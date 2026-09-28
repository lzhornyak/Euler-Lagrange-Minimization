"""Paper Lagrangians and initial conditions, TOML configs, rollouts, .npz I/O and the CLI."""
import argparse
import importlib
import json
import math
import tomllib

import jax
import jax.numpy as jnp
import numpy as np

from elm import nn
from elm.domain import Axis, Driven, Mur, Open, Pinned, Point, Rectangle
from elm.domain import State0D, State1D, State2D
from elm.integrator import Integrator

STATES = (State0D, State1D, State2D)


def pendulum(m=1.0, l=1.0, g=9.81):
    """Pendulum angle q[0]: L = 0.5 m l^2 qt^2 + m g l cos q."""
    def L(q, qt):
        return 0.5*m*l**2*qt[0]**2 + m*g*l*jnp.cos(q[0])
    return L


def double_pendulum(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
    """Planar double pendulum in the absolute angles q = (q1, q2)."""
    def L(q, qt):
        coupling = m2*l1*l2*qt[0]*qt[1]*jnp.cos(q[0] - q[1])
        T = 0.5*(m1 + m2)*l1**2*qt[0]**2 + 0.5*m2*l2**2*qt[1]**2 + coupling
        V = -((m1 + m2)*g*l1*jnp.cos(q[0]) + m2*g*l2*jnp.cos(q[1]))
        return T - V
    return L


def wave(c2=1.0, V=None, m2=1.0):
    """Wave density 0.5 qt^2 - 0.5 c2 |grad q|^2, minus a sine-Gordon or Klein-Gordon potential."""
    if V not in (None, "sine_gordon", "klein_gordon"):
        raise ValueError(f"unknown potential {V!r}")

    def L(q, qt, *grads):
        density = 0.5*qt**2 - 0.5*c2*sum(g**2 for g in grads)
        if V == "sine_gordon":
            density = density - m2*(1 - jnp.cos(q))
        elif V == "klein_gordon":
            density = density - 0.5*m2*q**2
        return density
    return L


def rest_mass(L):
    """d^2 L / d qt^2 at rest."""
    return jax.grad(jax.grad(L, 1), 1)(0.0, 0.0, 0.0)


def blend(left, right, x0, width, normalize=False):
    """Two-media density L(q, qt, qx, x): a sigmoid of `width` at x0 blends two Lagrangian specs.

    With `normalize` the right density is rescaled to the mass of the left one.
    """
    L_left = resolve(left, LAGRANGIANS)
    L_right = resolve(right, LAGRANGIANS)
    scale = rest_mass(L_left) / rest_mass(L_right) if normalize else 1.0

    def L(q, qt, qx, x):
        a = jax.nn.sigmoid((x - x0) / width)
        return (1 - a)*L_left(q, qt, qx) + a*scale*L_right(q, qt, qx)
    return L


def jets(dom, u):
    """The state of the space-time profile u(t, *xs) at t = 0: all its jets on the node grid."""
    d = len(dom.axes)
    grid = jnp.meshgrid(*[ax.start + jnp.arange(ax.n)*ax.dx for ax in dom.axes], indexing="ij")
    fields = []
    for k in range(2 ** (d + 1)):
        derivative = u
        for arg in range(d + 1):
            if k >> arg & 1:
                derivative = jax.grad(derivative, arg)
        fields.append(jnp.vectorize(derivative)(0.0, *grid))
    return STATES[d](*fields)


def ode_state(dom, q, qt):
    """An ODE state from given (q, qt)."""
    return State0D(jnp.asarray(q), jnp.asarray(qt))


def zeros(dom):
    """The rest state."""
    if isinstance(dom, Point):
        return State0D(jnp.zeros(dom.n_dof), jnp.zeros(dom.n_dof))
    cls = STATES[len(dom.axes)]
    shape = tuple(ax.n for ax in dom.axes)
    return cls(*[jnp.zeros(shape) for _ in cls._fields])


def gaussian(dom, center, alpha, c=0.0):
    """Bump exp(-alpha |x - center|^2) moving along the first axis at speed c."""
    centers = [center]*len(dom.axes) if np.ndim(center) == 0 else center

    def u(t, *xs):
        offsets = [x - s for x, s in zip(xs, centers)]
        offsets[0] = offsets[0] - c*t
        return jnp.exp(-alpha*sum(r**2 for r in offsets))
    return jets(dom, u)


def sine_mode(dom, k=1):
    """Standing mode k of a pinned interval or box: a product of sines at rest."""
    def u(t, *xs):
        profile = 1.0
        for x, ax in zip(xs, dom.axes):
            profile = profile * jnp.sin(k*jnp.pi*(x - ax.start)/(ax.end - ax.start))
        return profile
    return jets(dom, u)


def breather(dom, omega=0.6, x0=0.0):
    """Sine-Gordon breather of frequency omega centred at x0."""
    eta = math.sqrt(1 - omega**2)

    def u(t, x):
        return 4*jnp.arctan((eta/omega)*jnp.sin(omega*t)/jnp.cosh(eta*(x - x0)))
    return jets(dom, u)


def kink_antikink(dom, v=0.25, s0=0.0):
    """Sine-Gordon kink-antikink pair at relative speed v, at time s0 of the collision."""
    gamma = 1/math.sqrt(1 - v**2)

    def u(t, x):
        return 4*jnp.arctan(jnp.sinh(gamma*v*(t + s0))/(v*jnp.cosh(gamma*x)))
    return jets(dom, u)


LAGRANGIANS = dict(pendulum=pendulum, double_pendulum=double_pendulum, wave=wave,
                   model=nn.load, blend=blend)
INITS = dict(state=ode_state, zeros=zeros, gaussian=gaussian, sine_mode=sine_mode,
             breather=breather, kink_antikink=kink_antikink)
BCS = dict(pinned=Pinned, open=Open, driven=Driven, mur=Mur)


def resolve(spec, registry, *args):
    """Call the function named by spec["name"] with *args and the remaining keys.

    The name is a registry key or an import path "pkg.mod:fn".
    """
    kwargs = dict(spec)
    name = kwargs.pop("name")
    if ":" in name:
        module, _, fn_name = name.rpartition(":")
        fn = getattr(importlib.import_module(module), fn_name)
    else:
        fn = registry[name]
    return fn(*args, **kwargs)


def _face(spec):
    """A face condition from "pinned" or a table like {type = "mur", c = 1.0}."""
    if isinstance(spec, str):
        return BCS[spec]()
    kwargs = dict(spec)
    return BCS[kwargs.pop("type")](**kwargs)


def _axis(spec):
    kwargs = dict(spec)
    for side in ("left", "right"):
        if side in kwargs:
            kwargs[side] = _face(kwargs[side])
    return Axis(**kwargs)


def build_domain(cfg):
    """Point from [domain] n_dof, Axis from [domain.x], Rectangle from [domain.x] and [domain.y]."""
    spec = cfg["domain"]
    if "n_dof" in spec:
        return Point(spec["n_dof"])
    if "y" in spec:
        return Rectangle(_axis(spec["x"]), _axis(spec["y"]))
    return _axis(spec["x"])


def set_precision(cfg):
    """Switch JAX to float64 when the config sets x64 = true, else float32."""
    jax.config.update("jax_enable_x64", cfg.get("x64", False))


def barrier(dom, x, slits, half_width, thick=3):
    """Boolean (nx, ny) mask of a wall `thick` columns wide at x, open around y in slits."""
    ax, ay = dom.axes
    first = round((x - ax.start) / ax.dx) - thick // 2
    columns = np.arange(ax.n)
    wall = (columns >= first) & (columns < first + thick)
    y = np.linspace(ay.start, ay.end, ay.n)
    gaps = np.zeros(ay.n, dtype=bool)
    for s in slits:
        gaps |= np.abs(y - s) <= half_width
    return wall[:, None] & ~gaps[None, :]


def build(cfg):
    """Config dict -> (integrator, initial state, step kwargs); also sets the precision."""
    set_precision(cfg)
    dom = build_domain(cfg)
    integ = Integrator(dom, resolve(cfg["lagrangian"], LAGRANGIANS), **cfg["integrator"])
    state0 = resolve(cfg["init"], INITS, dom)
    step_kw = {}
    if "barrier" in cfg["run"]:
        step_kw["mask"] = barrier(dom, **cfg["run"]["barrier"])
    return integ, state0, step_kw


def simulate(integ, state, n_steps, every=1, **step_kw):
    """Roll out n_steps; returns the initial state and every `every`-th state."""
    out = [state]
    for i in range(n_steps):
        state = integ.step(state, t=i*integ.dt, **step_kw)
        if (i + 1) % every == 0:
            out.append(state)
    return out


def save(path, states, **meta):
    """Save a list of states to .npz, one stacked array per field, with meta as json."""
    fields = {name: np.stack([getattr(s, name) for s in states]) for name in states[0]._fields}
    np.savez(path, **fields, meta=json.dumps(meta))


def load(path):
    """Load (states, meta) written by save."""
    data = np.load(path)
    states_by_size = {len(cls._fields): cls for cls in STATES}
    cls = states_by_size[len(data.files) - 1]
    fields = [data[name] for name in cls._fields]
    states = [cls(*[field[i] for field in fields]) for i in range(len(fields[0]))]
    return states, json.loads(str(data["meta"]))


def run(cfg):
    """Build and simulate a config, saving to [run] out if given; returns the kept states."""
    integ, state0, step_kw = build(cfg)
    n_steps = cfg["run"]["n_steps"]
    every = cfg["run"].get("every", 1)
    states = simulate(integ, state0, n_steps, every, **step_kw)
    if "out" in cfg["run"]:
        save(cfg["run"]["out"], states, dt=integ.dt, every=every, n_steps=n_steps)
    return states


def train(cfg):
    """Train a model on the config's teacher Lagrangian with the [train] options."""
    set_precision(cfg)
    teacher = resolve(cfg["lagrangian"], LAGRANGIANS)
    return nn.train(teacher, build_domain(cfg), **cfg["train"])


def _override(cfg, assignment):
    """Apply one "--set section.key=value" (a TOML value) to the config."""
    key, value = assignment.split("=", 1)
    *sections, last = key.split(".")
    table = cfg
    for section in sections:
        table = table.setdefault(section, {})
    table[last] = tomllib.loads(f"v = {value}")["v"]


def main(argv=None):
    """CLI: elm sim|train CONFIG [--set section.key=value ...]."""
    parser = argparse.ArgumentParser(prog="elm")
    parser.add_argument("command", choices=["sim", "train"])
    parser.add_argument("config")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config entry with a TOML value")
    args = parser.parse_args(argv)
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    for assignment in args.set:
        _override(cfg, assignment)
    if args.command == "sim":
        run(cfg)
    else:
        train(cfg)
