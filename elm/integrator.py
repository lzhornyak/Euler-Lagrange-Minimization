import itertools
from typing import Any, Callable, NamedTuple
import jax
import jax.numpy as jnp
import equinox as eqx
from elm import root
from elm.basis import Hermite, el_residual, el_residual_0d, grid_vmap
from elm.domain import Point, Open, Driven, Mur, State0D
from elm.solver import newton_step


def _has_dt(k):
    """Whether jet k (bitmask order) carries a time derivative."""
    return k & 1 != 0


def _has_dx(k, axis):
    """Whether jet k (bitmask order) carries a derivative along spatial `axis`."""
    return k & (2 << axis) != 0


def _is_free(bc, axis, k):
    """Whether face condition `bc` on a face normal to `axis` leaves jet k to the Newton solve."""
    if isinstance(bc, Open):
        return True
    if isinstance(bc, Mur):
        return not (_has_dt(k) and not _has_dx(k, axis))
    return _has_dx(k, axis)


@eqx.filter_jit
def _extrapolate(state, dt):
    """Each jet as [old, guess] on a trailing time-level axis; values advance by their rates."""
    levels = []
    for k, x in enumerate(state):
        guess = x if _has_dt(k) else x + state[k + 1] * dt
        levels.append(jnp.stack([x, guess], -1))
    return tuple(levels)


@eqx.filter_jit
def _new_level(f):
    return tuple(x[..., 1] for x in f)


def _clamp(f, mask):
    """Hold every jet at zero on the barrier mask at the new time level."""
    if mask is None:
        return f
    return tuple(x.at[..., 1].set(jnp.where(mask, 0.0, x[..., 1])) for x in f)


def _windows(x, axes):
    """Stack the 3-node window around every patch center on new axes after the grid axes.

    A periodic axis centers a window on every node, a bounded one on nodes 1 .. n-2.
    """
    d = len(axes)
    for a, ax in enumerate(axes):
        if ax.periodic:
            nodes = [jnp.roll(x, 1, a), x, jnp.roll(x, -1, a)]
        else:
            n = x.shape[a]
            nodes = [jax.lax.slice_in_dim(x, i, n - 2 + i, axis=a) for i in range(3)]
        x = jnp.stack(nodes, d + a)
    return x


class _Span(NamedTuple):
    """Nodes along one axis that share a patch layout."""
    nodes: Any    # index into the node grid
    window: Any   # index into the stacked windows
    center: int   # position of the updated node within its window
    bc: Any       # face condition, None away from the faces


def _spans(ax):
    if ax.periodic:
        return [_Span(slice(None), slice(None), 1, None)]
    return [_Span(0, 0, 0, ax.left),
            _Span(slice(1, -1), slice(None), 1, None),
            _Span(-1, -1, 2, ax.right)]


class Integrator(eqx.Module):
    """ELM variational integrator on a Point, Axis or Rectangle domain.

    solver="newton" drives the squared EL residual down with n_refine damped Newton sweeps;
    solver="root" solves the square collocation system instead (see `root.solve` for the
    root_* and gmres_* options, which only the Axis and Rectangle root path uses).
    """
    domain: Any
    lagrangian: Callable
    dt: float
    n_refine: int
    damping: float
    solver: str
    sequential: bool
    position_dependent: bool
    root_kw: dict
    basis: Hermite
    quad: tuple

    def __init__(self, domain, lagrangian, dt, n_refine=5, damping=0.8, t_divs=5, x_divs=11,
                 y_divs=7, t_scale=None, solver="newton", sequential=False,
                 position_dependent=False, root_linear="auto", root_t_nodes="gauss",
                 gmres_tol=None, gmres_restart=10, gmres_maxiter=3):
        self.domain = domain
        self.lagrangian = lagrangian
        self.dt = dt
        self.n_refine = n_refine
        self.damping = damping
        self.solver = solver
        self.sequential = sequential
        self.position_dependent = position_dependent
        self.root_kw = dict(root_linear=root_linear, root_t_nodes=root_t_nodes,
                            gmres_tol=gmres_tol, gmres_restart=gmres_restart,
                            gmres_maxiter=gmres_maxiter)
        if solver == "root" and position_dependent:
            raise ValueError("solver='root' does not support position_dependent")
        self.basis = Hermite(dt, t_scale or dt, tuple(ax.dx for ax in domain.axes))
        x_divs = (x_divs, y_divs)[:len(domain.axes)]
        self.quad = self.basis.quadrature(t_divs, *x_divs)

    def step(self, state, t=0.0, mask=None):
        """Advance `state` (at absolute time t) by dt; mask: bool (nx, ny) barrier held at zero."""
        if isinstance(self.domain, Point):
            return self._step_point(state)
        if self.solver == "root":
            if mask is not None:
                raise ValueError("solver='root' does not support a barrier mask")
            return self._step_root(state)
        f = _extrapolate(state, self.dt)
        f = self._drive(f, t + self.dt)
        f = _clamp(self._absorb(f), mask)
        for _ in range(self.n_refine):
            f = self._sweep(f, mask)
        return type(state)(*_new_level(f))

    def _faces(self):
        """(axis, outward sign, index of the face nodes at the new time level, condition).

        A later axis leaves the corners alone.
        """
        axes = self.domain.axes
        for a, ax in enumerate(axes):
            for sign, node, bc in ((-1, 0, ax.left), (1, -1, ax.right)):
                index = [slice(1, -1)] * a + [node] + [slice(None)] * (len(axes) - a - 1)
                yield a, sign, tuple(index) + (1,), bc

    def _drive(self, f, t):
        """Set the driven faces to their prescribed values at time t."""
        f = list(f)
        for a, _, index, bc in self._faces():
            if not isinstance(bc, Driven):
                continue
            A, w = bc.amplitude, bc.omega
            for k in range(len(f)):
                if _has_dx(k, a):
                    continue
                if k == 0:
                    value = A * jnp.sin(w * t)
                elif k == 1:
                    value = A * w * jnp.cos(w * t)
                else:
                    value = 0.0
                f[k] = f[k].at[index].set(value)
        return tuple(f)

    def _absorb(self, f):
        """Mur faces: each rate jet follows its normal slope, qt = -sign c dq/dn."""
        f = list(f)
        for a, sign, index, bc in self._faces():
            if not isinstance(bc, Mur):
                continue
            for k in range(len(f)):
                if _has_dt(k) and not _has_dx(k, a):
                    slope = f[k - 1 + (2 << a)]
                    f[k] = f[k].at[index].set(-sign * bc.c * slope[index])
        return tuple(f)

    @eqx.filter_jit
    def _loss(self, patch, centers):
        """Quadrature-weighted squared EL residual of one patch; centers locate its middle node."""
        ts, xs, weights = self.quad
        d = len(xs)
        field = self.basis.field(patch)
        r = self.basis.t_scale / self.basis.dt

        def density(q, qt, *rest):
            grads, local = rest[:d], rest[d:]
            if self.position_dependent:
                coords = [c + s for c, s in zip(centers, local)]
                return self.lagrangian(q, qt, *grads, *coords)
            return self.lagrangian(q, qt, *grads)

        def residual(t, *x):
            return el_residual(density, field, r, t, *x)

        if d == 2:
            residual = jax.checkpoint(residual)
        return jnp.sum(weights * grid_vmap(residual, d + 1)(ts, *xs) ** 2)

    def _sites(self):
        """Node groups that share a patch layout: the interior first, then faces, then corners."""
        sites = itertools.product(*[_spans(ax) for ax in self.domain.axes])
        return sorted(sites, key=lambda site: sum(s.bc is not None for s in site))

    def _jet_blocks(self, site):
        """Free jets of the site's nodes, grouped into the blocks each solved by one Newton step.

        In 1D the values (q, qt) and slopes (qx, qxt) are separate blocks.
        """
        free = [k for k in range(2 << len(site))
                if all(_is_free(s.bc, a, k) for a, s in enumerate(site) if s.bc is not None)]
        if len(site) == 1:
            blocks = [[k for k in free if not _has_dx(k, 0)], [k for k in free if _has_dx(k, 0)]]
        else:
            blocks = [free]
        return [block for block in blocks if block]

    def _centers(self, site, batch_shape):
        """Coordinates of each window's middle node, shaped like the batch of updated nodes."""
        coords = []
        for ax, span in zip(self.domain.axes, site):
            if ax.periodic:
                middle = jnp.arange(ax.n)
            elif span.bc is None:
                middle = jnp.arange(1, ax.n - 1)
            else:
                middle = 1 if span.center == 0 else ax.n - 2
            coords.append(jnp.atleast_1d(ax.start + middle * ax.dx))
        return tuple(g.reshape(batch_shape) for g in jnp.meshgrid(*coords, indexing="ij"))

    def _solve_block(self, site, block, guess, windows):
        """One damped Newton step on `block` of every node of the site, from its window."""
        n_jets = len(windows)
        at_center = tuple(s.center for s in site) + (1,)
        centers = self._centers(site, guess.shape[:-1])
        patches = [w[tuple(s.window for s in site)] for w in windows]

        def loss(values, centers, *patch):
            patch = [patch[k].at[at_center].set(values[block.index(k)]) if k in block
                     else patch[k] for k in range(n_jets)]
            return self._loss(patch, centers)

        def newton(values, centers, *patch):
            return newton_step(loss, values, centers, *patch, damping=self.damping,
                               precond=len(site) == 2)

        n_batch = sum(s.bc is None for s in site)
        rows_in_sequence = n_batch == 2 and self.sequential
        n_vmapped = n_batch - 1 if rows_in_sequence else n_batch
        for _ in range(n_vmapped):
            newton = jax.vmap(newton)
        args = (guess, centers, *patches)
        if not rows_in_sequence:
            return newton(*args)

        def newton_on_row(row_args):
            return newton(*row_args)
        return jax.lax.map(newton_on_row, args)

    @eqx.filter_jit
    def _sweep(self, f, mask=None):
        """One Jacobi sweep: each node's free jets take a Newton step on the patch around it."""
        windows = [_windows(x, self.domain.axes) for x in f]
        out = list(f)
        for site in self._sites():
            nodes = tuple(s.nodes for s in site) + (1,)
            for block in self._jet_blocks(site):
                guess = jnp.stack([out[k][nodes] for k in block], -1)
                new = self._solve_block(site, block, guess, windows)
                for j, k in enumerate(block):
                    out[k] = out[k].at[nodes].set(new[..., j])
        return _clamp(self._absorb(out), mask)

    @eqx.filter_jit
    def _step_point(self, state):
        """ODE step: n_refine Newton (or root) iterations on the new (q, qt).

        The root solve collocates at the two Gauss points in time, whatever t_divs says.
        """
        q, qt = _extrapolate(state, self.dt)
        n = q.shape[0]
        ts, _, weights = self.quad
        if self.solver == "root":
            ts = self.basis.quadrature(2)[0]

        def coefficients(z):
            return jax.vmap(self.basis.fit_time)(q.at[:, 1].set(z[:n]), qt.at[:, 1].set(z[n:]))

        def residuals(z):
            a = coefficients(z)
            return jax.vmap(lambda t: el_residual_0d(self.lagrangian, self.basis, a, t))(ts)

        def flat_residuals(z):
            return residuals(z).ravel()

        def root_update(i, z):
            J = jax.jacfwd(flat_residuals)(z)
            return z - self.damping * jnp.linalg.solve(J, flat_residuals(z))

        def newton_update(i, z):
            def loss(z):
                return jnp.sum(weights[:, None] * residuals(z) ** 2)
            return newton_step(loss, z, damping=self.damping)

        update = root_update if self.solver == "root" else newton_update
        z = jax.lax.fori_loop(0, self.n_refine, update, jnp.concatenate([q[:, 1], qt[:, 1]]))
        return State0D(z[:n], z[n:])

    @eqx.filter_jit
    def _step_root(self, state):
        """PDE step by the square collocation root solve on the flattened node grid.

        It collocates on the unit-rate basis (t_scale = dt), whatever t_scale says.
        """
        d = len(self.domain.axes)
        f = _extrapolate(state, self.dt)
        old = jnp.stack([x[..., 0].reshape(-1) for x in f], 1)
        guess = jnp.stack([x[..., 1].reshape(-1) for x in f], 1)

        def density(q, qt, *rest):
            return self.lagrangian(q, qt, *rest[:d])

        basis = Hermite(self.dt, self.dt, self.basis.hs)
        with jax.default_matmul_precision("highest"):
            new = root.solve(density, basis, root.tables(self.domain.axes), old, guess,
                             n_refine=self.n_refine, damping=self.damping, **self.root_kw)
        new = new.reshape(f[0].shape[:-1] + (-1,))
        return type(state)(*(new[..., k] for k in range(len(f))))
