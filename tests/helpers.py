"""Small helpers shared by the tests."""
import jax
import jax.numpy as jnp
import numpy as np

from elm import Axis, Rectangle, nn

# Tiny networks and training runs: the tests check the plumbing, not the accuracy.
TINY_ARCH = dict(width_size=8, depth=1)
TINY_TRAIN = dict(TINY_ARCH, n_points=64, batch_size=16, n_steps=5)

RING = Axis(0.0, 10.0, 9, periodic=True)


def max_abs(a):
    return float(jnp.max(jnp.abs(a)))


def is_finite(state):
    """Whether every field of a state (or a single array) is finite."""
    return all(np.isfinite(np.asarray(field)).all() for field in jax.tree.leaves(state))


def energy(state, dx):
    """Discrete energy sum(0.5 qt^2 + 0.5 qx^2) dx of the unit-speed 1D wave."""
    return float(jnp.sum(0.5 * state.qt**2 + 0.5 * state.qx**2) * dx)


def unit_box(n=9):
    """The pinned unit square with n nodes per side."""
    return Rectangle(Axis(0.0, 1.0, n), Axis(0.0, 1.0, n))


def box_mode(dom, t):
    """The (1, 1) standing mode of the pinned unit square at time t, for c = 1."""
    x = jnp.linspace(0.0, 1.0, dom.x.n)
    profile = jnp.outer(jnp.sin(jnp.pi * x), jnp.sin(jnp.pi * x))
    return profile * jnp.cos(jnp.pi * jnp.sqrt(2.0) * t)


def el_residuals(model, teacher, dom, **sample_kw):
    """Pointwise EL residuals of `model` on 32 jets sampled from `teacher`."""
    z, dz = nn.sample(teacher, dom, 32, **sample_kw)

    def residual(z, dz):
        return nn.el_point(model, z, dz)
    return jax.vmap(residual)(z, dz)
