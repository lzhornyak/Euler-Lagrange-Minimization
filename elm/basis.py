import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx


def powers(s, n):
    """The monomials [1, s, ..., s**(n-1)]."""
    return jnp.array([s ** k for k in range(n)])


def hermite_fit(nodes, values, slopes, n):
    """Coefficients of the degree n-1 polynomial with the given values and slopes at the nodes."""
    value_rows = [[p ** k for k in range(n)] for p in nodes]
    slope_rows = [[k * p ** max(k - 1, 0) for k in range(n)] for p in nodes]
    A = jnp.array(value_rows + slope_rows)
    return jnp.linalg.solve(A, jnp.concatenate([values, slopes]))


def grid_vmap(fn, n_args):
    """Vectorise fn over the tensor grid of its n_args 1D array arguments."""
    for i in reversed(range(n_args)):
        in_axes = [None] * n_args
        in_axes[i] = 0
        fn = jax.vmap(fn, in_axes=tuple(in_axes))
    return fn


class Hermite(eqx.Module):
    """Hermite space-time basis: cubic in t on [0, t_scale], quintic in each spatial axis.

    A spatial axis spans 3 nodes at -h, 0, h (spacings `hs`); a Point has no spatial axes.
    """
    dt: float
    t_scale: float
    hs: tuple = ()

    def fn(self, a, s):
        """Evaluate the polynomial with coefficients `a` (last axis) at s."""
        return jnp.dot(a, powers(s, a.shape[-1]))

    def fit_time(self, q, qt):
        """Cubic-in-t coefficients from [old, new] values and physical rates.

        On a spatial basis they are zero-padded to 6 coefficients; the 2D field slices them
        back to 4, which keeps the f32 rollouts of the learned 2D figures bitwise.
        """
        a = hermite_fit((0.0, self.t_scale), q, qt * (self.dt / self.t_scale), 4)
        if self.hs:
            a = jnp.concatenate([a, jnp.zeros(2)])
        return a

    def fit_space(self, values, slopes, h):
        """Quintic coefficients from values and slopes at the nodes -h, 0, h."""
        return hermite_fit((-h, 0.0, h), values, slopes, 6)

    def field(self, patch):
        """Scalar field f(t, *xs) of a patch.

        `patch` holds the 2**(d+1) jets in bitmask order (bit 0: d/dt, bit 1 + a: d/dx_a),
        each shaped (3,)*d + (2,): the 3**d nodes at the [old, new] time levels.
        """
        fit = self.fit_time
        for _ in self.hs:
            fit = jax.vmap(fit)
        time_coeffs = [fit(value, rate) for value, rate in zip(patch[::2], patch[1::2])]
        if len(self.hs) == 1:
            return self._field_1d(*time_coeffs)
        return self._field_2d(*time_coeffs)

    def _field_1d(self, q, qx):
        at_nodes = jax.vmap(self.fn, in_axes=(0, None))

        def f(t, x):
            a = self.fit_space(at_nodes(q, t), at_nodes(qx, t), self.hs[0])
            return self.fn(a, x)
        return f

    def _field_2d(self, q, qx, qy, qxy):
        hx, hy = self.hs
        fit_x = jax.vmap(lambda v, s: self.fit_space(v, s, hx), in_axes=(1, 1))
        fit_y = jax.vmap(lambda v, s: self.fit_space(v, s, hy), in_axes=(1, 1))

        def f(t, x, y):
            cubic = powers(t, 6)[:4]
            Q, Qx, Qy, Qxy = (jnp.dot(a[..., :4], cubic) for a in (q, qx, qy, qxy))
            a_xy = fit_y(fit_x(Q, Qx), fit_x(Qy, Qxy))
            return jnp.dot(powers(x, 6), a_xy @ powers(y, 6))
        return f

    def quadrature(self, t_divs, *x_divs):
        """Gauss-Legendre nodes: t on [0, t_scale], each x on [-h, h]; returns (ts, xs, weights)."""
        nodes, weights = np.polynomial.legendre.leggauss(t_divs)
        ts = jnp.array((nodes + 1) / 2 * self.t_scale)
        W = jnp.array(weights / 2 * self.t_scale)
        xs = []
        for n, h in zip(x_divs, self.hs):
            nodes, weights = np.polynomial.legendre.leggauss(n)
            xs.append(jnp.array(nodes * h))
            W = W[..., None] * jnp.array(weights * h)
        return ts, tuple(xs), W


def jet(f, r, t, *xs):
    """(q, qt, *grad_x q) of field f at (t, *xs); r = t_scale/dt rescales the t-derivative."""
    q = f(t, *xs)
    qt = jax.grad(f, 0)(t, *xs) * r
    grads = [jax.grad(f, 1 + a)(t, *xs) for a in range(len(xs))]
    return (q, qt, *grads)


def el_residual(density, f, r, t, *xs):
    """Pointwise Euler-Lagrange residual of density(q, qt, *grads, *xs) on field f."""
    def dL(i, t, *xs):
        return jax.grad(density, i)(*jet(f, r, t, *xs), *xs)

    res = dL(0, t, *xs) - jax.grad(lambda s: dL(1, s, *xs))(t) * r
    for a in range(len(xs)):
        def flux(s, a=a):
            return dL(2 + a, t, *xs[:a], s, *xs[a + 1:])
        res = res - jax.grad(flux)(xs[a])
    return res


def el_residual_0d(L, basis, a, t):
    """ODE Euler-Lagrange residual at time t for per-DOF cubic coefficients a (n_dof, 4)."""
    r = basis.t_scale / basis.dt

    def state(s):
        q, dq = jax.jvp(lambda u: basis.fn(a, u), (s,), (jnp.ones_like(s),))
        return q, dq * r

    def dL_dqt(s):
        return jax.grad(L, 1)(*state(s))

    _, dL_dqt_dt = jax.jvp(dL_dqt, (t,), (jnp.ones_like(t),))
    return jax.grad(L, 0)(*state(t)) - dL_dqt_dt * r
