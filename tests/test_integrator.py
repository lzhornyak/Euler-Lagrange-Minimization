import jax.numpy as jnp
import numpy as np
import pytest

from elm import Axis, Driven, Integrator, Mur, Open, Point, Rectangle, State0D
from elm.basis import Hermite
from elm.sim import barrier, gaussian, pendulum, simulate, sine_mode, wave, zeros
from helpers import box_mode, energy, is_finite, max_abs, unit_box

DT, H = 0.1, 0.3
T_NODES = jnp.array([0.0, DT])
X_NODES = jnp.array([-H, 0.0, H])


def cubic(t):
    return 1 + 2 * t - 3 * t**2 + t**3


def cubic_dt(t):
    return 2 - 6 * t + 3 * t**2


def quintic(x):
    return 1 - x + x**2 + 2 * x**3 - x**4 + x**5


def quintic_dx(x):
    return -1 + 2 * x + 6 * x**2 - 4 * x**3 + 5 * x**4


@pytest.mark.parametrize("t_scale", [DT, 1.0])
def test_hermite_time_fit_is_exact_on_cubics(t_scale):
    basis = Hermite(DT, t_scale)
    coeffs = basis.fit_time(cubic(T_NODES), cubic_dt(T_NODES))
    for s in (0.3 * t_scale, 0.8 * t_scale):
        np.testing.assert_allclose(basis.fn(coeffs, s), cubic(s * DT / t_scale), rtol=1e-5)


def test_hermite_space_fit_is_exact_on_quintics():
    basis = Hermite(DT, DT, (H,))
    coeffs = basis.fit_space(quintic(X_NODES), quintic_dx(X_NODES), H)
    np.testing.assert_allclose(basis.fn(coeffs, 0.17), quintic(0.17), rtol=1e-5)


def test_hermite_field_1d_is_exact_on_products():
    basis = Hermite(DT, DT, (H,))
    x, t = jnp.meshgrid(X_NODES, T_NODES, indexing="ij")
    patch = [quintic(x) * cubic(t), quintic(x) * cubic_dt(t),
             quintic_dx(x) * cubic(t), quintic_dx(x) * cubic_dt(t)]
    field = basis.field(patch)
    np.testing.assert_allclose(field(0.04, 0.11), cubic(0.04) * quintic(0.11), rtol=1e-5)


def test_hermite_field_2d_is_exact_on_products():
    basis = Hermite(DT, DT, (H, H))
    x, y, t = jnp.meshgrid(X_NODES, X_NODES, T_NODES, indexing="ij")
    patch = []
    for k in range(8):
        in_x = quintic_dx if k & 2 else quintic
        in_y = quintic_dx if k & 4 else quintic
        in_t = cubic_dt if k & 1 else cubic
        patch.append(in_x(x) * in_y(y) * in_t(t))
    field = basis.field(patch)
    expected = cubic(0.04) * quintic(0.11) * quintic(-0.2)
    np.testing.assert_allclose(field(0.04, 0.11, -0.2), expected, rtol=1e-5)


def test_quadrature_is_exact_on_monomials():
    ts, (xs,), weights = Hermite(DT, 1.0, (H,)).quadrature(3, 4)
    integral = jnp.sum(weights * ts[:, None]**5 * xs[None, :]**6)
    np.testing.assert_allclose(integral, (1 / 6) * (2 * H**7 / 7), rtol=1e-4)


def pendulum_energy(state):
    return 0.5 * state.qt[0]**2 - 9.81 * jnp.cos(state.q[0])


def test_point_pendulum_conserves_energy():
    s0 = State0D(jnp.array([1.0]), jnp.array([0.0]))
    E0 = float(pendulum_energy(s0))
    final_q = {}
    for solver in ("newton", "root"):
        integ = Integrator(Point(1), pendulum(), 0.05, t_divs=2, damping=1.0, solver=solver)
        states = simulate(integ, s0, 100)
        assert max(abs(float(pendulum_energy(s)) - E0) for s in states) < 1e-3 * abs(E0)
        final_q[solver] = states[-1].q
    np.testing.assert_allclose(final_q["newton"], final_q["root"], atol=1e-3)


def test_axis_periodic_gaussian_travels_at_unit_speed():
    ring = Axis(0.0, 10.0, 40, periodic=True)
    s0 = gaussian(ring, center=5.0, alpha=1.0, c=1.0)
    integ = Integrator(ring, wave(), 0.1, t_divs=3, x_divs=6, n_refine=3)
    state = simulate(integ, s0, 20)[-1]
    x = jnp.arange(ring.n) * ring.dx
    assert max_abs(state.q - jnp.exp(-(x - 7.0)**2)) < 2e-2


def test_axis_open_faces_do_not_gain_energy():
    dom = Axis(-1.0, 1.0, 21, left=Open(), right=Open())
    s0 = gaussian(dom, center=0.0, alpha=10.0)
    states = simulate(Integrator(dom, wave(), 0.01, t_divs=3, x_divs=6), s0, 30)
    assert is_finite(states[-1])
    assert max(energy(s, dom.dx) for s in states) < 1.05 * energy(s0, dom.dx) + 1e-6


def test_axis_pinned_faces_stay_at_zero():
    dom = Axis(-1.0, 1.0, 21)
    s0 = sine_mode(dom)
    states = simulate(Integrator(dom, wave(), 0.01, t_divs=3, x_divs=6), s0, 30)
    assert is_finite(states[-1])
    assert max(energy(s, dom.dx) for s in states) < 1.05 * energy(s0, dom.dx) + 1e-6
    assert all(max_abs(s.q[jnp.array([0, -1])]) < 1e-6 for s in states)


def test_position_dependent_path_matches_uniform_medium():
    dom = Axis(0.0, 10.0, 21, periodic=True)
    s0 = gaussian(dom, center=5.0, alpha=1.0)
    L = wave(c2=0.5)

    def L_at_x(q, qt, qx, x):
        return L(q, qt, qx)

    uniform = Integrator(dom, L, 0.2, t_divs=3, x_divs=6)
    at_x = Integrator(dom, L_at_x, 0.2, t_divs=3, x_divs=6, position_dependent=True)
    a = simulate(uniform, s0, 3)[-1]
    b = simulate(at_x, s0, 3)[-1]
    for field_a, field_b in zip(a, b):
        np.testing.assert_allclose(field_a, field_b, atol=0)


BOX_SETTINGS = dict(t_divs=3, x_divs=5, y_divs=5, n_refine=5, damping=0.8)


def test_rectangle_pinned_eigenmode():
    dom = unit_box()
    integ = Integrator(dom, wave(), 0.05, **BOX_SETTINGS)
    state = simulate(integ, sine_mode(dom), 4)[-1]
    assert max_abs(state.q - box_mode(dom, 0.2)) < 1e-2


def test_sequential_sweep_matches_parallel_sweep():
    dom = unit_box()
    s0 = gaussian(dom, center=[0.5, 0.5], alpha=10.0)
    parallel = Integrator(dom, wave(), 0.05, **BOX_SETTINGS)
    sequential = Integrator(dom, wave(), 0.05, sequential=True, **BOX_SETTINGS)
    a = simulate(parallel, s0, 2)[-1]
    b = simulate(sequential, s0, 2)[-1]
    np.testing.assert_allclose(a.q, b.q, atol=1e-4)


def test_driven_face_mur_faces_and_barrier_mask():
    amplitude, omega, dt = 0.25, 2 * np.pi, 0.04
    x_axis = Axis(0.0, 2.0, 9, left=Driven(amplitude, omega), right=Mur())
    y_axis = Axis(0.0, 2.0, 9, left=Mur(), right=Mur())
    dom = Rectangle(x_axis, y_axis)
    mask = barrier(dom, x=1.0, slits=[1.0], half_width=0.3, thick=1)
    integ = Integrator(dom, wave(), dt, t_divs=3, x_divs=3, y_divs=3, n_refine=4, damping=0.5)
    states = simulate(integ, zeros(dom), 5, mask=mask)
    for i, state in enumerate(states[1:], start=1):
        assert is_finite(state)
        assert not any(np.asarray(field)[mask].any() for field in state)
        np.testing.assert_allclose(state.q[0], amplitude * np.sin(omega * i * dt), atol=1e-6)
