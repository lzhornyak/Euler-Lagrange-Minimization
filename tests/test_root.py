import numpy as np

from elm import Axis, Integrator, Open, Rectangle
from elm.sim import gaussian, simulate, sine_mode, wave
from helpers import box_mode, energy, is_finite, max_abs, unit_box

ROOT = dict(solver="root", t_divs=2, x_divs=2, damping=1.0)


def ring_and_pulse():
    """A periodic ring with a right-moving Gaussian pulse."""
    ring = Axis(0.0, 10.0, 21, periodic=True)
    return ring, gaussian(ring, center=5.0, alpha=2.0, c=1.0)


def test_root_1d_linear_solves_agree():
    ring, s0 = ring_and_pulse()
    final_q = {}
    for linear in ("dense", "banded", "krylov"):
        integ = Integrator(ring, wave(), 0.2, n_refine=2, root_linear=linear, **ROOT)
        final_q[linear] = simulate(integ, s0, 10)[-1].q
    np.testing.assert_allclose(final_q["banded"], final_q["dense"], atol=1e-5)
    np.testing.assert_allclose(final_q["krylov"], final_q["dense"], atol=1e-5)


def test_root_1d_radau_nodes_match_gauss_nodes():
    pinned = Axis(-1.0, 1.0, 21)
    for dom, s0 in (ring_and_pulse(), (pinned, sine_mode(pinned))):
        final = {}
        for t_nodes in ("gauss", "radau"):
            integ = Integrator(dom, wave(), 0.05, n_refine=2, root_t_nodes=t_nodes, **ROOT)
            final[t_nodes] = simulate(integ, s0, 20)[-1]
        energy_gap = abs(energy(final["radau"], dom.dx) - energy(final["gauss"], dom.dx))
        assert energy_gap < 1e-3 * energy(s0, dom.dx)
        assert max_abs(final["radau"].q - final["gauss"].q) < 1e-3


def test_root_1d_open_faces_are_free_ends():
    dom = Axis(0.0, 2.0, 41, left=Open(), right=Open())
    s0 = gaussian(dom, center=1.0, alpha=16.0, c=1.0)
    states = simulate(Integrator(dom, wave(), 0.02, n_refine=2, **ROOT), s0, 100)
    assert is_finite(states[-1].q)
    assert max(abs(float(s.q[-1])) for s in states) > 1.5
    assert abs(energy(states[-1], dom.dx) - energy(s0, dom.dx)) < 1e-2 * energy(s0, dom.dx)


def test_root_2d_pinned_eigenmode():
    dom = unit_box()
    integ = Integrator(dom, wave(), 0.02, y_divs=2, n_refine=2, **ROOT)
    state = simulate(integ, sine_mode(dom), 10)[-1]
    assert max_abs(state.q - box_mode(dom, 0.2)) < 1e-2


def test_root_2d_open_dense_matches_krylov():
    side = Axis(0.0, 1.0, 5, left=Open(), right=Open())
    dom = Rectangle(side, side)
    s0 = gaussian(dom, center=[0.5, 0.5], alpha=10.0)
    final = {}
    for linear in ("dense", "krylov"):
        integ = Integrator(dom, wave(), 0.02, y_divs=2, n_refine=1, root_linear=linear, **ROOT)
        final[linear] = simulate(integ, s0, 2)[-1]
    for dense_field, krylov_field in zip(final["dense"], final["krylov"]):
        np.testing.assert_allclose(dense_field, krylov_field, atol=1e-4)
