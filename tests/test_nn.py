import jax
import numpy as np
import pytest

from elm import Axis, Integrator, Open, Point, Rectangle, nn
from elm.sim import blend, double_pendulum, gaussian, rest_mass, simulate, wave
from helpers import RING, TINY_ARCH, TINY_TRAIN, el_residuals, is_finite, max_abs

SQUARE = Rectangle(Axis(-1.0, 1.0, 5), Axis(-1.0, 1.0, 5))

TEACHERS = {
    "wave1d_c2_0p25": (wave(c2=0.25), RING, dict(c=0.5)),
    "wave2d": (wave(), SQUARE, {}),
    "sine_gordon": (wave(V="sine_gordon"), RING, dict(qt_scale=0.6, amp_bound=2.8, k_bound=2.0)),
    "double_pendulum": (double_pendulum(), Point(2), {}),
}


@pytest.mark.parametrize("name", TEACHERS)
def test_teacher_has_zero_residual_on_its_own_samples(name):
    teacher, dom, sample_kw = TEACHERS[name]
    assert max_abs(el_residuals(teacher, teacher, dom, **sample_kw)) < 1e-3


def same_leaves(a, b):
    return all(np.array_equal(u, v) for u, v in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


@pytest.mark.parametrize("dom, center, solver", [(RING, 5.0, "newton"),
                                                 (SQUARE, [0.0, 0.0], "root")])
def test_train_save_load_and_step(tmp_path, dom, center, solver):
    path = str(tmp_path / "model.eqx")
    model = nn.train(wave(), dom, out=path, **dict(TINY_TRAIN, n_steps=20))
    loaded = nn.load(path, in_size=2 + len(dom.axes), **TINY_ARCH)
    assert same_leaves(model, loaded)
    assert is_finite(el_residuals(model, wave(), dom))
    integ = Integrator(dom, model, 0.05, t_divs=2, x_divs=2, y_divs=2, n_refine=1, solver=solver)
    state = simulate(integ, gaussian(dom, center=center, alpha=1.0), 1)[-1]
    assert is_finite(state)


def test_blend_normalize_rescales_the_right_medium(models):
    left_spec = dict(name="model", path=models["wave1d"], **TINY_ARCH)
    right_spec = dict(name="model", path=models["wave1d_slow"], **TINY_ARCH)
    left = nn.load(models["wave1d"], **TINY_ARCH)
    right = nn.load(models["wave1d_slow"], **TINY_ARCH)
    L = blend(left_spec, right_spec, x0=1.0, width=0.011, normalize=True)
    z = (0.3, -0.2, 0.1)
    np.testing.assert_allclose(L(*z, 0.0), left(*z), rtol=1e-5)
    scale = rest_mass(left) / rest_mass(right)
    np.testing.assert_allclose(L(*z, 2.0), scale * right(*z), rtol=1e-5)


def test_blend_of_two_media_changes_the_dynamics():
    dom = Axis(0.0, 2.0, 21, left=Open(), right=Open())
    s0 = gaussian(dom, center=1.0, alpha=16.0)
    settings = dict(t_divs=3, x_divs=6, n_refine=3)
    two_media = blend(dict(name="wave"), dict(name="wave", c2=0.25), x0=1.0, width=0.011)
    blended = Integrator(dom, two_media, 0.01, position_dependent=True, **settings)
    uniform = Integrator(dom, wave(), 0.01, **settings)
    a = simulate(blended, s0, 3)[-1]
    b = simulate(uniform, s0, 3)[-1]
    assert is_finite(a.q)
    assert not np.allclose(a.q, b.q, atol=1e-6)
