import math
import pathlib
import tomllib

import numpy as np
import pytest

from elm import State1D, State2D
from elm.sim import load, main, run, save
from helpers import TINY_ARCH, is_finite

CONFIG_DIR = pathlib.Path(__file__).parents[1] / "configs"
SIM_CONFIGS = sorted(p.stem for p in CONFIG_DIR.glob("*.toml") if not p.stem.startswith("train_"))

# Learned alternatives of the paper configs; a "path" names a model of the `models` fixture.
LEARNED = {
    "fig2_double_pendulum": dict(name="model", path="double_pendulum", arch="structured",
                                 n_dof=2),
    "fig3_wave1d": dict(name="model", path="wave1d"),
    "fig3_wave1d_root": dict(name="model", path="wave1d"),
    "fig5a_two_media": dict(name="blend", x0=1.0, width=0.011, normalize=True,
                            left=dict(name="model", path="wave1d"),
                            right=dict(name="model", path="wave1d_slow")),
}


def small_config(name, out):
    """Config `name` shrunk to a two-step f32 run on a coarse grid, saved to `out`."""
    cfg = tomllib.loads((CONFIG_DIR / f"{name}.toml").read_text())
    cfg["x64"] = False
    cfg["run"].update(n_steps=2, every=1, out=str(out))
    domain = cfg["domain"]
    is_2d = "y" in domain
    for axis in ("x", "y"):
        if axis in domain:
            domain[axis]["n"] = min(domain[axis]["n"], 9 if is_2d else 21)
    if is_2d:
        cfg["integrator"]["n_refine"] = 3
    if "barrier" in cfg["run"]:
        driven = dict(type="driven", amplitude=0.25, omega=2 * math.pi)
        domain["x"].update(end=2.0, n=9, left=driven)
        domain["y"].update(start=0.0, end=2.0)
        cfg["integrator"]["dt"] = 0.04
        cfg["run"]["barrier"].update(x=0.6, slits=[0.7, 1.3])
    return cfg


def with_model_paths(spec, models):
    """A Lagrangian spec with each model name replaced by its trained file and architecture."""
    spec = dict(spec)
    for side in ("left", "right"):
        if side in spec:
            spec[side] = with_model_paths(spec[side], models)
    if "path" in spec:
        spec.update(path=models[spec["path"]], **TINY_ARCH)
    return spec


@pytest.mark.parametrize("name", SIM_CONFIGS)
def test_config_runs(name, tmp_path):
    states = run(small_config(name, tmp_path / "out.npz"))
    assert len(states) == 3
    assert is_finite(states[-1])


@pytest.mark.parametrize("name", sorted(LEARNED))
def test_config_runs_with_learned_lagrangian(name, tmp_path, models):
    cfg = small_config(name, tmp_path / "out.npz")
    cfg["lagrangian"] = with_model_paths(LEARNED[name], models)
    assert is_finite(run(cfg)[-1])


@pytest.mark.parametrize("cls", [State1D, State2D])
def test_save_load_roundtrip(tmp_path, cls):
    rng = np.random.default_rng(0)
    states = [cls(*rng.random((len(cls._fields), 3, 4))) for _ in range(3)]
    save(tmp_path / "states.npz", states, dt=0.1, every=2)
    loaded, meta = load(tmp_path / "states.npz")
    assert type(loaded[0]) is cls
    assert meta == dict(dt=0.1, every=2)
    for got, want in zip(loaded, states):
        for u, v in zip(got, want):
            np.testing.assert_array_equal(u, v)


def test_cli_overrides_and_import_path(tmp_path):
    expected = run(small_config("fig3_wave1d", tmp_path / "direct.npz"))
    out = tmp_path / "cli.npz"
    overrides = ["domain.x.n=21", "run.n_steps=2", "run.every=1", f'run.out="{out}"',
                 'lagrangian.name="elm.sim:wave"']
    argv = ["sim", str(CONFIG_DIR / "fig3_wave1d.toml")]
    for override in overrides:
        argv += ["--set", override]
    main(argv)
    states, meta = load(out)
    assert len(states) == 3
    assert meta["n_steps"] == 2
    for u, v in zip(expected[-1], states[-1]):
        np.testing.assert_array_equal(u, v)
