import pytest

from elm import Point, nn
from elm.sim import double_pendulum, wave
from helpers import RING, TINY_TRAIN, el_residuals, is_finite

# name: (teacher, domain, architecture, seed)
TINY_MODELS = {
    "wave1d": (wave(), RING, "basic", 0),
    "wave1d_slow": (wave(c2=0.25), RING, "basic", 1),
    "double_pendulum": (double_pendulum(), Point(2), "structured", 0),
}


@pytest.fixture(scope="session")
def models(tmp_path_factory):
    """Paths of the TINY_MODELS, trained once per session."""
    folder = tmp_path_factory.mktemp("models")
    paths = {}
    for name, (teacher, dom, arch, seed) in TINY_MODELS.items():
        paths[name] = str(folder / f"{name}.eqx")
        model = nn.train(teacher, dom, arch, out=paths[name], seed=seed,
                         mass_anchor=arch == "structured", **TINY_TRAIN)
        assert is_finite(el_residuals(model, teacher, dom))
    return paths
