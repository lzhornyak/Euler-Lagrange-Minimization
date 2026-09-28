import os

os.environ.setdefault("JAX_COMPILATION_CACHE_DIR",
                      os.path.expanduser("~/.cache/jax_compilation_cache"))
from elm.domain import Point, Axis, Rectangle, Pinned, Open, Driven, Mur, State0D, State1D, State2D
from elm.integrator import Integrator
from elm.sim import simulate, save, load
