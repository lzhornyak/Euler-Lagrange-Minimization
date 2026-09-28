from collections import namedtuple
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Pinned:
    """q = 0 on the face (and its tangential derivatives)."""


@dataclass(frozen=True)
class Open:
    """All face DOFs free; the natural BC emerges from the Newton solve."""


@dataclass(frozen=True)
class Driven:
    """q = A sin(omega t), qt = A omega cos(omega t) on the face."""
    amplitude: float
    omega: float


@dataclass(frozen=True)
class Mur:
    """First-order absorbing face: qt + c dq/dn = 0 (2D)."""
    c: float = 1.0


@dataclass(frozen=True)
class Point:
    """0D domain (ODE) with n_dof degrees of freedom."""
    n_dof: int
    axes = ()


@dataclass(frozen=True)
class Axis:
    """Interval [start, end] with n nodes; dx = (end - start)/n if periodic else /(n-1)."""
    start: float
    end: float
    n: int
    left: Any = Pinned()
    right: Any = Pinned()
    periodic: bool = False

    @property
    def dx(self):
        return (self.end - self.start) / (self.n if self.periodic else self.n - 1)

    @property
    def axes(self):
        return (self,)


@dataclass(frozen=True)
class Rectangle:
    """Tensor product of two Axes; y.left is the bottom face, y.right the top."""
    x: Axis
    y: Axis

    @property
    def axes(self):
        return (self.x, self.y)


# Node states. Field k is the jet whose bitmask k has bit 0 for d/dt and bit 1 + a for d/dx_a.
State0D = namedtuple("State0D", "q qt")
State1D = namedtuple("State1D", "q qt qx qxt")
State2D = namedtuple("State2D", "q qt qx qxt qy qyt qxy qxyt")
