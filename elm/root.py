"""Square collocation root solver: one residual row per jet of every node, solved by Newton.

A node's rows are its pinned jets, EL collocation points in its space-time cell and, on an
open face, natural BC rows. Row coordinates are relative to the center of the node's window.
"""
import itertools
from typing import Any, NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jsla
from elm.basis import el_residual, grid_vmap, jet
from elm.domain import Open, Pinned
from elm.solver import solve_cyclic_block_tridiag

# Row kinds. Natural rows set the flux dL/dq_{x_a} (kind FLUX + a) and, at an open 2D corner,
# its y-derivative (kind FLUX + 2) to zero, together with their time rates, at the new level.
PIN, EL, FLUX = 0, 1, 2


def _gauss_pair(mid, width):
    """The two Gauss-Legendre points of the interval of `width` centered at mid."""
    half = width / (2.0 * np.sqrt(3.0))
    return [mid - half, mid + half]


class _AxisNode(NamedTuple):
    """Where a node sits along one axis, and the cell its rows live in."""
    center: int     # window center node
    offset: int     # node position within the window: -1, 0 or 1
    x: float        # node coordinate relative to the window center
    face: Any       # face condition, None away from the faces
    mid: float      # cell center: the full cell around the node, or the half cell inside a face
    width: float    # cell width

    @property
    def el_points(self):
        """EL collocation points: two Gauss points in a full cell, the midpoint of a half cell."""
        if self.face is None:
            return _gauss_pair(self.mid, self.width)
        return [self.mid]


def _axis_node(ax, i):
    h = ax.dx
    if ax.periodic or 0 < i < ax.n - 1:
        return _AxisNode(i, 0, 0.0, None, 0.0, h)
    if i == 0:
        center, face = 1, ax.left
    else:
        center, face = ax.n - 2, ax.right
    offset = i - center
    x = offset * h
    return _AxisNode(center, offset, x, face, x - offset * h / 4, h / 2)


class _Row(NamedTuple):
    """One residual row of a node."""
    kind: int       # PIN, EL or FLUX + a
    jet: int        # the jet a PIN row holds at zero
    t_index: int    # EL row: time node; natural row: 0 for the value, 1 for the rate
    x: tuple        # row coordinates relative to the window center


def _window(nodes, shape):
    """Flat indices of the 3**d window nodes around the node's window center."""
    window = []
    for w in np.ndindex(*(3,) * len(shape)):
        index = [(node.center + wa - 1) % n for node, wa, n in zip(nodes, w, shape)]
        window.append(np.ravel_multi_index(index, shape))
    return window


def _natural_rows(nodes):
    """Natural BC rows of a node on open faces."""
    x_node = tuple(node.x for node in nodes)
    is_open = [isinstance(node.face, Open) for node in nodes]
    rows = []
    if len(nodes) == 2 and all(is_open):
        for a in range(2):
            point = tuple(x_node[b] if b == a else nodes[b].mid for b in range(2))
            rows += [_Row(FLUX + a, 0, t, point) for t in (0, 1)]
        rows += [_Row(FLUX + 2, 0, t, x_node) for t in (0, 1)]
        return rows
    for a in range(len(nodes)):
        if is_open[a]:
            along = [[node.x] if b == a else _gauss_pair(node.mid, node.width)
                     for b, node in enumerate(nodes)]
            points = list(itertools.product(*along))
            rows += [_Row(FLUX + a, 0, t, x) for t in (0, 1) for x in points]
    return rows


def _node_rows(nodes, n_jets):
    """The node's n_jets rows.

    The per-rank row order matches the reference layout (the dense and banded LU pivoting
    depend on it); only a mixed open/pinned 2D corner has excess rows to drop.
    """
    pinned = set()
    for a, node in enumerate(nodes):
        if isinstance(node.face, Pinned):
            pinned.update(k for k in range(n_jets) if not k & (2 << a))
    pin_rows = [_Row(PIN, k, 0, (0.0,) * len(nodes)) for k in sorted(pinned)]
    el_points = list(itertools.product(*[node.el_points for node in nodes]))
    el_rows = [_Row(EL, 0, t, x) for t in (0, 1) for x in el_points]
    if len(nodes) == 1:
        rows = pin_rows + _natural_rows(nodes) + el_rows
    else:
        rows = pin_rows + el_rows + _natural_rows(nodes)
    return rows[:n_jets]


class Table(NamedTuple):
    """Static row layout of the collocation system, one entry per node of the flattened grid."""
    window: np.ndarray   # (n_nodes, 3**d) flat indices of the window nodes
    slot: np.ndarray     # (n_nodes,) position of the node within its window
    kind: np.ndarray     # (n_nodes, n_jets) row kinds
    jet: np.ndarray      # the jet a PIN row holds at zero
    t_index: np.ndarray  # EL row: time node; natural row: 0 for the value, 1 for the rate
    x: np.ndarray        # (n_nodes, n_jets, d) row coordinates
    has_open: bool
    periodic: bool


def tables(axes):
    """Row table of the square collocation system on the given axes."""
    faces = [bc for ax in axes if not ax.periodic for bc in (ax.left, ax.right)]
    if not all(isinstance(bc, (Pinned, Open)) for bc in faces):
        raise ValueError("solver='root' supports only periodic, pinned and open faces")
    shape = tuple(ax.n for ax in axes)
    d = len(axes)
    n_jets = 2 ** (d + 1)
    windows, slots, kinds, jets, t_indices, xs = [], [], [], [], [], []
    for idx in np.ndindex(*shape):
        nodes = [_axis_node(ax, i) for ax, i in zip(axes, idx)]
        windows.append(_window(nodes, shape))
        slots.append(np.ravel_multi_index([node.offset + 1 for node in nodes], (3,) * d))
        rows = _node_rows(nodes, n_jets)
        kinds.append([row.kind for row in rows])
        jets.append([row.jet for row in rows])
        t_indices.append([row.t_index for row in rows])
        xs.append([row.x for row in rows])
    return Table(window=np.array(windows), slot=np.array(slots), kind=np.array(kinds),
                 jet=np.array(jets), t_index=np.array(t_indices), x=np.array(xs),
                 has_open=any(isinstance(bc, Open) for bc in faces),
                 periodic=all(ax.periodic for ax in axes))


def _finite(a):
    return jnp.where(jnp.isfinite(a), a, 0.0)


def _linear_method(root_linear, d, periodic):
    if root_linear == "banded" and not (d == 1 and periodic):
        raise ValueError("root_linear='banded' needs a periodic Axis")
    if root_linear != "auto":
        return root_linear
    if d == 2:
        return "krylov"
    if periodic and jax.devices()[0].platform == "cpu":
        return "banded"
    return "dense"


def _dense_solve(jacobians, window, rhs):
    """Assemble the global Jacobian from each node's rows (over its window jets) and solve."""
    n_nodes, n_jets, _ = jacobians.shape
    rows = np.arange(n_nodes * n_jets).reshape(n_nodes, n_jets, 1)
    cols = (window[:, :, None] * n_jets + np.arange(n_jets)).reshape(n_nodes, 1, -1)
    J = jnp.zeros((n_nodes * n_jets,) * 2, rhs.dtype).at[rows, cols].add(jacobians)
    return jnp.linalg.solve(J, rhs)


def solve(density, basis, table, old, guess, *, n_refine, damping, root_linear="auto",
          root_t_nodes="gauss", gmres_tol=None, gmres_restart=10, gmres_maxiter=3):
    """n_refine damped Newton steps on the collocation system for the new-level node jets.

    old, guess: (n_nodes, n_jets) jets at the old level and the first guess of the new one.
    root_linear: "dense", "banded" (1D periodic), "krylov" (block-Jacobi GMRES) or "auto".
    root_t_nodes: "gauss" or "radau" time collocation nodes.
    """
    n_nodes, n_jets = guess.shape
    d = len(basis.hs)
    r = basis.t_scale / basis.dt
    t_new = basis.t_scale
    if root_t_nodes == "gauss":
        t_nodes = basis.quadrature(2)[0]
    else:
        t_nodes = jnp.array([1.0 / 3.0, 1.0]) * basis.t_scale
    linear = _linear_method(root_linear, d, table.periodic)
    old_windows = old[table.window].reshape(n_nodes, -1)
    rows = (table.kind, table.jet, table.t_index, table.x, table.slot)

    def window_field(new_window, old_window):
        new = new_window.reshape((3,) * d + (n_jets,))
        old = old_window.reshape((3,) * d + (n_jets,))
        return basis.field(tuple(jnp.stack([old[..., k], new[..., k]], -1) for k in range(n_jets)))

    def periodic_residual(field):
        """EL rows on the full Gauss grid of a periodic cell."""
        def el(t, *x):
            return el_residual(density, field, r, t, *x)
        points = [0.5 * x for x in basis.quadrature(2, *(2,) * d)[1]]
        return grid_vmap(el, d + 1)(t_nodes, *points).reshape(n_jets)

    def row_residual(field, new_window, slot, kind, k, t_index, x):
        def flux(a, t, *xs):
            return jax.grad(density, 2 + a)(*jet(field, r, t, *xs), *xs)

        def natural(i, t):
            """Flux dL/dq_{x_i} at x; i = 2 is the y-derivative of the x-flux (2D corner)."""
            if i < d:
                return flux(i, t, *x)
            return jax.grad(lambda y: flux(0, t, x[0], y))(x[1])

        values = [jnp.take(new_window, slot * n_jets + k),
                  el_residual(density, field, r, jnp.take(t_nodes, t_index), *x)]
        conditions = [kind == PIN, kind == EL]
        n_naturals = (d + (d == 2)) if table.has_open else 0
        for i in range(n_naturals):
            values += [natural(i, t_new), jax.grad(natural, 1)(i, t_new) * r]
            conditions += [(kind == FLUX + i) & (t_index == 0), (kind == FLUX + i) & (t_index == 1)]
        return jnp.select(conditions, values)

    def cell_residual(new_window, old_window, kind, k, t_index, x, slot):
        """The rows of one node, from the jets of its window."""
        field = window_field(new_window, old_window)
        if table.periodic:
            return periodic_residual(field)

        def row(kind, k, t_index, x):
            return row_residual(field, new_window, slot, kind, k, t_index, x)
        return jax.vmap(row)(kind, k, t_index, x)

    def cell_and_jacobian(new_window, *args):
        return cell_residual(new_window, *args), jax.jacfwd(cell_residual)(new_window, *args)

    def own_block(new_window, old_window, kind, k, t_index, x, slot):
        """Jacobian of a node's rows with respect to its own jets."""
        start = (slot * n_jets,)

        def rows_of(own):
            window = jax.lax.dynamic_update_slice(new_window, own, start)
            return cell_residual(window, old_window, kind, k, t_index, x, slot)
        return jax.jacfwd(rows_of)(jax.lax.dynamic_slice(new_window, start, (n_jets,)))

    def residual(u_flat):
        new_windows = u_flat.reshape(n_nodes, n_jets)[table.window].reshape(n_nodes, -1)
        return jax.vmap(cell_residual)(new_windows, old_windows, *rows).reshape(-1)

    u = guess
    for _ in range(n_refine):
        new_windows = u[table.window].reshape(n_nodes, -1)
        if linear == "krylov":
            blocks = jax.vmap(own_block)(new_windows, old_windows, *rows)

            def block_jacobi(v):
                return jax.vmap(jnp.linalg.solve)(blocks, v.reshape(n_nodes, n_jets)).reshape(-1)

            def jacobian_times(v):
                return jax.jvp(residual, (u.reshape(-1),), (v,))[1]

            tol = gmres_tol or (1e-8 if guess.dtype == jnp.float64 else 1e-5)
            du, _ = jsla.gmres(jacobian_times, _finite(residual(u.reshape(-1))), tol=tol, atol=0.0,
                               restart=gmres_restart, maxiter=gmres_maxiter, M=block_jacobi)
        else:
            res, jacobians = jax.vmap(cell_and_jacobian)(new_windows, old_windows, *rows)
            res, jacobians = _finite(res).reshape(-1), _finite(jacobians)
            if linear == "banded":
                lower, diag, upper = jnp.split(jacobians, 3, axis=-1)
                du = solve_cyclic_block_tridiag(lower, diag, upper, res)
            else:
                du = _dense_solve(jacobians, table.window, res)
        u = u - damping * _finite(du).reshape(n_nodes, n_jets)
    return u
