# ELM: Euler-Lagrange Minimization

Variational integrators for ODEs and PDEs given by a (learned or analytic) Lagrangian density, from
*ELM* (arXiv:2605.07157). Each step drives the squared discrete Euler-Lagrange residual on Hermite
space-time cells to zero with damped Newton sweeps (`solver="newton"`), or solves the square
collocation system directly (`solver="root"`). Built on JAX and Equinox.

## Install

```bash
pixi install              # GPU (jax[cuda13]); then prefix commands with `pixi run`
pip install -e .          # or any environment with jax, equinox, numpy, optax
```

Run the tests with `pixi run test` (small float32 cases, several minutes on a GPU).

## Usage

Every run is one TOML file. `--set a.b=v` overrides any key; the value is parsed as TOML.

```bash
elm sim configs/fig3_wave1d.toml
elm sim configs/fig3_wave1d.toml --set run.n_steps=500 --set run.out='"quick.npz"'
elm train configs/train_wave1d.toml
```

Shipped configs, one per paper figure plus a few other systems:

| Config | Setup |
|---|---|
| `fig2_double_pendulum[_root].toml` | double pendulum from rest, f64 |
| `fig3_wave1d[_root].toml` | 1D wave on a periodic ring, n = 31 |
| `fig4_wave2d[_root].toml` | 2D wave in a pinned box |
| `fig5a_two_media.toml` | 1D pulse across a sigmoid step in c^2 (position-dependent L), open ends |
| `fig5b_double_slit.toml` | driven 2D wave through a double-slit barrier, Mur outflow |
| `sine_gordon_breather.toml`, `sine_gordon_kink_antikink.toml`, `klein_gordon.toml` | nonlinear examples |
| `wave1d_pinned.toml` | pinned standing mode (the paper runs c2 = 0.05, 0.25 and 1.0) |
| `train_*.toml` | training recipes for the learned densities |

The `_root` configs use `solver = "root"`. Trained models are not shipped: train one with the
matching `train_*.toml`, then pass the `[lagrangian]` table from the config's `# learned:` comment
with `--set` (last command below; for Fig. 5a, set `left`, `right` and `normalize`).
Common overrides:

```bash
elm sim configs/fig3_wave1d.toml --set domain.x.n=21 --set integrator.dt=0.4       # coarser grid
elm train configs/train_wave1d.toml --set lagrangian.c2=0.25 --set train.c=0.5 \
  --set train.width_size=200 --set train.amp_bound=2.0 \
  --set train.out='"wave1d_c2_0p25.eqx"'              # c^2 sweep; c2 = 1.0 uses train.c=1.0
elm sim configs/fig3_wave1d_root.toml --set integrator.root_linear='"krylov"'
elm sim configs/fig3_wave1d.toml \
  --set lagrangian='{name = "model", path = "wave1d_c2_0p05.eqx", width_size = 400}'
```

`elm sim` writes `run.out` as `.npz`: one array per state field, stacked over the saved frames,
plus `meta` (`dt`, `every`, `n_steps`). `elm.load(path)` reads it back as `(states, meta)`.

## Config schema

```toml
x64 = false                  # optional; true runs in float64

[domain]                     # Point: n_dof = 2
[domain.x]                   # Axis; add [domain.y] for a Rectangle
start = 0.0
end = 10.0
n = 31
periodic = true              # ring with dx = (end - start)/n; otherwise dx = (end - start)/(n - 1)
# left / right faces: "pinned" (default) | "open" | {type = "mur", c = 1.0}
#   | {type = "driven", amplitude = .., omega = ..}; on [domain.y], left/right = bottom/top

[integrator]                 # Integrator(domain, L, **integrator)
dt = 0.2
t_divs = 3                   # Gauss-Legendre points per cell in t (default 5)
x_divs = 6                   # ... in x (default 11); y_divs in y (default 7)
n_refine = 3                 # Newton sweeps per step (root: Newton iterations); default 5
damping = 0.8                # Newton step factor; default 0.8
# solver = "newton" | "root"
# t_scale = 1.0              # length of a cell's reference time interval (default dt)
# sequential = true          # 2D Newton: sweep the rows one at a time (less memory)
# position_dependent = true  # L takes the position x (and y) as trailing arguments
# root_linear = "auto" | "dense" | "banded" | "krylov"; root_t_nodes = "gauss" | "radau"
# gmres_tol, gmres_restart, gmres_maxiter: the "krylov" solve

[lagrangian]                 # registry name or "pkg.mod:fn"; the other keys are its kwargs
name = "wave"
c2 = 0.05

[init]                       # initial condition, same convention
name = "gaussian"
center = 5.0
alpha = 2.0

[run]
n_steps = 50000
every = 50                   # save every 50th state
out = "fig3_wave1d.npz"
# [run.barrier] x, slits, half_width, thick: a wall held at zero (2D Newton runs)

[train]                      # elm train: elm.nn.train(L, domain, **train)
```

Lagrangians (`elm.sim.LAGRANGIANS`):

| Name | Arguments | Density |
|---|---|---|
| `pendulum` | `m, l, g` | `L(q, qt)` |
| `double_pendulum` | `m1, m2, l1, l2, g` | `L(q, qt)` in the absolute angles |
| `wave` | `c2, V, m2` | `0.5 qt^2 - 0.5 c2 abs(grad q)^2`, minus `m2 (1 - cos q)` (`V = "sine_gordon"`) or `0.5 m2 q^2` (`V = "klein_gordon"`) |
| `model` | `path, arch, **arch_kw` | trained `elm.nn` model (`arch = "basic" \| "structured"`) |
| `blend` | `left, right, x0, width, normalize` | sigmoid blend in x of two nested specs; needs `position_dependent = true` |

Initial conditions (`elm.sim.INITS`): `state` (`q, qt`), `zeros`, `gaussian` (`center, alpha, c`),
`sine_mode` (`k`), `breather` (`omega, x0`), `kink_antikink` (`v, s0`).

Training (`[train]`): `arch, width_size, depth, in_size, use_trig, n_points, n_steps, batch_size,
norm_weight, seed, out`; sampler bounds `amp_bound, k_bound, n_modes, c, qt_scale, omega_bound`;
and `mass_anchor` for ODEs. The objective is the pointwise Euler-Lagrange residual of the model on
jets sampled from the teacher `[lagrangian]`, plus a gauge term. Only the rank (or `n_dof`) of
`[domain]` matters for training.

## Python API

```python
import elm
from elm.sim import gaussian, wave

dom = elm.Axis(0.0, 10.0, 31, periodic=True)
integ = elm.Integrator(dom, wave(c2=0.05), dt=0.2, t_divs=3, x_divs=6, n_refine=3, damping=0.8)
states = elm.simulate(integ, gaussian(dom, center=5.0, alpha=2.0), n_steps=100, every=10)
elm.save("out.npz", states, dt=0.2)
```

A Lagrangian is any JAX callable: `L(q, qt)` on a `Point`, `L(q, qt, qx)` on an `Axis`,
`L(q, qt, qx, qy)` on a `Rectangle`, with trailing `x` (and `y`) when `position_dependent=True`.
`integ.step(state, t=0.0, mask=None)` advances one step; `t` is the absolute time (used by driven
faces) and `mask` a boolean node grid held at zero.

## Notes

- The Newton sweeps use the exact Hessian and start from the first-order guess q + qt dt (the
  paper's Fig. 4 and 5b used Gauss-Newton and a second-order EL initial guess). The Fig. 3
  post-step zero-mode projection is not included.
- `solver = "root"` supports periodic, pinned and open faces on all ranks; it has no driven or Mur
  faces, barrier mask or position dependence, so Fig. 5a and 5b have no root config. It always
  collocates at two time nodes and, on an `Axis` or `Rectangle`, two Gauss points per cell axis;
  `t_divs`, `x_divs` and `y_divs` are ignored, and so is `t_scale` on an `Axis` or `Rectangle`.
- `root_linear = "auto"` picks `krylov` in 2D, `banded` on a periodic 1D ring on CPU, else `dense`;
  `banded` needs a periodic `Axis`. The `root_*` and `gmres_*` options apply to `Axis` and
  `Rectangle` only.
- Fig. 2 runs in f64 (`x64 = true`). Kink-antikink runs that start long before the collision (large
  `|s0|`) need it too, since `sinh` overflows in f32.

## License

MIT; see `LICENSE`.
