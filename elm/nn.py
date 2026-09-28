"""Neural Lagrangian densities and their training on a teacher's Euler-Lagrange residual."""
import jax
import jax.numpy as jnp
import equinox as eqx
import optax


class BasicModel(eqx.Module):
    """Softplus MLP Lagrangian density: a scalar of its flattened arguments."""
    layers: eqx.nn.MLP

    def __init__(self, key, width_size=200, in_size=3, depth=3):
        self.layers = eqx.nn.MLP(in_size, 1, width_size, depth, jax.nn.softplus, key=key)

    @eqx.filter_jit  # its own compile boundary: dropping it changes f32 rollouts at ~1e-4
    def __call__(self, *args):
        return self.layers(jnp.asarray(args).ravel())[0]


class StructuredODEModel(eqx.Module):
    """L(q, qt) = 0.5 qt^T M(q) qt - V(q) with M = C C^T, C lower-triangular, positive diagonal."""
    M_net: eqx.nn.MLP
    V_net: eqx.nn.MLP
    n_dof: int
    use_trig: bool

    def __init__(self, key, n_dof, width_size=200, depth=3, use_trig=True):
        key_M, key_V = jax.random.split(key)
        in_size = 2 * n_dof if use_trig else n_dof
        n_entries = n_dof * (n_dof + 1) // 2
        self.M_net = eqx.nn.MLP(in_size, n_entries, width_size, depth, jax.nn.softplus, key=key_M)
        self.V_net = eqx.nn.MLP(in_size, 1, width_size, depth, jax.nn.softplus, key=key_V)
        self.n_dof = n_dof
        self.use_trig = use_trig

    def __call__(self, q, qt):
        q, qt = jnp.atleast_1d(q), jnp.atleast_1d(qt)
        features = jnp.concatenate([jnp.sin(q), jnp.cos(q)]) if self.use_trig else q
        rows, cols = jnp.tril_indices(self.n_dof)
        raw = self.M_net(features)
        entries = jnp.where(rows == cols, jax.nn.softplus(raw) + 1e-6, raw)
        C = jnp.zeros((self.n_dof, self.n_dof)).at[rows, cols].set(entries)
        return 0.5 * jnp.sum((C.T @ qt) ** 2) - self.V_net(features)[0]


ARCHS = dict(basic=BasicModel, structured=StructuredODEModel)
ARCH_KEYS = ("width_size", "depth", "in_size", "use_trig")


def _deserialise_leaf(f, like):
    """Read one leaf; arrays are cast to the skeleton's dtype (the current float precision)."""
    if eqx.is_array(like):
        return jnp.load(f).astype(like.dtype)
    return eqx.default_deserialise_filter_spec(f, like)


def load(path, arch="basic", **arch_kw):
    """Load a .eqx model saved by `train` into an ARCHS[arch](**arch_kw) skeleton."""
    skeleton = ARCHS[arch](jax.random.PRNGKey(0), **arch_kw)
    return eqx.tree_deserialise_leaves(path, skeleton, filter_spec=_deserialise_leaf)


def el_point(L, z, dz):
    """Pointwise Euler-Lagrange residual at jets z = (q, qt, *grads).

    dz[a] holds the derivatives of z along coordinate a of (t, x, y).
    """
    flux_divergence = 0
    for a, dz_a in enumerate(dz):
        _, flux_derivative = jax.jvp(jax.grad(L, a + 1), z, dz_a)
        flux_divergence = flux_divergence + flux_derivative
    return jax.grad(L)(*z) - flux_divergence


def _uniform(key, shape, low, high):
    return jax.random.uniform(key, shape, minval=low, maxval=high)


def _sample_states(n_dof, n_points, seed, amp_bound, omega_bound):
    """Uniform (q, qt); the time derivative (qt, qtt) has qtt = 0 as a placeholder."""
    key_q, key_qt = jax.random.split(jax.random.PRNGKey(seed))
    q = _uniform(key_q, (n_points, n_dof), -amp_bound, amp_bound)
    qt = _uniform(key_qt, (n_points, n_dof), -omega_bound, omega_bound)
    return (q, qt), ((qt, jnp.zeros_like(qt)),)


def _sample_waves(d, n_points, seed, amp_bound, k_bound, n_modes, c, qt_scale):
    """Jets of sums of n_modes standing plane waves A sin(k.x + phi) cos(omega t + psi).

    The time derivative's qtt is a zero placeholder; qt_scale scales the rate jets.
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), 2 * d + 5)
    modes = (n_points, n_modes)
    x = [_uniform(keys[a], (n_points,), -3.0, 3.0)[:, None] for a in range(d)]
    t = _uniform(keys[d], (n_points,), 0.0, 3.0)[:, None]
    A = _uniform(keys[d + 1], modes, -amp_bound, amp_bound)
    k = [_uniform(keys[d + 2 + a], modes, -k_bound, k_bound) for a in range(d)]
    phi = _uniform(keys[2 * d + 2], modes, 0.0, 2 * jnp.pi)
    psi = _uniform(keys[2 * d + 3], modes, 0.0, 2 * jnp.pi)

    if d == 1:
        omega = c * jnp.abs(k[0])
    else:
        omega = c * jnp.sqrt(sum(k_a ** 2 for k_a in k))
    space_phase = sum(k_a * x_a for k_a, x_a in zip(k, x)) + phi
    sin_x, cos_x = jnp.sin(space_phase), jnp.cos(space_phase)
    cos_t, sin_t = jnp.cos(omega * t + psi), jnp.sin(omega * t + psi)

    q = jnp.sum(A * sin_x * cos_t, axis=1)
    qt = qt_scale * jnp.sum(A * sin_x * (-omega) * sin_t, axis=1)
    grads = [jnp.sum(A * k_a * cos_x * cos_t, axis=1) for k_a in k]
    grads_t = [qt_scale * jnp.sum(A * k_a * cos_x * (-omega) * sin_t, axis=1) for k_a in k]

    z = (q, qt, *grads)
    dz_t = (qt, jnp.zeros_like(qt), *grads_t)
    dz_x = []
    for a in range(d):
        hessian_row = [jnp.sum(A * (-k[a] * k[b]) * sin_x * cos_t, axis=1) for b in range(d)]
        dz_x.append((grads[a], grads_t[a], *hessian_row))
    return z, (dz_t, *dz_x)


def sample(teacher, dom, n_points, seed=42, amp_bound=2.0, k_bound=6.0, n_modes=5, c=1.0,
           qt_scale=1.0, omega_bound=5.0):
    """Random training jets (z, dz): uniform states on a Point, plane waves on a field.

    qtt is closed by the teacher's Euler-Lagrange equation, so the teacher's residual is zero.
    """
    if dom.axes:
        z, dz = _sample_waves(len(dom.axes), n_points, seed, amp_bound, k_bound, n_modes, c,
                              qt_scale)
    else:
        z, dz = _sample_states(dom.n_dof, n_points, seed, amp_bound, omega_bound)

    def close(z, dz):
        residual = el_point(teacher, z, dz)
        mass = jax.hessian(teacher, 1)(*z)
        if mass.ndim == 0:
            qtt = residual / mass
        else:
            qtt = jnp.linalg.solve(mass, residual)
        dz_t = (dz[0][0], qtt, *dz[0][2:])
        return (dz_t, *dz[1:])

    return z, jax.vmap(close)(z, dz)


def train(teacher, dom, arch="basic", out=None, seed=42, n_points=5000, n_steps=50000,
          batch_size=256, norm_weight=1.0, mass_anchor=False, **kw):
    """Fit ARCHS[arch] to the teacher's pointwise EL residual plus a gauge penalty; save to `out`.

    The gauge fixes L's freedom at rest: L(0) = 0, no first-order terms in the odd arguments
    (qt, grads) and a prescribed curvature. Adam with a one-cycle schedule.
    """
    d = len(dom.axes)
    n_inputs = 2 * dom.n_dof if d == 0 else 2 + d
    arch_kw = {key: kw.pop(key) for key in ARCH_KEYS if key in kw}
    if arch == "structured":
        arch_kw.setdefault("n_dof", dom.n_dof)
    else:
        arch_kw.setdefault("in_size", n_inputs)
    model = ARCHS[arch](jax.random.PRNGKey(seed), **arch_kw)
    data = sample(teacher, dom, n_points, seed, **kw)

    if d == 0:
        h = 1e-2
        odd = range(dom.n_dof, n_inputs)
        curvatures = []
        if mass_anchor:
            rest = jnp.zeros((2, dom.n_dof))
            curvatures = list(zip(odd, jnp.diag(jax.hessian(teacher, 1)(*rest))))
    else:
        h = 1.0
        odd = range(1, 2 + d)
        curvatures = [(1, 2.0)]

    def at(m, v):
        """The model at the flattened inputs v."""
        if d == 0:
            return m(*jnp.split(v, 2))
        return m(*v)

    def gauge(m):
        steps = jnp.eye(n_inputs) * h
        L0 = at(m, jnp.zeros(n_inputs))
        slopes = 0
        for i in odd:
            slope = (at(m, steps[i]) - at(m, -steps[i])) / (2 * h)
            slopes = slopes + slope ** 2
        curvature_errors = 0
        for i, target in curvatures:
            curvature = (at(m, steps[i]) - 2 * L0 + at(m, -steps[i])) / h ** 2
            curvature_errors = curvature_errors + (curvature - target) ** 2
        return L0 ** 2 + slopes + curvature_errors

    def loss(m, z, dz):
        residuals = jax.vmap(lambda z, dz: el_point(m, z, dz))(z, dz)
        return jnp.sum(residuals ** 2) / batch_size + norm_weight * gauge(m)

    optimizer = optax.adam(optax.cosine_onecycle_schedule(
        n_steps, peak_value=1e-3, pct_start=0.2, div_factor=25, final_div_factor=1e4))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(m, opt_state, key, data):
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (batch_size,), 0, n_points)
        batch = jax.tree.map(lambda a: a[idx], data)
        value, grads = eqx.filter_value_and_grad(loss)(m, *batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        return eqx.apply_updates(m, updates), opt_state, key, value

    key = jax.random.PRNGKey(seed + 1000)
    for i in range(n_steps):
        model, opt_state, key, value = step(model, opt_state, key, data)
        if i % 1000 == 0:
            print(f"step {i} loss {float(value):.6e}", flush=True)
    if out:
        eqx.tree_serialise_leaves(out, model)
    return model
