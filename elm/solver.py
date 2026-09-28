import jax
import jax.numpy as jnp


def newton_step(loss, x, *args, damping, precond=False):
    """One damped Newton step on loss(x, *args) with the exact Hessian.

    With `precond` the Jacobi-scaled system is solved (and the loss rematerialised); otherwise pinv.
    """
    if precond:
        loss = jax.checkpoint(loss)
    g = jax.grad(loss)(x, *args)
    H = jax.hessian(loss)(x, *args)
    if precond:
        scale = 1.0 / jnp.sqrt(jnp.maximum(jnp.abs(jnp.diag(H)), 1e-30))
        H_scaled = H * (scale[:, None] * scale[None, :])
        update = jnp.linalg.solve(H_scaled, g * scale) * scale
    else:
        update = jnp.linalg.pinv(H) @ g
    return x - damping * update


def solve_cyclic_block_tridiag(lower, diag, upper, rhs):
    """Solve lower_i u_{i-1} + diag_i u_i + upper_i u_{i+1} = rhs_i with periodic wrap-around.

    Block Thomas elimination on the open chain; the two wrap-around blocks are restored by Woodbury.
    """
    n, b, _ = diag.shape
    # Right-hand sides: rhs, then the wrap-around couplings to u_{n-1} (row 0) and u_0 (row n-1).
    R = jnp.zeros((n, b, 1 + 2 * b), dtype=diag.dtype)
    R = R.at[:, :, 0].set(rhs.reshape(n, b))
    R = R.at[0, :, 1:1 + b].set(lower[0])
    R = R.at[n - 1, :, 1 + b:].set(upper[n - 1])

    def eliminate(carry, row):
        D_prev, Y_prev = carry
        C, A, B_prev, R_i = row
        W = jnp.linalg.solve(D_prev.T, C.T).T
        D, Y = A - W @ B_prev, R_i - W @ Y_prev
        return (D, Y), (D, Y)

    first = (diag[0], R[0])
    _, (D, Y) = jax.lax.scan(eliminate, first, (lower[1:], diag[1:], upper[:-1], R[1:]))
    D = jnp.concatenate([diag[:1], D])
    Y = jnp.concatenate([R[:1], Y])

    def back_substitute(X_next, row):
        D_i, Y_i, B_i = row
        X_i = jnp.linalg.solve(D_i, Y_i - B_i @ X_next)
        return X_i, X_i

    X_last = jnp.linalg.solve(D[n - 1], Y[n - 1])
    _, X = jax.lax.scan(back_substitute, X_last, (D[:-1], Y[:-1], upper[:-1]), reverse=True)
    X = jnp.concatenate([X, X_last[None]])
    x, XU = X[:, :, 0], X[:, :, 1:]
    S = jnp.eye(2 * b, dtype=diag.dtype) + jnp.concatenate([XU[n - 1], XU[0]], axis=0)
    w = jnp.linalg.solve(S, jnp.concatenate([x[n - 1], x[0]]))
    return (x - XU @ w).reshape(n * b)
