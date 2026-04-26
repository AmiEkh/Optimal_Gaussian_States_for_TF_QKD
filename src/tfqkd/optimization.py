"""Optimization wrappers for the TF-QKD model."""
from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .key_rate import tf_qkd_fast


VSS_BOUNDS = (
    (1e-6, 1 - 1e-6),  # q
    (0.0, 2 * np.pi),  # phi0
    (0.0, 2 * np.pi),  # phi1
)

GS_BOUNDS = (
    (0.0, 0.99),       # epsilon
    (0.0, 2.0),        # displacement amplitude b
    (0.0, 2 * np.pi),  # phi0
    (0.0, 2 * np.pi),  # phi1
    (0.0, 2.0),        # squeezing amplitude s
    (0.0, 2 * np.pi),  # theta0
    (0.0, 2 * np.pi),  # theta1
)


def _negative_rate(x, loss_db, n_truncation, state, pd, f):
    """Objective function: minimize -R because scipy minimizes."""
    r, *_ = tf_qkd_fast(loss_db, n_truncation, x, state, pd=pd, f=f)

    if not np.isfinite(r):
        return 1e20

    return -float(r)


def _default_bounds(state: str):
    state = state.upper()

    if state == "VSS":
        return VSS_BOUNDS

    if state == "GS":
        return GS_BOUNDS

    raise ValueError(f"Unknown state type: {state!r}")


def _finalize_result(result, loss_db, n_truncation, state, pd, f, method):
    """Evaluate final optimized point and return a clean dictionary."""
    r, e_x, e_z, p_xx = tf_qkd_fast(
        loss_db=loss_db,
        n_truncation=n_truncation,
        parameters=result.x,
        state=state,
        pd=pd,
        f=f,
    )

    return {
        "loss_db": loss_db,
        "state": state.upper(),
        "method": method,
        "x": np.asarray(result.x, dtype=float),
        "rate": float(r),
        "e_x": float(e_x),
        "e_z": float(e_z),
        "p_xx": float(p_xx),
        "scipy_result": result,
    }


def optimize_de(
    loss_db: float,
    state: str,
    n_truncation: int = 6,
    pd: float = 1e-8,
    f: float = 1.16,
    bounds=None,
    seed: int | None = 1,
    polish: bool = True,
    maxiter: int = 100,
    popsize: int = 15,
):
    """Differential Evolution global optimization."""
    state = state.upper()
    bounds = _default_bounds(state) if bounds is None else bounds

    result = differential_evolution(
        _negative_rate,
        bounds=bounds,
        args=(loss_db, n_truncation, state, pd, f),
        seed=seed,
        polish=polish,
        maxiter=maxiter,
        popsize=popsize,
        updating="immediate",
        workers=1,
    )

    return _finalize_result(result, loss_db, n_truncation, state, pd, f, "DE")


def optimize_local(
    loss_db: float,
    state: str,
    x0,
    method: str = "L-BFGS-B",
    n_truncation: int = 6,
    pd: float = 1e-8,
    f: float = 1.16,
    bounds=None,
    maxiter: int = 500,
):
    """Local optimization / refinement."""
    state = state.upper()
    bounds = _default_bounds(state) if bounds is None else bounds

    method = method.upper()

    use_bounds = method in {"L-BFGS-B", "POWELL"}

    result = minimize(
        _negative_rate,
        np.asarray(x0, dtype=float),
        args=(loss_db, n_truncation, state, pd, f),
        method=method,
        bounds=bounds if use_bounds else None,
        options={"maxiter": maxiter},
    )

    return _finalize_result(result, loss_db, n_truncation, state, pd, f, method)


def optimize_pso(
    loss_db: float,
    state: str,
    n_truncation: int = 6,
    pd: float = 1e-8,
    f: float = 1.16,
    bounds=None,
    seed: int | None = 1,
    n_particles: int = 30,
    maxiter: int = 100,
    inertia: float = 0.7,
    cognitive: float = 1.5,
    social: float = 1.5,
):
    """Simple built-in Particle Swarm Optimization.

    No external package required.
    """
    rng = np.random.default_rng(seed)

    state = state.upper()
    bounds = _default_bounds(state) if bounds is None else bounds

    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    dim = len(bounds)

    positions = rng.uniform(lower, upper, size=(n_particles, dim))
    velocities = 0.1 * (upper - lower) * rng.normal(size=(n_particles, dim))

    personal_best = positions.copy()
    personal_best_values = np.array(
        [_negative_rate(x, loss_db, n_truncation, state, pd, f) for x in positions]
    )

    best_index = np.argmin(personal_best_values)
    global_best = personal_best[best_index].copy()
    global_best_value = personal_best_values[best_index]

    history = []

    for _ in range(maxiter):
        r1 = rng.random(size=(n_particles, dim))
        r2 = rng.random(size=(n_particles, dim))

        velocities = (
            inertia * velocities
            + cognitive * r1 * (personal_best - positions)
            + social * r2 * (global_best - positions)
        )

        positions = positions + velocities
        positions = np.clip(positions, lower, upper)

        values = np.array(
            [_negative_rate(x, loss_db, n_truncation, state, pd, f) for x in positions]
        )

        improved = values < personal_best_values
        personal_best[improved] = positions[improved]
        personal_best_values[improved] = values[improved]

        best_index = np.argmin(personal_best_values)

        if personal_best_values[best_index] < global_best_value:
            global_best_value = personal_best_values[best_index]
            global_best = personal_best[best_index].copy()

        history.append(-global_best_value)

    class Result:
        pass

    result = Result()
    result.x = global_best
    result.fun = global_best_value
    result.success = True
    result.message = "PSO completed"
    result.history = np.array(history)

    return _finalize_result(result, loss_db, n_truncation, state, pd, f, "PSO")


def optimize_bo(
    loss_db: float,
    state: str,
    n_truncation: int = 6,
    pd: float = 1e-8,
    f: float = 1.16,
    bounds=None,
    seed: int | None = 1,
    n_calls: int = 50,
    n_initial_points: int = 10,
):
    """Bayesian Optimization using scikit-optimize.

    Requires:
        pip install scikit-optimize
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except ImportError as exc:
        raise ImportError(
            "Bayesian optimization requires scikit-optimize. "
            "Install it with: pip install scikit-optimize"
        ) from exc

    state = state.upper()
    bounds = _default_bounds(state) if bounds is None else bounds

    space = [
        Real(low, high, name=f"x{i}")
        for i, (low, high) in enumerate(bounds)
    ]

    def objective(x):
        return _negative_rate(np.asarray(x), loss_db, n_truncation, state, pd, f)

    bo_result = gp_minimize(
        objective,
        dimensions=space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=seed,
    )

    class Result:
        pass

    result = Result()
    result.x = np.array(bo_result.x, dtype=float)
    result.fun = bo_result.fun
    result.success = True
    result.message = "Bayesian optimization completed"
    result.skopt_result = bo_result

    return _finalize_result(result, loss_db, n_truncation, state, pd, f, "BO")


def optimize(
    loss_db: float,
    state: str,
    method: str = "DE",
    n_truncation: int = 6,
    pd: float = 1e-8,
    f: float = 1.16,
    bounds=None,
    seed: int | None = 1,
    x0=None,
    **kwargs,
):
    """Unified optimizer.

    Supported methods:
        DE
        PSO
        BO
        L-BFGS-B
        Nelder-Mead
        Powell
    """
    method_clean = method.upper()

    if method_clean == "DE":
        return optimize_de(
            loss_db=loss_db,
            state=state,
            n_truncation=n_truncation,
            pd=pd,
            f=f,
            bounds=bounds,
            seed=seed,
            **kwargs,
        )

    if method_clean == "PSO":
        return optimize_pso(
            loss_db=loss_db,
            state=state,
            n_truncation=n_truncation,
            pd=pd,
            f=f,
            bounds=bounds,
            seed=seed,
            **kwargs,
        )

    if method_clean == "BO":
        return optimize_bo(
            loss_db=loss_db,
            state=state,
            n_truncation=n_truncation,
            pd=pd,
            f=f,
            bounds=bounds,
            seed=seed,
            **kwargs,
        )

    if method_clean in {"L-BFGS-B", "NELDER-MEAD", "POWELL"}:
        if x0 is None:
            raise ValueError(f"x0 must be provided for local method {method!r}.")

        return optimize_local(
            loss_db=loss_db,
            state=state,
            x0=x0,
            method=method_clean,
            n_truncation=n_truncation,
            pd=pd,
            f=f,
            bounds=bounds,
            **kwargs,
        )

    raise ValueError(
        f"Unknown method {method!r}. "
        "Choose from: DE, PSO, BO, L-BFGS-B, Nelder-Mead, Powell."
    )


def optimize_vss(
    loss_db: float,
    method: str = "DE",
    n_truncation: int = 6,
    pd: float = 1e-8,
    f: float = 1.16,
    bounds=VSS_BOUNDS,
    seed: int | None = 1,
    x0=None,
    **kwargs,
):
    """Optimize symmetric VSS parameters [q, phi0, phi1]."""
    return optimize(
        loss_db=loss_db,
        state="VSS",
        method=method,
        n_truncation=n_truncation,
        pd=pd,
        f=f,
        bounds=bounds,
        seed=seed,
        x0=x0,
        **kwargs,
    )


def optimize_gs(
    loss_db: float,
    method: str = "DE",
    n_truncation: int = 6,
    pd: float = 1e-8,
    f: float = 1.16,
    bounds=GS_BOUNDS,
    seed: int | None = 1,
    x0=None,
    **kwargs,
):
    """Optimize symmetric GS parameters [epsilon, b, phi0, phi1, s, theta0, theta1]."""
    return optimize(
        loss_db=loss_db,
        state="GS",
        method=method,
        n_truncation=n_truncation,
        pd=pd,
        f=f,
        bounds=bounds,
        seed=seed,
        x0=x0,
        **kwargs,
    )


def local_refine(
    x0,
    loss_db: float,
    n_truncation: int,
    state: str,
    pd: float,
    f: float = 1.16,
    bounds=None,
    method: str = "L-BFGS-B",
):
    """Backward-compatible local refinement wrapper."""
    return optimize_local(
        loss_db=loss_db,
        state=state,
        x0=x0,
        method=method,
        n_truncation=n_truncation,
        pd=pd,
        f=f,
        bounds=bounds,
    )


if __name__ == "__main__":
    print("\n=== 1) Objective function check ===")

    N = 3
    loss_db = 0.0
    pd = 1e-8

    x_vss = np.array([0.85, 0.0, np.pi])

    obj = _negative_rate(
        x_vss,
        loss_db,
        N,
        "VSS",
        pd,
        1.16,
    )

    print("Objective -R =", obj)

    assert np.isfinite(obj)
    assert obj <= 0.0 or np.isclose(obj, 0.0)

    print("\n=== 2) Default bounds check ===")

    vss_bounds = _default_bounds("VSS")
    gs_bounds = _default_bounds("GS")

    print("VSS bounds:", vss_bounds)
    print("GS bounds:", gs_bounds)

    assert len(vss_bounds) == 3
    assert len(gs_bounds) == 7

    print("\n=== 3) Differential Evolution: VSS ===")

    out_de = optimize_vss(
        loss_db=loss_db,
        method="DE",
        n_truncation=N,
        pd=pd,
        seed=1,
        polish=False,
        maxiter=2,
        popsize=4,
    )

    print("DE output:")
    print("x    =", out_de["x"])
    print("rate =", out_de["rate"])

    assert out_de["state"] == "VSS"
    assert out_de["method"] == "DE"
    assert out_de["x"].shape == (3,)
    assert out_de["rate"] >= 0.0

    print("\n=== 4) PSO: VSS ===")

    out_pso = optimize_vss(
        loss_db=loss_db,
        method="PSO",
        n_truncation=N,
        pd=pd,
        seed=1,
        n_particles=8,
        maxiter=3,
    )

    print("PSO output:")
    print("x    =", out_pso["x"])
    print("rate =", out_pso["rate"])

    assert out_pso["state"] == "VSS"
    assert out_pso["method"] == "PSO"
    assert out_pso["x"].shape == (3,)
    assert out_pso["rate"] >= 0.0

    print("\n=== 5) Local optimization: L-BFGS-B ===")

    out_local = optimize_vss(
        loss_db=loss_db,
        method="L-BFGS-B",
        x0=out_pso["x"],
        n_truncation=N,
        pd=pd,
        maxiter=5,
    )

    print("Local output:")
    print("x    =", out_local["x"])
    print("rate =", out_local["rate"])

    assert out_local["state"] == "VSS"
    assert out_local["method"] == "L-BFGS-B"
    assert out_local["x"].shape == (3,)
    assert out_local["rate"] >= 0.0

    print("\n=== 6) Local optimization: Nelder-Mead ===")

    out_nm = optimize_vss(
        loss_db=loss_db,
        method="Nelder-Mead",
        x0=x_vss,
        n_truncation=N,
        pd=pd,
        maxiter=5,
    )

    print("Nelder-Mead output:")
    print("x    =", out_nm["x"])
    print("rate =", out_nm["rate"])

    assert out_nm["state"] == "VSS"
    assert out_nm["method"] == "NELDER-MEAD"
    assert out_nm["x"].shape == (3,)
    assert out_nm["rate"] >= 0.0

    print("\n=== 7) Local optimization: Powell ===")

    out_powell = optimize_vss(
        loss_db=loss_db,
        method="Powell",
        x0=x_vss,
        n_truncation=N,
        pd=pd,
        maxiter=5,
    )

    print("Powell output:")
    print("x    =", out_powell["x"])
    print("rate =", out_powell["rate"])

    assert out_powell["state"] == "VSS"
    assert out_powell["method"] == "POWELL"
    assert out_powell["x"].shape == (3,)
    assert out_powell["rate"] >= 0.0

    print("\n=== 8) GS smoke test with PSO ===")

    out_gs = optimize_gs(
        loss_db=loss_db,
        method="PSO",
        n_truncation=N,
        pd=pd,
        seed=1,
        n_particles=6,
        maxiter=2,
    )

    print("GS PSO output:")
    print("x    =", out_gs["x"])
    print("rate =", out_gs["rate"])

    assert out_gs["state"] == "GS"
    assert out_gs["method"] == "PSO"
    assert out_gs["x"].shape == (7,)
    assert out_gs["rate"] >= 0.0

    print("\n=== 9) Backward-compatible local_refine ===")

    refined = local_refine(
        x0=out_pso["x"],
        loss_db=loss_db,
        n_truncation=N,
        state="VSS",
        pd=pd,
        bounds=VSS_BOUNDS,
        method="L-BFGS-B",
    )

    print("Refined output:")
    print("x    =", refined["x"])
    print("rate =", refined["rate"])

    assert refined["state"] == "VSS"
    assert refined["rate"] >= 0.0

    print("\n=== 10) Optional BO test ===")
    print("Skipping by default because it requires scikit-optimize.")
    print("To test it, install: pip install scikit-optimize")
    print("Then run:")
    print("""
out_bo = optimize_vss(
    loss_db=0.0,
    method="BO",
    n_truncation=3,
    pd=1e-8,
    n_calls=12,
    n_initial_points=5,
)
""")

    print("\noptimization.py sanity checks passed")