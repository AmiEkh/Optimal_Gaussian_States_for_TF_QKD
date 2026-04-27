from __future__ import annotations

import sys
import json
import time
import traceback
from pathlib import Path
from multiprocessing import freeze_support
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, CubicSpline, PchipInterpolator

# =============================================================================
# Project path setup
# =============================================================================
# Expected structure:
# project_root/
#   src/tfqkd/
#   scripts/run_vss_q_optimization.py
#
# If this script is inside scripts/, PROJECT_ROOT is parent of scripts/.
# If you run it from another location, edit PROJECT_ROOT manually.

PROJECT_ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "scripts" else Path.cwd()
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tfqkd.key_rate import tf_qkd_fast


# =============================================================================
# User settings
# =============================================================================

N_TRUNCATION = 6
P_DARK = 1e-8
F_RECONCILIATION = 1.16

# Fixed VSS phases
PHI0_FIXED = 0.0
PHI1_FIXED = np.pi

# Optimize q only
Q_BOUNDS = (1e-6, 1.0 - 1e-6)

# Loss points to optimize directly
LOSS_DB_VALUES = np.linspace(0.0, 120.0, 21)

# Parallel over loss points
PARALLEL_OVER_LOSSES = True
MAX_WORKERS = 4
SEED = 1

# PSO settings. Tune these to match your MATLAB PSO settings.
# MATLAB common defaults are often close to:
# swarm_size around 30-100, inertia around 0.7-1.1, cognitive/social around 1.49.
PSO_SETTINGS = {
    "n_particles": 40,
    "maxiter": 120,
    "inertia": 0.7298,
    "cognitive": 1.49618,
    "social": 1.49618,
    "velocity_clamp_fraction": 0.25,
    "tol_rate": 1e-16,
    "patience": 25,
}

# Neighbor search around PSO optimum
NEIGHBOR_STEP = 1e-3
NEIGHBOR_HALF_WIDTH = 0.02
NEIGHBOR_REFINE_LEVELS = [1e-3, 5e-4, 1e-4]

# Dead/invalid key-rate threshold
RATE_FLOOR = 1e-20
VALID_RATE_THRESHOLD = 1.01e-20

# Spline/high-resolution settings
HIGH_RES_POINTS = 1000
SPLINE_KIND = "pchip"  # "pchip", "cubic", or "smoothing"
SMOOTHING_FACTOR = 0.0


# =============================================================================
# Helper functions
# =============================================================================

def vss_parameters_from_q(q: float) -> list[float]:
    return [float(q), float(PHI0_FIXED), float(PHI1_FIXED)]


def evaluate_vss_q(loss_db: float, q: float) -> dict:
    """Evaluate TF-QKD fast rate for fixed-phase VSS with parameter q only."""
    q = float(np.clip(q, Q_BOUNDS[0], Q_BOUNDS[1]))
    params = vss_parameters_from_q(q)

    try:
        rate, e_x, e_z, p_xx = tf_qkd_fast(
            loss_db=float(loss_db),
            n_truncation=N_TRUNCATION,
            parameters=params,
            state="VSS",
            pd=P_DARK,
            f=F_RECONCILIATION,
        )
    except Exception:
        return {
            "rate": np.nan,
            "e_x": np.nan,
            "e_z": np.nan,
            "p_xx": np.nan,
            "valid": False,
        }

    valid = (
        np.isfinite(rate)
        and rate > VALID_RATE_THRESHOLD
        and np.isfinite(e_x)
        and np.isfinite(e_z)
        and np.isfinite(p_xx)
        and p_xx > 0.0
        and 0.0 <= e_x <= 1.0
    )

    return {
        "rate": float(rate) if np.isfinite(rate) else np.nan,
        "e_x": float(e_x) if np.isfinite(e_x) else np.nan,
        "e_z": float(e_z) if np.isfinite(e_z) else np.nan,
        "p_xx": float(p_xx) if np.isfinite(p_xx) else np.nan,
        "valid": bool(valid),
    }


def objective_q(loss_db: float, q: float) -> float:
    """Minimize -rate. Invalid/negative-floor points get a large penalty."""
    out = evaluate_vss_q(loss_db, q)
    rate = out["rate"]

    if not out["valid"] or not np.isfinite(rate):
        return 1e20

    return -float(rate)


def pso_optimize_q(loss_db: float, seed: int | None = None) -> dict:
    """One-dimensional PSO over q only."""
    rng = np.random.default_rng(seed)

    q_min, q_max = Q_BOUNDS
    span = q_max - q_min

    n_particles = int(PSO_SETTINGS["n_particles"])
    maxiter = int(PSO_SETTINGS["maxiter"])
    w = float(PSO_SETTINGS["inertia"])
    c1 = float(PSO_SETTINGS["cognitive"])
    c2 = float(PSO_SETTINGS["social"])
    vmax = float(PSO_SETTINGS["velocity_clamp_fraction"]) * span
    tol_rate = float(PSO_SETTINGS["tol_rate"])
    patience = int(PSO_SETTINGS["patience"])

    # Include a few deterministic physically meaningful guesses plus random particles.
    deterministic = np.array([0.05, 0.1, 0.2, 0.5, 0.8, 0.9], dtype=float)
    deterministic = deterministic[(deterministic > q_min) & (deterministic < q_max)]

    q_positions = rng.uniform(q_min, q_max, size=n_particles)
    q_positions[: min(len(deterministic), n_particles)] = deterministic[:n_particles]
    velocities = rng.uniform(-0.1 * span, 0.1 * span, size=n_particles)

    personal_best_q = q_positions.copy()
    personal_best_obj = np.array([objective_q(loss_db, q) for q in q_positions])

    best_idx = int(np.argmin(personal_best_obj))
    global_best_q = float(personal_best_q[best_idx])
    global_best_obj = float(personal_best_obj[best_idx])

    history = []
    stagnant = 0
    previous_best = global_best_obj

    for iteration in range(maxiter):
        r1 = rng.random(n_particles)
        r2 = rng.random(n_particles)

        velocities = (
            w * velocities
            + c1 * r1 * (personal_best_q - q_positions)
            + c2 * r2 * (global_best_q - q_positions)
        )
        velocities = np.clip(velocities, -vmax, vmax)

        q_positions = q_positions + velocities
        q_positions = np.clip(q_positions, q_min, q_max)

        values = np.array([objective_q(loss_db, q) for q in q_positions])

        improved = values < personal_best_obj
        personal_best_q[improved] = q_positions[improved]
        personal_best_obj[improved] = values[improved]

        best_idx = int(np.argmin(personal_best_obj))
        if personal_best_obj[best_idx] < global_best_obj:
            global_best_q = float(personal_best_q[best_idx])
            global_best_obj = float(personal_best_obj[best_idx])

        best_rate = -global_best_obj if np.isfinite(global_best_obj) and global_best_obj < 1e19 else RATE_FLOOR
        history.append(best_rate)

        if abs(global_best_obj - previous_best) < tol_rate:
            stagnant += 1
        else:
            stagnant = 0
        previous_best = global_best_obj

        if stagnant >= patience:
            break

    eval_out = evaluate_vss_q(loss_db, global_best_q)

    return {
        "q_pso": float(global_best_q),
        "rate_pso": float(eval_out["rate"]) if eval_out["valid"] else RATE_FLOOR,
        "valid_pso": bool(eval_out["valid"]),
        "iterations": iteration + 1,
        "history": history,
    }


def neighbor_search_q(loss_db: float, q_center: float) -> dict:
    """Grid-based neighbor search to make sure q does not change at the 0.001 level."""
    if not np.isfinite(q_center):
        return {
            "q_neighbor": np.nan,
            "rate_neighbor": RATE_FLOOR,
            "stable_001": False,
            "neighbor_delta": np.nan,
        }

    q_best = float(q_center)
    best_eval = evaluate_vss_q(loss_db, q_best)
    best_rate = best_eval["rate"] if best_eval["valid"] else RATE_FLOOR

    for step in NEIGHBOR_REFINE_LEVELS:
        left = max(Q_BOUNDS[0], q_best - NEIGHBOR_HALF_WIDTH)
        right = min(Q_BOUNDS[1], q_best + NEIGHBOR_HALF_WIDTH)
        grid = np.arange(left, right + 0.5 * step, step)

        for q in grid:
            out = evaluate_vss_q(loss_db, float(q))
            if out["valid"] and out["rate"] > best_rate:
                q_best = float(q)
                best_rate = float(out["rate"])

    stable_001 = abs(q_best - q_center) < 1e-3

    return {
        "q_neighbor": float(q_best),
        "rate_neighbor": float(best_rate),
        "stable_001": bool(stable_001),
        "neighbor_delta": float(q_best - q_center),
    }


def optimize_one_loss(loss_db: float, seed: int | None = None) -> dict:
    t0 = time.time()
    loss_db = float(loss_db)

    pso_out = pso_optimize_q(loss_db, seed=seed)

    if pso_out["valid_pso"]:
        neighbor_out = neighbor_search_q(loss_db, pso_out["q_pso"])
        q_final = neighbor_out["q_neighbor"]
        rate_final = neighbor_out["rate_neighbor"]
        stable_001 = neighbor_out["stable_001"]
        neighbor_delta = neighbor_out["neighbor_delta"]
    else:
        q_final = np.nan
        rate_final = RATE_FLOOR
        stable_001 = False
        neighbor_delta = np.nan

    final_eval = evaluate_vss_q(loss_db, q_final) if np.isfinite(q_final) else None
    valid_final = final_eval is not None and final_eval["valid"]

    if not valid_final:
        q_final = np.nan
        phi1_final = np.nan
        rate_final = RATE_FLOOR
        e_x = np.nan
        e_z = np.nan
        p_xx = np.nan
    else:
        phi1_final = PHI1_FIXED
        rate_final = float(final_eval["rate"])
        e_x = float(final_eval["e_x"])
        e_z = float(final_eval["e_z"])
        p_xx = float(final_eval["p_xx"])

    return {
        "loss_db": loss_db,
        "state": "VSS",
        "method": "PSO_1D_q_fixed_phases",
        "q_opt": q_final,
        "phi0": PHI0_FIXED if np.isfinite(q_final) else np.nan,
        "phi1": phi1_final,
        "rate": rate_final,
        "e_x": e_x,
        "e_z": e_z,
        "p_xx": p_xx,
        "q_pso": pso_out["q_pso"] if pso_out["valid_pso"] else np.nan,
        "rate_pso": pso_out["rate_pso"],
        "iterations": pso_out["iterations"],
        "stable_001": stable_001,
        "neighbor_delta": neighbor_delta,
        "elapsed_s": time.time() - t0,
    }


def fit_q_and_recompute_rate(df: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    """Fit q_opt on valid region and continue it through the dead zone."""
    valid = df["q_opt"].notna() & np.isfinite(df["q_opt"]) & (df["rate"] > VALID_RATE_THRESHOLD)
    valid_df = df.loc[valid].sort_values("loss_db")

    if len(valid_df) < 2:
        raise RuntimeError("Not enough valid q_opt points for fitting.")

    x = valid_df["loss_db"].to_numpy(dtype=float)
    y = valid_df["q_opt"].to_numpy(dtype=float)

    # Fit over valid values only, then extrapolate/continue over dead zone.
    if SPLINE_KIND.lower() == "pchip":
        fitter = PchipInterpolator(x, y, extrapolate=True)
    elif SPLINE_KIND.lower() == "cubic":
        fitter = CubicSpline(x, y, extrapolate=True)
    elif SPLINE_KIND.lower() == "smoothing":
        k = min(3, len(x) - 1)
        fitter = UnivariateSpline(x, y, s=SMOOTHING_FACTOR, k=k)
    else:
        raise ValueError("SPLINE_KIND must be 'pchip', 'cubic', or 'smoothing'.")

    loss_hr = np.linspace(float(df["loss_db"].min()), float(df["loss_db"].max()), HIGH_RES_POINTS)
    q_fit = np.asarray(fitter(loss_hr), dtype=float)
    q_fit = np.clip(q_fit, Q_BOUNDS[0], Q_BOUNDS[1])

    rows = []
    for loss, q in zip(loss_hr, q_fit):
        out = evaluate_vss_q(float(loss), float(q))
        rate = out["rate"] if out["valid"] else RATE_FLOOR
        rows.append({
            "loss_db": float(loss),
            "q_fit": float(q),
            "phi0": PHI0_FIXED,
            "phi1": PHI1_FIXED,
            "rate_from_q_fit": float(rate),
            "e_x": out["e_x"] if out["valid"] else np.nan,
            "e_z": out["e_z"] if out["valid"] else np.nan,
            "p_xx": out["p_xx"] if out["valid"] else np.nan,
            "valid_rate": bool(out["valid"]),
        })

    return pd.DataFrame(rows), fitter


def make_run_dir() -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "data" / "optimization_results" / f"vss_q_fixed_phase_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_settings(run_dir: Path) -> None:
    settings = {
        "state": "VSS",
        "optimized_variables": ["q"],
        "fixed_parameters": {"phi0": PHI0_FIXED, "phi1": PHI1_FIXED},
        "n_truncation": N_TRUNCATION,
        "p_dark": P_DARK,
        "f_reconciliation": F_RECONCILIATION,
        "q_bounds": Q_BOUNDS,
        "loss_db_values": LOSS_DB_VALUES.tolist(),
        "parallel_over_losses": PARALLEL_OVER_LOSSES,
        "max_workers": MAX_WORKERS,
        "seed": SEED,
        "pso_settings": PSO_SETTINGS,
        "neighbor_step": NEIGHBOR_STEP,
        "neighbor_half_width": NEIGHBOR_HALF_WIDTH,
        "neighbor_refine_levels": NEIGHBOR_REFINE_LEVELS,
        "rate_floor": RATE_FLOOR,
        "valid_rate_threshold": VALID_RATE_THRESHOLD,
        "spline_kind": SPLINE_KIND,
        "high_res_points": HIGH_RES_POINTS,
    }

    with open(run_dir / "settings.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4)

    pd.DataFrame([
        {"setting": k, "value": json.dumps(v) if isinstance(v, (list, dict, tuple)) else v}
        for k, v in settings.items()
    ]).to_csv(run_dir / "settings.csv", index=False)


def save_plots(df: pd.DataFrame, df_hr: pd.DataFrame, run_dir: Path) -> None:
    # q plot
    plt.figure(figsize=(7, 4.5))
    plt.plot(df["loss_db"], df["q_opt"], "o", label="optimized q")
    plt.plot(df_hr["loss_db"], df_hr["q_fit"], "-", label="spline-fitted q")
    plt.xlabel("Loss (dB)")
    plt.ylabel("q")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "q_opt_and_spline.png", dpi=300)
    plt.close()

    # rate plot
    plt.figure(figsize=(7, 4.5))
    valid = df["rate"] > VALID_RATE_THRESHOLD
    plt.semilogy(df.loc[valid, "loss_db"], df.loc[valid, "rate"], "o", label="optimized rate")
    plt.semilogy(df_hr["loss_db"], df_hr["rate_from_q_fit"], "-", label="rate from spline q")
    plt.xlabel("Loss (dB)")
    plt.ylabel("Key generation rate")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "rate_optimized_and_spline.png", dpi=300)
    plt.close()

    # combined plot
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(df["loss_db"], df["q_opt"], "o", label="optimized q")
    ax1.plot(df_hr["loss_db"], df_hr["q_fit"], "-", label="spline q")
    ax1.set_xlabel("Loss (dB)")
    ax1.set_ylabel("q")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.semilogy(df_hr["loss_db"], df_hr["rate_from_q_fit"], "--", label="rate from spline q")
    ax2.set_ylabel("Key generation rate")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(run_dir / "combined_q_and_rate.png", dpi=300)
    plt.close(fig)


def main() -> None:
    run_dir = make_run_dir()
    save_settings(run_dir)

    print("Run folder:", run_dir)
    print("Optimizing VSS with fixed phi0=0 and phi1=pi")
    print("Variable: q only")
    print("Loss values:", LOSS_DB_VALUES)

    all_rows = []
    failed_rows = []
    t0 = time.time()

    if PARALLEL_OVER_LOSSES:
        print(f"Running in parallel over loss points with MAX_WORKERS={MAX_WORKERS}")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for i, loss in enumerate(LOSS_DB_VALUES):
                this_seed = None if SEED is None else SEED + i
                future = executor.submit(optimize_one_loss, float(loss), this_seed)
                futures[future] = float(loss)

            for future in as_completed(futures):
                loss = futures[future]
                try:
                    row = future.result()
                    all_rows.append(row)
                    print(
                        f"loss={loss:.3f} dB | q={row['q_opt']} | "
                        f"rate={row['rate']:.6e} | stable_001={row['stable_001']} | "
                        f"elapsed={row['elapsed_s']:.2f}s"
                    )
                except Exception as exc:
                    failed_rows.append({
                        "loss_db": loss,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    })
                    print(f"FAILED loss={loss:.3f} dB: {exc}")
    else:
        print("Running serially over loss points")
        for i, loss in enumerate(LOSS_DB_VALUES):
            this_seed = None if SEED is None else SEED + i
            try:
                row = optimize_one_loss(float(loss), this_seed)
                all_rows.append(row)
                print(
                    f"loss={loss:.3f} dB | q={row['q_opt']} | "
                    f"rate={row['rate']:.6e} | stable_001={row['stable_001']} | "
                    f"elapsed={row['elapsed_s']:.2f}s"
                )
            except Exception as exc:
                failed_rows.append({
                    "loss_db": float(loss),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                print(f"FAILED loss={loss:.3f} dB: {exc}")

    elapsed_total = time.time() - t0
    print(f"Total elapsed time: {elapsed_total:.2f} s")

    df = pd.DataFrame(all_rows).sort_values("loss_db") if all_rows else pd.DataFrame()
    failed_df = pd.DataFrame(failed_rows)

    raw_path = run_dir / "vss_q_optimization_raw.csv"
    failed_path = run_dir / "failed_runs.csv"
    df.to_csv(raw_path, index=False)
    failed_df.to_csv(failed_path, index=False)

    print("Saved raw results:", raw_path)

    if df.empty:
        print("No successful optimization rows. Exiting.")
        return

    # Spline fit only on valid q values, then continue through dead zone.
    try:
        df_hr, _fitter = fit_q_and_recompute_rate(df)
        fit_path = run_dir / "vss_q_spline_high_resolution.csv"
        df_hr.to_csv(fit_path, index=False)
        save_plots(df, df_hr, run_dir)
        print("Saved high-resolution spline results:", fit_path)
        print("Saved plots in:", run_dir)
    except Exception as exc:
        fit_error_path = run_dir / "spline_fit_error.txt"
        fit_error_path.write_text(traceback.format_exc(), encoding="utf-8")
        print("Spline fitting failed:", exc)
        print("Error saved to:", fit_error_path)

    print("Done.")


if __name__ == "__main__":
    freeze_support()
    main()
