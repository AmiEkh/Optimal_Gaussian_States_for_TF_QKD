from __future__ import annotations

import sys
import time
import json
import traceback
from pathlib import Path
from multiprocessing import freeze_support
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pandas_pd
import matplotlib.pyplot as plt


# ============================================================
# 1) Project path setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tfqkd.key_rate import tf_qkd_fast
from tfqkd.optimization import optimize_vss, optimize_gs


# ============================================================
# 2) User settings
# ============================================================

STATE = "VSS"          # "VSS" or "GS"
METHOD = "DE"         # "DE", "PSO", "BO", "L-BFGS-B", "Nelder-Mead", "Powell"

N_TRUNCATION = 6
P_DARK = 1e-8
F_RECONCILIATION = 1.16

LOSS_DB_RANGE = np.linspace(0, 120, 7)

PARALLEL_OVER_LOSSES = True
MAX_WORKERS = 3

SEED = 1

POLISH = True

# For PSO
PSO_KWARGS = {
    "n_particles": 20,
    "maxiter": 50,
}

# For DE
DE_KWARGS = {
    "maxiter": 30,
    "popsize": 8,
    "polish": POLISH,
}

# For GS, safer bounds for low truncation
GS_BOUNDS = (
    (0.0, 0.3),        # epsilon
    (0.0, 0.8),        # b
    (0.0, 2 * np.pi),  # phi0
    (0.0, 2 * np.pi),  # phi1
    (0.0, 0.8),        # s
    (0.0, 2 * np.pi),  # theta0
    (0.0, 2 * np.pi),  # theta1
)


# ============================================================
# 3) Output directory
# ============================================================

def make_run_dir() -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "data" / "optimization_results" / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_settings(run_dir: Path) -> None:
    settings = {
        "state": STATE,
        "method": METHOD,
        "n_truncation": N_TRUNCATION,
        "p_dark": P_DARK,
        "f_reconciliation": F_RECONCILIATION,
        "loss_db_range": LOSS_DB_RANGE.tolist(),
        "parallel_over_losses": PARALLEL_OVER_LOSSES,
        "max_workers": MAX_WORKERS,
        "seed": SEED,
        "pso_kwargs": PSO_KWARGS,
        "de_kwargs": DE_KWARGS,
    }

    with open(run_dir / "settings.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4)

    rows = []
    for key, value in settings.items():
        rows.append({
            "setting": key,
            "value": json.dumps(value) if isinstance(value, (list, dict, tuple)) else value,
        })

    pandas_pd.DataFrame(rows).to_csv(run_dir / "settings.csv", index=False)


# ============================================================
# 4) One-loss optimization
# ============================================================

def optimize_one_loss(loss_db: float, seed: int | None = None) -> dict:
    t0 = time.time()

    state = STATE.upper()
    method = METHOD.upper()

    if method == "PSO":
        kwargs = PSO_KWARGS.copy()
    elif method == "DE":
        kwargs = DE_KWARGS.copy()
    else:
        kwargs = {}

    if state == "VSS":
        out = optimize_vss(
            loss_db=loss_db,
            method=METHOD,
            n_truncation=N_TRUNCATION,
            pd=P_DARK,
            f=F_RECONCILIATION,
            seed=seed,
            **kwargs,
        )

    elif state == "GS":
        out = optimize_gs(
            loss_db=loss_db,
            method=METHOD,
            n_truncation=N_TRUNCATION,
            pd=P_DARK,
            f=F_RECONCILIATION,
            bounds=GS_BOUNDS,
            seed=seed,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown STATE={STATE!r}")

    elapsed = time.time() - t0

    row = {
        "loss_db": float(loss_db),
        "state": out["state"],
        "method": out["method"],
        "rate": out["rate"],
        "e_x": out["e_x"],
        "e_z": out["e_z"],
        "p_xx": out["p_xx"],
        "success": getattr(out["scipy_result"], "success", True),
        "message": str(getattr(out["scipy_result"], "message", "")),
        "elapsed_s": elapsed,
    }

    x = np.asarray(out["x"], dtype=float)

    if state == "VSS":
        names = ["q", "phi0", "phi1"]
    else:
        names = ["epsilon", "b", "phi0", "phi1", "s", "theta0", "theta1"]

    for name, value in zip(names, x):
        row[name] = float(value)

    row["x_json"] = json.dumps(x.tolist())

    return row


# ============================================================
# 5) Main runner
# ============================================================

def main() -> None:
    run_dir = make_run_dir()
    save_settings(run_dir)

    print("Run folder:", run_dir)
    print("State:", STATE)
    print("Method:", METHOD)
    print("N_truncation:", N_TRUNCATION)
    print("p_dark:", P_DARK)
    print("Loss values:", LOSS_DB_RANGE)

    print("\n=== Basic sanity check ===")

    vss_parameters = [0.9, 0.0, np.pi]
    gs_parameters = [0.0, 0.3, 0.0, np.pi, 0.1, 0.0, np.pi]

    print("VSS:", tf_qkd_fast(0.0, N_TRUNCATION, vss_parameters, "VSS", P_DARK))
    print("GS :", tf_qkd_fast(0.0, N_TRUNCATION, gs_parameters, "GS", P_DARK))

    all_rows = []
    failed_rows = []

    t_total = time.time()

    if PARALLEL_OVER_LOSSES:
        print(f"\nRunning parallel optimization with max_workers={MAX_WORKERS}")

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}

            for i, loss_db in enumerate(LOSS_DB_RANGE):
                this_seed = None if SEED is None else SEED + i
                future = executor.submit(optimize_one_loss, float(loss_db), this_seed)
                futures[future] = float(loss_db)

            for future in as_completed(futures):
                loss_db = futures[future]

                try:
                    row = future.result()
                    all_rows.append(row)
                    print(
                        f"Done loss={loss_db:.3f} dB | "
                        f"rate={row['rate']:.6e} | "
                        f"elapsed={row['elapsed_s']:.2f} s"
                    )

                except Exception as exc:
                    tb = traceback.format_exc()
                    failed_rows.append({
                        "loss_db": loss_db,
                        "error": str(exc),
                        "traceback": tb,
                    })
                    print(f"FAILED loss={loss_db:.3f} dB: {exc}")

    else:
        print("\nRunning serial optimization")

        for i, loss_db in enumerate(LOSS_DB_RANGE):
            try:
                this_seed = None if SEED is None else SEED + i
                row = optimize_one_loss(float(loss_db), this_seed)
                all_rows.append(row)
                print(
                    f"Done loss={loss_db:.3f} dB | "
                    f"rate={row['rate']:.6e} | "
                    f"elapsed={row['elapsed_s']:.2f} s"
                )

            except Exception as exc:
                tb = traceback.format_exc()
                failed_rows.append({
                    "loss_db": float(loss_db),
                    "error": str(exc),
                    "traceback": tb,
                })
                print(f"FAILED loss={loss_db:.3f} dB: {exc}")

    elapsed_total = time.time() - t_total
    print(f"\nTotal elapsed time: {elapsed_total:.2f} s")

    results_df = (
        pandas_pd.DataFrame(all_rows).sort_values("loss_db")
        if all_rows
        else pandas_pd.DataFrame()
    )

    failed_df = pandas_pd.DataFrame(failed_rows)

    results_path = run_dir / "optimization_results.csv"
    failed_path = run_dir / "failed_runs.csv"

    results_df.to_csv(results_path, index=False)
    failed_df.to_csv(failed_path, index=False)

    print("\nSaved:")
    print(results_path)
    print(failed_path)

    if not results_df.empty:
        print("\nResults:")
        print(results_df)

        plt.figure(figsize=(6, 4))
        plt.semilogy(results_df["loss_db"], results_df["rate"], marker="o")
        plt.xlabel("Loss (dB)")
        plt.ylabel("Optimized key rate")
        plt.grid(True, which="both")
        plt.tight_layout()

        fig_path = run_dir / "optimized_key_rate.png"
        plt.savefig(fig_path, dpi=300)
        print("Saved figure:", fig_path)

        plt.show()


if __name__ == "__main__":
    freeze_support()
    main()