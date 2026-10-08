"""
Phase 1 — 1D response profiles per parameter, WITHOUT seeds (competitor setting).

Goal (F6 fix): choose the v3 grid VALUES from measured data instead of ad hoc.
For each of 3 representative instances (one per objective count), vary ONE param
over a wide range (others fixed at the DQN-v2 chosen combo) and measure HV + time:

    NBL   : 20, 30, 50, 70, 100, 130, 160, 200   <- censoring test above 100
    alpha : 5, 8, 10, 15, 20, 25, 30, 40
    L     : 2, 3, 5, 8, 10
    kappa : 0.05, 0.1, 0.15, 0.2, 0.3            <- censoring test above 0.2

Grid design rules applied afterwards (documented in GRID_DESIGN_V3.md):
    - place grid knots where the response curve actually bends
    - extend ranges where the curve is still rising at the old boundary (censoring)
    - drop dead zones (flat/worse regions)
    - always include the Ben Mansour defaults (10, 100, 5, 0.05)

No seeds: this is the COMPETITOR landscape (DQN params alone vs baseline).
Writes results/grid_sensitivity_profiles.json.
Estimated runtime ~1h40-2h.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

V3_ROOT = Path(__file__).resolve().parents[1]          # racine du dépôt
BASELINE = V3_ROOT / "work" / "grid_sensitivity"      # fichiers de sortie du solveur
BASELINE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(V3_ROOT / "solver"))
sys.path.insert(0, str(V3_ROOT))

import moacp_mut  # type: ignore[import-not-found]

from hypervolume.hv_calculator import InstanceBounds, compute_hv, parse_all_runs

N_RUNS = 12

# (instance, weights file, fixed combo = DQN-v2 chosen params)
SWEEPS = [
    ("750.2", "Weights_2obj_FQ200.txt", dict(alpha=25, NBL=50, L=3, kappa=0.2)),
    ("500.3", "Weights_3obj_FQ100.txt", dict(alpha=15, NBL=100, L=8, kappa=0.2)),
    ("250.4", "Weights_4obj_FQ40.txt", dict(alpha=20, NBL=50, L=8, kappa=0.05)),
]

PROFILES = {
    "NBL":   [20, 30, 50, 70, 100, 130, 160, 200],
    "alpha": [5, 8, 10, 15, 20, 25, 30, 40],
    "L":     [2, 3, 5, 8, 10],
    "kappa": [0.05, 0.1, 0.15, 0.2, 0.3],
}

OUT = V3_ROOT / "results" / "grid_sensitivity_profiles.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_config(instance_path: Path, weights_path: Path, cf: dict) -> tuple[list[np.ndarray], float]:
    prev = os.getcwd()
    try:
        os.chdir(BASELINE)
        for f in ("all_runs.txt", "pareto_sizes.txt"):
            try:
                (BASELINE / f).unlink()
            except FileNotFoundError:
                pass
        moacp_mut.set_runtime_params(alpha_val=cf["alpha"], L_val=cf["L"], kappa_val=cf["kappa"])
        moacp_mut.set_iterations(cf["NBL"])
        t0 = time.time()
        moacp_mut.run_moacp_ex(
            str(instance_path).encode(), str(weights_path).encode(),
            b"",  # NO SEEDS — competitor landscape
            N_RUNS, -1,
        )
        elapsed = (time.time() - t0) / N_RUNS
        fronts = parse_all_runs(BASELINE / "all_runs.txt", BASELINE / "pareto_sizes.txt")
        return fronts, elapsed
    finally:
        os.chdir(prev)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for inst, weights_file, fixed in SWEEPS:
        log(f"=== {inst} === fixed: {fixed}")
        cache = json.loads((V3_ROOT / "data" / "reference" / f"{inst}.json").read_text())
        bounds = InstanceBounds(low=np.array(cache["bounds_low"]),
                                high=np.array(cache["bounds_high"]))
        instance_path = V3_ROOT / "data" / "instances" / f"{inst}.txt"
        weights_path = V3_ROOT / "data" / "instances" / weights_file

        # build config list: union of all profile points, dedup on full tuple
        seen: set = set()
        configs: list[tuple[str, str, dict]] = []  # (param, label, config)
        for param, values in PROFILES.items():
            for v in values:
                cf = {**fixed, param: v}
                key = (cf["alpha"], cf["NBL"], cf["L"], cf["kappa"])
                if key in seen:
                    continue
                seen.add(key)
                configs.append((param, f"{param}={v}", cf))

        rows = []
        for param, label, cf in configs:
            try:
                fronts, t_run = run_config(instance_path, weights_path, cf)
                hvs = np.array([compute_hv(f, bounds) for f in fronts])
                row = {
                    "param": param, "label": label, "config": cf,
                    "hv_mean": float(hvs.mean()), "hv_std": float(hvs.std()),
                    "time_per_run": float(t_run),
                    "hv_runs": hvs.tolist(),
                }
                log(f"  {label:<12} HV={row['hv_mean']:.4f}±{row['hv_std']:.4f}  {t_run:.2f}s/run")
            except Exception as e:  # a pathological value must not kill the night
                row = {"param": param, "label": label, "config": cf, "error": str(e)}
                log(f"  {label:<12} ERROR: {e}")
            rows.append(row)
            # incremental save after every config
            all_results[inst] = {"fixed": fixed, "rows": rows}
            OUT.write_text(json.dumps(all_results, indent=2))

    log(f"DONE. Saved: {OUT}")


if __name__ == "__main__":
    main()
