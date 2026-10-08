# -*- coding: utf-8 -*-
"""Run mobkp's Pareto Local Search (Jesus, Paquete, Derbel, Liefooghe -- GECCO
2021) on the nine thesis instances, as an external competitor.

NOTHING in the algorithm is adapted. The mobkp and mooutils sources are used
byte-for-byte as published at github.com/adbjesus/{mobkp,mooutils}. The only
local file is run_pls.cpp, a thin driver that reproduces the official CLI app's
PLS branch exactly (empty initial solution -> unordered_set -> FIFO queue ->
default identity item ordering -> flip_exchange_pls with a timeout). The driver
exists only because the official app links Boost.multiprecision / glpk / fmt /
CLI11, none of which are installed here; it adds no heuristic of its own.

Two phases, because PLS's only stopping rule is wall-clock time (there is no
evaluation counter to match against our 200k-evaluation protocol):

  Phase A -- MATCHED BUDGET.  timeout = the baseline's own mean runtime on that
      instance (PAPER2_METRICS.json/runtime_s_per_run). This is the headline,
      like-for-like comparison: same machine, same wall-clock, so neither side
      is handed extra compute.

  Phase B -- CEILING PROBE.  one run at a long timeout, to record where PLS
      converges on its own and what it can reach with effectively unlimited
      time. PLS terminates by itself when its queue empties, so a converged
      instance costs only its true convergence time. This pre-empts the
      objection "you did not give the competitor enough time".

Each run's wall-clock time is measured here, outside the binary, so no timing
code is injected into the algorithm.

Outputs (canonical competitor layout, same as dcnsga3/drlosemcmo/cmodrl):
    competitor_fronts/mobkp_fronts/raw_mobkp_<inst>.txt
    competitor_fronts/mobkp_fronts/sizes_mobkp_<inst>.txt
    competitor_fronts/mobkp_fronts/timing_mobkp.json
"""
from __future__ import annotations

import argparse
import os
import json
import subprocess
import sys
import time
from pathlib import Path

V3 = Path(__file__).resolve().parents[2]   # racine du dépôt
OUT = V3 / "competitor_fronts" / "mobkp_fronts"
OUT.mkdir(parents=True, exist_ok=True)

# Unmodified upstream build + the converted ZT instances.
BUILD = Path(os.environ.get("MOBKP_BUILD", "external/mobkp_build"))
EXE = BUILD / "run_pls.exe"
INSTDIR = BUILD / "instances"

INST = ["250.2", "250.3", "250.4", "500.2", "500.3", "500.4", "750.2", "750.3", "750.4"]

# Baseline mean wall-clock per run, measured on this machine.
# Source: competitor_fronts/PAPER2_METRICS.json -> runtime_s_per_run.baseline
BASELINE_S = {
    "250.2": 0.39841158390045167,
    "250.3": 0.9345488786697388,
    "250.4": 2.176331377029419,
    "500.2": 1.1798480033874512,
    "500.3": 3.427227783203125,
    "500.4": 7.72820463180542,
    "750.2": 2.5554901361465454,
    "750.3": 6.557087659835815,
    "750.4": 15.447346019744874,
}

N_RUNS = 20          # same repetition count as every other external competitor
PROBE_TIMEOUT = 180  # seconds, phase B upper bound


def one_run(inst: str, timeout_s: float, tag: str) -> tuple[list[list[int]], float, bool]:
    """Run PLS once. Returns (front, wall_seconds, converged_before_timeout)."""
    tmp = OUT / f".tmp_{inst}_{tag}.txt"
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(EXE), str(INSTDIR / f"{inst}.dat"), f"{timeout_s:.6f}", str(tmp)],
        capture_output=True, text=True,
    )
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"{inst} {tag}: exit {proc.returncode}\n{proc.stderr}")

    front = []
    with open(tmp) as fh:
        for line in fh:
            line = line.strip()
            if line:
                front.append([int(x) for x in line.split()])
    tmp.unlink(missing_ok=True)

    # PLS stops early when its queue empties; a run that finished clearly under
    # its own timeout explored the whole reachable neighbourhood.
    converged = wall < timeout_s * 0.97
    return front, wall, converged


def phase_a(instances: list[str], timing: dict) -> None:
    for inst in instances:
        budget = BASELINE_S[inst]
        fronts, walls, conv = [], [], []
        for r in range(N_RUNS):
            f, w, c = one_run(inst, budget, f"a{r}")
            fronts.append(f)
            walls.append(w)
            conv.append(c)

        raw = OUT / f"raw_mobkp_{inst}.txt"
        siz = OUT / f"sizes_mobkp_{inst}.txt"
        with open(raw, "w") as fr, open(siz, "w") as fs:
            for f in fronts:
                for pt in f:
                    fr.write(" ".join(str(v) for v in pt) + "\n")
                fs.write(f"{len(f)}\n")

        sizes = [len(f) for f in fronts]
        identical = len({tuple(sorted(map(tuple, f))) for f in fronts}) == 1
        timing.setdefault("phase_a", {})[inst] = {
            "budget_s": budget,
            "n_runs": N_RUNS,
            "wall_s": walls,
            "wall_s_mean": sum(walls) / len(walls),
            "front_sizes": sizes,
            "converged_before_timeout": conv,
            "all_runs_identical": identical,
        }
        print(f"[A] {inst}: budget={budget:.3f}s  wall={sum(walls)/len(walls):.3f}s  "
              f"|front|={sizes[0]}..{max(sizes)}  converged={sum(conv)}/{N_RUNS}  "
              f"identical={identical}", flush=True)


def phase_b(instances: list[str], timing: dict) -> None:
    for inst in instances:
        f, w, c = one_run(inst, PROBE_TIMEOUT, "b")
        raw = OUT / f"probe_mobkp_{inst}.txt"
        with open(raw, "w") as fr:
            for pt in f:
                fr.write(" ".join(str(v) for v in pt) + "\n")
        timing.setdefault("phase_b", {})[inst] = {
            "timeout_s": PROBE_TIMEOUT,
            "wall_s": w,
            "front_size": len(f),
            "converged": c,
        }
        print(f"[B] {inst}: wall={w:.2f}s  |front|={len(f)}  "
              f"converged={'YES' if c else 'NO (hit %ds cap)' % PROBE_TIMEOUT}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="*", default=INST)
    ap.add_argument("--phase", choices=["a", "b", "ab"], default="ab")
    args = ap.parse_args()

    if not EXE.exists():
        sys.exit(f"missing {EXE} -- build it first")

    tf = OUT / "timing_mobkp.json"
    timing = json.loads(tf.read_text()) if tf.exists() else {}

    if "a" in args.phase:
        phase_a(args.instances, timing)
    if "b" in args.phase:
        phase_b(args.instances, timing)

    tf.write_text(json.dumps(timing, indent=1))
    print(f"\nsaved -> {tf}")


if __name__ == "__main__":
    main()
