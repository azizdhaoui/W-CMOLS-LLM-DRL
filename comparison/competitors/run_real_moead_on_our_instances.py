# -*- coding: utf-8 -*-
"""Run the REAL MOEA/D (Li & Zhang, the authors' own C++ knapsack code) on OUR
EXACT thesis instances -- instead of relying on the precomputed POF/ results
that shipped with the .rar.

WHY this is necessary (verified 2026-07-26): the instances bundled with the
reference code are NOT bit-identical to ours. Differences found:
  - capacities: theirs are floored integers, ours are exact sum(W)/2 (e.g.
    6962.5 vs 6962). MATHEMATICALLY IRRELEVANT -- weights are integers, so
    totalW <= 6962.5 accepts exactly the same solutions as totalW <= 6962.
  - 500.3: W[obj2,item248] theirs 48 vs ours 59
  - 750.4: W/P swapped-looking on items 573 & 594 of obj4
    (theirs W=87,P=21 / W=48,P=16 ; ours W=21,P=84 / W=16,P=48)
So the shipped MOEA/D results for 500.3 and 750.4 were produced on slightly
DIFFERENT data. Our instance files are authoritative for this thesis, so the
algorithm must be re-run on them.

The algorithm itself is NOT modified: we only translate our instance data into
the input format its own loader expects (same role as MOKPX.m for PlatEMO),
and we reuse its own weight-vector files and its own population sizes from
TestInstances.txt.

Usage:
  python experiments/run_real_moead_on_our_instances.py --prepare   # build run dir
  python experiments/run_real_moead_on_our_instances.py --run       # execute
Output: <rundir>/POF/POF_MOEAD_KS<inst>_R<run>.dat  (the algorithm's own format)
"""
from __future__ import annotations
import argparse, shutil, subprocess, sys, time
import os
from pathlib import Path
import numpy as np

V3 = Path(__file__).resolve().parents[2]   # racine du dépôt
SRC_REF = Path(os.environ.get("MOEAD_SRC", "external/moead"))   # code C++ des auteurs (poids, liste des instances)
RUNDIR = V3 / "work" / "moead"
EXE_SRC = Path(os.environ.get("MOEAD_EXE", "external/moead/moead_kp.exe"))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mokp_instance import MOKPInstance  # noqa: E402

INST = ['250.2', '250.3', '250.4', '500.2', '500.3', '500.4', '750.2', '750.3', '750.4']


def write_ref_format(inst: str, out: Path) -> None:
    """Translate our instance into the reference loader's text format.

    Reference format (LoadInstance in knapsackProblem.cpp):
        knapsack problem specification (K knapsacks, n items)
        =
        knapsack 1:
         capacity: +C1
         item 1:
          weight: +w
          profit: +p
         ...
        =                  <-- REQUIRED separator before EVERY "knapsack k:"
        knapsack 2:
         ...

    The "=" before each knapsack block is MANDATORY, not decorative: after the
    item loop, LoadInstance does one extra `Stream >> s` before re-testing
    `s == "knapsack"`. With the "=" present it lands on "knapsack"; without it
    it lands on the block index ("2:"), the while-loop exits, and ONLY THE FIRST
    KNAPSACK IS LOADED -- the run then silently degenerates to a 1-objective
    problem and the archive collapses to a single point. (Diagnosed 2026-07-26
    after a first attempt produced exactly 1 point per run.)

    Capacity is written as floor(C): our C may be x.5, but with integer weights
    floor(C) admits exactly the same feasible set.
    """
    m = MOKPInstance(str(V3 / "data" / "instances" / f"{inst}.txt"))
    P = np.array(m.profits, dtype=int)      # (K, n)
    W = np.array(m.weights, dtype=int)      # (K, n)
    C = np.array(m.capacities, dtype=float)
    K, n = P.shape
    L = [f"knapsack problem specification ({K} knapsacks, {n} items)"]
    for k in range(K):
        L.append("=")                      # mandatory separator -- see docstring
        L.append(f"knapsack {k + 1}:")
        L.append(f" capacity: +{int(np.floor(C[k]))}")
        for j in range(n):
            L.append(f" item {j + 1}:")
            L.append(f"  weight: +{W[k, j]}")
            L.append(f"  profit: +{P[k, j]}")
    out.write_text("\n".join(L) + "\n")


def prepare() -> None:
    if not EXE_SRC.exists():
        sys.exit(f"compiled exe not found: {EXE_SRC}\nBuild it first (MSVC cl).")
    RUNDIR.mkdir(parents=True, exist_ok=True)
    (RUNDIR / "Instances").mkdir(exist_ok=True)
    (RUNDIR / "POF").mkdir(exist_ok=True)
    shutil.copy2(EXE_SRC, RUNDIR / "moead_kp.exe")
    # the algorithm's OWN weight vectors and its OWN population sizes
    shutil.copytree(SRC_REF / "weights", RUNDIR / "weights", dirs_exist_ok=True)
    shutil.copy2(SRC_REF / "TestInstances.txt", RUNDIR / "TestInstances.txt")
    for inst in INST:
        items, k = inst.split(".")
        write_ref_format(inst, RUNDIR / "Instances" / f"knapsack_{items}.{k}")
        print(f"wrote Instances/knapsack_{items}.{k}  (from our data/{inst}.txt)")
    print(f"\nprepared: {RUNDIR}")
    print("weight vectors + population sizes: the algorithm's own (unchanged)")


def run() -> None:
    exe = RUNDIR / "moead_kp.exe"
    if not exe.exists():
        sys.exit("run --prepare first")
    t0 = time.time()
    p = subprocess.run([str(exe)], cwd=str(RUNDIR), capture_output=True, text=True)
    print(f"exit={p.returncode}  elapsed={(time.time()-t0)/60:.1f} min")
    tail = (p.stdout or "")[-1500:]
    if tail:
        print("--- stdout tail ---"); print(tail)
    if p.stderr:
        print("--- stderr tail ---"); print(p.stderr[-800:])
    got = sorted((RUNDIR / "POF").glob("POF_MOEAD_KS*_R*.dat"))
    print(f"\nPOF files produced: {len(got)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.prepare:
        prepare()
    if a.run:
        run()
    if not (a.prepare or a.run):
        ap.print_help()
