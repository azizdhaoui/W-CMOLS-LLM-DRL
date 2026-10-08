# -*- coding: utf-8 -*-
"""Run PISA selectors (SPEA2, NSGA-II, IBEA) on OUR exact thesis instances.

WHY PISA is the right competitor source
---------------------------------------
PISA is by Zitzler & Thiele -- the same authors as the Zitzler-Thiele knapsack
benchmark this whole thesis uses, and the same PISA whose assessment tools
(hyp_ind, eps_ind, mann-whit, wilcoxon-sign) the thesis already uses for its
metrics. Its architecture puts the ALGORITHM (selector) and the PROBLEM
(variator) in two SEPARATE PROGRAMS that talk over files.

Compliance
----------
* The SELECTORS (spea2.exe / nsga2.exe / ibea.exe = the actual algorithms) are
  the official precompiled Windows binaries, byte-for-byte as downloaded from
  sop.tik.ee.ethz.ch. NOT modified, NOT recompiled.
* The knapsack VARIATOR (the problem module) is recompiled from PISA's own C
  source with exactly two edits, both non-algorithmic:
    1. variator_user.h: `#define PISA_UNIX` -> `#define PISA_WIN`, which the
       source itself instructs on that very line ("replace with PISA_WIN if
       compiling for Windows").
    2. variator_user.c: the block that GENERATED a random instance
       (RandomInt(10,100) for weights/profits, capacities = 0.5*weightSum) now
       READS our instance from pisa_instance.txt instead.
  PISA's ORIGINAL greedy repair (profitWeightRatios / selectOrder / qsort) is
  untouched -- so unlike the pymoo runs, the repair here is the benchmark
  authors' own, not ours. This is the same data-injection role MOKPX.m plays
  for PlatEMO.

Budget: maxgen * lambda = 2000 * 100 = 200,000 evaluations -- identical to the
fair budget used for every other external competitor in the thesis.

Usage:
  python experiments/run_pisa_competitors.py --test          # 1 run, 250.2, spea2
  python experiments/run_pisa_competitors.py --all           # full campaign
Output: competitor_fronts/pisa_fronts/raw_<sel>_<inst>.txt (+ sizes_)
"""
from __future__ import annotations
import argparse, shutil, subprocess, sys, time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

MAX_WORKERS = 2  # 4 physical cores total; kept at 2 so two selector campaigns can run concurrently without oversubscribing the CPU

V3 = Path(__file__).resolve().parents[2]   # racine du dépôt
PISA = Path(os.environ.get("PISA_DIR", "external/pisa"))   # binaires PISA
VARIATOR_EXE = PISA / "ksrc" / "knapsack_c_source" / "knapsack.exe"
SELECTORS = {
    "spea2": PISA / "sel_spea2" / "spea2_win" / "spea2.exe",
    "nsga2": PISA / "sel_nsga2" / "nsga2_win" / "nsga2.exe",
    "ibea":  PISA / "sel_ibea" / "ibea_src_win" / "ibea.exe",  # recompiled natively (MSVC), no Cygwin dependency
}
SEL_PARAM_SRC = {
    "spea2": PISA / "sel_spea2" / "spea2_win" / "spea2_param.txt",
    "nsga2": PISA / "sel_nsga2" / "nsga2_win" / "nsga2_param.txt",
    "ibea":  PISA / "sel_ibea" / "ibea_win" / "IBEAEPS_param.txt",
}
WORK = V3 / "work" / "pisa"
OUT = V3 / "competitor_fronts" / "pisa_fronts"

INST = ['250.2', '250.3', '250.4', '500.2', '500.3', '500.4', '750.2', '750.3', '750.4']
POP = 100          # PISA default alpha=mu=lambda
MAXGEN = 2000      # 2000 * 100 lambda = 200,000 evaluations (thesis fair budget)
NRUNS = 20

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mokp_instance import MOKPInstance  # noqa: E402


def write_instance(inst: str, path: Path) -> tuple[int, int]:
    """pisa_instance.txt: '<dim> <length>', then weights, profits, capacities."""
    m = MOKPInstance(str(V3 / "data" / "instances" / f"{inst}.txt"))
    P = np.array(m.profits, dtype=float)
    W = np.array(m.weights, dtype=float)
    C = np.array(m.capacities, dtype=float)
    dim, length = P.shape
    parts = [f"{dim} {length}"]
    for arr in (W, P):
        for i in range(dim):
            parts.append(" ".join(f"{v:.0f}" for v in arr[i]))
    parts.append(" ".join(f"{v:.10g}" for v in C))
    path.write_text("\n".join(parts) + "\n")
    return dim, length


def one_run(sel: str, inst: str, run: int, timeout_s: int = 1800):
    """Launch variator + selector concurrently; return the final front (profits)."""
    wd = WORK / f"{sel}_{inst}_r{run}"
    if wd.exists():
        shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True)
    dim, length = write_instance(inst, wd / "pisa_instance.txt")
    (wd / "PISA_cfg").write_text(
        f"alpha {POP}\nmu {POP}\nlambda {POP}\ndim {dim}\n")
    (wd / "knapsack_param.txt").write_text(
        f"seed {run}\nlength {length}\nmaxgen {MAXGEN}\n"
        f"outputfile knapsack_output.txt\nmutation_type 1\nrecombination_type 1\n"
        f"mutation_probability 1\nrecombination_probability 0.5\n"
        f"bit_turn_probability 0.01\n")
    # selector param: reuse PISA's own defaults, only reseed
    sp = SEL_PARAM_SRC[sel].read_text().splitlines()
    sp = [(f"seed {run}" if l.strip().startswith("seed") else l) for l in sp]
    selparam = wd / f"{sel}_param.txt"
    selparam.write_text("\n".join(sp) + "\n")
    shutil.copy2(VARIATOR_EXE, wd / "knapsack.exe")
    shutil.copy2(SELECTORS[sel], wd / f"{sel}.exe")

    base = "PISA_"
    # poll interval MUST be > 0.01 per PISA's own docs; 0.01 exactly leaves the
    # selector stuck at PISA_sta=1 forever (diagnosed 2026-07-27, was the cause
    # of the IBEA campaign never starting).
    pv = subprocess.Popen([str(wd / "knapsack.exe"), "knapsack_param.txt", base, "0.02"],
                          cwd=str(wd), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    time.sleep(0.4)   # let the variator create the state file first
    ps = subprocess.Popen([str(wd / f"{sel}.exe"), selparam.name, base, "0.02"],
                          cwd=str(wd), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    t0 = time.time()
    try:
        pv.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        pv.kill(); ps.kill()
        return None, f"variator timeout after {timeout_s}s"
    try:
        ps.wait(timeout=60)
    except subprocess.TimeoutExpired:
        ps.kill()
    err = (pv.stderr.read() or b"").decode(errors="replace").strip()
    outf = wd / "knapsack_output.txt"
    if not outf.exists():
        return None, f"no output file. stderr: {err[:300]}"
    # PISA knapsack output, one archive member per line:
    #     <index> <obj_1> ... <obj_dim> <bitstring>
    # The objectives are already converted back to PROFITS (maximization) by
    # the variator; the trailing field is the decision vector and is ignored.
    rows = []
    for line in outf.read_text().splitlines():
        f = line.split()
        if len(f) < dim + 1:
            continue
        try:
            rows.append([float(v) for v in f[1:1 + dim]])
        except ValueError:
            continue
    if not rows:
        return None, f"unparsable output. stderr: {err[:300]}"
    F = np.array(rows, dtype=float)
    return F, f"ok in {time.time()-t0:.1f}s ({len(F)} pts)"


def _isnum(s: str) -> bool:
    try:
        float(s); return True
    except ValueError:
        return False


def nd_max(F: np.ndarray) -> np.ndarray:
    F = np.unique(F, axis=0)
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        if not keep[i]:
            continue
        dominated = np.all(F <= F[i], 1) & np.any(F < F[i], 1)
        keep &= ~dominated
        keep[i] = True
    return F[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sel", default=None)
    a = ap.parse_args()
    if not VARIATOR_EXE.exists():
        sys.exit(f"patched variator not built: {VARIATOR_EXE}")
    if a.test:
        F, msg = one_run("spea2", "250.2", 1, timeout_s=900)
        print("TEST:", msg)
        if F is not None:
            G = nd_max(F)
            print(f"raw {len(F)} pts -> non-dominated {len(G)}")
            print("sample:", G[:3].tolist())
        return
    if a.all:
        OUT.mkdir(parents=True, exist_ok=True)
        sels = [a.sel] if a.sel else list(SELECTORS)
        for sel in sels:
            for inst in INST:
                raw = OUT / f"raw_{sel}_{inst}.txt"
                szf = OUT / f"sizes_{sel}_{inst}.txt"
                if raw.exists():
                    print(f"{sel} {inst}: done, skip"); continue
                results = {}
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                    futs = {ex.submit(one_run, sel, inst, run): run
                            for run in range(1, NRUNS + 1)}
                    for fut in as_completed(futs):
                        run = futs[fut]
                        F, msg = fut.result()
                        if F is None:
                            print(f"  {sel} {inst} r{run} FAILED: {msg}"); continue
                        G = nd_max(F)
                        results[run] = G
                        print(f"  {sel} {inst} r{run}: {msg} -> nd {len(G)}")
                allpts  = [results[r] for r in sorted(results)]
                sizes   = [len(results[r]) for r in sorted(results)]
                if not allpts:
                    print(f"{sel} {inst}: no successful run"); continue
                S = np.vstack(allpts)
                with open(raw, "w") as fh:
                    for p in S:
                        fh.write(" ".join(f"{v:.2f}" for v in p) + "\n")
                szf.write_text("\n".join(str(s) for s in sizes) + "\n")
                print(f"{sel} {inst}: COMPLETE ({len(sizes)} runs)")
    if not (a.test or a.all):
        ap.print_help()


if __name__ == "__main__":
    main()
