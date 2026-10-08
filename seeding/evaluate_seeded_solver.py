"""Évaluation de W-CMOLS et des variantes amorcées (50 exécutions chronométrées par instance).

usage : python evaluate_seeded_solver.py <system> <nbl> <inst> [inst ...]
  system : baseline | G | L | ctrl | U2
    baseline : archive vide (W-CMOLS)
    G        : cloud glouton canonique
    L        : nouveau L (clouds/llm_<inst>.txt, 45 décisions)
    ctrl     : même dispositif sans LLM (clouds/llm_ablation_<inst>.txt)
    U2       : 755 G + 755 nouveau L (échantillonnage régulier, comme gen_union_seeds.py)
Sortie : results/<system>_nbl<nbl>_<inst>.json
"""
import json
import sys
import time

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])]
import common as C

system, nbl, insts = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
OUT = C.HERE / "results"; OUT.mkdir(exist_ok=True)
FR = OUT / "fronts"; FR.mkdir(exist_ok=True)
import os as _os
RUNS = int(_os.environ.get("RUNS", "50"))
SUFFIX = _os.environ.get("SUFFIX", "")
H2 = C.ROOT


def read(p):
    return [l.strip() for l in open(p, encoding="utf-8", errors="replace") if l.strip() and not l.startswith("#")]


def stride(sols, k):
    step = len(sols) / k
    return [sols[int(i * step)] for i in range(k)]


def cloud_for(inst):
    if system == "baseline":
        return None
    if system == "G":
        return H2 / "clouds" / f"greedy_{inst}.txt"
    if system == "L":
        return C.HERE / "clouds" / f"llm_{inst}.txt"
    if system == "ctrl":
        return C.HERE / "clouds" / f"llm_ablation_{inst}.txt"
    if system == "U2":
        g = read(H2 / "clouds" / f"greedy_{inst}.txt")
        l = read(C.HERE / "clouds" / f"llm_{inst}.txt")
        merged = stride(g, 755) + stride(l, 755)
        p = C.HERE / "clouds" / f"union_{inst}.txt"
        lines = ["# U2 : 755 G + 755 nouveau L", f"# Total: {len(merged)}"]
        for i, s in enumerate(merged, 1):
            lines += [f"# Sol {i}", s]
        p.write_text("\n".join(lines) + "\n")
        return p
    raise SystemExit(f"unknown system {system}")


Q = {}   # variantes configurées par l'agent : voir configuration/evaluate_agent.py
params_of = {}
if system in Q:
    base, f = Q[system]
    ev = json.loads((C.ROOT / "models" / f).read_text())
    params_of = {k: v["params"] for k, v in ev["instances"].items()}
    system_cloud = base
else:
    system_cloud = system

NAME = {"baseline": "wcmols", "G": "seeded_greedy", "L": "seeded_llm", "U2": "seeded_union", "ctrl": "seeded_llm_ablation"}
for inst in insts:
    tag = "dqn" if system in Q else f"nbl{nbl}"
    name = NAME[system] + ("" if nbl == 100 else f"_nbl{nbl}")
    out = OUT / f"{name}{SUFFIX}_{inst}.json"
    if out.exists():
        print(f"{inst}: {system} {tag} déjà fait", flush=True); continue
    system_saved, system = system, system_cloud
    cloud = cloud_for(inst)
    system = system_saved
    pr = params_of.get(inst)
    t0 = time.time()
    hv, dt = C.solve(cloud, inst, RUNS, C.HERE / "work" / f"final_{system}", nbl=nbl, reinject=False, params=pr,
                      save_as=FR / f"{name}_{inst}")
    r = {"system": system, "nbl": (pr or {}).get("NBL", nbl), "params": pr, "instance": inst, "runs": len(hv),
         "hv_mean": float(np.mean(hv)),
         "hv_runs": hv, "s_per_run": dt, "solver_total_s": dt * len(hv),
         "wall_s_incl_hv": time.time() - t0, "reinjection": False, "f_mut": -1}
    out.write_text(json.dumps(r, indent=1))
    print(f"{inst}: {system} {tag} HV {r['hv_mean']:.4f} | {dt:.2f} s/run | solver {dt*len(hv)/60:.1f} min", flush=True)
