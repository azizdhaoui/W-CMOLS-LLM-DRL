"""Témoin sans LLM avec N graines (λ en réseau simplexe), budget de cloud ≤ 1510.

usage : python screen_direction_count.py N inst [inst ...]
"""
import json
import sys

import numpy as np
from scipy.stats import mannwhitneyu

import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])]
import common as C

n_req = int(sys.argv[1]); insts = sys.argv[2:]
M = json.loads((C.ROOT / "data/reference/wcmols_metrics.json").read_text())
OUT = C.HERE / "results"; CL = C.HERE / "clouds"
for inst in insts:
    k, n, caps, w, p = C.load(inst)
    lams = C.lambdas_n(k, n_req)
    per_seed = 1510 // len(lams) - 1
    seeds = [np.flatnonzero(C.greedy(l, w, p, caps)).tolist() for l in lams]
    sols = C.expand(seeds, inst, per_seed)
    cloud = CL / f"direction_screening_N{len(lams)}_{inst}.txt"
    C.write_cloud(sols, cloud, f"control N={len(lams)}")
    RUNS = int(__import__("os").environ.get("RUNS", "50"))
    hv, dt = C.solve(cloud, inst, RUNS, C.HERE / "work" / f"direction_screening_N{len(lams)}_{inst}")
    g = M["hv_runs"]["s1_greedy"][inst]
    r = {"instance": inst, "n_seeds": len(lams), "cloud": len(sols),
         "distinct": len({tuple(s) for s in sols}), "hv_mean": float(np.mean(hv)), "hv_runs": hv,
         "G_mean": float(np.mean(g)), "s_per_run": dt,
         "p_vs_G_greater": float(mannwhitneyu(hv, g, alternative="greater").pvalue),
         "p_vs_G_less": float(mannwhitneyu(hv, g, alternative="less").pvalue)}
    (OUT / "screening" / f"direction_screening_N{len(lams)}_{inst}_r{RUNS}.json").write_text(json.dumps(r, indent=1))
    print(f"{inst}: N={len(lams)} cloud={len(sols)} distinct={r['distinct']} HV {r['hv_mean']:.4f} vs G {r['G_mean']:.4f} "
          f"({100*(r['hv_mean']/r['G_mean']-1):+.2f}%) p>G={r['p_vs_G_greater']:.3g} p<G={r['p_vs_G_less']:.3g}", flush=True)
