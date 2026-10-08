"""Cloud L : N directions gloutonnes (couverture) + 5 graines centrales modifiées par le LLM.

usage :
  python build_llm_cloud.py prompts            écrit seeding/llm/prompts/<inst>_s<idx>.txt (5 par instance)
  python build_llm_cloud.py run  llm|control   construit le cloud, lance 50 runs, écrit results/pipeline/
"""
import json
import sys
import time

import numpy as np
from scipy.stats import mannwhitneyu

import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])]
import common as C

N_BY_K = {2: 30, 3: 28, 4: 56}
N_LLM = 5
N_REMOVE, N_ADD = 10, 10
P2 = C.HERE / "seeding/llm/prompts"; R2 = C.HERE / "results/pipeline"; D2 = C.HERE / "seeding/llm/decisions"


def setup(inst):
    k, n, caps, w, p = C.load(inst)
    lams = C.lambdas_n(k, N_BY_K[k])
    centre = np.full(k, 1 / k)
    order = np.argsort([np.linalg.norm(np.asarray(l) - centre) for l in lams])
    return k, n, caps, w, p, lams, sorted(order[:N_LLM].tolist())


def fmt(v):
    return "[" + ", ".join(str(int(x)) for x in v) + "]"


def write_prompts():
    P2.mkdir(exist_ok=True)
    for inst in C.INSTANCES:
        k, n, caps, w, p, lams, llm_idx = setup(inst)
        for i in llm_idx:
            lam = lams[i]; sel = C.greedy(lam, w, p, caps)
            r = C.ratio(lam, w, p, caps); v = np.asarray(lam) @ p
            free = caps - (w * sel).sum(1); f = C.objs(sel, p)
            inside, outside = np.flatnonzero(sel), np.flatnonzero(~sel)
            rem = inside[np.argsort(r[inside])][:N_REMOVE]
            add = outside[np.argsort(-r[outside])][:N_ADD]
            L = [f"Instance {inst}, seed {i}. lambda = {[round(x, 3) for x in lam]}",
                 "GOAL: raise S = sum_k lambda_k*f_k, every knapsack within capacity. No gain -> empty lists.",
                 f"Free capacity = {fmt(free)}   f = {fmt(f)}   S = {float(np.dot(lam, f)):.1f}",
                 "REMOVE candidates (in bag, lowest ratio):"]
            L += [f"  {j:<4} w={fmt(w[:, j])} v={v[j]:.1f}" for j in rem]
            L += ["ADD candidates (outside, highest ratio), need = room missing per knapsack:"]
            L += [f"  {j:<4} w={fmt(w[:, j])} v={v[j]:.1f} need={fmt(np.maximum(0, w[:, j] - free))}" for j in add]
            L += ['Answer: {"remove": [...], "add": [...]}']
            (P2 / f"{inst}_s{i:02d}.txt").write_text("\n".join(L) + "\n")


def build_seeds(inst, use_llm):
    k, n, caps, w, p, lams, llm_idx = setup(inst)
    dec = json.loads((D2 / f"{inst}.json").read_text()) if use_llm else {}
    seeds, log = [], []
    for i, lam in enumerate(lams):
        st = C.greedy(lam, w, p, caps); sel, note = st, "greedy"
        if use_llm and i in llm_idx:
            d = dec[str(i)]; ed = st.copy()
            # contrôle des identifiants : existent, retraits dans le sac, ajouts hors du sac
            rem = [j for j in d["remove"] if isinstance(j, int) and 0 <= j < n and st[j]]
            add = [j for j in d["add"] if isinstance(j, int) and 0 <= j < n and not st[j]]
            if len(rem) != len(d["remove"]) or len(add) != len(d["add"]):
                log.append({"seed": i, "invalid_ids": [j for j in d["remove"] + d["add"] if j not in rem + add]})
            ed[rem] = False; ed[add] = True
            ed = C.repair_fill(ed, lam, w, p, caps)
            a, b = C.objs(ed, p), C.objs(st, p)
            if C.dominates(b, a) or a == b:
                note = "llm rejected"
            else:
                sel, note = ed, "llm accepted"
            log.append({"seed": i, "start": b, "final": C.objs(sel, p), "note": note})
        seeds.append(np.flatnonzero(sel).tolist())
    return seeds, 1510 // len(lams) - 1, log


def run(variant):
    R2.mkdir(exist_ok=True)
    M = json.loads((C.ROOT / "data/reference/wcmols_metrics.json").read_text())
    for inst in C.INSTANCES:
        out = R2 / f"{variant}_{inst}.json"
        if out.exists():
            continue
        t0 = time.time()
        seeds, per_seed, log = build_seeds(inst, variant == "llm")
        sols = C.expand(seeds, inst, per_seed)
        build_s = time.time() - t0
        cloud = C.HERE / "clouds" / f"{ {'llm': 'llm', 'control': 'llm_ablation'}[variant]}_{inst}.txt"
        C.write_cloud(sols, cloud, f"cloud {variant}")
        hv, dt = C.solve(cloud, inst, 50, C.HERE / "work" / f"cloud_{variant}")
        g, u = M["hv_runs"]["s1_greedy"][inst], M["hv_runs"]["s1_union"][inst]
        r = {"variant": variant, "instance": inst, "hv_mean": float(np.mean(hv)), "hv_runs": hv,
             "s_per_run": dt, "build_s": build_s, "cloud": len(sols),
             "G_mean": float(np.mean(g)), "U_mean": float(np.mean(u)),
             "p_vs_G_greater": float(mannwhitneyu(hv, g, alternative="greater").pvalue),
             "p_vs_G_less": float(mannwhitneyu(hv, g, alternative="less").pvalue),
             "p_vs_U_greater": float(mannwhitneyu(hv, u, alternative="greater").pvalue),
             "llm_log": log}
        out.write_text(json.dumps(r, indent=1))
        print(f"{inst}: {variant} HV {r['hv_mean']:.4f} | G {r['G_mean']:.4f} ({100*(r['hv_mean']/r['G_mean']-1):+.2f}%) "
              f"| U {r['U_mean']:.4f} | p>G={r['p_vs_G_greater']:.3g} p<G={r['p_vs_G_less']:.3g} | {dt:.2f}s/run", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "prompts":
        write_prompts()
    else:
        run(sys.argv[2])
