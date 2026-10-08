"""Chapitre 6 (et compléments des chapitres 4-5) avec les clouds finaux.

Fronts des treize concurrents, bornes et hypervolume communs (hypervolume.hv_calculator),
epsilon additif, amincissement et plafonds : competitor_data.py. Seuls nos
systèmes changent : leurs fronts viennent de results/fronts/ (solveur sans
réinjection, mêmes germes). W-CMOLS reprend ses fronts canoniques, reproduits
à l'identique par la réexécution (chapitre 4).

Sortie : results/competitor_comparison.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, mannwhitneyu, wilcoxon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
import competitor_data as D                                    # noqa: E402
from competitor_data import CH6, REF_MAX, RUN_MAX, eps_plus, gwaco_runs, nd, runs_of, thin   # noqa: E402
from hypervolume.hv_calculator import InstanceBounds, compute_hv, thin_front   # noqa: E402

FR = ROOT / "results" / "fronts"
OURS = {
    "W":  D.W_FRONTS,
    "G":  (FR / "seeded_greedy_{i}_raw.txt", FR / "seeded_greedy_{i}_sizes.txt"),
    "L":  (FR / "seeded_llm_{i}_raw.txt", FR / "seeded_llm_{i}_sizes.txt"),
    "U":  (FR / "seeded_union_{i}_raw.txt", FR / "seeded_union_{i}_sizes.txt"),
    "GQ": (FR / "agent_greedy_{i}_raw.txt", FR / "agent_greedy_{i}_sizes.txt"),
    "LQ": (FR / "agent_llm_{i}_raw.txt", FR / "agent_llm_{i}_sizes.txt"),
    "UQ": (FR / "agent_union_{i}_raw.txt", FR / "agent_union_{i}_sizes.txt"),
}
LABEL = {"W": "W-CMOLS", "G": "SW-CMOLS-G", "L": "SW-CMOLS-L", "U": "SW-CMOLS-U",
         "GQ": "SW-CMOLS-GQ", "LQ": "SW-CMOLS-LQ", "UQ": "SW-CMOLS-UQ"}
COMP = [(k, lab) for k, lab, _r, _s in CH6]
N20 = 20
I = D.INSTANCES


def bnd(inst):
    c = json.loads((ROOT / "data/reference" / f"{inst}.json").read_text())
    return InstanceBounds(low=np.array(c["bounds_low"]), high=np.array(c["bounds_high"]))


def fronts(key, inst, cap):
    if key in OURS:
        raw, sz = OURS[key]
    else:
        raw, sz = next((r, s) for k, _l, r, s in CH6 if k == key)
    return runs_of(raw, sz, inst, cap)


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for rank, j in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[j])); adj[j] = run
    return adj


def spacing_extent(f, lo, hi):
    y = (f - lo) / np.where(hi - lo == 0, 1, hi - lo)
    if len(y) > 400:
        y = thin(y, 400)
    if len(y) < 2:
        return np.nan, np.nan
    d = np.abs(y[:, None, :] - y[None, :, :]).sum(2); np.fill_diagonal(d, np.inf)
    di = d.min(1)
    return float(np.mean(np.abs(di - di.mean()))), float(np.linalg.norm(y.max(0) - y.min(0)))


def main():
    avail = [k for k in OURS if all(Path(str(OURS[k][0]).format(i=i)).exists() for i in I)]
    print("nos systèmes disponibles :", avail, flush=True)
    res = {"available": avail, "hv20": {}, "hv50_card": {}, "div": {}, "eps": {}}
    F20 = {}
    for inst in I:
        b = bnd(inst); lo, hi = D.bounds(inst)
        for k in avail + [c[0] for c in COMP]:
            fr = fronts(k, inst, N20)
            F20[(k, inst)] = fr
            res["hv20"].setdefault(k, {})[inst] = [float(compute_hv(f, b)) for f in fr]
        for k in avail:                                          # cardinalité (50) et diversité (20)
            f50 = fronts(k, inst, 50)
            res["hv50_card"].setdefault(k, {})[inst] = float(np.mean([len(f) for f in f50]))
            se = [spacing_extent(f, lo, hi) for f in F20[(k, inst)]]
            res["div"].setdefault(k, {})[inst] = [float(np.nanmean([s for s, _ in se])), float(np.nanmean([e for _, e in se]))]
        print(inst, "HV ok", flush=True)

    m = lambda k, i: float(np.mean(res["hv20"][k][i]))
    # tableaux 6.2 / 6.3 : significativité par instance
    for champ in [c for c in ("U", "UQ") if c in avail]:
        sig = {}
        for k, _ in COMP + [("W", "W")]:
            sig[k] = [bool(mannwhitneyu(res["hv20"][champ][i], res["hv20"][k][i], alternative="greater").pvalue < 0.05) for i in I]
        res[f"sig_{champ}"] = sig
        res[f"better_{champ}"] = {k: [m(champ, i) > m(k, i) for i in I] for k, _ in COMP + [("W", "W")]}
        tot_sig = sum(sum(v) for k, v in sig.items() if k != "W"); tot_b = sum(sum(v) for k, v in res[f"better_{champ}"].items() if k != "W")
        print(f"{champ}: moyenne supérieure {tot_b}/117, significatif {tot_sig}/117", flush=True)
        # contrôle par blocs
        cols = [[m(champ, i) for i in I]] + [[m(k, i) for i in I] for k, _ in COMP]
        chi, pf = friedmanchisquare(*cols)
        raw = [float(wilcoxon([m(champ, i) for i in I], [m(k, i) for i in I], alternative="greater").pvalue) for k, _ in COMP]
        res[f"friedman_{champ}"] = {"chi2": float(chi), "p": float(pf), "wilcoxon_raw": dict(zip([c[0] for c in COMP], raw)),
                                    "holm": dict(zip([c[0] for c in COMP], holm(np.array(raw)).tolist()))}
        print(f"  Friedman chi2={chi:.2f} p={pf:.3g} ; Holm max={max(holm(np.array(raw))):.4g}", flush=True)
    # tableau 6.4 : gain par nombre d'objectifs sur W (20 répétitions)
    res["gain_by_m"] = {c: {mm: float(np.mean([100 * (m(c, i) / m("W", i) - 1) for i in I if i.endswith(f".{mm}")]))
                            for mm in (2, 3, 4)} for c in ("U", "UQ") if c in avail}
    # tableau 6.5 : réduction au nombre moyen de points de MOEA-D-2WA (et contrôle MOEA/D, DCNSGA-III)
    res["equal_card"] = {}
    for comp in ("moead2wa", "moead", "dcnsga3"):
        row = {}
        for inst in I:
            b = bnd(inst); n = int(round(np.mean([len(f) for f in F20[(comp, inst)]])))
            row[inst] = {"n": n, "comp": m(comp, inst)}
            for c in [x for x in ("U", "UQ", "W") if x in avail]:
                row[inst][c] = float(np.mean([compute_hv(thin_front(f, n), b) if len(f) > n else compute_hv(f, b) for f in F20[(c, inst)]]))
        res["equal_card"][comp] = row
    # epsilon additif : référence = union non dominée de tous les systèmes du mémoire
    for inst in I:
        lo, hi = D.bounds(inst); rng = np.where(hi - lo == 0, 1.0, hi - lo)
        pool_sys = {k: F20[(k, inst)] for k in avail + [c[0] for c in COMP]}
        for key in D.GWACO_KEYS:
            r = gwaco_runs(key, inst, D.N_RUNS["gwaco"])
            if r:
                pool_sys[key] = r
        pool = np.unique(np.vstack([f for v in pool_sys.values() for f in v]), axis=0)
        R = thin(nd(thin((pool - lo) / rng, 6000)), REF_MAX)
        for k in avail + [c[0] for c in COMP]:
            res["eps"].setdefault(k, {})[inst] = [float(eps_plus(nd(thin((f - lo) / rng, RUN_MAX)), R)) for f in F20[(k, inst)]]
        print(inst, "eps ok |R| =", len(R), flush=True)
    for champ in [c for c in ("U", "UQ") if c in avail]:
        res[f"eps_sig_{champ}"] = {k: [bool(mannwhitneyu(res["eps"][champ][i], res["eps"][k][i], alternative="less").pvalue < 0.05) for i in I]
                                   for k, _ in COMP + [("W", "W")]}
    (ROOT / "results" / "competitor_comparison.json").write_text(json.dumps(res, indent=1))
    print("écrit -> results/competitor_comparison.json")


if __name__ == "__main__":
    main()
