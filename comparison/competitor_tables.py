"""Tableaux Markdown du chapitre 6 (et 4.5 / 5.2) à partir de results/competitor_comparison.json.

Sortie : results/competitor_tables.md (tableaux prêts à coller + chiffres clés pour le texte).
Les temps des concurrents viennent de results/competitor_times.json (tableau 6.7).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R = json.loads((ROOT / "results" / "competitor_comparison.json").read_text())
I = ["250.2", "250.3", "250.4", "500.2", "500.3", "500.4", "750.2", "750.3", "750.4"]
COMP = [("moead", "MOEA/D"), ("eagmoead", "EAG-MOEA/D"), ("cmoead", "C-MOEA-D"), ("moead2wa", "MOEA-D-2WA"),
        ("gwasfga", "GWASF-GA"), ("moeadgr", "MOEA/D-GR"), ("ibea", "IBEA"), ("spea2", "SPEA2"),
        ("nsga2", "NSGA-II"), ("mobkp", "mobkp/PLS"), ("dcnsga3", "DCNSGA-III"), ("cmodrl", "CMODRL"),
        ("idbea", "I-DBEA")]
DAG = {"eagmoead", "gwasfga", "moeadgr", "ibea", "spea2", "nsga2"}
LAB = dict(COMP) | {"W": "*W-CMOLS*", "U": "*SW-CMOLS-U*", "UQ": "*SW-CMOLS-UQ*", "G": "SW-CMOLS-G",
                    "L": "SW-CMOLS-L", "GQ": "SW-CMOLS-GQ", "LQ": "SW-CMOLS-LQ"}
f4 = lambda x: f"{x:.4f}".replace(".", ",")
f2 = lambda x: f"{x:.2f}".replace(".", ",")
pc = lambda x: (f"{x:+.1f}".replace(".", ",")) + "~\\%"
m = lambda k, i: float(np.mean(R["hv20"][k][i]))
mm = lambda k: float(np.mean([m(k, i) for i in I]))
out = []


def hv_table(champ, num):
    order = sorted([c for c, _ in COMP], key=mm, reverse=True)
    rows = ["W", champ] + order
    best = {i: max(rows, key=lambda k: m(k, i)) for i in I}
    bestmean = max(rows, key=mm)
    out.append(f"\n### Tableau {num} ({champ})\n")
    out.append("| Système | " + " | ".join(I) + " | Moyenne | Sign. |")
    out.append("|:---|" + "---:|" * 10 + ":---:|")
    for k in rows:
        cells = [("**" + f4(m(k, i)) + "**") if best[i] == k else f4(m(k, i)) for i in I]
        mean = ("**" + f4(mm(k)) + "**") if k == bestmean else f4(mm(k))
        sig = "—" if k in ("W", champ) else f"{sum(R[f'sig_{champ}'][k])}/9"
        lab = LAB[k] + (" †" if k in DAG else "")
        out.append(f"| {lab} | " + " | ".join(cells) + f" | {mean} | {sig} |")
    tb = sum(sum(v) for k, v in R[f"better_{champ}"].items() if k != "W")
    ts = sum(sum(v) for k, v in R[f"sig_{champ}"].items() if k != "W")
    out.append(f"\n{champ} : moyenne supérieure {tb}/117 ; significatif {ts}/117")
    miss = [(k, i) for k, _ in COMP for n, i in enumerate(I) if not R[f"sig_{champ}"][k][n]]
    out.append(f"{champ} : couples non significatifs : {miss}")
    worse = [(k, i, f4(m(champ, i)), f4(m(k, i))) for k, _ in COMP for i in I if m(champ, i) <= m(k, i)]
    out.append(f"{champ} : couples où la moyenne n'est pas supérieure : {worse}")
    gaps = sorted([(100 * (m(champ, i) / m(k, i) - 1), k, i) for k, _ in COMP for i in I])
    out.append(f"{champ} : écart minimal {gaps[0]} ; maximal {gaps[-1]}")
    fr = R[f"friedman_{champ}"]
    out.append(f"{champ} : Friedman chi2={fr['chi2']:.2f} p={fr['p']:.3g} ; Wilcoxon bruts {fr['wilcoxon_raw']} ; Holm {fr['holm']}")
    out.append(f"{champ} vs W-CMOLS : sign {sum(R[f'sig_{champ}']['W'])}/9")


hv_table("U", "6.2")
hv_table("UQ", "6.3")
beatW = [lab for c, lab in COMP if mm(c) > mm("W")]
out.append(f"\nConcurrents au-dessus de W-CMOLS (20 runs, {f4(mm('W'))}) : {beatW}")
out.append("IBEA > W par instance : " + str(sum(m('ibea', i) > m('W', i) for i in I)))
best_comp = {i: max([c for c, _ in COMP], key=lambda k: m(k, i)) for i in I}
out.append(f"Meilleur concurrent par instance : {best_comp}")

# 6.4 gain par nombre d'objectifs
out.append("\n### Tableau 6.4\n")
out.append("| Nombre d'objectifs | Instances concernées | Gain SW-CMOLS-U | Gain SW-CMOLS-UQ |")
out.append("|:---|:---|---:|---:|")
for k in (2, 3, 4):
    ins = ", ".join(i for i in I if i.endswith(f".{k}"))
    out.append(f"| {k} objectifs | {ins} | ${pc(R['gain_by_m']['U'][str(k)])}$ | ${pc(R['gain_by_m']['UQ'][str(k)])}$ |")
out.append("Gains 2-obj U par instance : " + str({i: round(100 * (m('U', i) / m('W', i) - 1), 1) for i in I if i.endswith('.2')}))
out.append("Gains 2-obj UQ par instance : " + str({i: round(100 * (m('UQ', i) / m('W', i) - 1), 1) for i in I if i.endswith('.2')}))

# 6.5 cardinalité égale
ec = R["equal_card"]
out.append("\n### Tableau 6.5\n")
out.append("| Instance | Points | MOEA-D-2WA | SW-CMOLS-U *(réduit)* | SW-CMOLS-UQ *(réduit)* | SW-CMOLS-U *(complet)* |")
out.append("|:---|---:|---:|---:|---:|---:|")
for i in I:
    r = ec["moead2wa"][i]; b = max(("comp", "U", "UQ"), key=lambda k: r[k])
    cell = lambda k: ("**" + f4(r[k]) + "**") if b == k else f4(r[k])
    out.append(f"| {i} | {r['n']} | {cell('comp')} | {cell('U')} | {cell('UQ')} | {f4(m('U', i))} |")
for comp in ("moead2wa", "moead", "dcnsga3"):
    wins = {c: sum(ec[comp][i][c] > ec[comp][i]["comp"] for i in I) for c in ("U", "UQ", "W")}
    out.append(f"cardinalité égale vs {comp} : {wins} ; écarts U % : " +
               str({i: round(100 * (ec[comp][i]['U'] / ec[comp][i]['comp'] - 1), 1) for i in I}))
out.append("W réduit vs 2WA sur 250.4 : " + f4(ec["moead2wa"]["250.4"]["W"]))
card = R["hv50_card"]
out.append("Tailles moyennes des fronts (50 runs) : " + str({k: (round(min(v.values())), round(max(v.values()))) for k, v in card.items()}))

# 6.6 epsilon
E = {k: float(np.mean([np.mean(R["eps"][k][i]) for i in I])) for k in R["eps"]}
allk = ["U", "W"] + [c for c, _ in COMP]
rank_eps = {k: n + 1 for n, k in enumerate(sorted(allk, key=lambda k: E[k]))}
rank_hv = {k: n + 1 for n, k in enumerate(sorted(allk, key=mm, reverse=True))}
out.append("\n### Tableau 6.6\n")
out.append("| Système | $I_{\\varepsilon+}$ moyen | Rang $\\varepsilon$ | Rang HV | SW-CMOLS-U l'emporte |")
out.append("|:---|---:|---:|---:|---:|")
for k in sorted(allk, key=lambda k: E[k]):
    if k == "U":
        out.append(f"| *SW-CMOLS-U* | **{f4(E[k])}** | **{rank_eps[k]}** | **{rank_hv[k]}** | — |")
        continue
    bet = sum(np.mean(R["eps"]["U"][i]) < np.mean(R["eps"][k][i]) for i in I)
    sg = sum(R["eps_sig_U"][k])
    lab = ("W-CMOLS" if k == "W" else LAB[k]) + (" †" if k in DAG else "")
    out.append(f"| {lab} | {f4(E[k])} | {rank_eps[k]} | {rank_hv[k]} | {bet}/9 ({sg} sign.) |")
for champ in ("U", "UQ"):
    tb = sum(sum(np.mean(R["eps"][champ][i]) < np.mean(R["eps"][k][i]) for i in I) for k in [c for c, _ in COMP] + ["W"])
    ts = sum(sum(R[f"eps_sig_{champ}"][k]) for k in [c for c, _ in COMP] + ["W"])
    miss = [(k, i) for k in [c for c, _ in COMP] + ["W"] for n, i in enumerate(I) if not R[f"eps_sig_{champ}"][k][n]]
    out.append(f"eps {champ} : {f4(E[champ])} ; meilleur sur {tb}/126, sign {ts}/126 ; non sign : {miss}")
out.append("eps autres nôtres : " + str({k: f4(E[k]) for k in E if k in ("G", "L", "GQ", "LQ")}))

# 6.7 temps
T = {}
for k, f in (("W", "wcmols"), ("G", "seeded_greedy"), ("L", "seeded_llm"), ("U", "seeded_union"),
             ("GQ", "agent_greedy"), ("LQ", "agent_llm"), ("UQ", "agent_union")):
    T[k] = [json.loads((ROOT / "results" / f"{f}_{i}.json").read_text())["s_per_run"] for i in I]
CT = json.loads((ROOT / "results" / "competitor_times.json").read_text(encoding="utf-8"))
out.append("\n### Tableau des temps\n")
out.append("| Système | " + " | ".join(I) + " | Moyenne | Env. |")
out.append("|:---|" + "---:|" * 10 + ":---|")
for k in ("W", "G", "L", "U", "GQ", "LQ", "UQ"):
    out.append(f"| {LAB[k]} | " + " | ".join(f2(x) for x in T[k]) + f" | {f2(np.mean(T[k]))} | Cython |")
for c, lab in COMP:
    t = CT[lab]
    cells = ["—" if x is None else f2(x) for x in t["s_per_run"] + [t["mean"]]]
    out.append(f"| {lab} | " + " | ".join(cells) + f" | {t['env']} |")

(ROOT / "results" / "competitor_tables.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
