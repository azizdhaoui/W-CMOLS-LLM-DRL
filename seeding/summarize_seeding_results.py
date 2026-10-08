"""Tests statistiques de l'amorçage : W-CMOLS et variantes amorcées G, L, U (50 répétitions par instance).

- chaque variante contre W-CMOLS : Mann-Whitney U unilatéral, instance par instance ;
- variantes entre elles (L contre G, U contre G, U contre L) : Mann-Whitney U unilatéral par instance,
  et Wilcoxon apparié unilatéral sur les neuf moyennes ;
- stabilité sur les instances à deux objectifs : moyenne, médiane, écart-type, minimum et nombre
  d'effondrements (répétitions sous la moitié de la médiane de W-CMOLS sur la même instance).

Sortie : results/seeding_results_summary.md
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
I = ["250.2", "250.3", "250.4", "500.2", "500.3", "500.4", "750.2", "750.3", "750.4"]
SYS = {"W": "wcmols", "G": "seeded_greedy", "L": "seeded_llm", "U": "seeded_union"}
NAME = {"W": "W-CMOLS", "G": "SW-CMOLS-G", "L": "SW-CMOLS-L", "U": "SW-CMOLS-U"}
HV = {k: {i: np.array(json.loads((RES / f"{f}_{i}.json").read_text())["hv_runs"]) for i in I} for k, f in SYS.items()}
f4 = lambda x: f"{x:.4f}".replace(".", ",")
pct = lambda x: f"{x:+.1f} %".replace(".", ",")


def mw(a, b):
    return mannwhitneyu(a, b, alternative="greater").pvalue


md = ["# Amorçage : tests statistiques", "",
      "Hypervolume moyen sur 50 répétitions. (*) : amélioration significative par rapport à W-CMOLS, "
      "Mann-Whitney U unilatéral, p < 0,05.", "",
      "| Instance | W-CMOLS | SW-CMOLS-G | SW-CMOLS-L | SW-CMOLS-U |", "|:---|---:|---:|---:|---:|"]
for i in I:
    cells = [f4(HV["W"][i].mean())] + [f4(HV[k][i].mean()) + ("*" if mw(HV[k][i], HV["W"][i]) < 0.05 else "")
                                       for k in ("G", "L", "U")]
    md.append(f"| {i} | " + " | ".join(cells) + " |")
md.append("| Moyenne | " + " | ".join(f4(np.mean([HV[k][i].mean() for i in I])) for k in SYS) + " |")
md += ["", "## Gain sur W-CMOLS", ""]
for k in ("G", "L", "U"):
    g = [100 * (HV[k][i].mean() / HV["W"][i].mean() - 1) for i in I]
    sig = sum(mw(HV[k][i], HV["W"][i]) < 0.05 for i in I)
    md.append(f"- {NAME[k]} : gain moyen {pct(np.mean(g))} (de {pct(min(g))} à {pct(max(g))}), significatif sur {sig}/9 instances.")

md += ["", "## Variantes entre elles", ""]
for a, b in (("L", "G"), ("U", "G"), ("U", "L")):
    ma, mb = [HV[a][i].mean() for i in I], [HV[b][i].mean() for i in I]
    better = [i for i in I if HV[a][i].mean() > HV[b][i].mean()]
    sig = [i for i in I if mw(HV[a][i], HV[b][i]) < 0.05]
    worse = [i for i in I if mannwhitneyu(HV[a][i], HV[b][i], alternative="less").pvalue < 0.05]
    p = wilcoxon(ma, mb, alternative="greater").pvalue
    md.append(f"- {NAME[a]} contre {NAME[b]} : moyenne supérieure sur {len(better)}/9, significativement supérieure sur "
              f"{len(sig)}/9, significativement inférieure sur {len(worse)}/9" + (f" ({', '.join(worse)})" if worse else "") +
              f" ; Wilcoxon apparié unilatéral sur les neuf moyennes p = {p:.4f}".replace(".", ",") + ".")

md += ["", "## Stabilité sur les instances à deux objectifs", "",
       "Effondrement : répétition sous la moitié de la médiane de W-CMOLS sur la même instance.", "",
       "| Instance | Système | Moyenne | Médiane | Écart-type | Minimum | Effondrements |",
       "|:---|:---|---:|---:|---:|---:|---:|"]
for i in ("250.2", "500.2", "750.2"):
    thr = np.median(HV["W"][i]) / 2
    for n, k in enumerate(("W", "L", "U")):
        h = HV[k][i]
        md.append(f"| {i if n == 0 else ''} | {NAME[k]} | {f4(h.mean())} | {f4(np.median(h))} | {f4(h.std(ddof=1))} | "
                  f"{f4(h.min())} | {int((h < thr).sum())} / {len(h)} |")

(RES / "seeding_results_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
