"""Effet propre des 45 décisions : L (avec LLM) contre ctrl (même dispositif, graines centrales gloutonnes).

Même solveur (sans réinjection), mêmes germes, 50 répétitions. Sortie : tableau Markdown + chiffres clés.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

HERE = Path(__file__).resolve().parent
RF = HERE.parent / "results"
I = ["250.2", "250.3", "250.4", "500.2", "500.3", "500.4", "750.2", "750.3", "750.4"]
f4 = lambda x: f"{x:.4f}".replace(".", ",")
pc = lambda x: f"{x:+.2f}".replace(".", ",") + "~\\%"
L = {i: json.loads((RF / f"seeded_llm_{i}.json").read_text())["hv_runs"] for i in I}
C = {i: json.loads((RF / f"seeded_llm_ablation_{i}.json").read_text())["hv_runs"] for i in I}
rows, better, worse, ml, mc = [], 0, 0, [], []
for i in I:
    a, b = np.mean(L[i]), np.mean(C[i])
    pg = mannwhitneyu(L[i], C[i], alternative="greater").pvalue
    pl = mannwhitneyu(L[i], C[i], alternative="less").pvalue
    better += pg < 0.05; worse += pl < 0.05
    ml.append(a); mc.append(b)
    star = "\\*" if pg < 0.05 else ("(−)" if pl < 0.05 else "")
    rows.append(f"| {i} | {f4(b)} | {f4(a)}{star} | ${pc(100 * (a / b - 1))}$ | {pg:.3g} |".replace(f"{pg:.3g}", f"{pg:.3g}".replace(".", ",")))
gain = np.mean([100 * (a / b - 1) for a, b in zip(ml, mc)])
p1 = wilcoxon(ml, mc, alternative="greater").pvalue
p2 = wilcoxon(ml, mc).pvalue
nb = sum(a > b for a, b in zip(ml, mc))
out = ["| Instance | Sans LLM | Avec LLM (*SW-CMOLS-L*) | Écart | $p$ (unilatéral) |", "|:---|---:|---:|---:|---:|"] + rows
out.append(f"| Moyenne | {f4(np.mean(mc))} | {f4(np.mean(ml))} | ${pc(gain)}$ | |")
out.append("")
out.append(f"moyenne supérieure {nb}/9 ; significatif meilleur {better}/9 ; significatif moins bon {worse}/9 ; "
           f"gain moyen {gain:+.3f} % ; Wilcoxon unilatéral p={p1:.4g} ; bilatéral p={p2:.4g}")
txt = "\n".join(out)
(RF / "llm_effect_summary.md").write_text(txt + "\n", encoding="utf-8")
print(txt)
