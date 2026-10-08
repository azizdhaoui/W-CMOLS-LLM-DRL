"""Chiffres du chapitre 5 pour la contribution 2 finale (politiques réentraînées, 2026-09-25).

Lit les résultats canoniques (50 répétitions) : W-CMOLS, variantes statiques G/L/U,
variantes contrôlées GQ/LQ/UQ et, s'il existe, le contrôle sans cloud WQ.
Mêmes tests que la version publiée :
  - Mann-Whitney U unilatéral contre W-CMOLS (tableau 5.1) ;
  - variante contrôlée contre sa variante statique, dans les deux sens, par instance,
    puis Wilcoxon apparié bilatéral sur les neuf moyennes ;
  - temps cumulés et respect du budget (0,8 x temps de W-CMOLS).
Sortie : results/C2_CH5_FINAL.md
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

HERE = Path(__file__).resolve().parent
RF = HERE.parent / "results"
I = ["250.2", "250.3", "250.4", "500.2", "500.3", "500.4", "750.2", "750.3", "750.4"]
TAG = {"W": "baseline_nbl100", "G": "G_nbl100", "L": "L_nbl100", "U": "U2_nbl100",
       "GQ": "GQ_dqn", "LQ": "LQ_dqn", "UQ": "UQ_dqn", "WQ": "WQ_dqn",
       "GQp": "GQ_dqn_pub", "LQp": "LQ_dqn_pub", "UQp": "UQ_dqn_pub"}
D = {k: {i: json.loads((RF / f"{t}_{i}.json").read_text()) for i in I}
     for k, t in TAG.items() if all((RF / f"{t}_{i}.json").exists() for i in I)}
hv = lambda k, i: float(np.mean(D[k][i]["hv_runs"]))
runs = lambda k, i: D[k][i]["hv_runs"]
T = lambda k: sum(D[k][i]["s_per_run"] for i in I)
fr = lambda x: f"{x:.4f}".replace(".", ",")
pc = lambda x: f"{x:+.2f} %".replace(".", ",")
out = []

# Tableau 5.1
out.append("## Tableau 5.1 (HV moyen, 50 répétitions ; * = MW unilatéral > W-CMOLS, p < 0,05)\n")
cols = ["W", "G", "GQ", "L", "LQ", "U", "UQ"]
out.append("| Instance | " + " | ".join(cols) + " |")
out.append("|" + "---|" * (len(cols) + 1))
for i in I:
    best = max(hv(k, i) for k in cols)
    cells = []
    for k in cols:
        c = fr(hv(k, i))
        if k.endswith("Q") and mannwhitneyu(runs(k, i), runs("W", i), alternative="greater").pvalue < 0.05:
            c += "*"
        if abs(hv(k, i) - best) < 5e-5:
            c = f"**{c}**"
        cells.append(c)
    out.append(f"| {i} | " + " | ".join(cells) + " |")
out.append("| Moyenne | " + " | ".join(fr(np.mean([hv(k, i) for i in I])) for k in cols) + " |\n")

# Gains contre W-CMOLS
for k in ("GQ", "LQ", "UQ"):
    g = np.mean([100 * (hv(k, i) / hv("W", i) - 1) for i in I])
    w = sum(mannwhitneyu(runs(k, i), runs("W", i), alternative="greater").pvalue < 0.05 for i in I)
    out.append(f"- {k} vs W-CMOLS : gain moyen par instance {pc(g)}, significatif {w}/9")

# Effet propre : contrôlée vs statique, même cloud
out.append("\n## Contrôlée vs statique (même cloud)\n")
for q, s in (("GQ", "G"), ("LQ", "L"), ("UQ", "U")):
    d = 100 * (np.mean([hv(q, i) for i in I]) / np.mean([hv(s, i) for i in I]) - 1)
    better = [i for i in I if mannwhitneyu(runs(q, i), runs(s, i), alternative="greater").pvalue < 0.05]
    worse = [i for i in I if mannwhitneyu(runs(q, i), runs(s, i), alternative="less").pvalue < 0.05]
    p = wilcoxon([hv(q, i) for i in I], [hv(s, i) for i in I]).pvalue
    out.append(f"- {q} vs {s} : HV moyen {pc(d)} ; meilleure {len(better)} {better} ; moins bonne {len(worse)} {worse} ; "
               f"Wilcoxon bilatéral p = {p:.2f} ; temps {T(q):.2f} s vs {T(s):.2f} s ({pc(100 * (T(q) / T(s) - 1))})")

# Nouvelle politique vs politique publiée
out.append("\n## Politique réentraînée vs publiée (même cloud)\n")
for q in ("GQ", "LQ", "UQ"):
    if q + "p" not in D:
        continue
    p_ = q + "p"
    d = 100 * (np.mean([hv(q, i) for i in I]) / np.mean([hv(p_, i) for i in I]) - 1)
    worse = [i for i in I if mannwhitneyu(runs(q, i), runs(p_, i), alternative="less").pvalue < 0.05]
    better = [i for i in I if mannwhitneyu(runs(q, i), runs(p_, i), alternative="greater").pvalue < 0.05]
    out.append(f"- {q} : HV {pc(d)} ; moins bonne {worse} ; meilleure {better} ; temps {T(q):.2f} s vs {T(p_):.2f} s "
               f"({pc(100 * (T(q) / T(p_) - 1))})")

# Temps
out.append("\n## Temps par exécution (s)\n")
out.append("| Instance | W | GQ | LQ | UQ | budget (0,8 x W) | UQ / budget |")
out.append("|---|---|---|---|---|---|---|")
for i in I:
    b = 0.8 * D["W"][i]["s_per_run"]
    out.append(f"| {i} | " + " | ".join(f"{D[k][i]['s_per_run']:.2f}" for k in ("W", "GQ", "LQ", "UQ"))
               + f" | {b:.2f} | {D['UQ'][i]['s_per_run'] / b:.2f} |")
out.append("| Total | " + " | ".join(f"{T(k):.2f}" for k in ("W", "GQ", "LQ", "UQ")) + f" | {0.8 * T('W'):.2f} | |\n")
for k in ("GQ", "LQ", "UQ"):
    faster = [i for i in I if D[k][i]["s_per_run"] < D["W"][i]["s_per_run"]]
    r = [D[k][i]["s_per_run"] / (0.8 * D["W"][i]["s_per_run"]) for i in I]
    out.append(f"- {k} : plus rapide que W-CMOLS sur {len(faster)}/9 ; 450 exécutions = {T(k) * 50 / 60:.1f} min ; "
               f"rapport mesuré / budget de {min(r):.2f} à {max(r):.2f}, {sum(abs(x - 1) <= 0.15 for x in r)}/9 à 15 % près, "
               f"{sum(x <= 1 for x in r)}/9 sous le budget")
out.append(f"- W : 450 exécutions = {T('W') * 50 / 60:.1f} min ; U : {T('U') * 50 / 60:.1f} min")
out.append("\n## Configurations choisies\n")
for k in ("GQ", "LQ", "UQ"):
    out.append(f"- {k} : " + " ; ".join(f"{i} ({D[k][i]['params']['alpha']}, {D[k][i]['params']['NBL']}, "
                                         f"{D[k][i]['params']['L']}, {D[k][i]['params']['kappa']})" for i in I))

if "WQ" in D and "source" in D["WQ"]["250.2"]:
    out.append("\n## Contrôle sans cloud (politique GQ finale sur W-CMOLS)\n")
    g = np.mean([100 * (hv("WQ", i) / hv("W", i) - 1) for i in I])
    better = [i for i in I if mannwhitneyu(runs("WQ", i), runs("W", i), alternative="greater").pvalue < 0.05]
    worse = [i for i in I if mannwhitneyu(runs("WQ", i), runs("W", i), alternative="less").pvalue < 0.05]
    p = wilcoxon([hv("WQ", i) for i in I], [hv("W", i) for i in I]).pvalue
    out.append(f"- HV moyen {fr(np.mean([hv('WQ', i) for i in I]))} vs {fr(np.mean([hv('W', i) for i in I]))} ; gain moyen {pc(g)} ; "
               f"meilleure {len(better)} ; moins bonne {len(worse)} ; Wilcoxon p = {p:.2f} ; temps {T('WQ'):.2f} s vs {T('W'):.2f} s "
               f"({pc(100 * (T('WQ') / T('W') - 1))})")

(RF / "C2_CH5_FINAL.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
