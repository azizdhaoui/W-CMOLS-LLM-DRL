"""Résumé des validations refaites sur l'agent final : oracle, instance exclue, ablation.

Lit results/validation/oracle_<inst>.json, agent_union_leave_one_out_budget0.8.json et
agent_greedy_ablation_<v>_budget0.8.json ; écrit results/validation/validation_summary.md.
Référence W-CMOLS : wcmols_<inst>.json (50 répétitions, mêmes germes).
"""
import json

import numpy as np
from scipy.stats import mannwhitneyu

import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1]), str(_Path(__file__).resolve().parents[1] / 'configuration')]
import common as C

RF = C.HERE / "results"
OUT = RF / "validation"


def fr(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def vs_w(hv, i):
    w = json.loads((RF / f"wcmols_{i}.json").read_text())
    hv, hw = np.array(hv), np.array(w["hv_runs"])
    u, p = mannwhitneyu(hv, hw, alternative="greater")
    gain = (hv.mean() - hw.mean()) / hw.mean() * 100
    return gain, float(p), float(u) / (len(hv) * len(hw)), w["s_per_run"]


def conf(p):
    return f"α={p['alpha']}, NBL={p['NBL']}, L={p['L']}, κ={fr(p['kappa'], 2).rstrip('0').rstrip(',')}"


def oracle_part(md):
    rows = [json.loads((OUT / f"oracle_{i}.json").read_text())
            for i in C.INSTANCES if (OUT / f"oracle_{i}.json").exists()]
    md += ["## 1. Écart à l'oracle empirique (agent UQ final, cloud union)", "",
           f"Instances terminées : {len(rows)}/9. Étape 1 : chaque configuration faisable, 3 répétitions ; "
           "étape 2 : les 12 meilleures + le choix de l'agent, 10 répétitions.", "",
           "| Instance | Faisables | HV oracle | HV de l'agent | Écart | Rang de l'agent (étape 1) | Config. oracle | Config. agent |",
           "|:---|---:|---:|---:|---:|---:|:---|:---|"]
    for r in rows:
        md.append(f"| {r['instance']} | {r['n_feasible']} / 432 | {fr(r['oracle']['hv_mean'], 4)} | "
                  f"{fr(r['agent']['hv_mean'], 4)} | {'+' if r['gap_pct'] >= 0 else ''}{fr(r['gap_pct'])} % | "
                  f"{r['agent_rank_stage1']} / {r['n_feasible']} | {conf(r['oracle']['params'])} | {conf(r['agent']['params'])} |")
    if rows:
        g = np.array([r["gap_pct"] for r in rows])
        md += ["", f"- Écart médian : **{fr(np.median(g))} %**, moyen : {fr(g.mean())} %, maximal : {fr(g.max())} % "
               f"({rows[int(g.argmax())]['instance']}).",
               f"- Écart < 1 % sur {int((g < 1).sum())} instance(s) ; l'agent retrouve exactement l'oracle sur "
               f"{sum(r['agent_is_oracle'] for r in rows)} instance(s).", ""]


def loo_part(md):
    f = RF / "agent_union_leave_one_out_budget0.8.json"
    res = json.loads(f.read_text()) if f.exists() else {}
    md += ["## 2. Instance jamais vue (validation croisée, 9 agents entraînés depuis zéro sur 8 instances)", "",
           f"Instances terminées : {len(res)}/9. Évaluation : 50 répétitions, budget 0,8 × W-CMOLS, "
           "Mann-Whitney U unilatéral contre W-CMOLS.", "",
           "| Instance exclue | Gain d'HV | p | A12 | Configuration choisie | Temps / budget | Verdict |",
           "|:---|---:|---:|---:|:---|---:|:---|"]
    won, t_ctrl, t_w, ratios = 0, 0.0, 0.0, []
    for i in C.INSTANCES:
        if i not in res:
            continue
        r = res[i]
        gain, p, a12, tw = vs_w(r["hv_runs"], i)
        ok = p < 0.05 and gain > 0
        won += ok
        t_ctrl += 50 * r["s_per_run"]; t_w += 50 * tw
        ratios.append((r["s_per_run"] / r["budget"], i))
        md.append(f"| {i} | {'+' if gain >= 0 else ''}{fr(gain)} % | {p:.1e} | {fr(a12, 3)} | {conf(r['params'])} | "
                  f"{fr(r['s_per_run'] / r['budget'])} | {'gagné' if ok else 'non conclu'} |")
    if res:
        mx = max(ratios)
        md += ["", f"- Bilan : **{won}/{len(res)}** instances exclues gagnées (p < 0,05).",
               f"- Temps cumulé des 50 répétitions : {fr(t_ctrl, 0)} s contre {fr(t_w, 0)} s pour W-CMOLS ; "
               f"dépassement maximal du budget : {fr(mx[0])} × ({mx[1]}).", ""]


def abl_part(md):
    md += ["## 3. Ablation Dueling / rejeu priorisé (cloud glouton, entraînement depuis zéro, 225 épisodes)", "",
           "| Variante | Gain moyen d'HV vs W-CMOLS | Instances gagnées | HV moyen | Temps total / exécution |",
           "|:---|---:|:---|---:|---:|"]
    names = {"full": "Agent complet (Dueling + rejeu priorisé)", "noduel": "Sans Dueling",
             "noper": "Sans rejeu priorisé"}
    per = {}
    for v, name in names.items():
        f = RF / f"agent_greedy_ablation_{v}_budget0.8.json"
        if not f.exists():
            md.append(f"| {name} | (pas encore) | | | |"); continue
        res = json.loads(f.read_text())
        st = [vs_w(res[i]["hv_runs"], i) for i in C.INSTANCES if i in res]
        gains = [s[0] for s in st]
        won = sum(s[1] < 0.05 and s[0] > 0 for s in st)
        per[v] = {i: res[i]["hv_mean"] for i in res}
        md.append(f"| {name} | +{fr(np.mean(gains))} % | {won} / {len(st)} | "
                  f"{fr(np.mean([res[i]['hv_mean'] for i in res]), 4)} | "
                  f"{fr(sum(res[i]['s_per_run'] for i in res), 1)} s |")
    ref = json.loads((RF / "agent_greedy_budget0.8.json").read_text())
    st = [vs_w(ref[i]["hv_runs"], i) for i in C.INSTANCES]
    md += [f"| *Pour mémoire : GQ final (amorcé, rapport)* | +{fr(np.mean([s[0] for s in st]))} % | "
           f"{sum(s[1] < 0.05 and s[0] > 0 for s in st)} / 9 | "
           f"{fr(np.mean([ref[i]['hv_mean'] for i in C.INSTANCES]), 4)} | "
           f"{fr(sum(ref[i]['s_per_run'] for i in C.INSTANCES), 1)} s |", ""]
    if "full" in per:
        for v in ("noduel", "noper"):
            if v in per:
                d = [(per["full"][i] - per[v][i]) / per[v][i] * 100 for i in C.INSTANCES if i in per[v]]
                md.append(f"- Complet contre {names[v].lower()} : meilleur sur {sum(x > 0 for x in d)}/{len(d)} "
                          f"instances, écart moyen {'+' if np.mean(d) >= 0 else ''}{fr(np.mean(d))} %.")
    md += [""]


def card(name, i):
    f = RF / "fronts" / f"{name}_{i}_sizes.txt"
    return float(np.mean([int(x) for x in f.read_text().split()])) if f.exists() else float("nan")


def smac_part(md, suf="", title="## 4. SMAC3 dans les conditions finales, objectif HV (cloud glouton, 25 configurations faisables x 10 répétitions)",
              old=True):
    f = OUT / f"smac3{suf}.json"
    md += [title, ""]
    if not f.exists():
        md += ["(pas encore)", ""]; return
    res = json.loads(f.read_text())["instances"]
    gq = json.loads((RF / "agent_greedy_budget0.8.json").read_text())
    sd = RF / "agent_greedy_same_day_budget0.8.json"                     # temps de GQ mesurés le même jour
    t_gq = json.loads(sd.read_text()) if sd.exists() else gq
    md += ["| Instance | Config. SMAC3 | HV SMAC3 | HV GQ | Gain SMAC3 vs W | Gain GQ vs W | Solutions SMAC3 / GQ | Temps/exéc. SMAC3 / GQ | Recherche |",
           "|:---|:---|---:|---:|---:|---:|---:|---:|---:|"]
    gs, gg, cs, cg, t_all, n_all = [], [], 0.0, 0.0, 0.0, 0
    for i in C.INSTANCES:
        if i not in res:
            continue
        r = res[i]
        g_gq = vs_w(gq[i]["hv_runs"], i)[0]
        gs.append(r["gain_pct"]); gg.append(g_gq)
        c_s, c_g = card(f"smac3{suf}", i), card("agent_greedy_budget0.8", i)
        cs += c_s; cg += c_g
        t_all += r["tuning_wall_s"]; n_all += r["n_solver_runs_tuning"]
        md.append(f"| {i} | {conf(r['params'])} | {fr(r['hv_mean'], 4)} | {fr(gq[i]['hv_mean'], 4)} | "
                  f"+{fr(r['gain_pct'])} % | +{fr(g_gq)} % | {c_s:.0f} / {c_g:.0f} | "
                  f"{fr(r['s_per_run'])} / {fr(t_gq[i]['s_per_run'])} s | {fr(r['tuning_wall_s'] / 60, 1)} min |")
    better = sum(res[i]["hv_mean"] > gq[i]["hv_mean"] for i in res)
    md += ["", f"- SMAC3 : +{fr(np.mean(gs))} % en moyenne, gagne {sum(res[i]['win'] for i in res)}/{len(res)} ; "
           f"GQ sur les mêmes instances : +{fr(np.mean(gg))} %. SMAC3 meilleur que GQ en HV sur {better}/{len(res)}.",
           f"- Solutions par exécution (somme des instances) : SMAC3 {cs:.0f}, GQ {cg:.0f}.",
           f"- Coût de recherche de SMAC3 : {n_all} exécutions, {fr(t_all / 60, 0)} min au total, "
           f"max {fr(max(res[i]['tuning_wall_s'] for i in res) / 60, 1)} min sur une instance.",
           f"- Temps par exécution (somme) : SMAC3 {fr(sum(res[i]['s_per_run'] for i in res), 1)} s, GQ "
           f"{fr(sum(t_gq[i]['s_per_run'] for i in res), 1)} s ({'même jour' if sd.exists() else 'autre jour'})."]
    md.append("")


def sameday_part(md):
    f = RF / "agent_greedy_same_day_budget0.8.json"
    md += ["## 6. Temps de GQ remesuré le même jour que SMAC3 (mêmes germes, 50 répétitions)", ""]
    if not f.exists():
        md += ["(pas encore)", ""]; return
    new = json.loads(f.read_text()); ref = json.loads((RF / "agent_greedy_budget0.8.json").read_text())
    same = all(abs(new[i]["hv_mean"] - ref[i]["hv_mean"]) < 1e-9 for i in new)
    md += [f"- HV identique à la campagne canonique : {'oui' if same else 'NON'} (déterminisme).",
           f"- Temps total par exécution : {fr(sum(new[i]['s_per_run'] for i in new), 1)} s le même jour, "
           f"contre {fr(sum(ref[i]['s_per_run'] for i in ref), 1)} s dans la campagne canonique.", ""]


def retime_part(md):
    f = OUT / "same_day_timing.json"
    md += ["## 7. Temps des sept systèmes remesurés le même jour, en alternance (50 répétitions)", ""]
    if not f.exists():
        md += ["(pas encore)", ""]; return
    r = json.loads(f.read_text())
    sy = ["W", "G", "GQ", "L", "LQ", "U", "UQ"]
    done = [i for i in C.INSTANCES if all(f"{x}_{i}" in r for x in sy)]
    md.append(f"Instances terminées : {len(done)}/9 ; HV identique à la campagne canonique : "
              f"{sum(v['hv_identical'] for v in r.values())}/{len(r)} mesures.")
    if not done:
        md.append(""); return
    T = lambda x, key="s_per_run": sum(r[f"{x}_{i}"][key] for i in done)
    md += ["", "| Système | Temps / exéc. (même jour) | Temps / exéc. (rapport) |", "|:---|---:|---:|"]
    md += [f"| {x} | {fr(T(x), 1)} s | {fr(T(x, 's_per_run_canon'), 1)} s |" for x in sy]
    red = lambda a, b, key: (1 - T(a, key) / T(b, key)) * 100
    md.append("")
    for q, st in (("GQ", "G"), ("LQ", "L"), ("UQ", "U"), ("UQ", "W")):
        md.append(f"- {q} contre {st} : −{fr(red(q, st, 's_per_run'), 0)} % le même jour "
                  f"(rapport : −{fr(red(q, st, 's_per_run_canon'), 0)} %).")
    for q in ("GQ", "LQ", "UQ"):
        n = sum(r[f"{q}_{i}"]["s_per_run"] < r[f"W_{i}"]["s_per_run"] for i in done)
        md.append(f"- {q} plus rapide que W-CMOLS, même jour : {n}/{len(done)} instances.")
    ratios = [r[f"UQ_{i}"]["s_per_run"] / (0.8 * r[f"W_{i}"]["s_per_run"]) for i in done]
    md.append(f"- UQ / budget (0,8 × W du même jour) : de {fr(min(ratios))} à {fr(max(ratios))} ; "
              f"sous le budget sur {sum(x <= 1 for x in ratios)}/{len(done)}.")
    lf = RF / "agent_union_leave_one_out_budget0.8.json"
    if lf.exists() and len(done) == 9:
        loo = json.loads(lf.read_text())
        md.append(f"- Instance jamais vue : {fr(50 * sum(loo[i]['s_per_run'] for i in done), 0)} s contre "
                  f"{fr(50 * T('W'), 0)} s pour W-CMOLS remesuré (au lieu de 1 978 s mesurés le 23 sept.).")
    md.append("")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# Validations refaites sur l'agent final", "",
          "Généré par `summarize_validations.py`.", ""]
    oracle_part(md); loo_part(md); abl_part(md); smac_part(md)
    smac_part(md, "_reward", "## 5. SMAC3 avec la note de l'agent (HV + cardinalité − pénalité de temps), mêmes conditions", old=False)
    sameday_part(md)
    retime_part(md)
    (OUT / "validation_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
