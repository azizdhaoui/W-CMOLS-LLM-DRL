"""Écart à l'oracle empirique, refait sur l'agent final (UQ, cloud union u2).

Criblage de toutes les configurations autorisées, dans les conditions de l'évaluation finale (evaluate_agent.py, UQ, r = 0,8, NOGUARD) :
  - modèle de coût calibré et budget 0,8 x temps mesuré de W-CMOLS, comme evaluate_agent ;
  - solveur sans réinjection, cloud u2, mêmes germes ;
  - étape 1 : chaque configuration faisable (temps prédit <= budget), N1 = 3 répétitions ;
  - étape 2 : les K_TOP = 12 meilleures + le choix de l'agent final, N2 = 10 répétitions.
Écart = (HV oracle - HV agent) / HV oracle, à N2 répétitions pour les deux.

usage : python oracle_gap.py [inst ...]      (reprend là où il s'est arrêté)
Sortie : results/validation/oracle_<inst>.json (+ oracle_<inst>_stage1.json, partiel)
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1]), str(_Path(__file__).resolve().parents[1] / 'configuration')]
import common as C
from evaluate_agent import RF, V3, calibrated_model  # noqa: F401  (même calibration que l'évaluation finale)
from dqn_agent import AutoregressiveDQNAgentV3
from dqn_agent.budget_mask import predict_time

N1, K_TOP, N2 = 3, 12, 10
R = 0.8
OUT = RF / "validation"
WORK = C.HERE / "work" / "oracle_gap"
AGENT_RUN = RF / "agent_union_budget0.8.json"     # choix de l'agent final, évaluation canonique


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def key(p):
    return f"a{p['alpha']}_N{p['NBL']}_L{p['L']}_k{p['kappa']}"


def main():
    insts = sys.argv[1:] or C.INSTANCES
    OUT.mkdir(parents=True, exist_ok=True)
    agent = AutoregressiveDQNAgentV3.load(V3 / "models" / "policy_union" / "checkpoint.pt")
    grids = agent.grids
    cm = calibrated_model("seeded_union", "calibration_agent_union")
    final = json.loads(AGENT_RUN.read_text())
    for i in insts:
        out = OUT / f"oracle_{i}.json"
        if out.exists():
            log(f"{i}: déjà fait"); continue
        tw = json.loads((RF / f"wcmols_{i}.json").read_text())
        budget = R * tw["s_per_run"]
        combos = [dict(zip(("alpha", "NBL", "L", "kappa"), c)) for c in
                  itertools.product(grids["alpha"], grids["NBL"], grids["L"], grids["kappa"])]
        feas = [p for p in combos if predict_time(cm, i, **p) <= budget]
        agent_p = final[i]["params"]
        assert key(agent_p) in {key(p) for p in feas}, (i, agent_p)
        cloud = C.HERE / "clouds" / f"union_{i}.txt"
        log(f"{i}: {len(feas)}/432 faisables, budget {budget:.2f} s, "
            f"étape 1 prédite {sum(predict_time(cm, i, **p) for p in feas) * N1 / 60:.1f} min")

        # étape 1 (reprise possible combinaison par combinaison)
        part = OUT / f"oracle_{i}_stage1.json"
        s1 = json.loads(part.read_text()) if part.exists() else {}
        t0 = time.time()
        for j, p in enumerate(feas, 1):
            if key(p) in s1:
                continue
            hv, dt = C.solve(cloud, i, N1, WORK, reinject=False, params=p)
            s1[key(p)] = {"params": p, "hv_mean": float(np.mean(hv)), "s_per_run": dt}
            part.write_text(json.dumps(s1, indent=1))
            if j % 10 == 0 or j == len(feas):
                log(f"  {i} étape 1 : {j}/{len(feas)} ({(time.time() - t0) / 60:.1f} min)")
        ranked = sorted(s1.values(), key=lambda d: -d["hv_mean"])
        rank_agent = 1 + [key(d["params"]) for d in ranked].index(key(agent_p))

        # étape 2 : 12 meilleures + choix de l'agent, 10 répétitions
        cands = [d["params"] for d in ranked[:K_TOP]]
        if key(agent_p) not in {key(p) for p in cands}:
            cands.append(agent_p)
        hv_w = np.array(tw["hv_runs"])
        fine = []
        for p in cands:
            hv, dt = C.solve(cloud, i, N2, WORK, reinject=False, params=p)
            hv = np.array(hv)
            fine.append({"params": p, "hv_mean": float(hv.mean()), "hv_std": float(hv.std()),
                         "hv_runs": hv.tolist(), "s_per_run": dt,
                         "p_vs_W": float(mannwhitneyu(hv, hv_w, alternative="greater")[1])})
        fine.sort(key=lambda d: -d["hv_mean"])
        oracle = fine[0]
        ag = next(d for d in fine if key(d["params"]) == key(agent_p))
        gap = (oracle["hv_mean"] - ag["hv_mean"]) / oracle["hv_mean"] * 100
        res = {"instance": i, "budget_s": budget, "n_feasible": len(feas), "n_total": 432,
               "N1": N1, "K_TOP": K_TOP, "N2": N2, "oracle": oracle, "agent": ag,
               "agent_is_oracle": key(oracle["params"]) == key(agent_p),
               "agent_rank_stage1": rank_agent, "gap_pct": float(gap),
               "agent_rank_stage2": 1 + [key(d["params"]) for d in fine].index(key(agent_p)),
               "stage2": fine}
        out.write_text(json.dumps(res, indent=1))
        log(f"  {i}: oracle {oracle['params']} HV {oracle['hv_mean']:.4f} | agent {agent_p} "
            f"HV {ag['hv_mean']:.4f} | écart {gap:+.2f} % | rang étape 1 : {rank_agent}/{len(feas)}")


if __name__ == "__main__":
    main()
