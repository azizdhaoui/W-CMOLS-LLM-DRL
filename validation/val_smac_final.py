"""SMAC3 dans les conditions de l'évaluation finale, face à SW-CMOLS-GQ.

SMAC3 par instance (25 essais x 10 répétitions par
instance, cloud glouton, mêmes 432 configurations), dans les conditions de l'évaluation
finale de GQ (c2_screen.py GQ 0.8, NOGUARD) :
  - solveur sans réinjection, mêmes germes, même calcul d'HV ;
  - faisabilité : modèle de coût calibré de c2_screen, budget 0,8 x temps mesuré de W-CMOLS.
Une configuration hors budget reçoit la pénalité (coût 1, aucune exécution) sans compter
parmi les 25 essais : SMAC3 évalue donc 25 configurations faisables par instance. Boucle ask/tell ; plan initial de 6 configurations, comme avec n_trials = 25.
La cible est déterministe (germes fixes), d'où deterministic=True : pas d'essai répété.

Variante OBJ=reward : SMAC3 maximise la note de l'agent (gain d'HV − pénalité de temps
+ 0,2 x gain de cardinalité, mêmes références que l'entraînement) au lieu de l'HV seul.

usage : [OBJ=reward] python val_smac_final.py [inst ...]      (reprend là où il s'est arrêté)
Sortie : results/validation/smac_final.json, fronts dans results/fronts/smac_final_<inst>
"""
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1]), str(_Path(__file__).resolve().parents[1] / 'contribution2')]
import common as C
from c2_screen import RF, V3, calibrated_model
from rl_agent_v3 import AutoregressiveDQNAgentV3
from rl_agent_v3.budget_mask import predict_time

from ConfigSpace import Categorical, ConfigurationSpace  # noqa: E402
from smac import HyperparameterOptimizationFacade, Scenario  # noqa: E402
from smac.runhistory.dataclasses import TrialValue  # noqa: E402

N_TRIALS, N_TUNE, N_EVAL, R = 25, 10, 50, 0.8
MAX_ASK, MAX_WALL = 1500, 90 * 60    # garde-fous : demandes à SMAC3, durée par instance
OBJ = os.environ.get("OBJ", "hv")                  # hv (défaut) ou reward (note de l'agent)
SUF = "" if OBJ == "hv" else f"_{OBJ}"
OUT = RF / "validation" / f"smac_final{SUF}.json"
WORK = C.HERE / "work" / "val_smac"
DIMS = ("alpha", "NBL", "L", "kappa")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def score(i, hv, dt):
    """Valeur maximisée par SMAC3 : HV moyen, ou note de l'agent (environment_v3 + train_v31)."""
    if OBJ == "hv":
        return float(np.mean(hv))
    cache = json.loads((C.ROOT / "data/reference" / f"{i}.json").read_text())
    k = cache["n_objectives"]
    stat = float(np.quantile(hv, 0.25)) if k == 2 and len(hv) >= 4 else float(np.median(hv))
    gain = (stat - cache["baseline_hv_mean"]) / cache["baseline_hv_mean"]
    t_w = json.loads((V3 / "models" / "baseline_times_final.json").read_text())[i]["baseline_time_fresh"]
    pen = 0.10 * max(0.0, dt / t_w - 1.0)                      # λ = 0,10 (lambda3 = 0,10 en version finale)
    sizes = [int(x) for x in (WORK / "pareto_sizes.txt").read_text().split()]
    bc = json.loads((V3 / "models" / "baseline_card.json").read_text())[i]
    return float(gain - pen + 0.2 * (int(np.median(sizes)) - bc) / bc)   # ν = 0,2


def tune(i, grids, cm):
    tw = json.loads((RF / f"baseline_nbl100_{i}.json").read_text())
    budget = R * tw["s_per_run"]
    cloud = C.ROOT / "clouds" / f"greedy_cloud_{i}.txt"
    cs = ConfigurationSpace(seed=42)
    cs.add([Categorical(d, list(grids[d])) for d in DIMS])
    combos = list(itertools.product(*[grids[d] for d in DIMS]))
    feas = {c for c in combos if predict_time(cm, i, **dict(zip(DIMS, c))) <= budget}

    sc = Scenario(cs, deterministic=True, n_trials=10 * MAX_ASK, seed=42,
                  output_directory=C.HERE / "work" / f"val_smac_logs{SUF}" / i)
    init = HyperparameterOptimizationFacade.get_initial_design(sc, n_configs=6)
    smac = HyperparameterOptimizationFacade(sc, lambda config, seed=0: 0.0, initial_design=init,
                                            overwrite=True)
    t0, n_ask, tested = time.time(), 0, {}
    while len(tested) < N_TRIALS and n_ask < MAX_ASK and time.time() - t0 < MAX_WALL:
        info = smac.ask(); n_ask += 1
        p = {d: (float(info.config[d]) if d == "kappa" else int(info.config[d])) for d in DIMS}
        c = tuple(p[d] for d in DIMS)
        if c not in feas:
            smac.tell(info, TrialValue(cost=1.0)); continue        # hors budget : pénalité, aucune exécution
        if c not in tested:
            hv, dt = C.solve(cloud, i, N_TUNE, WORK, reinject=False, params=p)
            tested[c] = score(i, hv, dt)
        smac.tell(info, TrialValue(cost=-tested[c]))                # SMAC minimise
    t_tune = time.time() - t0
    if not tested:
        raise RuntimeError(f"{i}: aucune configuration faisable testée ({n_ask} demandes)")
    best = max(tested, key=tested.get)
    p = dict(zip(DIMS, best))
    log(f"{i}: SMAC3 retient {p} ({len(tested)} configurations testées sur {len(feas)} faisables, "
        f"{n_ask} demandes, {len(tested) * N_TUNE} exécutions, {t_tune / 60:.1f} min)")

    hv, dt = C.solve(cloud, i, N_EVAL, WORK, reinject=False, params=p,
                     save_as=RF / "fronts" / f"smac_final{SUF}_{i}")
    hv, hw = np.array(hv), np.array(tw["hv_runs"])
    u, pv = mannwhitneyu(hv, hw, alternative="greater")
    gain = (hv.mean() - hw.mean()) / hw.mean() * 100
    return {"params": p, "n_feasible": len(feas), "n_configs_tested": len(tested), "n_asks": n_ask,
            "n_solver_runs_tuning": len(tested) * N_TUNE, "tuning_wall_s": t_tune, "budget_s": budget,
            "objective": OBJ, "tuning_score": {"_".join(map(str, k)): v for k, v in tested.items()},
            "hv_runs": hv.tolist(), "hv_mean": float(hv.mean()), "s_per_run": dt,
            "gain_pct": float(gain), "p_vs_W": float(pv), "A12": float(u) / (len(hv) * len(hw)),
            "win": bool(pv < 0.05 and gain > 0)}


def main():
    insts = sys.argv[1:] or C.INSTANCES
    grids = AutoregressiveDQNAgentV3.load(V3 / "models" / "ckpt_v31_greedy_b1_final" / "checkpoint.pt").grids
    cm = calibrated_model("G_nbl100", "GQ_dqn_pub")                 # même faisabilité que l'éval de GQ
    res = json.loads(OUT.read_text())["instances"] if OUT.exists() else {}
    for i in insts:
        if i in res:
            log(f"{i}: déjà fait"); continue
        log(f"=== SMAC3 {i} ({N_TRIALS} essais x {N_TUNE} répétitions) ===")
        res[i] = tune(i, grids, cm)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({
            "tool": "SMAC3 2.4 (HyperparameterOptimizationFacade), par instance",
            "objective": OBJ, "protocol": f"{N_TRIALS} essais x {N_TUNE} répétitions, espace = configurations faisables "
                        f"(budget {R} x W-CMOLS, modèle calibré de c2_screen GQ), cloud glouton, "
                        f"solveur sans réinjection, éval {N_EVAL} répétitions",
            "instances": res}, indent=1))
        r = res[i]
        log(f"  {i}: HV {r['hv_mean']:.4f} gain {r['gain_pct']:+.2f} % p {r['p_vs_W']:.1e} "
            f"{'gagné' if r['win'] else 'non conclu'}")


if __name__ == "__main__":
    main()
