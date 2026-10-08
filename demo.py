"""Démonstration rapide : W-CMOLS, amorçage par le cloud union, puis configuration par l'agent DQN.

usage : python demo.py [instance] [répétitions]      (par défaut : 250.2, 3 répétitions, moins d'une minute)

Les exécutions emploient les mêmes germes que les campagnes : les hypervolumes obtenus doivent être
identiques aux premières répétitions des résultats publiés dans results/.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "configuration")]
sys.stdout.reconfigure(encoding="utf-8")
import common as C                      # noqa: E402
import evaluate_agent as A              # noqa: E402

INST = sys.argv[1] if len(sys.argv) > 1 else "250.2"
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
WORK = ROOT / "work" / "demo"


def published(name):
    return json.loads((ROOT / "results" / name).read_text())


def report(label, hv, ref, dt):
    same = np.allclose(hv, ref[:len(hv)])
    print(f"  {label:<34} HV moyen {np.mean(hv):.4f}   {dt:5.2f} s/exécution   "
          f"{'identique au résultat publié' if same else 'DIFFÉRENT du résultat publié'}")
    return np.mean(hv)


print(f"Instance {INST}, {RUNS} répétitions\n")

print("1. Solveur de référence, archive initiale vide")
hv, dt = C.solve(None, INST, RUNS, WORK / "wcmols", reinject=False)
w = report("W-CMOLS", hv, published(f"wcmols_{INST}.json")["hv_runs"], dt)

print("\n2. Amorçage par le cloud union (glouton + modèle de langage), réglages par défaut")
dec = json.loads((ROOT / "seeding" / "llm" / "decisions" / f"{INST}.json").read_text())
k, d = next((k, d) for k, d in dec.items() if d["remove"] or d["add"])
print(f"  exemple de décision du LLM (graine {k}) : retirer {d['remove']}, ajouter {d['add']}")
hv, dt = C.solve(ROOT / "clouds" / f"union_{INST}.txt", INST, RUNS, WORK / "union", reinject=False)
u = report("SW-CMOLS-U", hv, published(f"seeded_union_{INST}.json")["hv_runs"], dt)

print("\n3. Configuration choisie par l'agent DQN, budget 0,8 x temps de W-CMOLS")
agent = A.AutoregressiveDQNAgentV3.load(ROOT / "models" / "policy_union" / "checkpoint.pt")
agent.epsilon = 0.0
cm = A.calibrated_model("seeded_union", "calibration_agent_union")
ref = json.loads((ROOT / "data" / "reference" / f"{INST}.json").read_text())
t_w = published(f"wcmols_{INST}.json")["s_per_run"]
state = A.encode_state(ref["n_objectives"], A.N_ITEMS[INST.split(".")[0]], ref["baseline_hv_mean"], ref["baseline_run_time"])
guard = None if A.predict_time(cm, INST, **A.GUARD) > 0.8 * t_w else A.GUARD
mask = A.BudgetMask(cm, INST, agent.grids, 0.8 * t_w, guard=guard)
p = agent.decode(agent.policy(state, mask_fn=mask.mask_fn))
print(f"  configuration choisie : alpha={p['alpha']}, NBL={p['NBL']}, L={p['L']}, kappa={p['kappa']}"
      f"   (temps prédit {A.predict_time(cm, INST, **p):.2f} s, budget {0.8 * t_w:.2f} s)")
hv, dt = C.solve(ROOT / "clouds" / f"union_{INST}.txt", INST, RUNS, WORK / "agent", reinject=False, params=p)
uq = report("SW-CMOLS-UQ", hv, published("agent_union_budget0.8.json")[INST]["hv_runs"], dt)

print(f"\nGain d'hypervolume sur W-CMOLS : SW-CMOLS-U {100 * (u / w - 1):+.1f} %, SW-CMOLS-UQ {100 * (uq / w - 1):+.1f} %"
      f"  ({RUNS} répétitions ; 50 dans results/)")
