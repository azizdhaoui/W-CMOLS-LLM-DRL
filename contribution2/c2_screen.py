"""Criblage rapide de la contribution 2 sans réentraînement.

Même politique entraînée (checkpoint), mais :
  1. modèle de coût recalibré sur les temps mesurés de la campagne finale
     (intercept de chaque instance corrigé pour le cloud utilisé) ;
  2. budget par instance = r x temps mesuré de W-CMOLS (plus de redistribution).
Chaque configuration décodée est exécutée RUNS fois (mêmes germes que les
premières répétitions officielles) sur le cloud du système.

usage : python c2_screen.py <system: UQ|GQ> <r> [<r> ...]
Sortie : results/c2_screen_<system>_r<r>.json
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])]
import common as C

V3 = C.ROOT
sys.path.insert(0, str(V3))
from rl_agent_v3 import AutoregressiveDQNAgentV3, BudgetMask, encode_state, load_cost_model  # noqa: E402
from rl_agent_v3.budget_mask import predict_time  # noqa: E402

RF = C.HERE / "results"
RUNS = int(os.environ.get("RUNS", "10"))
GUARD = {"alpha": 10, "NBL": 100, "L": 5, "kappa": 0.05}
NOGUARD = os.environ.get("NOGUARD") == "1"   # garde-fou seulement s'il tient dans le budget
ONLY = os.environ.get("ONLY")   # validation croisée : évaluer une seule instance
SYS = {"UQ": ("union_b1_card", "U2_nbl100", "UQ_dqn_pub", C.HERE / "clouds" / "u2_{i}.txt"),
       "GQ": ("greedy_b1_card", "G_nbl100", "GQ_dqn_pub", C.ROOT / "clouds" / "greedy_cloud_{i}.txt"),
       "LQ": ("union_b1_card", "L_nbl100", "LQ_dqn_pub", C.HERE / "clouds" / "v8b_llm_{i}.txt"),
       # contrôle sans cloud : politique de GQ appliquée à W-CMOLS (archive vide)
       "WQ": ("greedy_b1_card", "baseline_nbl100", "WQ_dqn_pub", None)}
N_ITEMS = {"250": 250, "500": 500, "750": 750}


def calibrated_model(static_tag, q_tag):
    """Intercept de chaque instance corrigé par le rapport mesuré / prédit (moyenne log
    sur la variante statique et la variante contrôlée, même cloud)."""
    cm = load_cost_model(V3 / "models" / "cost_model_v31.json")
    cm = json.loads(json.dumps(cm))
    for i in C.INSTANCES:
        logs = []
        for tag in (static_tag, q_tag):
            d = json.loads((RF / f"{tag}_{i}.json").read_text())
            p = d.get("params") or GUARD
            logs.append(math.log(d["s_per_run"] / predict_time(cm, i, **p)))
        cm["intercepts"][i] += float(np.mean(logs))
    return cm


def main():
    system = sys.argv[1]; ratios = [float(r) for r in sys.argv[2:]]
    ck_tag, static_tag, q_tag, cloud_pat = SYS[system]
    ck_tag = os.environ.get("CKPT_TAG", ck_tag)          # politique réentraînée, si fournie
    extra = os.environ.get("TAG", "")                    # suffixe des fichiers de sortie
    agent = AutoregressiveDQNAgentV3.load(V3 / "models" / f"ckpt_v31_{ck_tag}" / "checkpoint.pt")
    agent.epsilon = 0.0
    cm = calibrated_model(static_tag, q_tag)
    for r in ratios:
        variant = f"r{r:g}{'_ng' if NOGUARD else ''}{extra}"
        out = RF / (f"c2_screen_{system}_{variant}.json" if RUNS == 10 else f"c2_{system}_{variant}_n{RUNS}.json")
        res = json.loads(out.read_text()) if out.exists() else {}
        for i in C.INSTANCES:
            if i in res or (ONLY and i != ONLY):
                continue
            cache = json.loads((C.ROOT / "data/reference" / f"{i}.json").read_text())
            tw = json.loads((RF / f"baseline_nbl100_{i}.json").read_text())["s_per_run"]
            state = encode_state(cache["n_objectives"], N_ITEMS[i.split(".")[0]],
                                 cache["baseline_hv_mean"], cache["baseline_run_time"])
            guard = None if NOGUARD and predict_time(cm, i, **GUARD) > r * tw else GUARD
            mask = BudgetMask(cm, i, agent.grids, r * tw, guard=guard)
            p = agent.decode(agent.policy(state, mask_fn=mask.mask_fn))
            hv, dt = C.solve(Path(str(cloud_pat).format(i=i)) if cloud_pat else None, i, RUNS, C.HERE / "work" / f"c2_{system}",
                             reinject=False, params=p,
                             save_as=(RF / "fronts" / f"c2_{system}_{variant}_{i}") if RUNS != 10 else None)
            res[i] = {"params": p, "pred": predict_time(cm, i, **p), "budget": r * tw,
                      "s_per_run": dt, "hv_runs": hv, "hv_mean": float(np.mean(hv))}
            out.write_text(json.dumps(res, indent=1))
            print(f"r={r:g} {i}: {p} pred {res[i]['pred']:.2f}s budget {r*tw:.2f}s -> {dt:.2f}s HV {np.mean(hv):.4f}", flush=True)


if __name__ == "__main__":
    main()
