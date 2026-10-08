"""Recalibre le modèle de coût v3.1 sur la campagne finale (clouds finaux, solveur sans réinjection).

Principe : les élasticités (NBL, L, alpha, kappa) sont conservées — ajustées sur 900 mesures,
R² = 0,993. Seul l'intercept de chaque instance est corrigé, par cloud, de la moyenne
(en log) du rapport temps mesuré / temps prédit sur les deux campagnes disponibles pour
ce cloud (variante statique aux paramètres de référence + variante contrôlée).
Justification : ce rapport est stable par (instance, cloud) — par ex. 1,59 et 1,59
sur 250.3 avec le cloud U — donc un décalage multiplicatif par instance suffit.

Sortie : models/cost_model_final_<cloud>.json  (cloud = U, G, L, W)
"""
import json
import math
import sys

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])]
import common as C

V3 = C.ROOT
sys.path.insert(0, str(V3))
from rl_agent_v3 import load_cost_model  # noqa: E402
from rl_agent_v3.budget_mask import predict_time  # noqa: E402

RF = C.HERE / "results"
GUARD = {"alpha": 10, "NBL": 100, "L": 5, "kappa": 0.05}
CLOUDS = {"U": ("U2_nbl100", "UQ_dqn_pub"), "G": ("G_nbl100", "GQ_dqn_pub"),
          "L": ("L_nbl100", "LQ_dqn_pub"), "W": ("baseline_nbl100", "WQ_dqn_pub")}

base = load_cost_model(V3 / "models" / "cost_model_v31.json")
for cloud, tags in CLOUDS.items():
    cm = json.loads(json.dumps(base))
    report = {}
    for i in C.INSTANCES:
        logs = []
        for tag in tags:
            d = json.loads((RF / f"{tag}_{i}.json").read_text())
            p = d.get("params") or GUARD
            logs.append(math.log(d["s_per_run"] / predict_time(base, i, **p)))
        cm["intercepts"][i] += float(np.mean(logs))
        report[i] = [round(math.exp(x), 3) for x in logs]
    cm["calibration"] = {"source": "campagne finale 2026-09-24, 50 répétitions",
                         "campaigns": list(tags), "measured_over_predicted": report}
    out = V3 / "models" / f"cost_model_final_{cloud}.json"
    out.write_text(json.dumps(cm, indent=1))
    print(cloud, out.name, {i: v for i, v in report.items()})
