"""Cardinalité (50 rép.), temps et diversité (espacement/étendue, 20 rép.) de nos sept systèmes.
Même code que ch6_v8.py (fronts, spacing_extent) ; sortie results/OURS_METRICS.json."""
import json, sys
import numpy as np
sys.argv = [sys.argv[0]]
import ch6_v8 as C
out = {"card50": {}, "div20": {}, "time": {}}
TAG = {"W": "baseline_nbl100", "G": "G_nbl100", "L": "L_nbl100", "U": "U2_nbl100", "GQ": "GQ_dqn", "LQ": "LQ_dqn", "UQ": "UQ_dqn"}
for inst in C.I:
    lo, hi = C.D.bounds(inst)
    for k in C.OURS:
        f50 = C.fronts(k, inst, 50)
        out["card50"].setdefault(k, {})[inst] = float(np.mean([len(f) for f in f50]))
        se = [C.spacing_extent(f, lo, hi) for f in f50[:20]]
        out["div20"].setdefault(k, {})[inst] = [float(np.nanmean([s for s, _ in se])), float(np.nanmean([e for _, e in se]))]
        out["time"].setdefault(k, {})[inst] = json.loads((C.ROOT / "results" / f"{TAG[k]}_{inst}.json").read_text())["s_per_run"]
    print(inst, flush=True)
(C.ROOT / "results" / "OURS_METRICS.json").write_text(json.dumps(out, indent=1))
for k in C.OURS:
    print(k, "card", [round(out["card50"][k][i]) for i in C.I], "esp", round(np.mean([out["div20"][k][i][0] for i in C.I]), 4),
          "ete", round(np.mean([out["div20"][k][i][1] for i in C.I]), 3))
