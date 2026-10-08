"""Sonde des bornes basses des grilles alpha et NBL (contribution 2).

Question : faut-il étendre les grilles vers le bas (alpha = 5, NBL = 20) pour gagner du temps ?
Constat qui la motive : sur 63 décisions, alpha = 10 (borne basse) est choisi 40 fois et
NBL = 30 (borne basse) 15 fois ; alpha = 30, 40 et NBL = 160 ne sont jamais choisis.

Protocole : pour chaque instance, la configuration retenue par la variante A
(results/c2_screen_UQ_r0.8.json, cloud U, 10 répétitions) sert de référence ;
on la réexécute avec alpha = 5, avec NBL = 20, puis les deux, mêmes germes (10 répétitions).
Sortie : results/c2_grid_probe.json
"""
import json
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1])]
import common as C

RF = C.HERE / "results"
RUNS = 10
ref = json.loads((RF / "c2_screen_UQ_r0.8.json").read_text())
out_p = RF / "c2_grid_probe.json"
out = json.loads(out_p.read_text()) if out_p.exists() else {}
VARIANTS = {"alpha5": {"alpha": 5}, "nbl20": {"NBL": 20}, "alpha5_nbl20": {"alpha": 5, "NBL": 20}}

for i in C.INSTANCES:
    base = ref[i]["params"]
    for name, change in VARIANTS.items():
        key = f"{i}|{name}"
        if key in out:
            continue
        p = {**base, **change}
        hv, dt = C.solve(C.HERE / "clouds" / f"u2_{i}.txt", i, RUNS, C.HERE / "work" / "c2_probe",
                         reinject=False, params=p)
        out[key] = {"params": p, "ref_params": base, "s_per_run": dt, "hv_mean": float(np.mean(hv)),
                    "ref_s_per_run": ref[i]["s_per_run"], "ref_hv_mean": ref[i]["hv_mean"], "hv_runs": hv}
        out_p.write_text(json.dumps(out, indent=1))
        print(f"{i} {name}: {p} t {dt:.2f}s (ref {ref[i]['s_per_run']:.2f}) HV {np.mean(hv):.4f} "
              f"(ref {ref[i]['hv_mean']:.4f}, {100*(np.mean(hv)/ref[i]['hv_mean']-1):+.2f} %)", flush=True)
