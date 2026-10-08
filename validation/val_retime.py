"""Temps des sept systèmes remesurés le même jour, en alternance (contrôle de l'effet du jour).

Les temps du chapitre 5 viennent de deux nuits différentes : W-CMOLS, G, L, U le 23-24 sept.,
GQ, LQ, UQ le 25 sept. Or le même système peut varier d'environ 25 % d'un jour à l'autre.
Ici, pour chaque instance, les sept systèmes passent l'un après l'autre (ordre tourné d'une
instance à l'autre), 50 répétitions, mêmes germes, même solveur sans réinjection.
L'HV doit être identique à la campagne canonique (déterminisme) : c'est vérifié.

usage : python val_retime.py      (reprend là où il s'est arrêté)
Sortie : results/validation/retime_sameday.json
"""
import json
import time

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1]), str(_Path(__file__).resolve().parents[1] / 'contribution2')]
import common as C

RF = C.HERE / "results"
OUT = RF / "validation" / "retime_sameday.json"
WORK = C.HERE / "work" / "val_retime"
RUNS = 50
GREEDY = C.ROOT / "clouds" / "greedy_cloud_{i}.txt"
CLOUD = {"W": None, "G": GREEDY, "GQ": GREEDY,
         "L": C.HERE / "clouds" / "v8b_llm_{i}.txt", "LQ": C.HERE / "clouds" / "v8b_llm_{i}.txt",
         "U": C.HERE / "clouds" / "u2_{i}.txt", "UQ": C.HERE / "clouds" / "u2_{i}.txt"}
CANON = {"W": "baseline_nbl100", "G": "G_nbl100", "L": "L_nbl100", "U": "U2_nbl100",
         "GQ": "GQ_dqn", "LQ": "LQ_dqn", "UQ": "UQ_dqn"}
ORDER = ["W", "G", "GQ", "L", "LQ", "U", "UQ"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for k, i in enumerate(C.INSTANCES):
        order = ORDER[k % 7:] + ORDER[:k % 7]                    # ordre tourné
        for s in order:
            key = f"{s}_{i}"
            if key in res:
                continue
            canon = json.loads((RF / f"{CANON[s]}_{i}.json").read_text())
            params = canon.get("params")                         # None pour les variantes statiques
            cloud = CLOUD[s]
            hv, dt = C.solve(C.Path(str(cloud).format(i=i)) if cloud else None, i, RUNS, WORK,
                             nbl=100, reinject=False, params=params)
            res[key] = {"system": s, "instance": i, "params": params, "s_per_run": dt,
                        "s_per_run_canon": canon["s_per_run"], "hv_mean": float(np.mean(hv)),
                        "hv_identical": bool(np.allclose(hv, canon["hv_runs"], rtol=0, atol=1e-12)),
                        "date": time.strftime("%Y-%m-%d %H:%M")}
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(res, indent=1))
            log(f"{i} {s:>2}: {dt:.2f} s/exéc. (canonique {canon['s_per_run']:.2f}) "
                f"HV {'identique' if res[key]['hv_identical'] else 'DIFFÉRENT'}")


if __name__ == "__main__":
    main()
