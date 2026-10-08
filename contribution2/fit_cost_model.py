"""
v3.1 — Fit the W-CMOLS wall-clock cost model from the 900 measured training
episodes (competitor_training_log.csv). Used for budget action-masking.

Model (log space, per-instance intercept, shared elasticities):
    ln t = intercept[inst] + b_nbl*ln(NBL) + b_l*ln(L) + b_a*ln(alpha) + b_k*kappa

Fit quality (2026-06-11): R2=0.993 (trimmed), median multiplicative error 6.9%,
p90 ~30% (heavy tail = machine-load outliers incl. PC sleep; 5% trimmed refit).

Measured elasticities: NBL^1.57, alpha^1.61, L^0.17, e^(0.40*kappa)
=> alpha drives cost as much as NBL — a budget-constrained agent can trade
   lower alpha for higher NBL at constant wall-clock.

Output: results/cost_model_v31.json
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "models"
LOG = RESULTS / "logs_v3" / "competitor_training_log.csv"
OUT = RESULTS / "cost_model_v31.json"

TRIM_QUANTILE = 0.95  # drop 5% worst residuals (load spikes / PC sleep), refit


def fit() -> dict:
    rows = [r for r in csv.DictReader(open(LOG)) if float(r["run_time"]) > 0.05]
    insts = sorted({r["instance"] for r in rows})
    imap = {k: i for i, k in enumerate(insts)}

    X = np.zeros((len(rows), 4 + len(insts)))
    y = np.zeros(len(rows))
    for j, r in enumerate(rows):
        X[j, 0] = math.log(float(r["NBL"]))
        X[j, 1] = math.log(float(r["L"]))
        X[j, 2] = math.log(float(r["alpha"]))
        X[j, 3] = float(r["kappa"])
        X[j, 4 + imap[r["instance"]]] = 1.0
        y[j] = math.log(float(r["run_time"]))

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = np.abs(y - X @ coef)
    keep = res < np.quantile(res, TRIM_QUANTILE)
    coef, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)

    pred = X @ coef
    ss = np.sum((y[keep] - pred[keep]) ** 2) / np.sum((y[keep] - np.mean(y[keep])) ** 2)
    mult_err = np.exp(np.abs(y - pred)) - 1

    model = {
        "form": "ln_t = intercept[inst] + b_nbl*ln(NBL) + b_l*ln(L) + b_a*ln(alpha) + b_k*kappa",
        "b_nbl": float(coef[0]),
        "b_l": float(coef[1]),
        "b_a": float(coef[2]),
        "b_k": float(coef[3]),
        "intercepts": {k: float(coef[4 + imap[k]]) for k in insts},
        "fit": {
            "n_episodes": len(rows),
            "r2_log_trimmed": float(1 - ss),
            "median_mult_error": float(np.median(mult_err)),
            "p90_mult_error": float(np.quantile(mult_err, 0.90)),
            "trim_quantile": TRIM_QUANTILE,
            "source_log": str(LOG.name),
        },
    }
    return model


def predict_time(model: dict, instance: str, alpha: float, nbl: float,
                 l_val: float, kappa: float) -> float:
    """Predicted wall-clock seconds for one W-CMOLS run."""
    ln_t = (model["intercepts"][instance]
            + model["b_nbl"] * math.log(nbl)
            + model["b_l"] * math.log(l_val)
            + model["b_a"] * math.log(alpha)
            + model["b_k"] * kappa)
    return math.exp(ln_t)


if __name__ == "__main__":
    m = fit()
    OUT.write_text(json.dumps(m, indent=2))
    print(f"saved {OUT}")
    print(f"R2={m['fit']['r2_log_trimmed']:.3f} med_err={m['fit']['median_mult_error']*100:.1f}% "
          f"p90={m['fit']['p90_mult_error']*100:.1f}%")
    print(f"elasticities: NBL^{m['b_nbl']:.2f} alpha^{m['b_a']:.2f} "
          f"L^{m['b_l']:.2f} e^({m['b_k']:.2f}*kappa)")
    # sanity: compare predictions vs the 8 measured eval configs (50-run averages)
    eval_known = {
        "250.2": (10, 70, 8, 0.2, 17.78 / 50),
        "250.3": (20, 160, 3, 0.2, 328.30 / 50),
        "250.4": (15, 160, 10, 0.2, 672.46 / 50),
        "500.2": (25, 70, 3, 0.05, 172.57 / 50),
        "500.3": (25, 160, 10, 0.1, 1881.95 / 50),
        "500.4": (20, 160, 5, 0.2, 3797.31 / 50),
        "750.2": (15, 160, 3, 0.2, 391.57 / 50),
        "750.3": (25, 160, 3, 0.2, 3074.20 / 50),
    }
    print("\nsanity vs eval-measured (s/run):")
    for inst, (a, n, l, k, t_meas) in eval_known.items():
        t_pred = predict_time(m, inst, a, n, l, k)
        print(f"  {inst}: pred={t_pred:6.2f}  measured={t_meas:6.2f}  "
              f"ratio={t_pred / t_meas:.2f}")
