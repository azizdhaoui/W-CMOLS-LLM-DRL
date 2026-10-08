"""
v3.1 — Budget action-masking for the autoregressive decode.

Hard time constraint (replaces the too-weak lambda penalty, measured 5-20x
under-priced in v3): a sub-action value is feasible iff there EXISTS a
completion of the remaining dims whose predicted wall-clock stays under the
per-instance budget. Cost is monotone increasing in every param (elasticities
NBL^1.57, alpha^1.61, L^0.17, e^(0.40*kappa), all > 0), so the cheapest
completion is simply the minimum of each remaining grid.

Cost model: results/cost_model_v31.json (fit: experiments/fit_cost_model.py,
R2=0.993, median error 6.9% vs held-out eval times).

Usage:
    mask = BudgetMask(cost_model, instance="500.3", grids=GRIDS_V3,
                      budget_seconds=fresh_baseline_time * 1.0)
    fn = mask.mask_fn          # (dim, chosen_idx: dict[str,int]) -> bool array
    agent.select_action(state, mask_fn=fn)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

PARAM_ORDER = ("alpha", "NBL", "L", "kappa")


def load_cost_model(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def predict_time(model: dict, instance: str, alpha: float, NBL: float,
                 L: float, kappa: float) -> float:
    """Predicted wall-clock seconds for one W-CMOLS run."""
    ln_t = (model["intercepts"][instance]
            + model["b_nbl"] * math.log(NBL)
            + model["b_l"] * math.log(L)
            + model["b_a"] * math.log(alpha)
            + model["b_k"] * kappa)
    return math.exp(ln_t)


class BudgetMask:
    """Per-instance feasibility masks for sequential (autoregressive) decode."""

    def __init__(self, cost_model: dict, instance: str,
                 grids: dict[str, tuple], budget_seconds: float,
                 guard: dict[str, float] | None = None):
        """guard: combo toujours décodable (ex. params baseline) — répare le bug
        de bord à parité : baseline coûte PILE 1.0x le budget, l'erreur ±7% du
        modèle de coût pouvait l'exclure du faisable (250.3 -17% en éval @1.0).
        À budget >= 1.0, un agent calibré ne doit jamais pouvoir perdre."""
        if instance not in cost_model["intercepts"]:
            raise KeyError(f"instance {instance} absent from cost model")
        self.model = cost_model
        self.instance = instance
        self.grids = {k: tuple(v) for k, v in grids.items()}
        self.budget = float(budget_seconds)
        self.guard = dict(guard) if guard else None

    def _combo_time(self, combo: dict[str, float]) -> float:
        return predict_time(self.model, self.instance, **combo)

    def mask_fn(self, dim: str, chosen_idx: dict[str, int]) -> np.ndarray:
        """Bool array over grids[dim]: True = a feasible completion exists.

        chosen_idx: dims already decoded this step -> grid INDEX.
        Never returns an all-False mask: if the budget is unreachable even at
        the cheapest completion, only the cheapest value of `dim` is allowed
        (degenerate-but-valid policy instead of a crash).
        """
        fixed = {d: self.grids[d][i] for d, i in chosen_idx.items()}
        remaining = [d for d in PARAM_ORDER if d != dim and d not in fixed]
        cheapest = {d: min(self.grids[d]) for d in remaining}

        times = np.array([
            self._combo_time({**fixed, dim: v, **cheapest})
            for v in self.grids[dim]
        ])
        allowed = times <= self.budget
        # garde-fou : le chemin guard (params baseline) reste toujours décodable
        if self.guard is not None:
            on_guard_path = all(self.grids[d][i] == self.guard[d]
                                for d, i in chosen_idx.items())
            if on_guard_path and self.guard[dim] in self.grids[dim]:
                allowed[self.grids[dim].index(self.guard[dim])] = True
        if not allowed.any():
            allowed[int(np.argmin(times))] = True
        return allowed

    def feasible_combos(self) -> int:
        """Count of full combos under budget (diagnostics)."""
        count = 0
        for a in self.grids["alpha"]:
            for n in self.grids["NBL"]:
                for l in self.grids["L"]:
                    for k in self.grids["kappa"]:
                        if self._combo_time(
                                {"alpha": a, "NBL": n, "L": l, "kappa": k}) <= self.budget:
                            count += 1
        return count
