"""
Environment v3 — same reward semantics as Hybrid v2, but seeds are OPTIONAL.

    seeds_path = None  -> competitor mode (random init, DQN params alone)
    seeds_path = file  -> hybrid mode (LLM seeds + DQN params)

Reward (unchanged from v2 for fair comparison):
    reward = (median_HV - baseline_hv) / baseline_hv
             - lambda_per_obj * max(0, time_ratio - 1)
    LAMBDA_PER_OBJ = {2: 0.10, 3: 0.05, 4: 0.10}

State encoding (unchanged): [n_obj/4, log(n_items), baseline_hv, log(t_base)].
HV bounds: data/reference/<instance>.json (same bounds for every system).
"""
from __future__ import annotations

import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_BASELINE = _ROOT / "work" / "train"          # sorties du solveur pendant l'entraînement
_BASELINE.mkdir(parents=True, exist_ok=True)
for p in (str(_ROOT / "solver"), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import moacp_mut  # type: ignore[import-not-found]

from hypervolume.hv_calculator import InstanceBounds, compute_hv, parse_all_runs  # noqa: E402

LAMBDA_PER_OBJ: dict[int, float] = {2: 0.10, 3: 0.05, 4: 0.10}


def encode_state(n_objectives: int, n_items: int, baseline_hv: float, baseline_time: float) -> np.ndarray:
    """Identical encoding to v1/v2 for comparability."""
    return np.array([
        n_objectives / 4.0,
        math.log(n_items / 250.0) / math.log(3.0) if n_items > 0 else 0.0,
        float(np.clip(baseline_hv, 0.0, 1.0)),
        math.log(max(baseline_time, 0.4) / 0.4) / math.log(40.0),
    ], dtype=np.float32)


class EnvironmentV3:
    """W-CMOLS environment, optional LLM seeds, v2 reward semantics."""

    def __init__(
        self,
        instance_path: Path,
        weights_path: Path,
        seeds_path: Path | None,
        bounds: InstanceBounds,
        baseline_hv: float,
        baseline_time: float,
        n_objectives: int,
        n_items: int,
        n_reward_runs: int = 1,
        lambda_cost: float | None = None,
        solver=None,
    ):
        self.instance_path = Path(instance_path)
        self.weights_path = Path(weights_path)
        self.seeds_path = Path(seeds_path) if seeds_path is not None else None
        if self.seeds_path is not None and not self.seeds_path.exists():
            raise FileNotFoundError(f"LLM seeds file missing: {self.seeds_path}")

        # solveur : moacp_mut par défaut (réinjection active dès qu'un cloud est fourni) ;
        # la version finale passe moacp_noreinj (même code, réinjection désactivée).
        self.solver = solver if solver is not None else moacp_mut
        self.bounds = bounds
        self.baseline_hv = max(baseline_hv, 1e-9)
        self.baseline_time = max(baseline_time, 1e-3)
        self.n_objectives = n_objectives
        self.n_items = n_items
        self.n_reward_runs = max(1, int(n_reward_runs))
        self.lambda_cost = float(lambda_cost if lambda_cost is not None
                                 else LAMBDA_PER_OBJ.get(n_objectives, 0.1))

        self._state = encode_state(n_objectives, n_items, baseline_hv, baseline_time)
        self._all_runs = _BASELINE / "all_runs.txt"
        self._pareto_sizes = _BASELINE / "pareto_sizes.txt"
        self.episode_count = 0
        self.last_info: dict | None = None

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    def reset(self) -> np.ndarray:
        self.episode_count += 1
        self.last_info = None
        return self.state

    def step(self, params: dict) -> tuple[np.ndarray, float, bool, dict]:
        """params: decoded {'alpha':..,'NBL':..,'L':..,'kappa':..} (agent.decode)."""
        prev_cwd = os.getcwd()
        run_hvs: list[float] = []
        pareto_sizes: list[int] = []
        try:
            os.chdir(_BASELINE)
            for f in (self._all_runs, self._pareto_sizes):
                # Windows: l'AV/indexeur peut tenir le fichier juste après un run
                for attempt in range(12):
                    try:
                        f.unlink()
                        break
                    except FileNotFoundError:
                        break
                    except PermissionError:
                        if attempt == 11:
                            raise
                        time.sleep(0.5)
            self.solver.set_runtime_params(alpha_val=params["alpha"], L_val=params["L"],
                                         kappa_val=params["kappa"])
            self.solver.set_iterations(params["NBL"])
            fronts = None
            for _attempt in range(3):
                t0 = time.time()
                self.solver.run_moacp_ex(
                    str(self.instance_path).encode(),
                    str(self.weights_path).encode(),
                    (str(self.seeds_path).encode() if self.seeds_path is not None else b""),
                    self.n_reward_runs, -1,
                )
                mean_time = (time.time() - t0) / self.n_reward_runs
                try:  # W-CMOLS occasionally writes a malformed row -> re-run
                    fronts = parse_all_runs(self._all_runs, self._pareto_sizes)
                    break
                except (ValueError, OSError):
                    fronts = None
                    for _f in (self._all_runs, self._pareto_sizes):
                        try:
                            _f.unlink()
                        except FileNotFoundError:
                            pass
            if fronts is None:
                raise RuntimeError("W-CMOLS output unparseable after 3 attempts")
            for f in fronts[:self.n_reward_runs]:
                run_hvs.append(compute_hv(f, self.bounds) if f.size else 0.0)
                pareto_sizes.append(len(f))
        finally:
            os.chdir(prev_cwd)

        if not run_hvs:
            run_hvs, pareto_sizes, mean_time = [0.0], [0], self.baseline_time

        # Statistique risk-aware sur 2-obj (mesuré éval n10 2026-06-11) : le
        # paysage 2-obj sans seeds est BIMODAL (échecs catastrophiques α faible) ;
        # la médiane multi-runs les cache → q25 expose la queue, MW teste les rangs.
        if self.n_objectives == 2 and len(run_hvs) >= 4:
            hv_median = float(np.quantile(run_hvs, 0.25))
        else:
            hv_median = float(statistics.median(run_hvs))
        hv_gain = (hv_median - self.baseline_hv) / self.baseline_hv
        time_ratio = mean_time / self.baseline_time
        cost_penalty = self.lambda_cost * max(0.0, time_ratio - 1.0)
        reward = float(hv_gain - cost_penalty)

        info = {
            "hv": hv_median, "hv_gain": float(hv_gain), "hv_runs": run_hvs,
            "run_time": float(mean_time), "time_ratio": float(time_ratio),
            "cost_penalty": float(cost_penalty), "reward": reward,
            "params": params, "pareto_size": int(np.median(pareto_sizes)),
            "lambda_used": self.lambda_cost, "n_reward_runs": self.n_reward_runs,
            "seeded": self.seeds_path is not None,
        }
        self.last_info = info
        return self.state, reward, True, info
