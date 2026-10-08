"""
Extrait la trace de decision reelle de l'agent SW-CMOLS-Q pour les 9 instances.

Rejoue exactement la décision de l'agent entraîné (evaluate_agent.py) :
    cache instance -> encode_state -> BudgetMask -> decodage glouton masque,
en reproduisant pas a pas la boucle de AutoregressiveDQNAgentV3._masked_greedy
afin d'enregistrer, pour chaque tete et dans l'ordre autoregressif
(alpha, NBL, L, kappa) : les valeurs Q brutes, le masque de budget et le choix.

Sortie : models/decision_trace_<tag>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

V3_ROOT = Path(__file__).resolve().parents[1]
REPO = V3_ROOT
sys.path.insert(0, str(V3_ROOT))

from dqn_agent import (AutoregressiveDQNAgentV3, BudgetMask, encode_state,  # noqa: E402
                         load_cost_model)
from dqn_agent.budget_mask import predict_time  # noqa: E402

COST_MODEL = V3_ROOT / "models" / "cost_model_base.json"
INSTANCES = ["250.2", "250.3", "250.4", "500.2", "500.3", "500.4",
             "750.2", "750.3", "750.4"]
ORDER = ["alpha", "NBL", "L", "kappa"]
GUARD = {"alpha": 10, "NBL": 100, "L": 5, "kappa": 0.05}


def trace_decode(agent, state, mask_fn):
    """Copie fidele de _masked_greedy, en conservant les valeurs intermediaires."""
    st = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
    h = agent.q_main.trunk(st)
    chosen_idx: dict[str, int] = {}
    ctx_parts: list[torch.Tensor] = []
    heads = {}
    for d in ORDER:
        ctx = torch.cat(ctx_parts, dim=1) if ctx_parts else None
        q = agent.q_main._q_head(d, h, ctx)[0]
        allowed = np.asarray(mask_fn(d, chosen_idx), dtype=bool)
        q_masked = q.masked_fill(~torch.tensor(allowed), float("-inf"))
        a = int(q_masked.argmax())
        grid = list(agent.grids[d])
        heads[d] = {
            "grid": grid,
            "q": [float(v) for v in q.detach().numpy()],
            "allowed": [bool(v) for v in allowed],
            "chosen_index": a,
            "chosen": grid[a],
        }
        chosen_idx[d] = a
        ctx_parts.append(F.one_hot(torch.tensor([a]), num_classes=agent.dims[d]).float())
    params = {d: heads[d]["chosen"] for d in ORDER}
    return params, heads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="union_b1_card")
    ap.add_argument("--cost-model", default=str(COST_MODEL))
    ap.add_argument("--budgets-json", default=None,
                    help="budgets par instance (défaut : budgets.json du checkpoint)")
    ap.add_argument("--guard", choices=["always", "if_fits"], default="always")
    args = ap.parse_args()

    ckpt_dir = V3_ROOT / "models" / f"policy_{args.tag}"
    agent = AutoregressiveDQNAgentV3.load(ckpt_dir / "checkpoint.pt")
    agent.epsilon = 0.0
    bpath = Path(args.budgets_json) if args.budgets_json else ckpt_dir / "budgets.json"
    budgets = json.loads(bpath.read_text())
    budgets = budgets.get("budgets_seconds", budgets)
    cost_model = load_cost_model(args.cost_model)

    out = {"tag": args.tag, "cost_model": str(args.cost_model), "guard_rule": args.guard, "grids": {k: list(v) for k, v in agent.grids.items()},
           "order": ORDER, "default": GUARD, "instances": {}}

    for inst in INSTANCES:
        n_items = int(inst.split(".")[0])
        cache = json.loads((REPO / "data" / "reference" / f"{inst}.json").read_text())
        state = encode_state(cache["n_objectives"], n_items,
                             cache["baseline_hv_mean"], cache["baseline_run_time"])
        guard = GUARD
        if args.guard == "if_fits" and predict_time(cost_model, inst, **GUARD) > budgets[inst]:
            guard = None
        mask = BudgetMask(cost_model, inst, agent.grids, budgets[inst], guard=guard)

        t0 = time.perf_counter()
        params, heads = trace_decode(agent, state, mask.mask_fn)
        decide_ms = (time.perf_counter() - t0) * 1000.0

        # controle : identique a la politique officielle
        ref = agent.decode(agent.policy(state, mask_fn=mask.mask_fn))
        assert ref == params, f"{inst}: {ref} != {params}"

        out["instances"][inst] = {
            "n_objectives": cache["n_objectives"],
            "n_items": n_items,
            "baseline_hv_mean": cache["baseline_hv_mean"],
            "baseline_run_time": cache["baseline_run_time"],
            "state": [float(v) for v in state],
            "params": params,
            "heads": heads,
            "budget_s": float(budgets[inst]),
            "pred_time_s": float(predict_time(cost_model, inst, **params)),
            "default_pred_time_s": float(predict_time(cost_model, inst, **GUARD)),
            "decide_ms": decide_ms,
            "feasible_combos": int(mask.feasible_combos()),
            "guard_kept": guard is not None,
            "raw_argmax": {d: heads[d]["grid"][int(np.argmax(heads[d]["q"]))] for d in ORDER},
        }
        print(f"{inst}: {params}  pred={out['instances'][inst]['pred_time_s']:.2f}s "
              f"budget={budgets[inst]:.2f}s  combos={out['instances'][inst]['feasible_combos']}/432 "
              f"({decide_ms:.2f} ms)")

    dest = V3_ROOT / "models" / f"decision_trace_{args.tag}.json"
    dest.write_text(json.dumps(out, indent=1))
    print("->", dest)


if __name__ == "__main__":
    main()
