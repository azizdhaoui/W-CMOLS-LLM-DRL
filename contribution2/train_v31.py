"""
v3.1 — Budget-constrained fine-tuning (warm-start from the v3 checkpoint).

Fixes the v3 runtime problem (lambda penalty measured 5-20x too weak: the
agent rationally bought HV with time, NBL=160 everywhere). v3.1 enforces a
HARD per-instance wall-clock budget via action masking during the
autoregressive decode (rl_agent_v3/budget_mask.py + cost_model_v31.json,
R2=0.993), instead of relying on the soft penalty.

    budget_seconds[inst] = baseline_time_fresh[inst] * --budget-ratio (1.0)

Goal (Aziz 2026-06-11): win HV + p-value vs baseline/LLM with per-run time
<= baseline (LLM_improvement runs baseline params, so its time ~ baseline's).

Changes vs train_v3.py:
    - warm-start (default: ckpt_v3_competitor) -> short fine-tune, default
      25 ep/instance = 225 episodes, epsilon restart 0.4 -> 0.05 (decay 0.99)
    - masked select_action (greedy AND random exploration)
    - env baseline_time = FRESH re-measured time (baseline_times_fresh.json),
      so the reward's time_ratio is correct too; baseline HV stays cached
    - small symmetric speed bonus: reward += MU * max(0, 1 - time_ratio)
      (the v3 reward never rewarded being fast, only punished being slow)

Prereqs: results/cost_model_v31.json + results/baseline_times_fresh.json.

Output:
    results/ckpt_v31_<mode>/checkpoint.pt (+ checkpoint_ep<N>.pt every 75)
    results/ckpt_v31_<mode>/budgets.json   (budgets used, for the eval script)
    results/logs_v3/v31_<mode>_training_log.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

V3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rl_agent_v3 import AutoregressiveDQNAgentV3, BudgetMask, load_cost_model
from rl_agent_v3.budget_mask import predict_time
from train_v3 import INSTANCES, load_cache, make_env  # noqa: E402

COST_MODEL = V3_ROOT / "models" / "cost_model_v31.json"
FRESH_TIMES = V3_ROOT / "models" / "baseline_times_fresh.json"
MU_SPEED_BONUS = 0.03


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["competitor", "hybrid", "greedy", "union"], required=True)
    p.add_argument("--episodes-per-instance", type=int, default=25)
    p.add_argument("--n-reward-runs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--budget-ratio", type=float, default=1.0,
                   help="budget = fresh baseline time x this ratio")
    p.add_argument("--warm-start", type=str, default="auto",
                   help="'auto' (competitor: ckpt_v3_competitor, hybrid: "
                        "ckpt_v31_competitor), 'none' (scratch), or a path")
    p.add_argument("--epsilon-start", type=float, default=0.4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--suffix", type=str, default="",
                   help="appended to the output tag (e.g. _n10) to keep variants apart")
    p.add_argument("--no-dueling", action="store_true", help="Ablation: disable Dueling heads")
    p.add_argument("--no-per", action="store_true", help="Ablation: disable PER (alpha=0)")
    p.add_argument("--nu", type=float, default=0.0,
                   help="cardinality reward weight: reward += nu*(pareto_size-base_card)/base_card")
    p.add_argument("--no-speed-bonus", action="store_true",
                   help="drop the MU speed bonus (use with --nu so nothing pushes NBL back down)")
    p.add_argument("--budgets-json", type=str, default=None,
                   help="JSON {inst: per-run budget seconds} overriding --budget-ratio "
                        "(lets total time be capped: sum(50*budget) <= target)")
    # --- version finale (sept. 2026) : toutes les options gardent l'ancien comportement par défaut
    p.add_argument("--solver", choices=["mut", "noreinj"], default="mut",
                   help="noreinj = solveur final (réinjection désactivée)")
    p.add_argument("--seeds-pattern", type=str, default=None,
                   help="chemin du cloud, avec {inst} (remplace le cloud du mode)")
    p.add_argument("--cost-model", type=str, default=None, help="modèle de coût (défaut : cost_model_v31.json)")
    p.add_argument("--fresh-times", type=str, default=None,
                   help="temps de référence de W-CMOLS (défaut : baseline_times_fresh.json)")
    p.add_argument("--lambda3", type=float, default=None, help="pénalité de temps à 3 objectifs (défaut 0,05)")
    p.add_argument("--exclude", type=str, default=None,
                   help="validation croisée : instance retirée de l'entraînement (ex. 500.4)")
    p.add_argument("--guard", choices=["always", "if_fits"], default="always",
                   help="if_fits : garde-fou autorisé seulement si son coût prédit tient dans le budget")
    return p.parse_args()


def resolve_warm_start(args: argparse.Namespace) -> Path | None:
    if args.warm_start == "none":
        return None
    if args.warm_start == "auto":
        name = {"competitor": "ckpt_v3_competitor",
                "hybrid": f"ckpt_v31_competitor_b{args.budget_ratio:g}",
                # greedy : warm-start de la politique déjà calibrée pour runs SEEDÉS
                "greedy": "ckpt_v31_hybrid_b1",
                # union : warm-start du meilleur checkpoint greedy (A10)
                "union": "ckpt_v31_greedy_b1_n10"}[args.mode]
        return V3_ROOT / "models" / name / "checkpoint.pt"
    return Path(args.warm_start)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cost_model_path = Path(args.cost_model) if args.cost_model else COST_MODEL
    fresh_path = Path(args.fresh_times) if args.fresh_times else FRESH_TIMES
    for f in (cost_model_path, fresh_path):
        if not f.exists():
            sys.exit(f"MISSING PREREQ: {f} (run fit_cost_model.py / "
                     f"measure_baseline_fresh.py first)")
    cost_model = load_cost_model(cost_model_path)
    fresh = json.loads(fresh_path.read_text())
    solver = None
    if args.solver == "noreinj":
        sys.path.insert(0, str(V3_ROOT / "solver"))
        import moacp_noreinj as solver  # type: ignore[import-not-found]
    base_card = (json.loads((V3_ROOT / "models" / "baseline_card.json").read_text())
                 if args.nu > 0 else {})
    custom_budgets = (json.loads(Path(args.budgets_json).read_text())
                      if args.budgets_json else None)

    TRAIN = [x for x in INSTANCES if x[0] != args.exclude]
    assert args.exclude is None or len(TRAIN) == len(INSTANCES) - 1, args.exclude
    total_episodes = args.episodes_per_instance * len(TRAIN)
    tag = f"{args.mode}_b{args.budget_ratio:g}{args.suffix}"   # e.g. competitor_b1.3
    out_ckpt = V3_ROOT / "models" / f"ckpt_v31_{tag}"
    out_logs = V3_ROOT / "models" / "logs_v3"
    out_ckpt.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    warm = resolve_warm_start(args)
    log("=" * 70)
    log(f"DQN v3.1 BUDGET-CONSTRAINED — mode={args.mode.upper()}")
    log(f"  budget_ratio={args.budget_ratio} (vs FRESH baseline times)")
    log(f"  warm-start: {warm if warm else 'from scratch'}")
    log(f"  episodes: {total_episodes}, eps {args.epsilon_start}->0.05, "
        f"lr={args.lr}, mu_speed={MU_SPEED_BONUS}, seed={args.seed}")
    log("=" * 70)

    # envs with FRESH baseline times (correct reward time_ratio).
    # NOTE: the STATE keeps the cached-time encoding (set in __init__, before
    # this override) — intentional: the warm-started v3 Q-values were trained
    # on that state distribution; fresh time only drives reward and budget.
    envs, masks, budgets = {}, {}, {}
    for inst, wf, n_items in TRAIN:
        env = make_env(inst, wf, n_items, args.mode, args.n_reward_runs)
        if args.mode == "greedy":   # make_env ne connaît que hybrid/competitor
            env.seeds_path = (V3_ROOT
                              / "clouds" / f"greedy_cloud_{inst}.txt")
            assert env.seeds_path.exists(), env.seeds_path
        elif args.mode == "union":  # A10 : cloud-union greedy+LLM (1510 total)
            env.seeds_path = (V3_ROOT
                              / "clouds" / f"u2_{inst}.txt")
            assert env.seeds_path.exists(), env.seeds_path
        if args.seeds_pattern:
            # chemin absolu : le solveur lit le cloud depuis son propre dossier de travail
            env.seeds_path = Path(args.seeds_pattern.format(inst=inst)).resolve()
            assert env.seeds_path.exists(), env.seeds_path
        if solver is not None:
            env.solver = solver
        if args.lambda3 is not None and env.n_objectives == 3:
            env.lambda_cost = args.lambda3
        t_fresh = fresh[inst]["baseline_time_fresh"]
        env.baseline_time = max(t_fresh, 1e-3)
        envs[inst] = env
        budgets[inst] = (custom_budgets[inst] if custom_budgets else t_fresh * args.budget_ratio)
        log(f"  {inst}: budget={budgets[inst]:.2f}s/run "
            f"(fresh={t_fresh:.2f}s, cache était {fresh[inst]['stale_factor']:.2f}x)")

    # agent: warm-start or scratch (grids come from the checkpoint when warm)
    if warm is not None:
        if args.no_dueling or args.no_per:
            sys.exit("ablation flags require --warm-start none (clean from-scratch)")
        agent = AutoregressiveDQNAgentV3.load(warm, seed=args.seed)
        log(f"  warm-started: step_count={agent.step_count}, grids={agent.grids}")
    else:
        from rl_agent_v3 import GRIDS_V3
        per_alpha_val = 0.0 if args.no_per else 0.6
        agent = AutoregressiveDQNAgentV3(grids=GRIDS_V3,
                                         per_beta_anneal_steps=total_episodes,
                                         seed=args.seed,
                                         dueling=not args.no_dueling,
                                         per_alpha=per_alpha_val)
        log(f"  from scratch: dueling={not args.no_dueling}, "
            f"per_alpha={per_alpha_val}")
    agent.epsilon = args.epsilon_start
    agent.epsilon_decay = 0.99
    for g in agent.optimizer.param_groups:
        g["lr"] = args.lr

    baseline_guard = {"alpha": 10, "NBL": 100, "L": 5, "kappa": 0.05}
    for inst in budgets:
        g = baseline_guard
        if args.guard == "if_fits" and predict_time(cost_model, inst, **baseline_guard) > budgets[inst]:
            g = None
        masks[inst] = BudgetMask(cost_model, inst, agent.grids, budgets[inst], guard=g)
        log(f"  {inst}: feasible combos = {masks[inst].feasible_combos()}/432")

    (out_ckpt / "budgets.json").write_text(json.dumps({
        "budget_ratio": args.budget_ratio,
        "budgets_seconds": budgets,
        "mu_speed_bonus": MU_SPEED_BONUS,
        "warm_start": str(warm) if warm else None,
        "cost_model": str(cost_model_path),
        "fresh_times": str(fresh_path),
        "solver": args.solver, "seeds_pattern": args.seeds_pattern,
        "lambda3": args.lambda3, "guard": args.guard, "nu": args.nu,
        "speed_bonus": not args.no_speed_bonus, "exclude": args.exclude,
    }, indent=2))

    log_path = out_logs / f"v31_{tag}_training_log.csv"
    with open(log_path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["global_ep", "instance", "epsilon",
                         "alpha", "NBL", "L", "kappa",
                         "hv", "hv_gain", "run_time", "time_ratio",
                         "pred_time", "budget", "reward", "loss"])

        global_ep = 0
        best_per_inst: dict[str, dict] = {}
        t_total = time.time()

        for _round in range(args.episodes_per_instance):
            for inst, wf, n_items in TRAIN:
                env = envs[inst]
                state = env.reset()
                action = agent.select_action(state, explore=True,
                                             mask_fn=masks[inst].mask_fn)
                params = agent.decode(action)
                next_state, reward, done, info = env.step(params)
                # symmetric speed bonus (v3 reward only punished slowness)
                if not args.no_speed_bonus:
                    reward += MU_SPEED_BONUS * max(0.0, 1.0 - info["time_ratio"])
                # cardinality-aware term (v4): reward more solutions vs baseline
                if args.nu > 0:
                    bc = base_card.get(inst, 1.0)
                    reward += args.nu * (info["pareto_size"] - bc) / max(bc, 1.0)
                reward = float(reward)
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.update()
                agent.end_episode(reward, loss)

                global_ep += 1
                if info["hv"] > best_per_inst.get(inst, {}).get("hv", -np.inf):
                    best_per_inst[inst] = info.copy()

                t_pred = predict_time(cost_model, inst, **params)
                writer.writerow([
                    global_ep, inst, f"{agent.epsilon:.4f}",
                    params["alpha"], params["NBL"], params["L"], params["kappa"],
                    f"{info['hv']:.6f}", f"{info['hv_gain']:.6f}",
                    f"{info['run_time']:.3f}", f"{info['time_ratio']:.3f}",
                    f"{t_pred:.3f}", f"{budgets[inst]:.3f}",
                    f"{reward:.6f}", f"{loss:.6f}" if loss is not None else "",
                ])
                fp.flush()

                if global_ep % 25 == 0 or global_ep == 1:
                    elapsed = time.time() - t_total
                    eta = elapsed / global_ep * (total_episodes - global_ep)
                    log(f"  ep={global_ep:4d}/{total_episodes} inst={inst} "
                        f"eps={agent.epsilon:.3f} "
                        f"a={params['alpha']:>2},N={params['NBL']:>3},"
                        f"L={params['L']},k={params['kappa']:.2f} "
                        f"hv={info['hv']:.4f} ratio={info['time_ratio']:.2f} "
                        f"r={reward:+.4f} [ETA {eta/60:.0f} min]")
                if global_ep % 75 == 0:
                    agent.save(out_ckpt / f"checkpoint_ep{global_ep}.pt")

    elapsed = time.time() - t_total
    log(f"Fine-tuning done in {elapsed/60:.1f} min")
    agent.save(out_ckpt / "checkpoint.pt")
    log(f"Checkpoint: {out_ckpt / 'checkpoint.pt'}")

    (out_ckpt / "best_per_instance.json").write_text(json.dumps({
        k: {kk: v[kk] for kk in ("hv", "hv_gain", "run_time", "time_ratio",
                                 "reward", "params", "lambda_used")}
        for k, v in best_per_inst.items()
    }, indent=2))

    log("")
    log("Greedy MASKED policy after fine-tuning:")
    for inst, wf, n_items in TRAIN:
        env = envs[inst]
        p = agent.decode(agent.policy(env.state, mask_fn=masks[inst].mask_fn))
        t_pred = predict_time(cost_model, inst, **p)
        log(f"  {inst}: alpha={p['alpha']} NBL={p['NBL']} L={p['L']} "
            f"kappa={p['kappa']}  pred={t_pred:.1f}s budget={budgets[inst]:.1f}s")


if __name__ == "__main__":
    main()
