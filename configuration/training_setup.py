"""
Phase 3 — Train DQN v3 (autoregressive heads + data-driven grids).

Two modes:
    --mode competitor : NO seeds (random init) -> the LLM competitor
    --mode hybrid     : LLM seeds (llm_smart_cloud_*) -> Hybrid v3

Protocol kept identical to v2 for fair comparison:
    900 episodes (100/instance, round-robin), n_reward_runs=1, seed=42,
    lr=1e-3, gamma=0.95, batch=32, buffer 10k, tau=0.005,
    eps 1.0 -> 0.05 (decay 0.995), PER alpha=0.6 beta 0.4 -> 1.0.

Output:
    results/ckpt_v3_<mode>/checkpoint.pt   (+ checkpoint_ep<N>.pt every 100)
    results/training_logs/<mode>_training_log.csv
    results/ckpt_v3_<mode>/best_per_instance.json
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
PROJECT_ROOT = V3_ROOT
sys.path.insert(0, str(V3_ROOT))

from dqn_agent import (
    AutoregressiveDQNAgentV3, EnvironmentV3, GRIDS_V3, LAMBDA_PER_OBJ,
)
from dqn_agent.environment import InstanceBounds  # re-exported import

INSTANCES = [
    ("250.2", "Weights_2obj_FQ200.txt", 250),
    ("250.3", "Weights_3obj_FQ100.txt", 250),
    ("250.4", "Weights_4obj_FQ40.txt", 250),
    ("500.2", "Weights_2obj_FQ200.txt", 500),
    ("500.3", "Weights_3obj_FQ100.txt", 500),
    ("500.4", "Weights_4obj_FQ40.txt", 500),
    ("750.2", "Weights_2obj_FQ200.txt", 750),
    ("750.3", "Weights_3obj_FQ100.txt", 750),
    ("750.4", "Weights_4obj_FQ40.txt", 750),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["competitor", "hybrid"], required=True)
    p.add_argument("--episodes-per-instance", type=int, default=100)
    p.add_argument("--n-reward-runs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_cache(instance: str) -> dict:
    return json.loads(
        (PROJECT_ROOT / "data" / "reference" / f"{instance}.json").read_text())


def make_env(instance: str, weights_file: str, n_items: int,
             mode: str, n_reward_runs: int) -> EnvironmentV3:
    cache = load_cache(instance)
    bounds = InstanceBounds(low=np.array(cache["bounds_low"]),
                            high=np.array(cache["bounds_high"]))
    seeds_path = (PROJECT_ROOT / "clouds" / f"llm_{instance}.txt"
                  if mode == "hybrid" else None)
    return EnvironmentV3(
        instance_path=PROJECT_ROOT / "data" / "instances" / f"{instance}.txt",
        weights_path=PROJECT_ROOT / "data" / "instances" / weights_file,
        seeds_path=seeds_path,
        bounds=bounds,
        baseline_hv=cache["baseline_hv_mean"],
        baseline_time=cache["baseline_run_time"],
        n_objectives=cache["n_objectives"],
        n_items=n_items,
        n_reward_runs=n_reward_runs,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    total_episodes = args.episodes_per_instance * len(INSTANCES)
    out_ckpt = V3_ROOT / "models" / f"ckpt_v3_{args.mode}"
    out_logs = V3_ROOT / "models" / "training_logs"
    out_ckpt.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log(f"DQN v3 training — mode={args.mode.upper()}")
    log(f"  Grids: alpha={GRIDS_V3['alpha']}")
    log(f"         NBL={GRIDS_V3['NBL']}")
    log(f"         L={GRIDS_V3['L']}  kappa={GRIDS_V3['kappa']}")
    n_combo = 1
    for v in GRIDS_V3.values():
        n_combo *= len(v)
    n_out = sum(len(v) for v in GRIDS_V3.values())
    log(f"  Joint combos: {n_combo}  |  network outputs: {n_out} (autoregressive)")
    log(f"  Episodes: {total_episodes} ({args.episodes_per_instance}/instance), "
        f"n_reward_runs={args.n_reward_runs}, seed={args.seed}")
    log("=" * 70)

    envs = {}
    for inst, wf, n_items in INSTANCES:
        envs[inst] = make_env(inst, wf, n_items, args.mode, args.n_reward_runs)
        log(f"  {inst}: baseline_hv={envs[inst].baseline_hv:.4f} "
            f"lam={envs[inst].lambda_cost} seeded={envs[inst].seeds_path is not None}")

    agent = AutoregressiveDQNAgentV3(
        grids=GRIDS_V3,
        per_beta_anneal_steps=total_episodes,
        seed=args.seed,
    )

    log_path = out_logs / f"{args.mode}_training_log.csv"
    with open(log_path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["global_ep", "instance", "epsilon",
                         "alpha", "NBL", "L", "kappa",
                         "hv", "hv_gain", "run_time", "time_ratio", "reward", "loss"])

        global_ep = 0
        best_per_inst: dict[str, dict] = {}
        t_total = time.time()

        for _round in range(args.episodes_per_instance):
            for inst, wf, n_items in INSTANCES:
                env = envs[inst]
                state = env.reset()
                action = agent.select_action(state, explore=True)
                params = agent.decode(action)
                next_state, reward, done, info = env.step(params)
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.update()
                agent.end_episode(reward, loss)

                global_ep += 1
                if info["hv"] > best_per_inst.get(inst, {}).get("hv", -np.inf):
                    best_per_inst[inst] = info.copy()

                writer.writerow([
                    global_ep, inst, f"{agent.epsilon:.4f}",
                    params["alpha"], params["NBL"], params["L"], params["kappa"],
                    f"{info['hv']:.6f}", f"{info['hv_gain']:.6f}",
                    f"{info['run_time']:.3f}", f"{info['time_ratio']:.3f}",
                    f"{reward:.6f}", f"{loss:.6f}" if loss is not None else "",
                ])
                fp.flush()

                if global_ep % 25 == 0 or global_ep == 1:
                    elapsed = time.time() - t_total
                    eta = elapsed / global_ep * (total_episodes - global_ep)
                    log(f"  ep={global_ep:4d}/{total_episodes} inst={inst} "
                        f"eps={agent.epsilon:.3f} "
                        f"a={params['alpha']:>2},N={params['NBL']:>3},L={params['L']},k={params['kappa']:.2f} "
                        f"hv={info['hv']:.4f} r={reward:+.4f} "
                        f"[ETA {eta/60:.0f} min]")
                if global_ep % 100 == 0:
                    agent.save(out_ckpt / f"checkpoint_ep{global_ep}.pt")

    elapsed = time.time() - t_total
    log(f"Training done in {elapsed/60:.1f} min")
    agent.save(out_ckpt / "checkpoint.pt")
    log(f"Checkpoint: {out_ckpt / 'checkpoint.pt'}")

    (out_ckpt / "best_per_instance.json").write_text(json.dumps({
        k: {kk: v[kk] for kk in ("hv", "hv_gain", "run_time", "time_ratio",
                                 "reward", "params", "lambda_used")}
        for k, v in best_per_inst.items()
    }, indent=2))

    # final greedy policy per instance
    log("")
    log("Greedy policy after training:")
    for inst, wf, n_items in INSTANCES:
        env = envs[inst]
        p = agent.decode(agent.policy(env.state))
        log(f"  {inst}: alpha={p['alpha']} NBL={p['NBL']} L={p['L']} kappa={p['kappa']}")


if __name__ == "__main__":
    main()
