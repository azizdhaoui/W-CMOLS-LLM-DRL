"""
DQN v3 — Autoregressive (conditioned) Dueling DQN + PER.

Key change vs v2 (F4 fix): the 4 parameter heads are no longer independent.
Parameters are selected SEQUENTIALLY, each head conditioned on the state AND
the choices already made:

    alpha ~ Q_a(s)
    NBL   ~ Q_N(s, alpha)
    L     ~ Q_L(s, alpha, NBL)
    kappa ~ Q_k(s, alpha, NBL, L)

so interactions like "alpha=25 works with NBL=50 but not NBL=100" are
representable BY CONSTRUCTION (the NBL head sees which alpha was picked),
while keeping a small output count (sum of grid sizes, not their product).
This is the standard autoregressive factorization used for large factored
action spaces (cf. Metz et al. 2017 "Discrete Sequential Prediction of
Continuous Actions", NAS controllers).

Second change (F6 fix): the grids are CONSTRUCTOR PARAMETERS chosen from
measured 1D response profiles (see GRID_DESIGN_V3.md), not hardcoded ad hoc
values. The grids are stored inside the checkpoint so evaluation always
decodes with the exact grids the network was trained on.

Kept from v2 (for fair comparison): Dueling per head, proportional PER,
soft target update (tau=0.005), epsilon-greedy, 1-step target
(target = r + (1-done)*gamma*maxQ', with done=True in our setting).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PARAM_ORDER = ("alpha", "NBL", "L", "kappa")

# Default grids = v2 grids. The real v3 grids are injected by training_setup.py from
# grids.py after the Phase-1 profile analysis (GRID_DESIGN_V3.md).
DEFAULT_GRIDS: dict[str, tuple] = {
    "alpha": (10, 15, 20, 25, 30, 40, 50),
    "NBL":   (30, 50, 70, 100),
    "L":     (3, 5, 8),
    "kappa": (0.05, 0.1, 0.2),
}


@dataclass(frozen=True)
class ActionV3:
    """Indices into the agent's grids, in PARAM_ORDER."""
    alpha_idx: int
    NBL_idx: int
    L_idx: int
    kappa_idx: int

    def as_tuple(self) -> tuple:
        return (self.alpha_idx, self.NBL_idx, self.L_idx, self.kappa_idx)


@dataclass
class Transition:
    state: np.ndarray
    action: tuple
    reward: float
    next_state: np.ndarray
    done: bool


# --------------------------------------------------------------------------- #
# Network: trunk + autoregressive conditioned dueling heads
# --------------------------------------------------------------------------- #

class AutoregressiveDuelingNet(nn.Module):
    """Q_d(s, a_d | a_<d): each head sees trunk(state) + one-hots of previous picks."""

    def __init__(self, state_dim: int, dims: dict[str, int],
                 hidden: int = 128, cond_hidden: int = 64,
                 dueling: bool = True):
        super().__init__()
        self.dims = dict(dims)
        self.dueling = dueling   # False = têtes Q simples (ablation E4)
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.cond = nn.ModuleDict()
        self.v_heads = nn.ModuleDict()
        self.a_heads = nn.ModuleDict()
        ctx = 0
        for d in PARAM_ORDER:
            self.cond[d] = nn.Sequential(nn.Linear(hidden + ctx, cond_hidden), nn.ReLU())
            if dueling:
                self.v_heads[d] = nn.Linear(cond_hidden, 1)
            self.a_heads[d] = nn.Linear(cond_hidden, dims[d])
            ctx += dims[d]   # next head also sees this head's one-hot

    def _q_head(self, d: str, h: torch.Tensor, ctx: torch.Tensor | None) -> torch.Tensor:
        x = h if ctx is None else torch.cat([h, ctx], dim=1)
        z = self.cond[d](x)
        a = self.a_heads[d](z)
        if not self.dueling:
            return a
        v = self.v_heads[d](z)
        return v + (a - a.mean(dim=1, keepdim=True))

    def q_conditioned(self, state: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Teacher-forced Q rows, conditioning each head on the TAKEN previous actions.

        actions: (B, 4) long tensor of grid indices in PARAM_ORDER.
        Returns {dim: (B, n_d) Q-values}.
        """
        h = self.trunk(state)
        out: dict[str, torch.Tensor] = {}
        ctx_parts: list[torch.Tensor] = []
        for i, d in enumerate(PARAM_ORDER):
            ctx = torch.cat(ctx_parts, dim=1) if ctx_parts else None
            out[d] = self._q_head(d, h, ctx)
            onehot = F.one_hot(actions[:, i], num_classes=self.dims[d]).float()
            ctx_parts.append(onehot)
        return out

    @torch.no_grad()
    def greedy_decode(self, state: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Sequential argmax decode. Returns (B,4) indices and per-head Q rows."""
        h = self.trunk(state)
        idxs: list[torch.Tensor] = []
        qrows: dict[str, torch.Tensor] = {}
        ctx_parts: list[torch.Tensor] = []
        for d in PARAM_ORDER:
            ctx = torch.cat(ctx_parts, dim=1) if ctx_parts else None
            q = self._q_head(d, h, ctx)
            a = q.argmax(dim=1)
            qrows[d] = q
            idxs.append(a)
            ctx_parts.append(F.one_hot(a, num_classes=self.dims[d]).float())
        return torch.stack(idxs, dim=1), qrows


# --------------------------------------------------------------------------- #
# Prioritized Experience Replay (identical mechanics to v2)
# --------------------------------------------------------------------------- #

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 10000, alpha: float = 0.6,
                 eps: float = 1e-6, seed: int | None = None):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.buffer: list[Transition | None] = [None] * capacity
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0
        self._rng = np.random.default_rng(seed)

    def push(self, transition: Transition) -> None:
        self.buffer[self.pos] = transition
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float):
        n = self.size
        if n == 0:
            raise RuntimeError("PER buffer is empty")
        prios = self.priorities[:n] ** self.alpha
        total = prios.sum()
        probs = prios / total if total > 0 else np.ones(n) / n
        idx = self._rng.choice(n, size=batch_size, p=probs)
        samples = [self.buffer[i] for i in idx]
        weights = (n * probs[idx]) ** (-beta)
        weights = weights / weights.max()
        return samples, idx, weights.astype(np.float32)

    def update_priorities(self, idx: np.ndarray, td_errors: np.ndarray) -> None:
        new_p = (np.abs(td_errors) + self.eps).astype(np.float64)
        self.priorities[idx] = new_p
        self.max_priority = max(self.max_priority, float(new_p.max()))

    def __len__(self) -> int:
        return self.size


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class AutoregressiveDQNAgentV3:
    """Autoregressive Dueling DQN + PER, grids injected and persisted."""

    def __init__(
        self,
        grids: dict[str, tuple] | None = None,
        state_dim: int = 4,
        hidden: int = 128,
        cond_hidden: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        replay_capacity: int = 10000,
        batch_size: int = 32,
        target_tau: float = 0.005,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        per_beta_anneal_steps: int = 900,
        device: str = "cpu",
        seed: int | None = None,
        dueling: bool = True,
    ):
        self.grids = {k: tuple(v) for k, v in (grids or DEFAULT_GRIDS).items()}
        self.dims = {k: len(v) for k, v in self.grids.items()}
        self.state_dim = state_dim
        self.hidden = hidden
        self.cond_hidden = cond_hidden
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_tau = target_tau
        self.per_beta_start = per_beta_start
        self.per_beta_end = per_beta_end
        self.per_beta_anneal_steps = per_beta_anneal_steps
        self.device = torch.device(device)

        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        self.dueling = dueling
        self.per_alpha = per_alpha
        self.q_main = AutoregressiveDuelingNet(state_dim, self.dims, hidden, cond_hidden,
                                               dueling=dueling).to(self.device)
        self.q_target = AutoregressiveDuelingNet(state_dim, self.dims, hidden, cond_hidden,
                                                 dueling=dueling).to(self.device)
        self.q_target.load_state_dict(self.q_main.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(self.q_main.parameters(), lr=lr)
        self.replay = PrioritizedReplayBuffer(capacity=replay_capacity, alpha=per_alpha, seed=seed)
        self.training_log: list[dict] = []
        self.step_count = 0

    # ------------------------------------------------------------------- #

    def decode(self, action: ActionV3) -> dict:
        return {
            "alpha": self.grids["alpha"][action.alpha_idx],
            "NBL":   self.grids["NBL"][action.NBL_idx],
            "L":     self.grids["L"][action.L_idx],
            "kappa": self.grids["kappa"][action.kappa_idx],
        }

    def _beta(self) -> float:
        progress = min(self.step_count / max(self.per_beta_anneal_steps, 1), 1.0)
        return self.per_beta_start + progress * (self.per_beta_end - self.per_beta_start)

    def select_action(self, state: np.ndarray, explore: bool = True,
                      mask_fn=None) -> ActionV3:
        """mask_fn (v3.1 budget constraint, optional):
        (dim, chosen_idx: dict[str,int]) -> bool array over grids[dim].
        None -> unconstrained v3 behaviour, bit-identical."""
        if explore and random.random() < self.epsilon:
            if mask_fn is None:
                return ActionV3(
                    alpha_idx=random.randrange(self.dims["alpha"]),
                    NBL_idx=random.randrange(self.dims["NBL"]),
                    L_idx=random.randrange(self.dims["L"]),
                    kappa_idx=random.randrange(self.dims["kappa"]),
                )
            chosen: dict[str, int] = {}
            for d in PARAM_ORDER:
                allowed = np.flatnonzero(mask_fn(d, chosen))
                chosen[d] = int(random.choice(allowed))
            return ActionV3(chosen["alpha"], chosen["NBL"], chosen["L"], chosen["kappa"])
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if mask_fn is None:
            idxs, _ = self.q_main.greedy_decode(s)
            ia, iN, iL, ik = (int(x) for x in idxs[0])
            return ActionV3(ia, iN, iL, ik)
        return self._masked_greedy(s, mask_fn)

    @torch.no_grad()
    def _masked_greedy(self, state_t: torch.Tensor, mask_fn) -> ActionV3:
        """Sequential argmax restricted to budget-feasible values (B=1)."""
        h = self.q_main.trunk(state_t)
        chosen: dict[str, int] = {}
        ctx_parts: list[torch.Tensor] = []
        for d in PARAM_ORDER:
            ctx = torch.cat(ctx_parts, dim=1) if ctx_parts else None
            q = self.q_main._q_head(d, h, ctx)[0]
            allowed = torch.tensor(np.asarray(mask_fn(d, chosen), dtype=bool),
                                   device=self.device)
            q = q.masked_fill(~allowed, float("-inf"))
            a = int(q.argmax())
            chosen[d] = a
            ctx_parts.append(F.one_hot(torch.tensor([a], device=self.device),
                                       num_classes=self.dims[d]).float())
        return ActionV3(chosen["alpha"], chosen["NBL"], chosen["L"], chosen["kappa"])

    def policy(self, state: np.ndarray, mask_fn=None) -> ActionV3:
        return self.select_action(state, explore=False, mask_fn=mask_fn)

    # ------------------------------------------------------------------- #

    def store_transition(self, state, action: ActionV3, reward, next_state, done):
        self.replay.push(Transition(state, action.as_tuple(), reward, next_state, done))

    def update(self) -> float | None:
        if len(self.replay) < self.batch_size:
            return None

        beta = self._beta()
        batch, idx, weights = self.replay.sample(self.batch_size, beta=beta)
        states = torch.tensor(np.array([b.state for b in batch]), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([b.next_state for b in batch]), dtype=torch.float32, device=self.device)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([b.done for b in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array([b.action for b in batch]), dtype=torch.long, device=self.device)
        w = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # Teacher-forced Q of the taken sub-actions
        q_rows = self.q_main.q_conditioned(states, actions)

        # Bootstrap term (zero in our 1-step setting, kept for genericity):
        # Double-DQN style — decode greedily with q_main, evaluate with q_target.
        with torch.no_grad():
            next_idx, _ = self.q_main.greedy_decode(next_states)
            q_next_rows = self.q_target.q_conditioned(next_states, next_idx)

        loss_total = torch.zeros(1, device=self.device)
        per_sample_td = torch.zeros(self.batch_size, device=self.device)

        for i, d in enumerate(PARAM_ORDER):
            q_pred = q_rows[d].gather(1, actions[:, i].unsqueeze(1)).squeeze(1)
            q_max = q_next_rows[d].gather(1, next_idx[:, i].unsqueeze(1)).squeeze(1)
            target = rewards + (1.0 - dones) * self.gamma * q_max
            td_err = target - q_pred
            per_sample_td = per_sample_td + td_err.detach().abs()
            elementwise = F.smooth_l1_loss(q_pred, target, reduction="none")
            loss_total = loss_total + (elementwise * w).mean()

        self.optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.q_main.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.replay.update_priorities(idx, (per_sample_td / 4.0).cpu().numpy())

        self.step_count += 1
        if self.step_count % 10 == 0:
            self._soft_update_target()
        return float(loss_total.item())

    def _soft_update_target(self) -> None:
        tau = self.target_tau
        for p_t, p_m in zip(self.q_target.parameters(), self.q_main.parameters()):
            p_t.data.mul_(1.0 - tau).add_(tau * p_m.data)

    def end_episode(self, reward: float, loss: float | None) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.training_log.append({
            "epsilon": self.epsilon, "reward": reward,
            "loss": loss, "beta": self._beta(),
        })

    # ------------------------------------------------------------------- #

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "grids":        self.grids,
            "state_dim":    self.state_dim,
            "hidden":       self.hidden,
            "cond_hidden":  self.cond_hidden,
            "dueling":      self.dueling,
            "per_alpha":    self.per_alpha,
            "q_main":       self.q_main.state_dict(),
            "q_target":     self.q_target.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "epsilon":      self.epsilon,
            "step_count":   self.step_count,
            "training_log": self.training_log,
        }, path)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "AutoregressiveDQNAgentV3":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(grids=ckpt["grids"], state_dim=ckpt["state_dim"],
                    hidden=ckpt["hidden"], cond_hidden=ckpt["cond_hidden"],
                    dueling=ckpt.get("dueling", True),
                    per_alpha=ckpt.get("per_alpha", 0.6), **kwargs)
        agent.q_main.load_state_dict(ckpt["q_main"])
        agent.q_target.load_state_dict(ckpt["q_target"])
        agent.optimizer.load_state_dict(ckpt["optimizer"])
        agent.epsilon = ckpt["epsilon"]
        agent.step_count = ckpt["step_count"]
        agent.training_log = ckpt["training_log"]
        return agent
