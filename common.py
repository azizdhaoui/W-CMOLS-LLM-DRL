"""Fonctions partagées : instances, graines gloutonnes, voisins (îles), solveur et hypervolume.

Test isolé : ne modifie aucun fichier canonique. Les îles reprennent à
l'identique generate_smart_neighbors.py ; le glouton reprend la formule du
cloud G (gen_weighted_greedy_seeds.py), pour que L et G ne diffèrent que par
l'étape de décision.
"""
from __future__ import annotations

import itertools
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = ROOT = Path(__file__).resolve().parent      # racine du dépôt
DATA = ROOT / "data" / "instances"
INSTANCES = ["250.2", "250.3", "250.4", "500.2", "500.3", "500.4",
             "750.2", "750.3", "750.4"]
WEIGHTS = {2: "Weights_2obj_FQ200.txt", 3: "Weights_3obj_FQ100.txt",
           4: "Weights_4obj_FQ40.txt"}
N_SEEDS = 10
NEIGHBORS = 150


def load(inst: str):
    lines = [l.strip() for l in (DATA / f"{inst}.txt").read_text().split("\n") if l.strip()]
    k, n = (int(x) for x in lines[0].split())
    pos, caps = 1, np.zeros(k)
    w, p = np.zeros((k, n), dtype=np.int64), np.zeros((k, n), dtype=np.int64)
    for kk in range(k):
        caps[kk] = float(lines[pos]); pos += 1
        for i in range(n):
            pos += 1
            w[kk, i] = int(float(lines[pos])); pos += 1
            p[kk, i] = int(float(lines[pos])); pos += 1
    return k, n, caps, w, p


def lambdas(k: int) -> list[list[float]]:
    if k == 2:
        return [[round(1 - t / 9, 6), round(t / 9, 6)] for t in range(10)]
    eye = [list(map(float, r)) for r in np.eye(k)]
    edges = []
    for i, j in itertools.combinations(range(k), 2):
        v = [0.0] * k; v[i] = v[j] = 0.5; edges.append(v)
    if k == 3:
        tilt = []
        for i in range(3):
            v = [1 / 6] * 3; v[i] = 2 / 3; tilt.append(v)
        return eye + edges + [[1 / 3] * 3] + tilt
    return eye + edges                                   # k == 4 : 4 + 6


def lattice(k: int, h: int) -> list[list[float]]:
    """Réseau simplexe de Das-Dennis : coins, arêtes et points intérieurs."""
    out = []
    for c in itertools.combinations(range(h + k - 1), k - 1):
        parts, prev = [], -1
        for x in c:
            parts.append(x - prev - 1); prev = x
        parts.append(h + k - 2 - prev)
        out.append([v / h for v in parts])
    return out


def lambdas_n(k: int, n_seeds: int) -> list[list[float]]:
    if k == 2:
        return [[round(1 - t / (n_seeds - 1), 6), round(t / (n_seeds - 1), 6)] for t in range(n_seeds)]
    h = 1
    while len(lattice(k, h + 1)) <= n_seeds:
        h += 1
    # coins exacts ; les autres points tirés de 20 % vers le centre, pour
    # qu'aucun objectif ne reçoive un poids nul hors des coins
    return [pt if max(pt) == 1 else [0.8 * x + 0.2 / k for x in pt] for pt in lattice(k, h)]


def ratio(lam, w, p, caps):
    return (np.asarray(lam) @ p) / np.maximum((w / caps[:, None]).sum(0), 1e-12)


def greedy(lam, w, p, caps) -> np.ndarray:
    r = ratio(lam, w, p, caps)
    sel, load_ = np.zeros(w.shape[1], bool), np.zeros(len(caps))
    for i in np.argsort(-r):
        if np.all(load_ + w[:, i] <= caps):
            sel[i] = True; load_ += w[:, i]
    return sel


def objs(sel, p):
    return [int(x) for x in (p * sel).sum(1)]


def feasible(sel, w, caps):
    return bool(np.all((w * sel).sum(1) <= caps))


def repair_fill(sel, lam, w, p, caps) -> np.ndarray:
    """Réparation et complétion selon le λ de la graine (pas de ratio non pondéré)."""
    sel = sel.copy(); r = ratio(lam, w, p, caps)
    while not feasible(sel, w, caps):
        inside = np.flatnonzero(sel)
        sel[inside[np.argmin(r[inside])]] = False
    load_ = (w * sel).sum(1)
    for i in np.argsort(-r):
        if not sel[i] and np.all(load_ + w[:, i] <= caps):
            sel[i] = True; load_ += w[:, i]
    return sel


def dominates(a, b):
    return all(x >= y for x, y in zip(a, b)) and a != b


# --- îles : copie fidèle de generate_smart_neighbors.py -----------------------
def island_weight_vectors(nf, per_seed=NEIGHBORS):
    islands = []
    if nf == 2:
        islands.append(([0.85, 0.15], per_seed // 3))
        islands.append(([0.15, 0.85], per_seed // 3))
        islands.append(([0.50, 0.50], per_seed - 2 * (per_seed // 3)))
    elif nf == 3:
        ps = per_seed // 5
        for o in range(nf):
            v = [0.10] * nf; v[o] = 0.80; islands.append((v, ps))
        pairs = list(itertools.combinations(range(nf), 2)); pp = per_seed // 10
        for a, b in pairs:
            v = [0.05] * nf; v[a] = v[b] = 0.475; islands.append((v, pp))
        rem = per_seed - ps * nf - pp * len(pairs)
        if rem > 0:
            islands.append(([1.0 / nf] * nf, rem))
    else:
        ps = per_seed // 7
        for o in range(nf):
            v = [0.05] * nf; v[o] = 1.0 - 0.05 * (nf - 1); islands.append((v, ps))
        pairs = list(itertools.combinations(range(nf), 2))[:3]; pp = per_seed // 15
        for a, b in pairs:
            v = [0.05] * nf; v[a] = v[b] = 0.45; islands.append((v, pp))
        rem = per_seed - ps * nf - pp * len(pairs)
        if rem > 0:
            islands.append(([1.0 / nf] * nf, rem))
    return islands


def expand(seeds: list[list[int]], inst: str, per_seed: int = NEIGHBORS) -> list[list[int]]:
    k, n, caps, w, p = load(inst)
    wl, pl = w.tolist(), p.tolist()
    islands = island_weight_vectors(k, per_seed)
    random.seed(42)
    out = []
    for seed in seeds:
        out.append(sorted(seed))
        for iwv, cnt in islands:
            ir = [(i, sum(iwv[f] * pl[f][i] for f in range(k)) /
                   sum(wl[f][i] / caps[f] for f in range(k))) for i in range(n)]
            for _ in range(cnt):
                nb = set(seed)
                if n >= 450 and k == 2:
                    nr, jl, jh = random.randint(20, 50), 0.60, 1.40
                elif n < 350:
                    nr, jl, jh = random.randint(2, 6), 0.92, 1.08
                elif n < 600:
                    nr, jl, jh = random.randint(5, 12), 0.85, 1.15
                else:
                    nr, jl, jh = random.randint(10, 25), 0.70, 1.30
                nr = min(nr, len(nb) - 1)
                for it in random.sample(list(nb), nr):
                    nb.remove(it)
                jit = sorted(((i, r * random.uniform(jl, jh)) for i, r in ir),
                             key=lambda x: x[1], reverse=True)
                rem = list(caps)
                for it in nb:
                    for f in range(k):
                        rem[f] -= wl[f][it]
                for it, _ in jit:
                    if it in nb:
                        continue
                    if all(rem[f] >= wl[f][it] for f in range(k)):
                        nb.add(it)
                        for f in range(k):
                            rem[f] -= wl[f][it]
                out.append(sorted(nb))
    return out


def write_cloud(sols, path: Path, head: str):
    lines = [f"# {head}", f"# Total: {len(sols)}"]
    for i, s in enumerate(sols, 1):
        lines += [f"# Sol {i}", " ".join(map(str, s))]
    path.write_text("\n".join(lines) + "\n")


# --- solveur + HV : mêmes réglages que le cloud G canonique -----------------
def solve(cloud, inst: str, runs: int, workdir: Path, nbl: int = 100, reinject: bool = True,
          params: dict | None = None, save_as=None):
    """cloud=None -> baseline (archive vide). reinject=False -> solveur sans réinjection."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(HERE / "solver"))
    import importlib
    solver = importlib.import_module("moacp_mut" if reinject else "moacp_noreinj")
    from rl_agent.hv_calculator import InstanceBounds, compute_hv, parse_all_runs
    k = int(inst.split(".")[1])
    pr = {"alpha": 10, "L": 5, "kappa": 0.05, "NBL": nbl, **(params or {})}
    workdir.mkdir(parents=True, exist_ok=True)
    prev = os.getcwd(); os.chdir(workdir)
    try:
        solver.set_runtime_params(alpha_val=pr["alpha"], L_val=pr["L"], kappa_val=pr["kappa"])
        solver.set_iterations(pr["NBL"])
        t0 = time.time()
        solver.run_moacp_ex(str(DATA / f"{inst}.txt").encode(),
                            str(DATA / WEIGHTS[k]).encode(),
                            (str(cloud).encode() if cloud else b""), runs, -1)
        dt = (time.time() - t0) / runs
        fronts = parse_all_runs(workdir / "all_runs.txt", workdir / "pareto_sizes.txt")
        if save_as is not None:                      # fronts bruts, même format que les artefacts canoniques
            import shutil
            Path(save_as).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(workdir / "all_runs.txt", f"{save_as}_raw.txt")
            shutil.copy2(workdir / "pareto_sizes.txt", f"{save_as}_sizes.txt")
    finally:
        os.chdir(prev)
    c = json.loads((ROOT / "data/reference" / f"{inst}.json").read_text())
    b = InstanceBounds(low=np.array(c["bounds_low"]), high=np.array(c["bounds_high"]))
    return [float(compute_hv(f, b)) for f in fronts], dt
