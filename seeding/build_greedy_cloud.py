"""
Option A (V4) — Générateur de seeds "weighted-greedy sweep" (zéro-LLM).

Rival classique des seeds LLM : N scalarisations -> N solutions greedy diverses
le long du front + mutants refill (épaisseur du nuage). Le greedy SIMPLE
(un seul balayage profit/poids) ne produit qu'un attracteur ; le greedy
PONDÉRÉ injecte la diversité par construction en balayant les vecteurs de
poids. Taille du nuage = 1510 solutions/instance (= la taille exacte des
nuages LLM, fairness).

Greedy MOKP (format Zitzler-Thiele, k sacs) :
    score(i | lam) = sum_k lam_k * profit_k(i) / sum_k (weight_k(i) / C_k)
    tri décroissant, insertion si TOUTES les contraintes de capacité tiennent.
Mutants : retirer ~10% des items choisis, re-remplir greedy (même lam) -> feasible.

Sortie : clouds/greedy_<inst>.txt
         (même format que llm_smart_cloud : lignes d'indices 0-based, en-têtes #)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "instances"
OUT_DIR = PROJECT_ROOT / "clouds"

INSTANCES = ["250.2", "250.3", "250.4", "500.2", "500.3", "500.4",
             "750.2", "750.3", "750.4"]
TARGET = 1510          # = taille des nuages LLM (fairness exacte)
MUTANTS_PER_BASE = 4   # par solution greedy de base
MUT_REMOVE_FRAC = 0.10
SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_instance(path: Path):
    """Format ZT strippé : 'k n' / [capacité / (label / weight / profit)*n] * k.

    Parsing POSITIONNEL (1 champ par ligne) — robuste aux labels corrompus du
    fichier source (' i74:', ' 291' sans deux-points dans 500.3, etc.).
    """
    lines = [ln.strip() for ln in path.read_text().split("\n") if ln.strip()]
    k, n = (int(x) for x in lines[0].split())
    assert len(lines) == 1 + k * (1 + 3 * n), \
        f"structure inattendue: {len(lines)} lignes vs {1 + k * (1 + 3 * n)}"
    pos = 1
    caps = np.zeros(k)
    weights = np.zeros((k, n))
    profits = np.zeros((k, n))
    for kk in range(k):
        caps[kk] = float(lines[pos]); pos += 1
        for i in range(n):
            pos += 1                                  # ligne label (ignorée)
            weights[kk, i] = float(lines[pos]); pos += 1
            profits[kk, i] = float(lines[pos]); pos += 1
    return k, n, caps, weights, profits


def greedy_fill(order, selected, weights, caps, load):
    """Remplit (in-place) selon `order` en respectant toutes les capacités."""
    for i in order:
        if selected[i]:
            continue
        new_load = load + weights[:, i]
        if np.all(new_load <= caps):
            selected[i] = True
            load = new_load
    return load


def greedy_solution(lam, weights, profits, caps, n):
    score = lam @ profits                       # (n,)
    wnorm = (weights / caps[:, None]).sum(axis=0)
    ratio = score / np.maximum(wnorm, 1e-12)
    order = np.argsort(-ratio)
    selected = np.zeros(n, dtype=bool)
    greedy_fill(order, selected, weights, caps, np.zeros(len(caps)))
    return selected, order


def lambda_set(k: int, n_base: int, rng) -> np.ndarray:
    """Coins + arêtes + grille/Dirichlet — diversité structurée sur le simplexe."""
    lams = [np.eye(k)[i] for i in range(k)]                      # extrêmes
    for i in range(k):                                           # arêtes (mi-chemin)
        for j in range(i + 1, k):
            v = np.zeros(k); v[i] = v[j] = 0.5
            lams.append(v)
    if k == 2:
        for t in np.linspace(0.02, 0.98, n_base - len(lams)):
            lams.append(np.array([t, 1.0 - t]))
    else:
        while len(lams) < n_base:
            lams.append(rng.dirichlet(np.ones(k)))
    return np.array(lams[:n_base])


def gen_instance(inst: str, rng) -> tuple[Path, int, int]:
    k, n, caps, weights, profits = parse_instance(DATA / f"{inst}.txt")
    n_base = int(np.ceil(TARGET / (1 + MUTANTS_PER_BASE)))
    lams = lambda_set(k, n_base, rng)

    sols: list[np.ndarray] = []
    for lam in lams:
        sel, order = greedy_solution(lam, weights, profits, caps, n)
        sols.append(sel.copy())
        chosen = np.flatnonzero(sel)
        for _ in range(MUTANTS_PER_BASE):       # mutants : remove 10% + refill
            mut = sel.copy()
            n_rm = max(1, int(len(chosen) * MUT_REMOVE_FRAC))
            mut[rng.choice(chosen, size=n_rm, replace=False)] = False
            load = (weights * mut).sum(axis=1)
            greedy_fill(order, mut, weights, caps, load)
            sols.append(mut)
    sols = sols[:TARGET]

    # validation : faisabilité de TOUTES les solutions sur TOUS les sacs
    for s in sols:
        assert np.all((weights * s).sum(axis=1) <= caps + 1e-9), "infeasible!"
    n_distinct = len({s.tobytes() for s in sols})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"greedy_{inst}.txt"
    with open(out, "w") as fp:
        fp.write("# Weighted-greedy sweep cloud (zero-LLM rival, V4 Option A)\n")
        fp.write(f"# Total: {len(sols)}\n")
        fp.write(f"# Base lams: {len(lams)} (corners+edges+sweep), "
                 f"mutants/base: {MUTANTS_PER_BASE}, seed={SEED}\n")
        for idx, s in enumerate(sols, 1):
            fp.write(f"# Sol {idx}\n")
            fp.write(" ".join(str(i) for i in np.flatnonzero(s)) + "\n")
    return out, len(sols), n_distinct


def main() -> None:
    rng = np.random.default_rng(SEED)
    for inst in INSTANCES:
        t0 = time.time()
        out, n_sols, n_distinct = gen_instance(inst, rng)
        log(f"{inst}: {n_sols} sols ({n_distinct} distinctes) en "
            f"{time.time()-t0:.1f}s -> {out.name}")
    log("done — setup zéro-LLM : quelques secondes/instance (vs ~24 min LLM)")


if __name__ == "__main__":
    main()
