"""Données et outils du chapitre 6 : instances, bornes, fronts, indicateur epsilon additif.

Les fronts bruts (nos systèmes : results/fronts/ ; concurrents : competitor_fronts/)
ne sont pas versionnés. Les valeurs calculées à partir d'eux sont dans results/CH6_V8.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import COMPETITOR_FRONTS as CF, FRONTS, REFERENCE  # noqa: E402

INSTANCES = ["250.2", "250.3", "250.4", "500.2", "500.3", "500.4", "750.2", "750.3", "750.4"]
N_RUNS = {"ours": 20, "competitor": 20, "gwaco": 30}
REF_MAX = 2000      # points conservés dans l'ensemble de référence
RUN_MAX = 1500      # points conservés par exécution

# fronts de W-CMOLS (archive initiale vide)
W_FRONTS = (FRONTS / "raw_baseline_{i}.txt", FRONTS / "sizes_baseline_{i}.txt")

# fronts des quatre variantes de l'algorithme de colonie de fourmis du laboratoire,
# employés seulement dans l'ensemble de référence de l'indicateur epsilon
GWACO_KEYS = ["gwaco_init", "gwaco_maj", "gwaco_both", "gwaco_heur"]
GW_SRC = CF / "gwaco"
GW_FOLDER = {"gwaco_heur": "resolution heuristique",
             "gwaco_init": "resolution initialisation",
             "gwaco_maj": "resolution mise à jour",
             "gwaco_both": "resolution ini + m à j"}
GW_FILES = {"250.3": "250_3_20fourmi_FQ100.txt", "250.4": "250_4_20fourmi_FQ40.txt",
            "500.3": "500_3_20fourmi_FQ100.txt", "500.4": "500_4_20fourmi_FQ40.txt",
            "750.3": "750_3_20fourmi_FQ100.txt", "750.4": "750_4_20fourmi_FQ40.txt"}


def bounds(inst: str):
    c = json.loads((REFERENCE / f"{inst}.json").read_text())
    return np.array(c["bounds_low"]), np.array(c["bounds_high"])


# treize concurrents : (clé, nom, front brut, cardinalités par exécution)
CH6 = [
    ("moead", "MOEA/D", CF / "moead_fronts/raw_moead_{i}.txt", CF / "moead_fronts/sizes_moead_{i}.txt"),
    ("eagmoead", "EAG-MOEA/D", CF / "mokpr_fronts/raw_eagmoead_{i}.txt", CF / "mokpr_fronts/sizes_eagmoead_{i}.txt"),
    ("cmoead", "C-MOEA-D", CF / "extra_fronts/raw_cmoead_{i}.txt", CF / "extra_fronts/sizes_cmoead_{i}.txt"),
    ("moead2wa", "MOEA-D-2WA", CF / "extra_fronts/raw_moead2wa_{i}.txt", CF / "extra_fronts/sizes_moead2wa_{i}.txt"),
    ("gwasfga", "GWASF-GA", CF / "mokpr_fronts/raw_gwasfga_{i}.txt", CF / "mokpr_fronts/sizes_gwasfga_{i}.txt"),
    ("moeadgr", "MOEA/D-GR", CF / "moeadgr_fronts/raw_moeadgr_{i}.txt", CF / "moeadgr_fronts/sizes_moeadgr_{i}.txt"),
    ("ibea", "IBEA", CF / "pisa_fronts/raw_ibea_{i}.txt", CF / "pisa_fronts/sizes_ibea_{i}.txt"),
    ("spea2", "SPEA2", CF / "pisa_fronts/raw_spea2_{i}.txt", CF / "pisa_fronts/sizes_spea2_{i}.txt"),
    ("nsga2", "NSGA-II", CF / "pisa_fronts/raw_nsga2_{i}.txt", CF / "pisa_fronts/sizes_nsga2_{i}.txt"),
    ("mobkp", "mobkp/PLS", CF / "mobkp_fronts/raw_mobkp_{i}.txt", CF / "mobkp_fronts/sizes_mobkp_{i}.txt"),
    ("dcnsga3", "DCNSGA-III", CF / "extra_fronts/raw_dcnsga3_{i}.txt", CF / "extra_fronts/sizes_dcnsga3_{i}.txt"),
    ("cmodrl", "CMODRL", CF / "rl_fronts/raw_cmodrl_{i}.txt", CF / "rl_fronts/sizes_cmodrl_{i}.txt"),
    ("idbea", "I-DBEA", CF / "extra_fronts/raw_idbea_{i}.txt", CF / "extra_fronts/sizes_idbea_{i}.txt"),
]


def nd(pts: np.ndarray) -> np.ndarray:
    """Filtre non domine, en maximisation."""
    keep = np.ones(len(pts), bool)
    for j in range(len(pts)):
        if not keep[j]:
            continue
        keep &= ~(np.all(pts <= pts[j], 1) & np.any(pts < pts[j], 1))
        keep[j] = True
    return pts[keep]


def thin(pts: np.ndarray, cap: int) -> np.ndarray:
    """Sous-echantillonnage deterministe preservant les extremes."""
    if len(pts) <= cap:
        return pts
    idx = set(np.argmax(pts, axis=0).tolist()) | set(np.argmin(pts, axis=0).tolist())
    rest = [i for i in range(len(pts)) if i not in idx]
    take = np.linspace(0, len(rest) - 1, cap - len(idx)).astype(int)
    return pts[sorted(idx | {rest[t] for t in take})]


def eps_plus(A: np.ndarray, R: np.ndarray, chunk: int = 256) -> float:
    """I_eps+(A, R) en maximisation. Identique a calc_ind_value de eps_ind.c.

    La boucle triple du C est vectorisee par blocs de references : pour chaque
    r, on calcule max_k (r_k - a_k) sur tous les a, on en prend le minimum, et
    le resultat global est le maximum sur r.
    """
    best = -np.inf
    for s in range(0, len(R), chunk):
        r = R[s:s + chunk]                       # (c, dim)
        d = r[:, None, :] - A[None, :, :]        # (c, |A|, dim)
        best = max(best, float(d.max(axis=2).min(axis=1).max()))
    return best


def runs_of(raw_tpl, sizes_tpl, inst: str, cap: int) -> list[np.ndarray]:
    """Fronts execution par execution, decoupes par le fichier de cardinalites."""
    raw, sizes = Path(str(raw_tpl).format(i=inst)), Path(str(sizes_tpl).format(i=inst))
    if not raw.exists() or not sizes.exists():
        return []
    sz = [int(x) for x in sizes.read_text().split() if x.strip()][:cap]
    if not sz:
        return []
    pts = np.loadtxt(raw, max_rows=sum(sz))
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    out, cur = [], 0
    for n in sz:
        f = pts[cur:cur + n]
        cur += n
        if len(f):
            out.append(np.unique(f, axis=0))
    return out


def gwaco_runs(key: str, inst: str, cap: int) -> list[np.ndarray]:
    import re
    if inst not in GW_FILES:
        return []
    base = GW_SRC / GW_FOLDER[key]
    info = (base / "info" / GW_FILES[inst]).read_text(errors="ignore")
    card = [int(x) for x in
            re.findall(r"cardinalit\S*\s+ensemble\s+Pareto\s+(\d+)", info)][:cap]
    if not card:
        return []
    pts = np.loadtxt(base / "resolution" / GW_FILES[inst], max_rows=sum(card))
    out, cur = [], 0
    for n in card:
        f = pts[cur:cur + n]
        cur += n
        if len(f):
            out.append(np.unique(f, axis=0))
    return out
