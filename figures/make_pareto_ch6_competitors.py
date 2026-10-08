# -*- coding: utf-8 -*-
"""Figures 6.1 a 6.9 du memoire : W-CMOLS, SW-CMOLS-UQ et quatre concurrents
externes, sur les neuf instances.

Derive de make_pareto_ch5_competitors.py (meme rendu, meme palette validee).
Les quatre concurrents traces sont :
  * MOEA/D et EAG-MOEA/D, les deux premiers du tableau 6.1 ;
  * GWASF-GA, cinquieme, seul representant de la scalarisation par fonction
    d'accomplissement ;
  * mobkp/PLS, seul concurrent dedie au sac a dos multi-objectif.
MOEA-D-2WA n'y figure plus : il est trace face a nos systemes dans les
figures 6.10 a 6.18. Les autres concurrents restent dans les tableaux.

La palette est EXACTEMENT celle du script d'origine (six couleurs validees
contre la deficience de vision des couleurs) ; seules les series changent.

EAG-MOEA/D et GWASF-GA : formulation a reparation native de PlatEMO
(section 6.2), fronts exportes par comparison/competitors/platemo/export_mokpr_campagnes.py.

Sortie -> figures/out/fig6_<inst>.png
"""
from __future__ import annotations
from pathlib import Path
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

V3 = Path(__file__).resolve().parents[1]
PLATEMO = V3 / "competitor_fronts"
MOKPR = PLATEMO / "mokpr_fronts"
HV = V3 / "results" / "fronts"
OUT = V3 / "figures" / "out"
# Version finale (septembre 2026) : fronts U et UQ des clouds finaux, solveur sans réinjection.
FINAL = V3 / "results" / "fronts"
OUT.mkdir(parents=True, exist_ok=True)

INSTANCES = ["250.2", "250.3", "250.4", "500.2", "500.3", "500.4", "750.2", "750.3", "750.4"]
OBJ = {"250.2": 2, "250.3": 3, "250.4": 4, "500.2": 2, "500.3": 3, "500.4": 4,
       "750.2": 2, "750.3": 3, "750.4": 4}
RUN_CAP = 20  # matches every external competitor's native run count

SYSTEMS = ["baseline", "hyb_union", "moead", "eagmoead", "gwasfga", "mobkp"]
LABEL = {"baseline": "W-CMOLS (init. aléatoire)", "hyb_union": "SW-CMOLS-UQ",
         "moead": "MOEA/D (2007)", "eagmoead": "EAG-MOEA/D (2015)",
         "gwasfga": "GWASF-GA (2017)", "mobkp": "mobkp / PLS (2022)"}
# CVD-validated with scripts/validate_palette.js (--mode light --pairs all):
# no FAIL. Worst CVD pair is mobkp/purple vs MOEA/D/blue at deutan dE 6.9,
# inside the 6-8 floor band, which is legal ONLY with secondary encoding --
# supplied here by a distinct marker shape per system (MARKER below). Normal-
# vision floor passes for every pair (worst 15.6). Two contrast-vs-surface
# WARNs (#e69f00, #56b4e9) are discharged by the always-present legend and by
# the numeric tables 6.1/6.2 carrying the same values.
# This 6-series set is the maximum that validates; see the module docstring for
# which systems were dropped from the figures to reach it.
COLOR = {"baseline": "#c62828", "hyb_union": "#e69f00", "moead": "#0072b2",
         "eagmoead": "#009e73", "mobkp": "#6a3d9a", "gwasfga": "#56b4e9"}
MARKER = {"baseline": "s", "hyb_union": "*", "moead": "P",
          "eagmoead": "^", "mobkp": "o", "gwasfga": "D"}
SIZE = {"baseline": 16, "hyb_union": 30, "moead": 22,
        "eagmoead": 22, "mobkp": 20, "gwasfga": 20}
ZORDER = {"baseline": 3, "hyb_union": 6, "moead": 5,
          "eagmoead": 5, "mobkp": 4, "gwasfga": 4}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.edgecolor": "#cccccc", "axes.linewidth": 0.9, "figure.dpi": 150})
RNG = np.random.default_rng(0)


def nd_filter(pts):
    keep = np.ones(len(pts), bool)
    for i in range(len(pts)):
        if not keep[i]:
            continue
        dominated = np.all(pts <= pts[i], 1) & np.any(pts < pts[i], 1)
        keep &= ~dominated
        keep[i] = True
    return pts[keep]


def load_capped(raw_path, sizes_path, cap_runs, cap_pts=3000):
    sizes = [int(x) for x in open(sizes_path) if x.strip()]
    cutoff = sum(sizes[:cap_runs])
    pts = np.loadtxt(raw_path, max_rows=cutoff)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    pts = np.unique(pts, axis=0)
    if len(pts) > cap_pts:
        pts = pts[np.linspace(0, len(pts) - 1, cap_pts).astype(int)]
    return nd_filter(pts)


def load_all(inst):
    return {
        "baseline": load_capped(HV / f"raw_pure_{inst}.txt", HV / f"sizes_pure_{inst}.txt", RUN_CAP),
        "hyb_union": load_capped(FINAL / f"UQ_dqn_{inst}_raw.txt", FINAL / f"UQ_dqn_{inst}_sizes.txt", RUN_CAP),
        "moead": load_capped(V3 / f"competitor_fronts/moead_fronts/raw_moead_{inst}.txt",
                              V3 / f"competitor_fronts/moead_fronts/sizes_moead_{inst}.txt", RUN_CAP),
        "eagmoead": load_capped(MOKPR / f"raw_eagmoead_{inst}.txt",
                                 MOKPR / f"sizes_eagmoead_{inst}.txt", RUN_CAP),
        "mobkp": load_capped(V3 / f"competitor_fronts/mobkp_fronts/raw_mobkp_{inst}.txt",
                              V3 / f"competitor_fronts/mobkp_fronts/sizes_mobkp_{inst}.txt", RUN_CAP),
        "gwasfga": load_capped(MOKPR / f"raw_gwasfga_{inst}.txt",
                                MOKPR / f"sizes_gwasfga_{inst}.txt", RUN_CAP),
    }


def fig_2d(inst, data):
    fig, ax = plt.subplots(figsize=(6.3, 4.9))
    for k in SYSTEMS:
        d = data[k]
        ax.scatter(d[:, 0], d[:, 1], s=SIZE[k], marker=MARKER[k], c=COLOR[k],
                   alpha=0.85, zorder=ZORDER[k], label=f"{LABEL[k]} (n={len(d)})")
    ax.set_xlabel("Profit objectif 1", fontsize=11)
    ax.set_ylabel("Profit objectif 2", fontsize=11)
    ax.set_title(f"Instance {inst} (2 objectifs, 20 répétitions) — plus haut/droite = mieux",
                 fontsize=12.5, weight="bold", pad=10)
    ax.grid(True, alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.92)
    fig.tight_layout()
    p = OUT / f"fig6_{inst}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p.name}")


def fig_3d(inst, data):
    fig = plt.figure(figsize=(5.0, 4.1))
    ax = fig.add_subplot(111, projection="3d")

    def sub(a, n):
        if len(a) <= n:
            return a
        idx = RNG.choice(len(a), n, replace=False)
        return a[idx]

    for k in SYSTEMS:
        d = sub(data[k], 450)
        ax.scatter(d[:, 0], d[:, 1], d[:, 2], s=SIZE[k] * 0.9, marker=MARKER[k], c=COLOR[k],
                   alpha=0.8, depthshade=False, label=f"{LABEL[k]} (n={len(data[k])})")
    ax.set_xlabel("Objectif 1", fontsize=10, labelpad=8)
    ax.set_ylabel("Objectif 2", fontsize=10, labelpad=8)
    ax.set_zlabel("Objectif 3", fontsize=10, labelpad=8)
    ax.set_title(f"Instance {inst} (3 objectifs, 20 répétitions) — points 3D réels",
                 fontsize=12.5, weight="bold", pad=6)
    ax.view_init(elev=22, azim=-60)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    # legende au niveau de la figure : en 3D, les nuages de points sont dessines
    # par-dessus une legende d'axes et en masquaient la derniere ligne.
    fig.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0.01, 0.92), framealpha=0.95)
    p = OUT / f"fig6_{inst}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p.name}")


def fig_scattermatrix(inst, data):
    m = OBJ[inst]
    pairs = list(combinations(range(m), 2))

    def sub(a, n):
        if len(a) <= n:
            return a
        idx = RNG.choice(len(a), n, replace=False)
        return a[idx]

    subd = {k: sub(data[k], 500) for k in SYSTEMS}
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.1))
    for ax, (i, j) in zip(axes.flat, pairs):
        for k in SYSTEMS:
            d = subd[k]
            ax.scatter(d[:, i], d[:, j], s=SIZE[k] * 0.75, marker=MARKER[k], c=COLOR[k],
                       alpha=0.75, zorder=ZORDER[k])
        ax.set_xlabel(f"Objectif {i+1}", fontsize=10)
        ax.set_ylabel(f"Objectif {j+1}", fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8.5)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker=MARKER[k], color="none", markerfacecolor=COLOR[k],
                       markersize=9, label=f"{LABEL[k]} (n={len(data[k])})") for k in SYSTEMS]
    fig.legend(handles=handles, fontsize=9.5, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.92)
    fig.suptitle(f"Instance {inst} ({m} objectifs, 20 répétitions) — "
                 f"matrice de nuages de points (paires d'objectifs)",
                 fontsize=13, weight="bold", y=1.01)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    p = OUT / f"fig6_{inst}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p.name}")


if __name__ == "__main__":
    for inst in INSTANCES:
        data = load_all(inst)
        for k, d in data.items():
            print(f"  {inst} {k}: {len(d)} non-dominated points")
        if OBJ[inst] == 2:
            fig_2d(inst, data)
        elif OBJ[inst] == 3:
            fig_3d(inst, data)
        else:
            fig_scattermatrix(inst, data)
    print(f"\ndone -> {OUT}")
