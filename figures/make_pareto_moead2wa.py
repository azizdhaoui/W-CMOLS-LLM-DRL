# -*- coding: utf-8 -*-
"""Pareto fronts: the thesis's three systems (baseline, S1-Union, Hybride-Union)
against MOEA-D-2WA (Jiao, Zeng, Li & Ong, Information Sciences 2021) on all 9
instances.

WHY THIS FIGURE SERIES EXISTS. MOEA-D-2WA is only the SECOND external algorithm
ever measured here to beat the thesis baseline (7/9 instances) -- the other being
the real MOEA/D of Li & Zhang. Both are decomposition-based, while every generic
MOEA, every RL-based method and the dedicated local search fall below the
baseline. Real MOEA/D is therefore plotted alongside, so the two baseline-beaters
can be read against each other and against our systems in one picture.

Five series, all four thesis-relevant systems plus the two competitors:
    baseline · S1-Union · Hybride-Union · MOEA/D (2007) · MOEA-D-2WA (2021)

S1-Union and Hybride-Union are BOTH kept here (unlike the chapter-5 competitor
figures, which drop one of them as visually redundant): the point of this series
is precisely to show that the two contributions behave alike against a genuinely
competitive rival, so collapsing them would remove the comparison being made.

20 runs for every system (RUN_CAP=20) for a like-for-like point count, even
though baseline/S1/Hyb use 50 runs in their own chapters' tables.

Output -> figures/out/fig_m2wa_<inst>.png
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
HV = V3 / "results" / "fronts"
OUT = V3 / "figures" / "out"
# Version finale (septembre 2026) : fronts U et UQ des clouds finaux, solveur sans réinjection.
FINAL = V3 / "results" / "fronts"
OUT.mkdir(parents=True, exist_ok=True)

INSTANCES = ["250.2", "250.3", "250.4", "500.2", "500.3", "500.4", "750.2", "750.3", "750.4"]
OBJ = {"250.2": 2, "250.3": 3, "250.4": 4, "500.2": 2, "500.3": 3, "500.4": 4,
       "750.2": 2, "750.3": 3, "750.4": 4}
RUN_CAP = 20

SYSTEMS = ["baseline", "s1_union", "hyb_union", "moead", "moead2wa"]
LABEL = {"baseline": "W-CMOLS (init. aléatoire)", "s1_union": "SW-CMOLS-U",
         "hyb_union": "SW-CMOLS-UQ", "moead": "MOEA/D (Zhang et Li, 2007)",
         "moead2wa": "MOEA-D-2WA (Jiao et al., 2021)"}
# CVD-validated with scripts/validate_palette.js (--mode light --pairs all):
# ALL CHECKS PASS. Worst CVD pair is MOEA-D-2WA/purple vs MOEA/D/blue at deutan
# dE 6.9 -- inside the 6-8 floor band, legal only with secondary encoding, which
# is supplied by a distinct marker shape per system (MARKER below). Normal-vision
# floor passes every pair (worst 15.6). The one contrast WARN (#e69f00) is
# discharged by the always-present legend and the numeric tables.
COLOR = {"baseline": "#c62828", "s1_union": "#e69f00", "hyb_union": "#009e73",
         "moead": "#0072b2", "moead2wa": "#6a3d9a"}
MARKER = {"baseline": "s", "s1_union": "*", "hyb_union": "^",
          "moead": "P", "moead2wa": "o"}
SIZE = {"baseline": 16, "s1_union": 30, "hyb_union": 22, "moead": 22, "moead2wa": 20}
ZORDER = {"baseline": 3, "s1_union": 6, "hyb_union": 5, "moead": 4, "moead2wa": 4}

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
        "s1_union": load_capped(FINAL / f"U2_nbl100_{inst}_raw.txt", FINAL / f"U2_nbl100_{inst}_sizes.txt", RUN_CAP),
        "hyb_union": load_capped(FINAL / f"UQ_dqn_{inst}_raw.txt", FINAL / f"UQ_dqn_{inst}_sizes.txt", RUN_CAP),
        "moead": load_capped(V3 / f"competitor_fronts/moead_fronts/raw_moead_{inst}.txt",
                              V3 / f"competitor_fronts/moead_fronts/sizes_moead_{inst}.txt", RUN_CAP),
        "moead2wa": load_capped(PLATEMO / f"extra_fronts/raw_moead2wa_{inst}.txt",
                                 PLATEMO / f"extra_fronts/sizes_moead2wa_{inst}.txt", RUN_CAP),
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
    p = OUT / f"fig_m2wa_{inst}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p.name}")


def fig_3d(inst, data):
    fig = plt.figure(figsize=(5.0, 4.1))
    ax = fig.add_subplot(111, projection="3d")

    def sub(a, n):
        if len(a) <= n:
            return a
        return a[RNG.choice(len(a), n, replace=False)]

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
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.92)
    fig.tight_layout()
    p = OUT / f"fig_m2wa_{inst}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p.name}")


def fig_scattermatrix(inst, data):
    m = OBJ[inst]
    pairs = list(combinations(range(m), 2))

    def sub(a, n):
        if len(a) <= n:
            return a
        return a[RNG.choice(len(a), n, replace=False)]

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
    fig.legend(handles=handles, fontsize=9.5, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.92)
    fig.suptitle(f"Instance {inst} ({m} objectifs, 20 répétitions) — "
                 f"matrice de nuages de points (paires d'objectifs)",
                 fontsize=13, weight="bold", y=1.01)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    p = OUT / f"fig_m2wa_{inst}.png"
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
