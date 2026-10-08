# -*- coding: utf-8 -*-
"""Fronts de Pareto des chapitres 4 et 5 (clouds G, L, U ; variantes GQ, LQ, UQ).

usage : python plot_fronts.py ch4   -> figures/fig4_<inst>.png : W, G, L, U
        python plot_fronts.py ch5   -> figures/fig5_<inst>.png : W, U, GQ, LQ, UQ
Fronts : results/fronts/ (50 répétitions) ; W-CMOLS : fronts canoniques (deck_data).
Fronts superposés, une figure par instance.
"""
from __future__ import annotations
import sys
from pathlib import Path
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "comparison")]
import competitor_data as D  # noqa: E402
FR = ROOT / "results" / "fronts"
OUT = ROOT / "figures" / "out"
OUT.mkdir(parents=True, exist_ok=True)
CH = sys.argv[1] if len(sys.argv) > 1 else "ch4"
PREFIX = "fig4_" if CH == "ch4" else "fig5_"

INSTANCES = ["250.2", "250.3", "250.4", "500.2", "500.3", "500.4", "750.2", "750.3", "750.4"]
OBJ = {i: int(i.split(".")[1]) for i in INSTANCES}
RUN_CAP = 50

SYSTEMS = ["W", "G", "L", "U"] if CH == "ch4" else ["W", "U", "GQ", "LQ", "UQ"]
LABEL = {"W": "W-CMOLS", "G": "SW-CMOLS-G", "L": "SW-CMOLS-L", "U": "SW-CMOLS-U" if CH == "ch4" else "SW-CMOLS-U (statique)",
         "GQ": "SW-CMOLS-GQ", "LQ": "SW-CMOLS-LQ", "UQ": "SW-CMOLS-UQ"}
FILES = {"G": "seeded_greedy", "L": "seeded_llm", "U": "seeded_union", "GQ": "agent_greedy", "LQ": "agent_llm", "UQ": "agent_union"}
# Palette CVD validée du script d'origine (toutes paires) ; mêmes couleurs par rôle.
COLOR = {"W": "#c62828", "G": "#009e73", "L": "#6a3d9a", "U": "#0072b2",
         "GQ": "#009e73", "LQ": "#6a3d9a", "UQ": "#e69f00"}
MARKER = {"W": "s", "G": "D", "L": "o", "U": "*", "GQ": "D", "LQ": "o", "UQ": "^"}
SIZE = {"W": 16, "G": 22, "L": 20, "U": 30, "GQ": 22, "LQ": 20, "UQ": 22}
ZORDER = {"W": 3, "G": 4, "L": 5, "U": 6, "GQ": 5, "LQ": 5, "UQ": 6}

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


def load_capped(raw_path, sizes_path, cap_runs, cap_pts=4000):
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
    out = {"W": load_capped(str(D.W_FRONTS[0]).format(i=inst), str(D.W_FRONTS[1]).format(i=inst), RUN_CAP)}
    for k in SYSTEMS[1:]:
        out[k] = load_capped(FR / f"{FILES[k]}_{inst}_raw.txt", FR / f"{FILES[k]}_{inst}_sizes.txt", RUN_CAP)
    return out


def fig_2d(inst, data):
    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    for k in SYSTEMS:
        d = data[k]
        ax.scatter(d[:, 0], d[:, 1], s=SIZE[k], marker=MARKER[k], c=COLOR[k],
                   alpha=0.85, zorder=ZORDER[k], label=f"{LABEL[k]} (n={len(d)})")
    ax.set_xlabel("Profit objectif 1", fontsize=11)
    ax.set_ylabel("Profit objectif 2", fontsize=11)
    ax.set_title(f"Instance {inst} (2 objectifs, 50 répétitions) — plus haut/droite = mieux",
                 fontsize=12.5, weight="bold", pad=10)
    ax.grid(True, alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.92)
    fig.tight_layout()
    p = OUT / f"{PREFIX}{inst}.png"
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
        d = sub(data[k], 500)
        ax.scatter(d[:, 0], d[:, 1], d[:, 2], s=SIZE[k] * 0.9, marker=MARKER[k], c=COLOR[k],
                   alpha=0.8, depthshade=False, label=f"{LABEL[k]} (n={len(data[k])})")
    ax.set_xlabel("Objectif 1", fontsize=10, labelpad=8)
    ax.set_ylabel("Objectif 2", fontsize=10, labelpad=8)
    ax.set_zlabel("Objectif 3", fontsize=10, labelpad=8)
    ax.set_title(f"Instance {inst} (3 objectifs, 50 répétitions) — points 3D réels",
                 fontsize=12.5, weight="bold", pad=6)
    ax.view_init(elev=22, azim=-60)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.92)
    fig.tight_layout()
    p = OUT / f"{PREFIX}{inst}.png"
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

    subd = {k: sub(data[k], 600) for k in SYSTEMS}
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
    fig.legend(handles=handles, fontsize=9.5, loc="lower center", ncol=len(SYSTEMS),
               bbox_to_anchor=(0.5, -0.02), framealpha=0.92)
    fig.suptitle(f"Instance {inst} ({m} objectifs, 50 répétitions) — "
                 f"matrice de nuages de points (paires d'objectifs)",
                 fontsize=13, weight="bold", y=1.01)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    p = OUT / f"{PREFIX}{inst}.png"
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
