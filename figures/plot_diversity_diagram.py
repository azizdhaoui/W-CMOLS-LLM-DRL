# -*- coding: utf-8 -*-
"""Geometric schematic for chapter 3, section 3.2.2.2 (Diversite : espacement et
etendue). Illustrates the two diversity measures on SYNTHETIC 2D point sets
(m=2, for pedagogical clarity only -- the real S1 clouds are higher-dimensional;
their actual measured Esp/Ete values are reported in section 3.4.6, not here).

Both quantities are computed with the EXACT formulas given in the chapter text
(03_systeme_s1.md, section 3.2.2.2), not asserted:
  d_i   = min_{j!=i} sum_k |y_i^k - y_j^k|                      (L1 nearest neighbour)
  Esp(C)= (1/n) sum_i |d_i - mean(d)|                           (mean absolute deviation)
  Ete(C)= || e ||_2,  e_k = max_i y_i^k - min_i y_i^k            (L2 norm of per-axis range)

Panel 1 (left, tall): concept illustration of d_i -- nearest-neighbour links on a
small irregular point set.
Panels 2x2 (right): the four Esp/Ete combinations named explicitly in the chapter
text ("un cloud peut etre tres uniformement reparti ... ou a l'inverse couvrir une
large region de facon irreguliere"), with the real computed Esp/Ete of each panel's
own synthetic points shown in its title -- so the classification is verified, not
just labelled.

Output -> figures/images/cloud_diversity.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

V3 = Path(__file__).resolve().parents[1]
OUT = V3 / "figures" / "images"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.dpi": 150})
RNG = np.random.default_rng(7)

POINT_COLOR = "#1a4dbd"
LINK_COLOR = "#c1650a"


def nn_l1(pts):
    n = len(pts)
    d = np.zeros(n)
    for i in range(n):
        diffs = np.abs(pts - pts[i]).sum(axis=1)
        diffs[i] = np.inf
        d[i] = diffs.min()
    return d


def espacement(pts):
    d = nn_l1(pts)
    return float(np.mean(np.abs(d - d.mean()))), d


def etendue(pts):
    e = pts.max(axis=0) - pts.min(axis=0)
    return float(np.linalg.norm(e))


def grid_cloud(lo, hi, n_side, jitter):
    xs = np.linspace(lo, hi, n_side)
    gx, gy = np.meshgrid(xs, xs)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    pts = pts + RNG.normal(0, jitter, pts.shape)
    return np.clip(pts, 0, 1)


def clumped_cloud(centers, per_clump, spread):
    parts = [RNG.normal(c, spread, (per_clump, 2)) for c in centers]
    return np.clip(np.vstack(parts), 0, 1)


# ------------------------------------------------------------- the 4 synthetic clouds
cloud_ll = grid_cloud(0.34, 0.64, 4, 0.006)                                    # low Esp, low Ete
cloud_lh = grid_cloud(0.04, 0.96, 4, 0.006)                                    # low Esp, high Ete
cloud_hl = clumped_cloud([(0.36, 0.35), (0.58, 0.60), (0.62, 0.36)], 6, 0.020)  # high Esp, low Ete
cloud_hh = clumped_cloud([(0.08, 0.10), (0.92, 0.12), (0.12, 0.90), (0.88, 0.90)], 5, 0.020)  # high Esp, high Ete

panels = [
    ("low", "low", cloud_ll, "(a) grille reguliere, region confinee"),
    ("low", "high", cloud_lh, "(b) grille reguliere, pleine amplitude"),
    ("high", "low", cloud_hl, "(c) amas irreguliers, region confinee"),
    ("high", "high", cloud_hh, "(d) amas irreguliers, pleine amplitude"),
]
results = {}
for esp_lvl, ete_lvl, pts, label in panels:
    esp, _ = espacement(pts)
    ete = etendue(pts)
    results[label] = (esp, ete, esp_lvl, ete_lvl)
    print(f"{label}: Esp={esp:.4f} ({esp_lvl}), Ete={ete:.4f} ({ete_lvl})")

# sanity: verify the intended ordering actually holds on the computed values
esp_vals = {lbl: results[lbl][0] for lbl in results}
ete_vals = {lbl: results[lbl][1] for lbl in results}
assert esp_vals[panels[0][3]] < esp_vals[panels[2][3]], "espacement (a) should be < (c)"
assert esp_vals[panels[1][3]] < esp_vals[panels[3][3]], "espacement (b) should be < (d)"
assert ete_vals[panels[0][3]] < ete_vals[panels[1][3]], "etendue (a) should be < (b)"
assert ete_vals[panels[2][3]] < ete_vals[panels[3][3]], "etendue (c) should be < (d)"
print("ordering verified: Esp(a)<Esp(c), Esp(b)<Esp(d), Ete(a)<Ete(b), Ete(c)<Ete(d)")

# ------------------------------------------------------------- concept point set (small, irregular)
concept_pts = np.array([
    [0.10, 0.75], [0.22, 0.80], [0.18, 0.60],
    [0.55, 0.50], [0.62, 0.55], [0.50, 0.35],
    [0.85, 0.20], [0.90, 0.35], [0.72, 0.15],
])

fig = plt.figure(figsize=(15.5, 7.6))
gs = gridspec.GridSpec(2, 3, width_ratios=[1.25, 1, 1], wspace=0.32, hspace=0.42)

# -------- concept panel
axc = fig.add_subplot(gs[:, 0])
d_concept = nn_l1(concept_pts)
d_bar = d_concept.mean()
for i in range(len(concept_pts)):
    diffs = np.abs(concept_pts - concept_pts[i]).sum(axis=1)
    diffs[i] = np.inf
    j = int(np.argmin(diffs))
    axc.plot([concept_pts[i, 0], concept_pts[j, 0]], [concept_pts[i, 1], concept_pts[j, 1]],
              color=LINK_COLOR, linewidth=1.3, alpha=0.85, zorder=2)
axc.scatter(concept_pts[:, 0], concept_pts[:, 1], s=90, color=POINT_COLOR, zorder=3,
             edgecolor="white", linewidth=0.8)
for k in (0, 3, 6):
    axc.annotate(f"$d_{{{k+1}}}$={d_concept[k]:.3f}", concept_pts[k], textcoords="offset points",
                  xytext=(9, 9), fontsize=8.6, color=LINK_COLOR)
axc.set_xlim(-0.02, 1.02)
axc.set_ylim(-0.02, 1.02)
axc.set_title("Notion de distance au plus proche voisin\n"
              f"$d_i = \\min_{{j \\neq i}}\\sum_k|y_i^k-y_j^k|$  (norme $L_1$)  —  ici $\\bar d$={d_bar:.3f}",
              fontsize=10.5, weight="bold")
axc.set_xlabel("Objectif 1 (normalisé)")
axc.set_ylabel("Objectif 2 (normalisé)")
axc.grid(True, alpha=0.2)
axc.spines[["top", "right"]].set_visible(False)
axc.text(0.02, -0.16, "Esp(C) = écart moyen absolu des $d_i$ à $\\bar d$   —   un espacement faible = distances $d_i$ homogènes",
          transform=axc.transAxes, fontsize=8.8, color="#444444")

# -------- 2x2 classification grid
positions = {(0, 0): gs[0, 1], (0, 1): gs[0, 2], (1, 0): gs[1, 1], (1, 1): gs[1, 2]}
row_of = {"low": 0, "high": 1}
col_of = {"low": 0, "high": 1}
for esp_lvl, ete_lvl, pts, label in panels:
    r, c = row_of[esp_lvl], col_of[ete_lvl]
    ax = fig.add_subplot(positions[(r, c)])
    ax.scatter(pts[:, 0], pts[:, 1], s=34, color=POINT_COLOR, alpha=0.85,
               edgecolor="white", linewidth=0.4)
    esp, ete, _, _ = results[label]
    ax.set_title(f"{label}\nEsp(C)={esp:.4f}   Ete(C)={ete:.4f}", fontsize=9.3)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.tick_params(labelsize=7.5)
    ax.grid(True, alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)

fig.text(0.685, 0.955, "Étendue faible", ha="center", fontsize=10.5, weight="bold", color="#444444")
fig.text(0.915, 0.955, "Étendue élevée", ha="center", fontsize=10.5, weight="bold", color="#444444")
fig.text(0.475, 0.72, "Espacement\nfaible", ha="center", va="center", fontsize=10.5,
          weight="bold", color="#444444", rotation=90)
fig.text(0.475, 0.28, "Espacement\nélevé", ha="center", va="center", fontsize=10.5,
          weight="bold", color="#444444", rotation=90)

fig.suptitle("Espacement et étendue : deux mesures de diversité complémentaires et non redondantes (illustration 2D)",
             fontsize=13, weight="bold", y=1.015)

fig.tight_layout(rect=(0.02, 0, 1, 0.98))
p = OUT / "cloud_diversity.png"
fig.savefig(p, bbox_inches="tight")
plt.close(fig)
print(f"saved {p}")
