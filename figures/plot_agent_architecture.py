# -*- coding: utf-8 -*-
"""Figure 5.1 du memoire (section 5.3.3) : reseau Q a decodage autoregressif.

Fondé directement sur dqn_agent/agent.py
(classe AutoregressiveDuelingNet) : tronc = Linear(4,128)+ReLU, Linear(128,128)+ReLU ;
chacune des 4 tetes = Linear(128+ctx,64)+ReLU, puis V(64,1) et A(64,|grille|),
combines en Q = V + (A - moyenne(A)) ; ctx concatene le one-hot de chaque parametre
deja decode (conditionnement autoregressif), dans l'ordre alpha -> NBL -> L -> kappa.
Grilles : GRIDS_V3 (dqn_agent/grids.py), celles des resultats du chapitre 5.

Mise en page verticale (tetes empilees) : la figure est imprimee sur la largeur
de la page ; une mise en page horizontale de 19 pouces ramenait le texte a
environ 5 pt a l'impression. Ici, environ 10 pouces de large : le texte des
blocs, en gras, reste autour de 9 pt une fois imprime (remarque de l'encadrante).

Le prefixe « Fig. 5.1 » n'est pas dans l'image : la legende Markdown l'ajoute.

Output -> figures/images/agent_architecture.png
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, RegularPolygon

V3 = Path(__file__).resolve().parents[1]
OUT = V3 / "figures" / "images"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 150})

INK = "#1f1f1f"
BOX_EDGE = "#4a5b7c"
BOX_FACE = "#eef1f8"
TRUNK_FACE = "#e4e9f5"
STATE_FACE = "#fbf6e3"
STATE_EDGE = "#8a6d00"
GATE_EDGE = "#a3342c"
GATE_FACE = "#f7e6e4"
CTX_COLOR = "#c1650a"
BUS_COLOR = "#4a5b7c"
RESULT_FACE = "#e2f0e5"
RESULT_EDGE = "#1a9850"

FS = 13.5          # texte des blocs
FS_HEAD = 14.5     # titre des blocs
FS_NOTE = 12.5     # annotations hors blocs

FIG_W, FIG_H = 10.6, 10.4
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0.2, 0.2 + FIG_H)
ax.axis("off")


def box(cx, cy, w, h, lines, face=BOX_FACE, edge=BOX_EDGE, header_idx=0, normal_idx=()):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                linewidth=1.5, edgecolor=edge, facecolor=face, zorder=3))
    n = len(lines)
    top = cy + h / 2 - 0.27
    bot = cy - h / 2 + 0.25
    step = (top - bot) / (n - 1) if n > 1 else 0
    for i, txt in enumerate(lines):
        y = top - i * step if n > 1 else cy
        fs = FS_HEAD if i == header_idx else FS
        weight = "normal" if i in normal_idx else "bold"
        ax.text(cx, y, txt, ha="center", va="center", fontsize=fs, weight=weight,
                color=INK, zorder=4)


def arrow(p1, p2, color=INK, rad=0.0, lw=1.6, mutation=13, zorder=2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=mutation,
                                 linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}",
                                 zorder=zorder, shrinkA=1, shrinkB=1))


# ------------------------------------------------------------ état et tronc commun
TOP_Y = 9.85
box(1.55, TOP_Y, 2.7, 1.0, ["État de l'instance", "s : 4 nombres"],
    face=STATE_FACE, edge=STATE_EDGE)
arrow((2.9, TOP_Y), (3.55, TOP_Y))
box(5.75, TOP_Y, 4.3, 1.0, ["Tronc commun", "4 → 128 → 128 neurones (ReLU)"], face=TRUNK_FACE)

# h : sortie du tronc, transmise à chaque tête par un bus vertical à gauche
BUS_X, BUS_TOP = 0.3, 8.95
ax.plot([5.75, 5.75], [TOP_Y - 0.5, BUS_TOP], color=BUS_COLOR, linewidth=1.8, zorder=1)
ax.plot([BUS_X, 5.75], [BUS_TOP, BUS_TOP], color=BUS_COLOR, linewidth=1.8, zorder=1)
ax.text(5.95, BUS_TOP, "h : 128 nombres, transmis à chaque tête", fontsize=FS_NOTE,
        color=BUS_COLOR, ha="left", va="center", style="italic")

# ------------------------------------------------------------ quatre têtes empilées
HEAD_CX, HEAD_W, HEAD_H = 3.2, 5.1, 1.42
ROWS = [7.85, 6.05, 4.25, 2.45]
GATE_X, GATE_R = 6.45, 0.36
PILL_CX, PILL_W, PILL_H = 8.35, 2.3, 0.64
COLL_X = 9.9

heads = [
    ("Tête α  (1/4)", "h → 64 neurones (ReLU)", "6 scores : {10, 15, 20, 25, 30, 40}", "α"),
    ("Tête NBL  (2/4)", "h + one-hot(α) → 64 neurones (ReLU)", "6 scores : {30, 50, 70, 100, 130, 160}", "NBL"),
    ("Tête L  (3/4)", "h + one-hot(α, NBL) → 64 neurones (ReLU)", "4 scores : {3, 5, 8, 10}", "L"),
    ("Tête κ  (4/4)", "h + one-hot(α, NBL, L) → 64 neurones (ReLU)", "3 scores : {0,05 ; 0,1 ; 0,2}", "κ"),
]

ax.plot([BUS_X, BUS_X], [BUS_TOP, ROWS[-1]], color=BUS_COLOR, linewidth=1.8, zorder=1)
for i, (y, (name, inp, notes, sym)) in enumerate(zip(ROWS, heads)):
    arrow((BUS_X, y), (HEAD_CX - HEAD_W / 2, y), color=BUS_COLOR, lw=1.6, mutation=12)
    box(HEAD_CX, y, HEAD_W, HEAD_H, [name, inp, "Q = V + (A − moy(A))", notes], normal_idx={1})
    # masque de budget, puis choix de la meilleure valeur autorisée
    arrow((HEAD_CX + HEAD_W / 2, y), (GATE_X - GATE_R, y))
    ax.add_patch(RegularPolygon((GATE_X, y), numVertices=4, radius=GATE_R, orientation=0.785398,
                                edgecolor=GATE_EDGE, facecolor=GATE_FACE, linewidth=1.5, zorder=3))
    ax.text(GATE_X, y, "✕", fontsize=13, ha="center", va="center", color=GATE_EDGE,
            weight="bold", zorder=4)
    arrow((GATE_X + GATE_R, y), (PILL_CX - PILL_W / 2, y))
    box(PILL_CX, y, PILL_W, PILL_H, [f"argmax → {sym}"], face="#ffffff")
    # vers la configuration finale
    ax.plot([PILL_CX + PILL_W / 2, COLL_X], [y, y], color=RESULT_EDGE, linewidth=1.6, zorder=1)
    # contexte autorégressif : la valeur choisie entre dans la tête suivante
    if i < 3:
        arrow((PILL_CX, y - PILL_H / 2), (HEAD_CX + 1.4, ROWS[i + 1] + HEAD_H / 2 + 0.03),
              color=CTX_COLOR, lw=1.8, mutation=16, rad=-0.12, zorder=5)

ax.text(GATE_X, ROWS[0] + 0.62, "masque de budget", fontsize=FS_NOTE, color=GATE_EDGE,
        ha="center", va="center", style="italic", weight="bold")
ax.text(7.95, ROWS[1] + 0.57, "choix → tête suivante (one-hot)",
        fontsize=FS_NOTE, color=CTX_COLOR, ha="center", va="center", weight="bold")

# ------------------------------------------------------------ configuration finale
RES_Y = 0.85
ax.plot([COLL_X, COLL_X], [ROWS[0], RES_Y], color=RESULT_EDGE, linewidth=1.6, zorder=1)
box(5.2, RES_Y, 4.6, 0.95, ["Configuration choisie", "(α, NBL, L, κ)"],
    face=RESULT_FACE, edge=RESULT_EDGE)
arrow((COLL_X, RES_Y), (5.2 + 2.3, RES_Y), color=RESULT_EDGE, lw=1.8)

fig.tight_layout()
p = OUT / "agent_architecture.png"
fig.savefig(p, bbox_inches="tight")
plt.close(fig)
print(f"saved {p}")
