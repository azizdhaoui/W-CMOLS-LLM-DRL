# -*- coding: utf-8 -*-
"""Les trois schemas du chapitre 1 (concepts fondamentaux).

POURQUOI. Le chapitre 1 etait le seul du memoire entierement en prose : huit
pages sans une image, alors qu'il pose les deux notions dont tout le reste
depend -- la dominance de Pareto et l'hypervolume. Le memoire de reference du
laboratoire porte six figures sur ses deux chapitres de fond.

CE QUI EST DESSINE, ET CE QUI EST CALCULE. Comme dans make_diversity_schema.py,
aucune quantite n'est affirmee : l'hypervolume affiche a la figure 1.2 est
calcule par balayage exact a partir des points reellement traces, avec la
formule meme du chapitre (section 1.2.4) :

    HV(A, r) = Vol( U_{a in A} [r, f(a)] )

de sorte que l'inegalite HV(A) > HV(B) illustrant la monotonie stricte est
verifiee sur les donnees du dessin, et non postulee par la legende.

LES POINTS SONT SYNTHETIQUES et en deux dimensions, pour la seule clarte
pedagogique : les fronts reels du memoire vont jusqu'a quatre objectifs et
figurent aux chapitres 3 a 5.

Sorties -> figures/images/concept_{dominance,hypervolume,mdp_sketch}.png
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

V3 = Path(__file__).resolve().parents[1]
OUT = V3 / "figures" / "images"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "figure.dpi": 150})

# Charte du memoire, reprise de make_diversity_schema.py.
BLEU = "#1a4dbd"     # ce qui est meilleur / l'agent
ORANGE = "#c1650a"   # ce qui porte le message de la figure
GRIS = "#7a7a7a"     # ce qui est moins bon / recessif
ENCRE = "#1c1c1c"


def cadre(ax, titre: str) -> None:
    """Axes recessifs : le trait de donnee doit primer sur le decor."""
    ax.set_xlabel("$f_1$  (à maximiser)")
    ax.set_ylabel("$f_2$  (à maximiser)")
    ax.set_title(titre, fontsize=10.5, pad=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bbbbbb")
    ax.tick_params(colors="#777777", labelsize=8.5)
    ax.grid(alpha=0.13, linewidth=0.6)
    ax.set_axisbelow(True)


def non_domines(P: np.ndarray) -> np.ndarray:
    """Filtre non domine, en maximisation (definition 1.1 du memoire)."""
    keep = np.ones(len(P), bool)
    for i in range(len(P)):
        if not keep[i]:
            continue
        keep &= ~(np.all(P <= P[i], 1) & np.any(P < P[i], 1))
        keep[i] = True
    return P[keep]


def hypervolume(P: np.ndarray, r: np.ndarray) -> float:
    """HV exact en 2D, maximisation, par balayage.

    Les points non domines tries par f1 decroissant ont f2 croissant ; l'union
    des boites [r, f(a)] se decoupe alors en bandes verticales disjointes.
    """
    F = non_domines(P)
    F = F[np.argsort(-F[:, 0])]
    # balayage : bande verticale entre f1 du point courant et f1 du suivant
    hv = 0.0
    for i, p in enumerate(F):
        x_suivant = F[i + 1][0] if i + 1 < len(F) else r[0]
        hv += (p[0] - x_suivant) * (p[1] - r[1])
    return float(hv)


# --------------------------------------------------------------------------
def figure_dominance() -> Path:
    """Figure 1.1 -- l'ordre est PARTIEL : deux quadrants sont incomparables."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.3))

    # --- panneau gauche : les quatre regions relatives a un point x
    x = np.array([5.0, 5.0])
    L = 10.0
    a1.add_patch(mpatches.Rectangle(x, L - x[0], L - x[1],
                                    facecolor=BLEU, alpha=0.15, lw=0))
    a1.add_patch(mpatches.Rectangle((0, 0), x[0], x[1],
                                    facecolor=GRIS, alpha=0.16, lw=0))
    for xy, w, h in (((0, x[1]), x[0], L - x[1]), ((x[0], 0), L - x[0], x[1])):
        a1.add_patch(mpatches.Rectangle(xy, w, h, facecolor=ORANGE,
                                        alpha=0.17, lw=0))

    a1.axhline(x[1], color="#999999", lw=0.8, ls="--")
    a1.axvline(x[0], color="#999999", lw=0.8, ls="--")
    a1.plot(*x, "o", ms=9, color=ENCRE, zorder=5)
    a1.annotate("$x$", x, textcoords="offset points", xytext=(9, -14),
                fontsize=12, fontweight="bold")

    a1.text(7.5, 7.5, "dominent $x$", ha="center", va="center",
            fontsize=9.5, color=BLEU, fontweight="bold")
    a1.text(2.5, 2.5, "dominées par $x$", ha="center", va="center",
            fontsize=9.5, color="#5a5a5a", fontweight="bold")
    for pos in ((2.5, 7.5), (7.5, 2.5)):
        a1.text(*pos, "incomparables\navec $x$", ha="center", va="center",
                fontsize=9.5, color=ORANGE, fontweight="bold")

    a1.set_xlim(0, L)
    a1.set_ylim(0, L)
    cadre(a1, "Les quatre régions relatives à une solution $x$")

    # --- panneau droit : un ensemble, son front
    rng = np.random.default_rng(11)
    P = rng.uniform(0.8, 9.2, size=(34, 2))
    F = non_domines(P)
    F = F[np.argsort(F[:, 0])]
    dom = np.array([p for p in P if not any(np.array_equal(p, f) for f in F)])

    a2.plot(dom[:, 0], dom[:, 1], "o", ms=5.5, color=GRIS, alpha=0.55,
            mew=0, label="dominées")
    a2.plot(F[:, 0], F[:, 1], "-", color=ORANGE, lw=1.6, alpha=0.75, zorder=3)
    a2.plot(F[:, 0], F[:, 1], "o", ms=8, color=ORANGE, mec="white", mew=1.4,
            zorder=4, label="front de Pareto")
    # fond opaque : sans lui la legende se pose sur un point domine
    a2.legend(loc="lower left", fontsize=9, framealpha=0.95,
              facecolor="white", edgecolor="none")
    a2.set_xlim(0, 10)
    a2.set_ylim(0, 10)
    cadre(a2, "Un ensemble de 34 solutions et son front")

    fig.tight_layout()
    p = OUT / "concept_dominance.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  {p.name}  ({len(F)} points non dominés sur {len(P)})")
    return p


# --------------------------------------------------------------------------
def figure_hypervolume() -> tuple[Path, float, float]:
    """Figure 1.2 -- l'hypervolume, et sa monotonie stricte VERIFIEE."""
    r = np.array([0.0, 0.0])
    A = np.array([[1.4, 8.6], [3.1, 7.2], [5.0, 5.4], [7.0, 3.0], [8.8, 1.5]])
    # B est domine par A : chaque point recule de 1.2 sur les deux objectifs.
    B = A - 1.2

    hv_a, hv_b = hypervolume(A, r), hypervolume(B, r)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), sharey=True)
    for ax, P, nom, coul, hv in ((axes[0], A, "$A$", BLEU, hv_a),
                                 (axes[1], B, "$B$", GRIS, hv_b)):
        F = P[np.argsort(-P[:, 0])]
        for i, p in enumerate(F):
            x_suiv = F[i + 1][0] if i + 1 < len(F) else r[0]
            ax.add_patch(mpatches.Rectangle((x_suiv, r[1]), p[0] - x_suiv,
                                            p[1] - r[1], facecolor=coul,
                                            alpha=0.20, lw=0))
        # contour en escalier de la region dominee
        xs, ys = [r[0]], [F[-1][1]]
        for i, p in enumerate(F[::-1]):
            xs += [p[0], p[0]]
            ys += [p[1], F[::-1][i + 1][1] if i + 1 < len(F) else r[1]]
        ax.plot(xs, ys, color=coul, lw=1.5, alpha=0.85, zorder=3)
        ax.plot(P[:, 0], P[:, 1], "o", ms=8, color=coul, mec="white", mew=1.4,
                zorder=4)
        ax.plot(*r, "s", ms=8, color=ORANGE, zorder=5)
        ax.annotate("point de référence $r$", r, textcoords="offset points",
                    xytext=(12, 6), fontsize=9, color=ORANGE)
        ax.set_xlim(-0.4, 10)
        ax.set_ylim(-0.4, 10)
        cadre(ax, f"Approximation {nom} — $HV = {hv:.2f}$".replace(".", "{,}"))
    axes[1].set_ylabel("")

    fig.suptitle("$A$ domine $B$, donc $HV(A) > HV(B)$ : la monotonie stricte "
                 f"({hv_a:.2f} > {hv_b:.2f})".replace(".", ","), fontsize=10.5, y=1.02)
    fig.tight_layout()
    p = OUT / "concept_hypervolume.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    assert hv_a > hv_b, "la monotonie stricte doit se verifier sur le dessin"
    print(f"  {p.name}  HV(A)={hv_a:.3f} > HV(B)={hv_b:.3f}  (calcule)")
    return p, hv_a, hv_b


# --------------------------------------------------------------------------
def figure_mdp() -> Path:
    """Figure 1.3 -- la boucle agent/environnement du MDP (definition 1.3)."""
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    for cx, nom, coul in ((2.3, "Agent\n$\\pi(a \\mid s)$", BLEU),
                          (7.7, "Environnement\n$P(s' \\mid s,a)$", GRIS)):
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx - 1.5, 1.65), 3.0, 1.25, boxstyle="round,pad=0.12",
            facecolor=coul, alpha=0.15, edgecolor=coul, linewidth=1.4))
        ax.text(cx, 2.28, nom, ha="center", va="center", fontsize=11,
                color=ENCRE)

    # action : agent -> environnement (au-dessus)
    ax.annotate("", xy=(6.2, 3.35), xytext=(3.8, 3.35),
                arrowprops=dict(arrowstyle="-|>", color=BLEU, lw=1.8,
                                connectionstyle="arc3,rad=-0.32"))
    ax.text(5.0, 4.0, "action  $a_t$", ha="center", fontsize=10, color=BLEU)

    # retour : environnement -> agent (en dessous)
    ax.annotate("", xy=(3.8, 1.2), xytext=(6.2, 1.2),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.8,
                                connectionstyle="arc3,rad=-0.32"))
    ax.text(5.0, 0.45, "récompense  $r_t$      état  $s_{t+1}$",
            ha="center", fontsize=10, color=ORANGE)

    # gamma n'est pas un composant place ENTRE l'agent et l'environnement :
    # c'est le poids des recompenses futures dans le retour. On l'ecrit donc
    # sous la fleche de recompense, sous sa forme utile.
    ax.text(5.0, 2.30,
            "$G_t = \\sum_{i \\geq 0} \\gamma^{\\,i} R_{t+i}$",
            ha="center", va="center", fontsize=11, color="#8a8a8a")
    ax.text(5.0, 1.80, "retour actualisé", ha="center", va="center",
            fontsize=8.5, color="#9a9a9a")

    fig.tight_layout()
    p = OUT / "concept_mdp_sketch.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  {p.name}")
    return p


if __name__ == "__main__":
    print("schémas du chapitre 1 :")
    figure_dominance()
    figure_hypervolume()
    figure_mdp()
    print(f"\n-> {OUT}")
