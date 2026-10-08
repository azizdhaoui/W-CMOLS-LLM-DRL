# -*- coding: utf-8 -*-
"""Les deux schémas du chapitre 3.

Le chapitre 2 etait le dernier entierement en prose : treize pages, dont 4 528
mots decrivant le solveur de reference sans un seul schema de sa boucle -- et
une decouverte, le balayage tronque des poids (section 2.4.8), enoncee par un
tableau de chiffres alors qu'elle se voit d'un coup d'oeil.

  Figure 3.1  La boucle W-CMOLS (section 3.4.2).
  Figure 3.2  Le balayage tronqué (section 3.5), tracé sur les vrais
              fichiers de poids du solveur.

RIEN N'EST REDESSINE DE MEMOIRE. La figure 2.2 lit
Weights_{2,3,4}obj_FQ*.txt, les fichiers que le solveur consomme, et
re-verifie au passage les quatre colonnes du tableau 2.3 : nombre de vecteurs,
plage de lambda_1 sur les vecteurs employes, et premier indice ou lambda_1 > 0.
Une divergence entre le dessin et le tableau interromprait la generation.
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
DATA = V3 / "data" / "instances"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "figure.dpi": 150})

BLEU = "#1a4dbd"
ORANGE = "#c1650a"
GRIS = "#7a7a7a"
ENCRE = "#1c1c1c"
T_REF = 100          # valeur de la configuration de reference (tableau 2.1)

FICHIERS = {2: "Weights_2obj_FQ200.txt",
            3: "Weights_3obj_FQ100.txt",
            4: "Weights_4obj_FQ40.txt"}


def lire(m: int) -> np.ndarray:
    lignes = [l for l in (DATA / FICHIERS[m]).read_text().splitlines() if l.strip()]
    return np.array([[float(x) for x in l.split()] for l in lignes])


# --------------------------------------------------------------------------
def figure_balayage() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7))
    resume = []

    for ax, m in zip(axes, (2, 3, 4)):
        W = lire(m)
        l1 = W[:, 0]
        n = len(W)
        employes = l1[:T_REF] if n > T_REF else l1
        # Premier indice ou lambda_1 > 0, compte A PARTIR DE 0 -- la convention
        # du memoire, qui ecrit « les iterations 1 a 100 consomment les indices
        # 0 a 99 ». C'est ce qui rend le cas m = 4 si net : le premier vecteur
        # utile porte l'indice 100, la troncature le manque d'une position.
        nz = np.flatnonzero(l1 > 0)
        premier = int(nz[0]) if len(nz) else None
        resume.append((m, n, min(n, T_REF), employes.min(), employes.max(),
                       premier))

        ax.plot(np.arange(1, n + 1), l1, lw=1.1, color=GRIS, alpha=0.75)
        k = min(n, T_REF)
        ax.plot(np.arange(1, k + 1), l1[:k], lw=2.0, color=BLEU, zorder=3)
        if n > T_REF:
            ax.axvspan(T_REF, n, color="#000000", alpha=0.045, lw=0)
            ax.axvline(T_REF, color=ORANGE, lw=1.5, ls="--", zorder=4)
            ax.annotate(f"$T = {T_REF}$", (T_REF, 0.93), color=ORANGE,
                        fontsize=9, ha="left" if m == 3 else "right",
                        xytext=(5 if m == 3 else -5, 0),
                        textcoords="offset points")
        ax.set_xscale("log")
        ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel("indice du vecteur de poids  (échelle log)")
        if m == 2:
            ax.set_ylabel("$\\lambda_1$")
        ax.set_title(f"$m = {m}$ — {n} vecteurs", fontsize=10.5, pad=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#bbbbbb")
        ax.tick_params(colors="#777777", labelsize=8.5)
        ax.grid(alpha=0.13, lw=0.6)
        ax.set_axisbelow(True)

        if n > T_REF and employes.max() == 0.0:
            ax.text(0.5, 0.5, "$\\lambda_1 = 0$\nsur tout le balayage",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10.5, color=ORANGE, fontweight="bold")

    fig.suptitle("Portion du simplexe atteinte en $T = 100$ itérations — "
                 "bleu : vecteurs employés ; gris : jamais atteints",
                 fontsize=10.5, y=1.03)
    fig.tight_layout()
    p = OUT / "weight_sweep_truncation.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)

    print("  vérification du tableau 3.2 sur les fichiers réels :")
    attendu = {2: (51, 1), 3: (626, 25), 4: (1001, 100)}
    for m, n, k, lo, hi, premier in resume:
        n_att, p_att = attendu[m]
        print(f"    m={m}  {n:5d} vecteurs (tableau : {n_att})   "
              f"λ1 employés ∈ [{lo:.3f} ; {hi:.3f}]   "
              f"premier λ1>0 à l'indice {premier} (tableau : {p_att})")
        assert n == n_att, f"m={m}: {n} vecteurs, le tableau 2.3 en annonce {n_att}"
        assert premier == p_att, (f"m={m}: premier λ1>0 a l'indice {premier}, "
                                  f"le tableau 2.3 annonce {p_att}")
    print(f"  {p.name}")
    return p


# --------------------------------------------------------------------------
def figure_boucle() -> Path:
    """Figure 3.1 -- la boucle de W-CMOLS (section 3.4.2)."""
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    etapes = [
        (1.55, 3.75, "Population initiale\n$\\alpha$ solutions", GRIS),
        (5.00, 3.75, "Vecteur de poids $\\lambda(g)$\nsuivant de la liste", ORANGE),
        (8.45, 3.75, "Fitness par indicateur\n$I_{\\varepsilon+}$ pondéré", BLEU),
        (8.45, 1.55, "Voisinage $(1, L)$\n+ contrôle de faisabilité", BLEU),
        (5.00, 1.55, "Archive globale\nextraction non dominée", GRIS),
    ]
    for cx, cy, txt, coul in etapes:
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx - 1.42, cy - 0.52), 2.84, 1.04,
            boxstyle="round,pad=0.10", facecolor=coul, alpha=0.14,
            edgecolor=coul, linewidth=1.3))
        ax.text(cx, cy, txt, ha="center", va="center", fontsize=9.2,
                color=ENCRE)

    fleches = [((2.97, 3.75), (3.58, 3.75)), ((6.42, 3.75), (7.03, 3.75)),
               ((8.45, 3.23), (8.45, 2.07)), ((7.03, 1.55), (6.42, 1.55))]
    for xy0, xy1 in fleches:
        ax.annotate("", xy=xy1, xytext=xy0,
                    arrowprops=dict(arrowstyle="-|>", color="#666666", lw=1.5))

    # retour de boucle : archive -> vecteur suivant
    ax.annotate("", xy=(5.00, 3.23), xytext=(5.00, 2.07),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6,
                                ls="--"))
    ax.text(5.18, 2.65, "itération suivante", fontsize=8.6, color=ORANGE,
            va="center")

    ax.annotate("", xy=(1.55, 2.07), xytext=(3.58, 1.55),
                arrowprops=dict(arrowstyle="-|>", color="#666666", lw=1.3,
                                connectionstyle="arc3,rad=0.28"))
    ax.text(1.30, 1.55, "front rendu\naprès $T$ itérations", fontsize=8.8,
            color="#555555", ha="center", va="center")

    ax.text(5.00, 4.75, "Une itération de W-CMOLS", ha="center",
            fontsize=11, color=ENCRE)
    fig.tight_layout()
    p = OUT / "wcmols_loop.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  {p.name}")
    return p


if __name__ == "__main__":
    print("schémas du chapitre 3 :")
    figure_boucle()
    figure_balayage()
    print(f"\n-> {OUT}")
