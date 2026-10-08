"""Figure de la diapositive 9 : construction réelle du cloud SW-CMOLS-L sur 250.2.

Voisins (cloud), 30 graines gloutonnes, et les 5 graines centrales avant / après
la décision du LLM (encart agrandi). Sortie : presentation/figures_c1/fig_pipeline_L_250.2.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1]), str(_Path(__file__).resolve().parents[1] / "seeding")]
import common as C
import build_llm_cloud as P

INST = "250.2"
OUT = C.ROOT / "figures" / "out"
OUT.mkdir(parents=True, exist_ok=True)

k, n, caps, w, p, lams, idx = P.setup(INST)
seeds_g, _, _ = P.build_seeds(INST, False)
_, _, log = P.build_seeds(INST, True)
cloud = [[int(x) for x in l.split()] for l in open(C.HERE / "clouds" / f"llm_{INST}.txt")
         if l.strip() and not l.startswith("#")]
F = lambda s: p[:, s].sum(1)
cl = np.array([F(s) for s in cloud])
sg = np.array([F(s) for s in seeds_g])

GREY, ORANGE, PURPLE = "#B7C3CF", "#F28E2B", "#7656A6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
fig, ax = plt.subplots(figsize=(7.4, 6.0), dpi=200)
ax.scatter(cl[:, 0], cl[:, 1], s=7, c=GREY, alpha=0.55, lw=0, label="voisins : 49 par graine (cloud de 1 500)")
ax.scatter(sg[:, 0], sg[:, 1], s=46, c=ORANGE, edgecolors="white", lw=0.6, zorder=3,
           label="30 graines gloutonnes (une par direction λ)")
for e in log:
    a, b = np.array(e["start"]), np.array(e["final"])
    ax.scatter(*b, s=120, marker="*", c=PURPLE, edgecolors="white", lw=0.5, zorder=5)
ax.scatter([], [], s=120, marker="*", c=PURPLE, label="5 graines centrales après le LLM")
ax.set_xlabel("Profit objectif 1  (f₁)")
ax.set_ylabel("Profit objectif 2  (f₂)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.2)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, fontsize=10, frameon=False)

# encart : zone centrale, avant -> après LLM
ins = ax.inset_axes([0.07, 0.07, 0.47, 0.47])
x0, x1 = 8800, 9400; y0, y1 = 9050, 9750
m = (cl[:, 0] > x0) & (cl[:, 0] < x1) & (cl[:, 1] > y0) & (cl[:, 1] < y1)
ins.scatter(cl[m, 0], cl[m, 1], s=8, c=GREY, alpha=0.6, lw=0)
for e in log:
    a, b = np.array(e["start"]), np.array(e["final"])
    ins.scatter(*a, s=60, c=ORANGE, edgecolors="white", lw=0.6, zorder=3)
    if not np.array_equal(a, b):
        ins.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.6), zorder=4)
    ins.scatter(*b, s=150, marker="*", c=PURPLE, edgecolors="white", lw=0.5, zorder=5)
ins.set_xlim(x0, x1); ins.set_ylim(y0, y1)
ins.set_title("zoom : graine gloutonne → graine après LLM", fontsize=9.5, color=PURPLE)
ins.tick_params(labelsize=7.5)
for s in ins.spines.values():
    s.set_edgecolor(PURPLE)
ax.indicate_inset_zoom(ins, edgecolor=PURPLE, alpha=0.6)
fig.tight_layout()
out = OUT / f"fig_pipeline_L_{INST}.png"
fig.savefig(out, bbox_inches="tight")
print("saved", out, [(e["seed"], e["start"], e["final"]) for e in log])
