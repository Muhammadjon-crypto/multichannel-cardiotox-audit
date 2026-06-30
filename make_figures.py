import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
import os

os.makedirs("figures", exist_ok=True)

# ---- global style: clean, publication quality ----
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "savefig.dpi": 600,
    "figure.dpi": 150,
})
# colorblind-friendly (Okabe-Ito)
C_HERG = "#0072B2"   # blue
C_NAV  = "#E69F00"   # orange
C_CAV  = "#D55E00"   # vermillion
GREY   = "#444444"

# ============================================================
# FIGURE 1 — Learning curve (data starvation)
# ============================================================
sizes = np.array([9412, 5000, 2830, 1500, 800, 463, 250, 120])
means = np.array([0.928, 0.899, 0.873, 0.842, 0.809, 0.782, 0.751, 0.698])
stds  = np.array([0.000, 0.004, 0.003, 0.007, 0.004, 0.005, 0.008, 0.008])

fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.plot(sizes, means, "-o", color=C_HERG, lw=1.6, ms=4, label="hERG model (mean of 5 runs)")
ax.fill_between(sizes, means - stds, means + stds, color=C_HERG, alpha=0.18,
                label="\u00b11 SD")
ax.axvline(2830, color=C_NAV, ls="--", lw=1.1)
ax.axvline(463,  color=C_CAV, ls="--", lw=1.1)
ax.text(2830, 0.705, " Nav1.5\n size", color=C_NAV, fontsize=7.5, va="bottom", ha="left")
ax.text(463, 0.93, "Cav1.2\nsize ", color=C_CAV, fontsize=7.5, va="top", ha="right")
ax.set_xscale("log")
ax.set_xlabel("Training compounds (log scale)")
ax.set_ylabel("External ROC-AUC")
ax.set_ylim(0.66, 0.95)
ax.grid(True, which="both", alpha=0.2, lw=0.5)
ax.legend(fontsize=6.6, loc="lower right", frameon=True, framealpha=0.9)
fig.tight_layout()
fig.savefig("figures/fig1_learning_curve.png", bbox_inches="tight")
plt.close(fig)

# ============================================================
# FIGURE 2 — Channel scarcity & imbalance (grouped bars + active%)
# ============================================================
channels = ["hERG\n(KCNH2)", "Nav1.5\n(SCN5A)", "Cav1.2\n(CACNA1C)"]
actives   = np.array([6057, 231, 32])
inactives = np.array([6044, 2599, 431])
active_pct = np.array([50.1, 8.2, 6.9])
x = np.arange(len(channels))

fig, ax = plt.subplots(figsize=(3.4, 2.7))
ax.bar(x, inactives, color="#BBBBBB", label="Inactives", width=0.62)
ax.bar(x, actives, bottom=inactives, color=C_HERG, label="Actives (blockers)", width=0.62)
ax.set_yscale("log")
ax.set_ylabel("Compounds (log scale)")
ax.set_xticks(x); ax.set_xticklabels(channels, fontsize=7.5)
for i in range(len(channels)):
    total = actives[i] + inactives[i]
    ax.text(i, total * 1.15, f"n={total:,}\n{active_pct[i]}% act.",
            ha="center", va="bottom", fontsize=7, color=GREY)
ax.set_ylim(10, 40000)
ax.legend(fontsize=7, loc="upper right", frameon=True)
fig.tight_layout()
fig.savefig("figures/fig2_scarcity.png", bbox_inches="tight")
plt.close(fig)

# ============================================================
# FIGURE 3 — Cross-channel overlap (annotated 3-set Venn, not to scale)
# ============================================================
fig, ax = plt.subplots(figsize=(3.4, 3.0))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
# three circles
r = 2.6
c_herg = (3.7, 6.2); c_nav = (6.3, 6.2); c_cav = (5.0, 4.0)
for center, col, lab in [(c_herg, C_HERG, "hERG"), (c_nav, C_NAV, "Nav1.5"), (c_cav, C_CAV, "Cav1.2")]:
    ax.add_patch(Circle(center, r, facecolor=col, alpha=0.22, edgecolor=col, lw=1.4))
# labels
ax.text(c_herg[0]-1.3, c_herg[1]+2.0, "hERG", color=C_HERG, fontsize=9, fontweight="bold", ha="center")
ax.text(c_nav[0]+1.3, c_nav[1]+2.0, "Nav1.5", color=C_NAV, fontsize=9, fontweight="bold", ha="center")
ax.text(c_cav[0], c_cav[1]-2.45, "Cav1.2", color=C_CAV, fontsize=9, fontweight="bold", ha="center")
# region counts
ax.text(2.7, 6.7, "11,769", fontsize=8, ha="center")          # hERG only
ax.text(7.3, 6.7, "2,392", fontsize=8, ha="center")           # Nav only
ax.text(5.0, 2.9, "216", fontsize=8, ha="center")             # Cav only
ax.text(5.0, 7.1, "216", fontsize=8, ha="center")             # hERG&Nav only
ax.text(3.7, 4.7, "26", fontsize=8, ha="center")              # hERG&Cav only
ax.text(6.3, 4.7, "131", fontsize=8, ha="center")             # Nav&Cav only
ax.text(5.0, 5.55, "90", fontsize=9, ha="center", fontweight="bold", color="#000000")  # all three
ax.set_title("Compounds measured per channel and shared\n(only 90 across all three)", fontsize=8.2)
fig.tight_layout()
fig.savefig("figures/fig3_overlap_venn.png", bbox_inches="tight")
plt.close(fig)

# ============================================================
# FIGURE 4 — Confidence-interval comparison (forest plot, ROC-AUC & MCC)
# ============================================================
rows = [
    ("Nav1.5 / Random",   0.952, 0.914, 0.975, 0.597, 0.476, 0.712, C_NAV),
    ("Nav1.5 / Scaffold", 0.945, 0.920, 0.965, 0.591, 0.491, 0.680, C_NAV),
    ("Cav1.2 / Random",   0.922, 0.782, 0.996, 0.658, 0.277, 0.900, C_CAV),
    ("Cav1.2 / Scaffold", 0.921, 0.697, 1.000, 0.627, 0.193, 0.909, C_CAV),
]
labels = [r[0] for r in rows]
ypos = np.arange(len(rows))[::-1]

fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.4), sharey=True)
# ROC-AUC
ax = axes[0]
for (lab, m, lo, hi, *_ , col), y in zip(rows, ypos):
    ax.plot([lo, hi], [y, y], color=col, lw=2.2)
    ax.plot(m, y, "o", color=col, ms=5)
ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=7.2)
ax.set_xlim(0.6, 1.02); ax.set_xlabel("ROC-AUC [95% CI]", fontsize=8)
ax.grid(True, axis="x", alpha=0.2, lw=0.5)
ax.set_title("Ranking power", fontsize=8.5)
# MCC
ax = axes[1]
for (lab, *_ , m, lo, hi, col), y in zip(rows, ypos):
    ax.plot([lo, hi], [y, y], color=col, lw=2.2)
    ax.plot(m, y, "o", color=col, ms=5)
ax.set_xlim(0.0, 1.02); ax.set_xlabel("MCC [95% CI]", fontsize=8)
ax.grid(True, axis="x", alpha=0.2, lw=0.5)
ax.set_title("Imbalance-robust score", fontsize=8.5)
fig.suptitle("Cav1.2 intervals span near-chance to near-perfect (8 test actives)", fontsize=8, y=1.04)
fig.tight_layout()
fig.savefig("figures/fig4_ci_forest.png", bbox_inches="tight")
plt.close(fig)

# ============================================================
# FIGURE 5 — Pipeline / workflow diagram
# ============================================================
fig, ax = plt.subplots(figsize=(6.8, 2.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")

def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                 facecolor=fc, edgecolor="#333333", lw=1.0, alpha=0.95))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=7.4, color="#111111")

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                  mutation_scale=11, color="#555555", lw=1.1))

box(1, 11, 17, 9, "Public data\nTDC (hERG)\nChEMBL (Nav1.5,\nCav1.2)", "#D9E8F5")
box(21, 11, 16, 9, "Clean & label\nIC50 \u2192 pIC50\npIC50 \u2265 6 = blocker", "#D9E8F5")
box(40, 11, 16, 9, "Featurize\n10 descriptors +\n2048-bit Morgan", "#FBE6CC")
box(59, 11, 17, 9, "Train models\nLR / RF / XGBoost\n(balanced weights)", "#FBE6CC")
box(79, 16, 19, 9, "Evaluate\nrandom & scaffold\nsplits", "#E2D4E8")
box(79, 4, 19, 9, "Audit\noverlap, starvation\nbootstrap 95% CI", "#E2D4E8")

arrow(18, 15.5, 21, 15.5)
arrow(37, 15.5, 40, 15.5)
arrow(56, 15.5, 59, 15.5)
arrow(76, 15.5, 79, 20.5)
arrow(76, 15.5, 79, 8.5)
fig.tight_layout()
fig.savefig("figures/fig5_pipeline.png", bbox_inches="tight")
plt.close(fig)

print("All figures written:")
for f in sorted(os.listdir("figures")):
    print("  figures/" + f)
