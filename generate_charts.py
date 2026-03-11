"""
Generate supplementary README charts that are NOT produced by the notebooks.
Charts from notebooks (ROC, risk-coverage, uncertainty boxplot, PCA, etc.)
are extracted directly from .ipynb outputs — see extract_notebook_images.py.

This script only generates:
  1. pipeline.png — inference pipeline diagram
  2. dataset.png — dataset split & class distribution
  3. hyperparam_grid.png — hyperparameter search space visualization
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np
import os

OUT = "assets"
os.makedirs(OUT, exist_ok=True)

C_MC = "#2196F3"
C_VI = "#FF9800"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


# ═══════════════════════════════════════════════════
# 1. Pipeline Overview Diagram
# ═══════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 3.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis("off")

boxes = [
    (0.3,  1.0, "Raw Image\n(JPG/PNG)",       "#E3F2FD", "#1565C0"),
    (2.5,  1.0, "Xception\nBackbone\n(frozen)", "#E8F5E9", "#2E7D32"),
    (4.7,  1.0, "2048-d\nFeatures",            "#FFF3E0", "#E65100"),
    (6.9,  1.0, "Bayesian Head\n(MC / VI)\n× 50 passes", "#F3E5F5", "#7B1FA2"),
    (9.1,  1.0, "P(fake) ± σ\nUncertainty",   "#FFEBEE", "#C62828"),
]

for bx, by, text, fc, ec in boxes:
    rect = matplotlib.patches.FancyBboxPatch((bx - 0.8, by - 0.55), 1.6, 1.3,
            boxstyle="round,pad=0.12", facecolor=fc, edgecolor=ec, linewidth=2)
    ax.add_patch(rect)
    ax.text(bx, by + 0.1, text, ha="center", va="center", fontsize=9.5, fontweight="bold", color=ec)

for i in range(len(boxes) - 1):
    x_start = boxes[i][0] + 0.85
    x_end   = boxes[i+1][0] - 0.85
    ax.annotate("", xy=(x_end, 1.1), xytext=(x_start, 1.1),
                arrowprops=dict(arrowstyle="-|>", color="#424242", lw=2))

ax.set_title("Inference Pipeline", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
fig.savefig(f"{OUT}/pipeline.png")
plt.close()
print("pipeline.png")


# ═══════════════════════════════════════════════════
# 2. Dataset Split Visualization
# ═══════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

splits = ["Train\n(932)", "Validation\n(234)", "Test\n(786)"]
sizes  = [932, 234, 786]
colors_split = ["#42A5F5", "#66BB6A", "#FFA726"]
bars = ax1.bar(splits, sizes, color=colors_split, width=0.55, edgecolor="white", linewidth=1.5)
ax1.set_title("Dataset Split", fontweight="bold")
ax1.set_ylabel("Number of Samples")
ax1.grid(axis="y", alpha=0.3)
for bar in bars:
    h = bar.get_height()
    ax1.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 5), textcoords="offset points", ha="center", fontsize=11, fontweight="bold")

labels_cls = ["Fake (~61%)", "Real (~39%)"]
sizes_cls  = [61, 39]
colors_cls = ["#EF5350", "#66BB6A"]
wedges, texts, autotexts = ax2.pie(sizes_cls, labels=labels_cls, colors=colors_cls,
                                    autopct="%1.0f%%", startangle=90, textprops={"fontsize": 11})
for at in autotexts:
    at.set_fontweight("bold")
ax2.set_title("Class Distribution\n(consistent across splits)", fontweight="bold")

plt.tight_layout()
fig.savefig(f"{OUT}/dataset.png")
plt.close()
print("dataset.png")


# ═══════════════════════════════════════════════════
# 3. Hyperparameter Search Grid Visualization
# ═══════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

kl_weights = [0.1, 0.5, 1.0, 2.0]
prior_stds = [0.5, 1.0, 2.0]
for ps in prior_stds:
    for kl in kl_weights:
        ax1.scatter(kl, ps, s=200, color=C_VI, edgecolors="white", linewidth=1.5, zorder=5)
ax1.set_xlabel("kl_weight")
ax1.set_ylabel("prior_std")
ax1.set_title("VI Hyperparameter Grid\n(12 configurations × 50 epochs)", fontweight="bold")
ax1.set_xticks(kl_weights)
ax1.set_yticks(prior_stds)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.1, 2.3)
ax1.set_ylim(0.2, 2.3)

dropout_rates = [0.1, 0.3, 0.5]
hidden_sizes  = [128, 256, 512]
for hs in hidden_sizes:
    for dr in dropout_rates:
        ax2.scatter(dr, hs, s=200, color=C_MC, edgecolors="white", linewidth=1.5, zorder=5)
ax2.set_xlabel("dropout_rate")
ax2.set_ylabel("hidden_size")
ax2.set_title("MC Dropout Hyperparameter Grid\n(9 configurations × 30 epochs)", fontweight="bold")
ax2.set_xticks(dropout_rates)
ax2.set_yticks(hidden_sizes)
ax2.grid(True, alpha=0.3)

fig.suptitle("Hyperparameter Search Space", fontsize=14, fontweight="bold", y=1.04)
plt.tight_layout()
fig.savefig(f"{OUT}/hyperparam_grid.png")
plt.close()
print("hyperparam_grid.png")

print("\nDone. 3 supplementary charts generated in ./assets/")
