"""E19 figure: top100 / VLM / union, scores and which object bleeding pixels get.

Reads the outputs of e19_union.py and e18_boundaries.py (one per vocabulary):
uv run python experiments/scripts/e19_figure.py
"""

import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("experiments/results")
FIGURE = Path("experiments/figures/e19_union.jpg")
SCENES = ("kitchen", "office")
VOCABULARIES = {
    "top100": "e18_boundaries.txt",
    "VLM": "e19_boundaries_vlm.txt",
    "top100 + VLM": "e19_boundaries_union.txt",
}
SCORES = {"pixel recall": 7, "precision": 8, "cross-class leakage": 10, "merges": 14}

# scores: working setting ((b) on), all classes
scores = {}
for line in (RESULTS / "e19_union.txt").read_text(encoding="utf-8").splitlines():
    cells = [c.strip() for c in line.strip("|").split("|")]
    if len(cells) == 15 and cells[2] == "on" and cells[3] == "all":  # noqa: PLR2004
        scores[cells[0], cells[1]] = {k: float(cells[i]) for k, i in SCORES.items()}

# bleeding pixels at confirmed GT depth edges: right / other object / no mask
bleeding = {}
for vocab, name in VOCABULARIES.items():
    text = (RESULTS / name).read_text(encoding="utf-8")
    rows = re.findall(r"^\| bleeding \| \d+ \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", text, re.M)
    # per scene: all edges, then confirmed edges
    for k, scene in enumerate(SCENES):
        bleeding[scene, vocab] = [float(v) for v in rows[2 * k + 1]]

COLORS = {"top100": "#2a78d6", "VLM": "#eb6834", "top100 + VLM": "#3a9a5b"}
fig, axes = plt.subplots(1, len(SCORES) + 1, figsize=(17, 3.8))
x = np.arange(len(SCENES))
width = 0.26
for ax, metric in zip(axes, SCORES, strict=False):
    for k, vocab in enumerate(VOCABULARIES):
        vals = [scores[s, vocab][metric] for s in SCENES]
        ax.bar(x + (k - 1) * width, vals, width * 0.95, color=COLORS[vocab], label=vocab)
    ax.set_xticks(x, SCENES)
    lower = metric in ("cross-class leakage", "merges")
    ax.set_title(f"{metric}\n({'lower' if lower else 'higher'} is better)", fontsize=9)
axes[3].legend(fontsize=8, frameon=False, loc="upper left")  # kitchen bars are low

ax = axes[-1]
labels, right, other, none = [], [], [], []
for scene in SCENES:
    for vocab in VOCABULARIES:
        labels.append(f"{scene}\n{vocab}")
        r, o, n = bleeding[scene, vocab]
        right.append(r)
        other.append(o)
        none.append(n)
y = np.arange(len(labels))[::-1]
ax.barh(y, right, color="#3a9a5b", label="right object")
ax.barh(y, other, left=right, color="#d64545", label="other object")
ax.barh(y, none, left=np.add(right, other), color="#c9c7c0", label="no mask")
ax.set_yticks(y, labels, fontsize=7)
ax.set_xlim(0, 1)
ax.set_title("bleeding pixels: SAM 3 mask\n(confirmed GT depth edges)", fontsize=9)
ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.08), ncol=3)
for ax in axes:
    ax.grid(axis="x" if ax is axes[-1] else "y", color="#e5e4df", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
fig.suptitle(
    "E19: prompt vocabulary top100 / VLM (Qwen3-VL-2B) / both in one run. Working setting "
    "(t0.3, cross-class 0.7, (a) vf 0.25, (b) multi-view filter), all classes",
    fontsize=10,
)
fig.tight_layout()
fig.savefig(FIGURE, dpi=130, pil_kwargs={"quality": 90})
print(f"figure: {FIGURE}")
