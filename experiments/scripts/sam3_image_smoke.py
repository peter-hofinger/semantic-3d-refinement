"""E2: SAM 3 image mode on one view; does it fit in 4 GB, and what does it find.

uv run python experiments/scripts/sam3_image_smoke.py
"""

import time
from pathlib import Path

import matplotlib as mpl
import numpy as np
import torch

mpl.use("Agg")
import matplotlib.pyplot as plt
from transformers import Sam3Model, Sam3Processor

from src.loaders import Views

SCENE, VIEW = "7b6477cb95", 6
PROMPTS = ["wall", "floor", "ceiling", "door", "window", "table", "chair",
           "monitor", "keyboard", "cabinet", "bookshelf", "whiteboard"]  # fmt: skip
OUT = Path("experiments/figures")
OUT.mkdir(parents=True, exist_ok=True)

views = Views.load(
    Path("data/processed/scannetpp") / SCENE / "views_n20_uniform_504.npz"
)
image = views.images[VIEW]

t0 = time.perf_counter()
model = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.bfloat16).to("cuda")
model = model.eval()
processor = Sam3Processor.from_pretrained("facebook/sam3")
torch.cuda.synchronize()
print(
    f"load: {time.perf_counter() - t0:.1f}s, "
    f"weights on GPU: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
)

torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
img_inputs = processor(images=image, return_tensors="pt").to("cuda", torch.bfloat16)
found = []
with torch.no_grad():
    vision = model.get_vision_features(pixel_values=img_inputs.pixel_values)
    for prompt in PROMPTS:
        text_inputs = processor(text=prompt, return_tensors="pt").to("cuda")
        out = model(vision_embeds=vision, **text_inputs)
        res = processor.post_process_instance_segmentation(
            out,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=img_inputs["original_sizes"].tolist(),
        )[0]
        for mask, score in zip(res["masks"], res["scores"], strict=True):
            found.append((prompt, float(score), mask.cpu().numpy().astype(bool)))
torch.cuda.synchronize()
print(
    f"inference ({len(PROMPTS)} prompts): {time.perf_counter() - t0:.1f}s, "
    f"peak GPU {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
)
for prompt in PROMPTS:
    s = [f"{sc:.2f}" for p, sc, _ in found if p == prompt]
    print(f"  {prompt:11s} {len(s)} found  scores {' '.join(s)}")

rng = np.random.default_rng(0)
over = image.astype(float).copy()
for _, _, m in sorted(found, key=lambda f: -f[2].sum()):  # large first, small on top
    over[m] = 0.45 * over[m] + 0.55 * rng.integers(40, 255, 3)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].imshow(image)
ax[0].set_title(f"{SCENE} view {VIEW}")
ax[1].imshow(over.astype(np.uint8))
ax[1].set_title(f"SAM 3 image mode: {len(found)} masks")
for p, sc, m in found:
    ys, xs = np.nonzero(m)
    if len(xs):
        ax[1].text(
            xs.mean(),
            ys.mean(),
            f"{p} {sc:.2f}",
            fontsize=7,
            color="white",
            ha="center",
            bbox={"facecolor": "black", "alpha": 0.5, "pad": 1},
        )
for a in ax:
    a.axis("off")
fig.tight_layout()
fig.savefig(
    OUT / f"e02_sam3_image_smoke_{SCENE}_v{VIEW}.jpg",
    dpi=80,
    pil_kwargs={"quality": 85},
)
print("figure ->", OUT)
