"""E12: best SAM 3 detection score per prompt and view, without a threshold.

Why did the floor synonyms (carpet, rug) add no mask: no detection, or below 0.5?

HF_HUB_OFFLINE=1 uv run python experiments/scripts/sam3_prompt_scores.py <scene> <prompts...>
"""

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import Sam3Model, Sam3Processor

from src.loaders import Views

scene, prompts = sys.argv[1], sys.argv[2:]
views = Views.load(
    Path("data/processed/scannetpp") / scene / "views_n20_uniform_504.npz"
)
model = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.bfloat16).to("cuda")
model = model.eval()
processor = Sam3Processor.from_pretrained("facebook/sam3")

best = np.zeros((len(prompts), len(views)))
for i, image in enumerate(views.images):
    inputs = processor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        vision = model.get_vision_features(
            pixel_values=inputs.pixel_values.to(torch.bfloat16)
        )
        for k, prompt in enumerate(prompts):
            text = processor(text=prompt, return_tensors="pt").to("cuda")
            res = processor.post_process_instance_segmentation(
                model(vision_embeds=vision, **text),
                threshold=0.0,
                mask_threshold=0.5,
                target_sizes=inputs["original_sizes"].tolist(),
            )[0]
            if len(res["scores"]):
                best[k, i] = res["scores"].float().max().item()

print(f"{scene}: best score per view (threshold 0.5 keeps a mask)")
print(f"  {'prompt':10s} " + " ".join(f"{i:4d}" for i in range(len(views))))
for k, prompt in enumerate(prompts):
    row = " ".join(f"{s:4.2f}" for s in best[k])
    print(f"  {prompt:10s} {row}   views >= 0.5: {(best[k] >= 0.5).sum()}")
