"""Run PanSt3R (Zust et al., ICCV 2025) on one scene's cached views, per pixel.

PanSt3R lives in `data/third_party/panst3r` (naver/panst3r @ 0253909, PanSt3R
Non-Commercial License) with its own environment, MUSt3R pinned at naver/must3r @
0c0d76b (the last commit before PanSt3R's, as the head needs torch >= 2.14). Checkpoint
v2 (`panst3r_v2_512_5ds.pth`, MD5 6a7a47a2...), its default post-processing
(standard v2), every view a keyframe, the NYU40 names as the text vocabulary.
Its inputs are resized from 504 x 378 to 512 x 384 (same aspect, no crop) as its demo
does; its point maps (global frame) and panoptic ids come back to 504 x 378 by nearest
neighbour, so no point is interpolated across a depth edge. A segment's score, which v2
does not output, is its mean mask confidence.

Writes `<out>`: points (V, H, W, 3), confidence (V, H, W), instance_ids (V, H, W) with
-1 for none, class_ids / class_names / scores per instance, seconds.

data/third_party/panst3r/.venv/bin/python experiments/scripts/panst3r_predict.py \
    <views.npz> <checkpoint> <out.npz>
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from must3r.datasets import ImgNorm
from must3r.engine.inference import postprocess
from must3r.model import get_pointmaps_activation
from must3r.model.blocks.attention import (
    has_xformers,
    toggle_memory_efficient_attention,
)
from must3r.tools.image import get_resize_function
from panst3r import PanSt3R
from panst3r.engine import panoptic_inference_v2
from PIL import Image

NYU40 = (
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window",
    "bookshelf", "picture", "counter", "blinds", "desk", "shelves", "curtain",
    "dresser", "pillow", "mirror", "floor mat", "clothes", "ceiling", "books",
    "refrigerator", "television", "paper", "towel", "shower curtain", "box",
    "whiteboard", "person", "night stand", "toilet", "sink", "lamp", "bathtub", "bag",
    "otherstructure", "otherfurniture", "otherprop",
)  # fmt: skip
SIZE = 512  # the checkpoint's resolution


def nearest(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Resize (V, h', w', C) to (V, h, w, C) by nearest neighbour."""
    return F.interpolate(
        t.permute(0, 3, 1, 2).float(), size=(h, w), mode="nearest-exact"
    ).permute(0, 2, 3, 1)


def main() -> None:
    """Predict, resize back to the views' pixels, save."""
    views_path, weights, out = map(Path, sys.argv[1:4])
    images = np.load(views_path)["images"]
    n, h, w, _ = images.shape
    device = torch.device("cuda")
    toggle_memory_efficient_attention(enabled=has_xformers)
    model = PanSt3R.from_checkpoint(str(weights))
    model.panoptic_decoder.text_encoder.change_mode(fixed_vocab=False)
    model = model.eval().to(device)
    resize, _, _ = get_resize_function(SIZE, model.must3r_encoder.patch_size, h, w)
    imgs = [resize(ImgNorm(Image.fromarray(im))).to(device) for im in images]
    shape = torch.tensor([im.shape[-2:] for im in imgs], device=device)
    t0 = time.perf_counter()
    with torch.no_grad():
        maps, pan = model.forward_inference_multi_ar(
            imgs, shape, list(NYU40), num_keyframes=n, max_bs=1, outdevice="cpu"
        )
        pred = panoptic_inference_v2(
            pan["pred_logits"],
            pan["pred_masks"],
            shape.cpu().numpy(),
            label_mode=model.panoptic_decoder.label_mode,
            device="cpu",
            multi_ar=True,
        )[0]
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    activation = get_pointmaps_activation(model.must3r_decoder)
    x = [postprocess(m[0], pointmaps_activation=activation) for m in maps]
    points = nearest(torch.stack([o["pts3d"].float() for o in x]), h, w)
    conf = nearest(torch.stack([o["conf"].float() for o in x])[..., None], h, w)
    ids = nearest(torch.stack(pred["pan"])[..., None], h, w)[..., 0].long()
    mask_conf = nearest(torch.stack(pred["conf"])[..., None], h, w)[..., 0]
    segments = pred["segments_info"]
    ids = torch.where(ids > 0, ids, -1).int()
    scores = [float(mask_conf[ids == s["id"]].mean()) for s in segments]
    np.savez_compressed(
        out,
        points=points.numpy().astype(np.float32),
        confidence=conf[..., 0].numpy().astype(np.float32),
        instance_ids=ids.numpy(),
        class_ids=np.array([s["id"] for s in segments], np.int32),
        class_names=np.array([NYU40[s["category_id"]] for s in segments]),
        scores=np.array(scores, np.float32),
        seconds=seconds,
    )
    print(f"{n} views, {len(segments)} segments, {seconds:.1f} s -> {out}")


if __name__ == "__main__":
    main()
