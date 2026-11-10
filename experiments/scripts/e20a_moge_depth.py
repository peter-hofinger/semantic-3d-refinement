"""E20a: MoGe-3's depth for prepared views, cached next to them (`moge3depth_<views>.npz`).

MoGe-3 (Wang et al. 2026, arXiv 2607.17967, MIT) predicts metric depth from one image; it
is given each view's horizontal field of view from the backbone's fitted intrinsics, so its
depth belongs to the backbone's camera. Pixels MoGe-3 marks invalid (sky, far background)
are stored as 0 with `valid` false. MoGe-3 needs Triton (Linux or Windows) and its own
pinned dependencies, so this script runs in MoGe's own environment and imports nothing
outside it and numpy:

  uv run --project data/third_party/MoGe python experiments/scripts/e20a_moge_depth.py \
      data/processed/hypersim/<scene>/views_n20_uniform_504.npz ...
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from moge.model.v3 import MoGeModel

CHECKPOINT = "Ruicheng/moge-3-vitl"


def main() -> None:
    """Write MoGe-3's depth (V, H, W) and valid mask next to every views file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("views", type=Path, nargs="+", help="prepared views (.npz)")
    parser.add_argument("--backbone", default="dejaview_k16")
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--refine-steps", type=int, default=3)
    args = parser.parse_args()
    model = MoGeModel.from_pretrained(args.checkpoint).to("cuda").eval()
    for path in args.views:
        images = np.load(path)["images"]
        k = np.load(path.with_name(f"{args.backbone}_{path.stem}.npz"))["intrinsics"]
        assert images.dtype == np.uint8, images.dtype
        assert k.shape == (len(images), 3, 3), k.shape
        width = images.shape[2]
        depths, valids = [], []
        with torch.no_grad():
            for image, ki in zip(images, k, strict=True):
                x = torch.from_numpy(image).cuda().permute(2, 0, 1).float() / 255
                fov_x = math.degrees(2 * math.atan(width / (2 * ki[0, 0])))
                pred = model.infer(x, fov_x=fov_x, refine_steps=args.refine_steps)
                depth = pred["depth"].float().cpu().numpy()
                valid = (
                    pred["mask"].bool().cpu().numpy() & np.isfinite(depth) & (depth > 0)
                )
                depths.append(np.where(valid, depth, 0).astype(np.float32))
                valids.append(valid)
        depth, valid = np.stack(depths), np.stack(valids)
        assert depth.shape == images.shape[:3], depth.shape
        dest = path.with_name(f"moge3depth_{path.stem}.npz")
        np.savez_compressed(dest, depth=depth, valid=valid)
        print(
            f"{path.parent.name}: {len(depth)} views, {valid.mean():.1%} valid -> {dest}"
        )


if __name__ == "__main__":
    main()
