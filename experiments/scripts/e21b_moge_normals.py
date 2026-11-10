"""E21b: MoGe-3's normals for prepared views, cached as DSINE's are (`moge3_<views>.npz`).

MoGe-3 (Wang et al. 2026, arXiv 2607.17967, MIT) runs per image; it is given each view's
horizontal field of view from the backbone's fitted intrinsics, as DSINE is given the
intrinsics, so both normal maps refer to the same camera. Its normals are in OpenCV's
camera frame; `to_world` fits the axis signs to the depth either way. MoGe-3 needs Triton
(Linux or Windows) and its own pinned dependencies, so this script runs in MoGe's own
environment and imports neither `src` nor anything outside it and numpy:

  uv run --project data/third_party/MoGe python experiments/scripts/e21b_moge_normals.py \
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
    """Write MoGe-3's camera-frame normals (V, H, W, 3) next to every views file."""
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
        assert images.shape[-1] == 3, images.shape
        assert k.shape == (len(images), 3, 3), k.shape
        width = images.shape[2]
        out = []
        with torch.no_grad():
            for image, ki in zip(images, k, strict=True):
                x = torch.from_numpy(image).cuda().permute(2, 0, 1).float() / 255
                fov_x = math.degrees(2 * math.atan(width / (2 * ki[0, 0])))
                pred = model.infer(x, fov_x=fov_x, refine_steps=args.refine_steps)
                out.append(pred["normal"].float().cpu().numpy())
        normals = np.stack(out).astype(np.float32)
        assert normals.shape == images.shape, normals.shape
        assert np.isfinite(normals).all()
        dest = path.with_name(f"moge3_{path.stem}.npz")
        np.savez_compressed(dest, normals=normals)
        print(f"{path.parent.name}: {len(normals)} views -> {dest}")


if __name__ == "__main__":
    main()
