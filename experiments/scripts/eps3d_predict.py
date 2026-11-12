"""Run EPS3D's geometry (Zhu et al., ICML 2026) on one scene's cached views, per pixel.

EPS3D lives in `data/third_party/eps3d` (Runsong123/EPS3D @ 147f774, Apache-2.0) with
its own environment; its released checkpoint (`EPS3D/model.safetensors`, which freezes
AnySplat's appearance head, unlike the paper). Its points are what its encoder unprojects
before it voxelizes them into Gaussians (`EncoderEPS3D.forward`, `pred_head_type:
depth`): the VGGT aggregator, camera head and depth head, each pixel's depth along the
predicted camera, under the same bfloat16 autocast. Only those three modules are built,
their weights loaded strictly from the checkpoint; the semantic and instance heads do
not feed the geometry. Our 504 px views go in as they are: both sides are multiples of
VGGT's 14 px patch, so no crop or resize, and every pixel gets a point. EPS3D's own
ScanNet evaluation runs at 256 x 256; a square crop of our 3:2 views would drop a third
of each view.

Writes `<out>`: points (V, H, W, 3) in its first camera's frame, confidence (V, H, W),
seconds.

data/third_party/eps3d/.venv/bin/python experiments/scripts/eps3d_predict.py \
    <views.npz> <model.safetensors> <out.npz>
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, "data/third_party/eps3d")
from src.model.encoder.vggt.heads.camera_head import CameraHead
from src.model.encoder.vggt.heads.dpt_head import DPTHead
from src.model.encoder.vggt.models.aggregator import Aggregator
from src.model.encoder.vggt.utils.geometry import (
    batchify_unproject_depth_map_to_point_map,
)
from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri

INTERMEDIATE = [4, 11, 17, 23]  # the checkpoint's `intermediate_layer_idx`
EMBED = 1024


def load(module: torch.nn.Module, weights: dict, prefix: str) -> torch.nn.Module:
    """Load the checkpoint's `encoder.<prefix>.*` weights into module, strictly."""
    head = f"encoder.{prefix}."
    module.load_state_dict(
        {k[len(head) :]: v for k, v in weights.items() if k.startswith(head)}
    )
    return module.eval()


def main() -> None:
    """Predict per-pixel points, save."""
    views_path, weights_path, out = map(Path, sys.argv[1:4])
    images = np.load(views_path)["images"]
    n, h, w, _ = images.shape
    assert h % 14 == 0, h
    assert w % 14 == 0, w
    device = torch.device("cuda")
    weights = load_file(str(weights_path))
    aggregator = load(
        Aggregator(img_size=518, patch_size=14, embed_dim=EMBED), weights, "aggregator"
    )
    camera_head = load(CameraHead(dim_in=2 * EMBED), weights, "camera_head")
    depth_head = load(
        DPTHead(
            dim_in=2 * EMBED, output_dim=2, activation="exp", conf_activation="expp1"
        ),
        weights,
        "depth_head",
    )
    for m in (aggregator, camera_head, depth_head):
        m.to(device)
    image = (
        torch.from_numpy(images).permute(0, 3, 1, 2)[None].float().div(255).to(device)
    )
    t0 = time.perf_counter()
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            tokens, start = aggregator(
                image.to(torch.bfloat16), intermediate_layer_idx=INTERMEDIATE
            )
        with torch.amp.autocast("cuda", enabled=False):
            pose = camera_head(tokens)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose, image.shape[-2:])
            depth, conf = depth_head(tokens, images=image, patch_start_idx=start)
            points = batchify_unproject_depth_map_to_point_map(
                depth, extrinsic, intrinsic
            )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    points = points[0].float().cpu().numpy()
    assert points.shape == (n, h, w, 3), points.shape
    np.savez_compressed(
        out,
        points=points.astype(np.float32),
        confidence=conf[0].float().cpu().numpy().astype(np.float32),
        seconds=seconds,
    )
    print(f"{n} views, {seconds:.1f} s -> {out}")


if __name__ == "__main__":
    main()
