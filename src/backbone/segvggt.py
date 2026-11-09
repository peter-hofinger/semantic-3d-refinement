"""SegVGGT (Qu et al., ECCV 2026) as backbone and segmenter in one forward pass.

SegVGGT is VGGT (Wang et al., CVPR 2025) with object queries: one pass over unposed
views gives VGGT's depth, confidence and cameras and, per query, a class and one mask
over all views, so its instances are linked across views by construction. Code and
weights are the authors' (IDEA-Research/SegVGGT, VGGT License, research use only) and
stay outside git under `data/third_party/SegVGGT`, pinned in `configs/segvggt.yaml`.
The ScanNet checkpoint's instance head has 18 object classes, no wall or floor.

Queries become one instance per pixel by Mask2Former's panoptic inference (Cheng et
al., CVPR 2022), whose query decoder SegVGGT's follows: keep a query if its best class
has probability p ≥ `object_threshold`, give each pixel to the kept query with the
largest p · m (m its mask probability), and drop a query left with less than
`overlap_threshold` of its own mask (m > 0.5). A kept instance scores p times its mean
mask probability inside the mask, SegVGGT's own instance score.

The cameras come from VGGT's pose encoding (field of view, centred principal point);
the ray map is the exact pinhole of that camera, so x = o + D · d holds per pixel.

The model runs in float32 throughout. VGGT's demo autocasts the trunk to bfloat16 and
keeps the heads in float32, but that split only holds on CUDA; float32 everywhere is
the same on every device and needs about 12 GB, more than a 10 GB card has.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F  # noqa: N812

from src.geometry import Geometry
from src.loaders import Views
from src.segmentation.base import NO_INSTANCE, Segmentation

log = logging.getLogger(__name__)

PATCH_SIZE = 14
# instance classes of the ScanNet checkpoint, in its order, spelled as NYU40 is
CLASSES = (
    "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "desk", "curtain", "refrigerator", "shower curtain",
    "toilet", "sink", "bathtub", "otherfurniture",
)  # fmt: skip


@dataclass(frozen=True)
class SegVGGTConfig:
    """Where the authors' code and weights live, and the panoptic thresholds."""

    repo: Path
    config: str  # Hydra config inside the repo
    checkpoint: Path
    object_threshold: float
    overlap_threshold: float

    @classmethod
    def load(cls, path: Path) -> SegVGGTConfig:
        """Read `configs/segvggt.yaml`."""
        raw = yaml.safe_load(path.read_text())
        return cls(
            Path(raw["repo"]),
            raw["config"],
            Path(raw["checkpoint"]),
            float(raw["object_threshold"]),
            float(raw["overlap_threshold"]),
        )


def to_cam_to_world(world_to_cam: np.ndarray) -> np.ndarray:
    """Invert (V, 3, 4) world-to-camera extrinsics into (V, 4, 4) camera-to-world."""
    assert world_to_cam.ndim == 3 and world_to_cam.shape[1:] == (3, 4)  # noqa: PLR2004
    rot, trans = world_to_cam[:, :, :3], world_to_cam[:, :, 3]
    c2w = np.tile(np.eye(4), (len(world_to_cam), 1, 1))
    c2w[:, :3, :3] = rot.transpose(0, 2, 1)
    c2w[:, :3, 3] = -np.einsum("vji,vj->vi", rot, trans)
    return c2w


def pinhole_rays(
    intrinsics: np.ndarray, cam_to_world: np.ndarray, height: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel ray origins and directions (V, H, W, 3), z-unit in the camera frame."""
    num = len(intrinsics)
    assert intrinsics.shape == (num, 3, 3) and cam_to_world.shape == (num, 4, 4)
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    pix = np.stack([u, v, np.ones_like(u)], axis=-1)
    cam = np.einsum("vij,hwj->vhwi", np.linalg.inv(intrinsics), pix)
    directions = np.einsum("vij,vhwj->vhwi", cam_to_world[:, :3, :3], cam)
    origins = np.broadcast_to(cam_to_world[:, None, None, :3, 3], directions.shape)
    return origins.astype(np.float32), directions.astype(np.float32)


def merge_queries(
    probs: np.ndarray, labels: np.ndarray, masks: np.ndarray, overlap_threshold: float
) -> tuple[np.ndarray, dict[int, int], dict[int, float]]:
    """Mask2Former's panoptic merge of kept queries: ids (V, H, W), labels, scores.

    `probs` (K,) are class probabilities, `labels` (K,) class indices, `masks`
    (K, V, H, W) mask probabilities. Ids are query indices into the K, NO_INSTANCE
    where no query covers the pixel.
    """
    num = len(probs)
    assert probs.shape == labels.shape == (num,) and masks.shape[0] == num
    assert np.all((masks >= 0) & (masks <= 1)), "masks are probabilities"
    ids = np.full(masks.shape[1:], NO_INSTANCE, dtype=np.int32)
    if num == 0:
        return ids, {}, {}
    own = masks > 0.5  # noqa: PLR2004
    winner = (probs[:, None, None, None] * masks).argmax(0)
    out_labels, out_scores = {}, {}
    for k in range(num):
        won = own[k] & (winner == k)
        if won.sum() == 0 or won.sum() < overlap_threshold * own[k].sum():
            continue
        ids[won] = k
        out_labels[k] = int(labels[k])
        out_scores[k] = float(probs[k] * masks[k][own[k]].mean())
    return ids, out_labels, out_scores


def load_model(cfg: SegVGGTConfig, device: torch.device) -> torch.nn.Module:
    """Load the authors' SegVGGT with the checkpoint's weights, in eval mode.

    Its per-frame attention readout (FADA) only feeds a training loss; with it off, the
    query cross-attention is fused and never holds a (heads, queries, tokens) matrix.
    """
    from hydra import compose, initialize_config_dir  # noqa: PLC0415
    from hydra.utils import instantiate  # noqa: PLC0415

    sys.path.insert(0, str(cfg.repo.resolve()))
    config = cfg.repo.resolve() / cfg.config
    with initialize_config_dir(version_base=None, config_dir=str(config.parent)):
        model = instantiate(compose(config_name=config.stem).model, _recursive_=False)
    state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state.get("model", state), strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith(("point_head", "track_head")) for k in missing), missing
    model.aggregator.supervise_attend_frame = False
    return model.to(device).eval()


def predict(
    model: torch.nn.Module, views: Views, cfg: SegVGGTConfig
) -> tuple[Geometry, Segmentation]:
    """Depth, cameras, pinhole rays and linked instances of all views in one pass."""
    from segvggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: PLC0415

    num, height, width = views.images.shape[:3]
    assert height % PATCH_SIZE == 0 and width % PATCH_SIZE == 0, (height, width)
    device = next(model.parameters()).device
    images = torch.from_numpy(views.images).permute(0, 3, 1, 2).float() / 255
    with torch.no_grad():
        out = model(images.to(device))
    w2c, k = pose_encoding_to_extri_intri(out["pose_enc"], (height, width))
    intrinsics = k[0].float().cpu().numpy()
    cam_to_world = to_cam_to_world(w2c[0].float().cpu().numpy())
    origins, directions = pinhole_rays(intrinsics, cam_to_world, height, width)
    geometry = Geometry(
        depth=out["depth"][0, ..., 0].float().cpu().numpy(),
        confidence=out["depth_conf"][0].float().cpu().numpy(),
        intrinsics=intrinsics.astype(np.float32),
        cam_to_world=cam_to_world.astype(np.float32),
        ray_origins=origins,
        ray_directions=directions,
    )

    class_probs = F.softmax(out["instance_labels"][0].float(), dim=-1)[:, :-1]
    probs, labels = class_probs.max(-1)
    keep = probs >= cfg.object_threshold
    logits = out["instance_maps"][0][keep].float()  # (K, V, h, w) at half resolution
    masks = F.interpolate(logits, size=(height, width), mode="bilinear").sigmoid()
    ids, kept_labels, scores = merge_queries(
        probs[keep].cpu().numpy(),
        labels[keep].cpu().numpy(),
        masks.cpu().numpy(),
        cfg.overlap_threshold,
    )
    classes = {i: CLASSES[c] for i, c in kept_labels.items()}
    segmentation = Segmentation(ids, classes, scores)
    assert geometry.depth.shape == ids.shape == (num, height, width)
    assert np.isfinite(geometry.depth).all() and np.all(geometry.depth > 0)
    return geometry, segmentation


def main() -> None:
    """Run SegVGGT on prepared views; save its geometry and instances next to them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("views", type=Path, nargs="+", help="prepared views (.npz)")
    parser.add_argument("--config", type=Path, default=Path("configs/segvggt.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = SegVGGTConfig.load(args.config)
    mps = torch.backends.mps.is_available()
    device = "cuda" if torch.cuda.is_available() else "mps" if mps else "cpu"
    model = load_model(cfg, torch.device(device))
    for path in args.views:
        views = Views.load(path)
        geometry, segmentation = predict(model, views, cfg)
        geometry.save(path.with_name(f"segvggt_{path.stem}.npz"))
        segmentation.save(path.with_name(f"segvggt-instances_{path.stem}.npz"))
        log.info(
            "%s: %d views, %d instances",
            views.scene_id,
            len(views),
            len(segmentation.classes),
        )


if __name__ == "__main__":
    main()
