"""Run IGGT's geometry (Li et al., 2025) on one scene's cached views, per pixel.

IGGT lives in `data/third_party/iggt` (lifuguan/IGGT_official @ bec6604; the README
says MIT, there is no licence file) with its own environment; its released checkpoint
(Hugging Face lifuguan/IGGT_official @ 12aa6fa, `iggt_checkpoint.pth`). Its points are
the ones its demo builds (`demo.py`, `_post_process_predictions`): each pixel's depth
from the depth head, unprojected along the camera the camera head predicts. The three
modules run as `IGGT.forward` runs them, the aggregator under the demo's bfloat16
autocast, both heads in float32; the point, track and instance heads, which the points
do not read, are dropped before the model goes to the GPU. Its aggregator keeps the
tokens of all 24 layers (EPS3D's keeps the 4 the heads read), which does not fit next to
float32 weights in 10 GB, so the aggregator's weights are held in bfloat16: under the
autocast every matmul casts them to bfloat16 anyway, so only its LayerNorm parameters
and embedding tokens lose precision. Our ETH3D views are 504 x 336, the demo's own input size, so
every pixel gets a point without a crop or resize.

Two shims, both for imports the released code cannot satisfy here, neither touching the
geometry: basicsr 1.4.2 imports `torchvision.transforms.functional_tensor`, which
torchvision 0.17 renamed to `functional` (the known basicsr incompatibility); and the
attention block takes apex's FusedRMSNorm and otherwise falls back to `src.model.norm`,
which the release does not contain. apex needs a CUDA build this machine cannot do;
its FusedRMSNorm is x / sqrt(mean(x^2) + 1e-5) * weight, which is `torch.nn.RMSNorm`
with eps 1e-5 and the same `weight` parameter, so that stands in for the fallback.

Writes `<out>`: points (V, H, W, 3) in its first camera's frame, confidence (V, H, W),
seconds.

data/third_party/iggt/.venv/bin/python experiments/scripts/iggt_predict.py \
    <views.npz> <iggt_checkpoint.pth> <out.npz>
"""

import functools
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional

sys.modules["torchvision.transforms.functional_tensor"] = (
    torchvision.transforms.functional
)
for _name in ("src", "src.model", "src.model.norm"):
    sys.modules[_name] = types.ModuleType(_name)
sys.modules["src.model.norm"].RMSNorm = functools.partial(torch.nn.RMSNorm, eps=1e-5)
sys.path.insert(0, "data/third_party/iggt")
from iggt.models.vggt import IGGT
from iggt.utils.geometry import unproject_depth_map_to_point_map
from iggt.utils.pose_enc import pose_encoding_to_extri_intri

GEOMETRY = ("aggregator.", "camera_head.", "depth_head.")  # the modules the points use


def main() -> None:
    """Predict per-pixel points, save."""
    views_path, weights_path, out = map(Path, sys.argv[1:4])
    images = np.load(views_path)["images"]
    n, h, w, _ = images.shape
    assert h % 14 == 0, h
    assert w % 14 == 0, w
    device = torch.device("cuda")
    model = IGGT()
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    own = model.state_dict()
    geometry = [k for k in own if k.startswith(GEOMETRY)]
    missing = [k for k in geometry if k not in state or state[k].shape != own[k].shape]
    assert not missing, missing[:5]
    model.load_state_dict(state, strict=False)
    for unused in ("point_head", "track_head", "part_adaptor", "part_head"):
        setattr(model, unused, None)
    model = model.eval().to(device)
    model.aggregator.to(torch.bfloat16)
    image = torch.from_numpy(images).permute(0, 3, 1, 2)[None].float().div(255)
    image = image.to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            tokens, start = model.aggregator(image)
        with torch.amp.autocast("cuda", enabled=False):
            pose = model.camera_head(tokens)[-1]
            depth, conf = model.depth_head(tokens, images=image, patch_start_idx=start)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose, (h, w))
    depth = depth[0].float().cpu().numpy()
    points = unproject_depth_map_to_point_map(
        depth, extrinsic[0].float().cpu().numpy(), intrinsic[0].float().cpu().numpy()
    )
    assert points.shape == (n, h, w, 3), points.shape
    np.savez_compressed(
        out,
        points=points.astype(np.float32),
        confidence=conf[0].float().cpu().numpy().astype(np.float32),
        seconds=seconds,
    )
    print(
        f"{n} views, {len(geometry)} geometry tensors loaded, {seconds:.1f} s -> {out}"
    )


if __name__ == "__main__":
    main()
