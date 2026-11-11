"""Run PE3R (Hu et al., CVPR 2026) on one scene's cached views, per pixel.

PE3R lives in `data/third_party/pe3r` (hujiecpp/PE3R @ a6c3683, no clear licence) with
its own environment; MASt3R, SAM 2.1, SAM and SigLIP come from the Hugging Face Hub, its
two own checkpoints from its `checkpoints` release. It is training-free and uses its
segments in the geometry itself: MASt3R sees each view with every segment painted 95 %
in its mean colour, and the global alignment is tuned per segment. This runs its own
`get_reconstructed_scene` with its demo's defaults (all pairs, 300 iterations of the
linear schedule; the GLB export it writes on the way goes to a temporary folder) and
takes `scene.get_pts3d()`, every pixel's point. Its loader resizes our 504 x 336 views
to 512 x 341 and centre-crops 512 x 336 (`dust3r.utils.image.load_images`); the points
come back to our pixels by `e26a_pixels.back_to_views`.

PE3R keeps all five models on the GPU, which does not fit in 10 GB. Its segmentation
(`get_cog_feats`) never reads MASt3R and its reconstruction never reads the segmentation
models, so `get_cog_feats` is wrapped to hold MASt3R on the CPU while it runs and the
segmentation models there afterwards: the same computation, other placement.

Writes `<out>`: points (V, H, W, 3), confidence (V, H, W), seconds.

data/third_party/pe3r/.venv/bin/python experiments/scripts/pe3r_predict.py \
    <views.npz> <out.npz>
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from e26a_pixels import back_to_views
from PIL import Image

ROOT = Path("data/third_party/pe3r").resolve()
SIZE = 512  # MASt3R's resolution in PE3R's loader
PATCH = 16


def to_model(h: int, w: int) -> tuple[np.ndarray, tuple[int, int]]:
    """PE3R's map of our pixel coordinates onto MASt3R's (resize, centre crop); shape."""
    scale = SIZE / max(h, w)
    rw, rh = round(w * scale), round(h * scale)
    cx, cy = rw // 2, rh // 2
    half_w, half_h = (2 * cx // PATCH) * PATCH // 2, (2 * cy // PATCH) * PATCH // 2
    a = np.array(
        [[rw / w, 0, -(cx - half_w)], [0, rh / h, -(cy - half_h)], [0, 0, 1.0]]
    )
    return a, (2 * half_h, 2 * half_w)


def main() -> None:
    """Reconstruct with PE3R, map back onto our pixels, save."""
    views_path, out = (Path(a).resolve() for a in sys.argv[1:3])
    images = np.load(views_path)["images"]
    n, h, w, _ = images.shape
    os.chdir(ROOT)  # PE3R's checkpoint paths are relative to its repository
    sys.path.insert(0, str(ROOT))
    sys.path.append(str(ROOT / "modules"))  # as pe3r_demo.py does: dust3r, mast3r
    # after the chdir: models.py puts './modules/ultralytics' on the path at import
    from modules.pe3r import demo  # noqa: PLC0415
    from modules.pe3r.models import Models  # noqa: PLC0415

    device = "cuda"
    models = Models(device=device)
    segment = demo.get_cog_feats

    def placed(images: object, pe3r: object) -> object:
        pe3r.mast3r.cpu()
        torch.cuda.empty_cache()
        out = segment(images, pe3r)
        for m in (pe3r.sam2, pe3r.mobilesamv2, pe3r.siglip, pe3r.yolov8.model):
            m.cpu()
        torch.cuda.empty_cache()
        pe3r.mast3r.to(device)
        return out

    demo.get_cog_feats = placed
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for i, im in enumerate(images):
            files.append(f"{tmp}/{i:03d}.png")
            Image.fromarray(im).save(files[-1])
        t0 = time.perf_counter()
        scene, _, _ = demo.get_reconstructed_scene(
            outdir=tmp,
            pe3r=models,
            device=device,
            silent=True,
            filelist=files,
            schedule="linear",
            niter=300,
            min_conf_thr=3.0,
            as_pointcloud=True,
            mask_sky=False,
            clean_depth=True,
            transparent_cams=True,
            cam_size=0.05,
            scenegraph_type="complete",
            winsize=1,
            refid=0,
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
    pts = np.stack([p.detach().cpu().numpy() for p in scene.get_pts3d()])
    conf = np.stack([c.detach().cpu().numpy() for c in scene.im_conf])[..., None]
    a, shape = to_model(h, w)
    assert pts.shape[1:3] == shape, (pts.shape, shape)
    points = back_to_views(pts, a, h, w)
    np.savez_compressed(
        out,
        points=points,
        confidence=back_to_views(conf, a, h, w)[..., 0],
        seconds=seconds,
    )
    print(f"{n} views, {seconds:.1f} s -> {out}")


if __name__ == "__main__":
    main()
