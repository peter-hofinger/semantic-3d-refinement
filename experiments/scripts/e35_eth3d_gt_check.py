"""E35 ETH3D loader: check the ground truth before any score is computed on it.

Per scene, on the views `eth3d-prepare` picked:
1. Camera model: project the SfM points of ETH3D's distorted calibration (points3D.txt)
   with our THIN_PRISM_FISHEYE reimplementation and compare with the 2D observations
   COLMAP stored for them: median reprojection error in pixels of the 6048 x 4032 image.
2. Depth convention: at the observations that land on a laser dot, GT depth over the
   point's camera z and over its distance to the camera centre, by image radius. z-depth
   gives 1 at every radius; distance along the ray falls towards the corners.
3. Multi-view consistency of the prepared GT (504 px): lift view i's GT, project into
   view j, median relative depth error where the point is visible (within 5%, as E34).
4. Coverage: share of prepared pixels with GT depth.
Figure: one view per scene, image and GT depth. ETH3D data: CC BY-NC-SA 4.0
(www.eth3d.net), Schöps et al., CVPR 2017.

uv run python experiments/scripts/e35_eth3d_gt_check.py > experiments/results/e35_eth3d_gt_check.txt
"""

import itertools
import tempfile
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from src.loaders.eth3d.prepare import DEPTH, DISTORTED_CALIB, extract, output_path
from src.loaders.eth3d.scene import (
    distort,
    load_config,
    quaternion_to_rotation,
    read_cameras,
)
from src.loaders.ground_truth import GroundTruth
from src.loaders.views import Views

VISIBLE = 0.05  # relative depth gap above which a point counts as occluded in view j
DEPTH_VIEWS = 5  # views per scene whose full-resolution depth maps are read for check 2
RADII = (0, 1000, 2000, 4000)  # image radius bins in distorted pixels
FIGURE = Path("experiments/figures/e35_eth3d_gt_check.jpg")


def read_observations(
    images_txt: str,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """Image name -> (world-to-camera R, t, observations x y point3D_id, camera id)."""
    lines = [line for line in images_txt.splitlines() if not line.startswith("#")]
    out = {}
    for header, obs in zip(lines[0::2], lines[1::2], strict=False):
        f = header.split()
        r = quaternion_to_rotation(np.array(f[1:5], dtype=np.float64))
        t = np.array(f[5:8], dtype=np.float64)
        o = np.array(obs.split(), dtype=np.float64).reshape(-1, 3)
        out[Path(f[9]).name] = (r, t, o[o[:, 2] >= 0], int(f[8]))
    return out


def read_points(points_txt: str) -> dict[int, np.ndarray]:
    """SfM point id -> world position."""
    out = {}
    for line in points_txt.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        out[int(f[0])] = np.array(f[1:4], dtype=np.float64)
    return out


def pair_error(
    depth_i: np.ndarray,
    k_i: np.ndarray,
    pose_i: np.ndarray,
    depth_j: np.ndarray,
    k_j: np.ndarray,
    pose_j: np.ndarray,
) -> tuple[float, float]:
    """Median relative depth error and overlap of view i's GT seen from view j (E34)."""
    h, w = depth_i.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    valid = depth_i > 0
    rays = (
        np.stack([u[valid], v[valid], np.ones(valid.sum())], -1) @ np.linalg.inv(k_i).T
    )
    world = (rays * depth_i[valid, None]) @ pose_i[:3, :3].T + pose_i[:3, 3]
    cam = (world - pose_j[:3, 3]) @ pose_j[:3, :3]
    front = cam[:, 2] > 0
    pix = cam[front] @ k_j.T
    pix = pix[:, :2] / pix[:, 2:]
    z = cam[front, 2]
    x, y = np.floor(pix[:, 0]).astype(int), np.floor(pix[:, 1]).astype(int)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    z_j = depth_j[y[inside], x[inside]]
    has_gt = z_j > 0
    rel = np.abs(z[inside][has_gt] - z_j[has_gt]) / z_j[has_gt]
    seen = rel < VISIBLE
    if seen.sum() < 100:
        return float("nan"), 0.0
    return float(np.median(rel[seen])), float(seen.sum() / valid.sum())


def main() -> None:
    """Print the four checks per scene and write the figure."""
    cfg = load_config(Path("configs/eth3d.yaml"))
    figure_rows = []
    print(
        "scene | kind | views | GT px | reproj. px (median, 90%) | n obs | depth/z by radius"
        " | depth/range by radius | MV pairs | MV median err | MV overlap"
    )
    for scene, kind in cfg.scenes.items():
        path = output_path(cfg, scene)
        views = Views.load(path)
        gt = GroundTruth.load(path.with_name(f"gt_{path.stem}.npz"))
        names = [f"{n}.JPG" for n in views.names]
        with tempfile.TemporaryDirectory() as tmp:
            calib = [
                f"{DISTORTED_CALIB}/{f}"
                for f in ("cameras.txt", "images.txt", "points3D.txt")
            ]
            files = extract(cfg.archive(scene, "dslr_jpg"), calib, Path(tmp))
            cams = read_cameras(files[calib[0]].read_text())
            obs = read_observations(files[calib[1]].read_text())
            points = read_points(files[calib[2]].read_text())
            pick = names[:: max(1, len(names) // DEPTH_VIEWS)][:DEPTH_VIEWS]
            depths = extract(
                cfg.archive(scene, "dslr_depth"),
                [f"{DEPTH}/{n}" for n in pick],
                Path(tmp),
            )
            reproj, ratios = [], []
            for name in names:
                r, t, o, cam_id = obs[name]
                cam = cams[cam_id]
                fx, fy, cx, cy = cam.params[:4]
                xw = np.stack([points[int(p)] for p in o[:, 2]])
                xc = xw @ r.T + t
                ud, vd = distort(
                    xc[:, 0] / xc[:, 2], xc[:, 1] / xc[:, 2], cam.params[4:]
                )
                reproj.append(np.hypot(ud * fx + cx - o[:, 0], vd * fy + cy - o[:, 1]))
                if name in pick:
                    d = np.fromfile(depths[f"{DEPTH}/{name}"], dtype=np.float32)
                    d = d.reshape(cam.height, cam.width)
                    col, row = (
                        np.floor(o[:, 0]).astype(int),
                        np.floor(o[:, 1]).astype(int),
                    )
                    ok = (
                        (col >= 0) & (col < cam.width) & (row >= 0) & (row < cam.height)
                    )
                    val = np.zeros(len(o))
                    val[ok] = d[row[ok], col[ok]]
                    hit = np.isfinite(val) & (val > 0)
                    radius = np.hypot(o[:, 0] - cx, o[:, 1] - cy)
                    ratios.append(
                        np.stack(
                            [val / xc[:, 2], val / np.linalg.norm(xc, axis=1), radius],
                            1,
                        )[hit]
                    )
        reproj = np.concatenate(reproj)
        ratios = np.concatenate(ratios)
        by_z, by_range = [], []
        for lo, hi in itertools.pairwise(RADII):
            m = (ratios[:, 2] >= lo) & (ratios[:, 2] < hi)
            by_z.append(f"{np.median(ratios[m, 0]):.4f}" if m.sum() >= 10 else "-")
            by_range.append(f"{np.median(ratios[m, 1]):.4f}" if m.sum() >= 10 else "-")

        n = len(views)
        errs, ovl = [], []
        k, poses = (
            views.intrinsics.astype(np.float64),
            views.cam_to_world.astype(np.float64),
        )
        for i in range(n):
            for j in range(n):
                if i != j:
                    e, o_ = pair_error(
                        gt.depth[i], k[i], poses[i], gt.depth[j], k[j], poses[j]
                    )
                    errs.append(e)
                    ovl.append(o_)
        errs, ovl = np.array(errs), np.array(ovl)
        hit = np.isfinite(errs)
        print(
            f"{scene} | {kind} | {n} | {100 * (gt.depth > 0).mean():.0f}% |"
            f" {np.median(reproj):.2f}, {np.percentile(reproj, 90):.2f} | {len(reproj)} |"
            f" {' '.join(by_z)} | {' '.join(by_range)} | {hit.sum()}/{len(errs)} |"
            f" {100 * np.median(errs[hit]):.3f}% | {100 * ovl[hit].mean():.1f}%"
        )
        mid = n // 2
        figure_rows.append(
            (scene, kind, views.names[mid], views.images[mid], gt.depth[mid])
        )
    print(f"radius bins (distorted px): {RADII}")

    fig, axes = plt.subplots(7, 4, figsize=(16, 19.5))
    for ax in axes.flat:
        ax.axis("off")
    for idx, (scene, kind, name, image, depth) in enumerate(figure_rows):
        a_img, a_dep = axes.flat[2 * idx], axes.flat[2 * idx + 1]
        a_img.imshow(image)
        a_img.set_title(f"{scene} ({kind}), {name}", fontsize=10)
        shown = np.where(depth > 0, depth, np.nan)
        im = a_dep.imshow(
            shown, cmap="viridis", vmin=0, vmax=np.nanpercentile(shown, 99)
        )
        a_dep.set_title(
            f"GT z-depth, {100 * (depth > 0).mean():.0f}% of pixels", fontsize=10
        )
        fig.colorbar(im, ax=a_dep, fraction=0.03, label="m")
    fig.suptitle(
        "E35: ETH3D high-res training scenes as prepared (504 px, middle view of 20)."
        " White = no laser depth (scan holes, sky, outside the scan).\n"
        "ETH3D data CC BY-NC-SA 4.0, Schöps et al., CVPR 2017.",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=80, pil_kwargs={"quality": 85})
    print(f"figure: {FIGURE}")


if __name__ == "__main__":
    main()
