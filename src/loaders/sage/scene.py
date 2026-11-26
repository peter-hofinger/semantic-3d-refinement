"""SAGE-10k rooms as labelled meshes, cameras and exact per-pixel ground truth.

SAGE-10k (Xia et al. 2026, "SAGE: Scalable Agentic 3D Scene Generation for Embodied
AI", arXiv 2602.10116; huggingface.co/datasets/nvidia/SAGE-10k, Apache-2.0) ships each
room as a layout file and TRELLIS object meshes, without images. The dataset's kit
(`data/third_party/sage/kits`, pinned revision) assembles a room: walls with their door
and window openings cut, floor, ceiling, doors, windows and every object moved by its
layout pose (`export_layout_to_mesh_dict_list_v2`). It also plans a collision-free
camera path through the room (`generate_camera_trajectory`). We use both unchanged.

Ground truth is ray cast on the same meshes through each pixel centre (open3d), so it
is exact: z-depth, one instance per object, door (leaf and frame) and window. Walls,
floor and ceiling become one stuff instance per class, `STUFF_BASE + NYU40 id`, as in
Hypersim. A ray that leaves the room through an opening has no ground truth (NO_MESH).

Classes: every instance keeps SAGE's own type and description (`labels_<views>.json`).
Its NYU40 class (`nyu40_class`) is what the evaluation scores, because SAM 3 is
prompted with NYU40 names and SAGE's 2000+ free-text types would never match them.
"""

from __future__ import annotations

import functools
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import open3d as o3d

from src.loaders.ground_truth import NO_MESH, STUFF_BASE
from src.loaders.hypersim.scene import NYU40
from src.loaders.sage.config import RenderConfig

STUFF = {"wall": "wall", "floor": "floor", "ceiling": "ceiling"}
FALLBACK_CLASS = "otherprop"  # a type with no known class word: almost always a prop
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class Part:
    """One mesh of a room and the ground-truth instance it belongs to."""

    key: str  # the kit's mesh id
    instance: int
    vertices: np.ndarray  # (V, 3) metres, SAGE world (z up)
    faces: np.ndarray  # (F, 3) int


@dataclass(frozen=True)
class Labels:
    """Ground-truth instances of a room: NYU40 class and SAGE's own label."""

    instance: dict[str, int]  # mesh id -> instance id
    nyu40: dict[int, str]  # instance id -> NYU40 class
    sage: dict[int, dict[str, str]]  # instance id -> {"type", "description"}


def cut(text: str, words: dict[str, str | None]) -> list[str]:
    """Cut `text` into its longest known words, left to right, skipping the rest."""
    s = re.sub(r"[^a-z]", "", text.lower())
    longest = max(map(len, words))
    out, i = [], 0
    while i < len(s):
        for n in range(min(longest, len(s) - i), 0, -1):
            if s[i : i + n] in words:
                out.append(s[i : i + n])
                i += n
                break
        else:
            i += 1
    return out


def nyu40_class(kind: str, words: dict[str, str | None], props: tuple[str, ...]) -> str:
    """NYU40 class of a SAGE type: a prop word beats a furniture word, the last wins."""
    classes = [c for w in cut(kind, words) if (c := words[w]) is not None]
    preferred = [c for c in classes if c in props] or classes
    return preferred[-1] if preferred else FALLBACK_CLASS


def label_parts(
    layout: dict, keys: list[str], words: dict[str, str | None], props: tuple[str, ...]
) -> Labels:
    """Instance of every mesh id of a one-room layout; door frames join their door."""
    (room,) = layout["rooms"]
    objects = {o["id"]: o for o in room["objects"]}
    doors = {d["id"] for d in room["doors"]}
    windows = {w["id"] for w in room["windows"]}
    stuff = {w["id"]: "wall" for w in room["walls"]}
    stuff |= {f"floor_{room['id']}": "floor", f"floor_{room['id']}_ceiling": "ceiling"}
    owner = {}
    for key in keys:
        frame_of = key.removesuffix("_frame")
        base = frame_of if frame_of in doors else key
        if base in objects or base in doors or base in windows:
            owner[key] = base
        elif key in stuff:
            owner[key] = STUFF[stuff[key]]
        else:
            msg = f"mesh {key} is no object, door, window, wall, floor or ceiling"
            raise ValueError(msg)
    things = sorted({o for o in owner.values() if o not in STUFF})
    ids = {o: i + 1 for i, o in enumerate(things)}
    ids |= {s: STUFF_BASE + NYU40.index(c) for s, c in STUFF.items()}
    nyu40, sage = {}, {}
    for o in set(owner.values()):
        if o in objects:
            kind, text = objects[o]["type"], objects[o].get("description", "")
            nyu40[ids[o]] = nyu40_class(kind, words, props)
        else:
            kind = STUFF.get(o) or ("door" if o in doors else "window")
            text = ""
            nyu40[ids[o]] = kind
        sage[ids[o]] = {"type": kind, "description": text}
    return Labels({k: ids[o] for k, o in owner.items()}, nyu40, sage)


def intrinsics(width: int, height: int, fov_x_deg: float) -> np.ndarray:
    """Pinhole K: square pixels, centred principal point, pixel corners at integers."""
    f = width / (2 * np.tan(np.radians(fov_x_deg) / 2))
    return np.array([[f, 0.0, width / 2], [0.0, f, height / 2], [0.0, 0.0, 1.0]])


def opencv_pose(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Return cam_to_world (OpenCV) of a planner pose (columns right, up, -forward)."""
    assert rotation.shape == (3, 3), rotation.shape  # noqa: S101
    assert position.shape == (3,), position.shape  # noqa: S101
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)  # noqa: S101
    c2w = np.eye(4)
    c2w[:3, :3] = rotation @ OPENGL_TO_OPENCV
    c2w[:3, 3] = position
    return c2w


def ray_cast(
    parts: list[Part], k: np.ndarray, cam_to_world: np.ndarray, height: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Instance ids (NO_MESH: no hit) and z-depth (0: no hit), (N, H, W) per view."""
    scene = o3d.t.geometry.RaycastingScene()
    owner = np.array([p.instance for p in parts] + [NO_MESH], dtype=np.int64)
    for i, p in enumerate(parts):
        gid = scene.add_triangles(
            p.vertices.astype(np.float32), p.faces.astype(np.uint32)
        )
        assert gid == i, (gid, i)  # noqa: S101
    v, u = np.mgrid[0:height, 0:width] + 0.5
    pixels = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(k).T  # camera z = 1
    ids, depth = [], []
    for c2w in cam_to_world:
        d = pixels @ c2w[:3, :3].T
        o = np.broadcast_to(c2w[:3, 3], d.shape)
        hit = scene.cast_rays(
            o3d.core.Tensor(np.concatenate([o, d], -1), o3d.core.float32)
        )
        t = hit["t_hit"].numpy()  # along a ray of camera z = 1: z-depth
        g = hit["geometry_ids"].numpy().astype(np.int64)
        found = np.isfinite(t)
        depth.append(np.where(found, t, 0.0))
        ids.append(owner[np.where(found, g, -1)])
    return np.stack(ids).astype(np.int32), np.stack(depth).astype(np.float32)


def kit(root: Path) -> tuple[ModuleType, ModuleType]:
    """Import the kit's assembly and camera-path modules; the second needs bpy."""
    path = str((root / "kits").resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    import render_video  # noqa: PLC0415  third-party, git-ignored, imported on use
    import tex_utils_local  # noqa: PLC0415

    return tex_utils_local, render_video


def assemble(layout: dict, folder: Path, root: Path) -> tuple[object, dict]:
    """Assemble a room with the kit: its floor plan and mesh id -> {mesh, texture}."""
    tex, _ = kit(root)
    plan = tex.dict_to_floor_plan(layout)
    return plan, tex.export_layout_to_mesh_dict_list_v2(plan, str(folder))


def camera_path(
    plan: object, meshes: dict, cfg: RenderConfig, seed: int, root: Path
) -> list[np.ndarray]:
    """Plan the kit's camera path, `path_frames` poses, seeded; OpenCV cam_to_world."""
    _, rv = kit(root)
    (room,) = plan.rooms
    lo = np.array([room.position.x, room.position.y, room.position.z])
    size = [room.dimensions.width, room.dimensions.length, room.dimensions.height]
    every = [m["mesh"] for m in meshes.values()]
    things = [meshes[o.id]["mesh"] for o in room.objects if o.id in meshes]
    np.random.seed(seed)  # noqa: NPY002  the kit draws from numpy's global generator
    # and from trimesh's surface sampler, unseeded by default: seed it for the planner
    sample = rv.trimesh.sample.sample_surface
    rv.trimesh.sample.sample_surface = functools.partial(sample, seed=seed)
    try:
        env = rv.CameraPlannerEnv([*lo, *(lo + size)], every, things)
    finally:
        rv.trimesh.sample.sample_surface = sample
    path = rv.generate_camera_trajectory(
        [*lo, *(lo + size)],
        every,
        cfg.path_frames,
        cfg.path_anchors,
        env=env,
        room=room,
        mesh_dict=meshes,
    )
    return [
        opencv_pose(np.asarray(p["position"]), np.asarray(p["rotation"])) for p in path
    ]
