"""SAGE-10k (issue #99): class words, room labels, cameras and ray-cast ground truth."""

from pathlib import Path

import numpy as np
import pytest

from src.evaluation.labeled import lift
from src.geometry import Geometry
from src.loaders.ground_truth import NO_MESH, STUFF_BASE
from src.loaders.hypersim.scene import NYU40
from src.loaders.sage.config import load_config
from src.loaders.sage.download import select_rooms
from src.loaders.sage.scene import (
    Part,
    intrinsics,
    label_parts,
    nyu40_class,
    opencv_pose,
    ray_cast,
)
from src.loaders.sage.usable import assign_splits
from src.loaders.splits import scene_folders
from src.refinement.multiview import MAX_PINHOLE_RESIDUAL, pinhole_residuals

CFG = load_config(Path("configs/sage.yaml"))


@pytest.mark.parametrize(
    ("kind", "nyu40"),
    [
        ("chair_3", "chair"),
        ("armchair", "chair"),
        ("coffeetable", "table"),
        ("nightstand", "night stand"),
        ("bookshelf", "bookshelf"),
        ("bookshelfbook", "books"),
        ("booksideboard", "books"),
        ("deskpen", "otherprop"),
        ("floorlamp", "lamp"),
        ("wallart", "picture"),
        ("uppercabinet", "cabinet"),
        ("glitch", "otherprop"),
    ],
)
def test_sage_types_map_to_nyu40(kind: str, nyu40: str) -> None:
    assert nyu40_class(kind, CFG.words, CFG.props) == nyu40


def test_config_classes_are_nyu40() -> None:
    assert {c for c in CFG.words.values() if c} <= set(NYU40)
    assert set(CFG.props) <= set(NYU40)


def box(lo: list[float], hi: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """An axis-aligned box as 12 triangles."""
    v = np.array(
        [
            [x, y, z]
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])
        ]
    )
    quads = [
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    ]
    return v.astype(float), np.array(
        [t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))]
    )


LAYOUT = {
    "rooms": [
        {
            "id": "r",
            "objects": [{"id": "o1", "type": "chair_2", "description": "a chair"}],
            "doors": [{"id": "d1"}],
            "windows": [{"id": "w1"}],
            "walls": [{"id": "wall_n"}, {"id": "wall_s"}],
        }
    ]
}


def test_room_parts_get_instances_and_both_vocabularies() -> None:
    keys = [
        "wall_n",
        "wall_s",
        "floor_r",
        "floor_r_ceiling",
        "o1",
        "d1",
        "d1_frame",
        "w1",
    ]
    labels = label_parts(LAYOUT, keys, CFG.words, CFG.props)
    i = labels.instance
    assert i["wall_n"] == i["wall_s"] == STUFF_BASE + NYU40.index("wall")
    assert i["floor_r_ceiling"] == STUFF_BASE + NYU40.index("ceiling")
    assert i["d1"] == i["d1_frame"] != i["o1"]
    assert labels.nyu40[i["o1"]] == "chair"
    assert labels.nyu40[i["w1"]] == "window"
    assert labels.sage[i["o1"]] == {"type": "chair_2", "description": "a chair"}
    with pytest.raises(ValueError, match="no object"):
        label_parts(LAYOUT, [*keys, "lamp_x"], CFG.words, CFG.props)


def test_camera_looks_along_the_planner_forward() -> None:
    forward, up = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    c2w = opencv_pose(np.array([1.0, 2.0, 3.0]), np.stack([right, up, -forward], 1))
    np.testing.assert_allclose(c2w[:3, 2], forward)  # OpenCV z: forward
    np.testing.assert_allclose(c2w[:3, 1], -up)  # OpenCV y: down
    np.testing.assert_allclose(np.linalg.det(c2w[:3, :3]), 1.0)
    k = intrinsics(504, 378, 60.0)
    edge = np.linalg.inv(k) @ [0.0, 189.0, 1.0]  # the left image border
    np.testing.assert_allclose(edge[0], -np.tan(np.radians(30.0)))


def test_ray_cast_depth_lands_on_the_meshes() -> None:
    room = [Part(n, STUFF_BASE + NYU40.index(c), *box(lo, hi)) for n, c, lo, hi in (
        ("floor", "floor", [0, 0, -0.1], [4, 6, 0]),
        ("far", "wall", [0, 6, 0], [3, 6.1, 2.7]),  # an opening at x > 3
        ("left", "wall", [-0.1, 0, 0], [0, 6, 2.7]),
        ("right", "wall", [4, 0, 0], [4.1, 6, 2.7]),
        ("ceiling", "ceiling", [0, 0, 2.7], [4, 6, 2.8]),
    )]  # fmt: skip
    cube = Part("cube", 1, *box([1.5, 2.5, 0], [2.5, 3.5, 1]))
    forward, up = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    c2w = opencv_pose(
        np.array([2.0, 0.5, 0.5]), np.stack([np.cross(forward, up), up, -forward], 1)
    )
    k = intrinsics(64, 48, 90.0)
    ids, depth = ray_cast([*room, cube], k, c2w[None], 48, 64)
    points = lift(depth[0].astype(np.float64), k, c2w)
    on_cube = ids[0] == 1
    assert on_cube.sum() > 50
    np.testing.assert_allclose(points[on_cube][:, 1], 2.5, atol=1e-4)  # its front face
    np.testing.assert_allclose(depth[0][on_cube], 2.0, atol=1e-4)  # z-depth = y - 0.5
    on_far = ids[0] == STUFF_BASE + NYU40.index("wall")
    far = on_far & (np.abs(points[..., 1] - 6.0) < 1e-3)
    np.testing.assert_allclose(depth[0][far], 5.5, atol=1e-4)
    assert (ids[0] == NO_MESH).any()  # through the opening: no ground truth
    assert (depth[0][ids[0] == NO_MESH] == 0).all()


def test_rooms_are_spread_evenly_and_seeded() -> None:
    names = [f"s{i:03d}" for i in range(100)]
    picked = select_rooms(names[::-1], 10, 0)
    assert picked == names[::10]
    assert select_rooms(names, 10, 3) == names[3::10]
    with pytest.raises(ValueError, match="asked for"):
        select_rooms(names, 101, 0)


def test_both_split_formats_give_scene_folders(tmp_path: Path) -> None:
    rooms = tmp_path / "splits.yaml"
    rooms.write_text("train: [a, b]\nval: [c]\n")
    assert scene_folders(rooms, ["val", "train"]) == {"val": ["c"], "train": ["a", "b"]}
    with pytest.raises(ValueError, match="unknown splits"):
        scene_folders(rooms, ["test"])
    hypersim = scene_folders(Path("configs/hypersim.yaml"), ["dev"])["dev"]
    assert "ai_006_007_cam_00" in hypersim


def test_rooms_whose_rays_are_no_pinhole_are_found() -> None:
    h, w = 90, 120
    k = np.array([[100.0, 0, 60.0], [0, 100.0, 45.0], [0, 0, 1]])
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    cam = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(k).T
    dirs = np.stack([cam, cam])
    shuffled = np.random.default_rng(0).permutation(h * w)  # rays in no order: broken
    dirs[1] = cam.reshape(-1, 3)[shuffled].reshape(h, w, 3)
    geometry = Geometry(
        np.ones((2, h, w), np.float32),
        np.ones((2, h, w), np.float32),
        np.stack([k, k]).astype(np.float32),
        np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
        np.zeros((2, h, w, 3)),
        dirs,
    )
    residual = pinhole_residuals(geometry)
    assert residual[0] < 1e-6
    assert residual[1] > MAX_PINHOLE_RESIDUAL


def test_held_out_rooms_are_the_last_usable_ones() -> None:
    names = ["a", "b", "c", "d", "e"]
    splits = assign_splits(names, {"a", "c", "d", "e"}, 2)
    assert splits == {"train": ["a", "c"], "val": ["d", "e"]}
    assert assign_splits(names, {"a"}, 2) == {"train": [], "val": ["a"]}
