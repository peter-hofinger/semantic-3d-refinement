import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.geometry import Geometry
from src.loaders.ground_truth import GroundTruth
from src.loaders.views import Views
from src.refinement.generator import (
    KINDS,
    Children,
    EmptyCrop,
    Generator,
    GeneratorConfig,
    ImplicitGenerator,
    OVoxelGenerator,
    RenderViews,
    SeededGenerator,
    Seeds,
    TrainScene,
    build,
    candidate_mask,
    child_corner,
    child_counts,
    child_scores,
    child_targets,
    cloud_loss,
    crop_losses,
    direct_terms,
    distance_loss,
    draw_crop,
    fixed_error,
    for_arm,
    gate_loss,
    lattice,
    load_rung,
    nearest,
    occupancy_loss,
    pick_views,
    profile,
    rate_factor,
    reach_distance,
    reachable_points,
    render_depth,
    render_terms,
    render_views,
    run_model,
    scene_children,
    scene_grid,
    split_arm,
    spread,
    train,
)
from src.refinement.volume import (
    ScenePoints,
    compact,
    scene_points,
    voxelize,
)

DIMS = (16, 16, 16)


def config(**over: object) -> GeneratorConfig:
    """A tiny CPU configuration; every field of the real one."""
    fields = {
        "root": Path("data"),
        "views": "v",
        "backbone": "b",
        "segmentation": "s",
        "tag": "t",
        "train": [],
        "val": [],
        "predict": [],
        "select": "train",
        "resolution": 16,
        "widths": [8, 16],
        "split": 2,
        "reach": 1,
        "offset": 1.0,
        "crop": 16,
        "margin": 4,
        "gen_cap": 2000,
        "gt_cap": 4000,
        "comp_cap": 2000,
        "fixed_crops": 1,
        "fixed_scenes": 1,
        "steps": 1,
        "eval_every": 1,
        "amp": False,
        "lr": 1e-2,
        "seeds": [0],
        "kind": "lattice",
    }
    return GeneratorConfig(**(fields | over))


def on_device(pts: ScenePoints, gt: torch.Tensor, render: object = None) -> tuple:
    """A scene as `crop_losses` takes it: pixels, their ground truth, render views."""
    return pts, gt, render


def plane(n: int = 6000, z: float = 8.3) -> ScenePoints:
    """Points on the plane z = const of a 16-cell grid whose cells are 0.1 wide."""
    g = torch.Generator().manual_seed(0)
    xy = torch.rand(n, 2, generator=g) * 1.2 + 0.2
    points = torch.cat([xy, torch.full((n, 1), z * 0.1)], 1)
    return ScenePoints(
        points,
        torch.tensor([0.0, 0.0, 1.0]).expand(n, 3).contiguous(),
        torch.full((n, 3), 0.5),
        torch.zeros(n),
        torch.zeros(n, dtype=torch.long),
        torch.zeros(3),
        0.1,
        DIMS,
    )


def test_lattice_is_centred_and_evenly_spaced() -> None:
    base = lattice(2)
    assert base.shape == (8, 3)
    torch.testing.assert_close(base.mean(0), torch.zeros(3))
    assert base.abs().max() == 0.25


def test_untrained_generator_emits_the_lattice_of_every_candidate() -> None:
    pts = plane()
    grid = voxelize(pts, semantics=False)
    model = Generator(grid.shape[0], [8, 16], 2, 1.0, semantics=False)
    candidates = candidate_mask(grid, 0)
    out = model(grid, candidates)
    pos, logit = out.pos, out.logit
    centre = torch.nonzero(candidates).flip(1).float() + 0.5
    torch.testing.assert_close(pos, centre[:, None] + lattice(2), atol=1e-6, rtol=0)
    assert torch.all(logit == 0)


def test_untrained_implicit_generator_emits_the_lattice_and_keeps_every_point() -> None:
    pts = plane()
    grid = voxelize(pts, semantics=False)
    model = build(config(kind="implicit"))
    assert isinstance(model, ImplicitGenerator)
    candidates = candidate_mask(grid, 0)
    out = model(grid, candidates)
    pos, score = out.pos, out.logit
    centre = torch.nonzero(candidates).flip(1).float() + 0.5
    torch.testing.assert_close(pos, centre[:, None] + lattice(2), atol=1e-6, rtol=0)
    assert torch.all(
        score > 0
    )  # the starting distance, softplus(0), is below the offset


def test_untrained_split_generator_emits_the_lattice_and_its_parents_the_halves() -> (
    None
):
    pts = plane()
    grid = voxelize(pts, semantics=False)
    model = build(config(kind="split"))
    candidates = candidate_mask(grid, 0)
    out = model(grid, candidates)
    centre = torch.nonzero(candidates).flip(1).float() + 0.5
    torch.testing.assert_close(out.pos, centre[:, None] + lattice(2), atol=1e-6, rtol=0)
    assert out.coarse is not None
    parents = torch.tensor([[-0.25, 0, 0], [0.25, 0, 0]])
    torch.testing.assert_close(
        out.coarse[0], centre[:, None] + parents, atol=1e-6, rtol=0
    )
    assert torch.all(out.logit == 0) and torch.all(out.coarse[1] == 0)


def test_untrained_gated_generator_emits_the_lattice_with_the_gate_at_zero() -> None:
    pts = plane()
    grid = voxelize(pts, semantics=False)
    candidates = candidate_mask(grid, 0)
    torch.manual_seed(0)
    gated = build(config(kind="gated"))
    out = gated.eval()(grid, candidates)
    centre = torch.nonzero(candidates).flip(1).float() + 0.5
    torch.testing.assert_close(out.pos, centre[:, None] + lattice(2), atol=1e-6, rtol=0)
    assert (
        out.gate is not None and torch.all(out.gate == 0) and torch.all(out.logit == 0)
    )


def test_the_gate_opens_a_parent_for_any_reachable_child_and_counts_it_once() -> None:
    voxel = torch.tensor(
        [[0, 0, 0], [1, 1, 1], [0, 1, 0], [4, 4, 4]]
    )  # three share parent 0
    reachable = torch.tensor([False, True, False, False])
    gate = torch.tensor([3.0, 3.0, 3.0, -3.0])  # parent 0 open, parent 2 closed
    got = float(gate_loss(gate, reachable, voxel))
    # parent 0: positive (weight 1 of 1), parent 2: negative; balanced: pos_weight 1
    want = (np.log1p(np.exp(-3.0)) + np.log1p(np.exp(-3.0))) / 2
    assert abs(got - want) < 1e-6


def test_candidates_are_the_occupied_voxels_dilated_by_reach() -> None:
    grid = torch.zeros(1, 9, 9, 9)
    grid[0, 4, 4, 4] = 1
    mask = candidate_mask(grid, 2)
    assert int(mask.sum()) == 5**3
    assert mask[2:7, 2:7, 2:7].all()


def test_chamfer_terms_count_the_reachable_points_only() -> None:
    gt = torch.tensor([[0.0, 0, 0], [4, 0, 0]])
    gen = torch.tensor([[0.0, 1, 0], [4, 0, 0], [9, 0, 0]])
    assert nearest(gen, gt).tolist() == [0, 1, 1]
    inner = gen[:2].clone().requires_grad_()
    ok = torch.tensor([True, True, False])
    terms = cloud_loss(inner, gen, (gt, gt), (ok[:2], ok))
    assert float(terms["acc"].detach()) == 0.5  # distances 1 and 0
    assert float(terms["comp"].detach()) == 0.5  # nearest of gen[:2]: 1 and 0
    terms["acc"].backward()
    torch.testing.assert_close(inner.grad[0], torch.tensor([0.0, 0.5, 0.0]))
    out = torch.tensor([True, False, True])  # the far point is the only one near gt[1]
    far = cloud_loss(inner.detach(), gen, (gt, gt), (out[:2], out))
    assert float(far["acc"]) == 1.0  # only gen[0] is counted
    assert (
        abs(float(far["comp"]) - (1 + 17**0.5) / 2) < 1e-6
    )  # gt[1]: gen[0], not gen[1]


def test_the_occupancy_loss_weighs_the_rarer_class_up_to_the_commoner() -> None:
    ok = torch.tensor([True, False, False, False])
    # one positive weighted 3, three negatives: (3 ln 2 + 3 ln 2) / 4 at logit 0
    assert abs(float(occupancy_loss(torch.zeros(4), ok)) - 1.5 * np.log(2)) < 1e-6


def test_the_distance_loss_caps_the_true_distance_at_twice_the_offset() -> None:
    score = torch.tensor([0.5, 1.0, -3.0])  # distances 0.5, 0, 4 for offset 1
    truth = torch.tensor([0.25, 0.0, 9.0])  # the last is capped at 2
    assert abs(float(distance_loss(score, truth, 1.0)) - (0.25 + 0 + 2) / 3) < 1e-6


def test_the_reach_distance_is_exact_to_a_quarter_cell() -> None:
    origin, size = torch.zeros(3), torch.tensor([16, 16, 16])
    surface = torch.tensor([[8.1, 8.1, 8.1]])
    home = torch.tensor(
        [[8.1, 8.1, 8.1], [8.6, 8.1, 8.1], [9.1, 8.9, 8.1], [11.0, 8.1, 8.1]]
    )
    got = reach_distance(home, surface, 2.0, origin, size)
    np.testing.assert_allclose(got.numpy(), [0.0, 0.5, 1.0, 2.25], atol=1e-6)
    assert reachable_points(home, surface, 1.0, origin, size).tolist() == [
        True,
        True,
        True,
        False,
    ]


def test_reachable_means_a_surface_within_the_offset_on_every_axis() -> None:
    origin, size = torch.zeros(3), torch.tensor([16, 16, 16])
    surface = torch.tensor([[8.0, 8, 8], [3, 3, 3]])
    home = torch.tensor([[8.9, 8.9, 7.1], [8.0, 9.6, 8], [3, 3, 4.6], [3.5, 2.5, 3.5]])
    got = reachable_points(home, surface, 1.0, origin, size)
    assert got.tolist() == [True, False, False, True]


@pytest.mark.parametrize("kind", ["lattice", "implicit", "split", "gated"])
def test_generator_trained_on_a_plane_puts_its_points_on_it(kind: str) -> None:
    pts = plane()
    gt = pts.points
    cfg = config(kind=kind)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = build(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    first = None
    for _ in range(60):
        terms = crop_losses(
            model, voxelize(pts, semantics=False), on_device(pts, gt), cfg, rng
        )
        loss = sum(terms.values())
        first = first if first is not None else float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss.detach()) < 0.6 * first
    model.eval()
    grid = voxelize(pts, semantics=False)
    out = model(grid, candidate_mask(grid, 1))
    kept = out.pos[out.logit > 0] * 0.1
    assert len(kept) > 100
    assert (
        float((kept[:, 2] - 0.83).abs().median().detach()) < 0.03
    )  # within a third of a cell


def plane_scene() -> TrainScene:
    """The plane as a scene whose pixels are their own ground truth."""
    pts = plane()
    return TrainScene(
        compact(pts), pts.points.clone(), torch.ones(len(pts.points), dtype=torch.bool)
    )


def test_training_logs_every_step_and_keeps_a_state_no_worse_than_the_start(
    tmp_path: Path,
) -> None:
    scene = plane_scene()
    cfg = config(steps=20, eval_every=5, fixed_scenes=3)  # more than the scenes
    model, log = train(
        cfg, ([scene], [scene]), torch.device("cpu"), seed=0, log_dir=tmp_path
    )
    assert [e["step"] for e in log] == [0, 5, 10, 15, 20]
    assert all(np.isfinite(e["train"]["chamfer_cells"]) for e in log)
    kept = min(e["train"]["chamfer_cells"] for e in log)
    assert kept <= log[0]["train"]["chamfer_cells"]
    assert isinstance(model, Generator)
    events = EventAccumulator(str(tmp_path))
    events.Reload()
    tags = set(events.Tags()["scalars"])
    assert {"train/loss", "val/loss", "fixed/val_chamfer_cells"} <= tags
    assert len(events.Scalars("train/loss")) == 20  # a point every step
    assert len(events.Scalars("val/loss")) == 20  # and a validation crop after each


def test_the_fixed_error_is_the_same_on_every_call() -> None:
    scene = plane_scene()
    model = build(config())
    first = fixed_error(model, [scene], config(), torch.device("cpu"))
    assert first == fixed_error(model, [scene], config(), torch.device("cpu"))


def test_every_generator_stays_under_a_million_parameters_at_the_chosen_widths() -> (
    None
):
    wide = config(widths=[64, 128, 256])
    for kind in KINDS:
        assert (
            sum(p.numel() for p in build(config(kind=kind), DIMS).parameters()) < 10**6
        )
    with pytest.raises(AssertionError):
        build(wide)


def test_the_profile_times_every_stage_of_a_step() -> None:
    stages = profile(config(kind="seeded"), plane_scene(), torch.device("cpu"), 4)
    assert {"to_device", "grid", "forward_loss", "backward", "val", "step_s"} <= set(
        stages
    )
    assert all(v >= 0 for v in stages.values())
    assert stages["step_s"] == pytest.approx(
        sum(stages[k] for k in ("to_device", "grid", "forward_loss", "backward", "val"))
    )


def test_spread_takes_evenly_spaced_scenes_and_all_for_zero() -> None:
    names = [str(i) for i in range(40)]
    assert spread(names, 0) == names
    assert spread(names, 5) == ["0", "10", "20", "29", "39"]
    assert spread(names, 40) == names


def test_a_rung_reads_its_scenes_and_an_arm_names_its_classes(tmp_path: Path) -> None:
    caps = (
        "resolution",
        "gen_cap",
        "gt_cap",
        "comp_cap",
        "fixed_crops",
        "fixed_scenes",
    )
    arms = {"lattice": {"kind": "lattice"}}
    arms["seeded-render"] = {"kind": "seeded", "loss": "render", "stride": 6}
    rung = {
        "train_splits": ["dev"], "val_scenes": ["a_cam_00"],
        "predict_scenes": ["x_cam_00"],
        "select": "train", "steps": 3, "run": ["lattice"], "sizes": [0, 1],
    }  # fmt: skip
    raw = dict.fromkeys(caps, 16) | {
        "root": "d", "views": "v", "backbone": "b", "segmentation": "s", "tag": "t",
        "split_file": str(Path(__file__).parent.parent / "configs/hypersim.yaml"),
        "widths": [8, 16], "split": 2, "reach": 1, "offset": 1.0, "crop": 16,
        "margin": 4,
        "eval_every": 1, "amp": False, "lr": 1e-3, "seeds": [0],
        "arms": arms,
        "rungs": {"one": rung},
    }  # fmt: skip
    path = tmp_path / "e.yaml"
    path.write_text(yaml.safe_dump(raw))
    loaded = load_rung(path, "one")
    assert loaded.cfg.train == [
        "ai_010_007_cam_00",
        "ai_006_007_cam_00",
        "ai_044_001_cam_00",
    ]
    assert loaded.cfg.val == ["a_cam_00"] and loaded.cfg.predict == ["x_cam_00"]
    assert loaded.run == ["lattice"] and loaded.sizes == [0, 1]
    plain = for_arm(loaded, "lattice")
    assert (plain.kind, plain.semantics, plain.loss) == ("lattice", "none", "chamfer")
    seen = for_arm(loaded, "seeded-render-sam3")
    assert (seen.kind, seen.semantics, seen.loss) == ("seeded", "sam3", "render")
    assert seen.stride == 6
    assert for_arm(loaded, "seeded-render", {"seeded-render": 5e-4}).lr == 5e-4
    assert for_arm(loaded, "seeded-render-gt", {"seeded-render": 5e-4}).lr == 5e-4
    assert split_arm("seeded-render-gt") == ("seeded-render", "gt")
    with pytest.raises(AssertionError):
        for_arm(loaded, "nonsense")


def seeds_of(pts: ScenePoints) -> Seeds:
    """The points of a scene as seeds, in cells."""
    return Seeds((pts.points - pts.lo) / pts.cell, pts.color, pts.log_conf)


def test_untrained_seeded_generator_returns_its_input_bit_for_bit() -> None:
    pts = plane()
    grid = voxelize(pts, semantics=False)
    model = SeededGenerator(grid.shape[0], [8, 16], 3, 1.0, semantics=False)
    seeds = seeds_of(pts)
    out = model(grid, seeds)
    assert out.pos.shape == (len(pts.points), 3, 3)
    assert torch.equal(out.pos[:, 0], seeds.pos)  # copy 0 stays on its seed
    assert torch.all(out.logit[:, 0] > 0) and torch.all(out.logit[:, 1:] < 0)


def test_seeded_generator_trained_on_a_noisy_plane_moves_its_points_onto_it() -> None:
    clean = plane()
    g = torch.Generator().manual_seed(1)
    noise = torch.zeros_like(clean.points)
    noise[:, 2] = (torch.rand(len(noise), generator=g) - 0.5) * 0.06  # +-0.3 cell
    noisy = dataclasses.replace(clean, points=clean.points + noise)
    cfg = config(kind="seeded", copies=2, lr=3e-3)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = build(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    for _ in range(80):
        loss = sum(
            crop_losses(
                model,
                voxelize(noisy, semantics=False),
                on_device(noisy, clean.points),
                cfg,
                rng,
            ).values()
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    out = model(voxelize(noisy, semantics=False), seeds_of(noisy))
    kept = out.pos[out.logit > 0] * 0.1
    before = float(noise[:, 2].abs().mean())
    assert float((kept[:, 2] - 0.83).abs().mean().detach()) < 0.7 * before


def camera(depth: float = 2.0) -> RenderViews:
    """An 8 x 8 camera at the origin looking along +z, ground truth z = depth."""
    ray = torch.zeros(1, 8, 8, 3)
    col, row = torch.meshgrid(
        torch.arange(8) + 0.5, torch.arange(8) + 0.5, indexing="xy"
    )
    ray[0, ..., 0], ray[0, ..., 1], ray[0, ..., 2] = (col - 4) / 8, (row - 4) / 8, 1
    z = torch.full((1, 8, 8), depth)
    return RenderViews(
        torch.zeros(1, 3),
        torch.eye(3)[None],
        torch.full((1, 2), 8.0),
        torch.full((1, 2), 4.0),
        z,
        torch.zeros(1, 8, 8),
        ray * z[..., None],
    )


def wall(z: float, step: float = 0.02) -> torch.Tensor:
    """A dense grid of points on the plane at depth z that fills the view."""
    axis = torch.arange(-1.0, 1.0, step) + step / 2
    x, y = torch.meshgrid(axis, axis, indexing="xy")
    return torch.stack(
        [x.flatten() * z / 2, y.flatten() * z / 2, torch.full_like(x.flatten(), z)], 1
    )


def test_render_draws_a_plane_at_its_depth_and_hides_what_is_behind_it() -> None:
    rv = camera()
    near, far = wall(2.0), wall(3.0)
    depth, alpha = render_depth(near, torch.ones(len(near)), rv, 0, 0.05)
    assert torch.allclose(depth[2:6, 2:6], torch.tensor(2.0), atol=1e-3)
    assert alpha[2:6, 2:6].min() > 0.99
    both = torch.cat([near, far])
    depth, _ = render_depth(both, torch.ones(len(both)), rv, 0, 0.05)
    assert torch.allclose(
        depth[2:6, 2:6], torch.tensor(2.0), atol=1e-2
    )  # far one is behind
    _, empty = render_depth(near, torch.zeros(len(near)), rv, 0, 0.05)
    assert empty.max() == 0


def test_render_depth_is_differentiable_in_the_points_and_their_opacity() -> None:
    rv = camera()
    points = wall(2.0, 0.1).requires_grad_()
    opacity = torch.full((len(points),), 0.5, requires_grad=True)
    depth, alpha = render_depth(points, opacity, rv, 0, 0.05)
    (depth.sum() + alpha.sum()).backward()
    assert points.grad[:, 2].abs().sum() > 0 and opacity.grad.abs().sum() > 0


def test_render_terms_measure_the_depth_error_in_cells_and_the_holes() -> None:
    rv = camera()
    box = (torch.full((3,), -5.0), torch.full((3,), 5.0))
    cell = 0.05
    on = wall(2.0)
    exact = render_terms(rv, [0], (on, torch.ones(len(on))), box, (cell, 0.25))
    assert float(exact["depth"]) < 0.05 and float(exact["cover"]) < 0.05
    off = on + torch.tensor([0.0, 0.0, 0.1])  # two cells behind the truth
    shifted = render_terms(rv, [0], (off, torch.ones(len(off))), box, (cell, 0.25))
    assert abs(float(shifted["depth"]) - 2.0) < 0.1
    nothing = render_terms(rv, [0], (on, torch.zeros(len(on))), box, (cell, 0.25))
    assert float(nothing["cover"]) > 0.99


def test_seeded_render_loss_moves_noisy_points_onto_the_plane() -> None:
    clean = plane()
    g = torch.Generator().manual_seed(1)
    noise = torch.zeros_like(clean.points)
    noise[:, 2] = (torch.rand(len(noise), generator=g) - 0.5) * 0.06  # +-0.3 cell
    noisy = dataclasses.replace(clean, points=clean.points + noise)
    rv = camera(0.83)
    rv = dataclasses.replace(
        rv,
        centre=torch.tensor([[0.8, 0.8, 0.0]]),
        point=rv.point + torch.tensor([0.8, 0.8, 0.0]),
    )  # looks at the plane's middle: 8 pixels cover x, y in 0.39-1.21
    cfg = config(kind="seeded", loss="render", copies=2, lr=3e-3, tau=0.25)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = build(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    depth = []
    for _ in range(60):
        terms = crop_losses(
            model,
            voxelize(noisy, semantics=False),
            on_device(noisy, clean.points, (rv, [0])),
            cfg,
            rng,
        )
        opt.zero_grad()
        sum(terms.values()).backward()
        opt.step()
        depth.append(float(terms["depth"].detach()))
    assert all(np.isfinite(depth)) and np.mean(depth[-10:]) < 0.7 * np.mean(depth[:3])


def test_render_views_put_the_cameras_and_ground_truth_in_the_grids_frame() -> None:
    k = np.array([[20.0, 0, 8], [0, 20.0, 8], [0, 0, 1]])
    c2w = np.tile(np.eye(4), (2, 1, 1))
    c2w[:, 0, 3] = np.arange(2) * 0.1
    v, u = np.mgrid[0:16, 0:16] + 0.5
    depth = np.tile((2 + 0.02 * u + 0.01 * v)[None], (2, 1, 1)).astype(np.float32)
    rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(k).T
    origins = np.broadcast_to(c2w[:, None, None, :3, 3], (2, 16, 16, 3))
    geometry = Geometry(
        depth, np.ones_like(depth), np.tile(k, (2, 1, 1)), c2w,
        origins.copy(), np.broadcast_to(rays, (2, 16, 16, 3)).copy(),
    )  # fmt: skip
    images = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    views = Views(
        "synthetic", ["a", "b"], images, geometry.intrinsics, geometry.cam_to_world
    )
    gt = GroundTruth(np.zeros(depth.shape, dtype=np.int32), depth.copy(), {})
    _, flat_rays, flat_depth, scale = scene_points(
        geometry, images, gt.instance_ids, {}, 8
    )
    rv = render_views((flat_rays, flat_depth), views, gt, scale, 4)
    np.testing.assert_allclose(rv.centre.numpy(), c2w[:, :3, 3] / scale, atol=1e-5)
    # a block's ground-truth point, drawn into its own camera, lands on its pixel
    for view in range(2):
        drawn, _ = render_depth(
            rv.point[view].reshape(-1, 3), torch.ones(16), rv, view, 0.05
        )
        torch.testing.assert_close(drawn, rv.depth[view], atol=1e-4, rtol=0)


def test_only_views_that_see_a_supervised_pixel_in_the_box_are_picked() -> None:
    rv = camera(2.0)
    both = dataclasses.replace(
        rv,
        depth=rv.depth.repeat(2, 1, 1),
        spread=rv.spread.repeat(2, 1, 1),
        point=rv.point.repeat(2, 1, 1, 1),
    )
    both.depth[1] = 0.0  # the second view has no ground truth
    box = (torch.full((3,), -10.0), torch.full((3,), 10.0))
    rng = np.random.default_rng(0)
    assert pick_views((both, [0, 1]), box, (1.0, 5), rng) == [0]
    far = (torch.full((3,), 100.0), torch.full((3,), 101.0))
    with pytest.raises(EmptyCrop):
        pick_views((both, [0, 1]), far, (1.0, 5), rng)


def test_a_crop_with_nothing_to_score_is_drawn_again_and_given_up_after_twenty() -> (
    None
):
    pts = plane()
    model = build(config())
    grid = voxelize(pts, semantics=False)
    rng = np.random.default_rng(0)
    cfg = config()
    terms = draw_crop(model, grid, on_device(pts, pts.points), cfg, rng)
    assert set(terms) >= {"acc", "comp", "occ"}
    nowhere = pts.points + 100.0  # a ground truth no crop can contain
    with pytest.raises(EmptyCrop):
        draw_crop(model, grid, on_device(pts, nowhere), cfg, rng)


def test_a_resumed_model_keeps_the_curve_of_its_training(tmp_path: Path) -> None:
    cfg = config(steps=10, eval_every=5, fixed_scenes=1)
    scene = plane_scene()
    args = (cfg, "t-lattice-s0", ([scene], [scene]), torch.device("cpu"), tmp_path)
    run_model(*args)
    first = yaml.safe_load((tmp_path / "t-lattice-s0.yaml").read_text())
    run_model(*args)  # the model is saved: it is loaded, not trained
    again = yaml.safe_load((tmp_path / "t-lattice-s0.yaml").read_text())
    assert [e["step"] for e in first["curve"]] == [0, 5, 10]
    assert again["curve"] == first["curve"]
    assert again["seconds"] == first["seconds"]


def plane_children(split: int = 2) -> tuple[Children, torch.Tensor]:
    """The plane's candidate children at the oracle: logits and vertices on target."""
    pts = plane()
    grid = voxelize(pts, semantics=False)
    voxel = torch.nonzero(candidate_mask(grid, 1)).flip(1)
    occupied, vertex = child_targets(voxel, pts.points / pts.cell, split, DIMS)
    corner = voxel.float()[:, None] + child_corner(split)
    children = Children(
        torch.where(occupied, 30.0, -30.0),
        corner + vertex / split,
        corner,
        occupied,
        vertex,
        torch.ones(len(voxel), dtype=torch.bool),
    )
    return children, voxel


def test_the_child_targets_are_the_occupancy_and_the_mean_of_the_points_in_each() -> (
    None
):
    voxel = torch.tensor([[0, 0, 0], [1, 0, 0]])
    points = torch.tensor(
        [[0.1, 0.1, 0.1], [0.3, 0.1, 0.1], [1.75, 0.75, 0.75], [5.0, 0.5, 0.5]]
    )
    occupied, vertex = child_targets(voxel, points, 2, (2, 1, 1))
    assert occupied.sum() == 2  # the last point lies in no listed voxel
    assert occupied[0, 0] and occupied[1, 7]  # lattice order: (ix * 2 + iy) * 2 + iz
    assert vertex[0, 0].tolist() == pytest.approx([0.4, 0.2, 0.2])
    assert vertex[1, 7].tolist() == pytest.approx([0.5, 0.5, 0.5])
    assert vertex[~occupied].abs().sum() == 0


@pytest.mark.parametrize("split", [2, 3, 4])
def test_the_direct_loss_is_zero_at_the_oracle_and_positive_off_it(split: int) -> None:
    children, _ = plane_children(split)
    assert children.occupied.any()
    terms = direct_terms(children, split)
    assert float(terms["occ"]) < 1e-9
    assert float(terms["vertex"]) == pytest.approx(0.0, abs=1e-5)
    flipped = dataclasses.replace(children, logit=-children.logit)
    assert float(direct_terms(flipped, split)["occ"]) > 10
    moved = dataclasses.replace(children, pos=children.pos + 0.1)  # 0.1 cell
    assert float(direct_terms(moved, split)["vertex"]) == pytest.approx(
        0.1 * split, abs=1e-4
    )


def test_the_oracle_children_score_one_everywhere() -> None:
    children, _ = plane_children()
    scores = child_scores(child_counts(children, 2))
    assert scores["precision"] == pytest.approx(1.0)
    assert scores["recall"] == pytest.approx(1.0)
    assert scores["vertex_err"] == pytest.approx(0.0, abs=1e-5)


def test_child_scores_pool_counts_by_hand() -> None:
    counts = {
        "kept": 4.0,
        "occupied": 5.0,
        "hit": 3.0,
        "vertex_sum": 1.5,
        "vertex_n": 15.0,
    }
    assert child_scores(counts) == pytest.approx(
        {"precision": 0.75, "recall": 0.6, "vertex_err": 0.1}
    )
    nothing = {
        "kept": 0.0,
        "occupied": 0.0,
        "hit": 0.0,
        "vertex_sum": 0.0,
        "vertex_n": 0.0,
    }
    assert child_scores(nothing) == {"precision": 0.0, "recall": 0.0, "vertex_err": 0.0}


def test_an_untrained_ovoxel_generator_keeps_nothing_and_centres_every_vertex() -> None:
    pts = plane()
    grid = voxelize(pts, semantics=False)
    candidates = candidate_mask(grid, 1)
    out = build(config(kind="ovoxel", loss="direct"))(grid, candidates)
    voxel = torch.nonzero(candidates).flip(1).float()
    assert (out.logit == 0).all()
    centres = voxel[:, None] + 0.5 + lattice(2)
    assert torch.allclose(out.pos.detach(), centres)


def test_a_vertex_never_leaves_its_child() -> None:
    torch.manual_seed(0)
    pts = plane()
    grid = voxelize(pts, semantics=False)
    candidates = candidate_mask(grid, 1)
    model = OVoxelGenerator(8, [8, 16], 3, semantics=False)
    for p in model.head.parameters():
        torch.nn.init.normal_(p, std=50.0)  # saturates the sigmoid
    out = model(grid, candidates)
    voxel = torch.nonzero(candidates).flip(1).float()
    placed = ((out.pos - voxel[:, None] - child_corner(3)) * 3).detach()
    assert placed.min() > -1e-5 and placed.max() < 1 + 1e-5  # float32 rounding


def fit_plane(cfg: GeneratorConfig, steps: int) -> torch.nn.Module:
    """Train a model of the config on the plane's direct loss and return it."""
    pts = plane()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = build(cfg, DIMS)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for _ in range(steps):
        grid = scene_grid(model, pts)
        terms = crop_losses(model, grid, on_device(pts, pts.points), cfg, rng)
        opt.zero_grad()
        sum(terms.values()).backward()
        opt.step()
    return model


def test_a_latent_generator_fits_the_targets_of_a_toy_scene() -> None:
    cfg = config(kind="latent", loss="direct", amp=False, lr=3e-2)
    model = fit_plane(cfg, 300)
    pts = plane()
    children = scene_children(model, pts, pts.points, cfg, torch.device("cpu"))
    scores = child_scores(child_counts(children, 2))
    assert scores["precision"] > 0.95 and scores["recall"] > 0.95
    assert scores["vertex_err"] < 0.05


def test_an_ovoxel_generator_trained_on_a_plane_finds_and_places_its_children() -> None:
    cfg = config(kind="ovoxel", loss="direct", lr=1e-2)
    model = fit_plane(cfg, 150)
    pts = plane()
    children = scene_children(model, pts, pts.points, cfg, torch.device("cpu"))
    scores = child_scores(child_counts(children, 2))
    assert scores["precision"] > 0.9 and scores["recall"] > 0.9
    assert scores["vertex_err"] < 0.1


def test_the_fixed_error_of_a_direct_model_reports_its_output_and_its_children() -> (
    None
):
    scene = plane_scene()
    cfg = config(kind="ovoxel", loss="direct")
    untrained = fixed_error(build(cfg), [scene], cfg, torch.device("cpu"))
    assert untrained["chamfer_cells"] == float("inf")  # nothing is kept yet
    assert untrained["precision"] == untrained["recall"] == 0.0
    model = fit_plane(dataclasses.replace(cfg, lr=1e-2), 100)
    trained = fixed_error(model, [scene], cfg, torch.device("cpu"))
    assert trained["chamfer_cells"] < 1.0
    assert trained["precision"] > 0.5 and trained["recall"] > 0.5
    assert trained == fixed_error(model, [scene], cfg, torch.device("cpu"))


def test_a_latent_table_is_memory_outside_the_parameter_limit() -> None:
    model = build(config(kind="latent", loss="direct", amp=False), DIMS)
    head = sum(p.numel() for n, p in model.named_parameters() if n != "table")
    assert head < 10**6 < sum(p.numel() for p in model.parameters()) * 100
    with pytest.raises(AssertionError):
        build(config(kind="latent", loss="direct"))  # a table needs its scene's grid
    with pytest.raises(AssertionError):
        build(config(kind="ovoxel", loss="direct", widths=[64, 128, 256]))


def test_a_cosine_decay_takes_the_rate_from_one_to_zero_over_the_steps(
    tmp_path: Path,
) -> None:
    assert rate_factor(0, 10, cosine=True) == pytest.approx(1.0)
    assert rate_factor(5, 10, cosine=True) == pytest.approx(0.5)
    assert rate_factor(10, 10, cosine=True) == pytest.approx(0.0, abs=1e-12)
    assert rate_factor(7, 10, cosine=False) == 1.0
    scene = plane_scene()
    cfg = config(steps=4, eval_every=4, cosine=True, lr=0.1)
    trained, log = train(
        cfg, ([scene], [scene]), torch.device("cpu"), seed=0, log_dir=tmp_path
    )
    assert isinstance(trained, Generator) and [e["step"] for e in log] == [0, 4]


def test_fixed_vertices_sit_at_their_childs_centre_and_only_the_logits_train() -> None:
    pts = plane()
    grid = voxelize(pts, semantics=False)
    candidates = candidate_mask(grid, 1)
    cfg = config(kind="ovoxel", loss="direct", fixed_vertex=True)
    model = build(cfg)
    for p in model.head.parameters():
        torch.nn.init.normal_(p, std=3.0)
    out = model(grid, candidates)
    voxel = torch.nonzero(candidates).flip(1).float()
    assert torch.allclose(out.pos, voxel[:, None] + 0.5 + lattice(2))
    scene = on_device(pts, pts.points)
    terms = crop_losses(model, grid, scene, cfg, np.random.default_rng(0))
    assert list(terms) == ["occ"]


def test_the_parameter_limit_is_raised_only_by_the_config() -> None:
    wide = config(kind="ovoxel", loss="direct", widths=[32, 64, 128])
    with pytest.raises(AssertionError):
        build(wide)
    raised = dataclasses.replace(wide, max_parameters=2_000_000)
    assert sum(p.numel() for p in build(raised).parameters()) > 10**6
