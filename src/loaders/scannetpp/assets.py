"""ScanNet++ asset names, where they live in a scene folder, and how they are served."""

from pathlib import PurePosixPath

# asset name -> (path inside data/<scene_id>/, served as a zip archive)
ASSETS: dict[str, tuple[str, bool]] = {
    "dslr_resized_dir": ("dslr/resized_images", True),
    "dslr_resized_mask_dir": ("dslr/resized_anon_masks", True),
    "dslr_resized_undistorted_dir": ("dslr/resized_undistorted_images", True),
    "dslr_resized_undistorted_mask_dir": ("dslr/resized_undistorted_masks", True),
    "dslr_nerfstudio_transform_path": ("dslr/nerfstudio/transforms.json", False),
    "dslr_nerfstudio_transform_undistorted_path": (
        "dslr/nerfstudio/transforms_undistorted.json",
        True,
    ),
    "dslr_colmap_dir": ("dslr/colmap", True),
    "dslr_train_test_lists_path": ("dslr/train_test_lists.json", False),
    "scan_mesh_path": ("scans/mesh_aligned_0.05.ply", True),
    "scan_mesh_mask_path": ("scans/mesh_aligned_0.05_mask.txt", False),
    "scan_mesh_segs_path": ("scans/segments.json", True),
    "scan_anno_json_path": ("scans/segments_anno.json", True),
    "scan_sem_mesh_path": ("scans/mesh_aligned_0.05_semantic.ply", True),
    "iphone_video_path": ("iphone/rgb.mkv", False),
    "iphone_video_mask_path": ("iphone/rgb_mask.mkv", True),
    "iphone_depth_path": ("iphone/depth.bin", True),
    "iphone_pose_intrinsic_imu_path": ("iphone/pose_intrinsic_imu.json", True),
    "iphone_colmap_dir": ("iphone/colmap", True),
}

# dataset-level files, relative to the data root
METADATA_FILES = [
    "splits/nvs_sem_train.txt",
    "splits/nvs_sem_val.txt",
    "splits/nvs_test.txt",
    "splits/sem_test.txt",
    "metadata/scene_types.json",
    "metadata/semantic_classes.txt",
    "metadata/instance_classes.txt",
    "metadata/semantic_benchmark/top100.txt",
    "metadata/semantic_benchmark/top100_instance.txt",
    "metadata/semantic_benchmark/map_benchmark.csv",
]


def local_path(scene_id: str, asset: str) -> PurePosixPath:
    """Path of an asset relative to the data root, after unzipping."""
    return PurePosixPath("data") / scene_id / ASSETS[asset][0]


def remote_path(scene_id: str, asset: str) -> str:
    """Path of an asset on the server (zipped assets end in .zip)."""
    path = local_path(scene_id, asset)
    return str(path.with_suffix(".zip") if ASSETS[asset][1] else path)
