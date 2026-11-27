"""Photoreal views of a SAGE room with Blender Cycles (bpy 4.5, `uv sync --group sage`).

The meshes and textures are the ones the kit assembled for the ground truth, so images
and ground truth see the same geometry. Lighting is the kit's (`setup_scene_lighting`:
a warm sun and a grid of ceiling point lights). Materials are a Principled BSDF on the
object's texture with SAGE's metallic and roughness where the layout gives them.

The camera is the ground truth's pinhole: horizontal sensor fit with the field of view
of K, square pixels, principal point centred. Blender cameras look down -z with y up,
so cam_to_world (OpenCV) is turned by diag(1, -1, -1).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import ModuleType

import bpy
import mathutils
import numpy as np
from PIL import Image

from src.loaders.sage.config import RenderConfig

GPU_KINDS = ("OPTIX", "CUDA", "METAL")
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def use_device(scene: bpy.types.Scene, device: str) -> str:
    """Pick the first GPU backend with a device, else the CPU; returns its name."""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in GPU_KINDS if device == "GPU" else ():
        try:
            prefs.compute_device_type = kind
        except TypeError:  # backend not built into this bpy
            continue
        prefs.get_devices()
        if any(d.type == kind for d in prefs.devices):
            for d in prefs.devices:
                d.use = d.type == kind
            scene.cycles.device = "GPU"
            return kind
    scene.cycles.device = "CPU"
    return "CPU"


def material(key: str, texture: dict) -> bpy.types.Material:
    """Make a Principled BSDF on the texture image, with SAGE's PBR values if given."""
    mat = bpy.data.materials.new(f"mat_{key}")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    pbr = texture.get("pbr_parameters") or {}
    bsdf.inputs["Roughness"].default_value = float(pbr.get("roughness", 1.0))
    bsdf.inputs["Metallic"].default_value = float(pbr.get("metallic", 0.0))
    image = nodes.new("ShaderNodeTexImage")
    image.image = bpy.data.images.load(texture["texture_map_path"])
    links.new(bsdf.inputs["Base Color"], image.outputs["Color"])
    return mat


def add_mesh(key: str, info: dict) -> None:
    """Link one kit mesh with its UVs and texture into the scene."""
    mesh = info["mesh"]
    data = bpy.data.meshes.new(f"mesh_{key}")
    data.from_pydata(mesh.vertices.tolist(), [], mesh.faces.tolist())
    data.update()
    texture = info.get("texture") or {}
    if Path(texture.get("texture_map_path", "")).is_file() and "fts" in texture:
        fts = np.asarray(texture["fts"])
        assert fts.shape == (len(mesh.faces), 3), (fts.shape, mesh.faces.shape)  # noqa: S101
        uv = data.uv_layers.new(name="UVMap")
        uv.data.foreach_set("uv", np.asarray(texture["vts"])[fts].ravel().tolist())
        data.materials.append(material(key, texture))
    obj = bpy.data.objects.new(key, data)
    bpy.context.scene.collection.objects.link(obj)


def render_views(
    meshes: dict,
    room: dict,
    cam_to_world: np.ndarray,
    cfg: RenderConfig,
    kit: ModuleType,
) -> tuple[np.ndarray, str]:
    """Render every pose: RGB (N, H, W, 3) uint8 and the device used."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = cfg.samples
    scene.cycles.use_denoising = True
    used = use_device(scene, cfg.device)
    scene.render.resolution_x, scene.render.resolution_y = cfg.width, cfg.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "Standard"
    for key, info in meshes.items():
        add_mesh(key, info)
    kit.setup_scene_lighting(scene, room)
    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.angle = np.radians(cfg.fov_x_deg)  # = intrinsics(): square, centred
    camera.data.clip_start, camera.data.clip_end = 0.01, 100.0
    images = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, c2w in enumerate(cam_to_world):
            camera.matrix_world = mathutils.Matrix((c2w @ FLIP).tolist())
            scene.render.filepath = f"{tmp}/{i:03d}.png"
            bpy.ops.render.render(write_still=True)
            image = Image.open(scene.render.filepath).convert("RGB")
            images.append(np.asarray(image))
    return np.stack(images), used
