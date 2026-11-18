"""Design tokens and drawing helpers of the issue-#76 figures (E57c).

One palette, used for the same thing everywhere: slate = error (a point that is worse than average), teal = what the oracle keeps and needs, steel blue = what the
confidence keeps, amber = bridged by the fusion's rule, rose = the seam the 8 cm gate leaves, violet = a reprojected seed, light gray = dropped. Clouds are splatted on a
slate-gray ground (`GROUND`), so holes in white walls stay visible. Figures are laid out in inches by `Layout`, not by the layout engine, so the panels line up.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from src.analysis.render import Camera, splat
from src.analysis.scene_eval import SceneInputs

INK = "#1B2A41"
MUTED = "#6B7785"
HAIR = "#D7DBE2"
GROUND = "#C4CBD6"
GROUP = {"reference": "#8A94A6", "oracle": "#1F7F73", "fusion": "#D98324"}
GROUP_NAME = {
    "reference": "No ground truth",
    "oracle": "Oracle (uses the GT)",
    "fusion": "Oracle + confidence (uses the GT)",
}
STEP = {
    "dropped": "#E3E6EB",
    "confidence": "#3B5B92",
    "oracle": "#1F9E89",
    "bridged": "#F2A541",
    "seam": "#D1495B",
    "seed": "#7A5BA6",
    "error": "#2B3A55",
}
# category codes of `e57c_two_fills` (white, red, green, blue, gold) -> the palette
CATEGORY_COLORS = [
    STEP["confidence"],
    STEP["dropped"],
    STEP["oracle"],
    STEP["seam"],
    STEP["bridged"],
]
ERROR_MAP = LinearSegmentedColormap.from_list(
    "error", ["#F6F7F9", "#B9C2D3", STEP["error"]]
)
NEED_MAP = LinearSegmentedColormap.from_list("need", ["#F6F7F9", "#BFE3DC", "#0F6B5E"])
NET_MAP = LinearSegmentedColormap.from_list(
    "net", ["#0F8F7E", "#BFE3DC", "#F6F7F9", "#B9C2D3", STEP["error"]]
)
ORDER_MAP = LinearSegmentedColormap.from_list(
    "order", ["#0F6B5E", "#7FC4B8", STEP["dropped"]]
)  # v from 0 (kept longest) to 1 (dropped first)


def apply_style() -> None:
    """Set the matplotlib defaults of the figures."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": INK,
            "axes.edgecolor": HAIR,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def to_rgb(color: str) -> np.ndarray:
    """Return a hex color as an RGB triple in [0, 1]."""
    return np.array(mpl.colors.to_rgb(color))


def cloud_image(
    inputs: SceneInputs,
    aligned: np.ndarray,
    keep: np.ndarray,
    camera: Camera,
    device: torch.device,
    colors: np.ndarray | None = None,
    ceiling: float | None = None,
) -> np.ndarray:
    """Splat the kept pixels of all views (image colors, or `colors` (V, H, W, 3)) from the camera on the ground color; (H, W, 3) in [0, 1]."""
    centers = inputs.views.cam_to_world[:, None, None, :3, 3]
    focal = inputs.views.intrinsics[:, None, None, 0, 0]
    radii = 0.5 * np.linalg.norm(aligned - centers, axis=-1) / focal
    colors = inputs.views.images / 255 if colors is None else colors
    ok = keep & np.isfinite(aligned).all(-1)
    if ceiling is not None:
        ok &= aligned[..., 2] < ceiling

    def tensor(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(x[ok])).float().to(device)

    image = splat(
        tensor(aligned), tensor(colors), tensor(radii), camera, background=-1.0
    )
    return np.clip(np.where(image[..., :1] < 0, to_rgb(GROUND), image), 0, 1)


class Layout:
    """A figure laid out in inches from the top left: `place` returns axes at (x, y, w, h)."""

    def __init__(self, width: float, height: float) -> None:
        """Open a figure of width x height inches."""
        self.width, self.height = width, height
        self.fig = plt.figure(figsize=(width, height))

    def place(self, x: float, y: float, w: float, h: float) -> plt.Axes:
        """Return an axes of w x h inches with its top left corner at (x, y) inches."""
        ax = self.fig.add_axes(
            [x / self.width, 1 - (y + h) / self.height, w / self.width, h / self.height]
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return ax

    def text(self, x: float, y: float, s: str, **kwargs: object) -> None:
        """Write text with its top left corner at (x, y) inches."""
        kwargs.setdefault("va", "top")
        self.fig.text(x / self.width, 1 - y / self.height, s, **kwargs)

    def save(self, path: str, dpi: int = 110) -> None:
        """Save the figure."""
        self.fig.savefig(path, dpi=dpi, pil_kwargs={"quality": 90})
        plt.close(self.fig)


def chips(
    layout: Layout,
    x: float,
    y: float,
    items: list[tuple[str, str]],
    gap: float = 0.28,
    size: float = 9.5,
) -> float:
    """Draw colored squares with labels in a row from (x, y); return the x where the row ends."""
    for color, label in items:
        layout.fig.patches.append(
            plt.Rectangle(
                (x / layout.width, 1 - (y + 0.16) / layout.height),
                0.15 / layout.width,
                0.15 / layout.height,
                transform=layout.fig.transFigure,
                facecolor=color,
                edgecolor="none",
            )
        )
        layout.text(x + 0.22, y - 0.01, label, fontsize=size, color=INK)
        x += 0.22 + 0.075 * len(label) * size / 9.5 + gap
    return x
