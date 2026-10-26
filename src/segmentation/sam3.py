"""SAM 3 video mode as a Segmenter: text-prompted instances tracked across views."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor

from src.loaders import Views
from src.segmentation.base import NO_INSTANCE, Segmentation

if TYPE_CHECKING:
    from src.geometry import Geometry

log = logging.getLogger(__name__)

CHECKPOINT = "facebook/sam3"
# frequent ScanNet++ classes
PROMPTS = (
    "wall", "floor", "ceiling", "door", "window", "table", "chair", "monitor",
    "keyboard", "cabinet", "shelf", "whiteboard", "box", "lamp", "sink", "refrigerator",
)  # fmt: skip
ID_STRIDE = 1000  # instance id = prompt index * ID_STRIDE + SAM 3 object id


@dataclass
class Sam3VideoSegmenter:
    """Treats the views as video frames; the tracker keeps instance ids across them."""

    prompts: tuple[str, ...] = PROMPTS
    checkpoint: str = CHECKPOINT
    _model: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)

    def _load(self) -> tuple[Any, Any]:
        if self._model is None:
            self._model = Sam3VideoModel.from_pretrained(
                self.checkpoint, dtype=torch.bfloat16
            )
            self._model = self._model.to("cuda").eval()
            self._processor = Sam3VideoProcessor.from_pretrained(self.checkpoint)
        return self._model, self._processor

    def segment(self, views: Views, geometry: Geometry | None = None) -> Segmentation:  # noqa: ARG002
        """Track every prompted class through the views in order."""
        model, processor = self._load()
        num, height, width = views.images.shape[:3]
        ids = np.full((num, height, width), NO_INSTANCE, dtype=np.int32)
        best = np.zeros((num, height, width), dtype=np.float32)
        classes: dict[int, str] = {}
        # one session per prompt: all prompts at once do not fit in 4 GB, and
        # grouping saves no time since detection runs per prompt anyway
        for k, prompt in enumerate(self.prompts):
            start = time.perf_counter()
            # frames and tracker state stay in CPU memory, only the model on the GPU
            session = processor.init_video_session(
                video=list(views.images),
                inference_device="cuda",
                inference_state_device="cpu",
                processing_device="cpu",
                video_storage_device="cpu",
                dtype=torch.bfloat16,
            )
            processor.add_text_prompt(session, prompt)
            found = set()
            with torch.no_grad():
                for out in model.propagate_in_video_iterator(session):
                    res = processor.postprocess_outputs(session, out)
                    i = out.frame_idx
                    masks = res["masks"].cpu().numpy().astype(bool)
                    scores = res["scores"].float().cpu().numpy()
                    objs = res["object_ids"].cpu().numpy() + k * ID_STRIDE
                    # overlapping masks: the higher score keeps the pixel
                    for obj, score, mask in zip(objs, scores, masks, strict=True):
                        win = mask & (score > best[i])
                        ids[i][win] = obj
                        best[i][win] = score
                        found.add(int(obj))
            classes.update(dict.fromkeys(found, prompt))
            log.info(
                "  %-12s %3d tracks  %.0fs",
                prompt,
                len(found),
                time.perf_counter() - start,
            )
        present = set(np.unique(ids).tolist())
        return Segmentation(ids, {k: v for k, v in classes.items() if k in present})


def main() -> None:
    """Run SAM 3 video mode on prepared views and cache the result next to them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("views", type=Path, nargs="+", help="prepared views (.npz)")
    parser.add_argument(
        "--max-gpu-gb",
        type=float,
        default=3.5,
        help="fail with out-of-memory above this instead of spilling into shared "
        "memory (Windows), which stalls the run",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_gb / total))
    segmenter = Sam3VideoSegmenter()
    for path in args.views:
        views = Views.load(path)
        log.info("%s: %d views, %d prompts", views.scene_id, len(views), len(PROMPTS))
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        seg = segmenter.segment(views)
        out = path.with_name(f"sam3video_{path.stem}.npz")
        seg.save(out)
        log.info(
            "%s: %d instances in %d views, %.0fs, peak GPU %.2f GB -> %s",
            views.scene_id,
            len(seg.classes),
            len(views),
            time.perf_counter() - start,
            torch.cuda.max_memory_allocated() / 1e9,
            out,
        )


if __name__ == "__main__":
    main()
