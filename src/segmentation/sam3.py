"""SAM 3 as a Segmenter: text-prompted instances, tracked or per view."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from transformers import (
    Sam3Model,
    Sam3Processor,
    Sam3VideoModel,
    Sam3VideoProcessor,
)

from src.loaders import Views
from src.segmentation.base import ID_STRIDE, NO_INSTANCE, Segmentation

if TYPE_CHECKING:
    from src.geometry import Geometry

log = logging.getLogger(__name__)

CHECKPOINT = "facebook/sam3"
# frequent ScanNet++ classes
PROMPTS = (
    "wall", "floor", "ceiling", "door", "window", "table", "chair", "monitor",
    "keyboard", "cabinet", "shelf", "whiteboard", "box", "lamp", "sink", "refrigerator",
)  # fmt: skip


def load_prompts(paths: list[Path]) -> dict[str, str]:
    """Prompt -> class from text files, in order, duplicates dropped.

    One prompt per line; `prompt = class` makes the prompt a synonym whose masks get
    that class (association only links masks of the same class). `#` starts a comment.
    """
    prompts: dict[str, str] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()  # noqa: PLW2901
            if line:
                prompt, _, cls = (s.strip() for s in line.partition("="))
                prompts.setdefault(prompt, cls or prompt)
    return prompts


@dataclass
class Sam3VideoSegmenter:
    """Treats the views as video frames; the tracker keeps instance ids across them."""

    prompts: tuple[str, ...] = PROMPTS
    synonyms: dict[str, str] = field(default_factory=dict)  # prompt -> class
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
                    _paint(ids[i], best[i], objs, scores, masks)
                    found.update(objs.tolist())
            classes.update(dict.fromkeys(found, self.synonyms.get(prompt, prompt)))
            log.info(
                "  %-12s %3d tracks  %.0fs",
                prompt,
                len(found),
                time.perf_counter() - start,
            )
        return _drop_hidden(ids, classes)


@dataclass
class Sam3ImageSegmenter:
    """Segments every view on its own; ids are NOT shared across views.

    Instance id = view index * ID_STRIDE + running index, so ids are unique over the
    scene and an association step can link them.
    """

    prompts: tuple[str, ...] = PROMPTS
    synonyms: dict[str, str] = field(default_factory=dict)  # prompt -> class
    checkpoint: str = CHECKPOINT
    threshold: float = 0.5
    _model: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)

    def _load(self) -> tuple[Any, Any]:
        if self._model is None:
            self._model = Sam3Model.from_pretrained(
                self.checkpoint, dtype=torch.bfloat16
            )
            self._model = self._model.to("cuda").eval()
            self._processor = Sam3Processor.from_pretrained(self.checkpoint)
        return self._model, self._processor

    def segment(self, views: Views, geometry: Geometry | None = None) -> Segmentation:  # noqa: ARG002
        """Run every prompt on every view, reusing the image features per view."""
        model, processor = self._load()
        num, height, width = views.images.shape[:3]
        ids = np.full((num, height, width), NO_INSTANCE, dtype=np.int32)
        best = np.zeros((num, height, width), dtype=np.float32)
        classes: dict[int, str] = {}
        for i, image in enumerate(views.images):
            inputs = processor(images=image, return_tensors="pt").to("cuda")
            sizes = inputs["original_sizes"].tolist()
            count = 0
            with torch.no_grad():
                vision = model.get_vision_features(
                    pixel_values=inputs.pixel_values.to(torch.bfloat16)
                )
                for prompt in self.prompts:
                    text = processor(text=prompt, return_tensors="pt").to("cuda")
                    res = processor.post_process_instance_segmentation(
                        model(vision_embeds=vision, **text),
                        threshold=self.threshold,
                        mask_threshold=0.5,
                        target_sizes=sizes,
                    )[0]
                    objs = i * ID_STRIDE + count + np.arange(len(res["scores"]))
                    count += len(objs)
                    assert count < ID_STRIDE, "too many masks in one view"  # noqa: S101
                    masks = res["masks"].cpu().numpy().astype(bool)
                    scores = res["scores"].float().cpu().numpy()
                    _paint(ids[i], best[i], objs, scores, masks)
                    cls = self.synonyms.get(prompt, prompt)
                    classes.update(dict.fromkeys(objs.tolist(), cls))
            log.info("  view %2d: %3d masks", i, len(np.unique(ids[i])) - 1)
        return _drop_hidden(ids, classes)


def _paint(
    ids: np.ndarray,
    best: np.ndarray,
    objs: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray,
) -> None:
    """Write masks into one view's id map; where masks overlap the higher score wins."""
    for obj, score, mask in zip(objs, scores, masks, strict=True):
        win = mask & (score > best)
        ids[win] = obj
        best[win] = score


def _drop_hidden(ids: np.ndarray, classes: dict[int, str]) -> Segmentation:
    """Keep classes only for ids that still own pixels after overlap resolution."""
    present = set(np.unique(ids).tolist())
    return Segmentation(ids, {k: v for k, v in classes.items() if k in present})


SEGMENTERS = {"video": Sam3VideoSegmenter, "image": Sam3ImageSegmenter}


def main() -> None:
    """Run SAM 3 on prepared views and cache the result next to them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("views", type=Path, nargs="+", help="prepared views (.npz)")
    parser.add_argument(
        "--mode",
        choices=sorted(SEGMENTERS),
        default="video",
        help="video: tracker ids across views; image: per-view ids (associate later)",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        action="append",
        help="prompt file (one per line, `prompt = class` for synonyms); repeat to "
        "combine in order; default: the 16 built-in prompts. The file names go into "
        "the output name: sam3<mode>-<stem>+<stem>_<views>.npz",
    )
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
    tag = f"sam3{args.mode}"
    if args.prompts:
        synonyms = load_prompts(args.prompts)
        segmenter = SEGMENTERS[args.mode](prompts=tuple(synonyms), synonyms=synonyms)
        tag += "-" + "+".join(p.stem for p in args.prompts)
    else:
        segmenter = SEGMENTERS[args.mode]()
    for path in args.views:
        views = Views.load(path)
        log.info(
            "%s: %d views, %d prompts, %d classes",
            views.scene_id,
            len(views),
            len(segmenter.prompts),
            len({segmenter.synonyms.get(p, p) for p in segmenter.prompts}),
        )
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        seg = segmenter.segment(views)
        out = path.with_name(f"{tag}_{path.stem}.npz")
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
