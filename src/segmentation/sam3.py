"""SAM 3 image mode as a Segmenter: text-prompted instances in every view."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from transformers import Sam3Model, Sam3Processor
from transformers.modeling_outputs import BaseModelOutputWithPooling

from src.device import best_device
from src.geometry import Geometry
from src.loaders import Views
from src.segmentation.base import ID_STRIDE, NO_INSTANCE, Segmentation

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
class Sam3ImageSegmenter:
    """Segments every view on its own; ids are NOT shared across views.

    Instance id = view index * ID_STRIDE + running index, so ids are unique over the
    scene and an association step can link them.
    """

    prompts: tuple[str, ...] = PROMPTS
    synonyms: dict[str, str] = field(default_factory=dict)  # prompt -> class
    checkpoint: str = CHECKPOINT
    threshold: float = 0.5
    batch: int = 4  # prompts per forward pass: 2.9 GB peak, under the 3.5 GB cap
    _model: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)

    def _load(self) -> tuple[Any, Any]:
        if self._model is None:
            self._model = Sam3Model.from_pretrained(
                self.checkpoint, dtype=torch.bfloat16
            )
            self._model = self._model.to(best_device()).eval()
            self._processor = Sam3Processor.from_pretrained(self.checkpoint)
        return self._model, self._processor

    def segment(self, views: Views, geometry: Geometry | None = None) -> Segmentation:  # noqa: ARG002
        """Every prompt on every view: text encoded once, prompts batched per view."""
        model, processor = self._load()
        device = model.device
        num, height, width = views.images.shape[:3]
        ids = np.full((num, height, width), NO_INSTANCE, dtype=np.int32)
        classes: dict[int, str] = {}
        all_scores: dict[int, float] = {}
        prompt_classes = [self.synonyms.get(p, p) for p in self.prompts]
        with torch.no_grad():
            # the processor pads every prompt to the same 32 tokens; the encoder runs
            # once per prompt and scene: batched, its bf16 output rounds differently,
            # one at a time it is bit for bit the old per-view encoding (E39)
            text = processor(text=list(self.prompts), return_tensors="pt").to(device)
            embeds = torch.cat(
                [
                    model.get_text_features(
                        input_ids=text.input_ids[k : k + 1],
                        attention_mask=text.attention_mask[k : k + 1],
                    ).pooler_output
                    for k in range(len(self.prompts))
                ]
            )
            for i, image in enumerate(views.images):
                inputs = processor(images=image, return_tensors="pt").to(device)
                sizes = inputs["original_sizes"].tolist()
                vision = model.get_vision_features(
                    pixel_values=inputs.pixel_values.to(torch.bfloat16)
                )
                # raw detections in prompt order, then the detector's query order
                masks, scores, owners = [], [], []
                for start in range(0, len(self.prompts), self.batch):
                    part = slice(start, start + self.batch)
                    n = len(self.prompts[part])
                    results = processor.post_process_instance_segmentation(
                        model(
                            vision_embeds=_repeat(vision, n),
                            text_embeds=BaseModelOutputWithPooling(
                                pooler_output=embeds[part]
                            ),
                            attention_mask=text.attention_mask[part],
                        ),
                        threshold=self.threshold,
                        mask_threshold=0.5,
                        target_sizes=sizes * n,
                    )
                    for k, res in enumerate(results, start):
                        if not len(res["scores"]):  # left unresized when empty
                            continue
                        masks.append(res["masks"].bool())
                        scores.append(res["scores"].float())
                        owners += [k] * len(res["scores"])
                empty = torch.zeros(0, height, width, dtype=torch.bool, device=device)
                score = torch.cat([torch.zeros(0, device=device), *scores])
                local = resolve_overlaps(torch.cat([empty, *masks]), score)
                present, view_ids = compact(local, i)
                ids[i] = view_ids.cpu().numpy()
                for raw, new_id in zip(
                    present.tolist(), range(len(present)), strict=True
                ):
                    classes[i * ID_STRIDE + new_id] = prompt_classes[owners[raw]]
                    all_scores[i * ID_STRIDE + new_id] = float(score[raw])
                log.info(
                    "  view %2d: %3d masks (%d detections)",
                    i,
                    len(present),
                    len(owners),
                )
        return Segmentation(ids, classes, scores=all_scores)


def _repeat(vision: Any, n: int) -> Any:  # noqa: ANN401
    """One image's vision features as a batch of n (views, no copy)."""
    return type(vision)(
        **{
            key: tuple(t.expand(n, *t.shape[1:]) for t in value)
            if isinstance(value, tuple)
            else value.expand(n, *value.shape[1:])
            for key, value in vision.items()
        }
    )


def resolve_overlaps(masks: Tensor, scores: Tensor) -> Tensor:
    """Each pixel's owner among masks (N, H, W): the highest score, the first on ties.

    Returns (H, W) int64 indices into the masks, -1 where no mask covers the pixel.
    """
    assert masks.dtype == torch.bool, masks.dtype  # noqa: S101
    assert masks.ndim == 3, masks.shape  # noqa: PLR2004, S101
    assert scores.shape == masks.shape[:1], (scores.shape, masks.shape)  # noqa: S101
    assert bool((scores > 0).all()), "a mask needs a positive score to own a pixel"  # noqa: S101
    if not len(masks):
        return torch.full(masks.shape[1:], NO_INSTANCE, device=masks.device)
    best, owner = torch.where(masks, scores[:, None, None], 0).max(0)
    return torch.where(best > 0, owner, NO_INSTANCE)


def compact(local: Tensor, view: int) -> tuple[Tensor, Tensor]:
    """Ids view * ID_STRIDE + rank for the masks left in `local` (-1: none).

    Raw detections that lost all their pixels to higher-scoring masks get no id, so
    only the visible masks count against ID_STRIDE. Returns the raw indices that are
    left, sorted, and the (H, W) int32 id map.
    """
    present = torch.unique(local[local >= 0])
    if len(present) >= ID_STRIDE:
        msg = f"view {view}: {len(present)} masks, ID_STRIDE is {ID_STRIDE}"
        raise ValueError(msg)
    rank = torch.searchsorted(present, local.clamp(min=0))
    ids = torch.where(local >= 0, view * ID_STRIDE + rank, NO_INSTANCE)
    return present, ids.to(torch.int32)


def main() -> None:
    """Run SAM 3 on prepared views and cache the result next to them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("views", type=Path, nargs="+", help="prepared views (.npz)")
    parser.add_argument(
        "--prompts",
        type=Path,
        action="append",
        help="prompt file (one per line, `prompt = class` for synonyms); repeat to "
        "combine in order; default: the 16 built-in prompts. The file names go into "
        "the output name: sam3image-<stem>+<stem>_<views>.npz",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="minimum detection score (default 0.5); per-mask scores are "
        "saved, other values add -t<value> to the output name",
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

    cuda = torch.cuda.is_available()
    if cuda:
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_gb / total))
    tag = "sam3image"
    options: dict[str, Any] = {}
    if args.prompts:
        synonyms = load_prompts(args.prompts)
        options.update(prompts=tuple(synonyms), synonyms=synonyms)
        tag += "-" + "+".join(p.stem for p in args.prompts)
    if args.threshold != 0.5:  # noqa: PLR2004
        options["threshold"] = args.threshold
        tag += f"-t{args.threshold:g}"
    segmenter = Sam3ImageSegmenter(**options)
    for path in args.views:
        views = Views.load(path)
        log.info(
            "%s: %d views, %d prompts, %d classes",
            views.scene_id,
            len(views),
            len(segmenter.prompts),
            len({segmenter.synonyms.get(p, p) for p in segmenter.prompts}),
        )
        if cuda:
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
            torch.cuda.max_memory_allocated() / 1e9 if cuda else float("nan"),
            out,
        )


if __name__ == "__main__":
    main()
