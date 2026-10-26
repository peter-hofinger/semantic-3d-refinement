"""Scene vocabulary from a VLM: the objects it names become SAM 3 prompts."""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from src.loaders import Views

log = logging.getLogger(__name__)

CHECKPOINT = "Qwen/Qwen3-VL-2B-Instruct"
# generic, fixed before any run; not tuned on the test scenes
QUESTION = (
    "List every distinct kind of object and surface visible in this image, "
    "including structural ones such as wall, floor and ceiling. "
    "Use short, common English nouns in singular form (one to three words), "
    "one per line, without numbering, counts, colors or descriptions."
)
MAX_WORDS = 3


def parse_names(text: str, *, truncated: bool = False) -> list[str]:
    """Object names from one answer: one per line, bullets and numbering removed.

    A truncated answer (token limit hit, usually a repetition loop) loses its last,
    possibly cut-off line.
    """
    lines = text.splitlines()
    names = []
    for line in lines[:-1] if truncated else lines:
        name = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*|\(.*?\)", "", line)
        name = re.sub(r"[^a-z \-]", "", name.lower()).strip(" -")
        if name and len(name.split()) <= MAX_WORDS:
            names.append(name)
    return names


@dataclass
class VlmVocabulary:
    """Asks a VLM for the objects in each view; the scene vocabulary is the union."""

    checkpoint: str = CHECKPOINT
    max_new_tokens: int = 256
    _model: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)

    def _load(self) -> tuple[Any, Any]:
        if self._model is None:
            # 4-bit weights: the bf16 model (4.3 GB) does not fit on a 4 GB GPU
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.checkpoint,
                quantization_config=quantization,
                dtype=torch.bfloat16,
                device_map="cuda",
            ).eval()
            self._processor = AutoProcessor.from_pretrained(self.checkpoint)
        return self._model, self._processor

    @torch.inference_mode()
    def ask(self, image: Image.Image) -> tuple[str, bool]:
        """Answer `QUESTION` for one image (greedy decoding); True if truncated."""
        model, processor = self._load()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": QUESTION},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
        )
        new = out[0, inputs["input_ids"].shape[1] :]
        truncated = len(new) == self.max_new_tokens
        return processor.decode(new, skip_special_tokens=True), truncated

    def scene(self, views: Views) -> tuple[list[str], dict[str, dict]]:
        """Union of the names over all views (first-seen order) and the raw answers."""
        vocabulary: dict[str, None] = {}
        answers = {}
        for name, image in zip(views.names, views.images, strict=True):
            start = time.perf_counter()
            text, truncated = self.ask(Image.fromarray(image))
            answers[name] = {"answer": text, "truncated": truncated}
            names = parse_names(text, truncated=truncated)
            vocabulary.update(dict.fromkeys(names))
            log.info(
                "%s: %d names, %d total, %.1fs",
                name,
                len(names),
                len(vocabulary),
                time.perf_counter() - start,
            )
        return list(vocabulary), answers


def main() -> None:
    """Write a SAM 3 prompt file per scene (and the raw answers) next to the views."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("views", type=Path, nargs="+", help="prepared views (.npz)")
    parser.add_argument("--checkpoint", default=CHECKPOINT)
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
    tag = "vlm-" + re.sub(r"[^a-z0-9]", "", args.checkpoint.split("/")[-1].lower())
    vlm = VlmVocabulary(checkpoint=args.checkpoint)
    for path in args.views:
        views = Views.load(path)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        vocabulary, answers = vlm.scene(views)
        out = path.with_name(f"{tag}_{path.stem}.txt")
        header = f"# {args.checkpoint}, union over {len(views)} views of {path.name}\n"
        out.write_text(header + "\n".join(vocabulary) + "\n", encoding="utf-8")
        out.with_suffix(".json").write_text(
            json.dumps(answers, indent=1), encoding="utf-8"
        )
        log.info(
            "%s: %d names, %.0fs, peak GPU %.2f GB -> %s",
            views.scene_id,
            len(vocabulary),
            time.perf_counter() - start,
            torch.cuda.max_memory_allocated() / 1e9,
            out,
        )


if __name__ == "__main__":
    main()
