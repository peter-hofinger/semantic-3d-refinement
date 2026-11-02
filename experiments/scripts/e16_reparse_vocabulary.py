"""E16: rebuild the VLM prompt files from the saved raw answers (no VLM rerun).

The first `vlm-vocabulary` run saved answers as {view: text} and kept the cut-off last
line of answers that hit the token limit (e.g. "c"). This re-tokenizes each answer to
find the truncated ones, rewrites the .json as {view: {answer, truncated}} (the format
`vlm-vocabulary` now writes) and rewrites the prompt file with `parse_names`.

Usage: uv run python experiments/scripts/e16_reparse_vocabulary.py <scene> [<scene> ...]
"""

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

from src.segmentation.vocabulary import CHECKPOINT, VlmVocabulary, parse_names

STEM = "vlm-qwen3vl2binstruct_views_n20_uniform_504"
ROOT = Path("data/processed/scannetpp")

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
limit = VlmVocabulary.max_new_tokens
for scene in sys.argv[1:]:
    path = ROOT / scene / f"{STEM}.json"
    answers = json.loads(path.read_text(encoding="utf-8"))
    vocabulary: dict[str, None] = {}
    for view, entry in answers.items():
        text = entry["answer"] if isinstance(entry, dict) else entry
        truncated = len(tokenizer(text)["input_ids"]) >= limit
        answers[view] = {"answer": text, "truncated": truncated}
        vocabulary.update(dict.fromkeys(parse_names(text, truncated=truncated)))
    path.write_text(json.dumps(answers, indent=1), encoding="utf-8")
    txt = path.with_suffix(".txt")
    header = txt.read_text(encoding="utf-8").splitlines()[0]
    txt.write_text(header + "\n" + "\n".join(vocabulary) + "\n", encoding="utf-8")
    num = sum(a["truncated"] for a in answers.values())
    print(f"{scene}: {num}/{len(answers)} answers truncated, {len(vocabulary)} names")
    print("  " + "; ".join(vocabulary))
