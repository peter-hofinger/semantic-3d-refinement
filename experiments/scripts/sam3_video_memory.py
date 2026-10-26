"""E3: SAM 3 video mode GPU memory and time per frame for a given prompt set.

uv run python experiments/scripts/sam3_video_memory.py 7b6477cb95 chair
uv run python experiments/scripts/sam3_video_memory.py 7b6477cb95 wall chair

Capped at 3.5 GB so it fails with OOM instead of spilling into Windows shared memory.
Run with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to match the logged numbers.
"""

import sys
import time
from pathlib import Path

import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor

from src.loaders import Views

total = torch.cuda.get_device_properties(0).total_memory / 1e9
torch.cuda.set_per_process_memory_fraction(min(1.0, 3.5 / total))
scene, prompts = sys.argv[1], sys.argv[2:]
views = Views.load(
    Path("data/processed/scannetpp") / scene / "views_n20_uniform_504.npz"
)

model = Sam3VideoModel.from_pretrained("facebook/sam3", dtype=torch.bfloat16)
model = model.to("cuda").eval()
processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
print(f"model on GPU: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

session = processor.init_video_session(
    video=list(views.images),
    inference_device="cuda",
    inference_state_device="cpu",
    processing_device="cpu",
    video_storage_device="cpu",
    dtype=torch.bfloat16,
)
processor.add_text_prompt(session, prompts)
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
with torch.no_grad():
    for out in model.propagate_in_video_iterator(session):
        res = processor.postprocess_outputs(session, out)
        print(
            f"frame {out.frame_idx:2d}  {time.perf_counter() - t0:6.1f}s  "
            f"objects {len(res['object_ids']):3d}  "
            f"peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB",
            flush=True,
        )
