from pathlib import Path

import pytest
import torch
import yaml

from experiments.scripts.e52a_segment_adapters import adapter_for, kept

CFG = {"models": {"lora_pool_gate": [False, True, True, True]}, "segments": ["gt"],
       "rank": 4, "lora_rank": 2, "grid_tiles": [3, 3]}  # fmt: skip


def saved_run(tmp_path: Path) -> tuple[Path, torch.nn.Module]:
    torch.manual_seed(0)
    adapter = adapter_for(CFG, "lora_pool_gate", torch.device("cpu"))
    for p in adapter.parameters():
        p.data.normal_()
    run = tmp_path / "e87a_main_gt_s0"
    run.mkdir()
    (run / "e52a.yaml").write_text(yaml.safe_dump({"config": CFG}), encoding="utf-8")
    torch.save({"state": adapter.state_dict(), "best_step": 750, "selected_on": "val"},
               run / "lora_pool_gate-gt.pt")  # fmt: skip
    return run, adapter


def test_kept_rebuilds_the_saved_state_from_the_runs_config(tmp_path: Path) -> None:
    run, adapter = saved_run(tmp_path)
    adapters, meta = kept(run, {"grid_tiles": [3, 3]}, torch.device("cpu"))
    ((label, (loaded, kind)),) = adapters.items()
    assert (label, kind) == ("e87a_main_gt_s0/lora_pool_gate-gt", "gt")
    assert meta[label] == {"best_step": 750, "selected_on": "val", "is_zero": False}
    want, got = adapter.state_dict(), loaded.state_dict()
    assert want.keys() == got.keys()
    assert all(torch.equal(want[k], got[k]) for k in want)


def test_kept_refuses_another_grid(tmp_path: Path) -> None:
    run, _ = saved_run(tmp_path)
    with pytest.raises(AssertionError):
        kept(run, {"grid_tiles": [4, 4]}, torch.device("cpu"))
