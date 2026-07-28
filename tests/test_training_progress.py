import json

import pytest

from tiny_transformer.training import (
    TrainingLogger,
    create_local_run_dir,
    resolve_gradient_accumulation_steps,
)


def test_training_logger_writes_local_files(tmp_path):
    run_dir = create_local_run_dir(tmp_path, "test run")
    logger = TrainingLogger(run_dir, total_steps=2, start_step=0, enable_tqdm=False)
    logger.save_config({"model": {"n_layer": 2}})
    logger.write("training started")
    logger.log_metrics("train", 0, {"loss": 1.5})
    logger.advance()
    logger.close()

    assert "training started" in (run_dir / "train.log").read_text(encoding="utf-8")
    metrics = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8"))
    assert metrics["event"] == "train"
    assert metrics["step"] == 0
    assert metrics["loss"] == 1.5
    assert json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["model"]["n_layer"] == 2


def test_accumulation_preserves_global_tokens_across_world_sizes():
    one_gpu = resolve_gradient_accumulation_steps(128, 256, 1, 2, 65536)
    two_gpu = resolve_gradient_accumulation_steps(128, 256, 2, 2, 65536)
    assert one_gpu == 2
    assert two_gpu == 1


def test_accumulation_rejects_incompatible_target():
    with pytest.raises(ValueError):
        resolve_gradient_accumulation_steps(128, 256, 2, 2, 70000)
