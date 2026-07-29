from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def create_local_run_dir(
    project_root: Path,
    run_name: str,
    resume_dir: str | None = None,
) -> Path:
    if resume_dir is not None:
        run_dir = Path(resume_dir)
        if not run_dir.is_absolute():
            run_dir = project_root / run_dir
        return run_dir

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", run_name).strip("-") or "run"
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return project_root / "logs" / safe_name / stamp


class TrainingLogger:
    """Terminal progress plus durable local text and JSONL logs."""

    def __init__(
        self,
        run_dir: Path,
        total_steps: int,
        start_step: int,
        enable_tqdm: bool = True,
        description: str = "pretrain",
    ) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.console_path = self.run_dir / "train.log"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self._console = self.console_path.open("a", encoding="utf-8", buffering=1)
        self._metrics = self.metrics_path.open("a", encoding="utf-8", buffering=1)
        self.progress = tqdm(
            total=total_steps,
            initial=start_step,
            desc=description,
            unit="step",
            dynamic_ncols=True,
            smoothing=0.05,
            disable=not enable_tqdm,
        )

    def save_config(self, config: dict) -> None:
        path = self.run_dir / "config.json"
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def write(self, message: str) -> None:
        line = f"[{_timestamp()}] {message}"
        if self.progress.disable:
            print(line, flush=True)
        else:
            self.progress.write(line)
        self._console.write(line + "\n")

    def log_metrics(self, event: str, step: int, metrics: dict) -> None:
        payload = {"time": _timestamp(), "event": event, "step": step, **metrics}
        self._metrics.write(json.dumps(payload, ensure_ascii=False, default=float) + "\n")

    def advance(self, steps: int = 1) -> None:
        self.progress.update(steps)

    def set_postfix(self, **values) -> None:
        formatted = {}
        for key, value in values.items():
            if value is None:
                continue
            if isinstance(value, float):
                formatted[key] = f"{value:.4g}"
            else:
                formatted[key] = value
        self.progress.set_postfix(formatted, refresh=False)

    def close(self) -> None:
        self.progress.close()
        self._metrics.close()
        self._console.close()
