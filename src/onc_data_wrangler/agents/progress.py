"""Pipeline progress tracking for the web UI."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StageProgress:
    """Progress state for a single pipeline stage."""

    stage: str
    status: str = "pending"  # pending | running | completed | failed | skipped
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    current: int = 0
    total: int = 0
    message: str = ""


@dataclass
class PipelineRun:
    """Tracks the state of a running pipeline."""

    run_id: str
    config_path: str
    stages: list[StageProgress] = field(default_factory=list)
    status: str = "running"  # running | completed | failed
    error: Optional[str] = None
    log_lines: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update_stage(self, stage: str, **kwargs) -> None:
        """Update a stage's status."""
        with self._lock:
            for s in self.stages:
                if s.stage == stage:
                    for k, v in kwargs.items():
                        setattr(s, k, v)
                    break

    def start_stage(self, stage: str) -> None:
        """Mark a stage as running."""
        self.update_stage(stage, status="running", started_at=time.time())

    def complete_stage(self, stage: str) -> None:
        """Mark a stage as completed."""
        self.update_stage(stage, status="completed", completed_at=time.time())

    def fail_stage(self, stage: str, error: str = "") -> None:
        """Mark a stage as failed."""
        self.update_stage(
            stage, status="failed", completed_at=time.time(), message=error
        )

    def add_log(self, level: str, message: str, stage: str = "") -> None:
        """Append a log entry."""
        with self._lock:
            self.log_lines.append(
                {
                    "timestamp": time.time(),
                    "level": level,
                    "message": message,
                    "stage": stage,
                }
            )

    def to_dict(self) -> dict:
        """Serialize to a dict for API responses."""
        with self._lock:
            return {
                "run_id": self.run_id,
                "config_path": self.config_path,
                "status": self.status,
                "current_stage": next(
                    (s.stage for s in self.stages if s.status == "running"), None
                ),
                "stages": [
                    {
                        "name": s.stage,
                        "status": s.status,
                        "started_at": s.started_at,
                        "completed_at": s.completed_at,
                        "progress": {
                            "current": s.current,
                            "total": s.total,
                            "message": s.message,
                        }
                        if s.total > 0
                        else None,
                    }
                    for s in self.stages
                ],
                "error": self.error,
            }


class ProgressCallback:
    """Callback for pipeline stages to report progress."""

    def __init__(self, run: PipelineRun, stage: str):
        self.run = run
        self.stage = stage

    def update(self, current: int, total: int, message: str = "") -> None:
        self.run.update_stage(
            self.stage, current=current, total=total, message=message
        )

    def log(self, level: str, message: str) -> None:
        self.run.add_log(level, message, self.stage)


class PipelineLogHandler(logging.Handler):
    """Captures log records into a PipelineRun."""

    def __init__(self, run: PipelineRun):
        super().__init__()
        self.run = run

    def emit(self, record: logging.LogRecord) -> None:
        self.run.add_log(
            level=record.levelname,
            message=self.format(record),
            stage=getattr(record, "stage", ""),
        )
