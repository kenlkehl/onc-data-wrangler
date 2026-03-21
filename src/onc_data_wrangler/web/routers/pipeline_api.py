"""Pipeline execution API router."""

from __future__ import annotations

import logging
import threading
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ...agents.progress import PipelineLogHandler, PipelineRun, StageProgress

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

# In-memory store of pipeline runs
_runs: dict[str, PipelineRun] = {}

ALL_STAGES = (
    "cohort",
    "prepare_notes",
    "extract",
    "harmonize",
    "propose_tables",
    "database",
    "metadata",
)


class PipelineRunRequest(BaseModel):
    config_path: str
    stages: list[str] | None = None
    resume: bool = False


def _run_pipeline_thread(
    run: PipelineRun,
    config_path: str,
    stages: list[str] | None,
    resume: bool,
) -> None:
    """Run the pipeline in a background thread."""
    # Attach a log handler to capture all pipeline logs
    pipeline_logger = logging.getLogger("onc_data_wrangler")
    handler = PipelineLogHandler(run)
    handler.setFormatter(logging.Formatter("%(name)s %(message)s"))
    pipeline_logger.addHandler(handler)

    try:
        from ...agents.pipeline import run_pipeline

        run_pipeline(
            config_path=config_path,
            stages=stages,
            resume=resume,
            progress=run,
        )
        run.status = "completed"
        run.add_log("INFO", "Pipeline completed successfully")
    except Exception as exc:
        run.status = "failed"
        run.error = str(exc)
        run.add_log("ERROR", f"Pipeline failed: {exc}")
        logger.error("Pipeline run %s failed: %s", run.run_id, exc, exc_info=True)
    finally:
        pipeline_logger.removeHandler(handler)


@router.post("/run")
async def start_pipeline(req: PipelineRunRequest) -> JSONResponse:
    """Launch a pipeline run in a background thread."""
    run_id = str(uuid.uuid4())[:8]

    # Determine which stages will run
    stage_names = req.stages or list(ALL_STAGES)
    stages = [StageProgress(stage=name) for name in stage_names]

    run = PipelineRun(
        run_id=run_id,
        config_path=req.config_path,
        stages=stages,
    )
    _runs[run_id] = run

    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(run, req.config_path, req.stages, req.resume),
        daemon=True,
    )
    thread.start()

    run.add_log("INFO", f"Pipeline started (run_id={run_id})")
    return JSONResponse({"run_id": run_id})


@router.get("/{run_id}/status")
async def get_pipeline_status(run_id: str) -> JSONResponse:
    """Get the current status of a pipeline run."""
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(404, f"Pipeline run not found: {run_id}")
    return JSONResponse(run.to_dict())


@router.get("/{run_id}/logs")
async def stream_pipeline_logs(run_id: str) -> StreamingResponse:
    """Stream pipeline log entries via SSE."""
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(404, f"Pipeline run not found: {run_id}")

    async def event_generator():
        cursor = 0
        import asyncio

        while True:
            with run._lock:
                new_logs = run.log_lines[cursor:]
                cursor = len(run.log_lines)
                is_done = run.status != "running"

            for entry in new_logs:
                import json

                data = json.dumps(entry)
                yield f"event: log\ndata: {data}\n\n"

            if is_done and not new_logs:
                yield f"event: done\ndata: {{}}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/runs")
async def list_runs() -> JSONResponse:
    """List recent pipeline runs."""
    runs = []
    for run in _runs.values():
        runs.append(
            {
                "run_id": run.run_id,
                "config_path": run.config_path,
                "status": run.status,
            }
        )
    return JSONResponse(runs)
