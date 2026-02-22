"""Chunked/serial extraction for long patient note histories.

Splits patient text into token-based chunks and runs iterative extraction,
producing a running summary that is updated with each chunk.
"""

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd

from ..llm.base import LLMClient
from .extractor import Extractor

logger = logging.getLogger(__name__)


def chunk_text_by_tokens(text: str, tokenizer, chunk_size: int = 40000, overlap: int = 200, boundary_marker: str = "\n--- ", boundary_window: int = 500) -> list[str]:
    """Split text into token-based chunks with overlap.

    Tries to split at document boundaries (e.g., note separators)
    when one falls within boundary_window tokens of the split point.

    Args:
        text: Full text to chunk.
        tokenizer: HuggingFace tokenizer instance.
        chunk_size: Maximum tokens per chunk.
        overlap: Overlap tokens between chunks.
        boundary_marker: String marking document boundaries.
        boundary_window: Token window to search for boundaries.

    Returns:
        List of text chunks.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    total = len(token_ids)
    chunks = []
    start = 0

    while start < total:
        end = min(start + chunk_size, total)

        # Try to find a boundary near the split point
        if end < total:
            search_start = max(start, end - boundary_window)
            window_text = tokenizer.decode(token_ids[search_start:end], skip_special_tokens=True)
            boundary_pos = window_text.rfind(boundary_marker)
            if boundary_pos != -1:
                pre_boundary = window_text[:boundary_pos]
                pre_tokens = tokenizer.encode(pre_boundary, add_special_tokens=False)
                end = search_start + len(pre_tokens)

        chunk_text = tokenizer.decode(token_ids[start:end], skip_special_tokens=True)
        chunks.append(chunk_text)
        start = max(start + 1, end - overlap)

    return chunks


def concatenate_patient_notes(patient_df: pd.DataFrame, text_column: str = "text", date_column: str = "date", type_column: str = "note_type") -> str:
    """Concatenate all notes for one patient chronologically.

    Args:
        patient_df: DataFrame with notes for one patient.
        text_column: Column containing note text.
        date_column: Column containing note date.
        type_column: Column containing note type.

    Returns:
        Concatenated text with note boundaries.
    """
    parts = []
    for _, row in patient_df.iterrows():
        note_type = str(row.get(type_column, "unknown")) if type_column in patient_df.columns else "unknown"
        date = str(row.get(date_column, "")) if date_column in patient_df.columns else ""
        text = str(row.get(text_column, ""))
        if len(text) < 10:
            continue
        parts.append("--- " + note_type + " | " + date + " ---\n" + text)
    return "\n\n".join(parts)


class ChunkedExtractor:
    """Chunked extraction pipeline for processing entire patient cohorts.

    Manages checkpointing, parallel processing, and iterative extraction.
    """

    def __init__(self, extractor: Extractor, tokenizer=None, chunk_size: int = 40000, overlap: int = 200, max_retries: int = 10, patient_workers: int = 8, checkpoint_interval: int = 50):
        self.extractor = extractor
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_retries = max_retries
        self.patient_workers = patient_workers
        self.checkpoint_interval = checkpoint_interval

    def extract_patient(self, patient_id: str, patient_text: str) -> dict:
        """Run extraction for a single patient.

        Args:
            patient_id: Patient identifier.
            patient_text: Concatenated patient notes.

        Returns:
            Dict with patient_id, extractions list, and chunks_failed count.
        """
        if self.tokenizer:
            chunks = chunk_text_by_tokens(patient_text, self.tokenizer, self.chunk_size, self.overlap)
        else:
            chunks = [patient_text]
        logger.info("Patient %s: %d chunks", patient_id, len(chunks))
        extractions = self.extractor.extract_iterative(chunks, max_retries=self.max_retries)
        return {"patient_id": patient_id, "extractions": extractions, "num_chunks": len(chunks)}

    def extract_cohort(self, notes_df: pd.DataFrame, output_dir: Path, patient_id_column: str = "record_id", text_column: str = "text", date_column: str = "date", type_column: str = "note_type", resume: bool = False) -> pd.DataFrame:
        """Extract data for an entire cohort of patients.

        Args:
            notes_df: DataFrame with all patient notes.
            output_dir: Directory for checkpoints and output shards.
            patient_id_column: Column with patient identifiers.
            text_column: Column with note text.
            date_column: Column with note dates.
            type_column: Column with note types.
            resume: Whether to resume from existing checkpoint.

        Returns:
            DataFrame with all extractions.
        """
        output_dir = Path(output_dir)
        ckpt = CheckpointManager(output_dir)

        grouped = notes_df.sort_values(by=[patient_id_column]).reset_index(drop=True)
        all_ids = dict(list(grouped.groupby(patient_id_column)))
        all_ids = sorted(all_ids.keys())

        if resume:
            completed = ckpt.load_completed()
        else:
            completed = set()

        pending = [pid for pid in all_ids if pid not in completed]
        logger.info("Cohort: %d total, %d completed, %d pending", len(all_ids), len(completed), len(pending))

        if not pending:
            logger.info("All patients already processed.")
            return ckpt.load_all_results()

        results_buffer = []
        shard_id = ckpt.count_existing_shards()
        processed = 0

        with ThreadPoolExecutor(max_workers=self.patient_workers) as executor:
            future_to_id = {}
            for pid in pending:
                patient_df = grouped[grouped[patient_id_column] == pid]
                patient_text = concatenate_patient_notes(patient_df, text_column, date_column, type_column)
                future = executor.submit(self.extract_patient, str(pid), patient_text)
                future_to_id[future] = pid

            for future in as_completed(future_to_id):
                pid = future_to_id[future]
                try:
                    result = future.result()
                except Exception:
                    logger.exception("Failed to process patient %s", pid)
                    result = {"patient_id": str(pid), "extractions": [], "num_chunks": 0}

                ckpt.append_result(str(pid), result.get("extractions", []))
                results_buffer.append(result)
                processed += 1

                if processed % self.checkpoint_interval == 0:
                    ckpt.save_shard(results_buffer, shard_id)
                    shard_id += 1
                    results_buffer = []

                if processed % 10 == 0:
                    logger.info("Progress: %d / %d", processed, len(pending))

        # Save remaining
        if results_buffer:
            ckpt.save_shard(results_buffer, shard_id)

        logger.info("Done. Processed %d patients.", processed)
        return ckpt.load_all_results()


class CheckpointManager:
    """Thread-safe checkpoint manager using JSONL + parquet shards."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / "checkpoint.jsonl"
        self._lock = threading.Lock()

    def load_completed(self) -> set[str]:
        """Load set of completed patient IDs, reconciled against shard data.

        A patient is considered completed if:
        - Its results appear in a saved parquet shard, OR
        - It was checkpointed with zero extractions (legitimately empty).

        Patients checkpointed with extractions that don't appear in any shard
        are treated as incomplete (crash between checkpoint write and shard
        flush) and will be re-processed.
        """
        shard_ids = set()
        for shard in self.output_dir.glob("shard_*.parquet"):
            try:
                df = pd.read_parquet(shard)
                if "patient_id" in df.columns:
                    shard_ids.update(str(x) for x in df["patient_id"].unique())
            except Exception:
                pass

        checkpoint_ids = set()
        empty_ids = set()
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        pid = str(record["patient_id"])
                        if record.get("num_extractions", 0) == 0:
                            empty_ids.add(pid)
                        else:
                            checkpoint_ids.add(pid)
                    except (json.JSONDecodeError, KeyError):
                        pass

        completed = shard_ids | empty_ids
        orphaned = checkpoint_ids - shard_ids
        if orphaned:
            logger.warning("%d patients in checkpoint but not in shards — will re-process", len(orphaned))

        return completed

    def append_result(self, patient_id: str, extractions: list):
        """Append one patient's results to the checkpoint."""
        record = {"patient_id": patient_id, "num_extractions": len(extractions), "extractions": extractions}
        with self._lock:
            with open(self.checkpoint_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    def save_shard(self, results: list, shard_id: int):
        """Save a batch of results as a parquet shard."""
        rows = []
        for patient_result in results:
            patient_id = patient_result.get("patient_id")
            for ext in patient_result.get("extractions", []):
                if isinstance(ext, dict) and len(ext) == 1:
                    category = next(iter(ext))
                    attrs = ext[category]
                    row = {"patient_id": patient_id, "category": category}
                    row.update(attrs)
                    rows.append(row)

        if not rows:
            return

        df = pd.DataFrame(rows)
        shard_path = self.output_dir / f"shard_{shard_id:04d}.parquet"
        df.to_parquet(shard_path, index=False)
        logger.info("Saved shard %s (%d rows)", shard_path, len(df))

    def count_existing_shards(self) -> int:
        return len(list(self.output_dir.glob("shard_*.parquet")))

    def load_all_results(self) -> pd.DataFrame:
        """Load all parquet shards into a single DataFrame."""
        shards = sorted(self.output_dir.glob("shard_*.parquet"))
        if not shards:
            return pd.DataFrame()
        dfs = [pd.read_parquet(s) for s in shards]
        return pd.concat(dfs, ignore_index=True)
