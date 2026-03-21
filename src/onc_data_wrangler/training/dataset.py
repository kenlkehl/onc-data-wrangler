"""Dataset builder for GRPO training.

Builds a HuggingFace Dataset from patient notes and silver-standard
extractions. Each item contains the prompt (clinical notes with the
summary ontology instructions) and the reference extraction for reward
computation.
"""

import logging
from pathlib import Path

import pandas as pd

from ..extraction.chunked import concatenate_patient_notes

logger = logging.getLogger(__name__)


def build_training_dataset(
    notes_df: pd.DataFrame,
    silver_labels: dict[str, list],
    tokenizer,
    patient_id_column: str = "record_id",
    text_column: str = "text",
    date_column: str = "date",
    type_column: str = "note_type",
    max_prompt_tokens: int = 40000,
    max_patients: int | None = None,
) -> list[dict]:
    """Build training samples from notes and silver labels.

    Each sample contains:
    - patient_id: str
    - prompt: str (the formatted prompt for summary generation)
    - reference_extraction: list[dict] (silver-standard structured extraction)

    Args:
        notes_df: DataFrame with patient notes.
        silver_labels: Dict mapping patient_id -> extraction list.
        tokenizer: HuggingFace tokenizer for truncation.
        patient_id_column: Column with patient IDs.
        text_column: Column with note text.
        date_column: Column with note dates.
        type_column: Column with note types.
        max_prompt_tokens: Maximum tokens for the notes portion of the prompt.
        max_patients: Optional limit on number of patients.

    Returns:
        List of training sample dicts.
    """
    from ..ontologies import OntologyRegistry
    summary_ont = OntologyRegistry.get("clinical_summary")
    system_prompt = summary_ont.format_for_prompt()

    from ..ontologies.builtins.clinical_summary.ontology import SUMMARY_FIRST_CHUNK
    prompt_template = SUMMARY_FIRST_CHUNK

    # Group notes by patient
    grouped = notes_df.sort_values(by=[patient_id_column]).reset_index(drop=True)
    patient_groups = dict(list(grouped.groupby(patient_id_column)))

    samples = []
    patient_ids = sorted(patient_groups.keys())
    if max_patients is not None:
        patient_ids = patient_ids[:max_patients]

    for pid in patient_ids:
        pid_str = str(pid)
        if pid_str not in silver_labels:
            logger.debug("Skipping patient %s: no silver labels", pid_str)
            continue

        pdf = patient_groups[pid]
        full_text = concatenate_patient_notes(pdf, text_column, date_column, type_column)

        # Truncate notes if too long
        if tokenizer is not None:
            tokens = tokenizer.encode(full_text, add_special_tokens=False)
            if len(tokens) > max_prompt_tokens:
                full_text = tokenizer.decode(tokens[:max_prompt_tokens], skip_special_tokens=True)

        prompt = prompt_template.format(
            system_prompt=system_prompt,
            chunk_text=full_text,
        )

        samples.append({
            "patient_id": pid_str,
            "prompt": prompt,
            "reference_extraction": silver_labels[pid_str],
        })

    logger.info("Built %d training samples from %d patients with silver labels",
                len(samples), len(silver_labels))
    return samples
