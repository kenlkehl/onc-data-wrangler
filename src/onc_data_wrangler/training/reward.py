"""Reward function for GRPO fine-tuning.

Computes a reward score by:
1. Extracting structured data from a generated summary using the target ontologies
2. Comparing the extracted fields to silver-standard reference extractions
3. Returning an F1-like score based on field-level matches
"""

import logging
from typing import Optional

from ..extraction.extractor import Extractor, parse_json_list
from ..llm.base import LLMClient

logger = logging.getLogger(__name__)


class RewardFunction:
    """Computes reward scores for generated clinical summaries.

    The reward measures how well a downstream structured extractor can
    recover known structured fields from the summary alone.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        ontology_ids: list[str],
        silver_labels: dict[str, list],
        cancer_type: str = "generic",
    ):
        self.extractor = Extractor(
            llm_client=llm_client,
            ontology_ids=ontology_ids,
            cancer_type=cancer_type,
        )
        self.silver_labels = silver_labels
        self.ontology_ids = ontology_ids

    def compute_reward(
        self,
        patient_id: str,
        generated_summary: str,
        max_tokens: int = 8000,
    ) -> float:
        """Compute reward for a generated summary.

        Args:
            patient_id: Patient identifier for looking up silver labels.
            generated_summary: The generated free-text summary.
            max_tokens: Max tokens for the extraction LLM call.

        Returns:
            Float reward in [0, 1] based on field-level F1 score.
        """
        reference = self.silver_labels.get(patient_id, [])
        if not reference:
            logger.warning("No silver labels for patient %s, returning 0.0", patient_id)
            return 0.0

        # Extract structured data from the generated summary
        try:
            extracted = self.extractor.extract_from_text(
                generated_summary, max_tokens=max_tokens,
            )
        except Exception:
            logger.exception("Extraction from summary failed for patient %s", patient_id)
            return 0.0

        if not extracted:
            return 0.0

        # Compare extracted vs reference
        return compute_field_f1(extracted, reference)


def compute_field_f1(extracted: list[dict], reference: list[dict]) -> float:
    """Compute field-level F1 score between extracted and reference data.

    Flattens both into sets of (category, field, value) triples and computes
    precision, recall, and F1.

    Args:
        extracted: Structured extraction from the summary.
        reference: Silver-standard extraction from full notes.

    Returns:
        F1 score in [0, 1].
    """
    extracted_fields = _flatten_to_field_set(extracted)
    reference_fields = _flatten_to_field_set(reference)

    if not reference_fields:
        return 1.0 if not extracted_fields else 0.0

    # Field presence matching (category + field name)
    ref_keys = {(cat, field) for cat, field, _ in reference_fields}
    ext_keys = {(cat, field) for cat, field, _ in reference_fields}

    # Value matching: for each reference field, check if extracted has a matching value
    ref_triples = reference_fields
    ext_triples = extracted_fields

    # Compute matches: a reference triple is "matched" if the extracted set
    # contains the same (category, field) with the same normalized value
    matches = 0
    for cat, fld, val in ref_triples:
        for e_cat, e_fld, e_val in ext_triples:
            if _category_match(cat, e_cat) and _field_match(fld, e_fld):
                if _value_match(val, e_val):
                    matches += 1
                    break
                else:
                    # Partial credit for having the right field but wrong value
                    matches += 0.25
                    break

    # Recall: fraction of reference fields recovered
    recall = matches / len(ref_triples) if ref_triples else 0.0

    # Precision: fraction of extracted fields that match something in reference
    precision_matches = 0
    for e_cat, e_fld, e_val in ext_triples:
        for cat, fld, val in ref_triples:
            if _category_match(cat, e_cat) and _field_match(fld, e_fld):
                if _value_match(val, e_val):
                    precision_matches += 1
                else:
                    precision_matches += 0.25
                break

    precision = precision_matches / len(ext_triples) if ext_triples else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _flatten_to_field_set(extractions: list[dict]) -> set[tuple[str, str, str]]:
    """Flatten extraction list into (category, field_name, value_str) triples."""
    fields = set()
    for entry in extractions:
        if not isinstance(entry, dict) or len(entry) != 1:
            continue
        category = next(iter(entry))
        attrs = entry[category]
        if not isinstance(attrs, dict):
            continue
        for field_name, value in attrs.items():
            if value is None:
                continue
            val_str = _normalize_value(value)
            if val_str:
                fields.add((category.lower().strip(), field_name.lower().strip(), val_str))
    return fields


def _normalize_value(value) -> str:
    """Normalize a value to a comparable string."""
    if isinstance(value, list):
        return "; ".join(str(v).lower().strip() for v in value if v is not None)
    return str(value).lower().strip()


def _category_match(cat1: str, cat2: str) -> bool:
    """Check if two category names match (fuzzy)."""
    return cat1 == cat2


def _field_match(field1: str, field2: str) -> bool:
    """Check if two field names match."""
    return field1 == field2


def _value_match(val1: str, val2: str) -> bool:
    """Check if two values match (with some fuzzy tolerance)."""
    if val1 == val2:
        return True
    # Check containment for longer values
    if len(val1) > 3 and len(val2) > 3:
        if val1 in val2 or val2 in val1:
            return True
    return False
