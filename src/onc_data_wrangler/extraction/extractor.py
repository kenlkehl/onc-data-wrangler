"""Unified extraction using ontology-defined fields.

Takes ontology IDs + optional cancer type to generate extraction prompts.
Supports both vLLM and Claude backends via the LLM abstraction layer.

When the ``clinical_summary`` ontology is the sole ontology, extraction
switches to free-text summary mode (see ``SummaryExtractor``).
"""

import json
import logging
import textwrap
from typing import Any, Optional

from ..llm.base import LLMClient, LLMResponse
from ..ontologies import MultiOntologyExtractor, OntologyRegistry

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM = textwrap.dedent("""\
    You are a clinical data extraction system. Your job is to extract
    structured information from clinical notes, imaging reports, and
    pathology reports according to the specified ontology schemas.

    Rules:
    - Your output must be valid JSON.
    - Output a JSON list of dictionaries. Each dictionary has a single key
      (the category name) whose value is a dictionary of subfields.
    - Never invent information not present in the document.
    - If information is missing, omit the subfield entirely.
    - All dates should be formatted as YYYY-MM-DD when possible.
    - Never output introductory text, concluding text, or commentary.
""")

FIRST_CHUNK_SUFFIX = textwrap.dedent("""\
    Here is the clinical document for this patient. Extract all relevant
    structured information as a JSON list.

    <DOCUMENT>
    {chunk_text}
    </DOCUMENT>

    Now, think carefully, then generate your JSON extraction.""")

UPDATE_CHUNK_TEMPLATE = textwrap.dedent("""\
    You previously extracted the following structured data from earlier
    portions of this patient's clinical record:

    <CURRENT_EXTRACTION>
    {running_json}
    </CURRENT_EXTRACTION>

    {instructions}

    Here is the next portion of the patient's clinical record:

    <DOCUMENT>
    {chunk_text}
    </DOCUMENT>

    Review this new text and update the extraction:
    - If a concept is already captured, MERGE any new details into the
      existing entry. Do NOT create a duplicate.
    - If a genuinely new concept appears, ADD it to the list.
    - If nothing new is found, return the existing extraction unchanged.

    Output the complete updated JSON list. Valid JSON only, no commentary.
    Think carefully, then generate your JSON extraction.""")


class Extractor:
    """Ontology-driven clinical data extractor.

    Uses the ontology system to define extraction fields rather than
    hardcoded category lists.
    """

    def __init__(self, llm_client: LLMClient, ontology_ids: list[str], cancer_type: Optional[str] = "generic"):
        self.llm_client = llm_client
        self.ontology_ids = ontology_ids
        self.cancer_type = cancer_type
        if not ontology_ids:
            ontology_ids = ["naaccr"]
        self.multi_extractor = MultiOntologyExtractor(ontology_ids)

    def build_extraction_prompt(self, cancer_type: Optional[str] = None) -> str:
        """Build the extraction instructions from ontology definitions."""
        ct = cancer_type or self.cancer_type
        ontology_prompt = self.multi_extractor.format_combined_prompt(ct)
        return EXTRACTION_SYSTEM + "\n" + ontology_prompt

    def extract_from_text(self, text: str, cancer_type: Optional[str] = None, max_tokens: Optional[int] = 8000) -> list[dict]:
        """Extract structured data from a single text document.

        Args:
            text: Clinical document text.
            cancer_type: Override cancer type for site-specific items.
            max_tokens: Maximum tokens for LLM response.

        Returns:
            List of extraction dictionaries.
        """
        instructions = self.build_extraction_prompt(cancer_type)
        prompt = instructions + FIRST_CHUNK_SUFFIX.format(chunk_text=text)
        response = self.llm_client.generate(prompt, max_tokens=max_tokens)
        parsed = parse_json_list(response.text)
        return parsed if parsed is not None else []

    def extract_single_chunk(self, chunk_text: str, running: Optional[list[dict]] = None, chunk_index: int = 0, total_chunks: int = 1, cancer_type: Optional[str] = None, max_tokens: Optional[int] = 8000, max_retries: int = 3) -> list[dict]:
        """Extract from a single chunk, given the running state from prior chunks.

        Args:
            chunk_text: Text of this chunk.
            running: Cumulative extraction from previous chunks (empty for first chunk).
            chunk_index: 0-based index of this chunk.
            total_chunks: Total number of chunks for this patient (for logging).
            cancer_type: Override cancer type.
            max_tokens: Max tokens per LLM call.
            max_retries: Retries on parse failure.

        Returns:
            Updated cumulative extraction. On total failure, returns running unchanged.
        """
        if running is None:
            running = []

        instructions = self.build_extraction_prompt(cancer_type)
        if chunk_index == 0 and not running:
            prompt = instructions + FIRST_CHUNK_SUFFIX.format(chunk_text=chunk_text)
        else:
            running_json = json.dumps(running, indent=1)
            prompt = UPDATE_CHUNK_TEMPLATE.format(
                running_json=running_json,
                instructions=instructions,
                chunk_text=chunk_text,
            )

        parsed = None
        for attempt in range(max_retries):
            try:
                response = self.llm_client.generate(prompt, max_tokens=max_tokens)
                parsed = parse_json_list(response.text)
                if parsed is not None:
                    break
            except Exception:
                logger.exception("Chunk %d/%d: LLM call failed (attempt %d/%d)", chunk_index + 1, total_chunks, attempt + 1, max_retries)
            else:
                if parsed is None:
                    logger.warning("Chunk %d/%d: JSON parse failed (attempt %d/%d)", chunk_index + 1, total_chunks, attempt + 1, max_retries)

        if parsed is not None:
            return parsed
        else:
            logger.warning("Chunk %d/%d: all retries failed, keeping previous extraction", chunk_index + 1, total_chunks)
            return running

    def extract_iterative(self, texts: list[str], cancer_type: Optional[str] = None, max_tokens: Optional[int] = 8000, max_retries: int = 3) -> list[dict]:
        """Extract from multiple text chunks iteratively.

        Processes chunks sequentially, maintaining a running extraction
        that is updated with each new chunk.

        Args:
            texts: List of text chunks in chronological order.
            cancer_type: Override cancer type.
            max_tokens: Maximum tokens per LLM call.
            max_retries: Retries per chunk on parse failure.

        Returns:
            Final merged extraction as a list of dicts.
        """
        running = []
        for i, chunk_text in enumerate(texts):
            running = self.extract_single_chunk(
                chunk_text, running, i, len(texts),
                cancer_type, max_tokens, max_retries,
            )
        return running


class SummaryExtractor:
    """Free-text clinical summary extractor.

    Produces a running free-text summary instead of structured JSON.
    Uses the clinical_summary ontology's prompt templates for iterative
    summarization across chunks.

    The running state is a plain string (the summary so far) rather than
    a list of dicts.  To fit into the same ``ChunkedExtractor`` pipeline,
    results are wrapped as ``[{"clinical_summary": {"summary": text}}]``.
    """

    def __init__(self, llm_client: LLMClient, cancer_type: Optional[str] = "generic"):
        self.llm_client = llm_client
        self.cancer_type = cancer_type
        self._ontology = OntologyRegistry.get("clinical_summary")
        # Import prompt templates from the ontology module
        from ..ontologies.builtins.clinical_summary.ontology import (
            SUMMARY_FIRST_CHUNK,
            SUMMARY_UPDATE_CHUNK,
        )
        self._first_chunk_template = SUMMARY_FIRST_CHUNK
        self._update_chunk_template = SUMMARY_UPDATE_CHUNK

    def _system_prompt(self) -> str:
        return self._ontology.format_for_prompt(self.cancer_type)

    def extract_from_text(self, text: str, cancer_type: Optional[str] = None, max_tokens: Optional[int] = 8000) -> list[dict]:
        """Summarise a single text document."""
        prompt = self._first_chunk_template.format(
            system_prompt=self._system_prompt(),
            chunk_text=text,
        )
        response = self.llm_client.generate(prompt, max_tokens=max_tokens)
        return _wrap_summary(response.text)

    def extract_single_chunk(self, chunk_text: str, running: Optional[list[dict]] = None, chunk_index: int = 0, total_chunks: int = 1, cancer_type: Optional[str] = None, max_tokens: Optional[int] = 8000, max_retries: int = 3) -> list[dict]:
        """Summarise a single chunk, updating the running summary."""
        prior_summary = _unwrap_summary(running)

        if chunk_index == 0 and not prior_summary:
            prompt = self._first_chunk_template.format(
                system_prompt=self._system_prompt(),
                chunk_text=chunk_text,
            )
        else:
            prompt = self._update_chunk_template.format(
                system_prompt=self._system_prompt(),
                prior_summary=prior_summary,
                chunk_text=chunk_text,
            )

        for attempt in range(max_retries):
            try:
                response = self.llm_client.generate(prompt, max_tokens=max_tokens)
                summary_text = response.text.strip()
                if summary_text:
                    return _wrap_summary(summary_text)
            except Exception:
                logger.exception(
                    "Summary chunk %d/%d: LLM call failed (attempt %d/%d)",
                    chunk_index + 1, total_chunks, attempt + 1, max_retries,
                )

        logger.warning(
            "Summary chunk %d/%d: all retries failed, keeping previous summary",
            chunk_index + 1, total_chunks,
        )
        return running if running else _wrap_summary("")

    def extract_iterative(self, texts: list[str], cancer_type: Optional[str] = None, max_tokens: Optional[int] = 8000, max_retries: int = 3) -> list[dict]:
        """Iteratively summarise multiple chunks."""
        running: list[dict] = []
        for i, chunk_text in enumerate(texts):
            running = self.extract_single_chunk(
                chunk_text, running, i, len(texts),
                cancer_type, max_tokens, max_retries,
            )
        return running


def _wrap_summary(text: str) -> list[dict]:
    """Wrap a plain-text summary into the extraction list format."""
    return [{"clinical_summary": {"summary": text.strip()}}]


def _unwrap_summary(running: Optional[list[dict]]) -> str:
    """Extract the plain-text summary from the extraction list format."""
    if not running:
        return ""
    for entry in running:
        if isinstance(entry, dict) and "clinical_summary" in entry:
            return entry["clinical_summary"].get("summary", "")
    return ""


def is_summary_only(ontology_ids: list[str]) -> bool:
    """Check if the ontology list consists solely of free-text ontologies."""
    if not ontology_ids:
        return False
    for oid in ontology_ids:
        ont = OntologyRegistry.get(oid)
        if not ont.is_free_text:
            return False
    return True


def create_extractor(llm_client: LLMClient, ontology_ids: list[str], cancer_type: Optional[str] = "generic"):
    """Factory that returns a SummaryExtractor or Extractor based on ontology types."""
    if is_summary_only(ontology_ids):
        return SummaryExtractor(llm_client, cancer_type)
    return Extractor(llm_client, ontology_ids, cancer_type)


def parse_json_list(text: str) -> list[dict] | None:
    """Best-effort parse of a JSON array from LLM output.

    Returns None on failure so callers can distinguish parse errors
    from a legitimately empty extraction ([]).
    """
    text = text.strip()

    # Handle markdown code blocks
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            part = parts[1]
            if part.lower().startswith("json"):
                part = part[4:]
            text = part.strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try finding JSON array boundaries
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return None
