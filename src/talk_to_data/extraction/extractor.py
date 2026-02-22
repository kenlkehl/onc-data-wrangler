"""Unified extraction using ontology-defined fields.

Takes ontology IDs + optional cancer type to generate extraction prompts.
Supports both vLLM and Claude backends via the LLM abstraction layer.
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
        return parsed

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
        instructions = self.build_extraction_prompt(cancer_type)
        running = []

        for i, chunk_text in enumerate(texts):
            if i == 0:
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
                    logger.exception("Chunk %d/%d: LLM call failed (attempt %d/%d)", i + 1, len(texts), attempt + 1, max_retries)
                else:
                    if parsed is None:
                        logger.warning("Chunk %d/%d: JSON parse failed (attempt %d/%d)", i + 1, len(texts), attempt + 1, max_retries)

            if parsed is not None:
                running = parsed
            else:
                logger.warning("Chunk %d/%d: all retries failed, keeping previous extraction", i + 1, len(texts))

        return running


def parse_json_list(text: str) -> list[dict]:
    """Best-effort parse of a JSON array from LLM output."""
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

    return []
