"""Domain-group-based extraction using ontology-driven fields.

Replaces single-pass and multi-pass extraction with the registry pattern:
sequential domain groups, per-field {value, confidence, evidence}, code
resolution, and higher-confidence-wins merging across chunks.

When the ``clinical_summary`` ontology is the sole ontology, extraction
switches to free-text summary mode (see ``SummaryExtractor``).
"""

import json
import logging
from typing import Any, Optional

from ..llm.base import LLMClient
from ..ontologies import OntologyRegistry
from .result import (
    ExtractionResult,
    HIGH_CONFIDENCE_THRESHOLD,
    merge_results,
    split_items_into_batches,
)
from .schema_builder import SchemaBuilder
from .code_resolver import GenericCodeResolver
from .domain_groups import (
    build_naaccr_domain_groups,
    build_generic_domain_groups,
    build_prior_state_block,
    build_prior_narratives_block,
    CHUNK_USER_TEMPLATE,
    NARRATIVE_USER_TEMPLATE,
)
from ..ontologies.protocols import DomainGroup

logger = logging.getLogger(__name__)

# Default items per LLM call
DEFAULT_ITEMS_PER_CALL = 50


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def parse_json_object(text: str) -> dict | None:
    """Best-effort parse of a JSON object from LLM output."""
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            part = parts[1]
            if part.lower().startswith("json"):
                part = part[4:]
            text = part.strip()

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        # Unwrap single-element array
        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
            return result[0]
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None


def parse_json_list(text: str) -> list[dict] | None:
    """Best-effort parse of a JSON array from LLM output.

    Returns None on failure so callers can distinguish parse errors
    from a legitimately empty extraction ([]).
    """
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            part = parts[1]
            if part.lower().startswith("json"):
                part = part[4:]
            text = part.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
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


# ---------------------------------------------------------------------------
# Main Extractor
# ---------------------------------------------------------------------------

class Extractor:
    """Domain-group-based clinical data extractor.

    Processes extraction in sequential domain groups (demographics -> staging
    -> surgery -> ...) with batching, per-field confidence/evidence tracking,
    and code resolution. Maintains the same public interface as the previous
    Extractor for backward compatibility with ChunkedExtractor.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        ontology_ids: list[str],
        cancer_type: Optional[str] = "generic",
        items_per_call: int = DEFAULT_ITEMS_PER_CALL,
    ):
        self.llm_client = llm_client
        self.ontology_ids = ontology_ids if ontology_ids else ["naaccr"]
        self.cancer_type = cancer_type or "generic"
        self.items_per_call = items_per_call
        self._schema_builder = SchemaBuilder()

        # Initialize per-ontology resources
        self._ontologies: dict[str, Any] = {}
        self._code_resolvers: dict[str, Any] = {}
        self._domain_groups: dict[str, list[DomainGroup]] = {}
        self._item_registries: dict[str, dict[str, Any]] = {}  # field_id -> item

        for oid in self.ontology_ids:
            ont = OntologyRegistry.get(oid)
            self._ontologies[oid] = ont
            self._init_ontology(oid, ont)

    def _init_ontology(self, oid: str, ont: Any) -> None:
        """Initialize code resolver, domain groups, and item registry for one ontology."""
        # Code resolver
        resolver = getattr(ont, "get_code_resolver", lambda: None)()
        if resolver is None:
            # Build generic resolver from ontology's valid_values
            all_items = self._collect_all_items(ont)
            resolver = GenericCodeResolver.from_data_items(all_items)
        self._code_resolvers[oid] = resolver

        # Domain groups
        if oid == "naaccr":
            from ..ontologies.builtins.naaccr.schema_registry import SchemaRegistry
            self._naaccr_schema_registry = SchemaRegistry()
            self._domain_groups[oid] = build_naaccr_domain_groups()
        else:
            self._domain_groups[oid] = build_generic_domain_groups(ont)

        # Item registry: field_id -> item object
        registry: dict[str, Any] = {}
        all_items = self._collect_all_items(ont)
        for item in all_items:
            fid = self._get_field_id(item)
            registry[fid] = item
        self._item_registries[oid] = registry

    def _collect_all_items(self, ont: Any) -> list:
        """Collect all DataItem objects from an ontology."""
        items = []
        for cat in ont.get_base_items():
            items.extend(cat.items)
        try:
            for cat in ont.get_site_specific_items("generic"):
                items.extend(cat.items)
        except Exception:
            pass
        return items

    @staticmethod
    def _get_field_id(item: Any) -> str:
        """Get the field_id from an item (NAACCR or generic)."""
        if hasattr(item, "field_id"):
            return str(item.field_id)
        if hasattr(item, "item_number"):
            return str(item.item_number)
        return getattr(item, "json_field", None) or getattr(item, "id", str(id(item)))

    # ------------------------------------------------------------------
    # Public interface (same as old Extractor)
    # ------------------------------------------------------------------

    def extract_from_text(
        self,
        text: str,
        cancer_type: Optional[str] = None,
        max_tokens: Optional[int] = 8000,
    ) -> list[dict]:
        """Extract structured data from a single text document."""
        return self.extract_single_chunk(text, [], 0, 1, cancer_type, max_tokens)

    def extract_single_chunk(
        self,
        chunk_text: str,
        running: Optional[list[dict]] = None,
        chunk_index: int = 0,
        total_chunks: int = 1,
        cancer_type: Optional[str] = None,
        max_tokens: Optional[int] = 8000,
        max_retries: int = 3,
    ) -> list[dict]:
        """Extract from a single chunk using domain-group processing.

        Returns list[dict] in the format ``[{category: {field: value}}]``
        for backward compatibility with ChunkedExtractor and downstream.
        """
        if running is None:
            running = []

        ct = cancer_type or self.cancer_type

        # Convert running list to internal ExtractionResult state
        internal_state = self._list_to_internal(running)

        # Process each ontology's domain groups
        for oid in self.ontology_ids:
            ont = self._ontologies[oid]
            resolver = self._code_resolvers[oid]
            groups = self._domain_groups[oid]
            item_registry = self._item_registries.get(oid, {})

            context: dict[str, str] = {"cancer_type": ct}

            for group in groups:
                try:
                    group_results = self._extract_domain_group(
                        group=group,
                        chunk_text=chunk_text,
                        internal_state=internal_state,
                        ont=ont,
                        oid=oid,
                        resolver=resolver,
                        item_registry=item_registry,
                        context=context,
                        chunk_index=chunk_index,
                        total_chunks=total_chunks,
                        max_tokens=max_tokens,
                        max_retries=max_retries,
                    )
                    internal_state = merge_results(internal_state, group_results)

                    # After demographics: resolve schema for NAACCR
                    if oid == "naaccr" and group.group_id == "demographics":
                        self._resolve_naaccr_schema(internal_state, context, groups)

                except Exception:
                    logger.exception(
                        "Error in domain group %s/%s, chunk %d/%d",
                        oid, group.group_id, chunk_index + 1, total_chunks,
                    )

        # Convert back to list[dict] format
        return self._internal_to_list(internal_state)

    def extract_iterative(
        self,
        texts: list[str],
        cancer_type: Optional[str] = None,
        max_tokens: Optional[int] = 8000,
        max_retries: int = 3,
    ) -> list[dict]:
        """Extract from multiple text chunks iteratively."""
        running: list[dict] = []
        for i, chunk_text in enumerate(texts):
            running = self.extract_single_chunk(
                chunk_text, running, i, len(texts),
                cancer_type, max_tokens, max_retries,
            )
        return running

    # ------------------------------------------------------------------
    # Domain group extraction
    # ------------------------------------------------------------------

    def _extract_domain_group(
        self,
        group: DomainGroup,
        chunk_text: str,
        internal_state: dict[str, ExtractionResult],
        ont: Any,
        oid: str,
        resolver: Any,
        item_registry: dict[str, Any],
        context: dict[str, str],
        chunk_index: int,
        total_chunks: int,
        max_tokens: Optional[int],
        max_retries: int,
    ) -> list[ExtractionResult]:
        """Extract one domain group's items from the chunk."""

        # Resolve items for this group
        if oid == "naaccr":
            items = self._resolve_naaccr_items(group, item_registry)
        else:
            items = self._resolve_generic_items(group, item_registry)

        if not items:
            return []

        # Filter out already-high-confidence items
        items = [
            item for item in items
            if internal_state.get(self._get_field_id(item)) is None
            or internal_state[self._get_field_id(item)].confidence < HIGH_CONFIDENCE_THRESHOLD
        ]

        if not items:
            return []

        # Batch by items_per_call
        batches = split_items_into_batches(items, self.items_per_call)

        all_results: list[ExtractionResult] = []
        for batch in batches:
            try:
                results = self._extract_batch(
                    batch=batch,
                    group=group,
                    chunk_text=chunk_text,
                    internal_state=internal_state,
                    oid=oid,
                    resolver=resolver,
                    context=context,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                )
                all_results.extend(results)
            except Exception:
                logger.exception(
                    "Error extracting batch in %s/%s", oid, group.group_id,
                )

        return all_results

    def _extract_batch(
        self,
        batch: list,
        group: DomainGroup,
        chunk_text: str,
        internal_state: dict[str, ExtractionResult],
        oid: str,
        resolver: Any,
        context: dict[str, str],
        chunk_index: int,
        total_chunks: int,
        max_tokens: Optional[int],
        max_retries: int,
    ) -> list[ExtractionResult]:
        """Extract a batch of items via a single LLM call."""
        # Build JSON format instructions
        json_instructions = self._schema_builder.build_json_format_instructions(
            batch, resolver,
        )

        # Build system prompt
        system_prompt = self._build_system_prompt(
            group, json_instructions, context,
        )

        # Build prior state block
        field_ids = [self._get_field_id(item) for item in batch]
        prior_block = build_prior_state_block(internal_state, field_ids)

        # Build user prompt
        if group.is_narrative:
            prior_narratives = build_prior_narratives_block(
                internal_state, field_ids,
            )
            user_prompt = NARRATIVE_USER_TEMPLATE.format(
                first_date="",
                last_date="",
                chunk_text=chunk_text,
                prior_narratives_block=prior_narratives,
                json_field_descriptions=json_instructions,
            )
        else:
            user_prompt = CHUNK_USER_TEMPLATE.format(
                first_date="",
                last_date="",
                chunk_text=chunk_text,
                tumor_context="",
                prior_state_block=prior_block,
                json_field_descriptions=json_instructions,
            )

        # Call LLM with retry
        parsed = None
        full_prompt = system_prompt + "\n\n" + user_prompt
        for attempt in range(max_retries):
            try:
                response = self.llm_client.generate(full_prompt, max_tokens=max_tokens or 8000)
                parsed = parse_json_object(response.text)
                if parsed is not None:
                    break
                logger.warning(
                    "Group %s batch JSON parse failed (attempt %d/%d)",
                    group.group_id, attempt + 1, max_retries,
                )
                # Include the failed output in the next attempt so the model
                # can see what went wrong and correct it.
                failed_text = response.text[:2000] if len(response.text) > 2000 else response.text
                full_prompt = (
                    system_prompt + "\n\n" + user_prompt
                    + "\n\n--- PREVIOUS ATTEMPT FAILED ---\n"
                    "Your previous response could not be parsed as valid JSON. "
                    "Here is what you returned:\n\n"
                    + failed_text
                    + "\n\nPlease try again and return ONLY a valid JSON object."
                )
            except Exception:
                logger.exception(
                    "Group %s batch LLM call failed (attempt %d/%d)",
                    group.group_id, attempt + 1, max_retries,
                )

        if parsed is None:
            return []

        # Parse response into ExtractionResult objects
        return self._parse_response(parsed, batch, oid, resolver, chunk_index, group.is_narrative)

    def _build_system_prompt(
        self,
        group: DomainGroup,
        json_instructions: str,
        context: dict[str, str],
    ) -> str:
        """Build the system prompt for a domain group."""
        template = group.system_prompt_template

        # Substitute known context variables
        format_kwargs = {"json_format_instructions": json_instructions}
        for key in ["primary_site", "histology", "primary_site_desc", "site_context",
                     "domain_name", "domain_context"]:
            if f"{{{key}}}" in template:
                format_kwargs[key] = context.get(key, "unknown")

        try:
            return template.format(**format_kwargs)
        except KeyError:
            # Fallback: just append JSON instructions
            return template + "\n\n" + json_instructions

    def _parse_response(
        self,
        response: dict,
        items: list,
        oid: str,
        resolver: Any,
        chunk_index: int,
        is_narrative: bool,
    ) -> list[ExtractionResult]:
        """Parse LLM JSON response into ExtractionResult objects."""
        results: list[ExtractionResult] = []

        # Build lookup: prompt_field_name -> item
        field_map: dict[str, Any] = {}
        for item in items:
            pfn = self._schema_builder._field_name(item)
            field_map[pfn] = item

        for field_name, payload in response.items():
            if field_name.startswith("_"):
                continue

            item = field_map.get(field_name)
            if item is None:
                logger.debug("LLM returned unknown field '%s'; skipping.", field_name)
                continue

            if not isinstance(payload, dict):
                # Handle flat value (no {value, confidence, evidence} wrapper)
                payload = {"value": str(payload), "confidence": 0.5, "evidence": ""}

            raw_value = str(payload.get("value", "")).strip()
            llm_confidence = float(payload.get("confidence", 0.5))
            evidence = str(payload.get("evidence", "")).strip()

            if not raw_value:
                continue

            field_id = self._get_field_id(item)

            if is_narrative:
                # No code resolution for narrative text
                length = getattr(item, "length", 0) or 0
                if length > 0:
                    raw_value = raw_value[:length]
                results.append(ExtractionResult(
                    field_id=field_id,
                    field_name=field_name,
                    extracted_value=raw_value,
                    resolved_code=raw_value,
                    confidence=round(llm_confidence, 4),
                    evidence_text=evidence[:300],
                    source_chunk_id="aggregated",
                    source_chunk_type="aggregated",
                    pass_number=chunk_index,
                    ontology_id=oid,
                ))
            else:
                # Resolve code
                resolved_code, resolution_confidence = resolver.resolve(field_id, raw_value)

                if resolution_confidence > 0.0:
                    final_confidence = min(llm_confidence, resolution_confidence)
                else:
                    final_confidence = llm_confidence * 0.5

                results.append(ExtractionResult(
                    field_id=field_id,
                    field_name=field_name,
                    extracted_value=raw_value,
                    resolved_code=resolved_code,
                    confidence=round(final_confidence, 4),
                    evidence_text=evidence[:300],
                    source_chunk_id="sequential",
                    source_chunk_type="sequential",
                    pass_number=chunk_index,
                    ontology_id=oid,
                ))

        return results

    # ------------------------------------------------------------------
    # NAACCR-specific helpers
    # ------------------------------------------------------------------

    def _resolve_naaccr_items(self, group: DomainGroup, item_registry: dict[str, Any]) -> list:
        """Resolve NAACCR item numbers to dictionary items."""
        items = []
        for fid in group.field_ids:
            item = item_registry.get(fid)
            if item is None:
                # Try loading from dictionary
                try:
                    ont = self._ontologies["naaccr"]
                    dict_obj = getattr(ont, "dictionary", None)
                    if dict_obj:
                        item = dict_obj.get_item(int(fid))
                        if item:
                            item_registry[fid] = item
                except (ValueError, TypeError):
                    pass
            if item is not None:
                # Skip retired items
                if getattr(item, "year_retired", ""):
                    continue
                items.append(item)
        return items

    def _resolve_generic_items(self, group: DomainGroup, item_registry: dict[str, Any]) -> list:
        """Resolve generic field_ids to item objects."""
        items = []
        for fid in group.field_ids:
            item = item_registry.get(fid)
            if item is not None:
                items.append(item)
        return items

    def _resolve_naaccr_schema(
        self,
        internal_state: dict[str, ExtractionResult],
        context: dict[str, str],
        groups: list[DomainGroup],
    ) -> None:
        """After demographics extraction, resolve schema and populate staging group."""
        primary_site_result = internal_state.get("400")
        histology_result = internal_state.get("522")

        primary_site = ""
        histology = ""
        if primary_site_result:
            primary_site = primary_site_result.resolved_code or primary_site_result.extracted_value
        if histology_result:
            histology = histology_result.resolved_code or histology_result.extracted_value

        schema = self._naaccr_schema_registry.get_schema_for_site_histology(
            primary_site, histology, None,
        )
        staging_items = self._naaccr_schema_registry.get_all_staging_items(schema)
        site_desc = self._naaccr_schema_registry.get_primary_site_description(schema)
        site_context = self._naaccr_schema_registry.get_site_context(schema)

        context["primary_site"] = primary_site or "unknown"
        context["histology"] = histology or "unknown"
        context["schema"] = schema
        context["primary_site_desc"] = site_desc
        context["site_context"] = site_context

        # Populate the dynamic staging group
        for group in groups:
            if group.group_id == "staging" and group.dynamic:
                group.field_ids = [str(n) for n in staging_items]
                break

    # ------------------------------------------------------------------
    # Format conversion (internal <-> list[dict])
    # ------------------------------------------------------------------

    def _list_to_internal(self, running: list[dict]) -> dict[str, ExtractionResult]:
        """Convert list[dict] format to internal ExtractionResult state.

        Handles the ``[{category: {field: value}}]`` format from old
        Extractor output.
        """
        state: dict[str, ExtractionResult] = {}
        if not running:
            return state

        # Check if we already have metadata from a previous round
        for entry in running:
            if not isinstance(entry, dict):
                continue

            # Check for embedded ExtractionResult metadata
            if "_extraction_results" in entry:
                for fid, result_dict in entry["_extraction_results"].items():
                    state[fid] = ExtractionResult.from_dict(result_dict)
                continue

            # Standard format: {category: {field: value}}
            for category, fields in entry.items():
                if category.startswith("_"):
                    continue
                if not isinstance(fields, dict):
                    continue
                for field_name, value in fields.items():
                    if field_name.startswith("_"):
                        continue
                    fid = field_name  # Use field name as ID for generic
                    state[fid] = ExtractionResult(
                        field_id=fid,
                        field_name=field_name,
                        extracted_value=str(value),
                        resolved_code=str(value),
                        confidence=0.5,  # Unknown confidence from old format
                        evidence_text="",
                        source_chunk_id="prior",
                        source_chunk_type="prior",
                        pass_number=0,
                    )

        return state

    def _internal_to_list(self, state: dict[str, ExtractionResult]) -> list[dict]:
        """Convert internal ExtractionResult state to list[dict] format.

        Groups results by ontology_id and category for backward compatibility.
        Also embeds the full ExtractionResult metadata for round-trip fidelity.
        """
        if not state:
            return []

        # Group by ontology
        by_ontology: dict[str, dict[str, str]] = {}
        for fid, result in state.items():
            oid = result.ontology_id or "extraction"
            if oid not in by_ontology:
                by_ontology[oid] = {}
            # Use field_name as key for the output dict
            by_ontology[oid][result.field_name] = result.resolved_code or result.extracted_value

        # Build list[dict] format
        output: list[dict] = []
        for oid, fields in by_ontology.items():
            output.append({oid: fields})

        # Embed metadata for round-trip
        metadata = {fid: result.to_dict() for fid, result in state.items()}
        output.append({"_extraction_results": metadata})

        return output


# ---------------------------------------------------------------------------
# SummaryExtractor (preserved unchanged)
# ---------------------------------------------------------------------------

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
        from ..ontologies.builtins.clinical_summary.ontology import (
            SUMMARY_FIRST_CHUNK,
            SUMMARY_UPDATE_CHUNK,
        )
        self._first_chunk_template = SUMMARY_FIRST_CHUNK
        self._update_chunk_template = SUMMARY_UPDATE_CHUNK

    def _system_prompt(self) -> str:
        return self._ontology.format_for_prompt(self.cancer_type)

    def extract_from_text(self, text: str, cancer_type: Optional[str] = None, max_tokens: Optional[int] = 8000) -> list[dict]:
        prompt = self._first_chunk_template.format(
            system_prompt=self._system_prompt(),
            chunk_text=text,
        )
        response = self.llm_client.generate(prompt, max_tokens=max_tokens)
        return _wrap_summary(response.text)

    def extract_single_chunk(self, chunk_text: str, running: Optional[list[dict]] = None, chunk_index: int = 0, total_chunks: int = 1, cancer_type: Optional[str] = None, max_tokens: Optional[int] = 8000, max_retries: int = 3) -> list[dict]:
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
        running: list[dict] = []
        for i, chunk_text in enumerate(texts):
            running = self.extract_single_chunk(
                chunk_text, running, i, len(texts),
                cancer_type, max_tokens, max_retries,
            )
        return running


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap_summary(text: str) -> list[dict]:
    return [{"clinical_summary": {"summary": text.strip()}}]


def _unwrap_summary(running: Optional[list[dict]]) -> str:
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


def create_extractor(
    llm_client: LLMClient,
    ontology_ids: list[str],
    cancer_type: Optional[str] = "generic",
    items_per_call: int = DEFAULT_ITEMS_PER_CALL,
    **kwargs,
):
    """Factory that returns the appropriate extractor based on ontology types."""
    if is_summary_only(ontology_ids):
        return SummaryExtractor(llm_client, cancer_type)
    return Extractor(llm_client, ontology_ids, cancer_type, items_per_call)
