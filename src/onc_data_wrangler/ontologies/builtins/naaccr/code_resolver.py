"""Resolve LLM outputs to valid NAACCR codes.

Ported from onc-registry-extraction/naaccr_pipeline/dictionary/code_resolver.py.
Implements the ``CodeResolverLike`` protocol from ``ontologies.protocols``.
"""

from __future__ import annotations

import logging
import re

from .dictionary import NAACCRDictionary, CodeEntry

logger = logging.getLogger(__name__)

# Optional dependency -- degrade gracefully when rapidfuzz is absent.
try:
    from rapidfuzz import fuzz as _fuzz, process as _process

    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

# Pre-compiled pattern for numeric ranges like "001-999", "00-88", etc.
_RANGE_RE = re.compile(r"^(\d+)\s*[-\u2013]\s*(\d+)$")


class NAACCRCodeResolver:
    """Map free-text / LLM output to a valid NAACCR code value.

    Implements the ``CodeResolverLike`` protocol, accepting ``field_id``
    (string representation of item_number) for the generic interface,
    while also supporting direct integer item_number calls.

    Resolution strategy (highest priority first):

    1. Exact match against code values -> 1.0
    2. Case-insensitive match against code values -> 0.95
    3. Exact match against code descriptions -> 0.9
    4. Fuzzy match against descriptions (rapidfuzz, score > 85) ->
       0.9 * (score / 100)
    5. Numeric-range check from allowable values -> 0.85
    6. No match -> ``(llm_output, 0.0)``
    """

    def __init__(self, dictionary: NAACCRDictionary) -> None:
        self._dict = dictionary

        # Per-item indexes built once at init time.
        self._code_index: dict[int, dict[str, CodeEntry]] = {}
        self._code_index_lower: dict[int, dict[str, CodeEntry]] = {}
        self._desc_index: dict[int, dict[str, CodeEntry]] = {}
        self._desc_list: dict[int, list[tuple[str, CodeEntry]]] = {}
        self._numeric_ranges: dict[int, list[tuple[int, int, int]]] = {}

        self._build_indexes()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_indexes(self) -> None:
        for item in self._dict.get_active_items():
            codes = self._dict.get_codes(item.item_number)
            if not codes:
                self._parse_numeric_ranges(item.item_number, item.allowable_values)
                continue

            exact: dict[str, CodeEntry] = {}
            lower: dict[str, CodeEntry] = {}
            desc: dict[str, CodeEntry] = {}
            desc_pairs: list[tuple[str, CodeEntry]] = []

            for ce in codes:
                exact[ce.code] = ce
                lower[ce.code.lower()] = ce
                d = ce.description.lower().strip()
                if d:
                    desc[d] = ce
                    desc_pairs.append((d, ce))

            self._code_index[item.item_number] = exact
            self._code_index_lower[item.item_number] = lower
            self._desc_index[item.item_number] = desc
            self._desc_list[item.item_number] = desc_pairs

            self._parse_numeric_ranges(item.item_number, item.allowable_values)

    def _parse_numeric_ranges(self, item_number: int, allowable: str) -> None:
        if not allowable:
            return
        ranges: list[tuple[int, int, int]] = []
        for token in re.split(r"[,;]\s*|\s+", allowable):
            token = token.strip()
            m = _RANGE_RE.match(token)
            if m:
                lo_str, hi_str = m.group(1), m.group(2)
                width = max(len(lo_str), len(hi_str))
                try:
                    ranges.append((int(lo_str), int(hi_str), width))
                except ValueError:
                    continue
        if ranges:
            self._numeric_ranges[item_number] = ranges

    # ------------------------------------------------------------------
    # Public API -- integer item_number interface
    # ------------------------------------------------------------------

    def resolve_by_item(self, item_number: int, llm_output: str) -> tuple[str, float]:
        """Resolve *llm_output* to a valid NAACCR code for *item_number*."""
        text = llm_output.strip()

        # 1. Exact code match
        exact = self._code_index.get(item_number, {})
        if text in exact:
            return (text, 1.0)

        # 2. Case-insensitive code match
        lower_idx = self._code_index_lower.get(item_number, {})
        hit = lower_idx.get(text.lower())
        if hit is not None:
            return (hit.code, 0.95)

        # 3. Exact description match
        desc_idx = self._desc_index.get(item_number, {})
        hit = desc_idx.get(text.lower())
        if hit is not None:
            return (hit.code, 0.9)

        # 4. Fuzzy description match
        desc_pairs = self._desc_list.get(item_number, [])
        if _HAS_RAPIDFUZZ and desc_pairs:
            descriptions = [d for d, _ in desc_pairs]
            result = _process.extractOne(
                text.lower(),
                descriptions,
                scorer=_fuzz.WRatio,
                score_cutoff=85,
            )
            if result is not None:
                matched_desc, score, idx = result
                entry = desc_pairs[idx][1]
                confidence = 0.9 * (score / 100.0)
                return (entry.code, round(confidence, 4))

        # 5. Numeric-range check
        ranges = self._numeric_ranges.get(item_number, [])
        if ranges and text.isdigit():
            val = int(text)
            for lo, hi, width in ranges:
                if lo <= val <= hi:
                    padded = text.zfill(width)
                    return (padded, 0.85)

        # 6. No match
        return (text, 0.0)

    def get_valid_codes_prompt_by_item(self, item_number: int) -> str:
        """Return a compact code-reference string for prompt injection."""
        codes = self._dict.get_codes(item_number)
        if not codes:
            return ""
        parts = [f"{c.code}={c.description}" for c in codes]
        return ", ".join(parts)

    def has_codes_by_item(self, item_number: int) -> bool:
        return bool(self._dict.get_codes(item_number))

    # ------------------------------------------------------------------
    # CodeResolverLike protocol -- string field_id interface
    # ------------------------------------------------------------------

    def resolve(self, field_id: str, llm_output: str) -> tuple[str, float]:
        """Resolve using string field_id (CodeResolverLike protocol)."""
        try:
            item_number = int(field_id)
        except (ValueError, TypeError):
            return (llm_output.strip(), 0.0)
        return self.resolve_by_item(item_number, llm_output)

    def get_valid_codes_prompt(self, field_id: str) -> str:
        """Get valid codes prompt using string field_id."""
        try:
            item_number = int(field_id)
        except (ValueError, TypeError):
            return ""
        return self.get_valid_codes_prompt_by_item(item_number)

    def has_codes(self, field_id: str) -> bool:
        """Check if codes exist using string field_id."""
        try:
            item_number = int(field_id)
        except (ValueError, TypeError):
            return False
        return self.has_codes_by_item(item_number)
