"""OMOP vocabulary dictionary loader for oncology concepts.

Loads a pre-filtered subset of the OMOP CONCEPT table (oncology-relevant
standard concepts) and provides lookup by concept_id, domain, vocabulary,
and free-text name search.

The ``OMOPConcept`` dataclass implements the ``DictionaryItemLike``
protocol so it integrates with the domain-group extraction pipeline.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODULE_DIR = Path(__file__).parent
DEFAULT_DATA_DIR = MODULE_DIR / "data"


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class OMOPConcept:
    """A single OMOP standard concept from the filtered vocabulary."""

    concept_id: int
    concept_name: str
    domain_id: str
    vocabulary_id: str
    concept_class_id: str
    concept_code: str

    # -- DictionaryItemLike protocol properties ----------------------------

    @property
    def field_id(self) -> str:
        return str(self.concept_id)

    @property
    def name(self) -> str:
        return self.concept_name

    @property
    def prompt_field_name(self) -> str:
        return self.concept_name

    @property
    def length(self) -> int:
        return 0  # unlimited

    @property
    def data_type(self) -> str:
        return "string"

    @property
    def description(self) -> str:
        return (
            f"{self.concept_name} [{self.vocabulary_id} {self.concept_code}] "
            f"({self.domain_id}/{self.concept_class_id})"
        )

    @property
    def allowable_values(self) -> str:
        return ""


# ---------------------------------------------------------------------------
# Dictionary loader
# ---------------------------------------------------------------------------

class OMOPDictionary:
    """In-memory index of oncology-relevant OMOP standard concepts.

    Loads from ``data/oncology_concepts.csv`` -- a pre-filtered subset of
    the OMOP CONCEPT table containing only standard, valid concepts in
    oncology-relevant domains and vocabularies.

    Usage::

        d = OMOPDictionary()
        d.load()
        concept = d.lookup_concept(4112853)
        drugs = d.get_concepts_by_domain('Drug')
    """

    def __init__(
        self,
        data_dir: Optional[str | Path] = None,
    ) -> None:
        self._data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

        # Primary index
        self._by_id: dict[int, OMOPConcept] = {}

        # Secondary indexes
        self._by_domain: dict[str, list[OMOPConcept]] = {}
        self._by_vocabulary: dict[str, list[OMOPConcept]] = {}
        self._by_code: dict[str, list[OMOPConcept]] = {}  # concept_code -> concepts
        self._name_index: list[tuple[str, OMOPConcept]] = []  # (lower_name, concept)

        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Parse oncology_concepts.csv and build look-up indexes."""
        path = self._data_dir / "oncology_concepts.csv"
        if not path.exists():
            logger.warning("OMOP oncology concepts file not found: %s", path)
            self._loaded = True
            return

        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    concept_id = int(row["concept_id"])
                except (ValueError, TypeError):
                    continue

                concept = OMOPConcept(
                    concept_id=concept_id,
                    concept_name=row.get("concept_name", "").strip(),
                    domain_id=row.get("domain_id", "").strip(),
                    vocabulary_id=row.get("vocabulary_id", "").strip(),
                    concept_class_id=row.get("concept_class_id", "").strip(),
                    concept_code=row.get("concept_code", "").strip(),
                )

                self._by_id[concept_id] = concept
                self._by_domain.setdefault(concept.domain_id, []).append(concept)
                self._by_vocabulary.setdefault(concept.vocabulary_id, []).append(concept)
                self._by_code.setdefault(concept.concept_code, []).append(concept)
                self._name_index.append((concept.concept_name.lower(), concept))

        self._loaded = True
        logger.info(
            "OMOP dictionary loaded: %d concepts across %d domains, %d vocabularies",
            len(self._by_id),
            len(self._by_domain),
            len(self._by_vocabulary),
        )

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def lookup_concept(self, concept_id: int) -> OMOPConcept | None:
        """Return the concept for *concept_id*, or ``None``."""
        return self._by_id.get(concept_id)

    def get_concepts_by_domain(self, domain: str) -> list[OMOPConcept]:
        """Return all concepts belonging to *domain* (e.g. 'Drug', 'Condition')."""
        return list(self._by_domain.get(domain, []))

    def get_concepts_by_vocabulary(self, vocab: str) -> list[OMOPConcept]:
        """Return all concepts from *vocab* (e.g. 'SNOMED', 'RxNorm')."""
        return list(self._by_vocabulary.get(vocab, []))

    def lookup_by_code(self, concept_code: str) -> list[OMOPConcept]:
        """Return concepts matching a source *concept_code*."""
        return list(self._by_code.get(concept_code, []))

    def search_by_name(self, query: str) -> list[OMOPConcept]:
        """Search concepts by name substring (case-insensitive).

        Returns up to 100 matching concepts, preferring exact prefix
        matches over substring matches.
        """
        q = query.lower()
        prefix_matches: list[OMOPConcept] = []
        substring_matches: list[OMOPConcept] = []

        for lower_name, concept in self._name_index:
            if lower_name.startswith(q):
                prefix_matches.append(concept)
            elif q in lower_name:
                substring_matches.append(concept)

            # Early exit once we have enough matches
            if len(prefix_matches) + len(substring_matches) >= 200:
                break

        results = prefix_matches + substring_matches
        return results[:100]

    @property
    def concept_count(self) -> int:
        """Total number of loaded concepts."""
        return len(self._by_id)

    @property
    def domains(self) -> list[str]:
        """Return all domain names in the dictionary."""
        return sorted(self._by_domain.keys())

    @property
    def vocabularies(self) -> list[str]:
        """Return all vocabulary names in the dictionary."""
        return sorted(self._by_vocabulary.keys())

    def get_all_concepts(self) -> list[OMOPConcept]:
        """Return all loaded concepts."""
        return list(self._by_id.values())
