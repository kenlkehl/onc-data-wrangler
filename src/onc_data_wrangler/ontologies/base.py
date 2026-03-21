"""
Ontology Base Classes

Provides abstract base classes for oncology data extraction ontologies.
All ontology implementations must inherit from OntologyBase.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class DataItem:
    """Represents a single extractable data element."""
    id: str
    name: str
    description: str
    data_type: str  # string, integer, date, etc.
    valid_values: Optional[Dict] = None
    extraction_hints: List[str] = field(default_factory=list)
    repeatable: bool = False
    required: bool = False
    json_field: str = None
    naaccr_item: Optional[str] = None
    human_readable_field: Optional[str] = None
    clinical_significance: Optional[str] = None
    required_for_staging: bool = False


@dataclass
class DataCategory:
    """A group of related data items."""
    id: str
    name: str
    description: str
    items: List[DataItem]
    context: str = ""
    per_diagnosis: bool = False


@dataclass
class ExtractionPass:
    """Defines a single pass in multi-pass extraction."""
    pass_id: str
    name: str
    categories: List[str]
    depends_on: List[str] = field(default_factory=list)
    context_from: List[str] = field(default_factory=list)


class OntologyBase(ABC):
    """
    Abstract base class for all ontologies.

    Each ontology implementation must:
    1. Define ontology_id, display_name, and version
    2. Implement methods to provide data items and templates
    3. Implement prompt formatting for LLM extraction
    """

    @property
    @abstractmethod
    def ontology_id(self) -> str:
        """Unique identifier for this ontology."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for display in prompts and UI."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Version string for this ontology implementation."""
        ...

    @property
    def is_free_text(self) -> bool:
        """Whether this ontology produces free-text output instead of structured JSON."""
        return False

    @property
    def description(self) -> str:
        """Short description for display in setup/selection UI."""
        return self.display_name

    @abstractmethod
    def get_base_items(self) -> List[DataCategory]:
        """Get base data items that apply to all cancer types."""
        ...

    @abstractmethod
    def get_site_specific_items(self, cancer_type: str) -> List[DataCategory]:
        """Get site-specific data items for a cancer type."""
        ...

    @abstractmethod
    def get_empty_summary_template(self) -> Dict[str, Any]:
        """Return empty JSON structure for this ontology."""
        ...

    @abstractmethod
    def get_empty_diagnosis_template(self, cancer_type: str) -> Dict[str, Any]:
        """Return empty JSON structure for a single diagnosis."""
        ...

    @abstractmethod
    def format_for_prompt(self, cancer_type: str = "generic") -> str:
        """Format ontology items as text for LLM prompts."""
        ...

    def get_supported_cancer_types(self) -> List[str]:
        """List of cancer types this ontology supports."""
        return []

    def detect_cancer_type(self, primary_site: str = None, histology: str = None, diagnosis_year: int = None) -> str:
        """Map primary site and histology to a cancer type identifier."""
        return "generic"

    def get_extraction_context(self) -> str:
        """Additional context to include in prompts about this ontology."""
        return ""

    def validate_output(self, output: Dict[str, Any]) -> List[str]:
        """Validate extracted output against ontology schema."""
        return []

    # ------------------------------------------------------------------
    # Optional hooks for domain-group-based extraction (Phase 9)
    # ------------------------------------------------------------------

    def get_code_resolver(self):
        """Return a CodeResolverLike for this ontology, or None.

        Default returns None; the Extractor will build a GenericCodeResolver
        from the ontology's valid_values.  Override for richer resolution
        (e.g., NAACCR returns its CSV-based NAACCRCodeResolver).
        """
        return None

    def get_schema_resolver(self):
        """Return a SchemaResolverLike for this ontology, or None.

        Default returns None.  Override for ontologies with site-specific
        schemas (e.g., NAACCR returns its SchemaRegistry).
        """
        return None

    def get_domain_groups(self, cancer_type: str = "generic") -> list:
        """Return DomainGroup objects for domain-group-based extraction.

        Default returns None, which tells the Extractor to auto-generate
        groups from DataCategory objects.  Override for hand-curated groups.
        """
        return None

    def get_domain_system_prompt(self, group_id: str) -> Optional[str]:
        """Return a custom system prompt for a specific domain group.

        Default returns None (use the generic or group-defined prompt).
        """
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id='{self.ontology_id}', version='{self.version}')"
