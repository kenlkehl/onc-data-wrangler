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


@dataclass
class DataCategory:
    """A group of related data items."""
    id: str
    name: str
    description: str
    items: List[DataItem]
    context: str = ""
    per_diagnosis: bool = False


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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id='{self.ontology_id}', version='{self.version}')"
