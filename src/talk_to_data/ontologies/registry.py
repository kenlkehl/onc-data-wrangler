"""
Ontology Registry

Central registry for discovering, registering, and instantiating ontologies.
Provides the MultiOntologyExtractor for combining multiple ontologies.
"""
import importlib
from pathlib import Path
from typing import Dict, List, Type, Optional, Any

from .base import OntologyBase


class OntologyRegistry:
    """
    Central registry for available ontologies.

    Ontologies self-register using the @register_ontology decorator
    or by calling OntologyRegistry.register() directly.
    """
    _ontologies: Dict[str, Type[OntologyBase]] = {}
    _instances: Dict[str, OntologyBase] = {}

    @classmethod
    def register(cls, ontology_class):
        """Register an ontology class. Can be used as a decorator or called directly."""
        temp_instance = ontology_class()
        ontology_id = temp_instance.ontology_id
        cls._ontologies[ontology_id] = ontology_class

    @classmethod
    def get(cls, ontology_id: str) -> OntologyBase:
        """Get an ontology instance by ID (cached)."""
        if ontology_id not in cls._ontologies:
            cls.auto_discover()
        if ontology_id not in cls._ontologies:
            available = cls.list_available()
            raise ValueError(f"Unknown ontology: '{ontology_id}'. Available ontologies: {available}")
        if ontology_id not in cls._instances:
            cls._instances[ontology_id] = cls._ontologies[ontology_id]()
        return cls._instances[ontology_id]

    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered ontology IDs."""
        if not cls._ontologies:
            cls.auto_discover()
        return sorted(cls._ontologies.keys())

    @classmethod
    def get_all(cls) -> List[OntologyBase]:
        """Get all registered ontology instances."""
        if not cls._ontologies:
            cls.auto_discover()
        return [cls.get(oid) for oid in cls._ontologies]

    @classmethod
    def auto_discover(cls):
        """Auto-discover ontologies in the builtins directory."""
        ontology_packages = ('naaccr', 'matchminer_ai', 'prissmm', 'omop', 'msk_chord', 'pan_top', 'generic_cancer', 'clinical_summary')
        for package_name in ontology_packages:
            try:
                importlib.import_module(f".builtins.{package_name}", package="talk_to_data.ontologies")
            except ImportError:
                pass

    @classmethod
    def clear(cls):
        """Clear all registered ontologies (mainly for testing)."""
        cls._ontologies.clear()
        cls._instances.clear()


# Decorator alias
register_ontology = OntologyRegistry.register


class MultiOntologyExtractor:
    """Combines multiple ontologies for extraction."""

    def __init__(self, ontology_ids: List[str]):
        if ontology_ids == ["all"]:
            ontology_ids = OntologyRegistry.list_available()
        self.ontology_ids = ontology_ids
        self.ontologies = [OntologyRegistry.get(oid) for oid in ontology_ids]

    def get_combined_template(self) -> Dict[str, Any]:
        """Generate combined template with separate sections per ontology."""
        combined = {}
        for ont in self.ontologies:
            combined[ont.ontology_id] = ont.get_empty_summary_template()
        combined["extraction_metadata"] = {
            "ontologies_used": self.ontology_ids,
            "extraction_date": None,
            "last_note_date": None,
            "notes_incorporated": 0,
        }
        return combined

    def get_combined_diagnosis_template(self, cancer_types: Dict[str, str] = None) -> Dict[str, Any]:
        """Generate combined diagnosis template for all ontologies."""
        combined = {}
        for ont in self.ontologies:
            cancer_type = (cancer_types or {}).get(ont.ontology_id, "generic")
            combined[ont.ontology_id] = ont.get_empty_diagnosis_template(cancer_type)
        return combined

    def format_combined_prompt(self, cancer_type: str = None, include_context: bool = True) -> str:
        """Build prompt incorporating all ontologies."""
        sections = []
        for ont in self.ontologies:
            sections.append("=" * 60)
            sections.append(f"=== {ont.display_name} (output to '{ont.ontology_id}' section) ===")
            if include_context:
                context = ont.get_extraction_context()
                if context:
                    sections.append(context)
            sections.append(ont.format_for_prompt(cancer_type))
        return "\n".join(sections)

    def detect_cancer_type(self, primary_site: str = None, histology: str = None, diagnosis_year: int = None) -> Dict[str, str]:
        """Detect cancer type for each ontology."""
        return {ont.ontology_id: ont.detect_cancer_type(primary_site, histology, diagnosis_year) for ont in self.ontologies}

    def validate_combined_output(self, output: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate output for all ontologies."""
        errors = {}
        for ont in self.ontologies:
            ont_errors = ont.validate_output(output.get(ont.ontology_id, {}))
            if ont_errors:
                errors[ont.ontology_id] = ont_errors
        return errors

    def __repr__(self) -> str:
        return f"MultiOntologyExtractor(ontologies={self.ontology_ids})"
