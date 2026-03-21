"""
Ontologies Module

Provides a modular ontology system for structured data extraction
from unstructured clinical documents.

Supported Ontologies:
- naaccr: NAACCR v25 cancer registry standards
- matchminer_ai: Custom concepts for clinical trial matching
- prissmm: PRISSMM/GENIE BPC data model
- omop: OMOP CDM with oncology extension
- msk_chord: MSK-CHORD/cBioPortal clinical data model
- pan_top: Pan-TOP thoracic oncology data extraction
- generic_cancer: Cancer-type-agnostic data extraction

Usage:
    from onc_data_wrangler.ontologies import OntologyRegistry, MultiOntologyExtractor

    # Get a single ontology
    naaccr = OntologyRegistry.get('naaccr')
    template = naaccr.get_empty_summary_template()
    prompt = naaccr.format_for_prompt('lung')

    # Use multiple ontologies
    extractor = MultiOntologyExtractor(['naaccr', 'matchminer_ai', 'prissmm'])
    combined_template = extractor.get_combined_template()
    combined_prompt = extractor.format_combined_prompt('lung')

    # List available ontologies
    available = OntologyRegistry.list_available()
"""
from .base import DataItem, DataCategory, OntologyBase
from .registry import OntologyRegistry, MultiOntologyExtractor, register_ontology

__version__ = "1.0.0"
__all__ = ("DataItem", "DataCategory", "OntologyBase", "OntologyRegistry", "MultiOntologyExtractor", "register_ontology")


def get_ontology(ontology_id: str) -> OntologyBase:
    """Convenience function to get an ontology by ID."""
    return OntologyRegistry.get(ontology_id)


def list_ontologies() -> list[str]:
    """Convenience function to list available ontologies."""
    return OntologyRegistry.list_available()


def create_extractor(ontology_ids: list[str]) -> MultiOntologyExtractor:
    """Convenience function to create a multi-ontology extractor."""
    return MultiOntologyExtractor(ontology_ids)
