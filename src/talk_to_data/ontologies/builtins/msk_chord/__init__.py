"""MSK-CHORD Ontology - MSK clinical health oncology research data model."""

from .ontology import MSKChordOntology
from ...registry import register_ontology

register_ontology(MSKChordOntology)

__all__ = ["MSKChordOntology"]
