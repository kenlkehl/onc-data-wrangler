"""Clinical Summary Ontology - free-text clinical summarization."""

from .ontology import ClinicalSummaryOntology
from ...registry import register_ontology

register_ontology(ClinicalSummaryOntology)

__all__ = ["ClinicalSummaryOntology"]
