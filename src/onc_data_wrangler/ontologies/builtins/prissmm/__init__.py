"""PRISSMM/GENIE BPC Ontology - biopharma collaborative data model."""

from .ontology import PRISSMMOntology
from ...registry import register_ontology

register_ontology(PRISSMMOntology)

__all__ = ["PRISSMMOntology"]
