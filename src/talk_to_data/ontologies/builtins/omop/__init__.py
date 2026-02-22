"""OMOP CDM Ontology - Common Data Model with oncology extension."""

from .ontology import OMOPOntology
from ...registry import register_ontology

register_ontology(OMOPOntology)

__all__ = ["OMOPOntology"]
