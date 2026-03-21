"""OMOP CDM Ontology - Common Data Model with oncology extension."""

from .ontology import OMOPOntology
from .dictionary import OMOPDictionary, OMOPConcept
from ...registry import register_ontology

register_ontology(OMOPOntology)

__all__ = ["OMOPOntology", "OMOPDictionary", "OMOPConcept"]
