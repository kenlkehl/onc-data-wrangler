"""NAACCR v25 Ontology - cancer registry standards with site-specific data items."""

from .ontology import NAACCROntology
from ...registry import register_ontology

register_ontology(NAACCROntology)

__all__ = ["NAACCROntology"]
