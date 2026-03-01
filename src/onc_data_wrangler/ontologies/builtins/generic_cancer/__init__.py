"""
Generic Cancer Ontology Module

Provides cancer-type-agnostic data items for structured extraction from
clinical notes across all cancer types.
"""

from .ontology import GenericCancerOntology
from ...registry import register_ontology

register_ontology(GenericCancerOntology)

__all__ = ["GenericCancerOntology"]
