"""
Pan-TOP Ontology Module

Provides Pan-TOP (Pan-Thoracic Oncology Platform) data items for structured
extraction from clinical notes of thoracic malignancy patients.

Supported Cancer Types:
- lung
- mesothelioma
- thymus
- generic (fallback)
"""

from .ontology import PanTOPOntology
from ...registry import register_ontology

register_ontology(PanTOPOntology)

__all__ = ["PanTOPOntology"]
