"""MatchMiner-AI Ontology - clinical trial matching concepts."""

from .ontology import MatchMinerAIOntology
from ...registry import register_ontology

register_ontology(MatchMinerAIOntology)

__all__ = ["MatchMinerAIOntology"]
