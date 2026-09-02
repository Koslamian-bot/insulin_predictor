"""
Ontology parsing and schema extraction subpackage.
"""

from module2.ontology.schema_types import NodeType, EdgeType, ONTOLOGY_CLASSES, ONTOLOGY_PROPERTIES
from module2.ontology.ontology_parser import OntologyParser, OntologySchema

__all__ = [
    "NodeType",
    "EdgeType",
    "ONTOLOGY_CLASSES",
    "ONTOLOGY_PROPERTIES",
    "OntologyParser",
    "OntologySchema",
]
