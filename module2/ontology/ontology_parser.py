"""
Ontology Parser & Schema Extractor.

Parses diabetes_insulin_ontology.owl using rdflib, extracts classes,
object properties, and datatype properties, and validates schema consistency.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any
import rdflib
from rdflib.namespace import RDF, RDFS, OWL, XSD

from module2.ontology.schema_types import (
    DEFAULT_ONTOLOGY_URI,
    ONTOLOGY_CLASSES,
    ONTOLOGY_PROPERTIES,
    NodeType,
    NODE_TYPE_TO_OWL_CLASS,
)

logger = logging.getLogger(__name__)


@dataclass
class PropertyMetadata:
    """Stores domain, range, and type for an ontology property."""
    name: str
    property_type: str  # "ObjectProperty" or "DatatypeProperty"
    domain: Optional[str] = None
    range_uri: Optional[str] = None
    comment: Optional[str] = None


@dataclass
class OntologySchema:
    """Cached in-memory representation of the parsed ontology."""
    ontology_uri: str
    classes: Set[str] = field(default_factory=set)
    subclass_hierarchy: Dict[str, List[str]] = field(default_factory=dict)
    object_properties: Dict[str, PropertyMetadata] = field(default_factory=dict)
    datatype_properties: Dict[str, PropertyMetadata] = field(default_factory=dict)

    def is_valid_class(self, class_name: str) -> bool:
        return class_name in self.classes

    def get_property(self, prop_name: str) -> Optional[PropertyMetadata]:
        return self.object_properties.get(prop_name) or self.datatype_properties.get(prop_name)


class OntologyParser:
    """
    Parses an OWL/RDF ontology file using rdflib.
    """

    def __init__(self, ontology_path: Optional[str] = None):
        self.ontology_path = ontology_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "diabetes_insulin_ontology.owl",
        )
        self.graph: Optional[rdflib.Graph] = None
        self.schema: Optional[OntologySchema] = None

    def parse(self) -> OntologySchema:
        """
        Load and parse the OWL ontology file.
        """
        if not os.path.exists(self.ontology_path):
            raise FileNotFoundError(f"Ontology file not found at: {self.ontology_path}")

        logger.info("Parsing ontology file: %s", self.ontology_path)
        g = rdflib.Graph()
        g.parse(self.ontology_path, format="xml")
        self.graph = g

        # 1. Determine ontology URI
        ontology_uri = DEFAULT_ONTOLOGY_URI
        for s in g.subjects(RDF.type, OWL.Ontology):
            ontology_uri = str(s)
            break

        # 2. Extract all OWL Classes
        classes: Set[str] = set()
        subclass_hierarchy: Dict[str, List[str]] = {}

        for s in g.subjects(RDF.type, OWL.Class):
            class_name = self._get_local_name(s)
            if class_name:
                classes.add(class_name)

        # Build subclass relationships
        for s, _, o in g.triples((None, RDFS.subClassOf, None)):
            sub = self._get_local_name(s)
            super_cls = self._get_local_name(o)
            if sub and super_cls:
                subclass_hierarchy.setdefault(super_cls, []).append(sub)

        # 3. Extract Object Properties
        object_properties: Dict[str, PropertyMetadata] = {}
        for s in g.subjects(RDF.type, OWL.ObjectProperty):
            prop_name = self._get_local_name(s)
            if prop_name:
                domain = self._get_property_attribute(g, s, RDFS.domain)
                range_uri = self._get_property_attribute(g, s, RDFS.range)
                comment = self._get_property_attribute(g, s, RDFS.comment)
                object_properties[prop_name] = PropertyMetadata(
                    name=prop_name,
                    property_type="ObjectProperty",
                    domain=domain,
                    range_uri=range_uri,
                    comment=comment,
                )

        # 4. Extract Datatype Properties
        datatype_properties: Dict[str, PropertyMetadata] = {}
        for s in g.subjects(RDF.type, OWL.DatatypeProperty):
            prop_name = self._get_local_name(s)
            if prop_name:
                domain = self._get_property_attribute(g, s, RDFS.domain)
                range_uri = self._get_property_attribute(g, s, RDFS.range)
                comment = self._get_property_attribute(g, s, RDFS.comment)
                datatype_properties[prop_name] = PropertyMetadata(
                    name=prop_name,
                    property_type="DatatypeProperty",
                    domain=domain,
                    range_uri=range_uri,
                    comment=comment,
                )

        self.schema = OntologySchema(
            ontology_uri=ontology_uri,
            classes=classes,
            subclass_hierarchy=subclass_hierarchy,
            object_properties=object_properties,
            datatype_properties=datatype_properties,
        )

        # Validation check against expected schema
        self._validate_schema(self.schema)
        logger.info(
            "Ontology parsed successfully: %d classes, %d object properties, %d datatype properties",
            len(classes),
            len(object_properties),
            len(datatype_properties),
        )
        return self.schema

    def _validate_schema(self, schema: OntologySchema) -> None:
        """Ensure all required classes and properties from schema_types exist."""
        missing_classes = ONTOLOGY_CLASSES - schema.classes
        if missing_classes:
            logger.warning("Ontology is missing expected classes: %s", missing_classes)

        all_props = set(schema.object_properties.keys()) | set(schema.datatype_properties.keys())
        missing_props = ONTOLOGY_PROPERTIES - all_props
        if missing_props:
            logger.warning("Ontology is missing expected properties: %s", missing_props)

    @staticmethod
    def _get_local_name(uri: rdflib.term.Identifier) -> str:
        """Extract local name after '#' or '/'."""
        s = str(uri)
        if "#" in s:
            return s.split("#")[-1]
        elif "/" in s:
            return s.split("/")[-1]
        return s

    @staticmethod
    def _get_property_attribute(g: rdflib.Graph, prop: rdflib.term.Identifier, attr: rdflib.term.Identifier) -> Optional[str]:
        """Fetch local name of domain/range/comment attribute."""
        for val in g.objects(prop, attr):
            return str(val)
        return None
