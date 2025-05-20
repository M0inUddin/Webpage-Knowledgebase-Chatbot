"""
Knowledge graph for tracking entities and relationships in the Manzil Chatbot.
"""

import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import networkx as nx

from utils.logging_config import get_logger
from utils.error_handlers import ErrorHandler

logger = get_logger("knowledge_base.knowledge_graph")


class KnowledgeGraph:
    """
    Manages a knowledge graph of entities and relationships extracted from
    user queries and responses.
    """

    def __init__(self):
        """Initialize the knowledge graph."""
        # Create directed graph
        self.graph = nx.DiGraph()

        # Define entity types
        self.entity_types = {
            "person": set(),
            "organization": set(),
            "product": set(),
            "topic": set(),
            "location": set(),
            "islamic_term": set(),
        }

        # Define important Islamic finance terms
        self.islamic_finance_terms = {
            "murabaha",
            "ijara",
            "musharaka",
            "sukuk",
            "takaful",
            "riba",
            "gharar",
            "zakat",
            "halal",
            "haram",
            "shariah",
            "fatwa",
            "wasiyyah",
            "qard",
            "sadaqah",
            "manzil",
        }

        # Entity synonyms for standardization
        self.entity_synonyms = {
            "sharia": "shariah",
            "shari'ah": "shariah",
            "sharia-compliant": "shariah-compliant",
            "mudaraba": "mudarabah",
            "ijara": "ijarah",
            "islamic will": "wasiyyah",
        }

        # Initialize with some basic entities about Manzil
        self._initialize_base_entities()

        logger.info("Knowledge graph initialized")

    def _initialize_base_entities(self):
        """Initialize the knowledge graph with base entities about Manzil."""
        # Add Manzil as organization
        self.add_entity(
            "Manzil",
            "organization",
            {
                "description": "A Canadian financial services company providing Shariah-compliant solutions",
                "website": "https://manzil.ca/",
            },
        )

        # Add main products/services
        products = [
            (
                "Halal Home Financing",
                "Murabaha mortgage financing that is Shariah-compliant",
            ),
            ("Halal Investing", "Shariah-compliant investment services"),
            ("Islamic Wills", "Wasiyyah creation services"),
            ("Halal Prepaid MasterCard", "Shariah-compliant payment card"),
            (
                "Manzil Communities",
                "Residential projects with Islamic financing options",
            ),
        ]

        for name, desc in products:
            self.add_entity(name, "product", {"description": desc})
            self.add_relationship("Manzil", "offers", name)

        # Add key Islamic finance terms
        terms = [
            ("Murabaha", "Cost-plus financing structure"),
            ("Shariah", "Islamic law derived from the Quran and Sunnah"),
            ("Riba", "Interest, which is prohibited in Islamic finance"),
            ("Zakat", "Islamic form of almsgiving or charity"),
            ("Wasiyyah", "Islamic will"),
        ]

        for name, desc in terms:
            self.add_entity(name, "islamic_term", {"description": desc})

    def add_entity(
        self, name: str, entity_type: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add an entity to the knowledge graph.

        Args:
            name (str): The name of the entity
            entity_type (str): The type of entity
            metadata (dict, optional): Additional metadata about the entity

        Returns:
            bool: Whether the entity was added successfully
        """
        with ErrorHandler(
            error_type="knowledge_graph", reraise=False, fallback_value=False
        ):
            # Standardize entity name
            name = name.strip()
            if name.lower() in self.entity_synonyms:
                name = self.entity_synonyms[name.lower()]

            # Check if entity type is valid
            if entity_type not in self.entity_types:
                logger.warning(f"Unknown entity type: {entity_type}")
                return False

            # Add to type set
            self.entity_types[entity_type].add(name)

            # Special handling for Islamic terms
            if (
                entity_type == "islamic_term"
                or name.lower() in self.islamic_finance_terms
            ):
                # Add to islamic_term type regardless of original type
                self.entity_types["islamic_term"].add(name)

            # Create or update node
            if not self.graph.has_node(name):
                # Add new node
                self.graph.add_node(
                    name, type=entity_type, first_seen=datetime.now().isoformat()
                )

                # Add metadata if provided
                if metadata:
                    for key, value in metadata.items():
                        self.graph.nodes[name][key] = value

                logger.debug(f"Added new entity: {name} ({entity_type})")
            else:
                # Update existing node
                self.graph.nodes[name]["last_updated"] = datetime.now().isoformat()

                # Update metadata if provided
                if metadata:
                    for key, value in metadata.items():
                        self.graph.nodes[name][key] = value

                logger.debug(f"Updated existing entity: {name}")

            return True

    def add_relationship(
        self,
        source_entity: str,
        relation_type: str,
        target_entity: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add a relationship between two entities.

        Args:
            source_entity (str): The source entity name
            relation_type (str): The type of relationship
            target_entity (str): The target entity name
            metadata (dict, optional): Additional metadata about the relationship

        Returns:
            bool: Whether the relationship was added successfully
        """
        with ErrorHandler(
            error_type="knowledge_graph", reraise=False, fallback_value=False
        ):
            # Standardize entity names
            source_entity = source_entity.strip()
            target_entity = target_entity.strip()

            if source_entity.lower() in self.entity_synonyms:
                source_entity = self.entity_synonyms[source_entity.lower()]

            if target_entity.lower() in self.entity_synonyms:
                target_entity = self.entity_synonyms[target_entity.lower()]

            # Ensure both entities exist
            if not self.graph.has_node(source_entity) or not self.graph.has_node(
                target_entity
            ):
                logger.warning(
                    f"Cannot create relationship: one or both entities do not exist"
                )
                return False

            # Check if relationship already exists
            if self.graph.has_edge(source_entity, target_entity):
                # Update existing relationship
                self.graph[source_entity][target_entity][
                    "last_updated"
                ] = datetime.now().isoformat()

                # If it's a different relation type, store as alternative
                current_type = self.graph[source_entity][target_entity].get("type")
                if current_type and current_type != relation_type:
                    alt_types = self.graph[source_entity][target_entity].get(
                        "alternative_types", []
                    )
                    if relation_type not in alt_types:
                        alt_types.append(relation_type)
                        self.graph[source_entity][target_entity][
                            "alternative_types"
                        ] = alt_types
            else:
                # Add new relationship
                self.graph.add_edge(
                    source_entity,
                    target_entity,
                    type=relation_type,
                    first_seen=datetime.now().isoformat(),
                )

            # Add metadata if provided
            if metadata:
                for key, value in metadata.items():
                    self.graph[source_entity][target_entity][key] = value

            logger.debug(
                f"Added relationship: {source_entity} --[{relation_type}]--> {target_entity}"
            )
            return True

    def get_entity(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an entity.

        Args:
            name (str): The name of the entity

        Returns:
            dict: Entity information or None if not found
        """
        with ErrorHandler(
            error_type="knowledge_graph", reraise=False, fallback_value=None
        ):
            # Standardize entity name
            name = name.strip()
            if name.lower() in self.entity_synonyms:
                name = self.entity_synonyms[name.lower()]

            if not self.graph.has_node(name):
                return None

            # Get node attributes
            attributes = dict(self.graph.nodes[name])

            # Get outgoing relationships
            outgoing = []
            for successor in self.graph.successors(name):
                edge_data = dict(self.graph[name][successor])
                outgoing.append(
                    {
                        "type": edge_data.get("type", "unknown"),
                        "target": successor,
                        "metadata": edge_data,
                    }
                )

            # Get incoming relationships
            incoming = []
            for predecessor in self.graph.predecessors(name):
                edge_data = dict(self.graph[predecessor][name])
                incoming.append(
                    {
                        "type": edge_data.get("type", "unknown"),
                        "source": predecessor,
                        "metadata": edge_data,
                    }
                )

            return {
                "name": name,
                "attributes": attributes,
                "outgoing_relationships": outgoing,
                "incoming_relationships": incoming,
            }

    def find_entities_by_type(self, entity_type: str) -> List[str]:
        """
        Find all entities of a specific type.

        Args:
            entity_type (str): The type of entity to find

        Returns:
            list: List of entity names
        """
        with ErrorHandler(
            error_type="knowledge_graph", reraise=False, fallback_value=[]
        ):
            if entity_type not in self.entity_types:
                logger.warning(f"Unknown entity type: {entity_type}")
                return []

            return list(self.entity_types[entity_type])

    def find_related_entities(
        self, entity_name: str, relation_type: Optional[str] = None, max_depth: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Find entities related to the given entity.

        Args:
            entity_name (str): The name of the entity
            relation_type (str, optional): Filter by relationship type
            max_depth (int): Maximum depth to search

        Returns:
            list: List of related entities
        """
        with ErrorHandler(
            error_type="knowledge_graph", reraise=False, fallback_value=[]
        ):
            # Standardize entity name
            entity_name = entity_name.strip()
            if entity_name.lower() in self.entity_synonyms:
                entity_name = self.entity_synonyms[entity_name.lower()]

            if not self.graph.has_node(entity_name):
                return []

            related_entities = []
            visited = {entity_name}
            queue = [(entity_name, 0)]

            while queue:
                current, depth = queue.pop(0)

                if depth > max_depth:
                    continue

                # Check outgoing relationships
                for successor in self.graph.successors(current):
                    if successor in visited:
                        continue

                    edge_data = self.graph[current][successor]
                    edge_type = edge_data.get("type", "unknown")

                    if relation_type is None or edge_type == relation_type:
                        related_entities.append(
                            {
                                "name": successor,
                                "type": self.graph.nodes[successor].get(
                                    "type", "unknown"
                                ),
                                "relation": edge_type,
                                "direction": "outgoing",
                                "depth": depth + 1,
                            }
                        )

                    if depth < max_depth:
                        visited.add(successor)
                        queue.append((successor, depth + 1))

                # Check incoming relationships
                for predecessor in self.graph.predecessors(current):
                    if predecessor in visited:
                        continue

                    edge_data = self.graph[predecessor][current]
                    edge_type = edge_data.get("type", "unknown")

                    if relation_type is None or edge_type == relation_type:
                        related_entities.append(
                            {
                                "name": predecessor,
                                "type": self.graph.nodes[predecessor].get(
                                    "type", "unknown"
                                ),
                                "relation": edge_type,
                                "direction": "incoming",
                                "depth": depth + 1,
                            }
                        )

                    if depth < max_depth:
                        visited.add(predecessor)
                        queue.append((predecessor, depth + 1))

            return related_entities

    def update_from_query(
        self, query: str, response: str, detected_entities: List[Dict[str, Any]]
    ):
        """
        Update the knowledge graph based on user query and response.

        Args:
            query (str): The user's query
            response (str): The chatbot's response
            detected_entities (list): Entities detected in the query/response
        """
        with ErrorHandler(error_type="knowledge_graph", reraise=False):
            if not detected_entities:
                return

            # Record this interaction
            timestamp = datetime.now().isoformat()
            interaction_id = f"interaction_{int(time.time())}"

            # Add a node for this interaction
            self.graph.add_node(
                interaction_id,
                type="interaction",
                timestamp=timestamp,
                query=query,
                response=response,
            )

            # Process each detected entity
            for entity in detected_entities:
                entity_name = entity.get("name", "").strip()
                entity_type = entity.get("type", "unknown")
                metadata = entity.get("metadata", {})

                if not entity_name:
                    continue

                # Add or update the entity
                self.add_entity(entity_name, entity_type, metadata)

                # Connect entity to this interaction
                self.graph.add_edge(
                    interaction_id, entity_name, type="mentions", timestamp=timestamp
                )

            # Look for relationships between entities
            if len(detected_entities) > 1:
                for i, entity1 in enumerate(detected_entities):
                    entity1_name = entity1.get("name", "").strip()
                    if not entity1_name:
                        continue

                    for entity2 in detected_entities[i + 1 :]:
                        entity2_name = entity2.get("name", "").strip()
                        if not entity2_name or entity1_name == entity2_name:
                            continue

                        # Check for explicit relationship
                        relation = entity1.get("relation_to", {}).get(entity2_name)
                        if relation:
                            self.add_relationship(
                                entity1_name,
                                relation,
                                entity2_name,
                                {"interaction": interaction_id},
                            )
                        else:
                            # Add co-occurrence relationship
                            self.add_relationship(
                                entity1_name,
                                "co-occurs_with",
                                entity2_name,
                                {"interaction": interaction_id},
                            )

            logger.debug(
                f"Updated knowledge graph from interaction {interaction_id} with {len(detected_entities)} entities"
            )

    def entity_summary(self, entity_name: str) -> str:
        """
        Generate a summary of what is known about an entity.

        Args:
            entity_name (str): The name of the entity

        Returns:
            str: A summary of the entity
        """
        with ErrorHandler(
            error_type="knowledge_graph",
            reraise=False,
            fallback_value=f"No information available about {entity_name}",
        ):
            # Standardize entity name
            entity_name = entity_name.strip()
            if entity_name.lower() in self.entity_synonyms:
                entity_name = self.entity_synonyms[entity_name.lower()]

            entity = self.get_entity(entity_name)
            if not entity:
                return f"No information available about {entity_name}"

            attributes = entity["attributes"]

            # Build summary
            summary = []
            summary.append(
                f"Summary for {entity_name} ({attributes.get('type', 'unknown')})"
            )

            # Add description if available
            if "description" in attributes:
                summary.append(f"\nDescription: {attributes['description']}")

            # Add other important attributes
            important_attrs = ["website", "category", "location"]
            for attr in important_attrs:
                if attr in attributes:
                    summary.append(f"{attr.capitalize()}: {attributes[attr]}")

            # Add outgoing relationships
            if entity["outgoing_relationships"]:
                summary.append("\nRelationships:")
                for rel in entity["outgoing_relationships"][:5]:  # Limit to 5
                    summary.append(f"- {entity_name} → {rel['target']} ({rel['type']})")

            # Add incoming relationships
            if entity["incoming_relationships"]:
                if not entity["outgoing_relationships"]:
                    summary.append("\nRelationships:")
                for rel in entity["incoming_relationships"][:5]:  # Limit to 5
                    summary.append(f"- {rel['source']} → {entity_name} ({rel['type']})")

            return "\n".join(summary)

    def get_islamic_finance_term_info(self, term: str) -> Dict[str, Any]:
        """
        Get information about an Islamic finance term.

        Args:
            term (str): The Islamic finance term

        Returns:
            dict: Information about the term or empty dict if not found
        """
        with ErrorHandler(
            error_type="knowledge_graph", reraise=False, fallback_value={}
        ):
            # Standardize term
            term = term.strip().lower()
            if term in self.entity_synonyms:
                term = self.entity_synonyms[term]

            # Check if we know this term
            if (
                term not in self.islamic_finance_terms
                and term not in self.entity_types["islamic_term"]
            ):
                return {}

            # Get entity if it exists
            for entity_name in self.entity_types["islamic_term"]:
                if entity_name.lower() == term:
                    return self.get_entity(entity_name) or {}

            return {}

    def export_graph(self, format_type: str = "json") -> str:
        """
        Export the knowledge graph to a specified format.

        Args:
            format_type (str): Export format ('json' or 'cytoscape')

        Returns:
            str: The exported graph data
        """
        with ErrorHandler(
            error_type="knowledge_graph", reraise=False, fallback_value="{}"
        ):
            if format_type == "json":
                # Basic JSON format
                data = {"nodes": [], "edges": []}

                # Add nodes
                for node, attrs in self.graph.nodes(data=True):
                    if attrs.get("type") == "interaction":
                        continue  # Skip interaction nodes for simplicity

                    node_data = {"id": node, "type": attrs.get("type", "unknown")}

                    # Add other attributes
                    for key, value in attrs.items():
                        if key not in ["id", "type"]:
                            node_data[key] = value

                    data["nodes"].append(node_data)

                # Add edges
                for source, target, attrs in self.graph.edges(data=True):
                    # Skip edges connected to interaction nodes
                    if (
                        self.graph.nodes[source].get("type") == "interaction"
                        or self.graph.nodes[target].get("type") == "interaction"
                    ):
                        continue

                    edge_data = {
                        "source": source,
                        "target": target,
                        "type": attrs.get("type", "unknown"),
                    }

                    data["edges"].append(edge_data)

                return json.dumps(data, indent=2)

            elif format_type == "cytoscape":
                # Cytoscape.js format
                elements = {"nodes": [], "edges": []}

                # Add nodes
                for node, attrs in self.graph.nodes(data=True):
                    if attrs.get("type") == "interaction":
                        continue  # Skip interaction nodes

                    node_data = {
                        "data": {
                            "id": node,
                            "label": node,
                            "type": attrs.get("type", "unknown"),
                        }
                    }

                    elements["nodes"].append(node_data)

                # Add edges
                edge_id = 0
                for source, target, attrs in self.graph.edges(data=True):
                    # Skip edges connected to interaction nodes
                    if (
                        self.graph.nodes[source].get("type") == "interaction"
                        or self.graph.nodes[target].get("type") == "interaction"
                    ):
                        continue

                    edge_data = {
                        "data": {
                            "id": f"e{edge_id}",
                            "source": source,
                            "target": target,
                            "label": attrs.get("type", "unknown"),
                        }
                    }

                    elements["edges"].append(edge_data)
                    edge_id += 1

                return json.dumps(elements, indent=2)

            else:
                return "{}"
