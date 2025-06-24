# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main authors: Shriram Chandran
#               Jakub Cudak

from dotenv import load_dotenv
from HOGDB.graph.graph_storage import GraphStorage
from HOGDB.graph.node import Node, Label, Property
from HOGDB.graph.edge import Edge
from HOGDB.db.db import Database
from HOGDB.db.schema import Schema
from HOGDB.graph.path import Path
from HOGDB.graph.subgraph import Subgraph, SubgraphEdge
from typing import List


# Load environment variables from the .env file
load_dotenv()

class GraphwithSubgraphStorage(GraphStorage):
    def __init__(self, db: Database) -> None:
        """
        Initialize GraphwithSubgraphStorage with a database connection.

        @param db: The database connection object.
        """
        super().__init__(db)

    def _delete_node_with_node_edges_from_database(
        self, tx, node: Node, edge_label: Label = Label("_adjacency")
    ):
        """
        Remove a node from the database with HO edges.

        @param tx: The transaction object.
        @param node: The node to be deleted.
        @param edge_label: The label of the edges to delete.
        @return: The transaction object after the operation.
        """
        return self._with_transaction(
            lambda tx: self.db.delete_node_with_node_edges(
                self.session, tx, node.labels, node.properties, edge_label
            ),
            tx,
        )

    def add_edge(self, edge: Edge) -> None:
        """
        Add an edge to the database.

        @param edge: The edge to be added.
        """
        edge_node = Node([Label("_edge"), edge.label], edge.properties)
        tx = self._add_node_to_database(None, edge_node)

        edge1 = Edge(edge.start_node, edge_node, Label("_adjacency"), [])
        tx = self._add_edge_to_database(tx, edge1)

        edge2 = Edge(edge_node, edge.end_node, Label("_adjacency"), [])
        tx = self._add_edge_to_database(tx, edge2)

        self._commit_transaction(tx)

    def add_subgraph(self, subgraph: Subgraph) -> None:
        """
        Add a subgraph to the database.

        @param subgraph: The subgraph to be added.
        """
        subgraph_node = Node(
            [Label("_subgraph")] + subgraph.labels, subgraph.properties
        )
        tx = self._add_node_to_database(None, subgraph_node)

        for node in subgraph.subgraph_nodes:
            node.labels = [Label("_node")] + node.labels
            edge = Edge(node, subgraph_node, Label("_node_membership"), [])
            tx = self._add_edge_to_database(tx, edge)
            node.labels.pop(0)

        for edge in subgraph.subgraph_edges:
            edge_node = Node([Label("_edge"), edge.label], edge.properties)
            edge = Edge(edge_node, subgraph_node, Label("_edge_membership"), [])
            tx = self._add_edge_to_database(tx, edge)

        self._commit_transaction(tx)

    def add_subgraph_edge(self, subgraph_edge: SubgraphEdge) -> None:
        """
        Add a subgraph edge to the database.

        @param subgraph_edge: The subgraph edge to be added.
        """
        subgraph_edge_node = Node(
            [Label("_subgraph_edge"), subgraph_edge.label], subgraph_edge.properties
        )
        tx = self._add_node_to_database(None, subgraph_edge_node)
        start_node = Node(
            [Label("_subgraph")] + subgraph_edge.start_subgraph.labels,
            subgraph_edge.start_subgraph.properties,
        )
        end_node = Node(
            [Label("_subgraph")] + subgraph_edge.end_subgraph.labels,
            subgraph_edge.end_subgraph.properties,
        )

        subgraph_edge1 = Edge(
            start_node,
            subgraph_edge_node,
            Label("_subgraph_adjacency"),
            [],
        )
        tx = self._add_edge_to_database(tx, subgraph_edge1)

        subgraph_edge2 = Edge(
            subgraph_edge_node,
            end_node,
            Label("_subgraph_adjacency"),
            [],
        )
        tx = self._add_edge_to_database(tx, subgraph_edge2)

        self._commit_transaction(tx)

    def delete_node(self, node: Node) -> None:
        """
        Remove a node from the database.

        @param node: The node to be deleted.
        """
        node.labels = [Label("_node")] + node.labels
        tx = self._delete_node_with_node_edges_from_database(None, node)
        node.labels.pop(0)
        self._commit_transaction(tx)

    def delete_edge(self, edge: Edge) -> None:
        """
        Remove an edge from the database.

        @param edge: The edge to be deleted.
        """
        edge_node = Node([Label("_edge"), edge.label], edge.properties)
        tx = self._delete_node_from_database(None, edge_node)
        self._commit_transaction(tx)

    def delete_subgraph(self, subgraph: Subgraph) -> None:
        """
        Remove a subgraph from the database.

        @param subgraph: The subgraph to be deleted.
        """
        subgraph_node = Node(
            [Label("_subgraph")] + subgraph.labels, subgraph.properties
        )
        tx = self._delete_node_with_node_edges_from_database(
            None, subgraph_node, Label("_subgraph_adjacency")
        )
        self._commit_transaction(tx)

    def delete_subgraph_edge(self, subgraph_edge: SubgraphEdge) -> None:
        """
        Remove a subgraph edge from the database.

        @param subgraph_edge: The subgraph edge to be deleted.
        """
        subgraph_edge_node = Node(
            [Label("_subgraph_edge"), subgraph_edge.label], subgraph_edge.properties
        )
        tx = self._delete_node_from_database(None, subgraph_edge_node)
        self._commit_transaction(tx)

    def update_edge(self, edge: Edge, update_properties: List[Property]) -> None:
        """
        Update the properties of an edge in the database.

        @param edge: The edge to be updated.
        @param update_properties: The properties to update.
        """
        edge_node = Node([Label("_edge"), edge.label], edge.properties)
        tx = self._update_node_in_database(None, edge_node, update_properties)
        self._commit_transaction(tx)

    def update_subgraph(self, subgraph: Subgraph, update_properties: List[Property]) -> None:
        """
        Update the properties of a subgraph in the database.

        @param subgraph: The subgraph to be updated.
        @param update_properties: The properties to update.
        """
        subgraph_node = Node(
            [Label("_subgraph")] + subgraph.labels, subgraph.properties
        )
        tx = self._update_node_in_database(None, subgraph_node, update_properties)
        self._commit_transaction(tx)

    def update_subgraph_edge(
        self, subgraph_edge: SubgraphEdge, update_properties: List[Property]
    ) -> None:
        """
        Update the properties of a subgraph edge in the database.

        @param subgraph_edge: The subgraph edge to be updated.
        @param update_properties: The properties to update.
        """
        subgraph_edge_node = Node(
            [Label("_subgraph_edge"), subgraph_edge.label], subgraph_edge.properties
        )
        tx = self._update_node_in_database(None, subgraph_edge_node, update_properties)
        self._commit_transaction(tx)

    def get_edge_count(self, label: Label = None) -> int:
        """
        Get the number of edges in the database.

        @param label: Optional label to filter edges. Defaults to None.
        @return: The count of edges.
        """
        count = self.db.node_count(
            self.session, [Label("_edge")] + ([label] if label else [])
        )
        return count

    def get_subgraph_count(self, labels: List[Label] = []) -> int:
        """
        Get the number of subgraphs in the database.

        @param labels: Optional labels to filter subgraphs. Defaults to an empty list.
        @return: The count of subgraphs.
        """
        count = self.db.node_count(self.session, [Label("_subgraph")] + labels)
        return count

    def get_subgraph_edge_count(self, label: Label = None) -> int:
        """
        Get the number of subgraph edges in the database.

        @param label: Optional label to filter subgraph edges. Defaults to None.
        @return: The count of subgraph edges.
        """
        count = self.db.node_count(
            self.session, [Label("_subgraph_edge")] + ([label] if label else [])
        )
        return count

    def get_edge(self, edge_pattern: Edge) -> Edge | None:
        """
        Get edges from the database.

        @param edge_pattern: The pattern to match edges.
        @return: The matched edge or None if not found.
        """
        records = self.db.match_node_edges(
            self.session,
            [Label("_node")] + edge_pattern.start_node.labels,
            edge_pattern.start_node.properties,
            [Label("_node")] + edge_pattern.end_node.labels,
            edge_pattern.end_node.properties,
            [Label("_edge"), edge_pattern.label],
            edge_pattern.properties,
            Label("_adjacency"),
        )
        assert len(records) <= 1
        if len(records) == 0:
            return None
        record = records.iloc[0]
        start_node = Node(
            [Label(label) for label in record["start_labels"] if label != "_node"],
            [
                Property(key, type(value), value)
                for key, value in record["start_properties"].items()
            ],
        )
        end_node = Node(
            [Label(label) for label in record["end_labels"] if label != "_node"],
            [
                Property(key, type(value), value)
                for key, value in record["end_properties"].items()
            ],
        )
        edge = Edge(
            start_node,
            end_node,
            next(Label(label) for label in record["edge_labels"] if label != "_edge"),
            [
                Property(key, type(value), value)
                for key, value in record["edge_properties"].items()
            ],
        )
        return edge

    def get_subgraph(self, subgraph_pattern: Subgraph) -> Subgraph | None:
        """
        Get a subgraph from the database.

        @param subgraph_pattern: The pattern to match subgraphs.
        @return: The matched subgraph or None if not found.
        """
        (subgraph_records, node_records, edge_records) = self.db.match_subgraph(
            self.session,
            [Label("_subgraph")] + subgraph_pattern.labels,
            subgraph_pattern.properties,
        )
        assert len(subgraph_records) <= 1
        if len(subgraph_records) == 0:
            return None
        subgraph_nodes = [
            Node(
                [Label(label) for label in node["labels"] if label != "_node"],
                [
                    Property(key, type(value), value)
                    for key, value in node["properties"].items()
                ],
            )
            for node in node_records.to_dict("records")
        ]
        subgraph_edges = [
            Edge(
                Node(
                    [
                        Label(label)
                        for label in edge["start_labels"]
                        if label != "_node"
                    ],
                    [
                        Property(key, type(value), value)
                        for key, value in edge["start_properties"].items()
                    ],
                ),
                Node(
                    [Label(label) for label in edge["end_labels"] if label != "_node"],
                    [
                        Property(key, type(value), value)
                        for key, value in edge["end_properties"].items()
                    ],
                ),
                next(Label(label) for label in edge["edge_labels"] if label != "_edge"),
                [
                    Property(key, type(value), value)
                    for key, value in edge["edge_properties"].items()
                ],
            )
            for edge in edge_records.to_dict("records")
        ]
        record = subgraph_records.iloc[0]
        subgraph = Subgraph(
            subgraph_nodes,
            subgraph_edges,
            [Label(label) for label in record["labels"] if label != "_subgraph"],
            [
                Property(key, type(value), value)
                for key, value in record["properties"].items()
            ],
        )
        return subgraph

    def get_subgraph_edge(self, subgraph_edge_pattern: SubgraphEdge) -> SubgraphEdge | None:
        """
        Get subgraph edges from the database.

        @param subgraph_edge_pattern: The pattern to match subgraph edges.
        @return: The matched subgraph edge or None if not found.
        """
        records = self.db.match_node_edges(
            self.session,
            [Label("_subgraph")] + subgraph_edge_pattern.start_subgraph.labels,
            subgraph_edge_pattern.start_subgraph.properties,
            [Label("_subgraph")] + subgraph_edge_pattern.end_subgraph.labels,
            subgraph_edge_pattern.end_subgraph.properties,
            [Label("_subgraph_edge"), subgraph_edge_pattern.label],
            subgraph_edge_pattern.properties,
            Label("_subgraph_adjacency"),
        )
        assert len(records) <= 1
        if len(records) == 0:
            return None
        record = records.iloc[0]
        start_subgraph = Subgraph(
            None,
            None,
            [Label(label) for label in record["start_labels"] if label != "_subgraph"],
            [
                Property(key, type(value), value)
                for key, value in record["start_properties"].items()
            ],
        )
        end_subgraph = Subgraph(
            None,
            None,
            [Label(label) for label in record["end_labels"] if label != "_subgraph"],
            [
                Property(key, type(value), value)
                for key, value in record["end_properties"].items()
            ],
        )
        edge = SubgraphEdge(
            start_subgraph,
            end_subgraph,
            next(
                Label(label)
                for label in record["edge_labels"]
                if label != "_subgraph_edge"
            ),
            [
                Property(key, type(value), value)
                for key, value in record["edge_properties"].items()
            ],
        )
        return edge

    def export_nodes_to_csv(
        self, file_name: str, labels: List[Label], node_schema: List[Schema]
    ) -> None:
        """
        Export nodes to a CSV file.

        @param file_name: The name of the CSV file.
        @param labels: The labels of nodes to export.
        @param node_schema: The schema of the nodes.
        """
        self.db.export_nodes_to_csv(
            self.session, file_name, [Label("_node")] + labels, node_schema
        )

    def export_edges_to_csv(
        self,
        file_name: str,
        start_node_labels: List[Label],
        start_node_schema: List[Schema],
        end_node_labels: List[Label],
        end_node_schema: List[Schema],
        edge_label: Label,
        edge_schema: List[Schema],
    ) -> None:
        """
        Export edges to a CSV file.

        @param file_name: The name of the CSV file.
        @param start_node_labels: Labels of the start nodes.
        @param start_node_schema: Schema of the start nodes.
        @param end_node_labels: Labels of the end nodes.
        @param end_node_schema: Schema of the end nodes.
        @param edge_label: Label of the edge.
        @param edge_schema: Schema of the edge.
        """
        self.db.export_node_edges_to_csv(
            self.session,
            file_name,
            [Label("_node")] + start_node_labels,
            start_node_schema,
            [Label("_node")] + end_node_labels,
            end_node_schema,
            [Label("_edge"), edge_label],
            edge_schema,
            Label("_adjacency"),
        )

    def export_subgraphs_to_csv(
        self,
        file_name: str,
        node_schema: Schema,
        edge_schema: Schema,
        subgraph_labels: List[Label],
        subgraph_schema: List[Schema],
    ) -> None:
        """
        Export subgraphs to a CSV file.

        @param file_name: The name of the CSV file.
        @param node_schema: The schema of the node list.
        @param edge_schema: The schema of the nodes in the edge list.
        @param subgraph_labels: Labels of the subgraphs.
        @param subgraph_schema: Schema of the subgraphs.
        """
        self.db.export_subgraphs_to_csv(
            self.session,
            file_name,
            node_schema,
            edge_schema,
            [Label("_subgraph")] + subgraph_labels,
            subgraph_schema,
        )

    def export_subgraph_edges_to_csv(
        self,
        file_name,
        start_subgraph_labels: List[Label],
        start_subgraph_schema: List[Schema],
        end_subgraph_labels: List[Label],
        end_subgraph_schema: List[Schema],
        edge_label: Label,
        edge_schema: List[Schema],
    ) -> None:
        """
        Export subgraph edges to a CSV file.

        @param file_name: The name of the CSV file.
        @param start_subgraph_labels: Labels of the start subgraphs.
        @param start_subgraph_schema: Schema of the start subgraphs.
        @param end_subgraph_labels: Labels of the end subgraphs.
        @param end_subgraph_schema: Schema of the end subgraphs.
        @param edge_label: Label of the edge.
        @param edge_schema: Schema of the edge.
        """
        self.db.export_node_edges_to_csv(
            self.session,
            file_name,
            [Label("_subgraph")] + start_subgraph_labels,
            start_subgraph_schema,
            [Label("_subgraph")] + end_subgraph_labels,
            end_subgraph_schema,
            [Label("_subgraph_edge"), edge_label],
            edge_schema,
            Label("_subgraph_adjacency"),
        )

    def import_nodes_from_csv(
        self,
        file_name: str,
        labels: List[Label],
        node_schema: List[Schema],
        as_url: bool = False,
        delimiter: str = ",",
    ) -> None:
        """
        Import nodes from a CSV file.

        @param file_name: The name of the CSV file.
        @param labels: Labels to assign to the imported nodes.
        @param node_schema: Schema of the nodes.
        @param as_url: Whether the file is a URL. Defaults to False.
        @param delimiter: The delimiter used in the CSV file. Defaults to ','.
        """
        self.db.import_nodes_from_csv(
            self.session,
            file_name,
            [Label("_node")] + labels,
            node_schema,
            as_url=as_url,
            delimiter=delimiter,
        )

    def import_edges_from_csv(
        self,
        file_path: str,
        start_node_labels: List[Label],
        start_node_schema: List[Schema],
        end_node_labels: List[Label],
        end_node_schema: List[Schema],
        edge_label: Label,
        edge_schema: List[Schema],
        as_url: bool = False,
        delimiter: str = ",",
    ) -> None:
        """
        Import edges from a CSV file.

        @param file_path: The path to the CSV file.
        @param start_node_labels: Labels of the start nodes.
        @param start_node_schema: Schema of the start nodes.
        @param end_node_labels: Labels of the end nodes.
        @param end_node_schema: Schema of the end nodes.
        @param edge_label: Label of the edge.
        @param edge_schema: Schema of the edge.
        @param as_url: Whether the file is a URL. Defaults to False.
        @param delimiter: The delimiter used in the CSV file. Defaults to ','.
        """
        self.db.import_node_edges_from_csv(
            self.session,
            file_path,
            [Label("_node")] + start_node_labels,
            start_node_schema,
            [Label("_node")] + end_node_labels,
            end_node_schema,
            [Label("_edge"), edge_label],
            edge_schema,
            Label("_adjacency"),
            as_url=as_url,
            delimiter=delimiter,
        )

    def import_subgraphs_from_csv(
        self,
        file_path: str,
        node_schema: Schema,
        node_schema_in_edge: Schema,
        common_schema: List[Schema],
        subgraph_labels: List[Label],
        subgraph_schema: List[Schema],
        as_url: bool = False,
        delimiter: str = ",",
    ) -> None:
        """
        Import subgraphs from a CSV file.

        @param file_path: The path to the CSV file.
        @param node_schema: Schema of the node list.
        @param node_schema_in_edge: Schema of the nodes in the edge list.
        @param common_schema: Common schema for all nodes with subgraph.
        @param subgraph_labels: Labels of the subgraphs.
        @param subgraph_schema: Schema of the subgraphs.
        @param as_url: Whether the file is a URL. Defaults to False.
        @param delimiter: The delimiter used in the CSV file. Defaults to ','.
        """
        self.db.import_subgraphs_from_csv(
            self.session,
            file_path,
            node_schema,
            node_schema_in_edge,
            common_schema,
            [Label("_subgraph")] + subgraph_labels,
            subgraph_schema,
            as_url=as_url,
            delimiter=delimiter,
        )

    def import_subgraph_edges_from_csv(
        self,
        file_path: str,
        start_subgraph_labels: List[Label],
        start_subgraph_schema: List[Schema],
        end_subgraph_labels: List[Label],
        end_subgraph_schema: List[Schema],
        edge_label: Label,
        edge_schema: List[Schema],
        as_url: bool = False,
        delimiter: str = ",",
    ) -> None:
        """
        Import subgraph edges from a CSV file.

        @param file_path: The path to the CSV file.
        @param start_subgraph_labels: Labels of the start subgraphs.
        @param start_subgraph_schema: Schema of the start subgraphs.
        @param end_subgraph_labels: Labels of the end subgraphs.
        @param end_subgraph_schema: Schema of the end subgraphs.
        @param edge_label: Label of the edge.
        @param edge_schema: Schema of the edge.
        @param as_url: Whether the file is a URL. Defaults to False.
        @param delimiter: The delimiter used in the CSV file. Defaults to ','.
        """
        self.db.import_node_edges_from_csv(
            self.session,
            file_path,
            [Label("_subgraph")] + start_subgraph_labels,
            start_subgraph_schema,
            [Label("_subgraph")] + end_subgraph_labels,
            end_subgraph_schema,
            [Label("_subgraph_edge"), edge_label],
            edge_schema,
            Label("_subgraph_adjacency"),
            as_url=as_url,
            delimiter=delimiter,
        )

    def _read_path(self, path: Path):
        """
        Read a path variable as a path with subgraphs.

        @param path: The path to read.
        @return: The path transformed with subgraphs.
        """
        return path.read_as_path_with_subgraphs()
