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
from HOGDB.graph.node_tuple import NodeTuple
from typing import List


# Load environment variables from the .env file
load_dotenv()

class GraphwithTupleStorage(GraphStorage):
    def __init__(self, db: Database) -> None:
        """
        Initialize GraphwithTupleStorage with a database connection.

        @param db: The database connection object.
        """
        super().__init__(db)

    def _delete_node_with_node_edges_from_database(self, tx, node: Node):
        """
        Remove a node from the database with HO edges.

        @param tx: The transaction object.
        @param node: The node to be deleted.
        @return: The transaction object after the operation.
        """
        return self._with_transaction(
            lambda tx: self.db.delete_node_with_node_edges(
                self.session, tx, node.labels, node.properties, Label("_adjacency")
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

    def add_node_tuple(self, node_tuple: NodeTuple) -> None:
        """
        Add a node-tuple to the database.

        @param node_tuple: The node-tuple to be added.
        """
        tuple_node = Node(
            [Label("_node_tuple")] + node_tuple.labels, node_tuple.properties
        )
        tx = self._add_node_to_database(None, tuple_node)

        for position, node in enumerate(node_tuple.nodes):
            node.labels = [Label("_node")] + node.labels
            edge = Edge(
                node,
                tuple_node,
                Label("_node_membership"),
                [Property("position_in_tuple", int, position)],
            )
            tx = self._add_edge_to_database(tx, edge)
            node.labels.pop(0)
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

    def delete_node_tuple(self, node_tuple: NodeTuple) -> None:
        """
        Remove a node-tuple from the database.

        @param node_tuple: The node-tuple to be deleted.
        """
        tuple_node = Node(
            [Label("_node_tuple")] + node_tuple.labels, node_tuple.properties
        )
        tx = self._delete_node_from_database(None, tuple_node)
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

    def update_node_tuple(
        self, node_tuple: NodeTuple, update_properties: List[Property]
    ) -> None:
        """
        Update the properties of a node-tuple in the database.

        @param node_tuple: The node-tuple to be updated.
        @param update_properties: The properties to update.
        """
        tuple_node = Node(
            [Label("_node_tuple")] + node_tuple.labels, node_tuple.properties
        )
        tx = self._update_node_in_database(None, tuple_node, update_properties)
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

    def get_node_tuple_count(self, labels: List[Label] = []) -> int:
        """
        Get the number of node-tuples in the database.

        @param labels: Optional labels to filter node-tuples. Defaults to an empty list.
        @return: The count of node-tuples.
        """
        count = self.db.node_count(self.session, [Label("_node_tuple")] + labels)
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

    def get_node_tuple(self, node_tuple_pattern: NodeTuple) -> NodeTuple | None:
        """
        Get a node-tuple from the database.

        @param node_tuple_pattern: The pattern to match node-tuples.
        @return: The matched node-tuple or None if not found.
        """
        (node_tuple_records, node_records) = self.db.match_node_tuple(
            self.session,
            [Label("_node_tuple")] + node_tuple_pattern.labels,
            node_tuple_pattern.properties,
        )
        assert len(node_tuple_records) <= 1
        if len(node_tuple_records) == 0:
            return None
        node_tuple_list = sorted(
            zip(
                [edge["position"] for edge in node_records.to_dict("records")],
                [
                    Node(
                        [Label(label) for label in node["labels"] if label != "_node"],
                        [
                            Property(key, type(value), value)
                            for key, value in node["properties"].items()
                        ],
                    )
                    for node in node_records.to_dict("records")
                ],
            )
        )
        nodes = [node for _, node in node_tuple_list]
        record = node_tuple_records.iloc[0]
        node_tuple = NodeTuple(
            nodes,
            [Label(label) for label in record["labels"] if label != "_node_tuple"],
            [
                Property(key, type(value), value)
                for key, value in record["properties"].items()
            ],
        )
        return node_tuple

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

    def export_node_tuples_to_csv(
        self,
        file_name: str,
        node_schema: Schema,
        node_tuple_labels: List[Label],
        node_tuple_schema: List[Schema],
    ) -> None:
        """
        Export node-tuples to a CSV file.

        @param file_name: The name of the CSV file.
        @param node_schema: The schema of the node list.
        @param node_tuple_labels: Labels of the node-tuples.
        @param node_tuple_schema: Schema of the node-tuples.
        """
        self.db.export_node_tuples_to_csv(
            self.session,
            file_name,
            node_schema,
            node_tuple_labels,
            node_tuple_schema,
        )

    def import_nodes_from_csv(
        self,
        file_name: str,
        labels: Label,
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

    def import_node_tuples_from_csv(
        self,
        file_path: str,
        node_schema: Schema,
        common_schema: List[Schema],
        node_tuple_labels: List[Label],
        node_tuple_schema: List[Schema],
        as_url: bool = False,
        delimiter: str = ",",
    ) -> None:
        """
        Import node-tuples from a CSV file.

        @param file_path: The path to the CSV file.
        @param node_schema: Schema of the node list.
        @param common_schema: Common schema for all nodes with tuple.
        @param node_tuple_labels: Labels of the node-tuples.
        @param node_tuple_schema: Schema of the node-tuples.
        @param as_url: Whether the file is a URL. Defaults to False.
        @param delimiter: The delimiter used in the CSV file. Defaults to ','.
        """
        self.db.import_node_tuples_from_csv(
            self.session,
            file_path,
            node_schema,
            common_schema,
            [Label("_node_tuple")] + node_tuple_labels,
            node_tuple_schema,
            as_url=as_url,
            delimiter=delimiter,
        )

    def _read_path(self, path: Path):
        """
        Read a path variable as a path with tuples.

        @param path: The path to read.
        @return: The path transformed with tuples.
        """
        return path.read_as_path_with_tuples()
