# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Shriram Chandran

from HOGDB.graph.graph_storage import GraphStorage
from HOGDB.graph.node import Node
from HOGDB.graph.edge import Edge
from HOGDB.graph.hyperedge import HyperEdge
from HOGDB.graph.path import Path
from HOGDB.db.db import Database
from HOGDB.db.label import Label
from HOGDB.db.schema import Schema
from HOGDB.db.property import Property
from typing import List
from dotenv import load_dotenv


load_dotenv()

class HyperGraphStorage(GraphStorage):
    def __init__(self, db: Database) -> None:
        """
        Initialize HyperGraphStorage with a database connection.

        @param db: The database connection object.
        """
        super().__init__(db)

    def add_hyperedge(self, edge: HyperEdge) -> None:
        """
        Add a hyperedge to the database.

        @param edge: The hyperedge to be added.
        """
        edge_node = Node(
            labels=[Label("_hyperedge"), edge.label], properties=edge.properties
        )
        tx = self._add_node_to_database(None, edge_node)

        for node in edge.nodes:
            db_edge1 = Edge(node, edge_node, Label("_adjacency"), edge.properties)
            db_edge2 = Edge(edge_node, node, Label("_adjacency"), edge.properties)
            tx = self._add_edge_to_database(tx, db_edge1)
            tx = self._add_edge_to_database(tx, db_edge2)

        self._commit_transaction(tx)

    def delete_hyperedge(self, edge: HyperEdge) -> None:
        """
        Remove a hyperedge from the database.

        @param edge: The hyperedge to be deleted.
        """
        edge_node = Node(
            labels=[Label("_hyperedge"), edge.label], properties=edge.properties
        )
        tx = self._delete_node_from_database(None, edge_node)
        self._commit_transaction(tx)

    def update_hyperedge(self, hyperedge: HyperEdge, update_properties: List[Property]) -> None:
        """
        Update the properties of a hyperedge in the database.

        @param hyperedge: The hyperedge to be updated.
        @param update_properties: The properties to update.
        """
        hyperedge_node = Node(
            labels=[Label("_hyperedge"), hyperedge.label],
            properties=hyperedge.properties,
        )
        tx = self._update_node_in_database(None, hyperedge_node, update_properties)
        self._commit_transaction(tx)

    def get_hyperedge(self, hyperedge_pattern: HyperEdge) -> HyperEdge | None:
        """
        Retrieve a hyperedge from the database.

        @param hyperedge_pattern: The pattern to match the hyperedge.
        @return: The matched hyperedge or None if not found.
        """
        (node_records, edge_records) = self.db.match_hyperedge(
            self.session,
            [],
            [hyperedge_pattern.label],
            hyperedge_pattern.properties,
        )
        assert len(edge_records) <= 1
        if len(edge_records) == 0:
            return None
        hyperedge_nodes = [
            Node(
                [Label(label) for label in node["labels"] if label != "_node"],
                [
                    Property(key, type(value), value)
                    for key, value in node["properties"].items()
                ],
            )
            for node in node_records.to_dict("records")
        ]
        record = edge_records.iloc[0]
        hyperedge = HyperEdge(
            hyperedge_nodes,
            next(Label(label) for label in record["labels"] if label != "_hyperedge"),
            [
                Property(key, type(value), value)
                for key, value in record["properties"].items()
            ],
        )
        return hyperedge

    def export_hyperedges_to_csv(
        self,
        file_name: str,
        node_labels: List[Label],
        node_schema: Schema,
        hyperedge_label: Label,
        hyperedge_schema: List[Schema],
    ) -> None:
        """
        Export hyperedges to a CSV file.

        @param file_name: The name of the CSV file.
        @param node_labels: Labels of the nodes in the hyperedge.
        @param node_schema: Schema of the nodes.
        @param hyperedge_label: Label of the hyperedge.
        @param hyperedge_schema: Schema of the hyperedge.
        """
        self.db.export_hyperedges_to_csv(
            self.session,
            file_name,
            [Label("_node")] + node_labels,
            node_schema,
            [Label("_hyperedge"), hyperedge_label],
            hyperedge_schema,
        )

    def import_nodes_from_csv(
        self, file_name: str, labels: Label, node_schema: List[Schema]
    ) -> None:
        """
        Import nodes from a CSV file.

        @param file_name: The name of the CSV file.
        @param labels: Labels to assign to the imported nodes.
        @param node_schema: Schema of the nodes.
        """
        self.db.import_nodes_from_csv(
            self.session, file_name, [Label("_node")] + labels, node_schema
        )

    def import_hyperedges_from_csv(
        self,
        file_name: str,
        node_labels: List[Label],
        node_schema: Schema,
        common_schema: List[Schema],
        hyperedge_label: Label,
        hyperedge_schema: List[Schema],
        as_url: bool = False,
        delimiter: str = ",",
    ) -> None:
        """
        Import hyperedges from a CSV file.

        @param file_name: The name of the CSV file.
        @param node_labels: Labels of the nodes in the hyperedge.
        @param node_schema: Schema of the nodes.
        @param common_schema: Common schema for all nodes in the hyperedge.
        @param hyperedge_label: Label of the hyperedge.
        @param hyperedge_schema: Schema of the hyperedge.
        @param as_url: Whether the file is a URL. Defaults to False.
        @param delimiter: The delimiter used in the CSV file. Defaults to ','.
        """
        self.db.import_hyperedges_from_csv(
            self.session,
            file_name,
            [Label("_node")] + node_labels,
            node_schema,
            common_schema,
            [Label("_hyperedge"), hyperedge_label],
            hyperedge_schema,
            as_url=as_url,
            delimiter=delimiter,
        )

    def get_hyperedge_count(self, labels: List[Label] = []) -> int:
        """
        Get the number of hyperedges in the database.

        @param labels: Optional labels to filter hyperedges.
        @return: The count of hyperedges.
        """
        return self.db.node_count(self.session, [Label("_hyperedge")] + labels)

    def _read_path(self, path: Path):
        """
        Read a path variable as a path with hypergraphs.

        @param path: The path to read.
        @return: The path transformed with hypergraphs.
        """
        return path.read_as_path_with_hypergraph()
