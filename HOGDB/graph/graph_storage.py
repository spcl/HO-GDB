# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main authors: Shriram Chandran
#               Jakub Cudak

from HOGDB.graph.node import Node
from HOGDB.graph.edge import Edge
from HOGDB.graph.path import Path
from HOGDB.db.db import Database
from HOGDB.db.label import Label
from HOGDB.db.property import Property
from HOGDB.db.schema import Schema
from typing import List
from dotenv import load_dotenv

import pandas as pd

load_dotenv()


class GraphStorage:
    def __init__(self, db: Database) -> None:
        """
        Initialize GraphStorage with a database connection.

        @param db: The database connection object.
        """
        self.db = db
        self._start_session()

    def close_connection(self) -> None:
        """
        Close the database connection.
        """
        self._end_session(self.session)
        self.db.close_driver()

    def _start_session(self) -> None:
        """
        Start a database session.
        """
        self.session = self.db.start_session()

    def _end_session(self, session) -> None:
        """
        End the database session.

        @param session: The session object to be ended.
        """
        self.db.end_session(session)

    def _with_transaction(self, operation, tx = None):
        """
        Context manager for database transaction handling.

        @param operation: The operation to execute within the transaction.
        @param tx: Optional transaction object. Defaults to None.
        @return: The transaction object after the operation.
        """
        tx = self.db._begin_transaction(self.session) if tx is None else tx
        operation(tx)
        return tx

    def _commit_transaction(self, tx) -> None:
        """
        Commit and close a transaction.

        @param tx: The transaction object to commit.
        """
        self.db._close_transaction(tx)

    def _with_transaction_return_records(self, operation, tx = None):
        """
        Context manager for database session handling.

        @param operation: The operation to execute within the transaction.
        @param tx: Optional transaction object. Defaults to None.
        @return: A tuple containing the transaction object and the records.
        """
        tx = self.db._begin_transaction(self.session) if tx is None else tx
        records = operation(tx)
        return tx, records

    def _add_node_to_database(self, tx, node: Node):
        """
        Add a node to the database.

        @param tx: The transaction object.
        @param node: The node to be added.
        @return: The transaction object after the operation.
        """
        return self._with_transaction(
            lambda tx: self.db.add_node(
                self.session, tx, labels=node.labels, properties=node.properties
            ),
            tx,
        )

    def _add_edge_to_database(self, tx, edge: Edge):
        """
        Add an edge to the database.

        @param tx: The transaction object.
        @param edge: The edge to be added.
        @return: The transaction object after the operation.
        """
        return self._with_transaction(
            lambda tx: self.db.add_edge(
                self.session,
                tx,
                start_node_labels=edge.start_node.labels,
                start_node_properties=edge.start_node.properties,
                end_node_labels=edge.end_node.labels,
                end_node_properties=edge.end_node.properties,
                edge_label=edge.label,
                edge_properties=edge.properties,
            ),
            tx,
        )

    def _delete_node_from_database(self, tx, node: Node):
        """
        Remove a node from the database.

        @param tx: The transaction object.
        @param node: The node to be deleted.
        @return: The transaction object after the operation.
        """
        return self._with_transaction(
            lambda tx: self.db.delete_node(
                self.session, tx, labels=node.labels, properties=node.properties
            ),
            tx,
        )

    def _delete_edge_from_database(self, tx, edge: Edge):
        """
        Remove an edge from the database.

        @param tx: The transaction object.
        @param edge: The edge to be deleted.
        @return: The transaction object after the operation.
        """
        return self._with_transaction(
            lambda tx: self.db.delete_edge(
                self.session,
                tx,
                start_node_labels=edge.start_node.labels,
                start_node_properties=edge.start_node.properties,
                end_node_labels=edge.end_node.labels,
                end_node_properties=edge.end_node.properties,
                edge_label=edge.label,
            ),
            tx,
        )

    def _update_node_in_database(
        self, tx, node: Node, update_properties: List[Property]
    ):
        """
        Update a node in the database.

        @param tx: The transaction object.
        @param node: The node to be updated.
        @param update_properties: The properties to update.
        @return: The transaction object after the operation.
        """
        return self._with_transaction(
            lambda tx: self.db.update_node(
                self.session, tx, node.labels, node.properties, update_properties
            ),
            tx,
        )

    def _update_edge_in_database(
        self, tx, edge: Edge, update_properties: List[Property]
    ):
        """
        Update an edge in the database.

        @param tx: The transaction object.
        @param edge: The edge to be updated.
        @param update_properties: The properties to update.
        @return: The transaction object after the operation.
        """
        return self._with_transaction(
            lambda tx: self.db.update_edge(
                self.session, tx, edge.label, edge.properties, update_properties
            ),
            tx,
        )

    def _get_nodes_from_database(self, node_pattern: Node) -> pd.DataFrame:
        """
        Get nodes from the database.

        @param node_pattern: The pattern to match nodes.
        @return: A dataframe containing the matched nodes.
        """
        return self.db.match_nodes(
            self.session, node_pattern.labels, node_pattern.properties
        )

    def _get_edges_from_database(self, edge_pattern: Edge) -> pd.DataFrame:
        """
        Get edges from the database.

        @param edge_pattern: The pattern to match edges.
        @return: A dataframe containing the matched edges.
        """
        return self.db.match_edges(
            self.session,
            edge_pattern.start_node.labels,
            edge_pattern.start_node.properties,
            edge_pattern.end_node.labels,
            edge_pattern.end_node.properties,
            edge_pattern.label,
            edge_pattern.properties,
        )

    def clear_graph(self) -> None:
        """
        Clear the graph storage.
        """
        self.db.clear_data(self.session)
        indexes = self.db.show_index_names(self.session)
        [self.db.drop_index(self.session, index) for index in indexes]

    def add_node(self, node: Node) -> None:
        """
        Add a node to the database.

        @param node: The node to be added.
        """
        node.labels = [Label("_node")] + node.labels
        tx = self._add_node_to_database(None, node)
        self._commit_transaction(tx)
        node.labels.pop(0)

    def add_edge(self, edge: Edge) -> None:
        """
        Add an edge to the database.

        @param edge: The edge to be added.
        """
        tx = self._add_edge_to_database(None, edge)
        self._commit_transaction(tx)

    def delete_node(self, node: Node) -> None:
        """
        Remove a node from the database.

        @param node: The node to be deleted.
        """
        node.labels = [Label("_node")] + node.labels
        tx = self._delete_node_from_database(None, node)
        self._commit_transaction(tx)
        node.labels.pop(0)

    def delete_edge(self, edge: Edge) -> None:
        """
        Remove an edge from the database.

        @param edge: The edge to be deleted.
        """
        tx = self._delete_edge_from_database(None, edge)
        self._commit_transaction(tx)

    def update_node(self, node: Node, update_properties: List[Property]) -> None:
        """
        Update a node in the database.

        @param node: The node to be updated.
        @param update_properties: The properties to update.
        """
        node.labels = [Label("_node")] + node.labels
        tx = self._update_node_in_database(None, node, update_properties)
        node.labels.pop(0)
        self._commit_transaction(tx)

    def update_edge(self, edge: Edge, update_properties: List[Property]) -> None:
        """
        Update an edge in the database.

        @param edge: The edge to be updated.
        @param update_properties: The properties to update.
        """
        tx = self._update_edge_in_database(None, edge, update_properties)
        self._commit_transaction(tx)

    def get_node_count(self, labels: List[Label] = None) -> int:
        """
        Get the number of nodes in the database.

        @param labels: Optional labels to filter nodes.
        @return: The count of nodes.
        """
        labels = labels if labels else []
        return self.db.node_count(self.session, [Label("_node")] + labels)

    def get_edge_count(self, label: Label = None) -> int:
        """
        Get the number of edges in the database.

        @param label: Optional label to filter edges.
        @return: The count of edges.
        """
        return self.db.edge_count(self.session, label)

    def get_node(self, node_pattern: Node) -> Node | None:
        """
        Get a node from the database.

        @param node_pattern: The pattern to match nodes.
        @return: The matched node or None if not found.
        """
        node_pattern.labels.insert(0, Label("_node"))
        records = self._get_nodes_from_database(node_pattern)
        node_pattern.labels.pop(0)
        assert len(records) <= 1
        if len(records) == 0:
            return None
        record = records.iloc[0]
        node = Node(
            [Label(label) for label in record["labels"] if label != "_node"],
            [
                Property(key, type(value), value)
                for key, value in record["properties"].items()
            ],
        )
        return node

    def get_edge(self, edge_pattern: Edge) -> Edge | None:
        """
        Get an edge from the database.

        @param edge_pattern: The pattern to match edges.
        @return: The matched edge or None if not found.
        """
        records = self._get_edges_from_database(edge_pattern)
        assert len(records) <= 1
        if len(records) == 0:
            return None
        record = records.iloc[0]
        edge = Edge(
            Node(
                [
                    Label(label)
                    for label in record["start_node_labels"]
                    if label != "_node"
                ],
                [
                    Property(key, type(value), value)
                    for key, value in record["start_node_properties"].items()
                ],
            ),
            Node(
                [
                    Label(label)
                    for label in record["end_node_labels"]
                    if label != "_node"
                ],
                [
                    Property(key, type(value), value)
                    for key, value in record["end_node_properties"].items()
                ],
            ),
            Label(record["edge_type"]),
            [
                Property(key, type(value), value)
                for key, value in record["edge_properties"].items()
            ],
        )
        return edge

    def import_nodes_from_csv(
        self,
        file_path: str,
        labels: List[Label],
        node_schema: List[Schema],
        as_url: bool = False,
        delimiter: str = ",",
    ) -> None:
        """
        Import nodes from a CSV file.

        @param file_path: The path to the CSV file.
        @param labels: Labels to assign to the imported nodes.
        @param node_schema: Schema of the nodes.
        @param as_url: Whether the file is a URL. Defaults to False.
        @param delimiter: The delimiter used in the CSV file. Defaults to ','.
        """
        self.db.import_nodes_from_csv(
            self.session,
            file_path,
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
        self.db.import_edges_from_csv(
            self.session,
            file_path,
            start_node_labels,
            start_node_schema,
            end_node_labels,
            end_node_schema,
            edge_label,
            edge_schema,
            as_url=as_url,
            delimiter=delimiter,
        )

    def export_nodes_to_csv(
        self, file_name: str, labels: List[Label], node_schema: List[Schema]
    ) -> None:
        """
        Export nodes to a CSV file.

        @param file_name: The name of the CSV file.
        @param labels: The labels of nodes to export.
        @param node_schema: The schema of the nodes.
        """
        self.db.export_nodes_to_csv(self.session, file_name, labels, node_schema)

    def export_edges_to_csv(
        self,
        file_name: str,
        start_node_label: Label,
        start_node_schema: List[Schema],
        end_node_label: Label,
        end_node_schema: List[Schema],
        edge_label: Label,
        edge_schema: List[Schema],
    ) -> None:
        """
        Export edges to a CSV file.

        @param file_name: The name of the CSV file.
        @param start_node_label: Label of the start nodes.
        @param start_node_schema: Schema of the start nodes.
        @param end_node_label: Label of the end nodes.
        @param end_node_schema: Schema of the end nodes.
        @param edge_label: Label of the edge.
        @param edge_schema: Schema of the edge.
        """
        self.db.export_edges_to_csv(
            self.session,
            file_name,
            start_node_label,
            start_node_schema,
            end_node_label,
            end_node_schema,
            edge_label,
            edge_schema,
        )

    def create_index(self, label: Label, property_keys: List[str]) -> None:
        """
        Create an index on a property.

        @param label: The label of the nodes.
        @param property_keys: The property keys to index.
        """
        self.db.create_index(self.session, label, property_keys)

    def show_indexes(self) -> List[str]:
        """
        Show all indexes.

        @return: A list of index names.
        """
        return self.db.show_index_names(self.session)

    def drop_index(self, index: str) -> None:
        """
        Drop an index.

        @param index: The name of the index to drop.
        """
        self.db.drop_index(self.session, index)

    def traverse_path(
        self,
        paths: List[Path],
        conditions: List[List[str]] = [],
        return_values: List[str] = [],
        sort: List[str] = None,
        limit: int = None,
    ) -> pd.DataFrame:
        """
        Traverse a path in the database.

        @param paths: A list of paths to traverse.
        @param conditions: Optional conditions for each path.
        @param return_values: The values to return from the traversal.
        @param sort: Optional sorting criteria.
        @param limit: Optional limit on the number of results.
        @return: A dataframe containing the traversal results.
        """
        vars_elements = [self._read_path(path) for path in paths]
        vars_list = [vars for vars, _ in vars_elements]
        elements_list = [elements for _, elements in vars_elements]
        conditions_list = (
            [[] for _ in range(len(vars_list))] if conditions == [] else conditions
        )
        assert len(vars_list) == len(elements_list) == len(conditions_list)
        records = self.db.traverse_path(
            self.session,
            vars_list,
            elements_list,
            conditions_list,
            return_values,
            sort,
            limit,
        )
        return records

    def _read_path(self, path: Path):
        """
        Read a path variable.

        @param path: The path to read.
        @return: The path for traversal.
        """
        return path.read_as_path()
