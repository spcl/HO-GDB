# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Shriram Chandran
#
# contributions: Jakub Cudak

from abc import ABC, abstractmethod
from HOGDB.db.label import Label
from HOGDB.db.property import Property
from HOGDB.db.schema import Schema
from typing import Dict, List, Optional, Tuple
import pandas as pd


class Session(ABC):
    """
    Abstract base class for a database session. Defines the interface for running queries and managing transactions.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def run(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        pass

    @abstractmethod
    def begin_transaction(self):
        pass

    @abstractmethod
    def close(self):
        pass


class Transaction(ABC):
    """
    Abstract base class for a database transaction. Defines the interface for running transactional queries and committing/closing.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def run(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        pass

    @abstractmethod
    def commit(self):
        pass

    @abstractmethod
    def close(self):
        pass


class Database(ABC):
    """
    Abstract base class that defines the interface for interacting with a graph database.
    """

    @staticmethod
    def format_properties(properties: Optional[Property]) -> str:
        """
        Convert a dictionary of properties to a Cypher-compatible string.

        @param properties: Dictionary of properties to be converted.
        @return: Cypher-compatible string.
        """
        if not properties:
            return ""
        return "{" + ", ".join(repr(property) for property in properties) + "}"

    @staticmethod
    def format_labels(labels: Optional[List[Label]]) -> str:
        """
        Convert a list of labels to a Cypher-compatible string.

        @param labels: List of labels.
        @return: Cypher-compatible string.
        """
        if not labels:
            return ""
        return ":" + ":".join(repr(label) for label in labels)

    @abstractmethod
    def export_nodes_to_csv(
        self,
        session: Session,
        file_name: str,
        labels: List[Label],
        node_schema: List[Schema],
    ) -> None:
        """
        Export nodes to a CSV file.

        @param session: Database session.
        @param file_name: Name and path of the output file.
        @param labels: List of node labels to export.
        @param node_schema: List of property schemas for the nodes.
        """
        pass

    @abstractmethod
    def export_edges_to_csv(
        self,
        session: Session,
        file_name: str,
        start_labels: List[Label],
        start_schema: List[Schema],
        end_labels: List[Label],
        end_schema: List[Schema],
        edge_label: Label,
        edge_schema: List[Schema],
    ) -> None:
        """
        Export edges to a CSV file.

        @param session: Database session.
        @param file_name: Name and path of the output file.
        @param start_labels: List of labels for the start nodes.
        @param start_schema: List of property schemas for the start nodes.
        @param end_labels: List of labels for the end nodes.
        @param end_schema: List of property schemas for the end nodes.
        @param edge_label: Export edges with that label.
        @param edge_schema: List of property schemas for the edges.
        """
        pass

    @abstractmethod
    def export_node_edges_to_csv(
        self,
        session: Session,
        file_name: str,
        start_labels: List[Label],
        start_schema: List[Schema],
        end_labels: List[Label],
        end_schema: List[Schema],
        node_edge_labels: List[Label],
        node_edge_schema: List[Schema],
        edge_label: Label,
    ) -> None:
        """
        Export HO edges to a CSV file.

        @param session: Database session.
        @param file_name: Name and path of the output file.
        @param start_labels: List of labels for the start nodes.
        @param start_schema: List of property schemas for the start nodes.
        @param end_labels: List of labels for the end nodes.
        @param end_schema: List of property schemas for the end nodes.
        @param node_edge_labels: List of labels for the HO edges.
        @param node_edge_schema: List of property schemas for the HO edges.
        @param edge_label: Label of the edges to the HO edges.
        """
        pass

    @abstractmethod
    def export_subgraphs_to_csv(
        self,
        session: Session,
        file_name: str,
        node_schema: Schema,
        edge_schema: Schema,
        subgraph_labels: List[Label],
        subgraph_schema: List[Schema],
    ) -> None:
        """
        Export subgraph collections to a CSV file.

        @param session: Database session.
        @param file_name: Name and path of the output file.
        @param node_schema: Node schema.
        @param edge_schema: Edge schema.
        @param subgraph_labels: List of subgraph collection labels to export.
        @param subgraph_schema: List of property schemas for the subgraph collections.
        """
        pass

    @abstractmethod
    def export_hyperedges_to_csv(
        self,
        session: Session,
        file_name: str,
        node_labels: List[Label],
        node_schema: Schema,
        hyperedge_labels: List[Label],
        hyperedge_schema: List[Schema],
    ) -> None:
        """
        Export hyperedges to a CSV file.

        @param session: Database session.
        @param file_name: Name and path of the output file.
        @param node_labels: List of node labels.
        @param node_schema: Property schema for the nodes.
        @param hyperedge_labels: Export hyperedges with those labels.
        @param hyperedge_schema: List of property schemas for the hyperedges.
        """
        pass

    @abstractmethod
    def export_node_tuples_to_csv(
        self,
        session: Session,
        file_name: str,
        node_schema: Schema,
        tuple_labels: List[Label],
        tuple_schema: List[Schema],
    ) -> None:
        """
        Export node-tuples to a CSV file.

        @param session: Database session.
        @param file_name: Name and path of the output file.
        @param node_schema: Property schema for the nodes.
        @param tuple_labels: Export node-tuples with those labels.
        @param tuple_schema: List of property schemas for the node-tuples.
        """
        pass

    @abstractmethod
    def import_nodes_from_csv(
        self,
        session: Session,
        file_name: str,
        labels: List[Label],
        node_schema: List[Schema],
        as_url: bool = False,
        batch_size: int = 10000,
        delimiter: str = ",",
    ) -> None:
        """
        Import nodes from a CSV file into the graph database.

        @param session: Database session.
        @param file_name: Name and path of the input file.
        @param labels: List of labels for the nodes.
        @param node_schema: List of property schemas for the nodes.
        @param as_url: Treat file_name as URL. Defaults to False.
        @param batch_size: Number of nodes to import at a time. Defaults to 10000.
        @param delimiter: Delimiter used in the CSV file. Defaults to ','.
        """
        pass

    @abstractmethod
    def import_edges_from_csv(
        self,
        session: Session,
        file_name: str,
        start_labels: List[Label],
        start_schema: List[Schema],
        end_labels: List[Label],
        end_schema: List[Schema],
        edge_label: Label,
        edge_schema: List[Schema],
        as_url: bool = False,
        batch_size: int = 10000,
        delimiter: str = ",",
    ) -> None:
        """
        Import edges from a CSV file into the graph databse.

        @param session: Database session.
        @param file_name: Name and path of the input file.
        @param start_labels: List of labels for the start nodes.
        @param start_schema: List of property schemas for the start nodes.
        @param end_labels: List of labels for the end nodes.
        @param end_schema: List of property schemas for the end nodes.
        @param edge_label: Label to be used for the edges.
        @param edge_schema: List of property schemas for the edges.
        @param as_url: Treat file_name as URL. Defaults to False.
        @param batch_size: Number of edges to import at a time. Defaults to 10000.
        @param delimiter: Delimiter used in the CSV file. Defaults to ','.
        """
        pass

    @abstractmethod
    def import_node_edges_from_csv(
        self,
        session: Session,
        file_name: str,
        start_labels: List[Label],
        start_schema: List[Schema],
        end_labels: List[Label],
        end_schema: List[Schema],
        node_edge_labels: List[Label],
        node_edge_schema: List[Schema],
        edge_label: Label,
        as_url: bool = False,
        batch_size: int = 10000,
        delimiter: str = ",",
    ) -> None:
        """
        Import HO edges from a CSV file into the graph database.

        @param session: Database session.
        @param file_name: Name and path of the input file.
        @param start_labels: List of labels for the start nodes.
        @param start_schema: List of property schemas for the start nodes.
        @param end_labels: List of labels for the end nodes.
        @param end_schema: List of property schemas for the end nodes.
        @param node_edge_label: Label to be used for the HO edges.
        @param node_edge_schema: List of property schemas for the HO edges.
        @param edge_label: Label for the edges to the HO edges.
        @param as_url: Treat file_name as URL. Defaults to False.
        @param batch_size: Number of edges to import at a time. Defaults to 10000.
        @param delimiter: Delimiter used in the CSV file. Defaults to ','.
        """
        pass

    @abstractmethod
    def import_hyperedges_from_csv(
        self,
        session: Session,
        file_name: str,
        node_labels: List[Label],
        node_schema: Schema,
        common_schema: List[Schema],
        hyperedge_labels: List[Label],
        hyperedge_schema: List[Schema],
        as_url: bool = False,
        batch_size: int = 5000,
        delimiter: str = ",",
    ) -> None:
        """
        Import hyperedges from a CSV file into the graph database.

        @param session: Database session.
        @param file_name: Name and path of the input file.
        @param node_labels: List of labels for the nodes.
        @param node_schema: Property schema for the nodes.
        @param common_schema: List of property schemas common to the nodes.
        @param hyperedge_labels: List of labels to be used for the hyperedges.
        @param hyperedge_schema: List of property schemas for the hyperedges.
        @param as_url: Treat file_name as URL. Defaults to False.
        @param batch_size: Number of hyperedges to import at a time. Defaults to 5000.
        @param delimiter: Delimiter used in the CSV file. Defaults to ','.
        """
        pass

    @abstractmethod
    def import_node_tuples_from_csv(
        self,
        session: Session,
        file_name: str,
        node_schema: Schema,
        common_schema: List[Schema],
        tuple_labels: List[Label],
        tuple_properties: List[Schema],
        as_url: bool = False,
        batch_size: int = 1000,
        delimiter: str = ",",
    ) -> None:
        """
        Import node-tuples from a CSV file into the graph database.

        @param session: Database session.
        @param file_name: Name and path of the input file.
        @param node_schema: Property schema for the nodes.
        @param common_schema: List of property schemas common to the nodes.
        @param tuple_labels: List of labels to be used for the node-tuples.
        @param tuple_schema: List of TODO schemas for the node-tuples.
        @param as_url: Treat file_name as URL. Defaults to False.
        @param batch_size: Number of hyperedges to import at a time. Defaults to 1000.
        @param delimiter: Delimiter used in the CSV file. Defaults to ','.
        """
        pass

    @abstractmethod
    def import_subgraphs_from_csv(
        self,
        session: Session,
        file_name: str,
        node_schema: Schema,
        edge_schema: Schema,
        common_schema: List[Schema],
        subgraph_labels: List[Label],
        subgraph_schema: List[Schema],
        as_url: bool = False,
        batch_size: int = 1000,
        delimiter: str = ",",
    ) -> None:
        """
        Import subgraph collections from a CSV file into the graph database.

        @param session: Database session.
        @param file_name: Name and path of the input file.
        @param node_schema: Property schema for the nodes.
        @param edge_schema: Property schema for the edges.
        @param common_schema:  List of property schemas common to the nodes.
        @param subgraph_labels: List of labels to be used for the subgraph collections.
        @param subgraph_schema: List of property schemas for the subgraph collections.
        @param as_url: Treat file_name as URL. Defaults to False.
        @param batch_size: Number of hyperedges to import at a time. Defaults to 1000.
        @param delimiter: Delimiter used in the CSV file. Defaults to ','.
        """
        pass

    @abstractmethod
    def add_node(
        self,
        session: Session,
        tx: Transaction,
        labels: List[Label],
        properties: List[Property],
    ) -> None:
        """
        Add a node to the database.

        @param session: Database session.
        @param tx: Current transaction.
        @param labels: List of labels for the node to be added.
        @param properties: List of properties for the node to be added.
        """
        pass

    @abstractmethod
    def delete_node(
        self,
        session: Session,
        tx: Transaction,
        labels: List[Label],
        properties: List[Property],
    ) -> None:
        """
        Delete a node and all its connected edges.

        @param tx: Current transaction.
        @param session: Database session.
        @param labels: List of labels for the node to be deleted.
        @param properties: List of properties for the node to be deleted.
        """
        pass

    @abstractmethod
    def delete_node_with_node_edges(
        self,
        session: Session,
        tx: Transaction,
        labels: List[Label],
        properties: List[Property],
        edge_label: Label,
    ) -> None:
        """
        Delete a node and all its connecting HO edges.

        @param session: Database session.
        @param tx: Current transaction.
        @param labels: List of labels for the node to be deleted.
        @param properties: List of properties for the node to be deleted.
        @param edge_label: Label of the edges.
        """
        pass

    @abstractmethod
    def add_edge(
        self,
        session: Session,
        tx: Transaction,
        start_node_labels: List[Label],
        start_node_properties: List[Property],
        end_node_labels: List[Label],
        end_node_properties: List[Property],
        edge_label: Label,
        edge_properties: List[Property],
    ) -> None:
        """
        Add an edge between two nodes.

        @param session: Database session.
        @param tx: Current transaction.
        @param start_node_labels: List of labels for the start node.
        @param start_node_properties: List of properties for the start node.
        @param end_node_labels: List of labels for the end node.
        @param end_node_properties: List of properties for the end node.
        @param edge_label: Label of the edge.
        @param edge_properties: List of properties of the edge.
        """
        pass

    @abstractmethod
    def delete_edge(
        self,
        session: Session,
        tx: Transaction,
        start_node_labels: List[Label],
        start_node_properties: List[Property],
        end_node_labels: List[Label],
        end_node_properties: List[Property],
        edge_label: Label,
    ) -> None:
        """
        Delete an edge between two nodes.

        @param session: Database session.
        @param tx: Current transaction.
        @param start_node_labels: List of labels for the start node.
        @param start_node_properties: List of properties for the start node.
        @param end_node_labels: List of labels for the end node.
        @param end_node_properties: List of properties for the end node.
        @param edge_label: Label of the edge.
        """
        pass

    @abstractmethod
    def update_node(
        self,
        session: Session,
        tx: Transaction,
        node_labels: List[Label],
        node_properties: List[Property],
        update_properties: List[Property],
    ) -> None:
        """
        Update properties of a node.

        @param session: Database session.
        @param tx: Current transaction.
        @param node_labels: List of labels of the node.
        @param node_properties: List of original properties of the node.
        @param update_properties: List of new properties for the node.
        """
        pass

    @abstractmethod
    def update_edge(
        self,
        session: Session,
        tx: Transaction,
        edge_label: Label,
        edge_properties: List[Property],
        update_properties: List[Property],
    ) -> None:
        """
        Update properties of an edge.

        @param session: Database session.
        @param tx: Current transaction.
        @param edge_label: Label of the edge.
        @param edge_properties: List of original properties of the edge.
        @param update_properties: List of new properties for the edge.
        """
        pass

    @abstractmethod
    def clear_data(self, session: Session) -> None:
        """
        Remove all data from the database.

        @param session: Database session.
        """
        pass

    @abstractmethod
    def match_nodes(
        self,
        session: Session,
        node_labels: List[Label],
        node_properties: List[Property],
    ) -> pd.DataFrame:
        """
        Match nodes by labels and properties.

        @param session: Database session.
        @param node_labels: List of labels of the nodes.
        @param node_properties: List of properties of the nodes.
        @return: Dataframe containing the matched nodes.
        """
        pass

    @abstractmethod
    def match_edges(
        self,
        session: Session,
        start_node_labels: List[Label],
        start_node_properties: List[Property],
        end_node_labels: List[Label],
        end_node_properties: List[Property],
        edge_label: Label,
        edge_properties: List[Property],
    ) -> pd.DataFrame:
        """
        Match edges by label and properties.

        @param session: Database session.
        @param start_node_labels: List of labels for the start nodes.
        @param start_node_properties: List of properties for the start nodes.
        @param end_node_labels: List of labels for the end nodes.
        @param end_node_properties: List of properties for the end nodes.
        @param edge_label: Label of the edges.
        @param edge_properties: List of properties of the edges.
        @return: Dataframe containing the matched edges with their start and end nodes.
        """
        pass

    @abstractmethod
    def match_node_edges(
        self,
        session: Session,
        start_node_labels: List[Label],
        start_node_properties: List[Property],
        end_node_labels: List[Label],
        end_node_properties: List[Property],
        node_edge_labels: List[Label],
        node_edge_properties: List[Property],
        edge_label: Label,
    ) -> None:
        """
        Match HO edges by label and properties.

        @param session: Database session.
        @param start_node_labels: List of labels for the start nodes.
        @param start_node_properties: List of properties for the start nodes.
        @param end_node_labels: List of labels for the end nodes.
        @param end_node_properties: List of properties for the end nodes.
        @param node_edge_labels: List of label of the HO edges.
        @param node_edge_properties: List of properties of the HO edges.
        @param edge_label: Label of the edges.
        @return: Dataframe containing the matched HO edges with their start and end nodes.
        """
        pass

    @abstractmethod
    def match_subgraph(
        self,
        session: Session,
        subgraph_labels: List[Label],
        subgraph_properties: List[Property],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Match a subgraph by labels and properties.

        @param session: Database session.
        @param subgraph_labels: List of labels of the subgraph.
        @param subgraph_properties: List of properties of the subgraph.
        @return: Triple of dataframes containing subgraph, node and edge information respectively.
        """
        pass

    @abstractmethod
    def match_subgraph_edges(
        self,
        session: Session,
        start_subgraph_labels: List[Label],
        start_subgraph_properties: List[Property],
        end_subgraph_labels: List[Label],
        end_subgraph_properties: List[Property],
        edge_label: Label,
        edge_properties: List[Property],
    ) -> pd.DataFrame:
        """
        Match subgraph edges by label and properties.

        @param session: Database session.
        @param start_subgraph_labels: List of labels for the start subgraphs.
        @param start_subgraph_properties: List of properties for the start subgraphs.
        @param end_subgraph_labels: List of labels for the end subgraphs.
        @param end_subgraph_properties: List of properties for the end subgraphs.
        @param edge_label: Label of the subgraph edges.
        @param edge_properties: List of properties of the subgraph edges.
        @return: Dataframe containing the matched subgraph edges with their start and end subgraphs.
        """
        pass

    @abstractmethod
    def match_hyperedge(
        self,
        session: Session,
        node_labels: List[Label],
        hyperedge_labels: List[Label],
        hyperedge_properties: List[Property],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Match a hyperedge by labels and properties.

        @param session: Database session.
        @param node_labels: List of labels for the nodes.
        @param hyperedge_labels: List of labels of the hyperedge.
        @param hyperedge_properties: List of properties of the hyperedge.
        @return: Tuple of dataframes containing the matched node information as well as the related
                 edge information.
        """
        pass

    @abstractmethod
    def match_node_tuple(
        self,
        session: Session,
        tuple_labels: List[Label],
        tuple_properties: List[Property],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Match a node-tuple by labels and properties.

        @param session: Database session.
        @param tuple_labels: List of labels of the node-tuple.
        @param tuple_properties: List of properties of the node-tuple.
        @return: Tuple of dataframes containing the matched node-tuple information as well as the
                 related node information.
        """
        pass

    @abstractmethod
    def create_index(
        self, session: Session, label: Label, properties: List[str]
    ) -> None:
        """
        Create an index on property keys and a label.

        @param session: Database session.
        @param label: Label used for indexing.
        @param properties: List of property keys to be indexed.
        """
        pass

    @abstractmethod
    def drop_index(self, session: Session, index_name: str) -> None:
        """
        Drop an index by name.

        @param session: Database session.
        @param index_name: Name of the index to be deleted.
        """
        pass

    @abstractmethod
    def show_index_names(self, session: Session) -> List[str]:
        """
        Show all index names in the database.

        @param session: Database session.
        @return: List of the names of indices in the database.
        """
        pass

    @abstractmethod
    def traverse_path(
        self,
        session: Session,
        variables_list: List[List[str]],
        elements_list: List[List[Tuple[List[Label], List[Property]]]],
        conditions_list: List[List[str]],
        return_values: List[str],
        sort: List[str] = None,
        limit: int = None,
    ) -> pd.DataFrame:
        """
        Traverse a path in the database.

        @param session: Database session.
        @param variables_list: TODO.
        @param elements_list: TODO.
        @param conditions_list: TODO.
        @param return_values: TODO.
        @param sort: TODO. Defaults to None.
        @param limit: TODO. Defaults to None.
        @return: Path information.
        """
        pass

    @abstractmethod
    def node_count(self, session: Session, node_labels: List[Label] = None) -> int:
        """
        Get number of nodes in the database that have the given list of labels.

        @param session: Database session.
        @param node_labels: List of node labels. Defaults to None.
        @return: Number of nodes with the given labels.
        """
        pass

    @abstractmethod
    def edge_count(self, session: Session, edge_label: Label = None) -> int:
        """
        Get number of edges in the database that have the given label.

        @param session: Database session.
        @param edge_label: Edge label. Defaults to None.
        @return: Number of edges with the given label.
        """
        pass

    @abstractmethod
    def start_session(self) -> Session:
        """
        Start a new database session.

        @return: Database session.
        """
        pass

    @abstractmethod
    def end_session(self, session) -> None:
        """
        Close the database session.

        @param session: Database session.
        """
        pass
