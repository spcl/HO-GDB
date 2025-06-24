# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main authors: Jakub Cudak
#               Shriram Chandran

from neo4j import (
    GraphDatabase,
    Session as Neo4jSession,
    Transaction as Neo4jTransaction,
)
from dotenv import load_dotenv
from HOGDB.db.db import Database
from HOGDB.db.label import Label
from HOGDB.db.property import Property
from HOGDB.db.schema import Schema
from HOGDB.proxy.proxy import ProxyDriver
from typing import Dict, List, Optional, Tuple
import os
import pandas as pd, csv

# Load environment variables from the .env file
load_dotenv("HOGDB/.env")


class Neo4jDatabase(Database):
    """
    The Neo4jDatabase class handles interactions with the Neo4j graph database using the provided configuration.

    Inherits from the Database class and implements its abstract methods.
    A concrete implementation of the Database class for Neo4j.
    """

    def __init__(
        self,
        db_name: str = None,
        db_uri: str = None,
        db_username: str = None,
        db_password: str = None,
        proxy_url: str = None,
        max_connection_lifetime: int = 300,
        max_connection_pool_size: int = 50,
        connection_timeout: int = 30,
    ) -> None:
        """
        Initialize the Neo4jDatabase instance. Takes into account the environmental variables if
        necessary.

        @param db_name: Name of the database. Defaults to None.
        @param db_uri: URI of the database. Defaults to None.
        @param db_username: Username of the database credentials. Defaults to None.
        @param db_password: Password of the database credentials. Defaults to None.
        @param proxy_url: URL to access the database. Defaults to None.
        @param max_connection_lifetime: Maximum lifetime in seconds for a given connection. Defaults to 300.
        @param max_connection_pool_size: TODO. Defaults to 50.
        @param connection_timeout: Connection timeout in seconds. Defaults to 30.
        """
        self._db_name = "neo4j" if db_name is None else db_name
        self._db_uri = os.getenv("DB_URI") if db_uri is None else db_uri
        self._db_username = (
            os.getenv("DB_USERNAME") if db_username is None else db_username
        )
        self._db_password = (
            os.getenv("DB_PASSWORD") if db_password is None else db_password
        )
        if proxy_url:
            self._driver = ProxyDriver(
                self._db_uri, self._db_username, self._db_password, proxy_url
            )
        else:
            self._driver = GraphDatabase.driver(
                self._db_uri,
                auth=(self._db_username, self._db_password),
                max_connection_lifetime=max_connection_lifetime,
                max_connection_pool_size=max_connection_pool_size,
                connection_timeout=connection_timeout,
            )
        self._driver.verify_connectivity()

    def close_driver(self) -> None:
        """
        Close the database driver.
        """
        self._driver.close()

    def _execute_query(
        self, session: Neo4jSession, query: str, parameters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Execute a non-transactional query in the given session and return the results.

        @param session: Database session.
        @param query: Query to run.
        @param parameters: Parameters for the query.
        @return: Results of the query.
        """
        return [record for record in session.run(query, parameters or {})]

    def _begin_transaction(self, session: Neo4jSession) -> Neo4jTransaction:
        """
        Begin a transaction in the given session.

        @param session: Database session.
        @return: Transaction.
        """
        return session.begin_transaction()

    def _close_transaction(self, tx: Neo4jTransaction) -> None:
        """
        Commit and close the transaction.

        @param tx: Transaction to commit and close.
        """
        tx.commit()
        tx.close()

    def _execute_in_transaction(
        self,
        session: Neo4jSession,
        tx: Neo4jTransaction,
        query: str,
        parameters: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Execute a query in the given transaction and return the results.

        @param session: Database session.
        @param tx: Current transaction.
        @param query: Query to run.
        @param parameters: Parameters for the query.
        @return: Results of the query.
        """
        return self._execute_query(tx, query, parameters)

    def _generate_query_strings(
        self, alias: str, schema: List[Schema]
    ) -> Tuple[str, List[str]]:
        """
        Utility method to generate query strings for properties and field names.

        @param alias: TODO.
        @param schema: List of property schemas to be used.
        @return: Tuple of a string, consisting of comma-separated properties, and a list of their
                 respective fieldnames.
        """
        properties_str = ", ".join(
            [f"{alias}.{s._property_to_field()}" for s in schema]
        )
        fields = [s.field_name for s in schema]
        return properties_str, fields

    def _write_to_csv(
        self, file_name: str, records: List[Dict], fields: List[str]
    ) -> None:
        """
        Utility method to write records to a CSV file.

        @param file_name: Name and path of the output file.
        @param records: Records to be written.
        @param fields: Column titles for the records.
        """
        df = pd.DataFrame(records, columns=fields)
        df.to_csv(file_name, index=False, header=True)

    def export_nodes_to_csv(
        self,
        session: Neo4jSession,
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
        labels_str = self.format_labels(labels)
        properties_str, fields = self._generate_query_strings("n", node_schema)
        query = f"""
        MATCH (n{labels_str})
        RETURN {properties_str}
        """
        self._write_to_csv(file_name, self._execute_query(session, query), fields)

    def export_edges_to_csv(
        self,
        session: Neo4jSession,
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
        start_str, start_fields = self._generate_query_strings("s", start_schema)
        end_str, end_fields = self._generate_query_strings("e", end_schema)
        edge_str, edge_fields = self._generate_query_strings("r", edge_schema)
        query = f"""
        MATCH (s{self.format_labels(start_labels)})-[r:{edge_label}]->(e{self.format_labels(end_labels)})
        RETURN {start_str}, {end_str}, {edge_str}
        """
        self._write_to_csv(
            file_name,
            self._execute_query(session, query),
            start_fields + end_fields + edge_fields,
        )

    def export_node_edges_to_csv(
        self,
        session: Neo4jSession,
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
        start_properties_str, start_fields = self._generate_query_strings(
            "s", start_schema
        )
        end_properties_str, end_fields = self._generate_query_strings("e", end_schema)
        edge_properties_str, edge_fields = self._generate_query_strings(
            "edge_node", node_edge_schema
        )
        query = f"""
        MATCH (edge_node{self.format_labels(node_edge_labels)})
        MATCH (s{self.format_labels(start_labels)})-[:{edge_label}]->(edge_node)-[:{edge_label}]->(e{self.format_labels(end_labels)})
        RETURN {start_properties_str}, {end_properties_str}, {edge_properties_str}
        """
        self._write_to_csv(
            file_name,
            self._execute_query(session, query),
            start_fields + end_fields + edge_fields,
        )

    def export_hyperedges_to_csv(
        self,
        session: Neo4jSession,
        file_name: str,
        node_labels: List[Label],
        node_schema: Schema,
        hyperedge_labels: List[Label],
        hyperedge_schema: List[Schema],
    ) -> None:
        """
        Export hyperedges to a CSV file, which are modeled as nodes in our Neo4j implementation.

        @param session: Database session.
        @param file_name: Name and path of the output file.
        @param node_labels: List of node labels.
        @param node_schema: Property schema for the nodes.
        @param hyperedge_labels: Export hyperedges with those labels.
        @param hyperedge_schema: List of property schemas for the hyperedges.
        """
        _, node_fields = self._generate_query_strings("e", [node_schema])
        hyperedge_properties_str, hyperedge_fields = self._generate_query_strings(
            "hyperedge_node", hyperedge_schema
        )
        hyperedge_labels_str = self.format_labels(hyperedge_labels)
        query = f"""
        MATCH (hyperedge_node{hyperedge_labels_str})
        WITH hyperedge_node, [(n{self.format_labels(node_labels)})-[r:_adjacency]->(hyperedge_node) | n.{node_schema.property_name}] AS node_list
        WITH hyperedge_node, REDUCE(s = "", name IN node_list | 
                CASE 
                    WHEN s = "" THEN name 
                    ELSE s + ";" + name 
                END) AS {node_schema.field_name},{hyperedge_properties_str}
        RETURN {node_schema.field_name}, {hyperedge_properties_str}
        """
        self._write_to_csv(
            file_name,
            self._execute_query(session, query),
            node_fields + hyperedge_fields,
        )

    def export_subgraphs_to_csv(
        self,
        session: Neo4jSession,
        file_name: str,
        node_schema: Schema,
        edge_schema: Schema,
        subgraph_labels: List[Label],
        subgraph_schema: List[Schema],
    ) -> None:
        """
        Export subgraph collections to a CSV file, which are modeled as nodes in our Neo4j
        implementation.

        @param session: Database session.
        @param file_name: Name and path of the output file.
        @param node_schema: Node schema.
        @param edge_schema: Edge schema.
        @param subgraph_labels: List of subgraph collection labels to export.
        @param subgraph_schema: List of property schemas for the subgraph collections.
        """
        subgraph_labels_str = self.format_labels(subgraph_labels)
        subgraph_properties_str, subgraph_fields = self._generate_query_strings(
            "subgraph_node", subgraph_schema
        )
        query = f"""
        MATCH (subgraph_node{subgraph_labels_str})
        WITH subgraph_node, [(n:_node)-[:_node_membership]->(subgraph_node) | n.{node_schema.property_name}] AS node_list, [(s:_node)-[:_adjacency]->(edge:_edge)-[:_adjacency]->(e:_node) WHERE (edge)-[:_edge_membership]->(subgraph_node) | [s.{edge_schema.property_name},e.{edge_schema.property_name}]] AS edge_list
        RETURN node_list AS {node_schema.field_name}, edge_list AS {edge_schema.field_name}, {subgraph_properties_str}
        """
        records = self._execute_query(session, query)
        nodes, edges = node_schema.field_name, edge_schema.field_name
        df = pd.DataFrame(
            records,
            columns=[nodes, edges] + subgraph_fields,
        )
        df[nodes] = df[nodes].apply(lambda node: ";".join(map(str, node)))
        df[edges] = df[edges].apply(
            lambda edge: ";".join(map(lambda node: ":".join(map(str, node)), edge))
        )
        df.to_csv(file_name, index=False, header=True, quoting=csv.QUOTE_NONE)

    def export_node_tuples_to_csv(
        self,
        session: Neo4jSession,
        file_name: str,
        node_schema: Schema,
        tuple_labels: List[Label],
        tuple_schema: List[Schema],
    ) -> None:
        """
        Export node-tuples to a CSV file, which are modeled as nodes in our Neo4j implementation.

        @param session: Database session.
        @param file_name: Name and path of the output file.
        @param node_schema: Property schema for the nodes.
        @param tuple_labels: Export node-tuples with those labels.
        @param tuple_schema: List of property schemas for the node-tuples.
        """
        tuple_labels_str = self.format_labels(tuple_labels)
        tuple_properties_str, subgraph_fields = self._generate_query_strings(
            "tuple_node", tuple_schema
        )
        query = f"""
        MATCH (tuple_node:_node_tuple{tuple_labels_str})
        WITH tuple_node, [(n:_node)-[r:_node_membership]->(tuple_node) | {{prop: n.{node_schema.property_name}, pos: r.position_in_tuple}}] AS node_list
        RETURN node_list AS {node_schema.field_name}, {tuple_properties_str}
        """
        records = self._execute_query(session, query)
        nodes = node_schema.field_name
        df = pd.DataFrame(
            records,
            columns=[nodes] + subgraph_fields,
        )
        df[nodes] = df[nodes].apply(
            lambda node: ";".join(
                map(
                    lambda x: str(x["prop"]),
                    sorted(node, key=lambda x: x["pos"]),
                )
            )
        )
        df.to_csv(file_name, index=False, header=True, quoting=csv.QUOTE_NONE)

    def import_nodes_from_csv(
        self,
        session: Neo4jSession,
        file_name: str,
        labels: List[Label],
        node_schema: List[Schema],
        as_url: bool = False,
        batch_size: int = 10000,
        delimiter: str = ",",
    ) -> None:
        """
        Import nodes from a CSV file into Neo4j.

        @param session: Database session.
        @param file_name: Name and path of the input file.
        @param labels: List of labels for the nodes.
        @param node_schema: List of property schemas for the nodes.
        @param as_url: Treat file_name as URL. Defaults to False.
        @param batch_size: Number of nodes to import at a time. Defaults to 10000.
        @param delimiter: Delimiter used in the CSV file. Defaults to ','.
        """
        labels_str = self.format_labels(labels)
        properties_str = (
            "{" + ", ".join([s._field_to_property("row") for s in node_schema]) + "}"
        )
        file_path = file_name if as_url else f"file:///{file_name}"
        query = f"""
        LOAD CSV WITH HEADERS FROM '{file_path}'
        AS row
        FIELDTERMINATOR '{delimiter}'
        CALL(row) {{
          WITH row
          CREATE ({labels_str} {properties_str})
        }} IN TRANSACTIONS OF {batch_size} ROWS"""
        self._execute_query(session, query)

    def import_edges_from_csv(
        self,
        session: Neo4jSession,
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
    ):
        """
        Import edges from a CSV file into Neo4j.

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
        start_properties = (
            "{" + ", ".join([s._field_to_property("row") for s in start_schema]) + "}"
        )
        end_properties = (
            "{" + ", ".join([s._field_to_property("row") for s in end_schema]) + "}"
        )
        edge_properties = (
            "{" + ", ".join([s._field_to_property("row") for s in edge_schema]) + "}"
        )
        file_path = file_name if as_url else f"file:///{file_name}"
        query = f"""
        LOAD CSV WITH HEADERS FROM '{file_path}'
        AS row
        FIELDTERMINATOR '{delimiter}'
        CALL(row) {{
          WITH row
          MATCH (start{self.format_labels(start_labels)} {start_properties})
          MATCH (end{self.format_labels(end_labels)} {end_properties})
          CREATE (start)-[r:{edge_label} {edge_properties}]->(end)
        }} IN TRANSACTIONS OF {batch_size} ROWS
        """
        self._execute_query(session, query)

    def import_node_edges_from_csv(
        self,
        session: Neo4jSession,
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
        Import HO edges from a CSV file into Neo4j, which are modeled as nodes in our Neo4j
        implementation.

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
        start_properties_str = (
            "{" + ", ".join([s._field_to_property("row") for s in start_schema]) + "}"
        )
        end_properties_str = (
            "{" + ", ".join([s._field_to_property("row") for s in end_schema]) + "}"
        )
        edge_properties_str = (
            "{"
            + ", ".join([s._field_to_property("row") for s in node_edge_schema])
            + "}"
        )
        file_path = file_name if as_url else f"file:///{file_name}"
        query = f"""
        LOAD CSV WITH HEADERS FROM '{file_path}'
        AS row
        FIELDTERMINATOR '{delimiter}'
        CALL(row) {{
          WITH row
          CREATE (edge_node{self.format_labels(node_edge_labels)} {edge_properties_str})
          WITH edge_node, row
          MATCH (start{self.format_labels(start_labels)} {start_properties_str}), (end{self.format_labels(end_labels)} {end_properties_str})
          CREATE (start)-[:{edge_label}]->(edge_node)-[:{edge_label}]->(end)
        }} IN TRANSACTIONS OF {batch_size} ROWS
        """
        self._execute_query(session, query)

    def import_hyperedges_from_csv(
        self,
        session: Neo4jSession,
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
        Import hyperedges from a CSV file into Neo4j.

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
        hyperedge_labels_str = self.format_labels(hyperedge_labels)
        hyperedge_schema_str = (
            "{"
            + ", ".join([s._field_to_property("row") for s in hyperedge_schema])
            + "}"
        )
        node_property_str = (
            "{" + node_schema.set_field("nodes[node_position]")._field_to_property()
        )
        common_schema_str = (
            (
                ", "
                + ", ".join([s._field_to_property("row") for s in common_schema])
                + "}"
            )
            if common_schema != []
            else "}"
        )
        file_path = file_name if as_url else f"file:///{file_name}"
        query = f"""
        LOAD CSV WITH HEADERS FROM '{file_path}'
       
        AS row
        FIELDTERMINATOR '{delimiter}'
        CALL(row) {{
          WITH row
          CREATE (hyperedge_node{hyperedge_labels_str}{hyperedge_schema_str})
          WITH row, hyperedge_node, split(row.{node_schema.field_name}, ';') as nodes
        UNWIND RANGE(0, SIZE(nodes) - 1) AS node_position
        MATCH (n{self.format_labels(node_labels)} {node_property_str}{common_schema_str})
          CREATE (n)-[:_adjacency]->(hyperedge_node)
          CREATE (hyperedge_node)-[:_adjacency]->(n)
        }} IN TRANSACTIONS OF {batch_size} ROWS
        """
        self._execute_query(session, query)

    def import_subgraphs_from_csv(
        self,
        session: Neo4jSession,
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
        Import subgraph collections from a CSV file into Neo4j, which are modeled as nodes in our
        Neo4j implementation.

        @param session: Database session.
        @param file_name: Name and path of the input file.
        @param node_schema: Property schema for the nodes.
        @param edge_schema: Property schema for the edges.
        @param common_schema: List of property schemas common to the nodes.
        @param subgraph_labels: List of labels to be used for the subgraph collections.
        @param subgraph_schema: List of property schemas for the subgraph collections.
        @param as_url: Treat file_name as URL. Defaults to False.
        @param batch_size: Number of hyperedges to import at a time. Defaults to 1000.
        @param delimiter: Delimiter used in the CSV file. Defaults to ','.
        """
        subgraph_labels_str = self.format_labels(subgraph_labels)
        subgraph_schema_str = (
            "{"
            + ", ".join([s._field_to_property("row") for s in subgraph_schema])
            + "}"
        )
        node_property_str, edge_property_str1, edge_property_str2 = (
            "{" + node_schema.set_field("node_value")._field_to_property(),
            "{" + edge_schema.set_field("edge_nodes[0]")._field_to_property(),
            "{" + edge_schema.set_field("edge_nodes[1]")._field_to_property(),
        )
        common_schema_str = (
            (
                ", "
                + ", ".join([s._field_to_property("row") for s in common_schema])
                + "}"
            )
            if common_schema != []
            else "}"
        )
        file_path = file_name if as_url else f"file:///{file_name}"
        query = f"""
        LOAD CSV WITH HEADERS FROM '{file_path}'
        AS row
        FIELDTERMINATOR '{delimiter}'
        CALL(row) {{
          WITH row
          CREATE (subgraph_node{subgraph_labels_str} {subgraph_schema_str})
          WITH row, subgraph_node, split(row.{node_schema.field_name}, ';') as nodes
          UNWIND nodes AS node_value
          MATCH (n:_node {node_property_str}{common_schema_str})
          CREATE (n)-[:_node_membership]->(subgraph_node)
          WITH row, subgraph_node, split(row.{edge_schema.field_name}, ';') as edges
          UNWIND edges AS edge_value
          WITH row, subgraph_node, edges, split(edge_value, ':') AS edge_nodes
          MATCH (start:_node {edge_property_str1}{common_schema_str})-[:_adjacency]->(edge:_edge)-[:_adjacency]->(end:_node {edge_property_str2}{common_schema_str})
          WITH DISTINCT edge, subgraph_node
          CREATE (edge)-[:_edge_membership]->(subgraph_node)
        }} IN TRANSACTIONS OF {batch_size} ROWS
        """
        self._execute_query(session, query)

    def import_node_tuples_from_csv(
        self,
        session: Neo4jSession,
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
        Import node tuples from a CSV file into Neo4j.

        @param session: Database session.
        @param file_name: Name and path of the input file.
        @param node_schema: Property schema for the nodes.
        @param common_schema: List of property schemas common to the nodes.
        @param tuple_labels: List of labels to be used for the node-tuples.
        @param tuple_schema: List of property schemas for the node-tuples.
        @param as_url: Treat file_name as URL. Defaults to False.
        @param batch_size: Number of hyperedges to import at a time. Defaults to 1000.
        @param delimiter: Delimiter used in the CSV file. Defaults to ','.
        """
        tuple_labels_str = self.format_labels(tuple_labels)
        tuple_properties_str = (
            "{"
            + ", ".join([s._field_to_property("row") for s in tuple_properties])
            + "}"
        )
        node_property_str = (
            "{" + node_schema.set_field("nodes[node_position]")._field_to_property()
        )
        common_schema_str = (
            (
                ", "
                + ", ".join([s._field_to_property("row") for s in common_schema])
                + "}"
            )
            if common_schema != []
            else "}"
        )
        file_path = file_name if as_url else f"file:///{file_name}"
        query = f"""
        LOAD CSV WITH HEADERS FROM '{file_path}'
        AS row
        FIELDTERMINATOR '{delimiter}'
        CALL(row) {{
          WITH row
          CREATE (tuple_node:_node_tuple{tuple_labels_str} {tuple_properties_str})
          WITH row, tuple_node, split(row.{node_schema.field_name}, ';') as nodes
          UNWIND RANGE(0, SIZE(nodes) - 1) AS node_position
          MATCH (n:_node {node_property_str}{common_schema_str})
          CREATE (n)-[:_node_membership{{position_in_tuple: toInteger(node_position)}}]->(tuple_node)
        }} IN TRANSACTIONS OF {batch_size} ROWS
        """
        self._execute_query(session, query)

    def import_subgraph_edges_from_csv(
        self,
        session: Neo4jSession,
        file_name: str,
        start_subgraph_label: Label,
        start_subgraph_schema: List[Schema],
        end_subgraph_label: Label,
        end_subgraph_schema: List[Schema],
        edge_label: Label,
        edge_schema: List[Schema],
        as_url: bool = False,
        batch_size: int = 1000,
        delimiter: str = ",",
    ) -> None:
        """
        Import subgraph edges from a CSV file into Neo4j.

        @param session: Database session.
        @param file_name: Name and path of the input file.
        @param start_subgraph_label: Label of the start subgraphs.
        @param start_subgraph_schema: List of property schemas for the start subgraphs.
        @param end_subgraph_label: Label of the end subgraphs.
        @param end_subgraph_schema: List of property schemas for the end subgraphs.
        @param edge_label: Label of the subgraph edges.
        @param edge_schema: List of property schemas for the subgraph edges.
        @param as_url: Treat file_name as URL. Defaults to False.
        @param batch_size: Number of hyperedges to import at a time. Defaults to 1000.
        @param delimiter: Delimiter used in the CSV file. Defaults to ','.
        """
        start_properties_str = (
            "{"
            + ", ".join([s._field_to_property("row") for s in start_subgraph_schema])
            + "}"
        )
        end_properties_str = (
            "{"
            + ", ".join([s._field_to_property("row") for s in end_subgraph_schema])
            + "}"
        )
        edge_properties_str = (
            "{" + ", ".join([s._field_to_property("row") for s in edge_schema]) + "}"
        )
        file_path = file_name if as_url else f"file:///{file_name}"
        query = f"""
        LOAD CSV WITH HEADERS FROM '{file_path}'
        AS row
        FIELDTERMINATOR '{delimiter}'
        CALL(row) {{
          WITH row
          CREATE (edge_node:_subgraph_edge:{repr(edge_label)} {edge_properties_str})
          WITH edge_node, row
          MATCH (start:_subgraph:{repr(start_subgraph_label)} {start_properties_str}), (end:_subgraph:{repr(end_subgraph_label)} {end_properties_str})
          CREATE (start)-[:_subgraph_adjacency]->(edge_node)-[:_subgraph_adjacency]->(end)
        }} IN TRANSACTIONS OF {batch_size} ROWS
        """
        self._execute_query(session, query)

    def add_node(
        self,
        session: Neo4jSession,
        tx: Neo4jTransaction,
        labels: List[Label],
        properties: List[Property],
    ) -> None:
        """
        Add a node to the database within a transaction.

        @param session: Database session.
        @param tx: Current transaction.
        @param labels: List of labels for the node to be added.
        @param properties: List of properties for the node to be added.
        """
        labels_str = self.format_labels(labels)
        properties_str = self.format_properties(properties)
        query = f"""
        CREATE (n{labels_str} {properties_str})
        """
        self._execute_in_transaction(session, tx, query)

    def delete_node(
        self,
        session: Neo4jSession,
        tx: Neo4jTransaction,
        labels: List[Label],
        properties: List[Property],
    ) -> None:
        """
        Delete a node and all its connected edges within a transaction.

        @param session: Database session.
        @param tx: Current transaction.
        @param labels: List of labels for the node to be deleted.
        @param properties: List of properties for the node to be deleted.
        """
        labels_str = self.format_labels(labels)
        properties_str = self.format_properties(properties)
        query = f"""
        MATCH (n{labels_str} {properties_str})
        DETACH DELETE n
        """
        self._execute_in_transaction(session, tx, query)

    def delete_node_with_node_edges(
        self,
        session: Neo4jSession,
        tx: Neo4jTransaction,
        labels: List[Label],
        properties: List[Property],
        edge_label: Label,
    ) -> None:
        """
        Delete a node and all its connecting HO edges. A HO edge is modeled as a node on the LPG
        level and therefore has to be deleted explicitely.

        @param session: Database session.
        @param tx: Current transaction.
        @param labels: List of labels for the node to be deleted.
        @param properties: List of properties for the node to be deleted.
        @param edge_label: Label of the edges.
        """
        labels_str = self.format_labels(labels)
        properties_str = self.format_properties(properties)
        query = f"""
        MATCH (n{labels_str} {properties_str})
        WITH n OPTIONAL MATCH (n)-[:{repr(edge_label)}]->(edge)
        DETACH DELETE edge
        WITH n OPTIONAL MATCH (edge)-[:{repr(edge_label)}]->(n)
        DETACH DELETE edge
        DETACH DELETE n
        """
        self._execute_in_transaction(session, tx, query)

    def add_edge(
        self,
        session: Neo4jSession,
        tx: Neo4jTransaction,
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
        start_labels_str = self.format_labels(start_node_labels)
        end_labels_str = self.format_labels(end_node_labels)
        start_properties_str = self.format_properties(start_node_properties)
        end_properties_str = self.format_properties(end_node_properties)
        edge_label_str = self.format_labels([edge_label])
        edge_properties_str = self.format_properties(edge_properties)
        query = f"""
        MATCH (start{start_labels_str} {start_properties_str})
        MATCH (end{end_labels_str} {end_properties_str})
        CREATE (start)-[r{edge_label_str} {edge_properties_str}]->(end)
        """
        self._execute_in_transaction(session, tx, query)

    def delete_edge(
        self,
        session: Neo4jSession,
        tx: Neo4jTransaction,
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
        start_labels_str = self.format_labels(start_node_labels)
        end_labels_str = self.format_labels(end_node_labels)
        start_properties_str = self.format_properties(start_node_properties)
        end_properties_str = self.format_properties(end_node_properties)
        edge_label_str = self.format_labels([edge_label])
        query = f"""
        MATCH (start{start_labels_str} {start_properties_str})-[r{edge_label_str}]->(end{end_labels_str} {end_properties_str})
        DELETE r
        """
        self._execute_in_transaction(session, tx, query)

    def update_node(
        self,
        session: Neo4jSession,
        tx: Neo4jTransaction,
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
        node_labels_str = self.format_labels(node_labels)
        node_properties_str = self.format_properties(node_properties)
        update_properties_str = self.format_properties(update_properties)
        query = f"""
        MATCH (n{node_labels_str} {node_properties_str})
        SET n += {update_properties_str}
        """
        self._execute_in_transaction(session, tx, query)

    def update_edge(
        self,
        session: Neo4jSession,
        tx: Neo4jTransaction,
        edge_label: Label,
        edge_properties: List[Property],
        update_properties: List[Property],
    ) -> None:
        """
        Update properties of an edge.

        @param tx: Current transaction.
        @param session: Database session.
        @param edge_label: Label of the edge.
        @param edge_properties: List of original properties of the edge.
        @param update_properties: List of new properties for the edge.
        """
        edge_label_str = self.format_labels([edge_label])
        edge_properties_str = self.format_properties(edge_properties)
        update_properties_str = self.format_properties(update_properties)
        query = f"""
        MATCH ()-[e{edge_label_str} {edge_properties_str}]->()
        SET e += {update_properties_str}
        """
        self._execute_in_transaction(session, tx, query)

    def node_count(self, session: Neo4jSession, node_labels: List[Label] = None) -> int:
        """
        Get number of nodes in the database that have the given list of labels.

        @param session: Database session.
        @param node_labels: List of node labels. Defaults to None.
        @return: Number of nodes with the given labels.
        """
        node_label_str = self.format_labels(node_labels) if node_labels else ""
        query = f"""
        MATCH (node{node_label_str})
        RETURN count(node) as count
        """
        records = self._execute_query(session, query)
        if (
            type(records) == list
            and len(records) == 1
            and type(records[0]) == list
            and len(records[0]) == 1
            and type(records[0][0]) == int
        ):
            return records[0][0]
        return records[0]["count"]

    def edge_count(self, session: Neo4jSession, edge_label: Label = None) -> int:
        """
        Get number of edges in the database that have the given label.

        @param session: Database session.
        @param edge_label: Edge label. Defaults to None.
        @return: Number of edges with the given label.
        """
        edge_label_str = self.format_labels([edge_label]) if edge_label else ""
        query = f"""
        MATCH ()-[edge{edge_label_str}]->()
        RETURN count(edge) as count
        """
        records = self._execute_query(session, query)
        if (
            type(records) == list
            and len(records) == 1
            and type(records[0]) == list
            and len(records[0]) == 1
            and type(records[0][0]) == int
        ):
            return records[0][0]
        return records[0]["count"]

    def match_nodes(
        self,
        session: Neo4jSession,
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
        node_label_str = self.format_labels(node_labels)
        node_properties_str = " " + self.format_properties(node_properties)
        query = f"""
        MATCH (node{node_label_str}{node_properties_str})
        RETURN labels(node), properties(node)
        """
        records = self._execute_query(session, query)
        df = pd.DataFrame(records, columns=["labels", "properties"])
        return df

    def match_edges(
        self,
        session: Neo4jSession,
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
        start_node_label_str = self.format_labels(start_node_labels)
        start_node_properties_str = " " + self.format_properties(start_node_properties)
        end_node_label_str = self.format_labels(end_node_labels)
        end_node_properties_str = " " + self.format_properties(end_node_properties)
        edge_label_str = self.format_labels([edge_label])
        edge_properties_str = " " + self.format_properties(edge_properties)
        query = f"""
        MATCH (start_node{start_node_label_str}{start_node_properties_str})-[edge{edge_label_str}{edge_properties_str}]->(end_node{end_node_label_str}{end_node_properties_str})
        RETURN labels(start_node), properties(start_node), labels(end_node), properties(end_node), type(edge), properties(edge)
        """
        records = self._execute_query(session, query)
        df = pd.DataFrame(
            records,
            columns=[
                "start_node_labels",
                "start_node_properties",
                "end_node_labels",
                "end_node_properties",
                "edge_type",
                "edge_properties",
            ],
        )
        return df

    def match_node_edges(
        self,
        session: Neo4jSession,
        start_node_labels: List[Label],
        start_node_properties: List[Property],
        end_node_labels: List[Label],
        end_node_properties: List[Property],
        node_edge_labels: List[Label],
        node_edge_properties: List[Property],
        edge_label: Label,
    ) -> pd.DataFrame:
        """
        Match HO edges by label and properties.

        @param session: Database session.
        @param tx: Current transaction.
        @param start_node_labels: List of labels for the start nodes.
        @param start_node_properties: List of properties for the start nodes.
        @param end_node_labels: List of labels for the end nodes.
        @param end_node_properties: List of properties for the end nodes.
        @param node_edge_labels: List of label of the HO edges.
        @param node_edge_properties: List of properties of the HO edges.
        @param edge_label: Label of the edges.
        @return: Dataframe containing the matched HO edges with their start and end nodes.
        """
        start_node_label_str = self.format_labels(start_node_labels)
        start_node_properties_str = " " + self.format_properties(start_node_properties)
        end_node_label_str = self.format_labels(end_node_labels)
        end_node_properties_str = " " + self.format_properties(end_node_properties)
        edge_label_str = self.format_labels(node_edge_labels)
        edge_properties_str = " " + self.format_properties(node_edge_properties)
        query = f"""
        MATCH (start_node{start_node_label_str}{start_node_properties_str})-[:{edge_label}]->(edge{edge_label_str}{edge_properties_str})-[:{edge_label}]->(end_node{end_node_label_str}{end_node_properties_str})
        RETURN labels(start_node), properties(start_node), labels(end_node), properties(end_node), labels(edge), properties(edge)
        """
        records = self._execute_query(session, query)
        df = pd.DataFrame(
            records,
            columns=[
                "start_labels",
                "start_properties",
                "end_labels",
                "end_properties",
                "edge_labels",
                "edge_properties",
            ],
        )
        return df

    def match_subgraph(
        self,
        session: Neo4jSession,
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
        subgraph_labels_str = self.format_labels(subgraph_labels)
        subgraph_properties_str = " " + self.format_properties(subgraph_properties)
        query = f"""
        MATCH (subgraph{subgraph_labels_str}{subgraph_properties_str})
        RETURN labels(subgraph), properties(subgraph)
        """
        subgraph_records = self._execute_query(session, query)
        assert len(subgraph_records) <= 1
        query = f"""
        MATCH (node:_node)-[:_node_membership]->(subgraph{subgraph_labels_str}{subgraph_properties_str})
        RETURN labels(node), properties(node)
        """
        node_records = self._execute_query(session, query)
        query = f"""
        MATCH (edge:_edge)-[:_edge_membership]->(subgraph{subgraph_labels_str}{subgraph_properties_str})
        MATCH (start:_node)-[:_adjacency]->(edge)-[:_adjacency]->(end:_node)
        RETURN labels(start), properties(start), labels(end), properties(end), labels(edge), properties(edge)
        """
        edge_records = self._execute_query(session, query)
        subgraph_df = pd.DataFrame(subgraph_records, columns=["labels", "properties"])
        node_df = pd.DataFrame(node_records, columns=["labels", "properties"])
        edge_df = pd.DataFrame(
            edge_records,
            columns=[
                "start_labels",
                "start_properties",
                "end_labels",
                "end_properties",
                "edge_labels",
                "edge_properties",
            ],
        )
        return (subgraph_df, node_df, edge_df)

    def match_subgraph_edges(
        self,
        session: Neo4jSession,
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
        start_label_str = self.format_labels(start_subgraph_labels)
        start_properties_str = " " + self.format_properties(start_subgraph_properties)
        end_label_str = self.format_labels(end_subgraph_labels)
        end_properties_str = " " + self.format_properties(end_subgraph_properties)
        edge_label_str = self.format_labels([Label("_subgraph_edge"), edge_label])
        edge_properties_str = " " + self.format_properties(edge_properties)
        query = f"""
        MATCH (start_node{start_label_str}{start_properties_str})-[:_subgraph_adjacency]->(edge{edge_label_str}{edge_properties_str})-[:_subgraph_adjacency]->(end_node{end_label_str}{end_properties_str})
        RETURN labels(start_node), properties(start_node), labels(end_node), properties(end_node), labels(edge), properties(edge)
        """
        records = self._execute_query(session, query)
        df = pd.DataFrame(
            records,
            columns=[
                "start_labels",
                "start_properties",
                "end_labels",
                "end_properties",
                "edge_labels",
                "edge_properties",
            ],
        )
        return df

    def match_hyperedge(
        self,
        session: Neo4jSession,
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
        node_labels_str = self.format_labels(node_labels)
        edge_labels_str = self.format_labels(hyperedge_labels)
        properties_str = self.format_properties(hyperedge_properties)
        query = f"""
        MATCH (edge{edge_labels_str}{properties_str})
        RETURN labels(edge), properties(edge)
        """
        edge_records = self._execute_query(session, query)
        assert len(edge_records) <= 1
        query = f"""
        MATCH (node{node_labels_str})-[:_adjacency]->(edge{edge_labels_str}{properties_str})
        RETURN labels(node), properties(node)
        """
        node_records = self._execute_query(session, query)
        node_df = pd.DataFrame(node_records, columns=["labels", "properties"])
        edge_df = pd.DataFrame(edge_records, columns=["labels", "properties"])
        return (node_df, edge_df)

    def match_node_tuple(
        self,
        session: Neo4jSession,
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
        tuple_labels_str = self.format_labels(tuple_labels)
        tuple_properties_str = " " + self.format_properties(tuple_properties)
        query = f"""
        MATCH (tuple{tuple_labels_str}{tuple_properties_str})
        RETURN labels(tuple), properties(tuple)
        """
        tuple_records = self._execute_query(session, query)
        assert len(tuple_records) <= 1
        query = f"""
        MATCH (node:_node)-[r:_node_membership]->(tuple{tuple_labels_str}{tuple_properties_str})
        RETURN labels(node), properties(node), r.position_in_tuple
        """
        node_records = self._execute_query(session, query)
        tuple_df = pd.DataFrame(tuple_records, columns=["labels", "properties"])
        node_df = pd.DataFrame(
            node_records, columns=["labels", "properties", "position"]
        )
        return (tuple_df, node_df)

    def create_index(
        self, session: Neo4jSession, label: Label, properties: List[str]
    ) -> None:
        """
        Create an index on property keys and a label.

        @param session: Database session.
        @param label: Label used for indexing.
        @param properties: List of property keys to be indexed.
        """
        properties_str = "_".join(properties)
        property_list = ",".join([f"n.{p}" for p in properties])
        query = f"""
        CREATE INDEX {label}_{properties_str}_index FOR (n:{label}) ON ({property_list})
        """
        self._execute_query(session, query)

    def drop_index(self, session: Neo4jSession, index_name: str) -> None:
        """
        Drop an index by name.

        @param session: Database session.
        @param index_name: Name of the index to be deleted.
        """
        query = f"""
        DROP INDEX {index_name} IF EXISTS
        """
        self._execute_query(session, query)

    def show_indexes(self, session: Neo4jSession) -> List[Tuple[str, List[str]]]:
        """
        Show all indexes in the database.

        @param session: Database session.
        @return: List of labels and property keys.
        """
        query = f"""
        SHOW INDEXES
        YIELD labelsOrTypes, properties
        RETURN labelsOrTypes[0] AS label, properties AS properties
        """
        records = self._execute_query(session, query)
        return [(record["label"], record["properties"]) for record in records]

    def show_index_names(self, session: Neo4jSession) -> List[str]:
        """
        Show all index names in the database.

        @param session: Database session.
        @return: List of the names of indices in the database.
        """
        query = f"""
        SHOW INDEXES
        YIELD name
        RETURN name
        """
        records = self._execute_query(session, query)
        return [
            record[0] if type(record[0]) == str else record["name"]
            for record in records
        ]

    def traverse_path(
        self,
        session: Neo4jSession,
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
        element_strs = [
            [
                f"{var}{self.format_labels(labels)} {self.format_properties(properties)}"
                for _, (variable, (labels, properties)) in enumerate(
                    zip(variables, elements)
                )
                if (var := (" " if variable is None else variable))
            ]
            for _, (variables, elements) in enumerate(
                zip(variables_list, elements_list)
            )
        ]
        patterns = [
            f"({element_str[0]})"
            + "".join(
                f"-[{element_str[i]}]->({element_str[i + 1]})"
                for i in range(1, len(element_str) - 1, 2)
            )
            for element_str in element_strs
        ]
        conditions = [
            f"WHERE {' AND '.join(conditions)}" if conditions else ""
            for conditions in conditions_list
        ]
        pattern = "".join(
            [
                f"""MATCH {pattern}
                {condition}
                """
                for pattern, condition in zip(patterns, conditions)
            ]
        )
        limit_str = f"LIMIT {limit}" if limit else ""
        sort_str = f"ORDER BY {', '.join(sort)}" if sort else ""
        return_str = f"{', '.join(return_values)}" if return_values else "*"
        query = f"""
        {pattern}
        RETURN {return_str}
        {sort_str}
        {limit_str}
        """
        records = self._execute_query(session, query)
        df = pd.DataFrame(records, columns=return_values)
        return df

    def clear_data(self, session: Neo4jSession) -> None:
        """
        Remove all data from the database.

        @param session: Database session.
        """
        query = """
        MATCH (n)
        DETACH DELETE n
        """
        self._execute_query(session, query)

    def start_session(self) -> Neo4jSession:
        """
        Start a new database session.

        @return: Database session.
        """
        return self._driver.session(database=self._db_name)

    def end_session(self, session) -> None:
        """
        Close the database session.

        @param session: Database session.
        """
        session.close()
