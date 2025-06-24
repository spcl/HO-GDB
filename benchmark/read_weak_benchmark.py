# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Shriram Chandran

from HOGDB.graph.graph_with_subgraph_storage import GraphwithSubgraphStorage
from HOGDB.db.db import Database
from HOGDB.db.neo4j import Neo4jDatabase
from HOGDB.db.label import Label
from HOGDB.db.schema import Schema
from HOGDB.db.property import Property
from HOGDB.graph.node import Node
from HOGDB.graph.edge import Edge
from HOGDB.graph.subgraph import Subgraph, SubgraphEdge
from HOGDB.graph.path import *
from typing import List, Tuple

import argparse
import time
import random
import pandas as pd
import multiprocessing


parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", type=str, default="debug", help="Mode of operation: 'debug' or 'full'."
)
parser.add_argument(
    "--results_to_csv", action="store_true", help="Save results to CSV."
)
parser.add_argument(
    "--import_dir",
    type=str,
    default=".",
    help="Directory containing CSV files for import.",
)
parser.add_argument(
    "--online", action="store_true", help="Use online CSV URLs for data import."
)
parser.add_argument(
    "--csv_url", type=str, default="", help="Base URL for online CSV directory."
)
args = parser.parse_args()

db_params = {
    "db_uri": "bolt://localhost:7687",
    "db_username": "neo4j",
    "db_password": "password",
}

nodes, edges, subgraphs, subgraph_edges = {}, {}, {}, {}


def read_subgraph_csvs(N: int) -> None:
    """
    Read graph data from CSV files and store it in global variables.

    @param N: Number of graphs.
    """
    global nodes, edges, subgraphs, subgraph_edges
    nodes, edges, subgraphs, subgraph_edges = {}, {}, {}, {}
    pref = f"{args.csv_url}/" if args.online else f"{args.import_dir}/"
    nodes_df = pd.read_csv(f"{pref}zinc_atoms_{N}.csv")
    nodes.update(
        {
            (row["atom_id"], row["graph_id"]): row["atom_features"]
            for _, row in nodes_df.iterrows()
        }
    )
    edges_df = pd.read_csv(f"{pref}zinc_bonds_{N}.csv")
    edges.update(
        {
            (row["start_atom"], row["end_atom"], row["graph_id"]): row["bond_features"]
            for _, row in edges_df.iterrows()
        }
    )
    subgraphs_df = pd.read_csv(f"{pref}zinc_cliques_{N}.csv")
    subgraphs.update(
        {
            (row["clique_id"], row["graph_id"]): row["clique_features"]
            for _, row in subgraphs_df.iterrows()
        }
    )
    subgraph_edges_df = pd.read_csv(f"{pref}zinc_tree_edges_{N}.csv")
    subgraph_edges.update(
        {
            (row["start_clique"], row["end_clique"], row["graph_id"]): row[
                "tree_edge_id"
            ]
            for _, row in subgraph_edges_df.iterrows()
        }
    )


def get_subgraphs(db: Database, N: int) -> None:
    """
    Read data of a graph with subgraph collections from CSV files and store it in a graph database
    as well as in global variables.

    @param db: Database where the higher-order graph is going to be stored.
    @param N: Number of graphs.
    """
    session = db.start_session()
    db.create_index(session, Label("Atom"), ["atom_id", "graph_id"])
    db.create_index(session, Label("Bond"), ["bond_id", "graph_id"])
    db.create_index(session, Label("Clique"), ["clique_id", "graph_id"])
    db.create_index(session, Label("TreeEdge"), ["tree_edge_id", "graph_id"])

    prefix = ""
    db.import_nodes_from_csv(
        session,
        f"{prefix}zinc_atoms_{N}.csv",
        [Label("_node"), Label("Atom")],
        [Schema("atom_id", int), Schema("graph_id", int), Schema("atom_features", int)],
        as_url=False,
    )
    db.import_node_edges_from_csv(
        session,
        f"{prefix}zinc_bonds_{N}.csv",
        [Label("_node"), Label("Atom")],
        [Schema("atom_id", int, "start_atom"), Schema("graph_id", int)],
        [Label("_node"), Label("Atom")],
        [Schema("atom_id", int, "end_atom"), Schema("graph_id", int)],
        [Label("_edge"), Label("Bond")],
        [
            Schema("bond_id", int),
            Schema("graph_id", int),
            Schema("bond_features", int),
        ],
        Label("_adjacency"),
        as_url=False,
    )
    db.import_nodes_from_csv(
        session,
        f"{prefix}zinc_cliques_{N}.csv",
        [Label("_subgraph"), Label("Clique")],
        [
            Schema("clique_id", int),
            Schema("graph_id", int),
            Schema("clique_features", int),
        ],
        as_url=False,
    )
    db.import_edges_from_csv(
        session,
        f"{prefix}zinc_clique_membership_{N}.csv",
        [Label("_node"), Label("Atom")],
        [Schema("atom_id", int), Schema("graph_id", int)],
        [Label("_subgraph"), Label("Clique")],
        [Schema("clique_id", int), Schema("graph_id", int)],
        Label("_node_membership"),
        [],
        as_url=False,
    )
    db.import_edges_from_csv(
        session,
        f"{prefix}zinc_clique_bond_membership_{N}.csv",
        [Label("_edge"), Label("Bond")],
        [Schema("bond_id", int), Schema("graph_id", int)],
        [Label("_subgraph"), Label("Clique")],
        [Schema("clique_id", int), Schema("graph_id", int)],
        Label("_edge_membership"),
        [],
        as_url=False,
    )
    db.import_node_edges_from_csv(
        session,
        f"{prefix}zinc_tree_edges_{N}.csv",
        [Label("_subgraph"), Label("Clique")],
        [Schema("clique_id", int, "start_clique"), Schema("graph_id", int)],
        [Label("_subgraph"), Label("Clique")],
        [Schema("clique_id", int, "end_clique"), Schema("graph_id", int)],
        [Label("_subgraph_edge"), Label("TreeEdge")],
        [Schema("tree_edge_id", int), Schema("graph_id", int)],
        Label("_subgraph_adjacency"),
        as_url=False,
    )
    session.close()


def clear_graph() -> None:
    """
    Remove all graphs and their elements from the graph database.
    """
    db = Neo4jDatabase(**db_params)
    gs = GraphwithSubgraphStorage(db)
    gs.clear_graph()
    gs.close_connection()


def gen_node_queries_subgraph(N: int) -> List[Tuple[int, int]]:
    """
    Generate a number of atom IDs and corresponding graph IDs to be used for querying the higher-
    order graph as "node queries".

    @param N: Number of pairs to generate.
    @return: List of atom IDs and graph IDs.
    """
    queries = []
    keys = list(nodes.keys())
    for i in range(N):
        atom_id, graph_id = random.choice(keys)
        queries.append((atom_id, graph_id))
    return queries


def gen_edge_queries_subgraph(N: int) -> List[Tuple[int, int, int]]:
    """
    Generate a number of edges (defined by their origin and target atom IDs) and corresponding graph
    IDs to be used for querying the higher-order graph as "edge queries".

    @param N: Number of triples to generate.
    @return: List of atom IDs (origin and target) and graph IDs.
    """
    queries = []
    keys = list(edges.keys())
    for i in range(N):
        start_atom, end_atom, graph_id = random.choice(keys)
        queries.append((start_atom, end_atom, graph_id))
    return queries


def gen_subgraph_queries_subgraph(N: int) -> List[Tuple[int, int]]:
    """
    Generate a number of subgraph collection IDs and corresponding graph IDs to be used for
    querying the higher-order graph as "subgraph queries".

    @param N: Number of pairs to generate.
    @return: List of subgraph collection IDs and graph IDs.
    """
    queries = []
    keys = list(subgraphs.keys())
    for i in range(N):
        clique_id, graph_id = random.choice(keys)
        queries.append((clique_id, graph_id))
    return queries


def gen_subgraph_edge_queries_subgraph(N: int) -> List[Tuple[int, int, int]]:
    """
    Generate a number of subgraph edges (defined by their origin and target subgraph collection IDs)
    and corresponding graph IDs to be used for querying the higher-order graph as "subgraph edge
    queries".

    @param N: Number of triples to generate.
    @return: List of subgraph collection IDs (origin and target) and graph IDs.
    """
    queries = []
    keys = list(subgraph_edges.keys())
    for i in range(N):
        start_clique, end_clique, graph_id = random.choice(keys)
        queries.append((start_clique, end_clique, graph_id))
    return queries


def run_read_node_query(queries: List[Tuple[int, int]]) -> None:
    """
    Run node retrieval queries on a graph with subgraph collections.

    @param queries: List of atom IDs and graph IDs.
    """
    db = Neo4jDatabase(**db_params)
    gs = GraphwithSubgraphStorage(db)
    for atom_id, graph_id in queries:
        node = gs.get_node(
            Node(
                [Label("Atom")],
                [
                    Property("atom_id", int, atom_id),
                    Property("graph_id", int, graph_id),
                ],
            )
        )
    gs.close_connection()


def run_read_edge_query(queries: List[Tuple[int, int, int]]) -> None:
    """
    Run edge retrieval queries on a graph with subgraph collections.

    @param queries: List of atom IDs (origin and target) and graph IDs.
    """
    db = Neo4jDatabase(**db_params)
    gs = GraphwithSubgraphStorage(db)
    for start_atom, end_atom, graph_id in queries:
        edge = gs.get_edge(
            Edge(
                Node(
                    [Label("Atom")],
                    [
                        Property("atom_id", int, start_atom),
                        Property("graph_id", int, graph_id),
                    ],
                ),
                Node(
                    [Label("Atom")],
                    [
                        Property("atom_id", int, end_atom),
                        Property("graph_id", int, graph_id),
                    ],
                ),
                Label("Bond"),
                [],
            )
        )
    gs.close_connection()


def run_read_subgraph_query(queries: List[Tuple[int, int]]) -> None:
    """
    Run subgraph collection retrieval queries on a graph with subgraph collections.

    @param queries: List of subgraph collection IDs and graph IDs.
    """
    db = Neo4jDatabase(**db_params)
    gs = GraphwithSubgraphStorage(db)
    for clique_id, graph_id in queries:
        sg = gs.get_subgraph(
            Subgraph(
                labels=[Label("Clique")],
                properties=[
                    Property("clique_id", int, clique_id),
                    Property("graph_id", int, graph_id),
                ],
            )
        )
    gs.close_connection()


def run_read_subgraph_edge_query(queries: List[Tuple[int, int, int]]) -> None:
    """
    Run subgraph edge retrieval queries on a graph with subgraph collections.

    @param queries: List of subgraph collection IDs (origin and target) and graph IDs.
    """
    db = Neo4jDatabase(**db_params)
    gs = GraphwithSubgraphStorage(db)
    for start_clique, end_clique, graph_id in queries:
        sg_edge = gs.get_subgraph_edge(
            SubgraphEdge(
                Subgraph(
                    labels=[Label("Clique")],
                    properties=[
                        Property("clique_id", int, start_clique),
                        Property("graph_id", int, graph_id),
                    ],
                ),
                Subgraph(
                    labels=[Label("Clique")],
                    properties=[
                        Property("clique_id", int, end_clique),
                        Property("graph_id", int, graph_id),
                    ],
                ),
                Label("TreeEdge"),
                [],
            )
        )
    gs.close_connection()


if __name__ == "__main__":
    results = {}
    num_queries = 64 if args.mode == "debug" else 4096
    num_processes_list = [2] if args.mode == "debug" else [8, 16, 32, 64, 128]

    for N in [20, 40, 80, 160, 320]:
        print(f"Processing dataset with N={N}...")
        db = Neo4jDatabase(**db_params)
        gs = GraphwithSubgraphStorage(db)
        gs.clear_graph()
        gs.close_connection()
        db = Neo4jDatabase(**db_params)
        get_subgraphs(db, N)
        read_subgraph_csvs(N)

        node_queries = gen_node_queries_subgraph(num_queries)
        edge_queries = gen_edge_queries_subgraph(num_queries)
        subgraph_queries = gen_subgraph_queries_subgraph(num_queries)
        subgraph_edge_queries = gen_subgraph_edge_queries_subgraph(num_queries)

        for num_processes in num_processes_list:
            print(f"Running with {num_processes} processes...")
            split = num_queries // num_processes

            # node reads
            processes = []
            total_time_start = time.perf_counter()
            for i in range(num_processes):
                process = multiprocessing.Process(
                    target=run_read_node_query,
                    args=(node_queries[split * i : split * (i + 1)],),
                )
                processes.append(process)
                process.start()
            for process in processes:
                process.join()
            results[f"node_time_N{N}_{num_processes}"] = (
                time.perf_counter() - total_time_start
            )

            # edge reads
            processes = []
            total_time_start = time.perf_counter()
            for i in range(num_processes):
                process = multiprocessing.Process(
                    target=run_read_edge_query,
                    args=(edge_queries[split * i : split * (i + 1)],),
                )
                processes.append(process)
                process.start()
            for process in processes:
                process.join()
            results[f"edge_time_N{N}_{num_processes}"] = (
                time.perf_counter() - total_time_start
            )

            # subgraph reads
            processes = []
            total_time_start = time.perf_counter()
            for i in range(num_processes):
                process = multiprocessing.Process(
                    target=run_read_subgraph_query,
                    args=(subgraph_queries[split * i : split * (i + 1)],),
                )
                processes.append(process)
                process.start()
            for process in processes:
                process.join()
            results[f"subgraph_time_N{N}_{num_processes}"] = (
                time.perf_counter() - total_time_start
            )

            # subgraph edge reads
            processes = []
            total_time_start = time.perf_counter()
            for i in range(num_processes):
                process = multiprocessing.Process(
                    target=run_read_subgraph_edge_query,
                    args=(subgraph_edge_queries[split * i : split * (i + 1)],),
                )
                processes.append(process)
                process.start()
            for process in processes:
                process.join()
            results[f"subgraph_edge_time_N{N}_{num_processes}"] = (
                time.perf_counter() - total_time_start
            )

    print(results)

    if args.results_to_csv:
        df = pd.DataFrame(results, index=[0])
        df.to_csv(f"read_weak_benchmark_{args.mode}.csv", index=False)
