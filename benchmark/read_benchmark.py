# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Shriram Chandran

from HOGDB.graph.graph_storage import GraphStorage
from HOGDB.graph.graph_with_subgraph_storage import GraphwithSubgraphStorage
from HOGDB.graph.hypergraph_storage import HyperGraphStorage
from HOGDB.db.db import Database
from HOGDB.db.neo4j import Neo4jDatabase
from HOGDB.db.label import Label
from HOGDB.db.schema import Schema
from HOGDB.db.property import Property
from HOGDB.graph.node import Node
from HOGDB.graph.edge import Edge
from HOGDB.graph.subgraph import Subgraph, SubgraphEdge
from HOGDB.graph.hyperedge import HyperEdge
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
    "--subgraph",
    action="store_true",
    help="Run benchmark for graphs with subgraph collections.",
)
parser.add_argument(
    "--hypergraph", action="store_true", help="Run benchmark for hypergraphs."
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

import_times = []

"""
Please note that the ZINC dataset (used for the higher-order graph with subgraph collections
experiment) stores each molecule in a separate graph, so an additional graph ID together with an
atom ID is necessary to identify a specific graph node.
"""


def read_subgraph_csvs() -> None:
    """
    Read graph data from CSV files and store it in global variables.
    """
    global nodes, edges, subgraphs, subgraph_edges
    nodes = {}
    edges = {}
    subgraphs = {}
    subgraph_edges = {}
    pref = f"{args.csv_url}/" if args.online else args.import_dir
    nodes_df = pd.read_csv(f"{pref}/zinc_atoms.csv")
    nodes.update(
        {
            (row["atom_id"], row["graph_id"]): row["atom_features"]
            for _, row in nodes_df.iterrows()
        }
    )
    edges_df = pd.read_csv(f"{pref}/zinc_bonds.csv")
    edges.update(
        {
            (row["start_atom"], row["end_atom"], row["graph_id"]): row["bond_features"]
            for _, row in edges_df.iterrows()
        }
    )
    subgraphs_df = pd.read_csv(f"{pref}/zinc_cliques.csv")
    subgraphs.update(
        {
            (row["clique_id"], row["graph_id"]): (row["clique_features"])
            for _, row in subgraphs_df.iterrows()
        }
    )
    subgraph_edges_df = pd.read_csv(f"{pref}/zinc_tree_edges.csv")
    subgraph_edges.update(
        {
            (row["start_clique"], row["end_clique"], row["graph_id"]): row[
                "tree_edge_id"
            ]
            for _, row in subgraph_edges_df.iterrows()
        }
    )


def read_hypergraph_csvs(labels: List[str]) -> None:
    """
    Read graph data from CSV files and store it in global variables.

    @param labels: List of labels for the hyperedges.
    """
    pref = f"{args.csv_url}/" if args.online else args.import_dir
    nodes_df = pd.read_csv(f"{pref}/mag10_nodes.csv")
    nodes.update({row["id"]: 1 for _, row in nodes_df.iterrows()})
    for label in labels:
        hyperedges_df = pd.read_csv(f"{pref}/mag10_hyperedges_{label}.csv")
        grouped = hyperedges_df.groupby("hyperedge")
        hedges = grouped["personid"].apply(list).tolist()
        hnodes_df = pd.read_csv(f"{pref}/mag10_hyperedges_properties_{label}.csv")
        edges.update(
            {(label, row["id"]): len(hedges[idx]) for idx, row in hnodes_df.iterrows()}
        )


def get_subgraphs(db: Database) -> None:
    """
    Read data of a graph with subgraph collections from CSV files and store it in a graph database
    as well as in global variables.

    @param db: Database where the higher-order graph is going to be stored.
    """
    session = db.start_session()
    db.create_index(session, Label("Atom"), ["atom_id", "graph_id"])
    db.create_index(session, Label("Bond"), ["bond_id", "graph_id"])
    db.create_index(session, Label("Clique"), ["clique_id", "graph_id"])
    db.create_index(session, Label("TreeEdge"), ["tree_edge_id", "graph_id"])

    start = time.perf_counter()

    pref = f"{args.csv_url}/" if args.online else ""
    db.import_nodes_from_csv(
        session,
        f"{pref}zinc_atoms.csv",
        [Label("_node"), Label("Atom")],
        [Schema("atom_id", int), Schema("graph_id", int), Schema("atom_features", int)],
        as_url=args.online,
    )
    db.import_node_edges_from_csv(
        session,
        f"{pref}zinc_bonds.csv",
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
        as_url=args.online,
    )
    db.import_nodes_from_csv(
        session,
        f"{pref}zinc_cliques.csv",
        [Label("_subgraph"), Label("Clique")],
        [
            Schema("clique_id", int),
            Schema("graph_id", int),
            Schema("clique_features", int),
        ],
        as_url=args.online,
    )
    db.import_edges_from_csv(
        session,
        f"{pref}zinc_clique_membership.csv",
        [Label("_node"), Label("Atom")],
        [Schema("atom_id", int), Schema("graph_id", int)],
        [Label("_subgraph"), Label("Clique")],
        [Schema("clique_id", int), Schema("graph_id", int)],
        Label("_node_membership"),
        [],
        as_url=args.online,
    )
    db.import_edges_from_csv(
        session,
        f"{pref}zinc_clique_bond_membership.csv",
        [Label("_edge"), Label("Bond")],
        [Schema("bond_id", int), Schema("graph_id", int)],
        [Label("_subgraph"), Label("Clique")],
        [Schema("clique_id", int), Schema("graph_id", int)],
        Label("_edge_membership"),
        [],
        as_url=args.online,
    )
    db.import_node_edges_from_csv(
        session,
        f"{pref}zinc_tree_edges.csv",
        [Label("_subgraph"), Label("Clique")],
        [Schema("clique_id", int, "start_clique"), Schema("graph_id", int)],
        [Label("_subgraph"), Label("Clique")],
        [Schema("clique_id", int, "end_clique"), Schema("graph_id", int)],
        [Label("_subgraph_edge"), Label("TreeEdge")],
        [Schema("tree_edge_id", int), Schema("graph_id", int)],
        Label("_subgraph_adjacency"),
        as_url=args.online,
    )

    end = time.perf_counter()
    print(f"Imported data in {end - start:.4f} seconds.")

    read_subgraph_csvs()


def get_hypergraphs(db: Database) -> None:
    """
    Read hypergraph data from CSV files and store it in a graph database as well as in global
    variables.

    @param db: Database where the hypergraph is going to be stored.
    """
    session = db.start_session()
    db.create_index(session, Label("Person"), ["id"])
    labels = [
        "CVPR",
        "FOCS",
        "ICCV",
        "ICML",
        "KDD",
        "NeurIPS",
        "SIGMOD",
        "STOC",
        "VLDB",
        "WWW",
    ]
    for label in labels:
        db.create_index(session, Label(label), ["id"])

    start = time.perf_counter()
    pref = f"{args.csv_url}/" if args.online else ""
    db.import_nodes_from_csv(
        session,
        f"{pref}/mag10_nodes.csv",
        [Label("_node"), Label("Person")],
        [Schema("id", int)],
        as_url=args.online,
    )

    for label in labels:
        db.import_nodes_from_csv(
            session,
            f"{pref}/mag10_hyperedges_properties_{label}.csv",
            [Label("_hyperedge"), Label(label)],
            [Schema("id", int)],
            as_url=args.online,
        )
        db.import_edges_from_csv(
            session,
            f"{pref}/mag10_hyperedges_{label}.csv",
            [Label("Person")],
            [Schema("id", int, "personid")],
            [Label(label)],
            [Schema("id", int, "hyperedge")],
            Label("_adjacency"),
            [],
            as_url=args.online,
        )
        db.import_edges_from_csv(
            session,
            f"{pref}/mag10_hyperedges_{label}.csv",
            [Label(label)],
            [Schema("id", int, "hyperedge")],
            [Label("Person")],
            [Schema("id", int, "personid")],
            Label("_adjacency"),
            [],
            as_url=args.online,
        )
    end = time.perf_counter()

    print(f"Imported data in {end - start:.4f} seconds.")
    read_hypergraph_csvs(labels)


def gen_node_queries_hypergraph(N: int) -> List[int]:
    """
    Generate a number of node IDs to be used for querying the hypergraph as "node queries".

    @param N: Number of node IDs to generate.
    @return: List of node IDs.
    """
    queries = []
    keys = list(nodes.keys())
    for i in range(N):
        person_id = random.choice(keys)
        queries.append(person_id)
    return queries


def gen_edge_queries_hypergraph(N: int) -> List[Tuple[str, int]]:
    """
    Generate a number of hyperedge IDs and their labels to be used for querying the hypergraph as
    "hyperedge queries".

    @param N: Number of hyperedges to generate.
    @return: List of labels and hyperedge IDs.
    """
    queries = []
    keys = list(edges.keys())
    for i in range(N):
        (label, id) = random.choice(keys)
        queries.append((label, id))
    return queries


def run_read_node_query_with_checks(i: int, queries: List[Tuple[int, int]]) -> None:
    """
    Run node retrieval queries on a graph with subgraph collections and measure the runtime of each
    individual query. Store these runtimes in a global variable. Additional checks, whether the
    right node is retrieved, are performed.

    @param i: Index for storing the runtimes.
    @param queries: List of atom IDs and graph IDs.
    """
    global read_times
    db = Neo4jDatabase(**db_params)
    gs = GraphwithSubgraphStorage(db)
    for atom_id, graph_id in queries:
        start = time.perf_counter()
        node = gs.get_node(
            Node(
                [Label("Atom")],
                [
                    Property("atom_id", int, atom_id),
                    Property("graph_id", int, graph_id),
                ],
            )
        )
        end = time.perf_counter()
        assert node["atom_id"] == atom_id
        assert node["atom_features"] == nodes[atom_id, graph_id]
        read_times[i].append(end - start)
    gs.close_connection()


def run_read_edge_query_with_checks(
    i: int, queries: List[Tuple[int, int, int]]
) -> None:
    """
    Run edge retrieval queries on a graph with subgraph collections and measure the runtime of each
    individual query. Store these runtimes in a global variable. Additional checks, whether the
    right edge is retrieved, are performed.

    @param i: Index for storing the runtimes.
    @param queries: List of atom IDs (origin and target) and graph IDs.
    """
    global read_times
    db = Neo4jDatabase(**db_params)
    gs = GraphwithSubgraphStorage(db)
    for start_atom, end_atom, graph_id in queries:
        start = time.perf_counter()
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
        end = time.perf_counter()
        assert edge["bond_features"] == edges[start_atom, end_atom, graph_id]
        read_times[i].append(end - start)
    gs.close_connection()


def run_read_subgraph_query_with_checks(i: int, queries: List[Tuple[int, int]]) -> None:
    """
    Run subgraph collection retrieval queries on a graph with subgraph collections and measure the
    runtime of each individual query. Store these runtimes in a global variable. Additional checks,
    whether the right subgraph collection is retrieved, are performed.

    @param i: Index for storing the runtimes.
    @param queries: List of subgraph collection IDs and graph IDs.
    """
    global read_times
    db = Neo4jDatabase(**db_params)
    gs = GraphwithSubgraphStorage(db)
    for clique_id, graph_id in queries:
        start = time.perf_counter()
        sg = gs.get_subgraph(
            Subgraph(
                labels=[Label("Clique")],
                properties=[
                    Property("clique_id", int, clique_id),
                    Property("graph_id", int, graph_id),
                ],
            )
        )
        end = time.perf_counter()
        assert sg["clique_id"] == clique_id
        assert sg["clique_features"] == subgraphs[clique_id, graph_id]
        read_times[i].append(end - start)
    gs.close_connection()


def run_read_subgraph_edge_query_with_checks(
    i: int, queries: List[Tuple[int, int, int]]
) -> None:
    """
    Run subgraph edge retrieval queries on a graph with subgraph collections and measure the
    runtime of each individual query. Store these runtimes in a global variable. Additional checks,
    whether the right subgraph edge is retrieved, are performed.

    @param i: Index for storing the runtimes.
    @param queries: List of subgraph collection IDs (origin and target) and graph IDs.
    """
    global read_times
    db = Neo4jDatabase(**db_params)
    gs = GraphwithSubgraphStorage(db)
    for start_clique, end_clique, graph_id in queries:
        start = time.perf_counter()
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
        end = time.perf_counter()
        assert (
            sg_edge["tree_edge_id"]
            == subgraph_edges[start_clique, end_clique, graph_id]
        )
        read_times[i].append(end - start)
    gs.close_connection()


def run_read_node_query_hypergraph_with_checks(i: int, queries: List[int]) -> None:
    """
    Run node retrieval queries on a hypergraph and measure the runtime of each individual query.
    Store these runtimes in a global variable. Additional checks, whether the right node is
    retrieved, are performed.

    @param i: Index for storing the runtimes.
    @param queries: List of node IDs.
    """
    global read_times
    db = Neo4jDatabase(**db_params)
    gs = HyperGraphStorage(db)
    for person_id in queries:
        start = time.perf_counter()
        node = gs.get_node(Node([Label("Person")], [Property("id", int, person_id)]))
        end = time.perf_counter()
        assert node["id"] == person_id
        read_times[i].append(end - start)
    gs.close_connection()


def run_read_node_query_hypergraph(queries: List[int]) -> None:
    """
    Run node retrieval queries on a hypergraph.

    @param queries: List of node IDs.
    """
    db = Neo4jDatabase(**db_params)
    gs = HyperGraphStorage(db)
    for person_id in queries:
        node = gs.get_node(Node([Label("Person")], [Property("id", int, person_id)]))
    gs.close_connection()


def run_read_edge_query_hypergraph_with_checks(
    i: int, queries: List[Tuple[str, int]]
) -> None:
    """
    Run hyperedge retrieval queries on a hypergraph and measure the runtime of each individual query.
    Store these runtimes in a global variable. Additional checks, whether the right hyperedge is
    retrieved, are performed.

    @param i: Index for storing the runtimes.
    @param queries: List of labels and hyperedge IDs.
    """
    global read_times
    db = Neo4jDatabase(**db_params)
    gs = HyperGraphStorage(db)
    for label, id in queries:
        start = time.perf_counter()
        edge = gs.get_hyperedge(HyperEdge([], Label(label), [Property("id", int, id)]))
        end = time.perf_counter()
        assert len(edge.nodes) == edges[(label, id)]
        read_times[i].append(end - start)
    gs.close_connection()


def run_read_edge_query_hypergraph(queries: List[Tuple[str, int]]) -> None:
    """
    Run hyperedge retrieval queries on a hypergraph.

    @param queries: List of labels and hyperedge IDs.
    """
    db = Neo4jDatabase(**db_params)
    gs = HyperGraphStorage(db)
    for label, id in queries:
        edge = gs.get_hyperedge(HyperEdge([], Label(label), [Property("id", int, id)]))
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
    global read_times
    results = {}
    if args.subgraph:
        db = Neo4jDatabase(**db_params)
        gs = GraphwithSubgraphStorage(db)
        gs.clear_graph()
        gs.close_connection()
        db = Neo4jDatabase(**db_params)
        get_subgraphs(db)

        num_queries = 64 if args.mode == "debug" else 4096
        node_queries = gen_node_queries_subgraph(num_queries)
        edge_queries = gen_edge_queries_subgraph(num_queries)
        subgraph_queries = gen_subgraph_queries_subgraph(num_queries)
        subgraph_edge_queries = gen_subgraph_edge_queries_subgraph(num_queries)
        read_times = [[]]
        run_read_node_query_with_checks(0, node_queries)
        results["node_read_times"] = read_times[0]
        read_times = [[]]
        run_read_edge_query_with_checks(0, edge_queries)
        results["edge_read_times"] = read_times[0]
        read_times = [[]]
        run_read_subgraph_query_with_checks(0, subgraph_queries)
        results["subgraph_read_times"] = read_times[0]
        read_times = [[]]
        run_read_subgraph_edge_query_with_checks(0, subgraph_edge_queries)
        results["subgraph_edge_read_times"] = read_times[0]

        num_processes_list = [2] if args.mode == "debug" else [8, 16, 32, 64, 128]
        for idx, num_processes in enumerate(num_processes_list):
            node_queries = gen_node_queries_subgraph(num_queries)
            edge_queries = gen_edge_queries_subgraph(num_queries)
            subgraph_queries = gen_subgraph_queries_subgraph(num_queries)
            subgraph_edge_queries = gen_subgraph_edge_queries_subgraph(num_queries)

            print(f"Running with {num_processes} processes...")
            split = num_queries // num_processes

            # node reads
            print("Nodes...")
            processes = []
            read_times = [[] for _ in range(num_processes)]
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
            results[f"total_node_time_{num_processes}"] = (
                time.perf_counter() - total_time_start
            )

            # edge reads
            print("Edges...")
            processes = []
            read_times = [[] for _ in range(num_processes)]
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
            results[f"total_edge_time_{num_processes}"] = (
                time.perf_counter() - total_time_start
            )

            # subgraph reads
            print("Subgraphs...")
            processes = []
            read_times = [[] for _ in range(num_processes)]
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
            results[f"total_subgraph_time_{num_processes}"] = (
                time.perf_counter() - total_time_start
            )

            # subgraph edge reads
            print("Subgraph edges...")
            processes = []
            read_times = [[] for _ in range(num_processes)]
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
            results[f"total_subgraph_edge_time_{num_processes}"] = (
                time.perf_counter() - total_time_start
            )

        print(results)

    elif args.hypergraph:
        db = Neo4jDatabase(**db_params)
        gs = HyperGraphStorage(db)
        gs.clear_graph()
        gs.close_connection()
        db = Neo4jDatabase(**db_params)
        get_hypergraphs(db)

        num_queries = 64 if args.mode == "debug" else 4096
        node_queries = gen_node_queries_hypergraph(num_queries)
        edge_queries = gen_edge_queries_hypergraph(num_queries)
        read_times = [[]]
        run_read_node_query_hypergraph_with_checks(0, node_queries)
        results["node_read_times"] = read_times[0]
        read_times = [[]]
        run_read_edge_query_hypergraph_with_checks(0, edge_queries)
        results["hyperedge_read_times"] = read_times[0]

        num_processes_list = [2] if args.mode == "debug" else [4, 8, 16, 32, 64, 128]
        for num_processes in num_processes_list:
            print(f"Running with {num_processes} processes...")
            split = num_queries // num_processes

            # node reads
            print("Nodes...")
            processes = []
            read_times = [[] for _ in range(num_processes)]
            total_time_start = time.perf_counter()
            for i in range(num_processes):
                process = multiprocessing.Process(
                    target=run_read_node_query_hypergraph,
                    args=(node_queries[split * i : split * (i + 1)],),
                )
                processes.append(process)
                process.start()
            for process in processes:
                process.join()
            results[f"total_node_time_{num_processes}"] = (
                time.perf_counter() - total_time_start
            )

            # edge reads
            print("Edges...")
            processes = []
            read_times = [[] for _ in range(num_processes)]
            total_time_start = time.perf_counter()
            for i in range(num_processes):
                process = multiprocessing.Process(
                    target=run_read_edge_query_hypergraph,
                    args=(edge_queries[split * i : split * (i + 1)],),
                )
                processes.append(process)
                process.start()
            for process in processes:
                process.join()
            results[f"total_edge_time_{num_processes}"] = (
                time.perf_counter() - total_time_start
            )

        print(results)

    else:
        raise ValueError("Please specify --subgraph or --hypergraph")
