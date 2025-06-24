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
from HOGDB.graph.graph_with_tuple_storage import GraphwithTupleStorage
from HOGDB.db.db import Database
from HOGDB.db.neo4j import Neo4jDatabase
from HOGDB.db.label import Label
from HOGDB.db.schema import Schema
from HOGDB.db.property import Property
from HOGDB.graph.node import Node
from HOGDB.graph.edge import Edge
from HOGDB.graph.node_tuple import NodeTuple
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
    "--tuples",
    action="store_true",
    help="Run benchmark for graphs with node-tuple collections.",
)
parser.add_argument(
    "--hypergraph", action="store_true", help="Run benchmark for hypergraphs."
)
parser.add_argument(
    "--split",
    type=str,
    default="mostly-reads",
    help="Query split type: 'mostly-reads', 'mixed', or 'write-heavy'.",
)
parser.add_argument("--no_import", action="store_true", help="Skip data import step.")
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
    "--csv_url", type=str, default="", help="Base URL for online CSV files."
)
args = parser.parse_args()

db_params = {
    "db_uri": "bolt://localhost:7687",
    "db_username": "neo4j",
    "db_password": "password",
}

import_times = []
nodes_count = 20000
nodes, edges, tuples, hedges = {}, {}, {}, {}
eupds, tupds, hupds = [], [], []


def read_tuple_csvs() -> None:
    """
    Read graph (with node-tuples) data from CSV files and store it in global variables.
    """
    global edges, tuples, eupds, tupds
    pref = f"{args.csv_url}/" if args.online else args.import_dir
    edges_df = pd.read_csv(f"{pref}/rgt_edges.csv")
    edges.update(
        {
            (row["source"], row["target"]): row["edge_feature"]
            for _, row in edges_df.iterrows()
        }
    )
    edges_df = pd.read_csv(f"{pref}/rgt_edges_upd.csv")
    eupds.extend(
        [
            (row["id"], row["source"], row["target"], row["edge_feature"])
            for _, row in edges_df.iterrows()
        ]
    )
    tuples_df = pd.read_csv(f"{pref}/rgt_tuple_properties.csv")
    tuples.update({row["id"]: row["tuple_feature"] for _, row in tuples_df.iterrows()})
    tuples_df = pd.read_csv(f"{pref}/rgt_tuples_upd.csv")
    tupds.extend(
        [
            (
                row["id"],
                list(map(int, str(row["nodes"]).split(";"))),
                row["tuple_feature"],
            )
            for _, row in tuples_df.iterrows()
        ]
    )


def read_hypergraph_csvs() -> None:
    """
    Read hypergraph data from CSV files and store it in global variables.
    """
    global nodes, hedges, hupds
    pref = f"{args.csv_url}/" if args.online else args.import_dir
    nodes_df = pd.read_csv(f"{pref}/rhg_nodes.csv")
    nodes.update({row["id"]: row["node_feature"] for _, row in nodes_df.iterrows()})
    hedges_df = pd.read_csv(f"{pref}/rhg_edge_properties.csv")
    hedges.update({row["id"]: row["feature"] for _, row in hedges_df.iterrows()})
    hedges_df = pd.read_csv(f"{pref}/rhg_edges_upd.csv")
    hupds.extend(
        [
            (
                row["edge_id"],
                list(map(int, str(row["nodes"]).split(";"))),
                row["edge_feature"],
            )
            for _, row in hedges_df.iterrows()
        ]
    )


def get_tuples(db: Database) -> None:
    """
    Read data of a graph with node-tuples from CSV files and store it in a graph database.

    @param db: Database where the higher-order graph is going to be stored.
    """
    session = db.start_session()
    db.create_index(session, Label("Node"), ["id"])
    db.create_index(session, Label("Edge"), ["id"])
    db.create_index(session, Label("Tuple"), ["id"])

    start = time.perf_counter()
    pref = f"{args.csv_url}/" if args.online else ""
    db.import_nodes_from_csv(
        session,
        f"{pref}rgt_nodes.csv",
        [Label("_node"), Label("Node")],
        [
            Schema("id", int),
            Schema("node_feature", float),
        ],
        as_url=args.online,
    )
    db.import_node_edges_from_csv(
        session,
        f"{pref}rgt_edges.csv",
        [Label("_node"), Label("Node")],
        [Schema("id", int, "source")],
        [Label("_node"), Label("Node")],
        [Schema("id", int, "target")],
        [Label("_edge"), Label("Node")],
        [
            Schema("id", int),
            Schema("edge_feature", float),
        ],
        Label("_adjacency"),
        as_url=args.online,
    )
    db.import_nodes_from_csv(
        session,
        f"{pref}rgt_tuple_properties.csv",
        [Label("_tuple"), Label("Tuple")],
        [
            Schema("id", int),
            Schema("tuple_feature", float),
        ],
        as_url=args.online,
    )
    db.import_edges_from_csv(
        session,
        f"{pref}rgt_tuple_edges.csv",
        [Label("_node"), Label("Node")],
        [Schema("id", int, "node_id")],
        [Label("_tuple"), Label("Tuple")],
        [Schema("id", int)],
        Label("_node_membership"),
        [],
        as_url=args.online,
    )
    end = time.perf_counter()
    print(f"Imported data in {end - start:.4f} seconds.")
    import_times.append(end - start)


def get_hypergraphs(db: Database) -> None:
    """
    Read data of a hypergraph from CSV files and store it in a graph database.

    @param db: Database where the hypergraph is going to be stored.
    """
    session = db.start_session()
    db.create_index(session, Label("Node"), ["id"])
    db.create_index(session, Label("Hyperedge"), ["id"])
    start = time.perf_counter()
    pref = f"{args.csv_url}/" if args.online else ""
    db.import_nodes_from_csv(
        session,
        f"{pref}rhg_nodes.csv",
        [Label("_node"), Label("Node")],
        [Schema("id", int), Schema("node_feature", float)],
        as_url=args.online,
    )
    db.import_nodes_from_csv(
        session,
        f"{pref}rhg_edge_properties.csv",
        [Label("_hyperedge"), Label("Hyperedge")],
        [Schema("id", int, "id"), Schema("edge_feature", float, "feature")],
        as_url=args.online,
    )
    db.import_edges_from_csv(
        session,
        f"{pref}rhg_edges.csv",
        [Label("Node")],
        [Schema("id", int, "node_id")],
        [Label("Hyperedge")],
        [Schema("id", int, "edge_id")],
        Label("_adjacency"),
        [],
        as_url=args.online,
    )
    db.import_edges_from_csv(
        session,
        f"{pref}rhg_edges.csv",
        [Label("Hyperedge")],
        [Schema("id", int, "edge_id")],
        [Label("Node")],
        [Schema("id", int, "node_id")],
        Label("_adjacency"),
        [],
        as_url=args.online,
    )
    end = time.perf_counter()

    print(f"Imported data in {end - start:.4f} seconds.")


def gen_node_reads(N: int) -> List[Tuple[int, Node]]:
    """
    Generate a number of nodes to be used for querying the higher-order graph as "node read
    queries".

    @param N: Number of query targets to generate.
    @return: List of type and nodes.
    """
    queries = []
    global nodes_count
    for _ in range(N):
        node_id = random.randint(0, nodes_count - 1)
        queries.append((1, Node([Label("Node")], [Property("id", int, node_id)])))
    return queries


def gen_edge_reads(N: int) -> List[Tuple[int, Edge]]:
    """
    Generate a number of edges to be used for querying the higher-order graph as "edge read
    queries".

    @param N: Number of query targets to generate.
    @return: List of type and edges.
    """
    queries = []
    keys = list(edges.keys())
    for _ in range(N):
        source, target = random.choice(keys)
        edge = Edge(
            Node([Label("Node")], [Property("id", int, source)]),
            Node([Label("Node")], [Property("id", int, target)]),
            Label("Edge"),
            [],
        )
        queries.append((2, edge))
    return queries


def gen_tuple_reads(N: int) -> List[Tuple[int, NodeTuple]]:
    """
    Generate a number of node-tuples to be used for querying the higher-order graph as "node-tuple
    read queries".

    @param N: Number of query targets to generate.
    @return: List of type and node-tuples.
    """
    queries = []
    keys = list(tuples.keys())
    for _ in range(N):
        tuple_id = random.choice(keys)
        node_tuple = NodeTuple(
            [],
            [Label("Tuple")],
            [Property("id", int, tuple_id)],
        )
        queries.append((3, node_tuple))
    return queries


def gen_edge_writes(N: int) -> List[Tuple[int, Edge]]:
    """
    Generate a number of edges with their labels and properties to be used for querying the
    higher-order graph as "edge write queries".

    @param N: Number of query targets to generate.
    @return: List of type and edges.
    """
    queries = random.sample(eupds, N)
    return [
        (
            4,
            Edge(
                Node([Label("Node")], [Property("id", int, source)]),
                Node([Label("Node")], [Property("id", int, target)]),
                Label("Edge"),
                [
                    Property("id", int, id),
                    Property("edge_feature", float, feat),
                ],
            ),
        )
        for id, source, target, feat in queries
    ]


def gen_tuple_writes(N: int) -> List[Tuple[int, NodeTuple]]:
    """
    Generate a number of node-tuples and their labels and properties to be used for querying the
    higher-order graph as "node-tuple write queries".

    @param N: Number of query targets to generate.
    @return: List of type and node-tuples.
    """
    queries = random.sample(tupds, N)
    return [
        (
            5,
            NodeTuple(
                [Node([Label("Node")], [Property("id", int, id)]) for id in nodes],
                [Label("Tuple")],
                [
                    Property("id", int, id),
                    Property("tuple_feature", float, feat),
                ],
            ),
        )
        for id, nodes, feat in queries
    ]


def run_tuple_queries(queries: List[Tuple[int, any]]) -> None:
    """
    Run queries on a graph with node-tuples.

    @param queries: Queries to be run.
    """
    db = Neo4jDatabase(**db_params)
    gs = GraphwithTupleStorage(db)
    query_map = {
        1: gs.get_node,
        2: gs.get_edge,
        3: gs.get_node_tuple,
        4: gs.add_edge,
        5: gs.add_node_tuple,
    }
    for query_type, query_arg in queries:
        _ = query_map[query_type](query_arg)
    gs.close_connection()


def gen_node_reads_h(N: int) -> List[Tuple[int, Node]]:
    """
    Generate a number of nodes to be used for querying the hypergraph as "node read queries".

    @param N: Number of query targets to generate.
    @return: List of type and nodes.
    """
    queries = []
    global nodes
    keys = list(nodes.keys())
    for _ in range(N):
        node_id = random.choice(keys)
        queries.append((1, Node([Label("Node")], [Property("id", int, node_id)])))
    return queries


def gen_edge_reads_h(N: int) -> List[Tuple[int, HyperEdge]]:
    """
    Generate a number of hyperedges to be used for querying the hypergraph as "hyperedge read
    queries".

    @param N: Number of query targets to generate.
    @return: List of type and hyperedges.
    """
    queries = []
    global hedges
    keys = list(hedges.keys())
    for _ in range(N):
        edge_id = random.choice(keys)
        queries.append(
            (2, HyperEdge([], Label("Hyperedge"), [Property("id", int, edge_id)]))
        )
    return queries


def gen_node_writes_h(N: int) -> List[Tuple[int, Node]]:
    """
    Generate a number of nodes as well as their labels and properties to be used for querying the
    hypergraph as "node write queries".

    @param N: Number of query targets to generate.
    @return: List of type and nodes.
    """
    queries = []
    global nodes
    keys = list(nodes.keys())
    for _ in range(N):
        node_id = random.choice(keys)
        queries.append(
            (
                3,
                (
                    Node([Label("Node")], [Property("id", int, node_id)]),
                    [Property("node_feature", float, random.uniform(0, 5))],
                ),
            )
        )
    return queries


def gen_edge_writes_h(N: int) -> List[Tuple[int, HyperEdge]]:
    """
    Generate a number of hyperedges as well as their labels and properties to be used for querying
    the hypergraph as "hyperedge write queries".

    @param N: Number of query targets to generate.
    @return: List of type and hyperedges.
    """
    global hupds
    qs = random.sample(hupds, N)
    queries = [
        (
            4,
            HyperEdge(
                [Node([Label("Node")], [Property("id", int, id)]) for id in _nodes],
                Label("Hyperedge"),
                [
                    Property("id", int, _id),
                    Property("edge_feature", float, feat),
                ],
            ),
        )
        for _id, _nodes, feat in qs
    ]
    return queries


def run_hypergraph_queries(queries: List[Tuple[int, any]]) -> None:
    """
    Run queries on a hypergraph.

    @param queries: Queries to be run.
    """
    db = Neo4jDatabase(**db_params)
    gs = HyperGraphStorage(db)
    query_map = {
        1: gs.get_node,
        2: gs.get_hyperedge,
        3: gs.update_node,
        4: gs.add_hyperedge,
    }

    def wrap_call(f):
        """
        Wrapper function to return the right type of function call.

        @param f: Function to be called.
        @return: Callable function.
        """

        def wrapper(arg):
            """
            Wrapper function to use right kind of parameter passing.

            @param arg: Function parameters.
            @return: Callable function.
            """
            return f(*arg) if isinstance(arg, tuple) else f(arg)

        return wrapper

    query_map = {k: wrap_call(v) for k, v in query_map.items()}
    for query_type, query_arg in queries:
        _ = query_map[query_type](query_arg)
    gs.close_connection()


if __name__ == "__main__":
    global read_times
    results = {}

    if args.tuples:
        if args.mode == "debug":
            splits_list = [
                {
                    "mostly-reads": [56, 56, 56, 4, 20],
                    "mixed": [32, 32, 32, 84, 12],
                    "write-heavy": [16, 16, 16, 120, 24],
                }
            ]
        else:
            splits_list = [
                {
                    "mostly-reads": [3892, 3892, 3892, 102, 510],
                    "mixed": [2048, 2048, 2048, 5120, 1024],
                    "write-heavy": [1024, 1024, 1024, 7680, 1536],
                }
            ]
        db = Neo4jDatabase(**db_params)
        gs = GraphwithTupleStorage(db)
        read_tuple_csvs()
        for scale, splits in enumerate(splits_list):
            split_num = splits[args.split]
            qs = gen_node_reads(split_num[0])
            qs.extend(gen_edge_reads(split_num[1]))
            qs.extend(gen_tuple_reads(split_num[2]))
            qs.extend(gen_edge_writes(split_num[3]))
            qs.extend(gen_tuple_writes(split_num[4]))
            random.shuffle(qs)

            num_processes_list = [2] if args.mode == "debug" else [8, 16, 32, 64, 128]
            for num_processes in num_processes_list:
                print(f"Running with {num_processes} processes...")
                process_split = len(qs) // num_processes
                db = Neo4jDatabase(**db_params)
                gs = GraphwithTupleStorage(db)
                gs.clear_graph()
                gs.close_connection()
                db = Neo4jDatabase(**db_params)
                get_tuples(db)

                processes = []
                read_times = [[] for _ in range(num_processes)]
                total_time_start = time.perf_counter()
                for i in range(num_processes):
                    process = multiprocessing.Process(
                        target=run_tuple_queries,
                        args=(qs[process_split * i : process_split * (i + 1)],),
                    )
                    processes.append(process)
                    process.start()
                for process in processes:
                    process.join()
                results[f"time_{args.split}_{scale}_{num_processes}"] = (
                    time.perf_counter() - total_time_start
                )
                print(f"Finished with {num_processes} processes...")
        print(results)

    elif args.hypergraph:
        if args.mode == "debug":
            splits = {
                "mostly-reads": [60, 60, 4, 4],
                "mixed": [32, 32, 32, 32],
                "write-heavy": [16, 16, 48, 48],
            }
        else:
            splits = {
                "mostly-reads": [3840, 3840, 256, 256],
                "mixed": [2048, 2048, 3456, 640],
                "write-heavy": [1024, 1024, 4736, 1408],
            }
        read_hypergraph_csvs()

        num_processes_list = [2] if args.mode == "debug" else [8, 16, 32, 64, 128]

        splits_num = splits[args.split]
        qs = gen_node_reads_h(splits_num[0])
        qs.extend(gen_edge_reads_h(splits_num[1]))
        qs.extend(gen_node_writes_h(splits_num[2]))
        qs.extend(gen_edge_writes_h(splits_num[3]))
        random.shuffle(qs)

        num_processes_list = [2] if args.mode == "debug" else [8, 16, 32, 64, 128]
        for num_processes in num_processes_list:
            db = Neo4jDatabase(**db_params)
            hs = HyperGraphStorage(db)
            hs.clear_graph()
            hs.close_connection()
            db = Neo4jDatabase(**db_params)
            get_hypergraphs(db)
            print(f"Running with {num_processes} processes...")
            process_split = len(qs) // num_processes

            processes = []
            read_times = [[] for _ in range(num_processes)]
            total_time_start = time.perf_counter()
            for i in range(num_processes):
                process = multiprocessing.Process(
                    target=run_hypergraph_queries,
                    args=(qs[process_split * i : process_split * (i + 1)],),
                )
                processes.append(process)
                process.start()
            for process in processes:
                process.join()
            results[f"time_{args.split}_{num_processes}"] = (
                time.perf_counter() - total_time_start
            )
            print(f"Finished with {num_processes} processes...")
        print(results)
    else:
        raise ValueError("Please specify --tuples or --hypergraph")
