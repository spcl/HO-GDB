# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Shriram Chandran

from torch_geometric.datasets import ZINC
from torch_geometric.data import Dataset
from rdkit.Chem.rdchem import BondType

import numpy as np
import pandas as pd
import hypernetx as hnx

import networkx as nx
import argparse
import random

from utils import JunctionTree, JunctionTreeData, mol_from_data


parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", type=str, default="debug", help="Mode of operation: 'debug' or 'full'."
)
parser.add_argument(
    "--root",
    type=str,
    default="datasets",
    help="Root directory containing raw datasets.",
)
parser.add_argument(
    "--dir", type=str, default=".", help="Output directory for generated CSV files."
)
parser.add_argument(
    "--data",
    type=str,
    default="zinc",
    help="Dataset to process: 'zinc', 'mag10', 'random_t', or 'random_h'.",
)
parser.add_argument(
    "--weak",
    action="store_true",
    help="Generate datasets for weak scaling (exclusive to ZINC dataset).",
)
args = parser.parse_args()
print(args)

bonds = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]


def get_zinc(data: Dataset, N: int) -> None:
    """
    Create subgraph collection based on the ZINC data and write the results into multiple CSV
    files.

    @param data: ZINC data (e.g., torch_geometric.data.Dataset or similar).
    @param N: Number of graphs.
    """
    N_str = f"_{N}" if args.weak else ""
    graphs = range(0, N)

    # Get atoms
    nodes, edges, subgraph_nodes, subgraph_edgel, subgraphs, subgraph_edges = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for graph in graphs:
        # Get graph index
        index = graph
        # Get nodes
        x = data[index].x.squeeze().tolist()
        nodes_list = [[id, graph, nt] for id, nt in enumerate(x)]

        # Get bonds
        e = data[index].edge_index.tolist()
        edge_attrs = data[index].edge_attr.tolist()
        st, en = e[0], e[1]
        edges_list = [
            [id, graph, s, t, edge_attrs[id]] for id, (s, t) in enumerate(zip(st, en))
        ]

        # Get cliques
        row, col = data[index].atom2clique_index.tolist()
        x_clique = data[index].x_clique.tolist()
        subgraphs_list = [
            [
                id,
                graph,
                _nodes,
                [id for id, _, s, t, _ in edges_list if s in _nodes and t in _nodes],
                f,
            ]
            for id, f in enumerate(x_clique)
            if (_nodes := [row[i] for i in range(len(col)) if col[i] == id])
        ]
        subgraph_node_mems, subgraph_edge_mems, subgraph_props = (
            [
                [id, graph, node_id]
                for id, graph, _nodes, _, _ in subgraphs_list
                for node_id in _nodes
            ],
            [
                [id, graph, edge]
                for id, graph, _, _edges, _ in subgraphs_list
                for edge in _edges
            ],
            [[id, graph, f] for id, graph, _, _, f in subgraphs_list],
        )

        # Get tree edges
        t = data[index].tree_edge_index.tolist()
        tst, ten = t[0], t[1]
        subgraph_edges_list = [
            [id, graph, s, t] for id, (s, t) in enumerate(zip(tst, ten))
        ]

        nodes.extend(nodes_list)
        edges.extend(edges_list)
        subgraph_nodes.extend(subgraph_node_mems)
        subgraph_edgel.extend(subgraph_edge_mems)
        subgraphs.extend(subgraph_props)
        subgraph_edges.extend(subgraph_edges_list)

    nodes_df = pd.DataFrame(nodes, columns=["atom_id", "graph_id", "atom_features"])
    nodes_df.to_csv(f"{args.dir}/zinc_atoms{N_str}.csv", index=False)

    edges_df = pd.DataFrame(
        edges,
        columns=["bond_id", "graph_id", "start_atom", "end_atom", "bond_features"],
    )
    edges_df.to_csv(f"{args.dir}/zinc_bonds{N_str}.csv", index=False)

    subgraphs_df = pd.DataFrame(
        subgraphs,
        columns=["clique_id", "graph_id", "clique_features"],
    )
    subgraphs_df.to_csv(f"{args.dir}/zinc_cliques{N_str}.csv", index=False)

    subgraph_node_membership_df = pd.DataFrame(
        subgraph_nodes,
        columns=["clique_id", "graph_id", "atom_id"],
    )
    subgraph_node_membership_df.to_csv(
        f"{args.dir}/zinc_clique_membership{N_str}.csv", index=False
    )

    subgraph_edge_membership_df = pd.DataFrame(
        subgraph_edgel,
        columns=["clique_id", "graph_id", "bond_id"],
    )
    subgraph_edge_membership_df.to_csv(
        f"{args.dir}/zinc_clique_bond_membership{N_str}.csv", index=False
    )

    subgraph_edges_df = pd.DataFrame(
        subgraph_edges,
        columns=["tree_edge_id", "graph_id", "start_clique", "end_clique"],
    )
    subgraph_edges_df.to_csv(f"{args.dir}/zinc_tree_edges{N_str}.csv", index=False)


def get_mag10(debug: bool = False) -> None:
    """
    Create a hypergraph based on the MAG-10 dataset and write the results into multiple CSV files.

    @param debug: Debug mode. Defaults to False.
    """
    nodes = set()

    # Parse hyperedges
    hyperedges = []
    with open(f"{args.root}/hyperedges.txt", "r") as f:
        for row in f:
            hyperedges.append(list(map(int, row.strip().split("\t"))))
            for node in hyperedges[-1]:
                nodes.add(node)

    labels = []
    with open(f"{args.root}/hyperedge-labels.txt", "r") as f:
        for row in f:
            labels.append(int(row.strip()))

    label_id = []
    with open(f"{args.root}/hyperedge-label-identities.txt", "r") as f:
        for row in f:
            label_id.append(row.strip())

    # Form nodes
    nodes = [id for id in nodes]

    # Form hyperedges
    hyperedges_properties, hyperedges_labels = [], []
    for label in range(len(label_id)):
        hyperedges_properties.append([])
        hyperedges_labels.append([])

    for i, label in enumerate(labels):
        hyperedges_properties[label - 1].append(i)
        for node in hyperedges[i]:
            hyperedges_labels[label - 1].append((node, i))

    if not (debug):
        nodes_df = pd.DataFrame(nodes, columns=["id"])
        nodes_df.to_csv(f"{args.dir}/mag10_nodes.csv", index=False)
        for i, label in enumerate(label_id):
            hyperedges_df = pd.DataFrame(
                hyperedges_labels[i], columns=["personid", "hyperedge"]
            )
            hyperedges_df.to_csv(
                f"{args.dir}/mag10_hyperedges_{label}.csv", index=False
            )
            hyperedges_properties_df = pd.DataFrame(
                hyperedges_properties[i], columns=["id"]
            )
            hyperedges_properties_df.to_csv(
                f"{args.dir}/mag10_hyperedges_properties_{label}.csv", index=False
            )

    else:
        # Form nodes
        nodes_d = nodes[:200]
        lt = nodes_d[-1]
        nodes_df = pd.DataFrame(nodes_d, columns=["id"])
        nodes_df.to_csv(f"{args.dir}/mag10_nodes.csv", index=False)

        # Form hyperedges
        hyperedges = [
            (id, [node for node in hyperedge if node < lt])
            for id, hyperedge in enumerate(hyperedges)
        ]
        hyperedges = [(id, edge) for id, edge in hyperedges if len(edge) > 1]

        hyperedges_properties_d, hyperedges_labels_d = [], []
        for label in range(len(label_id)):
            hyperedges_properties_d.append([])
            hyperedges_labels_d.append([])

        for id, edge in hyperedges:
            label = labels[id]
            hyperedges_properties_d[label - 1].append(id)
            for node in edge:
                hyperedges_labels_d[label - 1].append((node, id))

        for i, label in enumerate(label_id):
            hyperedges_df = pd.DataFrame(
                hyperedges_labels_d[i], columns=["personid", "hyperedge"]
            )
            hyperedges_df.to_csv(
                f"{args.dir}/mag10_hyperedges_{label}.csv", index=False
            )
            hyperedges_properties_df = pd.DataFrame(
                hyperedges_properties_d[i], columns=["id"]
            )
            hyperedges_properties_df.to_csv(
                f"{args.dir}/mag10_hyperedges_properties_{label}.csv", index=False
            )


def get_rand_tuple(n: int = 20000, m: int = 50) -> None:
    """
    Generate Node-Tuple Collection based on a Barabási-Albert graph with n nodes and a degree of m
    for each node and write the results into multiple CSV files.

    @param n: Number of nodes in the graph.
    @param m: Degree for each node.
    """
    G = nx.barabasi_albert_graph(n=n, m=m)
    print(
        f"Generated BA graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
    )

    node_range = range(G.number_of_nodes())
    nodes = [(id, random.uniform(0, 5)) for id in node_range]
    ins = 19000
    edges_l = list(G.edges())
    edges = [
        [id, u, v, random.uniform(5, 10)]
        for id, (u, v) in enumerate(edges_l)
        if u < ins and v < ins
    ]

    edges_upd = [
        [id, u, v, random.uniform(5, 10)]
        for id, (u, v) in enumerate(edges_l)
        if u >= ins or v >= ins
    ]

    df_nodes = pd.DataFrame(nodes, columns=["id", "node_feature"])
    df_edges = pd.DataFrame(edges, columns=["id", "source", "target", "edge_feature"])
    df_edges_upd = pd.DataFrame(
        edges_upd, columns=["id", "source", "target", "edge_feature"]
    )

    df_nodes.to_csv(f"{args.dir}/rgt_nodes.csv", index=False)
    df_edges.to_csv(f"{args.dir}/rgt_edges.csv", index=False)
    df_edges_upd.to_csv(f"{args.dir}/rgt_edges_upd.csv", index=False)

    degrees = [d for _, d in G.degree()]
    sum_degrees = sum(degrees)
    sum_degrees_ins = sum(degrees[:ins])
    ins_probabs = [d / sum_degrees_ins for d in degrees[:ins]]
    probabs = [d / sum_degrees for d in degrees]

    tuple_sizes = list(map(int, np.random.lognormal(mean=3.8, sigma=0.95, size=5600)))
    tuples = [
        [
            id,
            np.random.choice(node_range[:ins], size=sz, p=ins_probabs),
            random.uniform(10, 20),
        ]
        for id, sz in enumerate(tuple_sizes[:4000])
    ]
    tuples_upd = [
        [
            id,
            ";".join(map(str, np.random.choice(node_range, size=sz, p=probabs))),
            random.uniform(10, 20),
        ]
        for id, sz in enumerate(tuple_sizes[4000:])
    ]

    tuple_edges, tuple_properties = [
        [id, node] for id, _nodes, _ in tuples for node in _nodes
    ], [[id, feat] for id, _, feat in tuples]

    tuple_edges_df = pd.DataFrame(tuple_edges, columns=["id", "node_id"])
    tuple_edges_df.to_csv(f"{args.dir}/rgt_tuple_edges.csv", index=False)
    tuple_properties_df = pd.DataFrame(
        tuple_properties, columns=["id", "tuple_feature"]
    )
    tuple_properties_df.to_csv(f"{args.dir}/rgt_tuple_properties.csv", index=False)
    tuple_upd_df = pd.DataFrame(tuples_upd, columns=["id", "nodes", "tuple_feature"])
    tuple_upd_df.to_csv(f"{args.dir}/rgt_tuples_upd.csv", index=False)


def get_rand_hypergraph(
    num_edges: int,
    num_nodes: int,
    edge_size_range: Tuple[int, int] = (3, 10),
    edge_size_dist: Tuple[str, int] = ("peak_at", 20),
    sigmas: Tuple[int, int] = (2, 6),
    seed: int = None,
) -> None:
    """
    Generate a random hypergraph and write the results into multiple CSV files.

    @param num_edges: Number of edges in the initial graph.
    @param num_nodes: Number of nodes in the initial graph.
    @param edge_size_range: Range of edge sizes.
    @param edge_size_dist: Distribution or tuple describing edge size distribution.
    @param sigmas: Standard deviations for edge size distribution.
    @param seed: Seed for the random number generator. Defaults to None.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    min_size, max_size = edge_size_range

    def sample_edge_size() -> int:
        """
        Sample an edge size based on the provided edge_size_dist and edge_size_range.

        @return: Sampled edge size.
        """
        if callable(edge_size_dist):
            return edge_size_dist()
        if edge_size_dist[0] == "peak_at":
            values = np.arange(min_size, max_size + 1)
            peak = edge_size_dist[1]

            left = values[values <= peak]
            right = values[values > peak]

            left_probs = np.exp(-0.5 * ((left - peak) / sigmas[0]) ** 2)
            right_probs = np.exp(-0.5 * ((right - peak) / sigmas[1]) ** 2)

            probs = np.concatenate([left_probs, right_probs])
            probs /= probs.sum()

            return int(np.random.choice(values, p=probs))

        return random.randint(min_size, max_size)

    edges = {}
    node_degrees = {}
    node_id = 0

    init_size = sample_edge_size()
    initial_nodes = list(range(node_id, node_id + init_size))
    edges[0] = set(initial_nodes)

    for node in initial_nodes:
        node_degrees[node] = 1
    node_id += init_size
    edge_id = 1

    degree_list = []
    for node in initial_nodes:
        node_degrees[node] = 1
        degree_list.append(node)

    node_id += init_size
    edge_id = 1

    while edge_id < num_edges and node_id < num_nodes:
        edge_size = sample_edge_size()
        new_node = node_id
        node_id += 1

        if len(degree_list) < edge_size - 1:
            sampled = degree_list
        else:
            sampled = random.choices(degree_list, k=edge_size - 1)

        edge_nodes = set(sampled + [new_node])
        edges[edge_id] = edge_nodes

        for n in edge_nodes:
            node_degrees[n] = node_degrees.get(n, 0) + 1
            degree_list.append(n)

        edge_id += 1

    H = hnx.Hypergraph(edges)

    edge_sizes = H.edge_size_dist()
    nodes_to_insert_amount = 0.05

    nodes = list(H.nodes)
    num_to_insert = max(1, int(nodes_to_insert_amount * len(nodes)))

    node_weights = []
    for node in nodes:
        degree = len(H.nodes[node])
        weight = 1 / (degree + 1e-6)  # add epsilon to avoid division by zero
        node_weights.append(weight)

    total_weight = sum(node_weights)
    node_probs = [w / total_weight for w in node_weights]

    nodes_to_insert = set()
    while len(nodes_to_insert) < num_to_insert:
        sampled = random.choices(nodes, weights=node_probs, k=num_to_insert)
        nodes_to_insert.update(sampled)
    nodes_to_insert = list(nodes_to_insert)[:num_to_insert]
    edges_to_insert = set()
    for node in nodes_to_insert:
        edges_to_insert.update(H.nodes[node])

    # Write node CSV
    nodes_rows = [[node, random.uniform(0, 10)] for node in nodes]
    nodes_df = pd.DataFrame(nodes_rows, columns=["id", "node_feature"])
    nodes_df.to_csv(f"{args.dir}/rhg_nodes.csv", index=False)

    ins_edges = set(H.edges) - edges_to_insert
    edges_rows = [[node, edge] for edge in ins_edges for node in H.edges[edge]]
    edges_upd_rows = [
        [random.uniform(0, 10), ";".join(map(str, H.edges[edge])), edge]
        for edge in edges_to_insert
    ]
    edges_df = pd.DataFrame(edges_rows, columns=["node_id", "edge_id"])
    edges_df.to_csv(f"{args.dir}/rhg_edges.csv", index=False)
    edges_upd_df = pd.DataFrame(
        edges_upd_rows, columns=["edge_feature", "nodes", "edge_id"]
    )
    edges_upd_df.to_csv(f"{args.dir}/rhg_edges_upd.csv", index=False)

    edge_prop_rows = [[edge, random.uniform(0, 10)] for edge in ins_edges]
    edges_df = pd.DataFrame(edge_prop_rows, columns=["id", "feature"])
    edges_df.to_csv(f"{args.dir}/rhg_edge_properties.csv", index=False)


if __name__ == "__main__":
    assert args.mode in ["debug", "full"], "Invalid mode. Use 'debug' or 'full'."
    assert args.data in [
        "zinc",
        "mag10",
        "random_t",
        "random_h",
    ], "Invalid dataset. Use 'zinc', 'mag10', 'random_t', or 'random_h'."

    if args.data != "zinc":
        assert not args.weak, "--weak is exclusive to the 'zinc' dataset."

    if args.data == "zinc":
        transform = JunctionTree()
        data = ZINC(root=args.root, subset=True, split="test", pre_transform=transform)
        if args.weak:
            for N in [20, 40, 80, 160, 320]:
                get_zinc(data[0:N], N, args.dir)
        else:
            N = 400 if args.mode == "full" else 10
            get_zinc(data[0:N], N)

    elif args.data == "mag10":
        get_mag10(debug=(args.mode == "debug"))

    elif args.data == "random_t":
        get_rand_tuple()

    elif args.data == "random_h":
        get_rand_hypergraph(
            num_edges=50000,
            num_nodes=100000,
        )
