# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Shriram Chandran

import torch
from torch_geometric.datasets import ZINC
from torch_geometric.data import Data, Dataset, DataLoader

from rdkit.Chem.rdchem import BondType

from HOGDB.db.neo4j import Neo4jDatabase
from HOGDB.db.label import Label
from HOGDB.db.schema import Schema
from HOGDB.graph.graph_with_subgraph_storage import GraphwithSubgraphStorage
from HOGDB.graph.path import *
from model import Net

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import time
import argparse

from utils import JunctionTree, JunctionTreeData, mol_from_data


db_params = {
    "db_uri": "bolt://localhost:7687",
    "db_username": "neo4j",
    "db_password": "password",
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--no_train", action="store_true", help="Skip training and exit after data import."
)
parser.add_argument(
    "--device", type=int, default=0, help="CUDA device ID to use for training."
)
parser.add_argument(
    "--mode", type=str, default="debug", help="Mode of operation: 'debug' or 'full'."
)
parser.add_argument(
    "--root",
    type=str,
    default="datasets/zinc",
    help="Root directory containing the raw ZINC dataset.",
)
parser.add_argument(
    "--hidden_channels",
    type=int,
    default=128,
    help="Number of hidden channels in the GNN.",
)
parser.add_argument(
    "--num_layers", type=int, default=3, help="Number of layers in the GNN."
)
parser.add_argument(
    "--dropout", type=float, default=0.0, help="Dropout rate for the GNN."
)
parser.add_argument(
    "--epochs", type=int, default=300, help="Number of training epochs."
)
parser.add_argument(
    "--no_inter_message_passing",
    action="store_true",
    help="Disable inter-message passing in the GNN.",
)
parser.add_argument(
    "--online", action="store_true", help="Use online CSV URLs for data import."
)
parser.add_argument(
    "--csv_url", type=str, default="", help="Base URL for online CSV directory."
)
args = parser.parse_args()
print(args)

bonds = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]


db = Neo4jDatabase(**db_params)
gs = GraphwithSubgraphStorage(db)
gs.clear_graph()
gs.close_connection()

if args.mode == "full":
    N = 400
elif args.mode == "debug":
    N = 10
else:
    raise ValueError("Invalid mode. Use 'full' or 'debug'.")

transform = JunctionTree()
train_data = ZINC(root=args.root, subset=True, split="train", pre_transform=transform)
val_data = ZINC(root=args.root, subset=True, split="val", pre_transform=transform)
test_data = ZINC(root=args.root, subset=True, split="test", pre_transform=transform)
test_y = test_data[0:N].y

db = Neo4jDatabase(**db_params)
session = db.start_session()
start = time.perf_counter()
db.create_index(session, Label("Atom"), ["graph_id", "atom_id"])
db.create_index(session, Label("Bond"), ["graph_id", "bond_id"])
db.create_index(session, Label("Clique"), ["graph_id", "clique_id"])
db.create_index(session, Label("TreeEdge"), ["graph_id", "tree_edge_id"])
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


def query_data(gs: GraphwithSubgraphStorage) -> Data:
    """
    Queries graph data as a trainable torch_geometric data object.

    @param gs: A graph with subgraph collections.
    @return: Graph data as a torch_geometric data object.
    """
    # x
    path_x = Path()
    path_x.add(Node([Label("Atom")]), "a")
    data = JunctionTreeData()
    df = gs.traverse_path(
        [path_x], return_values=["a.atom_features"], sort=["a.graph_id", "a.atom_id"]
    )
    data.x = torch.tensor(df["a.atom_features"].tolist()).unsqueeze(1)

    # edge_index, edge_attr
    path_ei = Path()
    path_ei.add(Node([Label("Atom")]), "st")
    path_ei.add(Edge(label=Label("Bond")), "e")
    path_ei.add(Node([Label("Atom")]), "en")
    df = gs.traverse_path(
        [path_ei],
        return_values=["st.atom_id", "en.atom_id", "e.bond_features"],
        sort=["st.graph_id", "st.atom_id", "en.atom_id"],
    )
    data.edge_index = torch.tensor(df[["st.atom_id", "en.atom_id"]].T.values)
    data.edge_attr = torch.tensor(df["e.bond_features"].tolist())

    # tree_edge_index
    path_te = Path()
    path_te.add(Subgraph(labels=[Label("Clique")]), "st")
    path_te.add(SubgraphEdge(label=Label("TreeEdge")), "e")
    path_te.add(Subgraph(labels=[Label("Clique")]), "en")
    df = gs.traverse_path(
        [path_te],
        return_values=["st.clique_id", "en.clique_id"],
        sort=["st.graph_id", "st.clique_id", "en.clique_id"],
    )
    data.tree_edge_index = torch.tensor(df[["st.clique_id", "en.clique_id"]].T.values)

    # x_clique
    path_x_clique = Path()
    path_x_clique.add(Subgraph(labels=[Label("Clique")]), "c")
    df = gs.traverse_path(
        [path_x_clique],
        return_values=["c.clique_features"],
        sort=["c.graph_id", "c.clique_id"],
    )
    data.x_clique = torch.tensor(df["c.clique_features"].tolist())

    # num_cliques
    path_num_cliques = Path()
    path_num_cliques.add(Subgraph(labels=[Label("Clique")]), "c")
    df = gs.traverse_path(
        [path_num_cliques],
        return_values=["c.graph_id", "count(c)"],
        sort=["c.graph_id"],
    )
    data.num_cliques = torch.tensor(df["count(c)"].tolist())

    # atom2clique_index
    path_ac = Path()
    path_ac.add(Node([Label("Atom")]), "a")
    path_ac.add(Subgraph(labels=[Label("Clique")]), "c")
    df = gs.traverse_path(
        [path_ac],
        return_values=["a.atom_id", "c.clique_id"],
        sort=["a.graph_id", "a.atom_id", "c.clique_id"],
    )
    data.atom2clique_index = torch.tensor(df[["a.atom_id", "c.clique_id"]].T.values)
    return data


start = time.perf_counter()
gs = GraphwithSubgraphStorage(db)
test_dataset = query_data(gs)
end = time.perf_counter()
torch_time = end - start
print(f"Converted HO-GDB data to torch data in {torch_time:.4f} seconds.")


if args.mode == "debug":
    test_data._data = test_dataset
    test_data._data.y = test_y
    for data in [train_data, val_data, test_data]:
        for key, _ in data.slices.items():
            data.slices[key] = data.slices[key][: N + 1]
    for data in [train_data, val_data, test_data]:
        data = data[:N]
else:
    test_dataset.y = test_y
    for data_el in test_dataset.keys():
        el = test_dataset[data_el]
        if data_el in ["x", "x_clique", "edge_attr", "num_cliques", "y"]:
            rem_el = test_data._data[data_el][test_data.slices[data_el][N] :]
            el = torch.cat((el, rem_el), dim=0)
        else:
            rem_el = test_data._data[data_el][:, test_data.slices[data_el][N] :]
            el = torch.cat((el, rem_el), dim=1)
        assert torch.equal(test_data._data[data_el], el)
        test_data._data[data_el] = el

# Run model
if args.mode == "debug":
    train_loader = DataLoader(train_data, 20, shuffle=False, num_workers=1)
    val_loader = DataLoader(val_data, 20, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_data, 20, shuffle=False, num_workers=1)
else:
    train_loader = DataLoader(train_data, 128, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_data, 1000, shuffle=False, num_workers=12)
    test_loader = DataLoader(test_data, 1000, shuffle=False, num_workers=12)

device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
model = Net(
    hidden_channels=args.hidden_channels,
    out_channels=1,
    num_layers=args.num_layers,
    dropout=args.dropout,
    inter_message_passing=not args.no_inter_message_passing,
).to(device)

if args.no_train:
    exit()


def train() -> float:
    """
    Train the model.

    @return: Loss in one epoch.
    """
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = (model(data).squeeze() - data.y).abs().mean()
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader: Dataset) -> float:
    """
    Test the model.

    @param loader: Torch dataset loader.
    @return: Test loss.
    """
    model.eval()

    total_error = 0
    for data in loader:
        data = data.to(device)
        total_error += (model(data).squeeze() - data.y).abs().sum().item()
    return total_error / len(loader.dataset)


run_times = []
test_maes = []
num_epochs = args.epochs if args.mode == "full" else 10
for run in range(0, 1):
    print()
    print(f"Run {run}:")
    print()
    start = time.perf_counter()

    model.reset_parameters()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=0.00001
    )

    best_val_mae = test_mae = float("inf")
    for epoch in range(1, num_epochs + 1):
        lr = scheduler.optimizer.param_groups[0]["lr"]
        loss = train(epoch)
        val_mae = test(val_loader)
        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            test_mae = test(test_loader)

        print(
            f"Epoch: {epoch:03d}, LR: {lr:.5f}, Loss: {loss:.4f}, "
            f"Val: {val_mae:.4f}, Test: {test_mae:.4f}"
        )

    test_maes.append(test_mae)
    end = time.perf_counter()
    print(f"Run {run} time: {end - start:.4f} seconds")
    run_times.append(end - start)

test_mae = torch.tensor(test_maes)
print("===========================")
print(f"Final Test: {test_mae.mean():.4f} ± {test_mae.std():.4f}")
print(f"Total time: {sum(run_times):.4f} seconds")
print(f"Converted HO-GDB to torch data in {torch_time:.4f} seconds.")
