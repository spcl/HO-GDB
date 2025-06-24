# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Shriram Chandran

import random

import torch
from torch_geometric.data import Data
from torch_geometric.utils import tree_decomposition
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from typing import List, Tuple

bonds = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]


def mol_from_data(data: Data) -> Chem.Mol:
    """
    Get molecule data from the ZINC dataset.

    @param data: Raw data (torch_geometric.data.Data).
    @return: Molecule data (rdkit.Chem.Mol).
    """
    mol = Chem.RWMol()

    x = data.x if data.x.dim() == 1 else data.x[:, 0]
    for z in x.tolist():
        mol.AddAtom(Chem.Atom(z))

    row, col = data.edge_index
    mask = row < col
    row, col = row[mask].tolist(), col[mask].tolist()

    bond_type = data.edge_attr
    bond_type = bond_type if bond_type.dim() == 1 else bond_type[:, 0]
    bond_type = bond_type[mask].tolist()

    for i, j, bond in zip(row, col, bond_type):
        assert bond >= 1 and bond <= 4
        mol.AddBond(i, j, bonds[bond - 1])

    return mol.GetMol()


class JunctionTreeData(Data):
    """
    Class to store the junction tree data.
    """

    def __inc__(self, key: str, item, *args) -> torch.Tensor:
        """
        Returns the increment value for batching.

        @param key: Key to decide which data to return.
        @param item: The item to increment.
        @param *args: Additional arguments.
        @return: Increment value for batching.
        """
        if key == "tree_edge_index":
            return self.x_clique.size(0)
        elif key == "atom2clique_index":
            return torch.tensor([[self.x.size(0)], [self.x_clique.size(0)]])
        else:
            return super(JunctionTreeData, self).__inc__(key, item, *args)


class JunctionTree(object):
    """
    Class for the junction tree object.
    """

    def __call__(self, data: Data) -> JunctionTreeData:
        """
        Extract molecule data from the ZINC dataset and create a junction tree for that data.

        @param data: ZINC data (torch_geometric.data.Data).
        @return: Junction tree (JunctionTreeData).
        """
        mol = mol_from_data(data)
        out = tree_decomposition(mol, return_vocab=True)
        tree_edge_index, atom2clique_index, num_cliques, x_clique = out

        data = JunctionTreeData(**{k: v for k, v in data})

        data.tree_edge_index = tree_edge_index
        data.atom2clique_index = atom2clique_index
        data.num_cliques = num_cliques
        data.x_clique = x_clique

        return data
