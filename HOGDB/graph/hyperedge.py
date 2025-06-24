# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Shriram Chandran

from HOGDB.db.label import Label
from HOGDB.db.property import Property
from HOGDB.graph.graph_element import GraphElement
from HOGDB.graph.node import Node
from typing import List


class HyperEdge(GraphElement):
    """
    A HyperEdge is a graph element that connects multiple nodes.
    It can have properties and a label.
    """

    def __init__(
        self,
        nodes: List[Node] = None,
        label: Label = Label("_hyperedge"),
        properties: List[Property] = None,
    ):
        """
        Initialize a HyperEdge with a list of nodes, a label, and properties.

        @param nodes: The list of nodes connected by the HyperEdge. Defaults to None.
        @param label: The label of the HyperEdge. Defaults to a label object with the string "_hyperedge".
        @param properties: The properties of the HyperEdge. Defaults to None.
        """
        self.nodes = nodes if nodes is not None else []
        self.label = label
        self.properties = properties if properties is not None else []

    def __repr__(self) -> str:
        """
        Return a string representation of the HyperEdge.

        @return: The string representation of the HyperEdge.
        """
        properties_str = self.generate_properties_string(self.properties)
        repr_str = ""
        for node in self.nodes:
            repr_str += f"({repr(node)}),"
        repr_str = repr_str[:-1]
        repr_str += f"->(:{repr(self.label)}{properties_str})"
        return repr_str

    def __eq__(self, other) -> bool:
        """
        Check if two HyperEdges are equal.

        @param other: The other HyperEdge to compare.
        @return: True if the HyperEdges are equal, False otherwise.
        """
        if not isinstance(other, HyperEdge):
            return False
        return (
            self.nodes == other.nodes
            and self.label == other.label
            and self.properties == other.properties
        )

    def __getitem__(self, item) -> any | None:
        """
        Get the value of a property by its key.

        @param item: The key of the property.
        @return: The value of the property, or None if not found.
        """
        return next((prop.value for prop in self.properties if prop.key == item), None)
