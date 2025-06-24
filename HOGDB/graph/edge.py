# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Shriram Chandran
#
# contributions: Jakub Cudak

from HOGDB.db.label import Label
from HOGDB.db.property import Property
from HOGDB.graph.graph_element import GraphElement
from HOGDB.graph.node import Node
from typing import List


class Edge(GraphElement):
    """
    A class representing an edge in a property graph.
    An edge connects two nodes and can have a label and properties.
    """

    def __init__(
        self,
        start_node: Node = Node(),
        end_node: Node = Node(),
        label: Label = Label("_edge"),
        properties: List[Property] = None,
    ) -> None:
        """
        Initialize the Edge instance.

        @param start_node: The starting node of the edge. Defaults to an empty Node object.
        @param end_node: The ending node of the edge. Defaults to an empty Node object.
        @param label: The label of the edge. Defaults to a label object with the string "_edge".
        @param properties: The properties of the edge. Defaults to None.
        """
        self.start_node = start_node
        self.end_node = end_node
        self.label = label
        self.properties = properties if properties is not None else []

    def __repr__(self) -> str:
        """
        Return a string representation of the Edge instance.

        @return: A string representation of the edge.
        """
        properties_str = self.generate_properties_string(self.properties)
        return f"({repr(self.start_node)})-(:{repr(self.label)}{properties_str})->({repr(self.end_node)})"

    def __eq__(self, other) -> bool:
        """
        Check if two Edge instances are equal.

        @param other: The other edge to compare.
        @return: True if the edges are equal, False otherwise.
        """
        if not isinstance(other, Edge):
            return False
        return (
            self.start_node == other.start_node
            and self.end_node == other.end_node
            and self.label == other.label
            and self.properties == other.properties
        )

    def __getitem__(self, item: str) -> any | None:
        """
        Get the value of a property by its key.

        @param item: The key of the property.
        @return: The value of the property, or None if not found.
        """
        return next((prop.value for prop in self.properties if prop.key == item), None)
