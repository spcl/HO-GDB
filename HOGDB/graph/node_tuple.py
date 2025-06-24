# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Shriram Chandran

from HOGDB.db.label import Label
from HOGDB.db.property import Property
from HOGDB.graph.node import Node
from typing import List


class NodeTuple:
    """
    A NodeTuple represents an ordered tuple of nodes.
    It can have labels and properties.
    """

    def __init__(
        self,
        nodes: List[Node] = None,
        labels: List[Label] = None,
        properties: List[Property] = None,
    ) -> None:
        """
        Initialize a NodeTuple with a list of nodes, labels, and properties.

        @param nodes: A list of nodes in the node-tuple. Defaults to None.
        @param labels: A list of labels for the node-tuple. Defaults to None.
        @param properties: A list of properties for the node-tuple. Defaults to None.
        """
        self.nodes = nodes if nodes is not None else []
        self.labels = labels if labels is not None else []
        self.properties = properties if properties is not None else []

    def nodes_repr(self) -> str:
        """
        Return a string representation of the nodes in the NodeTuple.

        @return: A list of string representations of the nodes.
        """
        return [repr(node) for node in self.nodes]

    def __repr__(self) -> str:
        """
        Return a string representation of the NodeTuple.

        @return: A string representation of the NodeTuple.
        """
        labels_str = (
            ":" + ":".join(repr(label) for label in self.labels)
            if self.labels != None
            else ""
        )
        properties_str = (
            " {" + ", ".join([repr(property) for property in self.properties]) + "}"
            if self.properties != None
            else ""
        )
        return f"{labels_str}" + f"{properties_str}"

    def __eq__(self, value) -> bool:
        """
        Check if two NodeTuples are equal.

        @param value: The NodeTuple to compare with.
        @return: True if the NodeTuples are equal, False otherwise.
        """
        if not isinstance(value, NodeTuple):
            return False
        return (
            (self.nodes == value.nodes or self.nodes == None or value.nodes == None)
            and self.labels == value.labels
            and self.properties == value.properties
        )

    def __getitem__(self, item) -> any | None:
        """
        Get the value of a property by its key.

        @param item: The key of the property.
        @return: The value of the property, or None if not found.
        """
        return next((prop.value for prop in self.properties if prop.key == item), None)
