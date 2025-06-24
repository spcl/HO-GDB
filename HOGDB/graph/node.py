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
from typing import List


class Node(GraphElement):
    """
    A class representing a node in a property graph.
    A node can have multiple labels and properties.
    """

    def __init__(
        self,
        labels: List[Label] = None,
        properties: List[Property] = None,
    ) -> None:
        """
        Initialize the Node instance.

        @param labels: A list of labels for the node. Defaults to None.
        @param properties: A list of properties for the node. Defaults to None.
        """
        self.labels = labels if labels is not None else []
        self.properties = properties if properties is not None else []

    def __repr__(self) -> str:
        """
        Return a string representation of the Node instance.

        @return: A string representation of the node.
        """
        labels_str = self.generate_labels_string(self.labels)
        properties_str = self.generate_properties_string(self.properties)
        return f"{labels_str}" + f"{properties_str}"

    def __eq__(self, other) -> bool:
        """
        Check if two Node instances are equal.

        @param other: Another Node instance to compare.
        @return: True if the nodes are equal, False otherwise.
        """
        if not isinstance(other, Node):
            return False
        return self.labels == other.labels and self.properties == other.properties

    def __getitem__(self, item) -> any | None:
        """
        Get the value of a property by its key.

        @param item: The key of the property.
        @return: The value of the property, or None if not found.
        """
        return next((prop.value for prop in self.properties if prop.key == item), None)
