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
from HOGDB.graph.edge import Edge
from HOGDB.graph.graph_element import GraphElement
from typing import List, Optional


class Subgraph(GraphElement):
    """
    A Subgraph is a graph element that can contain nodes and edges.
    It can also have labels and properties.
    """

    def __init__(
        self,
        subgraph_nodes: List[Node] = None,
        subgraph_edges: List[Edge] = None,
        labels: List[Label] = None,
        properties: List[Property] = None,
    ) -> None:
        """
        Initialize a Subgraph with a list of nodes, edges, labels, and properties.

        @param subgraph_nodes: A list of nodes in the subgraph. Defaults to None.
        @param subgraph_edges: A list of edges in the subgraph. Defaults to None.
        @param labels: A list of labels for the subgraph. Defaults to None.
        @param properties: A list of properties for the subgraph. Defaults to None.
        """
        self.subgraph_nodes = subgraph_nodes if subgraph_nodes is not None else []
        self.subgraph_edges = subgraph_edges if subgraph_edges is not None else []
        self.labels = labels if labels is not None else []
        self.properties = properties if properties is not None else []

    def nodes_repr(self) -> str:
        """
        Return a string representation of the nodes in the Subgraph.

        @return: A list of string representations of the nodes.
        """
        return [repr(node) for node in self.subgraph_nodes]

    def edges_repr(self) -> str:
        """
        Return a string representation of the edges in the Subgraph.

        @return: A list of string representations of the edges.
        """
        return [repr(edge) for edge in self.subgraph_edges]

    def __repr__(self) -> str:
        """
        Return a string representation of the Subgraph.

        @return: A string representation of the Subgraph.
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
        Check if two Subgraphs are equal.

        @param value: The Subgraph to compare with.
        @return: True if the Subgraphs are equal, False otherwise.
        """
        if not isinstance(value, Subgraph):
            return False
        return self.labels == value.labels and self.properties == value.properties

    def __getitem__(self, item) -> any | None:
        """
        Get the value of a property by its key.

        @param item: The key of the property.
        @return: The value of the property, or None if not found.
        """
        return next((prop.value for prop in self.properties if prop.key == item), None)


class SubgraphEdge(GraphElement):
    """
    A SubgraphEdge is a graph element that connects two subgraphs.
    It can have properties and a label.
    """

    def __init__(
        self,
        start_subgraph: Subgraph = Subgraph(),
        end_subgraph: Subgraph = Subgraph(),
        label: Label = Label("_subgraph_edge"),
        properties: List[Property] = None,
    ) -> None:
        """
        Initialize a SubgraphEdge with a start subgraph, an end subgraph,
        a label, and properties.

        @param start_subgraph: The starting subgraph of the subgraph edge. Defaults to an empty Subgraph object.
        @param end_subgraph: The ending subgraph of the subgraph edge. Defaults to an empty Subgraph object.
        @param label: The label of the subgraph edge. Defaults to a label object with the string "_subgraph_edge".
        @param properties: A list of properties for the subgraph edge. Defaults to None.
        """
        self.start_subgraph = start_subgraph
        self.end_subgraph = end_subgraph
        self.label = label
        self.properties = properties if properties is not None else []

    def start_subgraph_repr(self) -> str:
        """
        Return a string representation of the start subgraph in the SubgraphEdge.

        @return: A string representation of the start subgraph.
        """
        return repr(self.start_subgraph)

    def end_subgraph_repr(self) -> str:
        """
        Return a string representation of the end subgraph in the SubgraphEdge.

        @return: A string representation of the end subgraph.
        """
        return repr(self.end_subgraph)

    def __repr__(self) -> str:
        """
        Return a string representation of the SubgraphEdge.

        @return: A string representation of the SubgraphEdge.
        """
        label_str = repr(self.label)
        properties_str = (
            " {" + ", ".join([repr(property) for property in self.properties]) + "}"
            if self.properties != None
            else ""
        )
        return f":{label_str}" + f"{properties_str}"

    def __eq__(self, value) -> bool:
        """
        Check if two SubgraphEdges are equal.

        @param value: The SubgraphEdge to compare with.
        @return: True if the SubgraphEdges are equal, False otherwise.
        """
        if not isinstance(value, SubgraphEdge):
            return False
        return (
            self.start_subgraph == value.start_subgraph
            and self.end_subgraph == value.end_subgraph
            and self.label == value.label
            and self.properties == value.properties
        )

    def __getitem__(self, item) -> any | None:
        """
        Get the value of a property by its key.

        @param item: The key of the property.
        @return: The value of the property, or None if not found.
        """
        return next((prop.value for prop in self.properties if prop.key == item), None)
