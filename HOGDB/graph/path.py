# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Shriram Chandran
#
# contributions: Jakub Cudak

from HOGDB.graph.graph_element import GraphElement
from HOGDB.graph.node import Node
from HOGDB.graph.edge import Edge
from HOGDB.graph.hyperedge import HyperEdge
from HOGDB.graph.subgraph import Subgraph, SubgraphEdge
from HOGDB.graph.node_tuple import NodeTuple
from HOGDB.db.label import Label
from typing import List


class PathElement:
    def __init__(self, element: GraphElement, variable: str = None) -> None:
        self.element = element
        self.variable = variable

    def __eq__(self, other) -> bool:
        if not isinstance(other, PathElement):
            return False
        return self.element == other.element and self.direction == other.direction


def structure_to_node(path_element: PathElement) -> PathElement:
    """
    Convert a graph element to a node representation.

    @param path_element: The path element containing the graph element.
    @return: A PathElement containing the node representation of the graph element.
    """
    el = path_element.element
    if isinstance(el, Node):
        node = Node([Label("_node")] + el.labels, el.properties)
    elif isinstance(el, Edge):
        node = Node([Label("_edge"), el.label], el.properties)
    elif isinstance(el, Subgraph):
        node = Node([Label("_subgraph")] + el.labels, el.properties)
    elif isinstance(el, SubgraphEdge):
        node = Node([Label("_subgraph_edge"), el.label], el.properties)
    elif isinstance(el, NodeTuple):
        node = Node([Label("_node_tuple")] + el.labels, el.properties)
    elif isinstance(el, HyperEdge):
        node = Node([Label("_hyperedge"), el.label], el.properties)
    else:
        raise ValueError(f"Unsupported graph element type: {type(el)}")
    return PathElement(node, path_element.variable)


class Path(GraphElement):
    def __init__(self, path: List[PathElement] = None) -> None:
        self.path = [] if path is None else path

    def add(self, element: PathElement) -> None:
        """
        Add a graph element to the path.

        @param element: The PathElement to add.
        """
        self.path.append(element)

    def add(self, element: GraphElement, variable: str = None) -> None:
        """
        Add a graph element to the path.

        @param element: The graph element to add (Node, Edge, Subgraph, etc.).
        @param variable: Optional variable name for the element.
        """
        self.path.append(PathElement(element, variable))

    def __repr__(self) -> str:
        """
        Return a string representation of the path.

        @return: A string representation of the path.
        """
        return f"Path({self.path})"

    def __eq__(self, other) -> bool:
        """
        Check equality of two paths.

        @param other: The other Path object to compare.
        @return: True if the paths are equal, False otherwise.
        """
        if not isinstance(other, Path):
            return False
        return self.path == other.path

    def read_as_path(self):
        """
        Read the path as a sequence of variables and elements.

        @return: A tuple containing a list of variables and a list of element details (labels and
                 properties).
        """
        return [el.variable for el in self.path], [
            (
                (
                    el.element.labels
                    if isinstance(el.element, Node)
                    else [el.element.label]
                ),
                el.element.properties,
            )
            for el in self.path
        ]

    def read_as_path_with_subgraphs(self):
        """
        Read the path as a sequence of variables and elements, including subgraph-specific edges.

        @return: A tuple containing a list of variables and a list of element details (labels and
                 properties).
        """
        path = [structure_to_node(self.path[0])]
        for i in range(1, len(self.path)):
            el, prev_el = self.path[i], self.path[i - 1]
            if isinstance(el.element, Node) and isinstance(prev_el.element, Edge):
                path.append(PathElement(Edge(None, None, Label("_adjacency"), [])))
            elif isinstance(el.element, Edge) and isinstance(prev_el.element, Node):
                path.append(PathElement(Edge(None, None, Label("_adjacency"), [])))
            elif isinstance(el.element, Subgraph):
                if isinstance(prev_el.element, Edge):
                    path.append(
                        PathElement(Edge(None, None, Label("_edge_membership"), []))
                    )
                elif isinstance(prev_el.element, Node):
                    path.append(
                        PathElement(Edge(None, None, Label("_node_membership"), []))
                    )
                elif isinstance(prev_el.element, SubgraphEdge):
                    path.append(
                        PathElement(Edge(None, None, Label("_subgraph_adjacency"), []))
                    )
                else:
                    raise ValueError(
                        f"Ill formed path: {prev_el.element} -> {el.element}"
                    )
            elif isinstance(el.element, SubgraphEdge) and isinstance(
                prev_el.element, Subgraph
            ):
                path.append(
                    PathElement(Edge(None, None, Label("_subgraph_adjacency"), []))
                )
            else:
                raise ValueError(f"Ill formed path: {prev_el.element} -> {el.element}")
            path.append(structure_to_node(el))
        return [el.variable for el in path], [
            (
                (
                    el.element.labels
                    if isinstance(el.element, Node)
                    else [el.element.label]
                ),
                el.element.properties,
            )
            for el in path
        ]

    def read_as_path_with_tuples(self):
        """
        Read the path as a sequence of variables and elements, including node-tuple-specific edges.

        @return: A tuple containing a list of variables and a list of element details (labels and
                 properties).
        """
        path = [structure_to_node(self.path[0])]
        for i in range(1, len(self.path)):
            el, prev_el = self.path[i], self.path[i - 1]
            if isinstance(el.element, Node) and isinstance(prev_el.element, Edge):
                path.append(PathElement(Edge(None, None, Label("_adjacency"), [])))
            elif isinstance(el.element, Edge) and isinstance(prev_el.element, Node):
                path.append(PathElement(Edge(None, None, Label("_adjacency"), [])))
            elif isinstance(el.element, NodeTuple) and isinstance(
                prev_el.element, Node
            ):
                path.append(
                    PathElement(Edge(None, None, Label("_node_membership"), []))
                )
            else:
                raise ValueError(f"Ill formed path: {prev_el.element} -> {el.element}")
            path.append(structure_to_node(el))
        return [el.variable for el in path], [
            (
                (
                    el.element.labels
                    if isinstance(el.element, Node)
                    else [el.element.label]
                ),
                el.element.properties,
            )
            for el in path
        ]

    def read_as_path_with_hypergraph(self):
        """
        Read the path as a sequence of variables and elements, including hypergraph-specific edges.

        @return: A tuple containing a list of variables and a list of element details (labels and
                 properties).
        """
        path = [structure_to_node(self.path[0])]
        for i in range(1, len(self.path)):
            el, prev_el = self.path[i], self.path[i - 1]
            if isinstance(el.element, HyperEdge) and isinstance(prev_el.element, Node):
                path.append(PathElement(Edge(None, None, Label("_adjacency"), [])))
            elif isinstance(el.element, Node) and isinstance(
                prev_el.element, HyperEdge
            ):
                path.append(PathElement(Edge(None, None, Label("_adjacency"), [])))
            else:
                raise ValueError(f"Ill formed path: {prev_el.element} -> {el.element}")
            path.append(structure_to_node(el))
        return [el.variable for el in path], [
            (
                (
                    el.element.labels
                    if isinstance(el.element, Node)
                    else [el.element.label]
                ),
                el.element.properties,
            )
            for el in path
        ]
