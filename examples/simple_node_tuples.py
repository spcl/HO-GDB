# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Shriram Chandran

from HOGDB.db.neo4j import Neo4jDatabase
from HOGDB.graph.graph_with_tuple_storage import *


def main() -> None:
    """
    Simple example code with a higher-order graph of a few nodes with a node-tuple.
    """
    # initialize database and graph storage
    db = Neo4jDatabase()
    gs = GraphwithTupleStorage(db)

    # clear the graph
    gs.clear_graph()

    # add nodes
    alice = Node(
        [Label("Person")],
        [Property("name", str, "Alice"), Property("role", str, "Engineer")],
    )
    bob = Node(
        [Label("Person")],
        [Property("name", str, "Bob"), Property("role", str, "Designer")],
    )
    carol = Node(
        [Label("Person")],
        [Property("name", str, "Carol"), Property("role", str, "Manager")],
    )
    gs.add_node(alice)
    gs.add_node(bob)
    gs.add_node(carol)

    # assert node count
    assert gs.get_node_count([Label("Person")]) == 3

    # add a node tuple (collaboration group)
    collaboration = NodeTuple(
        [alice, bob, carol],
        [Label("Collaboration")],
        [Property("project", str, "ProjectX"), Property("year", int, 2023)],
    )
    gs.add_node_tuple(collaboration)

    # assert node tuple count
    assert gs.get_node_tuple_count() == 1

    # retrieve and assert the node tuple
    retrieved_collaboration = gs.get_node_tuple(
        NodeTuple([], [], [Property("project", str, "ProjectX")])
    )
    assert retrieved_collaboration == collaboration

    # update the node tuple
    gs.update_node_tuple(
        NodeTuple([], [], [Property("project", str, "ProjectX")]),
        [Property("project", str, "ProjectY"), Property("year", int, 2024)],
    )

    # retrieve and assert the updated node tuple
    updated_collaboration = gs.get_node_tuple(
        NodeTuple([], [], [Property("project", str, "ProjectY")])
    )
    assert updated_collaboration == NodeTuple(
        [alice, bob, carol],
        [Label("Collaboration")],
        [Property("project", str, "ProjectY"), Property("year", int, 2024)],
    )

    # clean up
    gs.close_connection()
    print("Node tuple example executed successfully!")


if __name__ == "__main__":
    main()
