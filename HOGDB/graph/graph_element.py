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
from typing import List, Optional


class GraphElement:
    """
    A base class for graph elements (nodes and edges) in a property graph.
    """

    @staticmethod
    def generate_properties_string(properties: List[Property]) -> str:
        """
        Generate a string representation of properties for a graph element.

        @param properties: A list of properties.
        @return: A string representation of the properties.
        """
        return (
            " {" + ", ".join([repr(prop) for prop in properties]) + "}"
            if len(properties) > 0
            else ""
        )

    @staticmethod
    def generate_labels_string(labels: List[Label]) -> str:
        """
        Generate a string representation of labels for a graph element.

        @param labels: A list of labels.
        @return: A string representation of the labels.
        """
        return (
            ":" + ":".join(repr(label) for label in labels) if len(labels) > 0 else ""
        )
