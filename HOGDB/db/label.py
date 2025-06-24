# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Shriram Chandran

class Label:
    """
    A class representing an LPG label.
    """
    def __init__(self, label: str) -> None:
        """
        Initialize the Label instance.

        @param label: The label string.
        """
        self.label = label

    def __repr__(self) -> str:
        """
        Return a string representation of the Label instance.

        @return: A string representation of the label.
        """
        return self.label

    def __eq__(self, other) -> bool:
        """
        Check if two Label instances are equal.

        @param other: The other Label instance to compare.
        @return: True if the labels are equal, False otherwise.
        """
        return self.label == other.label

    def __hash__(self) -> int:
        """
        Return the hash of the Label instance.

        @return: The hash value of the label.
        """
        return hash(self.label)
