# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Shriram Chandran


def get_value_str(value: any) -> str:
    """
    Return a string representation of the passed value.

    @param value: Passed value.
    @return: String representation of the value.
    """
    if isinstance(value, str):
        return f"'{value}'"
    return str(value)


class Property:
    """
    A class representing an LPG property.
    """

    def __init__(self, key: str, property_type: type = None, value=None) -> None:
        """
        Initialize the Property instance.

        @param key: The key of the property.
        @param property_type: The type of the property. Defaults to None.
        @param value: The value of the property. Defaults to None.
        """
        self.key = key
        self.value = value

    def __repr__(self) -> str:
        """
        Return a string representation of the Property instance.

        @return: A string representation of the property.
        """
        return f"{self.key}: {get_value_str(self.value)}"

    def __eq__(self, other) -> bool:
        """
        Check if two Property instances are equal.

        @param other: The other Property instance to compare.
        @return: True if the properties are equal, False otherwise.
        """
        if not isinstance(other, Property):
            return False
        return self.key == other.key and self.value == other.value
