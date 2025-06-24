# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Shriram Chandran

from datetime import datetime, date


class Schema:
    """
    A class to represent a schema for a property in a database.
    This class is used to define the mapping between a property name in the graph,
    its type, and the field name in the table.
    """

    def __init__(
        self, property_name: str, property_type: type = None, field_name: str = None
    ) -> None:
        """
        Initialize the Schema instance.

        @param property_name: The name of the property in the graph.
        @param property_type: The type of the property. Defaults to str.
        @param field_name: The name of the field in the table. Defaults to the property name.
        """
        self.field_name = property_name if field_name == None else field_name
        self.property_name = property_name
        self.property_type = property_type if property_type else str

    def __eq__(self, other) -> bool:
        """
        Check if two Schema instances are equal.

        @param other: The other Schema instance to compare.
        @return: True if the schemas are equal, False otherwise.
        """
        if not isinstance(other, Schema):
            return False
        return (
            self.field_name == other.field_name
            and self.property_name == other.property_name
            and self.property_type == other.property_type
        )

    def _field_to_property(self, var: str = None) -> str:
        """
        Utility method to convert a field to a property string.

        @param var: Field string.
        """
        var_str = "" if var == None else f"{var}."
        if self.property_type == str:
            return f"{self.property_name}: {var_str}{self.field_name}"
        elif self.property_type == int:
            return f"{self.property_name}: toInteger({var_str}{self.field_name})"
        elif self.property_type == float:
            return f"{self.property_name}: toFloat({var_str}{self.field_name})"
        elif self.property_type == bool:
            return f"{self.property_name}: toBoolean({var_str}{self.field_name})"
        elif self.property_type == datetime:
            return f"{self.property_name}: datetime({var_str}{self.field_name})"
        elif self.property_type == date:
            return f"{self.property_name}: date({var_str}{self.field_name})"
        else:
            raise ValueError(f"Unsupported property type: {self.property_type}")

    def _property_to_field(self) -> str:
        """
        Utility method to convert a property to a field string.

        @return: Field string.
        """
        return f"{self.property_name} AS {self.field_name}"

    def set_property(self, new_property_name: str) -> "Schema":
        """
        Get a new schema with a new property name.

        @param new_property_name: The new property name.
        @return: A new Schema instance with the updated property name.
        """
        return Schema(new_property_name, self.property_type, self.field_name)

    def set_field(self, new_field_name: str) -> "Schema":
        """
        Get a new schema with a new field name.

        @param new_field_name: The new field name.
        @return: A new Schema instance with the updated field name.
        """
        return Schema(self.property_name, self.property_type, new_field_name)

    def __repr__(self) -> str:
        """
        Return a string representation of the Schema instance.

        @return: A string representation of the schema.
        """
        return f"Schema({self.property_name}, {self.property_type}, {self.field_name})"
