#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Seongwoo Kim

"""Singleton blackboard for sharing data across behavior tree nodes."""


class Blackboard:
    """Singleton blackboard for shared data storage."""

    _instance = None

    def __new__(cls):
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._data = {}
        return cls._instance

    def set_value(self, key: str, value):
        """Set a value in the blackboard."""
        self._data[key] = value

    def set(self, key: str, value):  # noqa: A003
        """Set a value in the blackboard (deprecated, use set_value)."""
        return self.set_value(key, value)

    def get(self, key: str, default=None):
        """Get a value from the blackboard."""
        return self._data.get(key, default)

    def has(self, key: str) -> bool:
        """Check if a key exists in the blackboard."""
        return key in self._data

    def clear(self):
        """Clear all data from the blackboard."""
        self._data.clear()
