#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared constants for the LeRobot engine + its mixins.

Pulled out so each module that needs them doesn't duplicate the
string literal (and so ``grep -rn _IMAGE_KEY_PREFIX`` returns one hit
instead of four). The module has no imports beyond the standard
library to keep it free of circular-import risk.
"""

from __future__ import annotations


IMAGE_KEY_PREFIX = "observation.images."
STATE_KEY = "observation.state"
