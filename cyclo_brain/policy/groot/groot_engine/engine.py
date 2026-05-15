#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""GR00T engine package entrypoint.

The implementation currently lives in ``runtime.inference_engine`` to preserve
the existing GR00T integration path. This module gives the new Engine process a
stable ``POLICY_ENGINE_MODULE=groot_engine`` target, matching the backend
package convention used by LeRobot.
"""

from runtime.inference_engine import GR00TInference, create_engine

__all__ = ["GR00TInference", "create_engine"]
