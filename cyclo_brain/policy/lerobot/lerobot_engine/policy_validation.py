# Copyright 2026 ROBOTIS CO., LTD.
# Licensed under the Apache License, Version 2.0

"""Dispatch checkpoint checks to the selected adapter before allocating weights."""

import json
from pathlib import Path


def validate_requested_policy(model_path, policy_id):
    from lerobot_engine.adapters import resolve_adapter

    if not policy_id.startswith("lerobot:"):
        return
    validator = resolve_adapter(policy_id.split(":", 1)[1]).requested_config_validator
    if validator is not None:
        with (Path(model_path) / "config.json").open() as stream:
            validator(json.load(stream))


def validate_checkpoint(config, model_path):
    from lerobot_engine.adapters import resolve_adapter

    validator = resolve_adapter(config.get("type")).checkpoint_validator
    if validator is not None:
        validator(config, model_path)
