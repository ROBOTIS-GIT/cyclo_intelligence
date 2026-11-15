"""Shared JSON progress output for learning CLIs."""

import json
import sys
from collections.abc import Mapping
from typing import Any


def json_line(value: Mapping[str, Any], *, stream: Any = sys.stdout) -> None:
    """Emit one deterministic, strict JSON object and flush it immediately."""

    print(
        json.dumps(
            dict(value),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        file=stream,
        flush=True,
    )

