"""GET_ACTION timing metadata for explicitly negotiated temporal observations."""

import json
import math


def _validate_wait(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 120:
        raise ValueError("observation wait must be finite and within 0..120 seconds")
    return float(value)


def encode_observation_wait(wait_s):
    return json.dumps({"observation_wait_s": _validate_wait(wait_s)}, separators=(",", ":"))


def action_latency(raw_metadata, roundtrip_s):
    if not isinstance(raw_metadata, str) or len(raw_metadata) > 256:
        raise ValueError("missing or oversized observation timing metadata")
    try:
        data = json.loads(raw_metadata)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid observation timing metadata") from exc
    if not isinstance(data, dict) or set(data) != {"observation_wait_s"}:
        raise ValueError("invalid observation timing metadata fields")
    wait = _validate_wait(data["observation_wait_s"])
    if not math.isfinite(roundtrip_s) or roundtrip_s < 0 or wait > roundtrip_s + 1e-6:
        raise ValueError("observation wait exceeds request roundtrip")
    return max(0., roundtrip_s - wait)
