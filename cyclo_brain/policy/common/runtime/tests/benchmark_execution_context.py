"""CPU-only serialization benchmark; no model, network or robot commands.

Run: python3 cyclo_brain/policy/common/runtime/tests/benchmark_execution_context.py
The baseline is the original asdict + JSON + bounded compression path. Both
encoders must produce identical wire bytes for every measured context.
"""

import base64
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
import time
import timeit
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_context.execution import ActionRecord, ExecutionContext, CONTEXT_COMPRESSION_THRESHOLD_BYTES


def baseline(context):
    data = asdict(context)
    if context.feedback_schema == 1:
        data.pop("feedback_schema")
        for plan in data["planning"]:
            plan.pop("command_start_id")
    raw = json.dumps(data, allow_nan=False, sort_keys=True, separators=(",", ":"))
    encoded = raw.encode("utf-8")
    if len(encoded) > CONTEXT_COMPRESSION_THRESHOLD_BYTES:
        return json.dumps({"encoding": "zlib+base64", "payload": base64.b64encode(
            zlib.compress(encoded)).decode("ascii")}, separators=(",", ":"))
    return raw


def median_ms(function, repeats=15):
    function()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1000)
    return round(statistics.median(samples), 3)


def main():
    for count in (20, 500, 2000):
        context = ExecutionContext("bench", 0, 1, "running", actions=tuple(
            ActionRecord("1", "published", "command", tuple(float(v) for v in range(22)),
                         command_id=i, event_id=i, recorded_s=100 + i / 100)
            for i in range(1, count + 1)
        ), latest_event_id=count)
        raw = context.to_json()
        assert raw == baseline(context)
        assert ExecutionContext.from_json(raw) == context
        print(json.dumps({
            "receipts": count, "dimensions": 22, "wire_bytes": len(raw.encode("utf-8")),
            "baseline_encode_median_ms": median_ms(lambda: baseline(context)),
            "current_encode_median_ms": median_ms(context.to_json),
            "parse_once_median_ms": median_ms(lambda: ExecutionContext.from_json(raw)),
        }))


def benchmark_snapshot():
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "sdk" / "action_chunk_processing"))
    from action_chunk_processing.tracked_buffer import ExecutionSnapshot, TrackedActionBuffer

    buffer = TrackedActionBuffer(postprocess=False)
    for _ in range(5000):
        buffer.clear("reset")

    def baseline_snapshot(cursor):
        with buffer._lock:
            if type(cursor) is not int or not 0 <= cursor <= buffer._event_id:
                raise ValueError("invalid event cursor")
            if buffer._events and cursor < buffer._events[0].event_id - 1:
                raise RuntimeError("execution history gap; reset consumer context")
            return ExecutionSnapshot(
                buffer._revision, buffer._event_id, tuple(buffer._pending), buffer._in_flight,
                tuple(event for event in buffer._events if event.event_id > cursor), buffer._failure,
            )

    for delta in (0, 1, 20, 4096):
        cursor = 5000 - delta
        assert baseline_snapshot(cursor) == buffer.snapshot(after_event_id=cursor)
        before = statistics.median(timeit.repeat(lambda: baseline_snapshot(cursor), repeat=7, number=1000)) * 1000
        after = statistics.median(timeit.repeat(lambda: buffer.snapshot(after_event_id=cursor), repeat=7, number=1000)) * 1000
        print(json.dumps({"retained_events": 4096, "unacknowledged_events": delta,
                          "baseline_snapshot_median_us": round(before, 3),
                          "current_snapshot_median_us": round(after, 3)}))


if __name__ == "__main__":
    main()
    benchmark_snapshot()
