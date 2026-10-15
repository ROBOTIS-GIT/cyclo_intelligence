#!/usr/bin/env python3
"""CPU-only trace regression tests; no model, network, or robot dependencies."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
trace_spec = importlib.util.spec_from_file_location("latency_trace", RUNTIME_ROOT / "main_runtime/latency_trace.py")
trace_module = importlib.util.module_from_spec(trace_spec)
trace_spec.loader.exec_module(trace_module)
AsyncLatencyTrace = trace_module.AsyncLatencyTrace
BufferedActionTrace = trace_module.BufferedActionTrace

class TraceWriterTests(unittest.TestCase):
    def test_disabled_and_environment_path(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            trace = AsyncLatencyTrace()
            self.assertFalse(trace.enabled)
            self.assertFalse(trace.emit("request_started"))
            self.assertTrue(trace.close())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            with mock.patch.dict(os.environ, {"INFERENCE_TRACE_PATH": str(path)}):
                trace = AsyncLatencyTrace()
                self.assertTrue(trace.emit("request_started", request_id="a"))
                self.assertTrue(trace.close())
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["event"] for row in rows], ["trace_started", "request_started", "trace_closed"])

    def test_concurrent_emit_flush_and_append(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            ids = set()
            for _ in range(2):
                trace = AsyncLatencyTrace(path, queue_size=1000)
                ids.add(trace.trace_id)
                workers = [threading.Thread(target=lambda: [trace.emit("sample") for _ in range(100)]) for _ in range(4)]
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join()
                self.assertTrue(trace.close())
                self.assertFalse(trace.emit("too_late"))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 804)
            self.assertEqual(len(ids), 2)
            for trace_id in ids:
                session = [row for row in rows if row["trace_id"] == trace_id]
                self.assertEqual([row["sequence"] for row in session], list(range(402)))
                self.assertEqual(session[-1]["written_events"], 400)
                self.assertTrue(all(row["schema_version"] == 1 and row["wall_time_ns"] > 0 for row in session))

    def test_slow_writer_does_not_grow_queue_or_write_on_caller(self):
        entered, release = threading.Event(), threading.Event()
        stderr = io.StringIO()
        original_write = AsyncLatencyTrace._write
        writer_threads = set()

        def blocked_write(stream, row):
            writer_threads.add(threading.current_thread().name)
            if row["event"] == "trace_started":
                entered.set()
                release.wait(2)
            original_write(stream, row)

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(stderr):
            path = Path(directory) / "events.jsonl"
            with mock.patch.object(AsyncLatencyTrace, "_write", staticmethod(blocked_write)):
                trace = AsyncLatencyTrace(path, queue_size=2)
                self.assertTrue(entered.wait(1))
                try:
                    for _ in range(100):
                        trace.emit("sample")
                    self.assertEqual(trace._queue.qsize(), 2)
                    self.assertEqual(trace.stats["dropped_events"], 98)
                    self.assertEqual(stderr.getvalue(), "")
                finally:
                    release.set()
                self.assertFalse(trace.close())
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows[-1]["dropped_events"], 98)
            self.assertEqual(writer_threads, {"inference-latency-trace"})
            self.assertIn("incomplete", stderr.getvalue())

    def test_unserializable_payload_and_io_failure_are_visible(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()) as stderr:
            path = Path(directory) / "events.jsonl"
            trace = AsyncLatencyTrace(path)
            trace.emit("bad", value=float("nan"))
            self.assertFalse(trace.close())
            self.assertEqual(trace.stats["serialization_errors"], 1)
            failed = AsyncLatencyTrace(directory)  # A directory cannot be opened as a JSONL file.
            self.assertFalse(failed.close())
            self.assertIsNotNone(failed.stats["writer_error"])
            self.assertIn("writer failed", stderr.getvalue())


class BufferedActionTraceTests(unittest.TestCase):
    def test_boundaries_follow_consumption_and_successful_publish(self):
        events = []

        class Sink:
            def emit(self, event, **payload):
                events.append({"event": event, **payload})

        trace = BufferedActionTrace(Sink())
        trace.popped(False, 1.0, 5)  # Initial startup is not an underflow.
        trace.installed("first", 5, 0, 2)
        trace.installed("second", 5, 2, 2)
        trace.popped(True, 1.1, 5)
        trace.published()
        trace.popped(True, 1.2, 5)
        trace.published()
        trace.popped(True, 1.3, 5)
        # A failed publish does not emit a false boundary.
        trace.popped(True, 1.4, 5)
        trace.published()
        self.assertEqual([e["request_id"] for e in events if e["event"] == "command_published"], ["first", "second"])
        self.assertEqual(len(trace._markers), 0)
        trace.popped(False, 1.5, 5)
        trace.popped(False, 1.6, 5)
        trace.reset(1.8)
        underflows = [e for e in events if e["event"] == "queue_underflow"]
        self.assertEqual([e["phase"] for e in underflows], ["start", "end"])
        self.assertAlmostEqual(underflows[-1]["duration_s"], 0.3)
        self.assertEqual(underflows[-1]["generation"], 5)
        trace.popped(False, 2.0, 6)
        self.assertEqual(len([e for e in events if e["event"] == "queue_underflow"]), 2)

    def test_markers_account_for_unattributed_existing_queue(self):
        events = []

        class Sink:
            def emit(self, event, **payload):
                events.append({"event": event, **payload})

        trace = BufferedActionTrace(Sink())
        trace.installed("new", 1, 2, 1)
        for i in range(2):
            trace.popped(True, float(i), 1)
            trace.published()
        self.assertFalse(any(e["event"] == "command_published" for e in events))
        trace.popped(True, 2.0, 1)
        trace.published()
        self.assertEqual(events[-1]["request_id"], "new")


if __name__ == "__main__":
    unittest.main()
