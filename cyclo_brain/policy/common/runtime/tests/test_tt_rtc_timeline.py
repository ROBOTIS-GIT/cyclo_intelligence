#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from main_runtime.tt_rtc_timeline import TTActionTimeline  # noqa: E402


class TTActionTimelineTests(unittest.TestCase):
    def test_keeps_source_prefix_while_outputting_at_control_rate(self) -> None:
        timeline = TTActionTimeline(source_hz=15.0, control_hz=100.0)
        source = np.arange(16, dtype=np.float64).reshape(16, 1)
        timeline.push_actions(source)

        outputs = [timeline.pop_action() for _ in range(61)]

        self.assertEqual(timeline.output_hz, 100.0)
        self.assertEqual(timeline.buffer_size, 6)
        np.testing.assert_allclose(timeline.peek_actions(), source[10:])
        np.testing.assert_allclose(outputs[0], np.asarray([0.0]))
        np.testing.assert_allclose(outputs[59], np.asarray([8.85]))
        np.testing.assert_allclose(outputs[60], np.asarray([9.0]))

    def test_six_source_actions_cover_four_hundred_milliseconds(self) -> None:
        timeline = TTActionTimeline(source_hz=15.0, control_hz=100.0)
        source = np.arange(16, dtype=np.float64).reshape(16, 1)
        timeline.push_actions(source)
        for _ in range(61):
            timeline.pop_action()

        remaining = [timeline.pop_action() for _ in range(40)]

        self.assertEqual(timeline.buffer_size, 0)
        np.testing.assert_allclose(remaining[-1], np.asarray([15.0]))

    def test_new_postfix_extends_existing_prefix_without_a_jump(self) -> None:
        timeline = TTActionTimeline(source_hz=15.0, control_hz=100.0)
        source = np.arange(16, dtype=np.float64).reshape(16, 1)
        postfix = np.arange(16, 26, dtype=np.float64).reshape(10, 1)
        timeline.push_actions(source)
        for _ in range(61):
            timeline.pop_action()

        captured_prefix = timeline.peek_actions()
        timeline.push_actions(postfix)
        self.assertEqual(timeline.buffer_size, 16)
        np.testing.assert_allclose(timeline.peek_actions()[:6], captured_prefix)

        boundary = [timeline.pop_action() for _ in range(41)]
        np.testing.assert_allclose(boundary[-2], np.asarray([15.0]))
        np.testing.assert_allclose(boundary[-1], np.asarray([15.15]))

    def test_clear_resets_source_phase(self) -> None:
        timeline = TTActionTimeline(source_hz=15.0, control_hz=100.0)
        timeline.push_actions(np.asarray([[0.0], [1.0]], dtype=np.float64))
        for _ in range(4):
            timeline.pop_action()
        timeline.clear()
        timeline.push_actions(np.asarray([[10.0], [20.0]], dtype=np.float64))

        np.testing.assert_allclose(timeline.pop_action(), np.asarray([10.0]))

    def test_late_refill_holds_then_resumes_without_skipping_actions(self) -> None:
        timeline = TTActionTimeline(source_hz=15.0, control_hz=100.0)
        timeline.push_actions(np.asarray([[0.0], [1.0]]))
        for _ in range(20):
            timeline.pop_action()
        for _ in range(100):
            self.assertIsNone(timeline.pop_action())
        np.testing.assert_allclose(timeline.last_action, [1.0])
        timeline.push_actions(np.asarray([[2.0], [3.0], [4.0]]))
        outputs = [timeline.pop_action() for _ in range(21)]
        np.testing.assert_allclose(np.asarray(outputs)[:, 0], 1.0 + np.arange(21) * 0.15)

    def test_rejects_a_control_rate_below_the_source_rate(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than or equal"):
            TTActionTimeline(source_hz=100.0, control_hz=15.0)


if __name__ == "__main__":
    unittest.main()
