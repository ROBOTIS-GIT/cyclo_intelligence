"""Small, model-neutral helpers shared by imitation-learning CLIs."""

from __future__ import annotations

import argparse
import math
import signal
import threading
from collections.abc import Callable
from typing import TypeVar

from cyclo_brain.algorithm.common.cli import json_line

from .dataset import parse_success_episode_csv


_ResultT = TypeVar("_ResultT")


def _base10_integer(value: str) -> int:
    try:
        return int(value, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a base-10 integer") from error


def positive_integer(value: str) -> int:
    """Parse a strictly positive base-10 integer for ``argparse``."""

    parsed = _base10_integer(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def non_negative_integer(value: str) -> int:
    """Parse a non-negative base-10 integer for ``argparse``."""

    parsed = _base10_integer(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


def positive_float(value: str) -> float:
    """Parse a finite, strictly positive floating-point value for ``argparse``."""

    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return parsed


def episode_csv(value: str) -> tuple[int, ...]:
    """Adapt the shared episode-index parser to an ``argparse`` type."""

    try:
        return parse_success_episode_csv(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def run_with_stop_signals(
    callback: Callable[[Callable[[], bool]], _ResultT],
) -> _ResultT:
    """Run a trainer with cooperative SIGINT/SIGTERM cancellation.

    The caller still owns result/error serialization.  Keeping that policy out
    of this helper preserves each model CLI's existing output contract.
    """

    stop_requested = threading.Event()
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, lambda _signum, _frame: stop_requested.set())
    signal.signal(signal.SIGTERM, lambda _signum, _frame: stop_requested.set())
    try:
        return callback(stop_requested.is_set)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


__all__ = [
    "episode_csv",
    "json_line",
    "non_negative_integer",
    "positive_float",
    "positive_integer",
    "run_with_stop_signals",
]
