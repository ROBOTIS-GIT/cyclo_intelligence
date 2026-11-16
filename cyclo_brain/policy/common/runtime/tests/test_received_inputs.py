"""Callback-time barriers do not turn latest values into observation history."""

import pytest

from inference_context.inputs import ReceivedValues, SampleQuery


def test_received_values_enforce_barrier_and_age_without_changing_values():
    value = object()
    provider = ReceivedValues({"state": value}, {"state": 10.1}, after_s=10.)
    assert provider.resolve(SampleQuery("state", max_age_s=1.), 10.2) == (value,)
    with pytest.raises(ValueError, match="stale"):
        provider.resolve(SampleQuery("state", max_age_s=1.), 11.2)
    with pytest.raises(ValueError, match="history"):
        provider.resolve(SampleQuery("state", (-.1, 0.)), 10.2)


@pytest.mark.parametrize("stamp", [None, True, -1, float("nan"), float("inf"), "10", 11.])
def test_missing_invalid_or_future_callback_timestamps_are_rejected(stamp):
    provider = ReceivedValues({"camera": object()}, {"camera": stamp})
    with pytest.raises(ValueError, match="timestamp"):
        provider.resolve(SampleQuery("camera"), 10.)


@pytest.mark.parametrize("stamp", [9.9, 10.])
def test_reading_a_cached_observation_again_does_not_cross_publication_barrier(stamp):
    provider = ReceivedValues({"joint": 1}, {"joint": stamp}, after_s=10.)
    for anchor in (10.1, 10.2, 11.):
        with pytest.raises(ValueError, match="barrier"):
            provider.resolve(SampleQuery("joint"), anchor)


@pytest.mark.parametrize("stamp", [-1, float("nan"), True, "10"])
def test_invalid_barrier_is_rejected(stamp):
    with pytest.raises(ValueError, match="barrier"):
        ReceivedValues({}, {}, after_s=stamp)
