import importlib.util
from pathlib import Path

import pandas as pd
import pytest


spec = importlib.util.spec_from_file_location(
    "rldx_prepare", Path(__file__).resolve().parents[1] / "scripts/prepare_dataset.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_task_column_and_official_task_index_layout():
    assert module.task_map(pd.DataFrame({"task_index": [0], "task": ["pick"]})) == {0: "pick"}
    assert module.task_map(pd.DataFrame({"task_index": [0]}, index=["pick"])) == {0: "pick"}


def test_numeric_index_is_never_used_as_instruction():
    with pytest.raises(ValueError, match="textual task"):
        module.task_map(pd.DataFrame({"task_index": [0]}))


def test_channel_order_is_not_guessed():
    with pytest.raises(ValueError, match="ordered, unique"):
        module.feature_names({"features": {"action": {"shape": [2], "names": ["a", "a"]}}}, "action")
