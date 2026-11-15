import io
import json
from pathlib import Path

import numpy as np
import pytest

from cyclo_data.recorder.rlt_execution_verification import (
    validate_dataset_execution_verification,
    verify_episode_execution,
    write_dataset_execution_verification,
)


_ACTOR = "a" * 64


def _event(root: Path, index: int, metadata: dict, **arrays) -> dict:
    name = f"recording_{index:06d}.npz"
    payload = dict(
        metadata,
        format="cyclo.rlt.recording/v1",
        recording_id="recording",
        publisher_dropped=0,
        file=name,
    )
    with io.BytesIO() as buffer:
        np.savez(
            buffer,
            metadata=np.frombuffer(json.dumps(payload).encode(), dtype=np.uint8),
            **arrays,
        )
        (root / name).write_bytes(buffer.getvalue())
    return payload


def _sidecar(tmp_path: Path, actions: list[np.ndarray]) -> Path:
    root = tmp_path / "rlt"
    root.mkdir()
    rows = [
        _event(
            root,
            0,
            {
                "event": "inference",
                "policy_identity": {"actor_sha256": _ACTOR},
            },
        )
    ]
    base_ns = 1_800_000_000_000_000_000
    for index, action in enumerate(actions, start=1):
        timestamp_ns = base_ns + (index - 1) * 100_000_000
        rows.append(
            _event(
                root,
                index,
                {
                    "event": "action_commanded",
                    "command_started_ns": timestamp_ns - 1_000,
                    "command_completed_ns": timestamp_ns,
                    "publish_success": True,
                    "action_policy_mode": "rlt",
                },
                commanded_action=np.asarray(action, dtype=np.float64).reshape(1, -1),
            )
        )
    (root / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (root / "recording_summary.json").write_text(
        json.dumps(
            {
                "format": "cyclo.rlt.recording_summary/v1",
                "recording_id": "recording",
                "records": len(rows),
                "recorder_dropped": 0,
                "publisher_dropped": 0,
                "publisher_summary_received": True,
                "error": "",
                "command_records": len(actions),
                "command_publish_failures": 0,
                "command_stream_complete": True,
                "execution_verified": False,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_episode_verification_matches_final_commands(tmp_path):
    actions = [
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([3.0, 4.0], dtype=np.float32),
    ]
    sidecar = _sidecar(tmp_path, actions)
    base = 1_800_000_000.0

    result = verify_episode_execution(
        sidecar=sidecar,
        episode_index=7,
        grid_log_times_sec=[base, base + 0.1],
        actions=actions,
        episode_success=True,
        fps=15.0,
    )

    assert result["execution_verified"] is True
    assert result["matched_frame_count"] == 2
    assert result["actor_sha256"] == [_ACTOR]
    assert result["reasons"] == []


def test_episode_verification_rejects_candidate_instead_of_command(tmp_path):
    commands = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    sidecar = _sidecar(tmp_path, commands)
    base = 1_800_000_000.0

    result = verify_episode_execution(
        sidecar=sidecar,
        episode_index=0,
        grid_log_times_sec=[base, base + 0.1],
        actions=[commands[0], np.array([30.0, 40.0])],
        episode_success=False,
        fps=15.0,
    )

    assert result["execution_verified"] is False
    assert result["mismatch_frame_indices"] == [1]
    assert "LeRobot actions" in result["reasons"][-1]


def test_dataset_gate_requires_verified_success_and_failure(tmp_path):
    action_file = tmp_path / "data" / "chunk-000" / "episode_000000.parquet"
    action_file.parent.mkdir(parents=True)
    action_file.write_bytes(b"verified-action-data")
    records = []
    for episode_index, successful in enumerate((True, False)):
        unsigned = {
            "format": "cyclo.rlt.episode_execution_verification/v1",
            "episode_index": episode_index,
            "execution_verified": True,
            "frame_count": 10,
            "matched_frame_count": 10,
            "mismatch_frame_indices": [],
            "max_abs_error": 0.0,
            "absolute_tolerance": 1e-5,
            "episode_success": successful,
            "command_record_count": 100,
            "actor_sha256": [_ACTOR],
            "recorder_dropped": 0,
            "publisher_dropped": 0,
            "reasons": [],
        }
        from cyclo_data.recorder.rlt_execution_verification import _canonical_fingerprint

        records.append(
            {
                **unsigned,
                "verification_fingerprint": _canonical_fingerprint(unsigned),
            }
        )

    write_dataset_execution_verification(tmp_path, records)
    manifest = validate_dataset_execution_verification(tmp_path)
    assert manifest["episode_count"] == 2
    assert manifest["success_episode_count"] == 1
    assert manifest["failure_episode_count"] == 1

    records[1]["execution_verified"] = False
    write_dataset_execution_verification(tmp_path, records)
    with pytest.raises(ValueError, match="has not been verified"):
        validate_dataset_execution_verification(tmp_path)
