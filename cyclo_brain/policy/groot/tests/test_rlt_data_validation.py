from pathlib import Path
import json
from unittest.mock import patch
from types import SimpleNamespace

import pytest
import torch

from cyclo_brain.algorithm.rl.tests.test_rlt_stage2_core import _run, _batch
from cyclo_brain.policy.groot.runtime.rlt_stage2_dataset import RLTStage2FeatureReplay
from cyclo_brain.policy.groot.runtime import rlt_data_validation as validation


def replay(tmp_path, sources=("vla", "human", "mlp")):
    fixture = _run(tmp_path / "encoder", action_dim=16)
    batch = _batch(fixture)
    count = len(sources)
    tensors = {name: getattr(batch, name)[:1].repeat((count,) + (1,) * (getattr(batch, name).ndim - 1))
               for name in RLTStage2FeatureReplay._TENSOR_NAMES}
    tensors["reference_actions"].fill_(0.1)
    tensors["executed_actions"].fill_(0.6)
    tensors["next_reference_actions"].fill_(0.1)
    tensors["bootstrap_discount"].fill_(0.9)
    tensors["bootstrap_discount"][-1].zero_()
    root = str(tmp_path / "dataset")
    metadata = {"dataset_roots": [root], "dataset_snapshots": [{"content_fingerprint": "abc"}],
                "row_provenance": [{"dataset_root": root, "episode_index": 0,
                                    "start_frame_index": i * 10, "action_source": source}
                                   for i, source in enumerate(sources)]}
    return RLTStage2FeatureReplay(fixture.learner.spec, tensors, metadata=metadata)


def test_validation_preserves_vla_references_and_behavior_actions(tmp_path):
    original = replay(tmp_path)
    prepared, result = validation.prepare_references(original)
    assert result["status"] == "ready"
    assert result["historical_mlp_output_required"] is False
    for name in RLTStage2FeatureReplay._TENSOR_NAMES:
        torch.testing.assert_close(prepared.tensors[name], original.tensors[name])
    prepared.save(tmp_path / "validated")
    restored = RLTStage2FeatureReplay.load(tmp_path / "validated")
    assert restored.metadata["data_validation"] == {
        "version": 2,
        "reference_rule": "preserve_saved_vla_reference",
        "actor_warmup_target": "executed_actions",
        "transition_count": 3,
        "unknown_source_rows": 0,
        "action_source_counts": {"vla": 10, "human": 10, "mlp": 10},
        "historical_mlp_output_required": False,
    }


def test_missing_sources_block_without_guessing_from_mlp_output(tmp_path):
    original = replay(tmp_path, (None, "human"))
    original.metadata["action_mean"] = "not evidence of execution"
    prepared, report = validation.prepare_references(original)
    assert prepared is None
    assert report["unknown_source_rows"] == 1
    assert report["status"] == "blocked"


def test_frame_sidecar_selects_human_frames_and_rejects_wrong_dataset(tmp_path):
    original = replay(tmp_path, (None,))
    path = tmp_path / "dataset/rlt/action_sources.json"
    path.parent.mkdir(parents=True)
    sidecar = {"format": validation.SOURCE_FORMAT, "dataset_content_fingerprint": "abc",
               "episodes": {"0": ["human"] * 4 + ["mlp"] * 6}}
    path.write_text(json.dumps(sidecar))
    prepared, result = validation.prepare_references(original)
    assert result["action_source_counts"] == {"human": 4, "mlp": 6}
    torch.testing.assert_close(
        prepared.tensors["reference_actions"], original.tensors["reference_actions"]
    )
    sidecar["dataset_content_fingerprint"] = "wrong"
    path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="does not match"):
        validation.prepare_references(original)


def test_validation_does_not_require_or_impute_a_successor_row(tmp_path):
    original = replay(tmp_path, ("human",))
    original.tensors["bootstrap_discount"].fill_(0.9)
    prepared, report = validation.prepare_references(original)
    assert report["status"] == "ready"
    torch.testing.assert_close(
        prepared.tensors["next_reference_actions"],
        original.tensors["next_reference_actions"],
    )


def test_validation_does_not_filter_or_rewrite_saved_transitions(tmp_path):
    original = replay(tmp_path)
    original.metadata["row_provenance"][-1]["start_frame_index"] = 25
    prepared, report = validation.prepare_references(original)
    assert report["transition_count"] == len(original)
    assert len(prepared) == len(original)
    for name in RLTStage2FeatureReplay._TENSOR_NAMES:
        torch.testing.assert_close(prepared.tensors[name], original.tensors[name])


def test_snapshot_change_rejected_with_model_action_dimensions(tmp_path):
    original = replay(tmp_path)
    with patch.object(validation, "open_rlt_stage2_source") as source:
        source.return_value.content_snapshot.return_value = {"content_fingerprint": "changed"}
        with pytest.raises(ValueError, match="changed"):
            validation.verify_snapshots(original)
        assert source.call_args.kwargs["action_dim"] == 16


def test_corrupt_saved_replay_rejected(tmp_path):
    original = replay(tmp_path)
    original.save(tmp_path / "saved")
    (tmp_path / "saved/replay.pt").write_bytes(b"broken")
    with pytest.raises(ValueError, match="digest"):
        RLTStage2FeatureReplay.load(tmp_path / "saved")


def test_saved_inputs_to_prepared_cache_without_vla_forward(tmp_path):
    original = replay(tmp_path)
    dataset = Path(original.metadata["dataset_roots"][0])
    original.save(dataset / "rlt/replay_cache")
    spec = original.spec
    provenance = SimpleNamespace(action_dim=spec.action_dim, reference_horizon=spec.reference_horizon,
                                 action_normalization_id=spec.action_normalization_id,
                                 action_codec_id=spec.action_codec_id)
    encoder = SimpleNamespace(representation_contract={}, config=SimpleNamespace(embedding_dim=spec.rl_token_dim),
                              representation_contract_fingerprint=spec.reference_contract_fingerprint,
                              artifact_fingerprint=spec.rl_token_artifact_fingerprint)
    request = {"checkpoint": "/unused", "encoder": "/unused", "dataset_paths": [str(dataset)], "candidates": [""]}
    with patch.object(validation, "build_groot_rlt_provenance", return_value=provenance), \
         patch.object(validation, "load_frozen_rl_token_encoder", return_value=encoder), \
         patch.object(validation, "validate_stage1_encoder_provenance"), \
         patch.object(validation, "verify_snapshots"), \
         patch("cyclo_brain.policy.groot.runtime.rlt_stage2_dataset.GR00TRLTStage2Extractor", side_effect=AssertionError("VLA must not run")):
        report = validation.validate(request, tmp_path / "output")
    assert report["status"] == "ready", report
    saved = RLTStage2FeatureReplay.load(tmp_path / "output/replay_cache", expected_spec=spec)
    torch.testing.assert_close(saved.tensors["reference_actions"], original.tensors["reference_actions"])
    torch.testing.assert_close(
        saved.tensors["next_reference_actions"], original.tensors["next_reference_actions"]
    )
    torch.testing.assert_close(saved.tensors["executed_actions"], original.tensors["executed_actions"])


def test_converted_intervention_column_is_used_automatically(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    original = replay(tmp_path, (None,))
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "meta/info.json").write_text(json.dumps({"features": {"intervention_mode": {"dtype": "bool"}}}))
    pq.write_table(pa.table({"episode_index": [0] * 10, "frame_index": list(range(10)),
                            "intervention_mode": [True] * 3 + [False] * 7}), root / "data/episode.parquet")
    prepared, result = validation.prepare_references(original)
    assert result["status"] == "ready"
    assert result["action_source_counts"] == {"human": 3, "policy": 7}
    torch.testing.assert_close(
        prepared.tensors["reference_actions"], original.tensors["reference_actions"]
    )


def test_v21_global_index_is_aligned_to_episode_frames_without_rewriting_actions(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    original = replay(tmp_path, (None,))
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "meta/info.json").write_text(json.dumps({
        "features": {"action_source": {"dtype": "string"}}
    }))
    pq.write_table(pa.table({
        "index": list(range(100, 110)),
        "episode_index": [0] * 10,
        "action_source": ["human"] * 3 + ["vla"] * 7,
    }), root / "data/episode.parquet")

    prepared, result = validation.prepare_references(original)

    assert result["status"] == "ready"
    assert result["action_source_counts"] == {"human": 3, "vla": 7}
    torch.testing.assert_close(
        prepared.tensors["reference_actions"], original.tensors["reference_actions"]
    )
    torch.testing.assert_close(
        prepared.tensors["executed_actions"], original.tensors["executed_actions"]
    )


def test_existing_verified_mlp_recording_is_detected_without_manual_labels(tmp_path):
    original = replay(tmp_path, (None,))
    root = tmp_path / "dataset"
    (root / "data").mkdir(parents=True)
    (root / "rlt").mkdir()
    path = root / "data/episode.parquet"
    path.write_bytes(b"recorded actions")
    proof = {"format": "cyclo.rlt.dataset_execution_verification/v1", "execution_verified": True,
             "action_data_files": [{"relative_path": "data/episode.parquet", "byte_count": path.stat().st_size,
                                     "sha256": validation.file_sha256(path)}],
             "episodes": [{"episode_index": 0, "execution_verified": True}]}
    (root / "rlt/execution_verification.json").write_text(json.dumps({**proof,
        "verification_fingerprint": validation.canonical_json_sha256(proof)}))
    prepared, result = validation.prepare_references(original)
    assert result["status"] == "ready"
    assert result["action_source_counts"] == {"mlp": 10}
    torch.testing.assert_close(prepared.tensors["reference_actions"], original.tensors["reference_actions"])
    path.write_bytes(b"changed actions")
    with pytest.raises(ValueError, match="stale"):
        validation.prepare_references(original)

