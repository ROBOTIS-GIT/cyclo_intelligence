"""Validate saved inputs and prepare intervention references, without VLA forward."""

from collections import Counter, defaultdict
import argparse
import json
import torch
from pathlib import Path

from cyclo_brain.algorithm.common import atomic_json_save, file_sha256, canonical_json_sha256
from cyclo_brain.algorithm.rl.rlt import RLTStage2Spec, load_frozen_rl_token_encoder
from .rlt_provenance import build_groot_rlt_provenance, validate_stage1_encoder_provenance
from .rlt_stage2_dataset import RLTStage2FeatureReplay, open_rlt_stage2_source
from .rlt_saved_replay import load_saved_replay


SOURCE_FORMAT = "cyclo.rlt.frame_action_sources/v1"


def _recorded_sources(root):
    """Read already aligned converted columns or the existing RLT execution proof."""
    root = Path(root)
    proof_path = root / "rlt/execution_verification.json"
    verified = set()
    if proof_path.is_file() and not proof_path.is_symlink():
        proof = json.loads(proof_path.read_text())
        digest = proof.pop("verification_fingerprint", None)
        if (proof.get("format") == "cyclo.rlt.dataset_execution_verification/v1" and
                proof.get("execution_verified") is True and
                digest == canonical_json_sha256(proof)):
            files = proof.get("action_data_files", [])
            actual = {p.relative_to(root).as_posix() for p in (root / "data").rglob("*.parquet")}
            if not files or {r["relative_path"] for r in files} != actual:
                raise ValueError("RLT execution proof does not cover all action files.")
            for record in files:
                path = root / record["relative_path"]
                if (path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or
                        path.stat().st_size != record["byte_count"] or file_sha256(path) != record["sha256"]):
                    raise ValueError("RLT execution proof is stale.")
            verified = {row["episode_index"] for row in proof["episodes"] if row.get("execution_verified") is True}
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text()) if info_path.is_file() else {}
    field = next((name for name in ("action_source", "intervention_mode", "is_intervention")
                  if name in info.get("features", {})), None)
    frames = {}
    if field:
        import pyarrow.parquet as pq
        episode_rows = defaultdict(list)
        for path in sorted((root / "data").rglob("*.parquet")):
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Unsafe action-source parquet path.")
            names = set(pq.read_schema(path).names)
            position_field = "frame_index" if "frame_index" in names else "index"
            if not {"episode_index", position_field, field}.issubset(names):
                raise ValueError("Action-source parquet has no frame-alignment columns.")
            table = pq.read_table(path, columns=["episode_index", position_field, field])
            for row in table.to_pylist():
                value = row[field]
                label = value if field == "action_source" else (
                    "human" if value is True else "policy" if value is False else "unknown")
                episode_rows[int(row["episode_index"])].append(
                    (int(row[position_field]), label, position_field)
                )
        for episode_index, values in episode_rows.items():
            position_fields = {value[2] for value in values}
            if len(position_fields) != 1:
                raise ValueError("Mixed action-source frame-index contracts.")
            values.sort(key=lambda value: value[0])
            positions = [value[0] for value in values]
            if len(positions) != len(set(positions)):
                raise ValueError("Duplicate action-source frame indices.")
            explicit_frames = next(iter(position_fields)) == "frame_index"
            for offset, (position, label, _) in enumerate(values):
                frame_index = position if explicit_frames else offset
                key = (episode_index, frame_index)
                if key in frames:
                    raise ValueError("Duplicate action-source frame indices.")
                frames[key] = label
    return verified, frames


def verify_snapshots(replay):
    roots = replay.metadata.get("dataset_roots", [])
    saved = replay.metadata.get("dataset_snapshots", [])
    if not roots or len(saved) != len(roots):
        raise ValueError("Dataset fingerprints are missing; prepare a replay with provenance.")
    actual = [open_rlt_stage2_source(Path(root), expected_fps=replay.spec.action_hz,
                                    action_dim=replay.spec.action_dim).content_snapshot()
              for root in roots]
    if actual != saved:
        raise ValueError("Dataset content changed after the replay was saved. Rebuild its replay.")


def _source_masks(replay):
    """Only use explicit, frame-aligned provenance. Folder names are not labels."""
    rows = replay.metadata.get("row_provenance", [])
    if len(rows) != len(replay):
        raise ValueError("Replay row provenance is incomplete.")
    sidecars = {}
    recorded = {}
    for root, snapshot in zip(replay.metadata["dataset_roots"], replay.metadata["dataset_snapshots"]):
        recorded[root] = _recorded_sources(root)
        path = Path(root) / "rlt/action_sources.json"
        if not path.exists():
            continue
        if path.is_symlink():
            raise ValueError("Action-source sidecar must not be a symlink.")
        value = json.loads(path.read_text())
        if (value.get("format") != SOURCE_FORMAT or
                value.get("dataset_content_fingerprint") != snapshot["content_fingerprint"]):
            raise ValueError(f"Action-source sidecar does not match the dataset: {root}")
        sidecars[root] = value["episodes"]
    labels = []
    for row in rows:
        source = row.get("action_sources", row.get("action_source"))
        verified, frames = recorded[row["dataset_root"]]
        if frames:
            source_indices = row.get("source_action_frame_indices")
            if not (
                isinstance(source_indices, list)
                and len(source_indices) == replay.spec.chunk_length
                and all(type(index) is int and index >= 0 for index in source_indices)
            ):
                start = row.get("action_start_frame_index", row["start_frame_index"])
                source_indices = list(range(start, start + replay.spec.chunk_length))
            source = [
                frames.get((row["episode_index"], index), "unknown")
                for index in source_indices
            ]
        elif source is None and row["episode_index"] in verified:
            source = "mlp"
        if row["dataset_root"] in sidecars:
            frames = sidecars[row["dataset_root"]].get(str(row["episode_index"]), [])
            start = row.get("action_start_frame_index", row["start_frame_index"])
            source = frames[start:start + replay.spec.chunk_length]
        if isinstance(source, str):
            source = [source] * replay.spec.chunk_length
        if not isinstance(source, list) or len(source) != replay.spec.chunk_length:
            source = ["unknown"] * replay.spec.chunk_length
        aliases = {"rlt": "mlp", "intervention": "human"}
        labels.append([aliases.get(item, item) if isinstance(item, str) and aliases.get(item, item) in
                       ("human", "vla", "mlp", "policy") else "unknown" for item in source])
    return rows, labels


def prepare_references(replay):
    """Validate action provenance without changing saved VLA references or behavior."""
    rows, labels = _source_masks(replay)
    counts = Counter(label for values in labels for label in values)
    unknown_rows = sum("unknown" in values for values in labels)
    summary = {"transition_count": len(replay), "unknown_source_rows": unknown_rows,
               "action_source_counts": dict(counts), "historical_mlp_output_required": False}
    if unknown_rows:
        return None, {**summary, "status": "blocked", "message":
            f"{unknown_rows}개 구간의 사람 개입/VLA/MLP 구분 정보가 없습니다. "
            "원본 intervention 신호와 변환 프레임의 매핑이 필요합니다. 저장 MLP 출력 누락은 문제가 아닙니다."}
    tensors = {key: value.clone() for key, value in replay.tensors.items()}
    if replay.metadata.get("replay_role") in {"tt_rtc_actor/v1", "tt_rtc_transition/v1"}:
        expected = torch.tensor([[label == "human" for label in values] for values in labels], dtype=torch.float32)
        if not torch.equal(tensors["imitation_mask"], expected):
            raise ValueError("TT-RTC human mask disagrees with recorded target action sources")
    metadata = dict(replay.metadata)
    metadata["data_validation"] = {
        "version": 2,
        "reference_rule": "preserve_saved_vla_reference",
        "actor_warmup_target": "executed_actions",
        **summary,
    }
    metadata["transition_count"] = len(replay)
    metadata["row_provenance"] = [
        {**row, "action_sources": values} for row, values in zip(rows, labels)
    ]
    return RLTStage2FeatureReplay(replay.spec, tensors, metadata=metadata), {
        **summary, "status": "ready", "message":
        f"{len(replay)}개 구간 검증 완료 · 저장된 VLA reference는 입력으로 유지하고 "
        "executed actions을 Actor warmup 정답으로 사용합니다."}


def validate(request, output):
    # Loading a frozen artifact/config is not a model forward pass.
    provenance = build_groot_rlt_provenance(Path(request["checkpoint"]))
    encoder = load_frozen_rl_token_encoder(request["encoder"], device="cpu")
    validate_stage1_encoder_provenance(encoder.representation_contract, provenance)
    spec = RLTStage2Spec(
        reference_contract_fingerprint=encoder.representation_contract_fingerprint,
        rl_token_artifact_fingerprint=encoder.artifact_fingerprint,
        rl_token_dim=encoder.config.embedding_dim, proprio_dim=provenance.action_dim,
        reference_horizon=provenance.reference_horizon, chunk_length=10,
        action_dim=provenance.action_dim, action_hz=15.0,
        action_normalization_id=provenance.action_normalization_id,
        action_codec_id=provenance.action_codec_id, model_domain="normalized", schema_version=1,
    )
    if request.get("expected_spec") and RLTStage2Spec(**request["expected_spec"]) != spec:
        raise ValueError("Selected bundle and frozen model/encoder specifications differ.")
    del encoder
    roots = request["dataset_paths"]
    candidates = request["candidates"]
    errors = []
    for candidate in candidates:
        try:
            replay = load_saved_replay(roots, spec=spec, output_root=output / "working",
                                       replay_bundle=candidate or None, training_phase=request.get("training_phase"))
            verify_snapshots(replay)
            if request.get("training_phase") in {"critic_warmup", "rl"} and replay.actor_only:
                raise ValueError("Prepare TT-RTC Critic/RL replay first; this replay only supports Actor warmup")
            prepared, report = prepare_references(replay)
            if prepared is None:
                if report not in errors:
                    errors.append(report)
                continue
            prepared.save(output / "replay_cache")
            return {**report, "replay_path": str(output), "source_bundle": candidate,
                    "manifest_sha256": file_sha256(output / "replay_cache/manifest.json")}
        except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
            errors.append({"status": "blocked", "message": str(exc)})
    if errors:
        return {**errors[0], "issues": errors[1:]}
    return {"status": "blocked", "message":
            "RL token·state·VLA reference를 포함한 저장 replay가 필요합니다."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    try:
        report = validate(json.loads(args.request), output)
    except Exception as exc:
        report = {"status": "blocked", "message": str(exc)}
    atomic_json_save(output / "report.json", report)


if __name__ == "__main__":
    main()
