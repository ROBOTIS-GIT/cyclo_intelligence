"""Evaluate a completed MultiTaskDiT value warm-up bundle deterministically."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from cyclo_brain.model.multi_task_dit.checkpoint_validation import (
    assert_deployment_artifacts,
    validate_policy_contract,
)
from cyclo_brain.model.multi_task_dit.flow_sde_adapter import (
    CYCLO_SG2_CAMERA_KEYS,
    MultiTaskDiTFlowAdapter,
)
from cyclo_brain.model.multi_task_dit.success_dataset import discover_episode_outcomes
from cyclo_brain.model.multi_task_dit.value_head import MultiTaskDiTValueHead

from .value_warmup import (
    VALUE_WARMUP_FORMAT,
    EpisodeBalancedChunkBoundaryDataset,
    module_sha256,
)
from .value_warmup_cli import (
    BUNDLE_FORMAT,
    REQUIRED_POLICY_ARTIFACTS,
    _canonical_observation_batch,
    _dataset_identity,
    _resolve_dataset_root,
    _resolve_device,
    _sha256_file,
)
from .value_warmup_eval import (
    CURRENT_VALUE_HEAD_HIDDEN_DIMS,
    VALUE_WARMUP_EVALUATION_FORMAT,
    assert_exact_value_head_reload,
    evaluate_value_predictions,
    samples_from_records,
    validate_current_value_head_state_dict,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument(
        "--dataset-root",
        action="append",
        type=Path,
        help=(
            "Repeat in manifest order. If omitted, use the immutable dataset paths "
            "recorded by the warm-up bundle."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--video-backend",
        choices=("pyav", "torchcodec", "video_reader"),
        default="pyav",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional atomic sidecar path outside the immutable warm-up bundle.",
    )
    return parser


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is invalid: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain a JSON object")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _validate_bundle_manifest(bundle: Path, manifest: Mapping[str, Any]) -> None:
    if manifest.get("format") != BUNDLE_FORMAT or manifest.get("status") != "complete":
        raise ValueError("value warm-up bundle must have the complete v1 format")
    artifacts = _require_mapping(manifest.get("artifacts"), name="manifest artifacts")
    expected_artifacts = {
        "model_path": "pretrained_model",
        "checkpoint_path": "training_state/value_warmup.pt",
        "progress_path": "progress.jsonl",
    }
    if dict(artifacts) != expected_artifacts:
        raise ValueError("value warm-up manifest artifact contract changed")
    for relative in expected_artifacts.values():
        if not (bundle / relative).is_file() and relative != "pretrained_model":
            raise FileNotFoundError(f"value warm-up artifact is missing: {bundle / relative}")
    if not (bundle / "pretrained_model").is_dir():
        raise FileNotFoundError("value warm-up pretrained_model directory is missing")

    config = _require_mapping(manifest.get("config"), name="manifest config")
    result = _require_mapping(manifest.get("result"), name="manifest result")
    if result.get("completed_steps") != config.get("steps"):
        raise ValueError("value warm-up did not complete its configured optimizer steps")
    base = _require_mapping(manifest.get("base"), name="manifest base")
    policy_hash = base.get("policy_sha256")
    if not isinstance(policy_hash, str) or not policy_hash.startswith("sha256:"):
        raise ValueError("manifest base policy hash is invalid")
    if (
        result.get("policy_sha256_before") != policy_hash
        or result.get("policy_sha256_after") != policy_hash
    ):
        raise ValueError("manifest reports a policy mutation during warm-up")


def _validate_policy_artifacts(pretrained_dir: Path, manifest: Mapping[str, Any]) -> None:
    base = _require_mapping(manifest["base"], name="manifest base")
    expected = _require_mapping(base.get("artifacts"), name="manifest base artifacts")
    if set(expected) != set(REQUIRED_POLICY_ARTIFACTS):
        raise ValueError("manifest policy artifact set changed")
    for name in REQUIRED_POLICY_ARTIFACTS:
        if _sha256_file(pretrained_dir / name) != expected[name]:
            raise RuntimeError(f"copied policy artifact hash mismatch: {name}")


def _validate_checkpoint(
    checkpoint: Mapping[str, Any], manifest: Mapping[str, Any]
) -> Mapping[str, torch.Tensor]:
    if checkpoint.get("format") != VALUE_WARMUP_FORMAT or checkpoint.get("status") != "complete":
        raise ValueError("value warm-up checkpoint must have the complete v1 format")
    if checkpoint.get("config") != manifest.get("config"):
        raise ValueError("checkpoint and manifest warm-up configs differ")
    if checkpoint.get("dataset_contract") != manifest.get("dataset_contract"):
        raise ValueError("checkpoint and manifest dataset contracts differ")
    if checkpoint.get("base_identity") != manifest.get("base"):
        raise ValueError("checkpoint and manifest base identities differ")
    if checkpoint.get("dataset_identities") != manifest.get("datasets"):
        raise ValueError("checkpoint and manifest dataset identities differ")
    if checkpoint.get("completed_steps") != manifest["result"].get("completed_steps"):
        raise ValueError("checkpoint and manifest completed steps differ")
    expected_hash = manifest["base"].get("policy_sha256")
    if (
        checkpoint.get("policy_sha256_before") != expected_hash
        or checkpoint.get("policy_sha256_after") != expected_hash
    ):
        raise ValueError("checkpoint reports a policy mutation during warm-up")
    optimizer = _require_mapping(checkpoint.get("value_optimizer"), name="value optimizer")
    if not optimizer.get("state") or not optimizer.get("param_groups"):
        raise ValueError("value warm-up optimizer state is empty")
    state = _require_mapping(checkpoint.get("value_head"), name="value head state")
    return state  # type: ignore[return-value]


def _resolve_roots(
    requested: Sequence[Path] | None, manifest: Mapping[str, Any]
) -> tuple[Path, ...]:
    identities = manifest.get("datasets")
    if not isinstance(identities, list) or not identities:
        raise ValueError("manifest dataset identities are missing")
    values = requested or [Path(item["path"]) for item in identities]
    roots = tuple(_resolve_dataset_root(value) for value in values)
    if len(roots) != len(identities) or len(set(roots)) != len(roots):
        raise ValueError("evaluation dataset roots must uniquely match the manifest order")
    return roots


def _validate_dataset_identities(
    roots: Sequence[Path], manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    expected = manifest["datasets"]
    actual = [
        _dataset_identity(root, discover_episode_outcomes(root)) for root in roots
    ]
    if actual != expected:
        raise RuntimeError(
            "evaluation datasets no longer exactly match the warm-up manifest identities"
        )
    return actual


def _validate_output_path(bundle: Path, raw_path: Path | None) -> Path | None:
    if raw_path is None:
        return None
    output = raw_path.expanduser().resolve(strict=False)
    if output == bundle or bundle in output.parents:
        raise ValueError("output-json must be outside the immutable warm-up bundle")
    if output.exists() and not output.is_file():
        raise ValueError("output-json must be a file path")
    return output


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if isinstance(args.batch_size, bool) or not isinstance(args.batch_size, int) or args.batch_size < 1:
        raise ValueError("evaluation batch-size must be a positive integer")
    bundle = args.bundle.expanduser().resolve(strict=True)
    if not bundle.is_dir():
        raise NotADirectoryError(bundle)
    output_json = _validate_output_path(bundle, args.output_json)
    manifest = _read_json(bundle / "run_manifest.json", name="run manifest")
    _validate_bundle_manifest(bundle, manifest)
    pretrained_dir = assert_deployment_artifacts(bundle / "pretrained_model")
    _validate_policy_artifacts(pretrained_dir, manifest)
    roots = _resolve_roots(args.dataset_root, manifest)
    dataset_identities = _validate_dataset_identities(roots, manifest)

    checkpoint_path = bundle / "training_state" / "value_warmup.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("value warm-up checkpoint must contain a mapping")
    value_state = _validate_checkpoint(checkpoint, manifest)

    from torchvision.transforms.v2 import Resize

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import get_policy_class, make_pre_post_processors

    device = _resolve_device(args.device)
    config_payload = _read_json(pretrained_dir / "config.json", name="policy config")
    policy_type = config_payload.get("type")
    if policy_type != "multi_task_dit":
        raise ValueError(f"value evaluation requires type='multi_task_dit', got {policy_type!r}")
    policy = get_policy_class(policy_type).from_pretrained(str(pretrained_dir)).to(device).eval()
    policy.requires_grad_(False)
    policy_contract = validate_policy_contract(policy)
    policy_hash_before = module_sha256(policy)
    if policy_hash_before != manifest["base"]["policy_sha256"]:
        raise RuntimeError("loaded evaluation policy does not match the warm-up base hash")
    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(pretrained_dir),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    datasets = tuple(
        LeRobotDataset(
            repo_id=f"local/value-warmup-eval-{index}-{root.name}",
            root=root,
            image_transforms=Resize((256, 256), antialias=True),
            delta_timestamps=None,
            force_cache_sync=False,
            download_videos=False,
            video_backend=args.video_backend,
            return_uint8=True,
        )
        for index, root in enumerate(roots)
    )
    observation_keys = ("observation.state", *CYCLO_SG2_CAMERA_KEYS)
    warmup_config = manifest["config"]
    dataset = EpisodeBalancedChunkBoundaryDataset(
        datasets,
        observation_keys=observation_keys,
        n_action_steps=int(policy.config.n_action_steps),
        gamma=float(warmup_config["gamma"]),
        dataset_names=tuple(str(root) for root in roots),
    )
    if dataset.contract() != manifest.get("dataset_contract"):
        raise RuntimeError("reconstructed evaluation dataset contract differs from the warm-up")

    adapter = MultiTaskDiTFlowAdapter(policy, freeze_observation_encoder=True)
    architecture = validate_current_value_head_state_dict(
        value_state, conditioning_dim=adapter.conditioning_dim
    )
    value_head = MultiTaskDiTValueHead(
        adapter.conditioning_dim, hidden_dims=CURRENT_VALUE_HEAD_HIDDEN_DIMS
    ).to(device)
    value_head.load_state_dict(value_state, strict=True)
    value_head.eval()

    conditioning_batches: list[torch.Tensor] = []
    prediction_batches: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(dataset.records), args.batch_size):
            stop = min(start + args.batch_size, len(dataset.records))
            observations, _targets = dataset.collate(tuple(range(start, stop)))
            raw = _canonical_observation_batch(
                observations,
                task_instruction=str(warmup_config["task_instruction"]),
                n_obs_steps=int(policy.config.n_obs_steps),
            )
            conditioning = adapter.encode_conditioning(preprocessor(raw))
            predictions = value_head(conditioning)
            conditioning_batches.append(conditioning.detach().cpu())
            prediction_batches.append(predictions.detach().cpu())

    all_predictions = torch.cat(prediction_batches)
    samples = samples_from_records(dataset.records, all_predictions)
    metrics = evaluate_value_predictions(samples)
    reload_contract = assert_exact_value_head_reload(
        value_state,
        conditioning_batches,
        prediction_batches,
        conditioning_dim=adapter.conditioning_dim,
        device=device,
    )
    policy_hash_after = module_sha256(policy)
    if policy_hash_after != policy_hash_before:
        raise RuntimeError("MultiTaskDiT policy changed during value evaluation")

    result = {
        "format": VALUE_WARMUP_EVALUATION_FORMAT,
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "training_dataset_diagnostic",
        "bundle_path": str(bundle),
        "integrity": {
            "bundle_format": manifest["format"],
            "bundle_status": manifest["status"],
            "completed_steps": checkpoint["completed_steps"],
            "policy_sha256_before": policy_hash_before,
            "policy_sha256_after": policy_hash_after,
            "policy_unchanged": True,
            "policy_contract": policy_contract,
            "dataset_identities": dataset_identities,
            "value_head": reload_contract,
            "optimizer_state_present": True,
        },
        "metrics": metrics,
        "limitations": [
            "This is an in-sample fit and reload diagnostic, not held-out performance.",
            "Success and failure episodes were collected in separate dataset roots.",
            architecture["technical_debt"],
        ],
    }
    if output_json is not None:
        _atomic_json(output_json, result)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    result = run_from_args(build_parser().parse_args(argv))
    print(json.dumps(result, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
