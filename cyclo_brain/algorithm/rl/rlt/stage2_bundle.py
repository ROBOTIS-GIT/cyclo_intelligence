"""Self-contained New/Resume bundle contract for compact RLT Stage 2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Literal

import torch
from torch import Tensor

from .rl_token import load_frozen_rl_token_encoder
from .shadow import RLTStage2InferenceSpec, load_groot_rlt_shadow_policy
from .stage1 import _atomic_torch_save
from .stage2 import (
    RLTStage2Config,
    RLTStage2FrozenSource,
    RLTStage2Learner,
    RLTStage2Spec,
    stage2_spec_fingerprint,
)


InitializationMode = Literal["new", "resume"]
_BUNDLE_FORMAT = "cyclo_brain.rlt.stage2_bundle/v1"
_TRAINING_STATE_FORMAT = "cyclo_brain.rlt.stage2_training/v1"
_ACTOR_ARTIFACT_FORMAT = "cyclo_brain.rlt.stage2_actor/v2"
_QUALIFICATION = "training_only_not_deployment_validated"
_ENCODER_RELATIVE = Path("artifacts/rl_token_encoder.pt")
_ACTOR_RELATIVE = Path("artifacts/rlt_actor.pt")
_TRAINING_RELATIVE = Path("training_state/rlt_stage2.pt")
_MANIFEST_RELATIVE = Path("manifest.json")
_MAX_MANIFEST_BYTES = 4 * 1024**2
_MAX_TRAINING_STATE_BYTES = 4 * 1024**3


def _canonical_fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"RLT Stage 2 bundle {name} is invalid")
    return value


def _lexical(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(Path(path).expanduser()))


def _regular_file(path: Path, name: str, *, maximum_bytes: int) -> os.stat_result:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"RLT Stage 2 {name} must be a non-symlink regular file")
    if not 0 < metadata.st_size <= maximum_bytes:
        raise ValueError(f"RLT Stage 2 {name} file size is invalid")
    return metadata


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, relative_path: Path) -> dict[str, Any]:
    metadata = _regular_file(path, str(relative_path), maximum_bytes=8 * 1024**3)
    return {
        "relative_path": relative_path.as_posix(),
        "byte_count": metadata.st_size,
        "sha256": _file_sha256(path),
    }


def _atomic_json_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                dict(payload),
                stream,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


def _atomic_copy(source: Path, destination: Path) -> None:
    source_metadata = _regular_file(
        source, "RL-token source artifact", maximum_bytes=8 * 1024**3
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        destination_metadata = destination.lstat()
        if stat.S_ISLNK(destination_metadata.st_mode):
            raise ValueError("RLT Stage 2 encoder destination must not be a symlink")
        if os.path.samefile(source, destination):
            return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output:
            before = os.fstat(input_stream.fileno())
            shutil.copyfileobj(input_stream, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
            after = os.fstat(input_stream.fileno())
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or before.st_size != source_metadata.st_size:
            raise RuntimeError("RLT Stage 2 encoder source changed while copying")
        os.replace(temporary, destination)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


def _secure_torch_load(path: Path) -> Mapping[str, Any]:
    _regular_file(path, "training checkpoint", maximum_bytes=_MAX_TRAINING_STATE_BYTES)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        payload = torch.load(stream, map_location="cpu", weights_only=True)
        after = os.fstat(stream.fileno())
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError("RLT Stage 2 checkpoint changed while loading")
    if not isinstance(payload, Mapping):
        raise ValueError("RLT Stage 2 checkpoint root must be a mapping")
    return payload


def _load_manifest(path: Path) -> Mapping[str, Any]:
    _regular_file(path, "bundle manifest", maximum_bytes=_MAX_MANIFEST_BYTES)
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError("RLT Stage 2 bundle manifest root must be a mapping")
    return payload


def _artifact_path(root: Path, record: Any, expected: Path, name: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {
        "relative_path",
        "byte_count",
        "sha256",
    }:
        raise ValueError(f"RLT Stage 2 {name} artifact record is invalid")
    if record.get("relative_path") != expected.as_posix():
        raise ValueError(f"RLT Stage 2 {name} artifact path is invalid")
    candidate = root / expected
    actual = _file_record(candidate, expected)
    if dict(record) != actual:
        raise ValueError(f"RLT Stage 2 {name} artifact digest disagrees")
    return candidate


def _tensor_mapping_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return set(left) == set(right) and all(
        isinstance(left[name], Tensor)
        and isinstance(right[name], Tensor)
        and left[name].shape == right[name].shape
        and left[name].dtype == right[name].dtype
        and torch.equal(left[name].detach().cpu(), right[name].detach().cpu())
        for name in left
    )


def _dtype_from_name(value: Any) -> torch.dtype:
    if value == "torch.float32":
        return torch.float32
    if value == "torch.float64":
        return torch.float64
    raise ValueError("RLT Stage 2 checkpoint dtype is unsupported")


@dataclass
class RLTStage2Run:
    """One in-memory Stage-2 learner and its immutable encoder source."""

    learner: RLTStage2Learner
    source: RLTStage2FrozenSource
    encoder_artifact_path: Path
    initialization_mode: InitializationMode
    parent_bundle_fingerprint: str | None = None

    @classmethod
    def new(
        cls,
        encoder_artifact: str | os.PathLike[str],
        *,
        spec: RLTStage2Spec,
        groot_checkpoint: str,
        groot_checkpoint_fingerprint: str,
        representation_contract_fingerprint: str,
        config: RLTStage2Config | None = None,
        actor_hidden_dims: Sequence[int] = (256, 256),
        critic_hidden_dims: Sequence[int] = (256, 256),
        random_seed: int = 0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "RLTStage2Run":
        encoder_path = _lexical(encoder_artifact)
        encoder = load_frozen_rl_token_encoder(encoder_path, device="cpu")
        if encoder.config.embedding_dim != spec.rl_token_dim:
            raise ValueError("RLT Stage 2 encoder width disagrees with spec")
        if encoder.artifact_fingerprint != spec.rl_token_artifact_fingerprint:
            raise ValueError("RLT Stage 2 encoder fingerprint disagrees with spec")
        if (
            encoder.representation_contract_fingerprint
            != representation_contract_fingerprint
        ):
            raise ValueError(
                "RLT Stage 2 encoder and GR00T representation contracts disagree"
            )
        source = RLTStage2FrozenSource(
            groot_checkpoint=groot_checkpoint,
            groot_checkpoint_fingerprint=groot_checkpoint_fingerprint,
            representation_contract_fingerprint=representation_contract_fingerprint,
            rl_token_artifact_fingerprint=encoder.artifact_fingerprint,
        )
        source.validate_spec(spec)
        learner = RLTStage2Learner.create(
            spec,
            config or RLTStage2Config(),
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            random_seed=random_seed,
            device=device,
            dtype=dtype,
        )
        return cls(
            learner=learner,
            source=source,
            encoder_artifact_path=encoder_path,
            initialization_mode="new",
        )

    @classmethod
    def resume(
        cls,
        bundle_root: str | os.PathLike[str],
        *,
        device: str | torch.device = "cpu",
        expected_groot_checkpoint_fingerprint: str | None = None,
    ) -> "RLTStage2Run":
        root = _lexical(bundle_root)
        metadata = root.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("RLT Stage 2 bundle root must be a real directory")
        manifest = _load_manifest(root / _MANIFEST_RELATIVE)
        required_manifest = {
            "format",
            "initialization",
            "source",
            "spec",
            "spec_fingerprint",
            "completed_critic_updates",
            "completed_actor_updates",
            "artifacts",
            "qualification",
            "manifest_fingerprint",
        }
        if (
            set(manifest) != required_manifest
            or manifest.get("format") != _BUNDLE_FORMAT
        ):
            raise ValueError("RLT Stage 2 bundle manifest fields are invalid")
        unsigned = {
            key: value
            for key, value in manifest.items()
            if key != "manifest_fingerprint"
        }
        manifest_fingerprint = _digest(
            manifest.get("manifest_fingerprint"), "manifest fingerprint"
        )
        if _canonical_fingerprint(unsigned) != manifest_fingerprint:
            raise ValueError("RLT Stage 2 bundle manifest fingerprint disagrees")
        if manifest.get("qualification") != _QUALIFICATION:
            raise ValueError("RLT Stage 2 bundle qualification is invalid")
        initialization = manifest.get("initialization")
        if not isinstance(initialization, Mapping) or set(initialization) != {
            "mode",
            "parent_bundle_fingerprint",
        }:
            raise ValueError("RLT Stage 2 bundle initialization is invalid")
        saved_mode = initialization.get("mode")
        saved_parent = initialization.get("parent_bundle_fingerprint")
        if saved_mode == "new":
            if saved_parent is not None:
                raise ValueError("RLT Stage 2 new bundle must not have a parent")
        elif saved_mode == "resume":
            _digest(saved_parent, "parent bundle fingerprint")
        else:
            raise ValueError("RLT Stage 2 bundle initialization mode is invalid")

        raw_spec = manifest.get("spec")
        if not isinstance(raw_spec, Mapping) or set(raw_spec) != {
            field.name for field in fields(RLTStage2InferenceSpec)
        }:
            raise ValueError("RLT Stage 2 bundle spec is invalid")
        spec = RLTStage2Spec(**dict(raw_spec))
        if stage2_spec_fingerprint(spec) != manifest.get("spec_fingerprint"):
            raise ValueError("RLT Stage 2 bundle spec fingerprint disagrees")
        raw_source = manifest.get("source")
        if not isinstance(raw_source, Mapping) or set(raw_source) != {
            field.name for field in fields(RLTStage2FrozenSource)
        }:
            raise ValueError("RLT Stage 2 bundle frozen source is invalid")
        source = RLTStage2FrozenSource(**dict(raw_source))
        source.validate_spec(spec)
        expected_groot = expected_groot_checkpoint_fingerprint
        if expected_groot is not None and source.groot_checkpoint_fingerprint != (
            _digest(expected_groot, "expected GR00T fingerprint")
        ):
            raise ValueError("RLT Stage 2 resume GR00T checkpoint disagrees")

        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != {
            "rl_token_encoder",
            "rlt_actor",
            "training_state",
        }:
            raise ValueError("RLT Stage 2 bundle artifacts are invalid")
        encoder_path = _artifact_path(
            root, artifacts["rl_token_encoder"], _ENCODER_RELATIVE, "encoder"
        )
        actor_path = _artifact_path(
            root,
            artifacts["rlt_actor"],
            _ACTOR_RELATIVE,
            "actor",
        )
        training_path = _artifact_path(
            root, artifacts["training_state"], _TRAINING_RELATIVE, "training state"
        )
        encoder = load_frozen_rl_token_encoder(encoder_path, device="cpu")
        if (
            encoder.artifact_fingerprint != source.rl_token_artifact_fingerprint
            or encoder.representation_contract_fingerprint
            != source.representation_contract_fingerprint
            or encoder.config.embedding_dim != spec.rl_token_dim
        ):
            raise ValueError("RLT Stage 2 bundled encoder contract disagrees")

        checkpoint = _secure_torch_load(training_path)
        if set(checkpoint) != {"format", "source", "learner"} or checkpoint.get(
            "format"
        ) != _TRAINING_STATE_FORMAT:
            raise ValueError("RLT Stage 2 training checkpoint fields are invalid")
        if checkpoint.get("source") != asdict(source):
            raise ValueError("RLT Stage 2 training checkpoint source disagrees")
        learner_state = checkpoint.get("learner")
        if not isinstance(learner_state, Mapping):
            raise ValueError("RLT Stage 2 learner checkpoint is invalid")
        contract = learner_state.get("contract")
        if not isinstance(contract, Mapping):
            raise ValueError("RLT Stage 2 learner contract is invalid")
        if contract.get("spec") != asdict(spec) or contract.get(
            "spec_fingerprint"
        ) != stage2_spec_fingerprint(spec):
            raise ValueError("RLT Stage 2 learner spec disagrees")
        try:
            config = RLTStage2Config(**dict(contract["config"]))
            learner = RLTStage2Learner.create(
                spec,
                config,
                actor_hidden_dims=tuple(contract["actor_hidden_dims"]),
                critic_hidden_dims=tuple(contract["critic_hidden_dims"]),
                random_seed=contract["random_seed"],
                device=device,
                dtype=_dtype_from_name(contract["dtype"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "RLT Stage 2 learner construction contract is invalid"
            ) from error
        learner.load_state_dict(learner_state)
        if (
            manifest.get("completed_critic_updates")
            != learner.completed_critic_updates
            or manifest.get("completed_actor_updates")
            != learner.completed_actor_updates
        ):
            raise ValueError("RLT Stage 2 bundle progress disagrees")

        shadow = load_groot_rlt_shadow_policy(
            encoder_path,
            actor_path,
            device=device,
            dtype=learner.dtype,
            expected_chunk_length=10,
            expected_action_dim=19,
        )
        if shadow.spec != spec or not _tensor_mapping_equal(
            shadow.actor.state_dict(), learner.actor.state_dict()
        ):
            raise ValueError("RLT Stage 2 actor artifact and training state disagree")
        return cls(
            learner=learner,
            source=source,
            encoder_artifact_path=encoder_path,
            initialization_mode="resume",
            parent_bundle_fingerprint=manifest_fingerprint,
        )

    def _actor_artifact(self) -> dict[str, Any]:
        source_fingerprint = _canonical_fingerprint(
            {"source": asdict(self.source), "spec": asdict(self.learner.spec)}
        )
        return {
            "format": _ACTOR_ARTIFACT_FORMAT,
            "spec": asdict(self.learner.spec),
            "spec_fingerprint": stage2_spec_fingerprint(self.learner.spec),
            "config": asdict(self.learner.config),
            "actor_hidden_dims": tuple(self.learner.actor.hidden_dims),
            "completed_critic_updates": self.learner.completed_critic_updates,
            "completed_actor_updates": self.learner.completed_actor_updates,
            "replay_artifact": {"contract": "precomputed_frozen_features/v1"},
            "source_manifest_fingerprint": source_fingerprint,
            "actor": {
                name: value.detach().cpu().clone()
                for name, value in self.learner.actor.state_dict().items()
            },
            "diagnostic_count": 0,
            "last_diagnostic": None,
            "qualification": _QUALIFICATION,
        }

    def save(self, bundle_root: str | os.PathLike[str]) -> Path:
        """Atomically publish artifacts first and the authoritative manifest last."""

        self.source.validate_spec(self.learner.spec)
        root = _lexical(bundle_root)
        if os.path.lexists(root):
            metadata = root.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("RLT Stage 2 bundle output must be a real directory")
        root.mkdir(parents=True, exist_ok=True)
        encoder_path = root / _ENCODER_RELATIVE
        actor_path = root / _ACTOR_RELATIVE
        training_path = root / _TRAINING_RELATIVE
        _atomic_copy(self.encoder_artifact_path, encoder_path)
        encoder = load_frozen_rl_token_encoder(encoder_path, device="cpu")
        if (
            encoder.artifact_fingerprint != self.source.rl_token_artifact_fingerprint
            or encoder.representation_contract_fingerprint
            != self.source.representation_contract_fingerprint
        ):
            raise RuntimeError("RLT Stage 2 copied encoder verification failed")
        _atomic_torch_save(actor_path, self._actor_artifact())
        _atomic_torch_save(
            training_path,
            {
                "format": _TRAINING_STATE_FORMAT,
                "source": asdict(self.source),
                "learner": self.learner.state_dict(),
            },
        )
        initialization = {
            "mode": self.initialization_mode,
            "parent_bundle_fingerprint": self.parent_bundle_fingerprint,
        }
        unsigned = {
            "format": _BUNDLE_FORMAT,
            "initialization": initialization,
            "source": asdict(self.source),
            "spec": asdict(self.learner.spec),
            "spec_fingerprint": stage2_spec_fingerprint(self.learner.spec),
            "completed_critic_updates": self.learner.completed_critic_updates,
            "completed_actor_updates": self.learner.completed_actor_updates,
            "artifacts": {
                "rl_token_encoder": _file_record(encoder_path, _ENCODER_RELATIVE),
                "rlt_actor": _file_record(actor_path, _ACTOR_RELATIVE),
                "training_state": _file_record(training_path, _TRAINING_RELATIVE),
            },
            "qualification": _QUALIFICATION,
        }
        manifest = {
            **unsigned,
            "manifest_fingerprint": _canonical_fingerprint(unsigned),
        }
        _atomic_json_save(root / _MANIFEST_RELATIVE, manifest)
        # Full round-trip verification is intentionally part of save.
        RLTStage2Run.resume(
            root,
            device=self.learner.device,
            expected_groot_checkpoint_fingerprint=(
                self.source.groot_checkpoint_fingerprint
            ),
        )
        self.encoder_artifact_path = encoder_path
        return root


__all__ = ["InitializationMode", "RLTStage2Run"]
