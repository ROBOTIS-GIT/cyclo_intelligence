"""Content identity for the exact local inputs consumed by ACT-TD3 training.

The identity deliberately excludes recording annotations, conversion caches,
``meta/frame_reuse.parquet``, and ``train_config.json`` because the offline
adapter and ACT checkpoint loader do not consume them.  Conversely, it binds
the selected episodes to every parquet/video file resolved through the loaded
LeRobot metadata object, the exact ACT policy/processor assets, and the robot
configuration plus its resolved URDF.

This module is intentionally independent of LeRobot and torch imports.  The
public builder accepts the already-loaded, LeRobotDataset-like and
ACTPhysicalActionDomain-like objects used by the caller.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any


_IDENTITY_SCHEMA = "cyclo.act_td3.training_data_identity.v1"
_MULTI_IDENTITY_SCHEMA = "cyclo.act_td3.training_data_identity.multi_root.v1"
_HASH_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ACTTD3TrainingIdentityFile:
    """One selected file in a training-data identity manifest."""

    component: str
    path: str
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class ACTTD3TrainingDataIdentity:
    """Frozen, JSON-friendly description of all identity-bound inputs."""

    identity: str
    file_count: int
    byte_count: int
    component_sha256: dict[str, str]
    manifest: tuple[ACTTD3TrainingIdentityFile, ...]
    virtual_contract: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return an ordinary JSON-serializable copy for logs and CLI output."""

        return asdict(self)


@dataclass(frozen=True)
class _SecureRoot:
    path: Path
    label: str


@dataclass(frozen=True)
class _SelectedFile:
    component: str
    logical_path: str
    path: Path


@dataclass(frozen=True)
class _FileSnapshot:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class _ResolvedInputs:
    files: tuple[_SelectedFile, ...]
    virtual_contract: dict[str, Any]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _prefixed_sha256(value: bytes) -> str:
    return f"sha256:{_sha256_bytes(value)}"


def _snapshot(file_stat: os.stat_result) -> _FileSnapshot:
    return _FileSnapshot(
        device=file_stat.st_dev,
        inode=file_stat.st_ino,
        mode=file_stat.st_mode,
        size=file_stat.st_size,
        mtime_ns=file_stat.st_mtime_ns,
        ctime_ns=file_stat.st_ctime_ns,
    )


def _secure_root(value: str | Path, *, label: str) -> _SecureRoot:
    lexical = Path(value).expanduser()
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    try:
        lexical_stat = os.lstat(lexical)
    except OSError as error:
        raise FileNotFoundError(f"{label} does not exist: {lexical}") from error
    if stat.S_ISLNK(lexical_stat.st_mode):
        raise ValueError(f"{label} must not be a symbolic link: {lexical}")
    if not stat.S_ISDIR(lexical_stat.st_mode):
        raise NotADirectoryError(f"{label} is not a directory: {lexical}")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise FileNotFoundError(f"{label} cannot be resolved: {lexical}") from error
    return _SecureRoot(path=resolved, label=label)


def _relative_path(value: str | Path, *, source: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"{source} returned a path outside the dataset root: {value}")
    if any(part in {"", "."} for part in relative.parts):
        raise ValueError(f"{source} returned a non-canonical relative path: {value}")
    return relative


def _secure_child(root: _SecureRoot, relative: str | Path, *, source: str) -> Path:
    relative_path = _relative_path(relative, source=source)
    current = root.path
    for part in relative_path.parts:
        current = current / part
        try:
            current_stat = os.lstat(current)
        except OSError as error:
            raise FileNotFoundError(
                f"{source} selected a missing file: {relative_path.as_posix()}"
            ) from error
        if stat.S_ISLNK(current_stat.st_mode):
            raise ValueError(
                f"{source} selected a symbolic link: {relative_path.as_posix()}"
            )
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(root.path)
    except ValueError as error:
        raise ValueError(
            f"{source} selected a path outside {root.label}: {relative_path.as_posix()}"
        ) from error
    resolved_stat = os.lstat(resolved)
    if not stat.S_ISREG(resolved_stat.st_mode):
        raise ValueError(
            f"{source} selected a non-regular file: {relative_path.as_posix()}"
        )
    return resolved


def _optional_child_exists(root: _SecureRoot, relative: str | Path, *, source: str) -> bool:
    relative_path = _relative_path(relative, source=source)
    candidate = root.path / relative_path
    if not os.path.lexists(candidate):
        return False
    _secure_child(root, relative_path, source=source)
    return True


def _secure_external_file(value: str | Path, *, source: str) -> Path:
    lexical = Path(value).expanduser()
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    try:
        lexical_stat = os.lstat(lexical)
    except OSError as error:
        raise FileNotFoundError(f"{source} does not exist: {lexical}") from error
    if stat.S_ISLNK(lexical_stat.st_mode):
        raise ValueError(f"{source} must not be a symbolic link: {lexical}")
    if not stat.S_ISREG(lexical_stat.st_mode):
        raise ValueError(f"{source} must be a regular file: {lexical}")
    return lexical.resolve(strict=True)


def _open_stable_file(path: Path) -> tuple[bytes, _FileSnapshot]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"identity input is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, _HASH_CHUNK_BYTES)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_snapshot = _snapshot(before)
    if before_snapshot != _snapshot(after):
        raise RuntimeError(f"identity input changed while it was read: {path}")
    current = os.lstat(path)
    if stat.S_ISLNK(current.st_mode) or before_snapshot != _snapshot(current):
        raise RuntimeError(f"identity input was replaced while it was read: {path}")
    return b"".join(chunks), before_snapshot


def _hash_stable_file(path: Path) -> tuple[str, int, _FileSnapshot]:
    """Hash a file incrementally while enforcing the same stability checks."""

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"identity input is not a regular file: {path}")
        while True:
            chunk = os.read(descriptor, _HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_snapshot = _snapshot(before)
    if before_snapshot != _snapshot(after):
        raise RuntimeError(f"identity input changed while it was hashed: {path}")
    current = os.lstat(path)
    if stat.S_ISLNK(current.st_mode) or before_snapshot != _snapshot(current):
        raise RuntimeError(f"identity input was replaced while it was hashed: {path}")
    return digest.hexdigest(), byte_count, before_snapshot


def _read_json(path: Path, *, source: str) -> Mapping[str, Any]:
    payload, _ = _open_stable_file(path)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{source} is not valid JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{source} JSON root must be an object: {path}")
    return value


def _recursive_parquet_files(root: _SecureRoot, relative_dir: Path) -> list[Path]:
    directory = root.path / relative_dir
    try:
        directory_stat = os.lstat(directory)
    except OSError as error:
        raise FileNotFoundError(
            f"LeRobot episode metadata directory is missing: {relative_dir.as_posix()}"
        ) from error
    if stat.S_ISLNK(directory_stat.st_mode):
        raise ValueError("LeRobot episode metadata directory must not be a symbolic link")
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise NotADirectoryError(
            f"LeRobot episode metadata path is not a directory: {relative_dir.as_posix()}"
        )

    result: list[Path] = []
    pending = [directory]
    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise ValueError(
                        "LeRobot episode metadata must not contain symbolic links: "
                        f"{Path(entry.path).relative_to(root.path).as_posix()}"
                    )
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.name.endswith(".parquet"):
                    result.append(Path(entry.path).relative_to(root.path))
    if not result:
        raise FileNotFoundError("LeRobot dataset has no meta/episodes parquet files")
    return sorted(result, key=lambda path: path.as_posix())


def _add_selection(
    selections: dict[tuple[str, str], _SelectedFile],
    *,
    component: str,
    logical_path: str,
    path: Path,
) -> None:
    key = (component, logical_path)
    selected = _SelectedFile(component, logical_path, path)
    previous = selections.get(key)
    if previous is not None and previous.path != path:
        raise ValueError(f"identity manifest path collision: {component}/{logical_path}")
    selections[key] = selected


def _add_root_file(
    selections: dict[tuple[str, str], _SelectedFile],
    root: _SecureRoot,
    relative: str | Path,
    *,
    component: str,
    source: str,
) -> Path:
    relative_path = _relative_path(relative, source=source)
    selected = _secure_child(root, relative_path, source=source)
    _add_selection(
        selections,
        component=component,
        logical_path=relative_path.as_posix(),
        path=selected,
    )
    return selected


def _checked_episode_indices(dataset: Any, meta: Any) -> tuple[int, ...]:
    total = getattr(meta, "total_episodes", None)
    if isinstance(total, bool) or not isinstance(total, Integral) or int(total) < 1:
        raise TypeError("LeRobot metadata total_episodes must be a positive integer")
    total_episodes = int(total)
    raw = getattr(dataset, "episodes", None)
    if raw is None:
        return tuple(range(total_episodes))
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError("LeRobot dataset episodes must be a sequence of indices or None")
    result: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("LeRobot episode indices must be integers")
        index = int(value)
        if index < 0 or index >= total_episodes:
            raise ValueError(f"LeRobot episode index is out of range: {index}")
        result.append(index)
    if not result:
        raise ValueError("ACT-TD3 requires at least one selected episode")
    if len(set(result)) != len(result):
        raise ValueError("LeRobot selected episode indices must be unique")
    return tuple(result)


def _checked_strings(value: Any, *, source: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{source} must be a sequence")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{source} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{source} must contain unique values")
    return result


def _plain_vector(value: Any, *, source: str) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{source} must be a one-dimensional sequence")
    result = list(value)
    if any(isinstance(item, Sequence) and not isinstance(item, (str, bytes)) for item in result):
        raise ValueError(f"{source} must be one-dimensional")
    return result


def _action_domain_contract(action_domain: Any) -> dict[str, Any]:
    names = _checked_strings(getattr(action_domain, "names", None), source="ACT action names")
    groups = _checked_strings(
        getattr(action_domain, "action_groups", None),
        source="ACT action groups",
    )
    passthrough_values = _plain_vector(
        getattr(action_domain, "passthrough_mask", None),
        source="ACT passthrough mask",
    )
    if any(not isinstance(value, bool) for value in passthrough_values):
        raise TypeError("ACT passthrough mask values must be boolean")
    low_values = _plain_vector(
        getattr(action_domain, "physical_low", None),
        source="ACT physical lower bounds",
    )
    high_values = _plain_vector(
        getattr(action_domain, "physical_high", None),
        source="ACT physical upper bounds",
    )
    if not (len(names) == len(passthrough_values) == len(low_values) == len(high_values)):
        raise ValueError("ACT action-domain vectors must have the same dimension")

    def exact_floats(values: list[Any], source: str) -> list[str]:
        result: list[str] = []
        for value in values:
            if isinstance(value, bool):
                raise TypeError(f"{source} values must be numeric")
            try:
                converted = float(value)
            except (TypeError, ValueError) as error:
                raise TypeError(f"{source} values must be numeric") from error
            if not math.isfinite(converted):
                raise ValueError(f"{source} values must be finite")
            result.append(converted.hex())
        return result

    return {
        "action_names": list(names),
        "action_groups": list(groups),
        "passthrough_mask": passthrough_values,
        "physical_low_float_hex": exact_floats(low_values, "ACT physical lower bounds"),
        "physical_high_float_hex": exact_floats(high_values, "ACT physical upper bounds"),
    }


def _validate_loaded_roots(dataset: Any, meta: Any, root: _SecureRoot) -> None:
    for owner, label in ((dataset, "LeRobot dataset root"), (meta, "LeRobot metadata root")):
        declared = getattr(owner, "root", None)
        if declared is None:
            raise TypeError(f"{label} is not available on the loaded dataset")
        try:
            resolved = Path(declared).expanduser().resolve(strict=True)
        except OSError as error:
            raise FileNotFoundError(f"{label} cannot be resolved: {declared}") from error
        if resolved != root.path:
            raise ValueError(
                f"{label} disagrees with dataset_root: {resolved} != {root.path}"
            )


def _checkpoint_selections(
    checkpoint_root: _SecureRoot,
    selections: dict[tuple[str, str], _SelectedFile],
) -> None:
    component = "act_checkpoint"
    _add_root_file(
        selections,
        checkpoint_root,
        "config.json",
        component=component,
        source="ACT checkpoint config",
    )

    monolithic = _optional_child_exists(
        checkpoint_root,
        "model.safetensors",
        source="ACT checkpoint weights",
    )
    sharded = _optional_child_exists(
        checkpoint_root,
        "model.safetensors.index.json",
        source="ACT checkpoint weight index",
    )
    if monolithic == sharded:
        if monolithic:
            raise ValueError(
                "ACT checkpoint ambiguously contains both monolithic and sharded safetensors"
            )
        raise FileNotFoundError(
            "ACT checkpoint must contain model.safetensors or model.safetensors.index.json"
        )
    if monolithic:
        _add_root_file(
            selections,
            checkpoint_root,
            "model.safetensors",
            component=component,
            source="ACT checkpoint weights",
        )
    else:
        index_path = _add_root_file(
            selections,
            checkpoint_root,
            "model.safetensors.index.json",
            component=component,
            source="ACT checkpoint weight index",
        )
        index = _read_json(index_path, source="ACT checkpoint weight index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ValueError("ACT checkpoint weight index must define a non-empty weight_map")
        shard_names = set()
        for parameter_name, shard_name in weight_map.items():
            if not isinstance(parameter_name, str) or not parameter_name:
                raise ValueError("ACT checkpoint weight_map keys must be non-empty strings")
            if not isinstance(shard_name, str) or not shard_name:
                raise ValueError("ACT checkpoint weight_map values must be non-empty strings")
            if not shard_name.endswith(".safetensors"):
                raise ValueError("ACT checkpoint weight_map must reference safetensors shards")
            shard_names.add(shard_name)
        for shard_name in sorted(shard_names):
            _add_root_file(
                selections,
                checkpoint_root,
                shard_name,
                component=component,
                source="ACT checkpoint weight shard",
            )

    for processor_name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        processor_path = _add_root_file(
            selections,
            checkpoint_root,
            processor_name,
            component=component,
            source="ACT checkpoint processor config",
        )
        processor = _read_json(processor_path, source="ACT checkpoint processor config")
        steps = processor.get("steps")
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            raise ValueError(f"ACT processor config must define a steps array: {processor_name}")
        for step in steps:
            if not isinstance(step, Mapping):
                raise ValueError(f"ACT processor steps must be objects: {processor_name}")
            if "state_file" not in step:
                continue
            state_file = step["state_file"]
            if not isinstance(state_file, str) or not state_file:
                raise ValueError(
                    f"ACT processor state_file must be a non-empty string: {processor_name}"
                )
            _add_root_file(
                selections,
                checkpoint_root,
                state_file,
                component=component,
                source="ACT checkpoint processor state",
            )


def _resolve_inputs(
    dataset: Any,
    dataset_root_value: str | Path,
    act_checkpoint_root_value: str | Path,
    action_domain: Any,
    *,
    robot_type: str,
    video_backend: str,
) -> _ResolvedInputs:
    if not isinstance(robot_type, str) or not robot_type:
        raise ValueError("robot_type must be a non-empty string")
    if not isinstance(video_backend, str) or not video_backend:
        raise ValueError("video_backend must be a non-empty string")

    dataset_root = _secure_root(dataset_root_value, label="LeRobot dataset root")
    checkpoint_root = _secure_root(act_checkpoint_root_value, label="ACT checkpoint root")
    meta = getattr(dataset, "meta", None)
    if meta is None:
        raise TypeError("loaded LeRobot dataset must expose metadata")
    _validate_loaded_roots(dataset, meta, dataset_root)

    loaded_backend = getattr(dataset, "_video_backend", video_backend)
    if loaded_backend != video_backend:
        raise ValueError(
            "video_backend disagrees with the loaded LeRobot dataset: "
            f"{video_backend!r} != {loaded_backend!r}"
        )
    metadata_robot_type = getattr(meta, "robot_type", None)
    if metadata_robot_type is not None and metadata_robot_type != robot_type:
        raise ValueError(
            "robot_type disagrees with LeRobot metadata: "
            f"{robot_type!r} != {metadata_robot_type!r}"
        )

    selections: dict[tuple[str, str], _SelectedFile] = {}
    info_path = _add_root_file(
        selections,
        dataset_root,
        "meta/info.json",
        component="dataset",
        source="LeRobot info metadata",
    )
    _add_root_file(
        selections,
        dataset_root,
        "meta/tasks.parquet",
        component="dataset",
        source="LeRobot tasks metadata",
    )
    for optional in ("meta/stats.json", "meta/subtasks.parquet"):
        if _optional_child_exists(dataset_root, optional, source="LeRobot optional metadata"):
            _add_root_file(
                selections,
                dataset_root,
                optional,
                component="dataset",
                source="LeRobot optional metadata",
            )
    for relative in _recursive_parquet_files(dataset_root, Path("meta/episodes")):
        _add_root_file(
            selections,
            dataset_root,
            relative,
            component="dataset",
            source="LeRobot episode metadata",
        )

    domain_info_path = _secure_external_file(
        getattr(action_domain, "dataset_info_path", ""),
        source="ACT action-domain dataset info",
    )
    if domain_info_path != info_path:
        raise ValueError(
            "ACT action domain was resolved from different dataset info metadata"
        )

    episode_indices = _checked_episode_indices(dataset, meta)
    get_data_file_path = getattr(meta, "get_data_file_path", None)
    get_video_file_path = getattr(meta, "get_video_file_path", None)
    if not callable(get_data_file_path) or not callable(get_video_file_path):
        raise TypeError("LeRobot metadata must expose data/video path resolvers")
    video_keys = _checked_strings(
        getattr(meta, "video_keys", None),
        source="LeRobot video keys",
    )
    for episode_index in episode_indices:
        _add_root_file(
            selections,
            dataset_root,
            get_data_file_path(episode_index),
            component="dataset",
            source=f"LeRobot episode {episode_index} data resolver",
        )
        for video_key in video_keys:
            _add_root_file(
                selections,
                dataset_root,
                get_video_file_path(episode_index, video_key),
                component="dataset",
                source=f"LeRobot episode {episode_index} video resolver",
            )

    _checkpoint_selections(checkpoint_root, selections)

    robot_config_path = _secure_external_file(
        getattr(action_domain, "robot_config_path", ""),
        source="ACT robot config",
    )
    urdf_path = _secure_external_file(
        getattr(action_domain, "urdf_path", ""),
        source="ACT robot URDF",
    )
    _add_selection(
        selections,
        component="robot",
        logical_path=f"robot_config/{robot_config_path.name}",
        path=robot_config_path,
    )
    _add_selection(
        selections,
        component="robot",
        logical_path=f"urdf/{urdf_path.name}",
        path=urdf_path,
    )

    virtual_contract = {
        "episode_indices": list(episode_indices),
        "robot_type": robot_type,
        "video_backend": video_backend,
        "video_keys": list(video_keys),
        **_action_domain_contract(action_domain),
    }
    ordered = tuple(
        sorted(
            selections.values(),
            key=lambda selected: (selected.component, selected.logical_path),
        )
    )
    return _ResolvedInputs(files=ordered, virtual_contract=virtual_contract)


def _hash_selected_file(
    selected: _SelectedFile,
) -> tuple[ACTTD3TrainingIdentityFile, _FileSnapshot]:
    digest, byte_count, snapshot = _hash_stable_file(selected.path)
    return (
        ACTTD3TrainingIdentityFile(
            component=selected.component,
            path=selected.logical_path,
            byte_count=byte_count,
            sha256=f"sha256:{digest}",
        ),
        snapshot,
    )


def _assert_snapshot(path: Path, expected: _FileSnapshot) -> None:
    try:
        current = os.lstat(path)
    except OSError as error:
        raise RuntimeError(f"identity input disappeared while hashing: {path}") from error
    if stat.S_ISLNK(current.st_mode) or _snapshot(current) != expected:
        raise RuntimeError(f"identity input changed while hashing: {path}")


def build_act_td3_training_data_identity(
    dataset: Any,
    dataset_root: str | Path,
    act_checkpoint_root: str | Path,
    action_domain: Any,
    *,
    robot_type: str,
    video_backend: str,
) -> ACTTD3TrainingDataIdentity:
    """Build a semantic SHA-256 identity for an ACT-TD3 offline run.

    The selection is resolved twice, before and after hashing.  Every opened
    file is checked with ``fstat`` before/after reading and checked again after
    the second resolution, so concurrent recording/conversion changes fail
    closed instead of producing a mixed checkpoint identity.
    """

    first = _resolve_inputs(
        dataset,
        dataset_root,
        act_checkpoint_root,
        action_domain,
        robot_type=robot_type,
        video_backend=video_backend,
    )
    manifest: list[ACTTD3TrainingIdentityFile] = []
    snapshots: list[tuple[Path, _FileSnapshot]] = []
    for selected in first.files:
        entry, snapshot = _hash_selected_file(selected)
        manifest.append(entry)
        snapshots.append((selected.path, snapshot))

    second = _resolve_inputs(
        dataset,
        dataset_root,
        act_checkpoint_root,
        action_domain,
        robot_type=robot_type,
        video_backend=video_backend,
    )
    first_selection = tuple(
        (selected.component, selected.logical_path, str(selected.path))
        for selected in first.files
    )
    second_selection = tuple(
        (selected.component, selected.logical_path, str(selected.path))
        for selected in second.files
    )
    if (
        first_selection != second_selection
        or _canonical_json_bytes(first.virtual_contract)
        != _canonical_json_bytes(second.virtual_contract)
    ):
        raise RuntimeError("training input manifest changed while hashing")
    for path, snapshot in snapshots:
        _assert_snapshot(path, snapshot)

    manifest_tuple = tuple(manifest)
    component_sha256: dict[str, str] = {}
    for component in ("dataset", "act_checkpoint", "robot"):
        component_entries = [
            {
                "path": entry.path,
                "byte_count": entry.byte_count,
                "sha256": entry.sha256,
            }
            for entry in manifest_tuple
            if entry.component == component
        ]
        if not component_entries:
            raise RuntimeError(f"training identity component is empty: {component}")
        component_sha256[component] = _prefixed_sha256(
            _canonical_json_bytes(
                {
                    "schema": _IDENTITY_SCHEMA,
                    "component": component,
                    "files": component_entries,
                }
            )
        )
    component_sha256["virtual_contract"] = _prefixed_sha256(
        _canonical_json_bytes(
            {
                "schema": _IDENTITY_SCHEMA,
                "component": "virtual_contract",
                "value": first.virtual_contract,
            }
        )
    )

    identity_payload = {
        "schema": _IDENTITY_SCHEMA,
        "files": [asdict(entry) for entry in manifest_tuple],
        "virtual_contract": first.virtual_contract,
    }
    return ACTTD3TrainingDataIdentity(
        identity=_prefixed_sha256(_canonical_json_bytes(identity_payload)),
        file_count=len(manifest_tuple),
        byte_count=sum(entry.byte_count for entry in manifest_tuple),
        component_sha256=component_sha256,
        manifest=manifest_tuple,
        virtual_contract=first.virtual_contract,
    )


def build_act_td3_multi_root_training_data_identity(
    datasets: Sequence[Any],
    dataset_roots: Sequence[str | Path],
    act_checkpoint_root: str | Path,
    action_domains: Sequence[Any],
    *,
    robot_type: str,
    video_backend: str,
) -> ACTTD3TrainingDataIdentity:
    """Bind ordered immutable LeRobot roots into one cumulative identity.

    Each root is first hashed with the strict single-root builder.  Dataset
    entries are then namespaced by root ordinal while checkpoint and robot
    inputs are retained once.  The ordered ``data_roots`` contract is what a
    child training round uses for append-only prefix validation.
    """

    if isinstance(datasets, (str, bytes)) or not isinstance(datasets, Sequence):
        raise TypeError("datasets must be an ordered sequence")
    if isinstance(dataset_roots, (str, bytes)) or not isinstance(
        dataset_roots, Sequence
    ):
        raise TypeError("dataset_roots must be an ordered sequence")
    if isinstance(action_domains, (str, bytes)) or not isinstance(
        action_domains, Sequence
    ):
        raise TypeError("action_domains must be an ordered sequence")
    datasets_tuple = tuple(datasets)
    roots_tuple = tuple(dataset_roots)
    domains_tuple = tuple(action_domains)
    if not datasets_tuple or not (
        len(datasets_tuple) == len(roots_tuple) == len(domains_tuple)
    ):
        raise ValueError(
            "datasets, dataset_roots, and action_domains must have the same non-zero length"
        )

    resolved_roots = tuple(
        _secure_root(root, label=f"LeRobot dataset root {index}").path
        for index, root in enumerate(roots_tuple)
    )
    if len(set(resolved_roots)) != len(resolved_roots):
        raise ValueError("ordered LeRobot dataset roots must be unique")

    identities = tuple(
        build_act_td3_training_data_identity(
            dataset,
            dataset_root=root,
            act_checkpoint_root=act_checkpoint_root,
            action_domain=domain,
            robot_type=robot_type,
            video_backend=video_backend,
        )
        for dataset, root, domain in zip(
            datasets_tuple, resolved_roots, domains_tuple, strict=True
        )
    )

    first = identities[0]
    common_virtual = {
        key: value
        for key, value in first.virtual_contract.items()
        if key != "episode_indices"
    }
    for index, identity in enumerate(identities[1:], start=1):
        candidate_virtual = {
            key: value
            for key, value in identity.virtual_contract.items()
            if key != "episode_indices"
        }
        if candidate_virtual != common_virtual:
            raise ValueError(
                f"LeRobot dataset root {index} has an incompatible virtual contract"
            )
        for component in ("act_checkpoint", "robot"):
            if identity.component_sha256.get(component) != first.component_sha256.get(
                component
            ):
                raise ValueError(
                    f"LeRobot dataset root {index} has a different {component} identity"
                )

    manifest: list[ACTTD3TrainingIdentityFile] = []
    for root_index, identity in enumerate(identities):
        for entry in identity.manifest:
            if entry.component == "dataset":
                manifest.append(
                    ACTTD3TrainingIdentityFile(
                        component="dataset",
                        path=f"data_root_{root_index:04d}/{entry.path}",
                        byte_count=entry.byte_count,
                        sha256=entry.sha256,
                    )
                )
            elif root_index == 0:
                manifest.append(entry)

    data_roots: list[dict[str, Any]] = []
    global_episode_start = 0
    global_episode_indices: list[int] = []
    for index, (root, identity) in enumerate(zip(resolved_roots, identities, strict=True)):
        local_indices = identity.virtual_contract.get("episode_indices")
        if not isinstance(local_indices, list) or not local_indices:
            raise RuntimeError("single-root identity returned invalid episode indices")
        global_episode_stop = global_episode_start + len(local_indices)
        global_indices = list(range(global_episode_start, global_episode_stop))
        global_episode_indices.extend(global_indices)
        data_roots.append(
            {
                "ordinal": index,
                "root": str(root),
                "name": root.name,
                "identity": identity.identity,
                "dataset_sha256": identity.component_sha256["dataset"],
                "episode_indices": list(local_indices),
                "global_episode_indices": global_indices,
                "file_count": sum(
                    entry.component == "dataset" for entry in identity.manifest
                ),
                "byte_count": sum(
                    entry.byte_count
                    for entry in identity.manifest
                    if entry.component == "dataset"
                ),
            }
        )
        global_episode_start = global_episode_stop

    virtual_contract = {
        **common_virtual,
        "episode_indices": global_episode_indices,
        "data_roots": data_roots,
    }
    manifest_tuple = tuple(
        sorted(manifest, key=lambda entry: (entry.component, entry.path))
    )
    dataset_entries = [
        {
            "path": entry.path,
            "byte_count": entry.byte_count,
            "sha256": entry.sha256,
        }
        for entry in manifest_tuple
        if entry.component == "dataset"
    ]
    component_sha256 = {
        "dataset": _prefixed_sha256(
            _canonical_json_bytes(
                {
                    "schema": _MULTI_IDENTITY_SCHEMA,
                    "component": "dataset",
                    "files": dataset_entries,
                }
            )
        ),
        "act_checkpoint": first.component_sha256["act_checkpoint"],
        "robot": first.component_sha256["robot"],
        "virtual_contract": _prefixed_sha256(
            _canonical_json_bytes(
                {
                    "schema": _MULTI_IDENTITY_SCHEMA,
                    "component": "virtual_contract",
                    "value": virtual_contract,
                }
            )
        ),
    }
    for index, identity in enumerate(identities):
        component_sha256[f"data_root_{index:04d}"] = identity.component_sha256[
            "dataset"
        ]

    identity_payload = {
        "schema": _MULTI_IDENTITY_SCHEMA,
        "files": [asdict(entry) for entry in manifest_tuple],
        "virtual_contract": virtual_contract,
    }
    return ACTTD3TrainingDataIdentity(
        identity=_prefixed_sha256(_canonical_json_bytes(identity_payload)),
        file_count=len(manifest_tuple),
        byte_count=sum(entry.byte_count for entry in manifest_tuple),
        component_sha256=component_sha256,
        manifest=manifest_tuple,
        virtual_contract=virtual_contract,
    )


__all__ = [
    "ACTTD3TrainingDataIdentity",
    "ACTTD3TrainingIdentityFile",
    "build_act_td3_multi_root_training_data_identity",
    "build_act_td3_training_data_identity",
]
