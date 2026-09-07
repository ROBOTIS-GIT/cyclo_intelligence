"""Validated, human-editable policy runtime catalog.

Each independently deployable policy runtime owns one ``manifest.yaml``.
Docker remains described by Compose; manifests only point at a Compose service
and describe the policies and UI capabilities provided by that runtime.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


MAX_POLICY_PARAMETERS_BYTES = 64 * 1024
_CONTROLS = {"text", "path", "number", "integer", "select", "toggle"}
_SOURCE_KINDS = {"submodule", "package"}
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_MANIFEST_FIELDS = {"schema_version", "runtime", "models"}
_RUNTIME_FIELDS = {
    "id", "label", "compose_service", "service_prefix", "checkpoint_root",
    "source", "services", "capabilities",
}
_SOURCE_FIELDS = {"kind", "location"}
_CAPABILITY_FIELDS = {
    "action_request_modes", "requires_hf_token", "operations",
}
_MODEL_FIELDS = {"id", "label", "aliases", "requires_instruction", "parameters"}
_PARAMETER_FIELDS = {
    "key", "label", "control", "binding", "default", "required",
    "min", "max", "step", "options", "visible_when",
}
_TASK_INFO_PARAMETER_KEYS = {"accelerationMode", "accelerationEnginePath"}


class CatalogError(ValueError):
    """Raised when a policy manifest or parameter payload is invalid."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid constant {value}")


def _require_mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CatalogError(f"{location} must be a mapping")
    return dict(value)


def _require_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{location} must be a non-empty string")
    return value.strip()


def _optional_bool(
    value: Mapping[str, Any],
    key: str,
    default: bool,
    location: str,
) -> bool:
    if key not in value:
        return default
    if not isinstance(value[key], bool):
        raise CatalogError(f"{location}.{key} must be boolean")
    return value[key]


def _reject_unknown_fields(
    value: Mapping[str, Any], allowed: set[str], location: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise CatalogError(f"{location} has unknown fields: {unknown}")


def _validate_parameter(parameter: Any, location: str) -> dict[str, Any]:
    item = _require_mapping(parameter, location)
    _reject_unknown_fields(item, _PARAMETER_FIELDS, location)
    key = _require_string(item.get("key"), f"{location}.key")
    item["key"] = key
    item["label"] = _require_string(item.get("label"), f"{location}.label")
    control = _require_string(item.get("control"), f"{location}.control")
    if control not in _CONTROLS:
        raise CatalogError(
            f"{location}.control must be one of {sorted(_CONTROLS)}"
        )
    item["control"] = control

    binding = _require_string(item.get("binding"), f"{location}.binding")
    if not binding.startswith(("task_info.", "policy_parameters.")):
        raise CatalogError(
            f"{location}.binding must start with task_info. or policy_parameters."
        )
    item["binding"] = binding
    if binding.startswith("policy_parameters.") and binding.split(".", 1)[1] != key:
        raise CatalogError(
            f"{location}.binding must end with the parameter key {key!r}"
        )
    if (
        binding.startswith("task_info.")
        and binding.split(".", 1)[1] not in _TASK_INFO_PARAMETER_KEYS
    ):
        raise CatalogError(f"{location}.binding references an unknown task_info field")
    item["required"] = _optional_bool(item, "required", False, location)

    options = item.get("options", [])
    if not isinstance(options, list):
        raise CatalogError(f"{location}.options must be a list")
    if control == "select" and not options:
        raise CatalogError(f"{location}.options is required for select controls")
    if control == "select" and any(not isinstance(option, str) for option in options):
        raise CatalogError(f"{location}.options must contain strings")
    if control != "select" and options:
        raise CatalogError(f"{location}.options is only valid for select controls")
    if len(options) != len(set(options)):
        raise CatalogError(f"{location}.options contains duplicates")
    item["options"] = options

    visible_when = item.get("visible_when", {})
    if not isinstance(visible_when, dict):
        raise CatalogError(f"{location}.visible_when must be a mapping")
    item["visible_when"] = visible_when

    for bound in ("min", "max", "step"):
        if bound not in item:
            continue
        if control not in {"number", "integer"}:
            raise CatalogError(
                f"{location}.{bound} is only valid for numeric controls"
            )
        value = item[bound]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CatalogError(f"{location}.{bound} must be numeric")
        if not math.isfinite(float(value)):
            raise CatalogError(f"{location}.{bound} must be finite")
    if "step" in item and item["step"] <= 0:
        raise CatalogError(f"{location}.step must be positive")
    if "min" in item and "max" in item and item["min"] > item["max"]:
        raise CatalogError(f"{location}.min cannot exceed max")
    if "default" in item:
        _validate_value(item, item["default"])
    return item


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CatalogError(f"could not read {manifest_path}: {exc}") from exc

    manifest = _require_mapping(raw, str(manifest_path))
    _reject_unknown_fields(manifest, _MANIFEST_FIELDS, str(manifest_path))
    if manifest.get("schema_version") != 1:
        raise CatalogError(f"{manifest_path}: schema_version must be 1")

    runtime = _require_mapping(manifest.get("runtime"), f"{manifest_path}.runtime")
    _reject_unknown_fields(runtime, _RUNTIME_FIELDS, f"{manifest_path}.runtime")
    runtime_id = _require_string(runtime.get("id"), f"{manifest_path}.runtime.id")
    if not _ID_RE.fullmatch(runtime_id):
        raise CatalogError(
            f"{manifest_path}.runtime.id must match {_ID_RE.pattern!r}"
        )
    runtime["id"] = runtime_id
    runtime["label"] = _require_string(
        runtime.get("label"), f"{manifest_path}.runtime.label"
    )
    for field in ("compose_service", "service_prefix", "checkpoint_root"):
        runtime[field] = _require_string(
            runtime.get(field), f"{manifest_path}.runtime.{field}"
        )
    if not (
        runtime["checkpoint_root"] == "/workspace"
        or runtime["checkpoint_root"].startswith("/workspace/")
    ):
        raise CatalogError(
            f"{manifest_path}.runtime.checkpoint_root must be under /workspace"
        )
    if runtime["compose_service"] != runtime_id or runtime["service_prefix"] != runtime_id:
        raise CatalogError(
            f"{manifest_path}: runtime id, compose_service, and service_prefix "
            "must be identical"
        )

    source = _require_mapping(runtime.get("source"), f"{manifest_path}.runtime.source")
    _reject_unknown_fields(source, _SOURCE_FIELDS, f"{manifest_path}.runtime.source")
    source_kind = _require_string(
        source.get("kind"), f"{manifest_path}.runtime.source.kind"
    )
    if source_kind not in _SOURCE_KINDS:
        raise CatalogError(
            f"{manifest_path}.runtime.source.kind must be one of "
            f"{sorted(_SOURCE_KINDS)}"
        )
    source["kind"] = source_kind
    source["location"] = _require_string(
        source.get("location"), f"{manifest_path}.runtime.source.location"
    )
    runtime["source"] = source

    capabilities = runtime.get("capabilities", {})
    if not isinstance(capabilities, dict):
        raise CatalogError(f"{manifest_path}.runtime.capabilities must be a mapping")
    _reject_unknown_fields(
        capabilities,
        _CAPABILITY_FIELDS,
        f"{manifest_path}.runtime.capabilities",
    )
    modes = capabilities.get("action_request_modes", ["async", "sync"])
    if not isinstance(modes, list) or not modes or any(
        mode not in {"async", "sync"} for mode in modes
    ):
        raise CatalogError(
            f"{manifest_path}.runtime.capabilities.action_request_modes is invalid"
        )
    capabilities["action_request_modes"] = list(dict.fromkeys(modes))
    operations = capabilities.get("operations", [])
    if not isinstance(operations, list) or any(
        not isinstance(value, str) or not value for value in operations
    ):
        raise CatalogError(f"{manifest_path}.runtime.capabilities.operations is invalid")
    capabilities["operations"] = list(dict.fromkeys(operations))
    capabilities["requires_hf_token"] = _optional_bool(
        capabilities,
        "requires_hf_token",
        False,
        f"{manifest_path}.runtime.capabilities",
    )
    runtime["capabilities"] = capabilities

    services = runtime.get("services", ["engine-process"])
    if not isinstance(services, list) or not services or any(
        not isinstance(service, str) or not service for service in services
    ):
        raise CatalogError(f"{manifest_path}.runtime.services must be a string list")
    if len(services) != len(set(services)):
        raise CatalogError(f"{manifest_path}.runtime.services contains duplicates")
    runtime["services"] = services

    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise CatalogError(f"{manifest_path}.models must be a non-empty list")
    seen_model_ids: set[str] = set()
    validated_models: list[dict[str, Any]] = []
    for index, raw_model in enumerate(models):
        location = f"{manifest_path}.models[{index}]"
        model = _require_mapping(raw_model, location)
        _reject_unknown_fields(model, _MODEL_FIELDS, location)
        model_id = _require_string(model.get("id"), f"{location}.id")
        if not _ID_RE.fullmatch(model_id):
            raise CatalogError(f"{location}.id must match {_ID_RE.pattern!r}")
        if model_id in seen_model_ids:
            raise CatalogError(f"{manifest_path}: duplicate model id {model_id!r}")
        seen_model_ids.add(model_id)
        model["id"] = model_id
        model["policy_id"] = f"{runtime_id}:{model_id}"
        model["label"] = _require_string(model.get("label"), f"{location}.label")
        model["requires_instruction"] = _optional_bool(
            model,
            "requires_instruction",
            False,
            location,
        )

        aliases = model.get("aliases", [])
        if not isinstance(aliases, list) or any(
            not isinstance(alias, str) or not alias.strip() for alias in aliases
        ):
            raise CatalogError(f"{location}.aliases must be a string list")
        aliases = [alias.strip() for alias in aliases]
        if len(aliases) != len(set(aliases)):
            raise CatalogError(f"{location}.aliases contains duplicates")
        if any(":" in alias for alias in aliases):
            raise CatalogError(f"{location}.aliases must be bare policy names")
        model["aliases"] = aliases

        parameters = model.get("parameters", [])
        if not isinstance(parameters, list):
            raise CatalogError(f"{location}.parameters must be a list")
        validated_parameters = [
            _validate_parameter(value, f"{location}.parameters[{param_index}]")
            for param_index, value in enumerate(parameters)
        ]
        parameter_keys = [parameter["key"] for parameter in validated_parameters]
        if len(parameter_keys) != len(set(parameter_keys)):
            raise CatalogError(f"{location}: duplicate parameter key")
        known_keys = set(parameter_keys)
        parameters_by_key = {
            parameter["key"]: parameter for parameter in validated_parameters
        }
        for parameter in validated_parameters:
            unknown_conditions = set(parameter["visible_when"]) - known_keys
            if unknown_conditions:
                raise CatalogError(
                    f"{location}: visible_when references unknown parameters "
                    f"{sorted(unknown_conditions)}"
                )
            for condition_key, condition_value in parameter["visible_when"].items():
                _validate_value(parameters_by_key[condition_key], condition_value)
        model["parameters"] = validated_parameters
        validated_models.append(model)

    return {
        "schema_version": 1,
        "runtime": runtime,
        "models": validated_models,
        "manifest_path": str(manifest_path),
    }


def load_catalog(
    policy_root: str | Path,
    *,
    compose_services: Iterable[str] | None = None,
) -> dict[str, Any]:
    root = Path(policy_root)
    paths = sorted(root.glob("*/manifest.yaml"))
    if not paths:
        raise CatalogError(f"no policy manifests found under {root}")

    manifests = [load_manifest(path) for path in paths]
    return catalog_from_manifests(manifests, compose_services=compose_services)


def catalog_from_manifests(
    manifests: Iterable[Mapping[str, Any]],
    *,
    compose_services: Iterable[str] | None = None,
) -> dict[str, Any]:
    runtimes: list[dict[str, Any]] = []
    policy_ids: set[str] = set()
    runtime_ids: set[str] = set()
    aliases: dict[str, str] = {}
    expected_services = set(compose_services) if compose_services is not None else None
    for manifest_value in manifests:
        manifest = dict(manifest_value)
        runtime = manifest["runtime"]
        runtime_id = runtime["id"]
        if runtime_id in runtime_ids:
            raise CatalogError(f"duplicate runtime id {runtime_id!r}")
        runtime_ids.add(runtime_id)
        if expected_services is not None and runtime["compose_service"] not in expected_services:
            raise CatalogError(
                f"runtime {runtime_id!r} references missing Compose service "
                f"{runtime['compose_service']!r}"
            )
        for model in manifest["models"]:
            if model["policy_id"] in policy_ids:
                raise CatalogError(f"duplicate policy id {model['policy_id']!r}")
            policy_ids.add(model["policy_id"])
            for alias in model.get("aliases", []):
                owner = aliases.get(alias)
                if owner is not None and owner != model["policy_id"]:
                    raise CatalogError(
                        f"duplicate policy alias {alias!r}: {owner!r} and "
                        f"{model['policy_id']!r}"
                    )
                aliases[alias] = model["policy_id"]
        runtimes.append({**runtime, "models": manifest["models"]})
    return {"schema_version": 1, "runtimes": runtimes}


def load_runtime_catalog(path: str | Path, runtime_id: str) -> dict[str, Any]:
    manifest = load_manifest(path)
    catalog = catalog_from_manifests([manifest])
    runtime = resolve_runtime(catalog, runtime_id)
    if runtime["service_prefix"] != runtime_id:
        raise CatalogError(
            f"manifest runtime {runtime['id']!r} does not match {runtime_id!r}"
        )
    return catalog


def resolve_runtime(catalog: Mapping[str, Any], runtime_id: str) -> dict[str, Any]:
    for runtime in catalog.get("runtimes", []):
        if runtime.get("id") == runtime_id:
            return dict(runtime)
    raise CatalogError(f"unknown policy runtime {runtime_id!r}")


def resolve_policy(catalog: Mapping[str, Any], policy_id: str) -> tuple[dict, dict]:
    for runtime in catalog.get("runtimes", []):
        for model in runtime.get("models", []):
            if model.get("policy_id") == policy_id or policy_id in model.get("aliases", []):
                return dict(runtime), dict(model)
    raise CatalogError(f"unknown policy id {policy_id!r}")


def resolve_policy_id(
    catalog: Mapping[str, Any],
    runtime_id: str,
    requested_policy_id: str,
    model_path: str | Path,
) -> str:
    requested = str(requested_policy_id or "").strip()
    if requested:
        runtime, model = resolve_policy(catalog, requested)
        if runtime["id"] != runtime_id:
            raise CatalogError(
                f"policy {requested!r} belongs to runtime {runtime['id']!r}, "
                f"not {runtime_id!r}"
            )
        return model["policy_id"]

    runtime = resolve_runtime(catalog, runtime_id)
    root = Path(model_path)
    config_paths = [root / "config.json", root / "pretrained_model" / "config.json"]
    checkpoint_type = ""
    for config_path in config_paths:
        if not config_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CatalogError(f"could not read checkpoint config {config_path}: {exc}") from exc
        checkpoint_type = str(config.get("type", "") or "").strip()
        if checkpoint_type:
            break
    if checkpoint_type:
        for model in runtime["models"]:
            if checkpoint_type == model["id"] or checkpoint_type in model["aliases"]:
                return model["policy_id"]
    if len(runtime["models"]) == 1:
        return runtime["models"][0]["policy_id"]
    if checkpoint_type:
        raise CatalogError(
            f"checkpoint policy type {checkpoint_type!r} is not supported by {runtime_id!r}"
        )
    raise CatalogError(
        "policy_id is required because the checkpoint does not identify a supported policy"
    )


def canonical_policy_parameters(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _parameter_is_visible(parameter: Mapping[str, Any], values: Mapping[str, Any]) -> bool:
    return all(values.get(key) == expected for key, expected in parameter["visible_when"].items())


def _validate_value(parameter: Mapping[str, Any], value: Any) -> Any:
    key = parameter["key"]
    control = parameter["control"]
    if control == "toggle":
        if not isinstance(value, bool):
            raise CatalogError(f"policy parameter {key!r} must be boolean")
    elif control == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise CatalogError(f"policy parameter {key!r} must be an integer")
    elif control == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CatalogError(f"policy parameter {key!r} must be numeric")
        if not math.isfinite(float(value)):
            raise CatalogError(f"policy parameter {key!r} must be finite")
    elif control in {"text", "path", "select"}:
        if not isinstance(value, str):
            raise CatalogError(f"policy parameter {key!r} must be a string")

    if parameter["options"] and value not in parameter["options"]:
        raise CatalogError(
            f"policy parameter {key!r} must be one of {parameter['options']}"
        )
    if "min" in parameter and value < parameter["min"]:
        raise CatalogError(f"policy parameter {key!r} is below its minimum")
    if "max" in parameter and value > parameter["max"]:
        raise CatalogError(f"policy parameter {key!r} exceeds its maximum")
    return value


def normalize_policy_parameters(
    catalog: Mapping[str, Any],
    policy_id: str,
    raw: str | None,
) -> str:
    if raw is not None and not isinstance(raw, str):
        raise CatalogError("policy_parameters_json must be a string")
    encoded = (raw or "").encode("utf-8")
    if len(encoded) > MAX_POLICY_PARAMETERS_BYTES:
        raise CatalogError(
            f"policy_parameters_json exceeds {MAX_POLICY_PARAMETERS_BYTES} bytes"
        )
    if not raw:
        payload: Any = {}
    else:
        try:
            payload = json.loads(
                raw,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            detail = getattr(exc, "msg", str(exc))
            raise CatalogError(
                f"policy_parameters_json is invalid JSON: {detail}"
            ) from exc
    if not isinstance(payload, dict):
        raise CatalogError("policy_parameters_json must contain a JSON object")

    _runtime, model = resolve_policy(catalog, policy_id)
    parameters = [
        parameter
        for parameter in model["parameters"]
        if parameter["binding"].startswith("policy_parameters.")
    ]
    known = {parameter["key"] for parameter in parameters}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise CatalogError(f"unknown policy parameters: {unknown}")

    values: dict[str, Any] = {}
    for parameter in parameters:
        key = parameter["key"]
        if key in payload:
            values[key] = payload[key]
        elif "default" in parameter:
            values[key] = parameter["default"]

    normalized: dict[str, Any] = {}
    for parameter in parameters:
        key = parameter["key"]
        visible = _parameter_is_visible(parameter, values)
        if not visible:
            continue
        if key not in values:
            if parameter["required"]:
                raise CatalogError(f"missing required policy parameter {key!r}")
            continue
        if (
            parameter["required"]
            and parameter["control"] in {"text", "path", "select"}
            and not values[key].strip()
        ):
            raise CatalogError(f"required policy parameter {key!r} cannot be empty")
        normalized[key] = _validate_value(parameter, values[key])
    canonical = canonical_policy_parameters(normalized)
    if len(canonical.encode("utf-8")) > MAX_POLICY_PARAMETERS_BYTES:
        raise CatalogError(
            f"policy_parameters_json exceeds {MAX_POLICY_PARAMETERS_BYTES} bytes"
        )
    return canonical
