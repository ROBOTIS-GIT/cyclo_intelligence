from pathlib import Path

import pytest

from catalog import CatalogError, load_catalog, normalize_policy_parameters


POLICY_ROOT = Path(__file__).resolve().parents[3]


def test_repository_catalog_matches_compose_services():
    catalog = load_catalog(POLICY_ROOT, compose_services={"lerobot", "groot"})

    assert [runtime["id"] for runtime in catalog["runtimes"]] == ["groot", "lerobot"]
    policy_ids = {
        model["policy_id"]
        for runtime in catalog["runtimes"]
        for model in runtime["models"]
    }
    assert "lerobot:act" in policy_ids
    assert "groot:n17" in policy_ids


def test_policy_parameter_payload_defaults_and_canonicalizes(tmp_path):
    policy_root = tmp_path / "policy"
    runtime = policy_root / "sample"
    runtime.mkdir(parents=True)
    (runtime / "manifest.yaml").write_text(
        """
schema_version: 1
runtime:
  id: sample
  label: Sample
  compose_service: sample
  service_prefix: sample
  checkpoint_root: /workspace/model/sample
  source: {kind: package, location: sample-policy==1.0.0}
models:
  - id: base
    label: Base
    parameters:
      - key: temperature
        label: Temperature
        control: number
        binding: policy_parameters.temperature
        default: 0.5
        min: 0.0
        max: 1.0
""",
        encoding="utf-8",
    )
    catalog = load_catalog(policy_root, compose_services={"sample"})

    assert normalize_policy_parameters(catalog, "sample:base", "") == '{"temperature":0.5}'
    assert normalize_policy_parameters(
        catalog, "sample:base", '{"temperature":0.25}'
    ) == '{"temperature":0.25}'

    with pytest.raises(CatalogError, match="unknown policy parameters"):
        normalize_policy_parameters(catalog, "sample:base", '{"typo":1}')
    with pytest.raises(CatalogError, match="exceeds its maximum"):
        normalize_policy_parameters(catalog, "sample:base", '{"temperature":2}')


def _write_manifest(policy_root: Path, runtime_id: str, body: str) -> Path:
    runtime = policy_root / runtime_id
    runtime.mkdir(parents=True)
    path = runtime / "manifest.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _parameter_catalog(tmp_path: Path):
    root = tmp_path / "policy"
    _write_manifest(root, "sample", """
schema_version: 1
runtime:
  id: sample
  label: Sample
  compose_service: sample
  service_prefix: sample
  checkpoint_root: /workspace/model/sample
  source: {kind: package, location: sample-policy==1.0.0}
models:
  - id: base
    label: Base
    parameters:
      - key: mode
        label: Mode
        control: select
        binding: policy_parameters.mode
        default: safe
        options: [safe, fast]
      - key: count
        label: Count
        control: integer
        binding: policy_parameters.count
        required: true
        min: 1
        max: 4
      - key: gain
        label: Gain
        control: number
        binding: policy_parameters.gain
        default: 0.5
        min: 0.0
        max: 1.0
        visible_when: {mode: fast}
""")
    return load_catalog(root, compose_services={"sample"})


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("[]", "must contain a JSON object"),
        ('{"count":"two"}', "must be an integer"),
        ('{"count":5}', "exceeds its maximum"),
        ('{"count":2,"mode":"turbo"}', "must be one of"),
        ('{"count":2,"unknown":true}', "unknown policy parameters"),
        ('{"count":2,"gain":NaN}', "invalid JSON"),
    ],
)
def test_policy_parameter_payload_rejects_invalid_values(tmp_path, payload, message):
    catalog = _parameter_catalog(tmp_path)

    with pytest.raises(CatalogError, match=message):
        normalize_policy_parameters(catalog, "sample:base", payload)


def test_policy_parameter_payload_enforces_required_and_size(tmp_path):
    catalog = _parameter_catalog(tmp_path)

    with pytest.raises(CatalogError, match="missing required"):
        normalize_policy_parameters(catalog, "sample:base", "{}")
    with pytest.raises(CatalogError, match="exceeds 65536 bytes"):
        normalize_policy_parameters(
            catalog,
            "sample:base",
            '{"count":2,"padding":"' + ("x" * 65536) + '"}',
        )


def test_policy_parameter_visibility_drops_hidden_values(tmp_path):
    catalog = _parameter_catalog(tmp_path)

    assert normalize_policy_parameters(
        catalog,
        "sample:base",
        '{"gain":0.9,"count":2,"mode":"safe"}',
    ) == '{"count":2,"mode":"safe"}'


def test_required_text_parameter_rejects_empty_string(tmp_path):
    root = tmp_path / "policy"
    _write_manifest(root, "sample", """
schema_version: 1
runtime:
  id: sample
  label: Sample
  compose_service: sample
  service_prefix: sample
  checkpoint_root: /workspace/model/sample
  source: {kind: package, location: sample-policy==1.0.0}
models:
  - id: base
    label: Base
    parameters:
      - key: prompt
        label: Prompt
        control: text
        binding: policy_parameters.prompt
        required: true
""")
    catalog = load_catalog(root, compose_services={"sample"})

    with pytest.raises(CatalogError, match="cannot be empty"):
        normalize_policy_parameters(catalog, "sample:base", '{"prompt":""}')


def test_catalog_rejects_missing_compose_service(tmp_path):
    root = tmp_path / "policy"
    _write_manifest(root, "sample", """
schema_version: 1
runtime:
  id: sample
  label: Sample
  compose_service: sample
  service_prefix: sample
  checkpoint_root: /workspace/model/sample
  source: {kind: package, location: sample-policy==1.0.0}
models: [{id: base, label: Base}]
""")

    with pytest.raises(CatalogError, match="missing Compose service"):
        load_catalog(root, compose_services={"other"})


def test_catalog_rejects_unimplemented_action_request_mode(tmp_path):
    root = tmp_path / "policy"
    _write_manifest(root, "sample", """
schema_version: 1
runtime:
  id: sample
  label: Sample
  compose_service: sample
  service_prefix: sample
  checkpoint_root: /workspace/model/sample
  source: {kind: package, location: sample-policy==1.0.0}
  capabilities:
    action_request_modes: [async, rtc]
models: [{id: base, label: Base}]
""")

    with pytest.raises(CatalogError, match="action_request_modes is invalid"):
        load_catalog(root, compose_services={"sample"})


def test_catalog_rejects_duplicate_bare_alias(tmp_path):
    root = tmp_path / "policy"
    template = """
schema_version: 1
runtime:
  id: {runtime}
  label: {runtime}
  compose_service: {runtime}
  service_prefix: {runtime}
  checkpoint_root: /workspace/model/{runtime}
  source: {{kind: package, location: {runtime}==1.0.0}}
models:
  - id: base
    label: Base
    aliases: [legacy]
"""
    _write_manifest(root, "one", template.format(runtime="one"))
    _write_manifest(root, "two", template.format(runtime="two"))

    with pytest.raises(CatalogError, match="duplicate policy alias"):
        load_catalog(root, compose_services={"one", "two"})


@pytest.mark.parametrize(
    ("parameter", "message"),
    [
        (
            "{key: gain, label: Gain, control: number, "
            "binding: policy_parameters.other}",
            "must end with the parameter key",
        ),
        (
            "{key: gain, label: Gain, control: number, "
            "binding: policy_parameters.gain, surprise: true}",
            "unknown fields",
        ),
        (
            "{key: mode, label: Mode, control: select, "
            "binding: policy_parameters.mode, options: [safe], default: bad}",
            "must be one of",
        ),
        (
            "{key: mode, label: Mode, control: select, "
            "binding: policy_parameters.mode, options: [1]}",
            "options must contain strings",
        ),
        (
            "{key: gain, label: Gain, control: number, "
            "binding: policy_parameters.gain, visible_when: {missing: true}}",
            "references unknown parameters",
        ),
        (
            "{key: prompt, label: Prompt, control: text, "
            "binding: policy_parameters.prompt, min: 1}",
            "only valid for numeric controls",
        ),
        (
            "{key: gain, label: Gain, control: number, "
            "binding: policy_parameters.gain, step: 0}",
            "step must be positive",
        ),
        (
            "{key: prompt, label: Prompt, control: text, "
            "binding: policy_parameters.prompt, required: 'false'}",
            "required must be boolean",
        ),
    ],
)
def test_catalog_rejects_malformed_parameter_schema(tmp_path, parameter, message):
    root = tmp_path / "policy"
    _write_manifest(root, "sample", f"""
schema_version: 1
runtime:
  id: sample
  label: Sample
  compose_service: sample
  service_prefix: sample
  checkpoint_root: /workspace/model/sample
  source: {{kind: package, location: sample-policy==1.0.0}}
models:
  - id: base
    label: Base
    parameters:
      - {parameter}
""")

    with pytest.raises(CatalogError, match=message):
        load_catalog(root, compose_services={"sample"})


@pytest.mark.parametrize(
    ("fragment", "message"),
    [
        ("runtime:\n  surprise: true", "unknown fields"),
        ("capabilities:\n    typo: true", "unknown fields"),
        ("model_extra: true", "unknown fields"),
    ],
)
def test_catalog_rejects_unknown_manifest_fields(tmp_path, fragment, message):
    root = tmp_path / "policy"
    if fragment.startswith("runtime:"):
        runtime_extra = "  surprise: true\n"
        capabilities = ""
        model_extra = ""
    elif fragment.startswith("capabilities:"):
        runtime_extra = ""
        capabilities = "  capabilities:\n    typo: true\n"
        model_extra = ""
    else:
        runtime_extra = ""
        capabilities = ""
        model_extra = "    model_extra: true\n"
    _write_manifest(root, "sample", f"""
schema_version: 1
runtime:
  id: sample
  label: Sample
  compose_service: sample
  service_prefix: sample
  checkpoint_root: /workspace/model/sample
  source: {{kind: package, location: sample-policy==1.0.0}}
{runtime_extra}{capabilities}models:
  - id: base
    label: Base
{model_extra}""")

    with pytest.raises(CatalogError, match=message):
        load_catalog(root, compose_services={"sample"})


def test_catalog_rejects_unknown_task_info_binding_and_duplicate_alias(tmp_path):
    root = tmp_path / "policy"
    path = _write_manifest(root, "sample", """
schema_version: 1
runtime:
  id: sample
  label: Sample
  compose_service: sample
  service_prefix: sample
  checkpoint_root: /workspace/model/sample
  source: {kind: package, location: sample-policy==1.0.0}
models:
  - id: base
    label: Base
    parameters:
      - key: typo
        label: Typo
        control: text
        binding: task_info.typo
""")
    with pytest.raises(CatalogError, match="unknown task_info field"):
        load_catalog(root, compose_services={"sample"})

    path.write_text(path.read_text(encoding="utf-8").replace(
        "    parameters:\n",
        "    aliases: [legacy, legacy]\n    parameters:\n",
    ), encoding="utf-8")
    with pytest.raises(CatalogError, match="aliases contains duplicates"):
        load_catalog(root, compose_services={"sample"})
