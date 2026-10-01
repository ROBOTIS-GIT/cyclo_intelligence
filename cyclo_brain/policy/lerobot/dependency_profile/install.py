"""Resolve Cyclo-owned LeRobot image profiles without owning model dependencies."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import tomllib

PROFILE_PATH = Path(__file__).with_name("dependencies.toml")
EXTRA_NAME = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*\Z")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def resolve(profile_path: Path, project_path: Path, profile_name: str) -> dict:
    with profile_path.open("rb") as file:
        config = tomllib.load(file)
    with (project_path / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)

    if config.get("schema_version") != 1 or set(config) != {"schema_version", "profiles", "models"}:
        raise ValueError("Invalid dependency profile schema")
    if project.get("project", {}).get("name") != "lerobot":
        raise ValueError("Expected a LeRobot source checkout")
    profiles = config["profiles"]
    if not isinstance(profiles, dict) or profile_name not in profiles:
        raise ValueError(f"Unknown installation profile: {profile_name}")
    available = {_canonical(name) for name in project["project"].get("optional-dependencies", {})}

    def check_extra(name: str) -> None:
        if not isinstance(name, str) or not EXTRA_NAME.fullmatch(name):
            raise ValueError(f"Invalid LeRobot extra: {name!r}")
        if _canonical(name) not in available:
            raise ValueError(f"LeRobot extra is not defined by this checkout: {name}")

    for name, entry in profiles.items():
        if not isinstance(entry, dict) or set(entry) != {"features"}:
            raise ValueError(f"Invalid profile definition: {name}")
        if (
            not isinstance(entry["features"], list)
            or any(not isinstance(extra, str) for extra in entry["features"])
            or len(entry["features"]) != len(set(entry["features"]))
        ):
            raise ValueError(f"Invalid features for profile: {name}")
        for extra in entry["features"]:
            check_extra(extra)

    models = config["models"]
    if not isinstance(models, dict) or not models:
        raise ValueError("No models declared in dependency profile")
    for model_id, entry in models.items():
        if not EXTRA_NAME.fullmatch(model_id):
            raise ValueError(f"Invalid model ID: {model_id}")
        if not isinstance(entry, dict) or set(entry) != {"extra", "profiles"}:
            raise ValueError(f"Invalid model definition: {model_id}")
        if not isinstance(entry["extra"], str):
            raise ValueError(f"Invalid extra for model: {model_id}")
        if entry["extra"]:
            check_extra(entry["extra"])
        if (
            not isinstance(entry["profiles"], list)
            or not entry["profiles"]
            or any(not isinstance(name, str) for name in entry["profiles"])
            or len(entry["profiles"]) != len(set(entry["profiles"]))
        ):
            raise ValueError(f"Invalid profiles for model: {model_id}")
        if any(name not in profiles for name in entry["profiles"]):
            raise ValueError(f"Unknown profile for model: {model_id}")

    selected_models = [model_id for model_id, entry in models.items() if profile_name in entry["profiles"]]
    if not selected_models:
        raise ValueError(f"No models selected for profile: {profile_name}")
    extras = list(dict.fromkeys(
        [_canonical(name) for name in profiles[profile_name]["features"]]
        + [_canonical(models[model_id]["extra"]) for model_id in selected_models if models[model_id]["extra"]]
    ))
    lock_path = project_path / "uv.lock"
    if not lock_path.is_file():
        raise ValueError("A uv.lock file is required for LeRobot source provenance")
    return {
        "profile": profile_name,
        "models": selected_models,
        "extras": extras,
        "profile_sha256": _sha256(profile_path),
        "pyproject_sha256": _sha256(project_path / "pyproject.toml"),
        "lock_sha256": _sha256(lock_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROFILE_PATH)
    parser.add_argument("--manager", choices=("check", "uv", "pip"), default="check")
    parser.add_argument("--expected-profile-sha256")
    parser.add_argument("--expected-lock-sha256")
    args = parser.parse_args()

    result = resolve(args.config, args.project, args.profile)
    if args.expected_profile_sha256 and result["profile_sha256"] != args.expected_profile_sha256:
        raise ValueError("Dependency profile hash does not match the pinned build input")
    if args.expected_lock_sha256 and result["lock_sha256"] != args.expected_lock_sha256:
        raise ValueError("LeRobot lockfile hash does not match the pinned build input")
    if args.manager == "uv":
        command = ["uv", "sync", "--locked", "--no-dev", "--no-cache"]
        for extra in result["extras"]:
            command.extend(("--extra", extra))
        subprocess.run(command, cwd=args.project, check=True)
    elif args.manager == "pip":
        subprocess.run(
            ["pip", "install", "--no-cache-dir", f".[{','.join(result['extras'])}]"],
            cwd=args.project,
            check=True,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
