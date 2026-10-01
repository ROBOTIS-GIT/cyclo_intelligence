"""Keep Cyclo's LeRobot images aligned with the supported policy catalog."""

import re
import tomllib
from pathlib import Path

import yaml


POLICY_DIR = Path(__file__).resolve().parents[1]
EXTRA_REFERENCE = re.compile(r"lerobot\[([^]]+)\]")


def test_cyclo_policy_extra_matches_manifest():
    with (POLICY_DIR / "lerobot/pyproject.toml").open("rb") as file:
        extras = tomllib.load(file)["project"]["optional-dependencies"]
    catalog = yaml.safe_load((POLICY_DIR / "manifest.yaml").read_text())

    policy_extras = {EXTRA_REFERENCE.fullmatch(item).group(1) for item in extras["cyclo-policies"]}
    policy_ids = {model["id"] for model in catalog["models"]}
    expected_extras = policy_ids - {"act", "pi0", "pi05", "wall_x"}
    if policy_ids & {"pi0", "pi05"}:
        expected_extras.add("pi")
    if "wall_x" in policy_ids:
        expected_extras.add("wallx")

    assert policy_extras == expected_extras
    assert set(extras["cyclo-inference"]) == {
        "lerobot[cyclo-policies]",
        "lerobot[training]",
        "lerobot[hilserl]",
        "lerobot[async]",
        "lerobot[peft]",
    }


def test_inference_images_use_composite_extra():
    for architecture in ("amd64", "arm64"):
        dockerfile = (POLICY_DIR / f"Dockerfile.{architecture}").read_text()
        assert '".[cyclo-inference]"' in dockerfile or "--extra cyclo-inference" in dockerfile
        assert "--extra smolvla" not in dockerfile

    amd64 = (POLICY_DIR / "Dockerfile.amd64").read_text()
    assert "lerobot/uv.lock" in amd64
    assert "uv sync --locked" in amd64
