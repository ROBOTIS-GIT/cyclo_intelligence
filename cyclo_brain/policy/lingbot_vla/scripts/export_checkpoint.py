"""Export a native LingBot checkpoint and its exact training transforms."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lingbot_vla_engine.bundle import CheckpointBundle, config_pairs, validate_assets


def export_checkpoint(weights, training_config, robot_config, norm_stats, metadata, output, base_model=None):
    weights, output = Path(weights).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    if not list(weights.glob("*.safetensors")):
        raise ValueError("--weights must be an upstream hf_ckpt directory")
    training = yaml.safe_load(Path(training_config).read_text())
    robot = yaml.safe_load(Path(robot_config).read_text())
    stats = json.loads(Path(norm_stats).read_text())
    metadata = json.loads(Path(metadata).read_text())
    validate_assets(training, robot, stats, metadata)
    if base_model:
        training["model"]["tokenizer_path"] = base_model
    # Native CLI snapshots serialize these dictionaries as Python literals.
    for key in ("joints", "norm_type"):
        training["data"][key] = [repr({k: v}) for k, v in config_pairs(training["data"][key], key).items()]
    robot["norm_stats"] = "norm_stats.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lingbot-export-", dir=output.parent) as temp:
        root = Path(temp) / "bundle"
        root.mkdir()
        shutil.copytree(weights, root / "checkpoints/export/hf_ckpt")
        (root / "lingbotvla_cli.yaml").write_text(yaml.safe_dump(training, sort_keys=False))
        (root / "robot_config.yaml").write_text(yaml.safe_dump(robot, sort_keys=False))
        (root / "norm_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        (root / "cyclo_input_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        CheckpointBundle(root)
        root.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("weights", "training-config", "robot-config", "norm-stats", "metadata", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    parser.add_argument("--base-model", help="Portable tokenizer/base model ID, e.g. Qwen/Qwen3-VL-4B-Instruct")
    args = parser.parse_args()
    print(export_checkpoint(**vars(args)))


if __name__ == "__main__":
    main()
