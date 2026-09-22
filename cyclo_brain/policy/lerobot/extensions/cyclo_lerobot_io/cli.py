"""Inspect or explicitly add channel metadata to an existing local checkpoint."""

import argparse
import json
from pathlib import Path

from .mapping import FILENAME, feature_dim, mapping_from_features, validate_mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset-info", type=Path, help="Dataset meta/info.json; verify training order yourself"
    )
    source.add_argument("--mapping", type=Path, help="Explicit ordered channel metadata")
    parser.add_argument(
        "--write", action="store_true", help="Confirm the training order and write; never overwrite"
    )
    args = parser.parse_args()
    root = args.checkpoint
    if not (root / "config.json").exists() and (root / "pretrained_model/config.json").exists():
        root = root / "pretrained_model"
    config = json.loads((root / "config.json").read_text())
    if args.dataset_info:
        info = json.loads(args.dataset_info.read_text())
        mapping = mapping_from_features(
            info["features"], repo_id=info.get("repo_id"), revision=info.get("revision")
        )
        if mapping is None:
            parser.error("Dataset has no channel names; provide --mapping with verified names")
        mapping["dataset"]["info_path"] = str(args.dataset_info.resolve())
    else:
        mapping = json.loads(args.mapping.read_text())
    validate_mapping(
        mapping,
        state_dim=feature_dim(config["input_features"], "observation.state"),
        action_dim=feature_dim(config["output_features"], "action"),
    )
    content = json.dumps(mapping, indent=2) + "\n"
    if args.write:
        with (root / FILENAME).open("x") as stream:
            stream.write(content)
    print(content, end="")
    if not args.write:
        print("Validated draft only. Verify the actual training channel order before using --write.")


if __name__ == "__main__":
    main()
