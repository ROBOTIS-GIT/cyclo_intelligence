"""Package an upstream run with explicit, reviewed Cyclo IO metadata."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from abot_engine.bundle import Bundle


def export(checkpoint, run_dir, metadata, output):
    checkpoint, run_dir, output = Path(checkpoint), Path(run_dir), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if checkpoint.suffix != ".pt" or not checkpoint.is_file():
        raise ValueError("Expected an upstream .pt state_dict checkpoint")
    meta = json.loads(Path(metadata).read_text())
    meta["checkpoint"] = "checkpoints/model.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".abot-export-", dir=output.parent) as directory:
        staging = Path(directory) / "bundle"
        (staging / "checkpoints").mkdir(parents=True)
        for name in ("config.yaml", "dataset_statistics.json"):
            shutil.copy2(run_dir / name, staging / name)
        (staging / "cyclo_input_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        # Validate metadata before copying a large checkpoint. The placeholder is
        # local to staging; no training files or existing bundles are modified.
        (staging / meta["checkpoint"]).touch()
        Bundle(staging, meta["robot_type"])
        shutil.copy2(checkpoint, staging / meta["checkpoint"])
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        staging.rename(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(export(args.checkpoint, args.run_dir, args.metadata, args.output))
