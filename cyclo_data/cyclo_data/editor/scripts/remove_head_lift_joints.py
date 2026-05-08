#!/usr/bin/env python3
"""
Post-process LeRobot v2.1 dataset to remove head/lift joints (indices 16, 17, 18).

These joints (head_joint1, head_joint2, lift_joint) are 100% NaN in action data
across all episodes, making them unusable for training.

Removes indices [16, 17, 18] from both observation.state and action,
reducing dimensionality from 22 to 19.

Usage:
    python remove_head_lift_joints.py /path/to/dataset_lerobot_v21
"""

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


REMOVE_INDICES = {16, 17, 18}
KEEP_INDICES = [i for i in range(22) if i not in REMOVE_INDICES]
EXPECTED_OLD_DIM = 22
EXPECTED_NEW_DIM = 19


def process_parquet_files(dataset_path: Path):
    """Remove indices [16,17,18] from observation.state and action in all parquet files."""
    parquet_dir = dataset_path / "data" / "chunk-000"
    parquet_files = sorted(parquet_dir.glob("episode_*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")

    for pf in parquet_files:
        table = pq.read_table(pf)

        new_columns = {}
        for col_name in ["observation.state", "action"]:
            col = table.column(col_name)
            # Extract values, filter indices, rebuild
            new_lists = []
            for row in col:
                vals = row.as_py()
                assert len(vals) == EXPECTED_OLD_DIM, f"{pf.name} {col_name}: expected {EXPECTED_OLD_DIM}, got {len(vals)}"
                new_vals = [vals[i] for i in KEEP_INDICES]
                new_lists.append(new_vals)

            new_arr = pa.array(new_lists, type=pa.list_(pa.float32(), EXPECTED_NEW_DIM))
            new_columns[col_name] = new_arr

        # Rebuild table with new columns
        col_names = table.column_names
        new_cols = []
        for name in col_names:
            if name in new_columns:
                new_cols.append(new_columns[name])
            else:
                new_cols.append(table.column(name))

        new_table = pa.table(dict(zip(col_names, new_cols)))

        # Preserve HuggingFace metadata but update feature shapes
        metadata = table.schema.metadata
        if metadata and b"huggingface" in metadata:
            hf_meta = json.loads(metadata[b"huggingface"])
            if "info" in hf_meta and "features" in hf_meta["info"]:
                features = hf_meta["info"]["features"]
                for key in ["observation.state", "action"]:
                    if key in features:
                        # Update the fixed_size_list length in HF metadata
                        if "feature" in features[key] and "length" in features[key]:
                            features[key]["length"] = EXPECTED_NEW_DIM
                hf_meta["info"]["features"] = features
            new_metadata = {**metadata, b"huggingface": json.dumps(hf_meta).encode()}
            new_table = new_table.replace_schema_metadata(new_metadata)

        pq.write_table(new_table, pf)
        print(f"  Processed {pf.name}: {len(table)} rows")

    print(f"Done: {len(parquet_files)} parquet files updated")


def process_episodes_stats(dataset_path: Path):
    """Remove indices [16,17,18] from stat vectors in episodes_stats.jsonl."""
    stats_path = dataset_path / "meta" / "episodes_stats.jsonl"
    lines = stats_path.read_text().strip().split("\n")
    print(f"Processing episodes_stats.jsonl: {len(lines)} episodes")

    new_lines = []
    for line in lines:
        entry = json.loads(line)
        stats = entry["stats"]

        for key in ["observation.state", "action"]:
            if key in stats:
                for stat_name in ["mean", "std", "min", "max"]:
                    if stat_name in stats[key]:
                        old_vals = stats[key][stat_name]
                        assert len(old_vals) == EXPECTED_OLD_DIM, (
                            f"Episode {entry['episode_index']} {key}.{stat_name}: "
                            f"expected {EXPECTED_OLD_DIM}, got {len(old_vals)}"
                        )
                        stats[key][stat_name] = [old_vals[i] for i in KEEP_INDICES]
                # count stays unchanged

        entry["stats"] = stats
        new_lines.append(json.dumps(entry))

    stats_path.write_text("\n".join(new_lines) + "\n")
    print("Done: episodes_stats.jsonl updated")


def process_info_json(dataset_path: Path):
    """Update shape and names in info.json."""
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    for key in ["observation.state", "action"]:
        feature = info["features"][key]

        old_shape = feature["shape"]
        assert old_shape == [EXPECTED_OLD_DIM], f"{key} shape: expected [{EXPECTED_OLD_DIM}], got {old_shape}"
        feature["shape"] = [EXPECTED_NEW_DIM]

        if feature.get("names"):
            old_names = feature["names"]
            assert len(old_names) == EXPECTED_OLD_DIM, f"{key} names: expected {EXPECTED_OLD_DIM}, got {len(old_names)}"
            feature["names"] = [old_names[i] for i in KEEP_INDICES]

        info["features"][key] = feature

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
        f.write("\n")

    print(f"Done: info.json updated (shape: [{EXPECTED_NEW_DIM}])")


def verify(dataset_path: Path):
    """Verify the post-processed dataset."""
    import math

    print("\n=== Verification ===")

    # Check info.json
    with open(dataset_path / "meta" / "info.json") as f:
        info = json.load(f)
    for key in ["observation.state", "action"]:
        shape = info["features"][key]["shape"]
        names = info["features"][key].get("names", [])
        assert shape == [EXPECTED_NEW_DIM], f"FAIL: {key} shape = {shape}"
        if names:
            assert len(names) == EXPECTED_NEW_DIM, f"FAIL: {key} names len = {len(names)}"
        print(f"  info.json {key}: shape={shape}, names_len={len(names)} OK")

    # Check first and last parquet
    parquet_dir = dataset_path / "data" / "chunk-000"
    parquet_files = sorted(parquet_dir.glob("episode_*.parquet"))
    for pf in [parquet_files[0], parquet_files[-1]]:
        table = pq.read_table(pf)
        for col_name in ["observation.state", "action"]:
            col = table.column(col_name)
            dim = len(col[0].as_py())
            assert dim == EXPECTED_NEW_DIM, f"FAIL: {pf.name} {col_name} dim = {dim}"

            # Check no NaN in action
            if col_name == "action":
                nan_count = sum(
                    1 for row in col for v in row.as_py()
                    if v is None or math.isnan(v)
                )
                assert nan_count == 0, f"FAIL: {pf.name} action has {nan_count} NaN values"

            print(f"  {pf.name} {col_name}: dim={dim} OK")

    # Check episodes_stats
    stats_path = dataset_path / "meta" / "episodes_stats.jsonl"
    lines = stats_path.read_text().strip().split("\n")
    for line in [lines[0], lines[-1]]:
        entry = json.loads(line)
        for key in ["observation.state", "action"]:
            for stat_name in ["mean", "std", "min", "max"]:
                vals = entry["stats"][key][stat_name]
                assert len(vals) == EXPECTED_NEW_DIM, (
                    f"FAIL: ep {entry['episode_index']} {key}.{stat_name} len = {len(vals)}"
                )
    print(f"  episodes_stats.jsonl: first/last episodes OK")

    # Full NaN check on all parquets
    total_nan = 0
    for pf in parquet_files:
        table = pq.read_table(pf)
        action_col = table.column("action")
        for row in action_col:
            for v in row.as_py():
                if v is None or math.isnan(v):
                    total_nan += 1
    assert total_nan == 0, f"FAIL: total NaN in action across all episodes: {total_nan}"
    print(f"  Full NaN check on all {len(parquet_files)} episodes: 0 NaN values OK")

    print("\n=== All verifications passed ===")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <dataset_path>")
        sys.exit(1)

    dataset_path = Path(sys.argv[1])
    if not dataset_path.exists():
        print(f"Error: {dataset_path} does not exist")
        sys.exit(1)

    print(f"Processing dataset: {dataset_path}")
    print(f"Removing indices {sorted(REMOVE_INDICES)} (head_joint1, head_joint2, lift_joint)")
    print(f"Dimension change: {EXPECTED_OLD_DIM} -> {EXPECTED_NEW_DIM}")
    print()

    process_parquet_files(dataset_path)
    print()
    process_episodes_stats(dataset_path)
    print()
    process_info_json(dataset_path)

    verify(dataset_path)


if __name__ == "__main__":
    main()
