from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "cyclo_data"))

from cyclo_data.reader import frame_timestamps


def test_load_frame_timestamps_uses_single_file_reader(tmp_path, monkeypatch):
    parquet_path = tmp_path / "cam_timestamps.parquet"
    pq.write_table(
        pa.table(
            {
                "frame_index": pa.array([0, 1, 2], type=pa.int32()),
                "header_stamp_ns": pa.array([10, 20, 30], type=pa.int64()),
                "recv_ns": pa.array([11, 21, 31], type=pa.int64()),
            }
        ),
        parquet_path,
    )

    def fail_read_table(*args, **kwargs):
        raise AssertionError("load_frame_timestamps must not use pq.read_table")

    monkeypatch.setattr(frame_timestamps.pq, "read_table", fail_read_table)

    loaded = frame_timestamps.load_frame_timestamps(Path(parquet_path), "cam")

    assert loaded.camera == "cam"
    assert loaded.frame_index.tolist() == [0, 1, 2]
    assert loaded.header_stamp_ns.tolist() == [10, 20, 30]
    assert loaded.recv_ns.tolist() == [11, 21, 31]
