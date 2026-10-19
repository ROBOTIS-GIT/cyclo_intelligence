"""Image-local evidence. Bench records the final image ID outside the image."""

import argparse
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import sys
import tempfile


def hashes(root):
    root = Path(root)
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


def main():
    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    import flash_attn
    from ABot.model.framework.ABot_M0 import ABot_M0
    from abot_training.native import TrainingPolicy

    parser = argparse.ArgumentParser()
    parser.add_argument("--flash-arch", required=True)
    parser.add_argument("--max-jobs", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as root:
        file = Path(root) / "test.parquet"
        table = pa.table({"state": [[1.0, 2.0]], "action": [[3.0, 4.0]]})
        pq.write_table(table, file)
        assert pq.read_table(file).equals(table)
    receipt = {
        "version": 1, "cyclo_commit": os.environ["CYCLO_COMMIT"],
        "upstream_commit": os.environ["ABOT_UPSTREAM_COMMIT"], "python": sys.executable,
        "python_version": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "pyarrow": {"version": pa.__version__, "parquet_roundtrip": True},
        "flash_attention": {"version": flash_attn.__version__,
            "source_sha256": "1e71dd64a9e0280e0447b8a0c2541bad4bf6ac65bdeaa2f90e51a9e57de0370d",
            "source": "flash_attn-2.8.3.tar.gz", "force_build": True,
            "cuda_archs": args.flash_arch, "max_jobs": args.max_jobs},
        "packages": {d.metadata["Name"]: d.version for d in distributions()},
        "adapter_sha256": hashes("/opt/cyclo-training"),
        "upstream_sha256": hashes("/abot"), "build_inputs_sha256": hashes("/opt/abot-build"),
    }
    Path("/opt/abot-build/receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"imports": [ABot_M0.__name__, TrainingPolicy.__name__], "pyarrow": receipt["pyarrow"]}))


if __name__ == "__main__":
    main()
