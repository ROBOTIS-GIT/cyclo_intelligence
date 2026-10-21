#!/usr/bin/env bash
# Restore the optional DLA compiler omitted by the Jetson LeRobot base image.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
vendor_dir="$repo_root/docker/workspace/runtime_vendor/jetson"
source_image=${CYCLO_TRT_VENDOR_IMAGE:-robotis/groot-zenoh:1.3.5-arm64}
mkdir -p "$vendor_dir"
vendor_container=$(docker create --entrypoint true "$source_image")
trap 'docker rm "$vendor_container" >/dev/null' EXIT
docker cp "$vendor_container:/usr/lib/aarch64-linux-gnu/nvidia/libnvdla_compiler.so" "$vendor_dir/libnvdla_compiler.so.tmp"
test -s "$vendor_dir/libnvdla_compiler.so.tmp"
mv "$vendor_dir/libnvdla_compiler.so.tmp" "$vendor_dir/libnvdla_compiler.so"
python3 - "$vendor_dir" "$source_image" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
path = root / 'libnvdla_compiler.so'
manifest = {'source_image': sys.argv[2], 'source_path': '/usr/lib/aarch64-linux-gnu/nvidia/libnvdla_compiler.so',
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'size': path.stat().st_size,
            'purpose': 'Jetson TensorRT dependency; GPU inference only, no DLA execution selected'}
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest))
PY
