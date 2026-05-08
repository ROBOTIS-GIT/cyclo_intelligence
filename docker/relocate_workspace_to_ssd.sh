#!/bin/bash
# Relocate cyclo_intelligence's workspace + huggingface bind-mount targets
# from the SD card (/dev/mmcblk0p1) to NVMe (/mnt/ssd). The docker-compose
# mounts (./workspace, ./huggingface) are kept as-is — they resolve via
# symlinks to /mnt/ssd/cyclo_intelligence/{workspace,huggingface}.
#
# Run with: sudo bash relocate_workspace_to_ssd.sh
set -euo pipefail

REPO=/home/robotis/cyclo_intelligence
SRC_W=$REPO/docker/workspace
SRC_H=$REPO/docker/huggingface
DST_W=/mnt/ssd/cyclo_intelligence/workspace
DST_H=/mnt/ssd/cyclo_intelligence/huggingface

echo "=== 1/6  containers using these mounts: stop them ==="
for c in cyclo_intelligence groot_server lerobot_server; do
    if docker ps -a --format '{{.Names}}' | grep -qx "$c"; then
        echo "  stopping $c"
        docker stop "$c" || true
    fi
done

echo "=== 2/6  prepare destination on NVMe ==="
mkdir -p "$DST_W" "$DST_H"
chown -R robotis:robotis /mnt/ssd/cyclo_intelligence

echo "=== 3/6  copy workspace -> $DST_W (this is the slow part) ==="
if [ -d "$SRC_W" ] && [ ! -L "$SRC_W" ]; then
    rsync -aHP --remove-source-files "$SRC_W"/ "$DST_W"/
    find "$SRC_W" -depth -type d -empty -delete || true
fi

echo "=== 4/6  copy huggingface -> $DST_H ==="
if [ -d "$SRC_H" ] && [ ! -L "$SRC_H" ]; then
    rsync -aHP --remove-source-files "$SRC_H"/ "$DST_H"/
    find "$SRC_H" -depth -type d -empty -delete || true
fi

echo "=== 5/6  symlink $SRC_W -> $DST_W and $SRC_H -> $DST_H ==="
[ -e "$SRC_W" ] && [ ! -L "$SRC_W" ] && rmdir "$SRC_W" || true
[ -e "$SRC_H" ] && [ ! -L "$SRC_H" ] && rmdir "$SRC_H" || true
[ -L "$SRC_W" ] || ln -s "$DST_W" "$SRC_W"
[ -L "$SRC_H" ] || ln -s "$DST_H" "$SRC_H"

echo "=== 6/6  result ==="
ls -la "$SRC_W" "$SRC_H"
df -h / /mnt/ssd | tail -2

echo
echo "Done. Restart containers with:"
echo "  docker start cyclo_intelligence groot_server lerobot_server"
echo "  docker start ai_worker  # (was the original failing one)"
