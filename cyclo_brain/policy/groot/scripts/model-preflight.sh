#!/bin/sh
set -eu
python /app/verify_model_overlay.py --overlay /app/model_overrides --base-root /gr00t --check-imports
exec /init "$@"
