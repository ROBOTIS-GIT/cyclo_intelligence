"""Source-tree path fallbacks for runtime modules.

The two-process policy runtime (Process A inference_server +
Process B control_publisher) needs to find sibling SDK packages
(`robot_client`, `post_processing`, `zenoh_ros2_sdk`) and the
orchestrator's per-robot YAMLs at module-import time. In a container
deployment the env vars POST_PROCESSING_SDK_PATH / ROBOT_CLIENT_SDK_PATH /
ORCHESTRATOR_CONFIG_PATH point at bind-mounted directories and that's
all that's needed. When running from a source checkout (host-side dev
or unit tests), the runtime files live deep in the tree and we need
to walk up to the repo root — that's what this helper handles.
"""

from pathlib import Path
from typing import Union


def dev_sdk_path(caller_file: Union[str, Path], depth: int, *parts: str) -> str:
    """Return ``<caller's Nth parent>/<parts>`` as a string, or '' if the
    file isn't deep enough.

    Used as the fallback in ``os.environ.get(ENV, dev_sdk_path(__file__, ...))``
    so an unset env var maps to a sensible source-tree path during dev,
    but doesn't crash with IndexError when the runtime file has been
    COPY'd into a shallow location like /app/runtime/ inside a container
    (in which case the env var is the only valid resolution).
    """
    parents = Path(caller_file).resolve().parents
    if len(parents) <= depth:
        return ""
    return str(parents[depth].joinpath(*parts))
