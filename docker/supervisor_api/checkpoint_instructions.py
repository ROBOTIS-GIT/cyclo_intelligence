"""Expose the same checkpoint language reader used by the policy runtime."""
import importlib.util
import os
from pathlib import Path

roots = [Path(__file__).resolve().parents[2], Path(os.environ.get(
    "CYCLO_SUPERVISOR_API_REPO_MOUNT", "/root/ros2_ws/src/cyclo_intelligence"
))]
core = next(root / "cyclo_brain/policy/common/runtime/checkpoint_instructions.py"
            for root in roots if (root / "cyclo_brain/policy/common/runtime/checkpoint_instructions.py").is_file())
spec = importlib.util.spec_from_file_location("cyclo_checkpoint_instructions", core)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MAX_BYTES = module.MAX_BYTES
MAX_ITEMS = module.MAX_ITEMS
read_checkpoint_instructions = module.read_checkpoint_instructions
