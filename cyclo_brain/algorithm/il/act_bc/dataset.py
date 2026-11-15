"""Backward-compatible ACT names for the shared LeRobot IL dataset adapter."""

from cyclo_brain.algorithm.il.common.dataset import (
    LEROBOT_CODEBASE_VERSION,
    DatasetDependencies,
    RootSelection,
    VirtualLeRobotDataset,
    load_virtual_lerobot_dataset,
    parse_success_episode_csv,
)

# Keep the established ACT public API stable while new model implementations
# depend on the neutral shared types directly.
LeRobotDatasetDependencies = DatasetDependencies
VirtualACTBCDataset = VirtualLeRobotDataset
load_virtual_act_bc_dataset = load_virtual_lerobot_dataset

__all__ = [
    "LEROBOT_CODEBASE_VERSION",
    "LeRobotDatasetDependencies",
    "RootSelection",
    "VirtualACTBCDataset",
    "load_virtual_act_bc_dataset",
    "parse_success_episode_csv",
]
