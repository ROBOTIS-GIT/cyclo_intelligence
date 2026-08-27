"""SMDP TD3 for executed prefixes produced by the official ACT policy."""

from .batch import ACTTD3Batch
from .config import (
    ACT_TD3_ACTOR_OBJECTIVES,
    ACTTD3Config,
    canonicalize_act_td3_actor_objective,
)
from .functional import (
    ACTSMDPReturns,
    actor_update_is_due,
    build_smdp_returns,
    masked_deterministic_bc_l1,
    q_weight_for_actor_update,
    smooth_target_action_chunks,
)
from .learner import ACTTD3Learner, ACTTD3UpdateResult
from .lerobot_offline import (
    ACTTD3LeRobotCollator,
    FixedHorizonLeRobotACTTD3Dataset,
    LeRobotACTTD3Transition,
    VirtualCumulativeLeRobotACTTD3Dataset,
)
from .offline_warmup import (
    ACTTD3CriticWarmupProgress,
    ACTTD3CriticWarmupRunner,
)
from .offline_training import (
    ACTTD3OfflineTrainingProgress,
    ACTTD3OfflineTrainingRunner,
    RLMetricHistoryPoint,
)
from .training_identity import (
    ACTTD3TrainingDataIdentity,
    ACTTD3TrainingIdentityFile,
    build_act_td3_multi_root_training_data_identity,
    build_act_td3_training_data_identity,
)

__all__ = [
    "ACTSMDPReturns",
    "ACT_TD3_ACTOR_OBJECTIVES",
    "ACTTD3Batch",
    "ACTTD3Config",
    "ACTTD3CriticWarmupProgress",
    "ACTTD3CriticWarmupRunner",
    "ACTTD3OfflineTrainingProgress",
    "ACTTD3OfflineTrainingRunner",
    "ACTTD3Learner",
    "ACTTD3LeRobotCollator",
    "ACTTD3TrainingDataIdentity",
    "ACTTD3TrainingIdentityFile",
    "ACTTD3UpdateResult",
    "FixedHorizonLeRobotACTTD3Dataset",
    "LeRobotACTTD3Transition",
    "RLMetricHistoryPoint",
    "VirtualCumulativeLeRobotACTTD3Dataset",
    "actor_update_is_due",
    "build_act_td3_multi_root_training_data_identity",
    "build_act_td3_training_data_identity",
    "build_smdp_returns",
    "canonicalize_act_td3_actor_objective",
    "masked_deterministic_bc_l1",
    "q_weight_for_actor_update",
    "smooth_target_action_chunks",
]
