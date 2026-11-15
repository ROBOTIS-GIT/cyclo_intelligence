"""Shared, dependency-free F2 10-to-10 training phase contract."""

CONTRACT_VERSION = 2
TRAINING_PHASES = ("actor_warmup", "critic_warmup", "rl")
ACTOR_WARMUP_DATA_SCOPES = ("all", "success_only")


def validate_training_request(phase, initialization_mode):
    if phase not in TRAINING_PHASES:
        raise ValueError("Select Actor warmup, Critic warmup, or RL explicitly")
    if initialization_mode not in ("new", "resume"):
        raise ValueError("Invalid weight initialization mode")
    if phase != "actor_warmup" and initialization_mode != "resume":
        raise ValueError("Critic warmup and RL require a previous Actor bundle (Resume)")


def validate_actor_warmup_data_scope(phase, scope):
    if scope not in ACTOR_WARMUP_DATA_SCOPES:
        raise ValueError("Select All replay or Success only for Actor warmup data")
    if phase != "actor_warmup" and scope != "all":
        raise ValueError("Success-only replay filtering is available only for Actor warmup")


def validate_phase_manifest(phase, manifest):
    """Check recorded completion, not a file name or a loss threshold."""
    actor = manifest.get("completed_actor_updates", 0)
    critic = manifest.get("completed_critic_updates", 0)
    state = manifest.get("phase_state")
    completed = state.get("completed_phases", []) if isinstance(state, dict) else []
    legacy = state is None and actor > 0 and critic > 0
    if phase == "actor_warmup" and critic:
        raise ValueError("Actor warmup cannot resume an already trained Critic")
    if phase == "critic_warmup" and not (legacy or "actor_warmup" in completed or "critic_warmup" in completed):
        raise ValueError("Complete Actor warmup before Critic warmup")
    if phase == "rl" and not (legacy or "critic_warmup" in completed):
        raise ValueError("Complete Critic warmup before RL")
