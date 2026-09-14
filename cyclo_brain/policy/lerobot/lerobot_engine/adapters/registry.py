"""Single selection point for model-specific execution and input contracts."""

from .definition import AdapterDefinition
from .lingbot_va import ADAPTER as LINGBOT_VA
from .multi_task_dit import ADAPTER as MULTI_TASK_DIT
from .diffusion import ADAPTER as DIFFUSION
from .xvla import ADAPTER as XVLA
from .wall_x import ADAPTER as WALL_X
from .groot import ADAPTER as GROOT
from .fastwam import ADAPTER as FASTWAM
from .molmoact2 import ADAPTER as MOLMOACT2


class AdapterRegistry:
    def __init__(self):
        self._definitions = {}
        self._chunk = AdapterDefinition()

    def register(self, policy_type, definition):
        if not isinstance(policy_type, str) or not policy_type.strip():
            raise ValueError("adapter policy type must be non-empty")
        if policy_type in self._definitions:
            raise ValueError(f"duplicate adapter for {policy_type}")
        if not isinstance(definition, AdapterDefinition):
            raise TypeError("expected AdapterDefinition")
        self._definitions[policy_type] = definition

    def resolve(self, policy_type):
        # Existing chunk policies keep the unchanged public chunk API.
        return self._definitions.get(policy_type, self._chunk)


registry = AdapterRegistry()
registry.register("multi_task_dit", MULTI_TASK_DIT)
registry.register("lingbot_va", LINGBOT_VA)
registry.register("diffusion", DIFFUSION)
registry.register("xvla", XVLA)
registry.register("wall_x", WALL_X)
registry.register("groot", GROOT)
registry.register("fastwam", FASTWAM)
registry.register("molmoact2", MOLMOACT2)


def resolve_adapter(policy_type):
    return registry.resolve(policy_type)
