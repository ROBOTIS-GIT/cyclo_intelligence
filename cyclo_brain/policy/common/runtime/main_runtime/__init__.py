"""Main process runtime.

The Main process owns the external lifecycle service and the robot-facing
control loop. It asks the Engine process for action lists through the internal
EngineCommand service.
"""

from .inference_requester import InferenceRequester
from .session_state import SessionState

__all__ = ["InferenceRequester", "SessionState"]
