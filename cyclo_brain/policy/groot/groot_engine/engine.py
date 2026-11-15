"""Choose the GR00T implementation once per loaded session."""
from runtime.inference_engine import GR00TInference


class RoutedGR00TEngine:
    def __init__(self):
        self._engine = GR00TInference()

    def load_policy(self, request):
        playground = getattr(request, "action_request_mode", "async") == "tt_rtc" or bool(getattr(request, "rlt_enabled", False))
        if playground:
            from runtime.playground_inference import GR00TInference as PlaygroundInference
            engine_type = PlaygroundInference
        else:
            engine_type = GR00TInference
        if type(self._engine) is not engine_type:
            self._engine.cleanup()
            self._engine = engine_type()
        return self._engine.load_policy(request)

    def get_action_chunk(self, request):
        return self._engine.get_action_chunk(request)

    def cleanup(self):
        return self._engine.cleanup()

    @property
    def is_ready(self):
        return self._engine.is_ready

    def policy_update_command(self, request):
        method = getattr(self._engine, "policy_update_command", None)
        return method(request) if method else {"success": False, "message": "Load an RLT bundle first"}


def create_engine():
    return RoutedGR00TEngine()


__all__ = ["GR00TInference", "create_engine"]
