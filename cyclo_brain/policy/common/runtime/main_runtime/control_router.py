"""Route a loaded session to ordinary/RAIL or Playground execution."""
from .control_loop import ControlLoop
from .playground_control_loop import ControlLoop as PlaygroundControlLoop


class ControlRouter:
    def __init__(self, requester, **options):
        self._requester = requester
        self._options = options
        loop_type = PlaygroundControlLoop if options.get("action_request_mode") == "tt_rtc" else ControlLoop
        self._loop = self._create_loop(loop_type)
        self._background = False

    def _create_loop(self, loop_type):
        options = dict(self._options)
        if loop_type is PlaygroundControlLoop:
            options.pop("trace_path", None)
            options.pop("rail_interpolation_hz", None)
        elif options.get("action_request_mode") == "tt_rtc":
            options["action_request_mode"] = "async"
        return loop_type(self._requester, **options)

    def configure(self, **options):
        playground = options.get("action_request_mode") == "tt_rtc" or options.get("rlt_enabled", False)
        if playground and options.get("vla_rail_enabled", False):
            raise ValueError("TT-RTC/RLT and VLA-RAIL cannot be enabled together")
        loop_type = PlaygroundControlLoop if playground else ControlLoop
        if type(self._loop) is not loop_type:
            self._loop.shutdown()
            self._loop = self._create_loop(loop_type)
            if self._background:
                self._loop.run_background()
        if playground:
            for key in ("vla_rail_enabled", "inference_hz", "chunk_align_window_s", "per_step_actions"):
                options.pop(key, None)
        else:
            for key in ("model_path", "rlt_enabled", "tt_rtc_horizon", "tt_rtc_action_dim",
                        "tt_rtc_min_delay_steps", "tt_rtc_max_delay_steps", "defer_initial_pose_sync_activation"):
                options.pop(key, None)
        self._loop.configure(**options)

    def run_background(self):
        self._background = True
        self._loop.run_background()

    def __getattr__(self, name):
        return getattr(self._loop, name)
