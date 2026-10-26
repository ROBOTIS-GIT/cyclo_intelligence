"""Local holonomic docking to a DICT_4X4_50 marker using calibrated head images.

Run from a clear pre-docking position with no active Navigation goal. Distance
is measured from base_frame to the marker face; marker_size excludes white margin.
"""
from orchestrator.bt.actions.base_action import BaseAction
from orchestrator.bt.bt_core import NodeStatus
from orchestrator.bt.docking.core import DockConfig


class ArucoDock(BaseAction):
    @classmethod
    def from_xml_params(cls, context, name, params):
        # Cleared optional editor fields mean use the constructor default.
        # Keep explicit non-empty values subject to normal validation.
        values = {key: value for key, value in params.items()
                  if value is not None and not (isinstance(value, str) and not value.strip())}
        try:
            action = cls(context.node, **values)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'ArucoDock {name}: {exc}') from exc
        action.name = name
        return action

    def __init__(self, node, marker_id: int = 10, marker_size: float = .30,
                 target_distance: float = .85, position_tolerance: float = .035,
                 yaw_tolerance: float = .06, max_speed: float = .10,
                 max_yaw_rate: float = .18, timeout: float = 120.,
                 lost_timeout: float = 3., stable_seconds: float = .7,
                 min_speed: float = .035, min_yaw_rate: float = .04,
                 image_topic: str = '',
                 camera_info_topic: str = '',
                 cmd_vel_topic: str = '/cmd_vel', scan_topic: str = '/scan',
                 odom_topic: str = '/odom', base_frame: str = 'base_link',
                 image_mode: str = 'raw', footprint_profile: str = 'circle',
                 camera_profile: str = 'VGA'):
        super().__init__(node, name='ArucoDock')
        self.config = DockConfig(*map(float, (target_distance, position_tolerance, yaw_tolerance,
                                              max_speed, max_yaw_rate, timeout, lost_timeout, stable_seconds, min_speed, min_yaw_rate)))
        self.marker_id, self.marker_size = int(marker_id), float(marker_size)
        if float(marker_id) != self.marker_id or not 0 <= self.marker_id < 50:
            raise ValueError('marker_id must be an integer in [0,49]')
        if not .05 <= self.marker_size <= 1.:
            raise ValueError('marker_size must be in [0.05,1] metres')
        from orchestrator.bt.docking.camera import normalize_profile
        self.camera_profile = normalize_profile(camera_profile)
        if bool(image_topic.strip()) != bool(camera_info_topic.strip()):
            raise ValueError('Set both image_topic and camera_info_topic, or leave both empty for automatic selection')
        self.transport = (image_topic,camera_info_topic,cmd_vel_topic,scan_topic,odom_topic,base_frame)
        if not all(isinstance(s,str) and s.strip() for s in self.transport[2:]):
            raise ValueError('Motion sensor topics and base_frame must be non-empty')
        from orchestrator.bt.docking.rectification import DockRectifier
        self.image_mode = DockRectifier(str(image_mode).strip().lower()).mode
        if footprint_profile not in ("circle", "sg2_autonomy_base"):
            raise ValueError("Unknown footprint_profile")
        self.footprint_profile = footprint_profile
        self.runtime = None
        self.failure = False
        self.outcome = None

    def tick(self):
        if self.outcome is not None: return self.outcome
        if self.failure: return NodeStatus.FAILURE
        if self.runtime is None:
            try:
                from orchestrator.bt.docking.runtime import DockRuntime
                from orchestrator.bt.docking.camera import camera_source
                image, info, mode = camera_source(self.node, *self.transport[:2], self.image_mode)
                self.runtime = DockRuntime(self.node, self.marker_id, self.marker_size,
                                           self.config, image, info, *self.transport[2:], image_mode=mode, footprint_profile=self.footprint_profile, camera_profile=self.camera_profile)
                self.runtime.start()
            except Exception as exc:
                self.log_error(f'ArucoDock initialization failed: {exc}')
                if self.runtime: self.runtime.close(); self.runtime = None
                self.failure = True
                return NodeStatus.FAILURE
        if self.runtime.result is None: return NodeStatus.RUNNING
        self.outcome = NodeStatus.SUCCESS if self.runtime.result else NodeStatus.FAILURE
        self.runtime.close()
        self.runtime = None
        return self.outcome

    def pause(self):
        if self.runtime: self.runtime.pause()

    def resume(self):
        if self.runtime: self.runtime.resume()

    def reset(self):
        if self.runtime: self.runtime.close(); self.runtime = None
        self.failure = False
        self.outcome = None
        super().reset()

    def destroy(self):
        self.reset()
