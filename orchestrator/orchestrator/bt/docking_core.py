"""Pure planar ArUco docking geometry and bounded holonomic control."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DockConfig:
    distance: float = .85
    xy_tolerance: float = .035
    yaw_tolerance: float = .06
    max_speed: float = .10
    max_yaw_rate: float = .18
    timeout: float = 120.
    lost_timeout: float = 3.
    stable_seconds: float = .7
    min_speed: float = .035
    min_yaw_rate: float = .04

    def __post_init__(self):
        bounds = {'distance': (.35, 2.), 'xy_tolerance': (.01, .15),
                  'yaw_tolerance': (.02, .25), 'max_speed': (.01, .2),
                  'max_yaw_rate': (.02, .4), 'timeout': (5., 180.),
                  'lost_timeout': (.5, 10.), 'stable_seconds': (.3, 3.), 'min_speed': (0., .08), 'min_yaw_rate': (0., .15)}
        for key, (low, high) in bounds.items():
            value = getattr(self, key)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'{key} must be finite and in [{low}, {high}]')
        if self.max_speed < math.sqrt(2)*self.min_speed or self.max_yaw_rate < self.min_yaw_rate:
            raise ValueError('Maximum speeds must exceed minimum command speeds')


def dock_error(position, normal, distance):
    """Marker +Z points OUT of its face. Target faces back toward the marker."""
    if not all(math.isfinite(float(v)) for v in (*position, *normal)):
        raise ValueError('Non-finite marker pose')
    nx, ny, nz = normal
    norm = math.hypot(nx, ny)
    if norm < .9 or abs(nz) > .35:
        raise ValueError('Docking requires a near-vertical marker')
    nx, ny = nx / norm, ny / norm
    px, py = position[:2]
    if px * nx + py * ny >= -.2:
        raise ValueError('Marker front must face the robot')
    if math.hypot(px, py) > 3.5:
        raise ValueError('Marker beyond local docking range (3.5 m)')
    return px + nx * distance, py + ny * distance, math.atan2(-ny, -nx)


def velocity(error, config):
    x, y, yaw = error
    if not all(math.isfinite(v) for v in error):
        raise ValueError('Non-finite docking error')
    settled = math.hypot(x, y) <= config.xy_tolerance and abs(yaw) <= config.yaw_tolerance
    if settled:
        return (0., 0., 0.), True
    # FFW's component-wise deadband otherwise stalls before the pose tolerance.
    axis_tol = config.xy_tolerance / math.sqrt(2)
    def axis(error):
        return 0. if abs(error) <= axis_tol else math.copysign(max(config.min_speed, .35*abs(error)), error)
    vx, vy = axis(x), axis(y)
    norm = math.hypot(vx, vy)
    if norm > config.max_speed:
        vx, vy = vx*config.max_speed/norm, vy*config.max_speed/norm
    if abs(yaw) > .45:
        vx = vy = 0.
    wz = 0. if abs(yaw) <= config.yaw_tolerance else math.copysign(
        min(config.max_yaw_rate, max(config.min_yaw_rate, .6*abs(yaw))), yaw)
    return (vx, vy, wz), False


def obstacle_in_motion(points, command, radius=.43, clearance=.04):
    """Conservative circular footprint plus swept translation corridor, in base frame."""
    vx, vy, wz = command
    speed = math.hypot(vx, vy)
    if speed < 1e-5 and abs(wz) < 1e-5:
        return False
    reach = clearance + speed * .5
    for x, y in points:
        if abs(wz) > .01 and math.hypot(x, y) < radius + clearance:
            return True
        if speed > 1e-5:
            along, across = (x*vx+y*vy)/speed, abs(x*vy-y*vx)/speed
            if 0 < along < radius+reach and across < radius+clearance:
                return True
    return False
