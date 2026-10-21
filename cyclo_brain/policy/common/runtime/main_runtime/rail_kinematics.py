"""NumPy-only FFW forward kinematics for execution-side TCP checks.

Inputs contain seven arm joints per action group, with grippers excluded and
the action-key order preserved. Results are metres in ``base_link``. The lift
is held at zero: its common translation cancels in displacement comparisons.
This provider does not estimate measured/world TCP poses or command a robot.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .rail_trajectory import validate_rail_keys


def _vector(text, name):
    try:
        value = np.array([float(component) for component in text.split()], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid URDF {name}') from exc
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f'URDF {name} requires three finite values')
    return value


def _rpy_rotation(rpy):
    """URDF fixed-axis roll/pitch/yaw: Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, cp, cy = np.cos(rpy)
    sr, sp, sy = np.sin(rpy)
    return np.array([
        [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
        [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
        [-sp, cp*sr, cp*cr],
    ])


@dataclass(frozen=True)
class _Step:
    rotation: object
    translation: np.ndarray
    column: object
    axis: object = None
    axis_index: object = None
    axis_sign: float = 1.


class RailKinematics:
    """Compile one or both FFW TCP chains from a local URDF once at LOAD."""

    frame = 'base_link'
    fixed_lift_position = 0.

    def __init__(self, path, keys):
        keys = list(keys)
        validate_rail_keys(keys)
        self.action_keys = tuple(key.split('.')[-1] for key in keys)
        self.sides = tuple('l' if key == 'arm_left' else 'r' for key in self.action_keys)
        self.arm_dof = 7*len(self.sides)
        self.joint_names = tuple(f'arm_{side}_joint{index}' for side in self.sides for index in range(1, 8))
        self.tcp_links = tuple(f'end_effector_{side}_link' for side in self.sides)
        self.urdf_path = str(Path(path).expanduser().resolve())
        document = Path(self.urdf_path).read_bytes()
        self.urdf_sha256 = hashlib.sha256(document).hexdigest()
        try:
            root = ET.fromstring(document)
        except ET.ParseError as exc:
            raise ValueError('Invalid RAIL robot URDF XML') from exc
        if root.tag != 'robot':
            raise ValueError('RAIL URDF requires a robot root element')
        parents = {}
        joint_names = set()
        for joint in root.findall('joint'):
            parent, child = joint.find('parent'), joint.find('child')
            name = joint.get('name')
            if (not name or parent is None or child is None
                    or not parent.get('link') or not child.get('link')):
                raise ValueError('URDF joint requires name, parent link and child link')
            if name in joint_names or child.get('link') in parents:
                raise ValueError('URDF contains duplicate joint names or multiple parents')
            joint_names.add(name)
            parents[child.get('link')] = joint
        self._chains = tuple(self._compile_chain(parents, side, index*7)
                             for index, side in enumerate(self.sides))

    @classmethod
    def from_urdf(cls, path, keys):
        return cls(path, keys)

    def _compile_chain(self, parents, side, first_column):
        expected = {f'arm_{side}_joint{index}': first_column+index-1 for index in range(1, 8)}
        link = f'end_effector_{side}_link'
        reverse = []
        visited = set()
        while link != self.frame:
            if link in visited:
                raise ValueError('Cycle in RAIL URDF TCP chain')
            visited.add(link)
            if link not in parents:
                raise ValueError(f'Missing URDF TCP chain link {link}')
            joint = parents[link]
            reverse.append(joint)
            link = joint.find('parent').get('link')
        steps, found = [], set()
        for joint in reversed(reverse):
            name, kind = joint.get('name'), joint.get('type')
            origin = joint.find('origin')
            origin = {} if origin is None else origin.attrib
            translation = _vector(origin.get('xyz', '0 0 0'), name+' origin xyz')
            rpy = _vector(origin.get('rpy', '0 0 0'), name+' origin rpy')
            rotation = _rpy_rotation(rpy) if np.any(rpy) else None
            if name in expected:
                if kind not in ('revolute', 'continuous') or joint.find('mimic') is not None:
                    raise ValueError(f'RAIL arm joint {name} must be an independent revolute/continuous joint')
                axis_element = joint.find('axis')
                axis = _vector(axis_element.get('xyz', '1 0 0') if axis_element is not None else '1 0 0',
                               name+' axis')
                length = np.linalg.norm(axis)
                if not np.isfinite(length) or length <= 0:
                    raise ValueError(f'RAIL joint {name} requires a nonzero finite axis')
                axis /= length
                nonzero = np.flatnonzero(axis)
                axis_index = int(nonzero[0]) if len(nonzero) == 1 else None
                axis_sign = float(axis[axis_index]) if axis_index is not None else 1.
                steps.append(_Step(rotation, translation, expected[name], axis, axis_index, axis_sign))
                found.add(name)
            elif kind == 'fixed':
                steps.append(_Step(rotation, translation, None))
            elif name == 'lift_joint' and kind == 'prismatic' and joint.find('mimic') is None:
                # A zero prismatic displacement leaves only this joint's origin.
                # Other uncontrolled moving ancestors must not silently default.
                axis_element = joint.find('axis')
                axis = _vector(axis_element.get('xyz', '1 0 0') if axis_element is not None else '1 0 0',
                               name+' axis')
                length = np.linalg.norm(axis)
                if not np.isfinite(length) or length <= 0:
                    raise ValueError(f'RAIL joint {name} requires a nonzero finite axis')
                steps.append(_Step(rotation, translation, None))
            else:
                raise ValueError(f'Unsupported active URDF chain joint {name} ({kind})')
        missing = set(expected)-found
        if missing:
            raise ValueError('Missing active arm joints in URDF TCP chain: '+', '.join(sorted(missing)))
        # TCP position needs no accumulated orientation. Walk from the TCP to
        # the base, applying each joint's motion and then its constant origin.
        return tuple(reversed(steps))

    def tcp_positions(self, q):
        """Map finite joint positions ``[..., arm_dof]`` to ``[..., arms, 3]``."""
        q = np.asarray(q, dtype=float)
        if q.ndim < 1 or q.shape[-1] != self.arm_dof or not np.isfinite(q).all():
            raise ValueError(f'RAIL FK requires finite arm positions with last dimension {self.arm_dof}')
        shape = q.shape[:-1]
        positions = []
        cosine, sine = np.cos(q), np.sin(q)
        for chain in self._chains:
            position = np.zeros(shape+(3,))
            for step in chain:
                if step.column is not None:
                    c, s = cosine[..., step.column], sine[..., step.column]
                    if step.axis_index is not None:
                        # FFW arm axes are coordinate axes. Rotate only the
                        # two affected components without allocating matrices.
                        a, b = (step.axis_index+1) % 3, (step.axis_index+2) % 3
                        s = s*step.axis_sign
                        rotated_a = c*position[..., a]-s*position[..., b]
                        position[..., b] = s*position[..., a]+c*position[..., b]
                        position[..., a] = rotated_a
                    else:
                        # General URDF axes retain the same Rodrigues rotation.
                        c, s = c[..., None], s[..., None]
                        axial = np.sum(position*step.axis, axis=-1, keepdims=True)
                        position = (c*position+s*np.cross(step.axis, position)
                                    +(1-c)*axial*step.axis)
                if step.rotation is not None:
                    position = position @ step.rotation.T
                position += step.translation
            positions.append(position)
        result = np.stack(positions, axis=-2)
        if not np.isfinite(result).all():
            raise ValueError('RAIL FK produced nonfinite TCP positions')
        return result

    __call__ = tcp_positions
