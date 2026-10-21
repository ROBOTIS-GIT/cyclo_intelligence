"""Independent geometric and malformed-model checks for the RAIL TCP provider."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from main_runtime.rail_kinematics import RailKinematics


ROBOT_URDF = Path(__file__).resolve().parents[5]/'shared/shared/robot_configs/urdf/ffw_sg2_follower.urdf'


def planar_model(tmp_path):
    """Seven-link planar chain with a closed-form cumulative-angle solution."""
    lengths = np.array([.11, .17, .09, .05, .13, .07, .19])
    root = ET.Element('robot', name='planar_arm')
    joint = ET.SubElement(root, 'joint', name='lift_joint', type='prismatic')
    ET.SubElement(joint, 'parent', link='base_link')
    ET.SubElement(joint, 'child', link='body')
    ET.SubElement(joint, 'origin', xyz='.1 -.2 .3')
    ET.SubElement(joint, 'axis', xyz='0 0 1')
    for index in range(1, 8):
        joint = ET.SubElement(root, 'joint', name=f'arm_r_joint{index}', type='revolute')
        ET.SubElement(joint, 'parent', link='body' if index == 1 else f'r{index-1}')
        ET.SubElement(joint, 'child', link=f'r{index}')
        ET.SubElement(joint, 'origin', xyz=f'{0. if index == 1 else lengths[index-2]} 0 0')
        ET.SubElement(joint, 'axis', xyz='0 0 3')  # Non-unit input must be normalized.
    joint = ET.SubElement(root, 'joint', name='tcp_joint', type='fixed')
    ET.SubElement(joint, 'parent', link='r7')
    ET.SubElement(joint, 'child', link='end_effector_r_link')
    ET.SubElement(joint, 'origin', xyz=f'{lengths[-1]} 0 0')
    path = tmp_path/'robot.urdf'
    ET.ElementTree(root).write(path)
    return path, lengths


def test_vectorized_planar_geometry_has_independent_closed_form(tmp_path):
    path, lengths = planar_model(tmp_path)
    provider = RailKinematics.from_urdf(path, ['action.arm_right'])
    q = np.random.default_rng(7).uniform(-1., 1., (3, 11, 7))
    cumulative = np.cumsum(q, axis=-1)
    expected = np.stack((.1+(lengths*np.cos(cumulative)).sum(axis=-1),
                         -.2+(lengths*np.sin(cumulative)).sum(axis=-1),
                         np.full((3, 11), .3)), axis=-1)
    np.testing.assert_allclose(provider.tcp_positions(q)[..., 0, :], expected, atol=1e-14)
    assert provider(q).shape == (3, 11, 1, 3)
    np.testing.assert_allclose(provider(q[0, 0]), expected[0, 0][None], atol=1e-14)
    assert provider(np.empty((0, 7))).shape == (0, 1, 3)


def test_nonzero_fixed_rpy_uses_urdf_fixed_axis_order(tmp_path):
    path, lengths = planar_model(tmp_path)
    root = ET.parse(path)
    root.find("joint[@name='lift_joint']/origin").set('rpy', f'{np.pi/2} 0 {np.pi/2}')
    root.write(path)
    q = np.zeros(7); q[0] = np.pi/2
    # Planar +Y rotated Rx(90) then Rz(90) is +Z.
    expected = np.array([.1, -.2, .3+lengths.sum()])
    np.testing.assert_allclose(RailKinematics(path, ['arm_right'])(q)[0], expected, atol=1e-14)


def test_arbitrary_axis_rotation_has_independent_geometric_solution(tmp_path):
    path, lengths = planar_model(tmp_path)
    root = ET.parse(path)
    root.find("joint[@name='arm_r_joint1']/axis").set('xyz', '1 1 1')
    root.write(path)
    # A 120-degree rotation around (1, 1, 1) cycles X -> Y -> Z.
    q = np.zeros((3, 7)); q[:, 0] = [0., 2*np.pi/3, 4*np.pi/3]
    expected = np.array([.1, -.2, .3])+lengths.sum()*np.eye(3)
    provider = RailKinematics(path, ['arm_right'])
    np.testing.assert_allclose(provider(q)[:, 0], expected, atol=1e-14)
    np.testing.assert_allclose(provider(q[1])[0], expected[1], atol=1e-14)
    assert provider(np.empty((0, 7))).shape == (0, 1, 3)


def test_negative_axis_continuous_joints_follow_signed_angles(tmp_path):
    path, lengths = planar_model(tmp_path)
    root = ET.parse(path)
    for index in range(1, 8):
        joint = root.find(f"joint[@name='arm_r_joint{index}']")
        joint.set('type', 'continuous')
        joint.find('axis').set('xyz', '0 0 -3')
    root.write(path)
    q = np.random.default_rng(16).uniform(-3., 3., (13, 7))
    angles = -np.cumsum(q, axis=-1)
    expected = np.stack((.1+(lengths*np.cos(angles)).sum(axis=-1),
                         -.2+(lengths*np.sin(angles)).sum(axis=-1),
                         np.full(13, .3)), axis=-1)
    np.testing.assert_allclose(RailKinematics(path, ['arm_right'])(q)[:, 0], expected, atol=1e-14)


def test_real_robot_zero_pose_and_action_key_order_are_preserved():
    provider = RailKinematics(ROBOT_URDF, ['action.arm_right', 'action.arm_left'])
    np.testing.assert_allclose(provider(np.zeros(14)),
                               [[-.0199, -.2275, .5751], [-.0199, .2275, .5751]], atol=1e-12)
    assert provider.joint_names[:7] == tuple(f'arm_r_joint{i}' for i in range(1, 8))
    assert provider.sides == ('r', 'l')
    assert provider.frame == 'base_link'
    assert len(provider.urdf_sha256) == 64
    q = np.random.default_rng(8).normal(0., .3, (5, 14))
    right = RailKinematics(ROBOT_URDF, ['arm_right'])(q[:, :7])[:, 0]
    left = RailKinematics(ROBOT_URDF, ['arm_left'])(q[:, 7:])[:, 0]
    np.testing.assert_allclose(provider(q), np.stack((right, left), axis=1), atol=1e-14)
    reverse = RailKinematics(ROBOT_URDF, ['arm_left', 'arm_right'])
    np.testing.assert_allclose(reverse(np.concatenate((q[:, 7:], q[:, :7]), axis=-1)),
                               provider(q)[:, ::-1], atol=1e-14)


@pytest.mark.parametrize('q', [0., np.zeros(8), np.zeros((2, 6)), np.full(7, np.nan), np.full(7, np.inf)])
def test_invalid_arm_data_is_rejected(tmp_path, q):
    path, _ = planar_model(tmp_path)
    with pytest.raises(ValueError, match='finite arm positions'):
        RailKinematics(path, ['arm_right'])(q)


@pytest.mark.parametrize('keys', [[], ['head'], ['arm_right', 'action.arm_right']])
def test_invalid_action_layout_is_rejected(tmp_path, keys):
    path, _ = planar_model(tmp_path)
    with pytest.raises(ValueError):
        RailKinematics(path, keys)


@pytest.mark.parametrize('mutation,match', [
    ('missing', 'Missing active arm joints'),
    ('wrong_kind', 'independent revolute/continuous'),
    ('mimic', 'independent revolute/continuous'),
    ('unknown_ancestor', 'Unsupported active'),
    ('bad_axis', 'nonzero finite axis'),
    ('bad_xyz', 'three finite'),
    ('nonfinite_rpy', 'three finite'),
    ('bad_lift_axis', 'nonzero finite axis'),
    ('nonfinite_lift_axis', 'three finite'),
    ('cycle', 'Cycle'),
    ('disconnected', 'Missing URDF TCP chain'),
])
def test_bad_chain_is_rejected_at_load(tmp_path, mutation, match):
    path, _ = planar_model(tmp_path)
    root = ET.parse(path)
    joint = root.find("joint[@name='arm_r_joint3']")
    if mutation == 'missing':
        joint.set('name', 'fixed_spacer'); joint.set('type', 'fixed')
    elif mutation == 'wrong_kind':
        joint.set('type', 'prismatic')
    elif mutation == 'mimic':
        ET.SubElement(joint, 'mimic', joint='arm_r_joint1')
    elif mutation == 'unknown_ancestor':
        root.find("joint[@name='lift_joint']").set('name', 'uncontrolled_base')
    elif mutation == 'bad_axis':
        joint.find('axis').set('xyz', '0 0 0')
    elif mutation == 'bad_xyz':
        joint.find('origin').set('xyz', '0 0')
    elif mutation == 'nonfinite_rpy':
        joint.find('origin').set('rpy', '0 nan 0')
    elif mutation == 'bad_lift_axis':
        root.find("joint[@name='lift_joint']/axis").set('xyz', '0 0 0')
    elif mutation == 'nonfinite_lift_axis':
        root.find("joint[@name='lift_joint']/axis").set('xyz', '0 inf 0')
    elif mutation == 'cycle':
        joint.find('parent').set('link', 'r7')
    else:
        joint.find('parent').set('link', 'missing_link')
    root.write(path)
    with pytest.raises(ValueError, match=match):
        RailKinematics(path, ['arm_right'])
