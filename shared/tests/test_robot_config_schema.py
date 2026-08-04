from pathlib import Path
import sys
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "shared"))

from shared.robot_configs import schema as robot_schema  # noqa: E402


def _assert_urdf_path_and_mesh_assets_resolve(robot_type, expected_urdf_name):
    section = robot_schema.load_robot_section(robot_type)

    urdf_path = Path(robot_schema.get_urdf_path(section))
    assert urdf_path == (
        REPO_ROOT
        / "shared"
        / "shared"
        / "robot_configs"
        / "urdf"
        / expected_urdf_name
    )
    assert urdf_path.exists()

    tree = ET.parse(urdf_path)
    mesh_filenames = {
        mesh.attrib["filename"]
        for mesh in tree.findall(".//mesh")
        if mesh.attrib.get("filename", "").startswith("package://ffw_description/")
    }
    assert mesh_filenames

    asset_root = (
        REPO_ROOT
        / "shared"
        / "shared"
        / "robot_configs"
        / "ffw_description"
    )
    missing = []
    for filename in sorted(mesh_filenames):
        rel_path = filename.removeprefix("package://ffw_description/").replace("//", "/")
        if not (asset_root / rel_path).exists():
            missing.append(rel_path)

    assert missing == []


def test_sh5_config_records_raw_tactile_topics():
    section = robot_schema.load_robot_section("ffw_sh5_rev1")

    tactile_topics = robot_schema.get_tactile_topics(section)
    assert tactile_topics == {
        "left_hand_pressure": {
            "topic": "/left_hand/finger_pressures",
            "msg_type": "robotis_interfaces/msg/HandPressures",
        },
        "right_hand_pressure": {
            "topic": "/right_hand/finger_pressures",
            "msg_type": "robotis_interfaces/msg/HandPressures",
        },
    }

    mcap_topics = robot_schema.get_mcap_record_topics(section)
    assert "/left_hand/finger_pressures" in mcap_topics
    assert "/right_hand/finger_pressures" in mcap_topics
    assert "/zed/zed_node/left/image_rect_color/compressed" not in mcap_topics
    assert "/zed/zed_node/right/image_rect_color/compressed" not in mcap_topics
    assert "/zed/zed_node/left/camera_info" not in mcap_topics
    assert "/zed/zed_node/right/camera_info" not in mcap_topics


def test_sh5_state_and_action_dimensions_match_layout():
    section = robot_schema.load_robot_section("ffw_sh5_rev1")

    state_dim = sum(
        len(cfg["joint_names"])
        for cfg in robot_schema.get_state_groups(section).values()
    )
    action_dim = sum(
        len(cfg["joint_names"])
        for cfg in robot_schema.get_action_groups(section).values()
    )

    assert state_dim == 60
    assert action_dim == 60


def test_sh5_urdf_path_and_mesh_assets_resolve():
    _assert_urdf_path_and_mesh_assets_resolve(
        "ffw_sh5_rev1",
        "ffw_sh5_follower.urdf",
    )


def test_f2_config_uses_realsense_head_camera_layout():
    section = robot_schema.load_robot_section("f2")

    assert robot_schema.get_image_topics(section) == {
        "cam_head": {
            "topic": "/camera_head/camera_head/color/image_rect_raw/compressed",
            "msg_type": "sensor_msgs/msg/CompressedImage",
        },
        "cam_left_wrist": {
            "topic": "/camera_left/camera_left/color/image_rect_raw/compressed",
            "msg_type": "sensor_msgs/msg/CompressedImage",
        },
        "cam_right_wrist": {
            "topic": "/camera_right/camera_right/color/image_rect_raw/compressed",
            "msg_type": "sensor_msgs/msg/CompressedImage",
        },
    }

    camera_info_topics = robot_schema.get_camera_info_topics(section)
    assert camera_info_topics["cam_head"] == "/camera_head/camera_head/color/camera_info"

    extra_topics = robot_schema.get_recording_extra_topics(section)
    assert "/camera_head/camera_head/color/camera_info" in extra_topics
    assert "/zed/zed_node/left/camera_info" not in extra_topics
    assert "/zed/zed_node/right/camera_info" not in extra_topics


def test_f2_state_and_action_dimensions_match_layout():
    section = robot_schema.load_robot_section("f2")

    state_dim = sum(
        len(cfg["joint_names"])
        for cfg in robot_schema.get_state_groups(section).values()
    )
    action_dim = sum(
        len(cfg["joint_names"])
        for cfg in robot_schema.get_action_groups(section).values()
    )

    assert state_dim == 22
    assert action_dim == 22


def test_f2_urdf_path_and_mesh_assets_resolve():
    _assert_urdf_path_and_mesh_assets_resolve(
        "f2",
        "ffw_f2_follower.urdf",
    )


def test_tactile_is_optional_for_existing_sg2_config():
    section = robot_schema.load_robot_section("ffw_sg2_rev1")

    assert robot_schema.get_tactile_topics(section) == {}
    assert "/left_hand/finger_pressures" not in robot_schema.get_mcap_record_topics(section)
    assert "/right_hand/finger_pressures" not in robot_schema.get_mcap_record_topics(section)


def test_malformed_topic_entries_are_ignored():
    section = {
        "observation": {
            "images": {
                "missing_topic": {
                    "msg_type": "sensor_msgs/msg/CompressedImage",
                },
            },
            "state": {
                "missing_topic": {
                    "msg_type": "sensor_msgs/msg/JointState",
                    "joint_names": ["joint_a"],
                },
            },
            "tactile": {
                "missing_topic": {
                    "msg_type": "robotis_interfaces/msg/HandPressures",
                },
            },
        },
        "action": {
            "missing_topic": {
                "msg_type": "trajectory_msgs/msg/JointTrajectory",
                "joint_names": ["joint_a"],
            },
        },
    }

    assert robot_schema.get_image_topics(section) == {}
    assert robot_schema.get_state_groups(section) == {}
    assert robot_schema.get_tactile_topics(section) == {}
    assert robot_schema.get_action_groups(section) == {}
    assert robot_schema.get_joint_state_topics(section) == []
    assert robot_schema.get_action_topics(section) == []
    assert robot_schema.get_action_topic_types(section) == []
