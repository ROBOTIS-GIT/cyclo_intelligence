import numpy as np
import sys
import types


class _StubDependency:
    def __init__(self, *args, **kwargs):
        pass


bag_reader_module = types.ModuleType("cyclo_data.reader.bag_reader")
bag_reader_module.BagReader = _StubDependency
sys.modules.setdefault("cyclo_data.reader.bag_reader", bag_reader_module)

metadata_manager_module = types.ModuleType("cyclo_data.reader.metadata_manager")
metadata_manager_module.MetadataManager = _StubDependency
sys.modules.setdefault("cyclo_data.reader.metadata_manager", metadata_manager_module)

video_metadata_module = types.ModuleType("cyclo_data.reader.video_metadata_extractor")
video_metadata_module.VideoMetadataExtractor = _StubDependency
sys.modules.setdefault(
    "cyclo_data.reader.video_metadata_extractor",
    video_metadata_module,
)

from cyclo_data.converter.base_converter import (
    ConversionConfig,
    EpisodeData,
    RosbagToLerobotConverterBase,
)


def test_feature_names_allow_state_action_dimension_mismatch(tmp_path):
    converter = RosbagToLerobotConverterBase(
        ConversionConfig(repo_id="test", output_dir=tmp_path)
    )
    arm_names = [f"joint_{i}" for i in range(57)]
    mobile_names = ["linear_x", "linear_y", "angular_z"]

    converter._joint_order_by_group = {
        "leader_upper_body": arm_names,
        "leader_mobile": mobile_names,
    }
    converter._action_joint_names = []
    converter._state_joint_names = []

    episode = EpisodeData(
        episode_index=0,
        observation_state=[np.zeros(60, dtype=np.float32)],
        action=[np.zeros(57, dtype=np.float32)],
        observation_state_names=arm_names + mobile_names,
        action_names=arm_names,
    )

    converter._build_features([episode])

    assert converter._features["observation.state"]["shape"] == (60,)
    assert converter._features["observation.state"]["names"] == arm_names + mobile_names
    assert converter._features["action"]["shape"] == (57,)
    assert converter._features["action"]["names"] == arm_names


def test_feature_names_ignore_mismatched_config_fallback(tmp_path):
    converter = RosbagToLerobotConverterBase(
        ConversionConfig(repo_id="test", output_dir=tmp_path)
    )
    parsed_action_names = [f"arm_joint_{i}" for i in range(57)]
    converter._joint_order_by_group = {
        "leader_upper_body": parsed_action_names,
        "leader_mobile": ["linear_x", "linear_y", "angular_z"],
    }

    episode = EpisodeData(
        episode_index=0,
        observation_state=[np.zeros(60, dtype=np.float32)],
        action=[np.zeros(57, dtype=np.float32)],
        action_names=parsed_action_names,
    )

    converter._build_features([episode])

    assert converter._features["action"]["names"] == parsed_action_names
    assert converter._features["action"]["names"] != [f"joint_{i}" for i in range(57)]


def test_feature_names_can_use_later_episode_names(tmp_path):
    converter = RosbagToLerobotConverterBase(
        ConversionConfig(repo_id="test", output_dir=tmp_path)
    )
    action_names = [f"arm_joint_{i}" for i in range(57)]

    episodes = [
        EpisodeData(
            episode_index=0,
            observation_state=[np.zeros(60, dtype=np.float32)],
            action=[np.zeros(57, dtype=np.float32)],
        ),
        EpisodeData(
            episode_index=1,
            observation_state=[np.zeros(60, dtype=np.float32)],
            action=[np.zeros(57, dtype=np.float32)],
            action_names=action_names,
        ),
    ]

    converter._build_features(episodes)

    assert converter._features["action"]["names"] == action_names


def test_feature_names_ignore_none_name_sources(tmp_path):
    converter = RosbagToLerobotConverterBase(
        ConversionConfig(repo_id="test", output_dir=tmp_path)
    )
    action_names = [f"arm_joint_{i}" for i in range(57)]
    converter._state_joint_names = None
    converter._action_joint_names = None

    first = EpisodeData(
        episode_index=0,
        observation_state=[np.zeros(60, dtype=np.float32)],
        action=[np.zeros(57, dtype=np.float32)],
    )
    first.observation_state_names = None
    first.action_names = None
    second = EpisodeData(
        episode_index=1,
        observation_state=[np.zeros(60, dtype=np.float32)],
        action=[np.zeros(57, dtype=np.float32)],
        action_names=action_names,
    )

    converter._build_features([first, second])

    assert converter._features["action"]["names"] == action_names
