"""Offline CPU integration tests using real datasets, models and the training CLI."""

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from cyclo_lerobot_io.mapping import FILENAME


def fingerprint(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def dataset_root(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "dataset"
    features = {
        key: {"dtype": "float32", "shape": (22,), "names": [f"j{i}" for i in range(22)]}
        for key in ("observation.state", "action")
    }
    features["observation.images.head"] = {
        "dtype": "image",
        "shape": (64, 64, 3),
        "names": ["height", "width", "channels"],
    }
    ds = LeRobotDataset.create(
        "fixture/channels", root=root, fps=10, features=features, use_videos=False, image_writer_threads=1
    )
    for episode in range(4):
        for frame in range(8):
            vector = np.arange(22, dtype=np.float32) + frame / 10 + episode
            ds.add_frame(
                {
                    "observation.state": vector,
                    "action": vector + 0.2,
                    "observation.images.head": np.full((64, 64, 3), frame * 24, dtype=np.uint8),
                    "task": "move",
                }
            )
        ds.save_episode()
    ds.finalize()
    return root


def train_cli(arguments, log_path, distributed=False):
    launch = [sys.executable, "-m"]
    if distributed:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        launch += [
            "torch.distributed.run",
            "--nnodes=1",
            "--nproc_per_node=2",
            "--master_addr=127.0.0.1",
            f"--master_port={port}",
            "-m",
        ]
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "WANDB_MODE": "disabled", "OMP_NUM_THREADS": "1"}
    with log_path.open("w") as log:
        with subprocess.Popen(
            launch + ["lerobot.scripts.lerobot_train", *arguments],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as process:
            try:
                code = process.wait(timeout=240)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
    assert code == 0, log_path.read_text()[-14000:]


@pytest.mark.parametrize("policy", ["act", "diffusion"])
def test_real_training_save_reload_resume(tmp_path, dataset_root, policy):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_train_eval_datasets
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from cyclo_lerobot_io.selection import ChannelSelectionAdapter

    yaml_path = tmp_path / "channels.yaml"
    selection = {
        "state_names": [f"j{i}" for i in range(16)],
        "action_names": [f"j{i}" for i in reversed(range(16))],
    }
    yaml_path.write_text(yaml.safe_dump(selection))
    original = fingerprint(dataset_root)
    output = tmp_path / "output"
    args = [
        f"--policy.type={policy}",
        "--policy.device=cpu",
        "--policy.push_to_hub=false",
        "--policy.pretrained_backbone_weights=null",
        "--dataset.repo_id=fixture/channels",
        f"--dataset.root={dataset_root}",
        "--dataset.eval_split=0.5",
        f"--output_dir={output}",
        "--steps=2",
        "--batch_size=2",
        "--num_workers=0",
        "--env_eval_freq=0",
        "--eval_steps=1",
        "--max_eval_samples=2",
        "--save_freq=1",
        "--log_freq=1",
        "--training_adapter.name=cyclo_channels",
        f"--training_adapter.config_path={yaml_path}",
    ]
    args += (
        [
            "--policy.chunk_size=4",
            "--policy.n_action_steps=2",
            "--policy.dim_model=32",
            "--policy.n_heads=4",
            "--policy.dim_feedforward=64",
            "--policy.n_encoder_layers=1",
            "--policy.n_decoder_layers=1",
            "--policy.use_vae=false",
        ]
        if policy == "act"
        else [
            "--policy.n_obs_steps=2",
            "--policy.horizon=4",
            "--policy.n_action_steps=2",
            "--policy.drop_n_last_frames=0",
            "--policy.down_dims=[16,32]",
            "--policy.n_groups=4",
            "--policy.diffusion_step_embed_dim=16",
            "--policy.num_train_timesteps=4",
            "--policy.num_inference_steps=2",
            "--policy.spatial_softmax_num_keypoints=4",
        ]
    )
    train_cli(args, tmp_path / "train.log")
    checkpoints = sorted(path for path in (output / "checkpoints").iterdir() if not path.is_symlink())
    assert len(checkpoints) == 2
    checkpoint = checkpoints[-1] / "pretrained_model"
    mapping = json.loads((checkpoint / FILENAME).read_text())
    assert mapping["state_names"] == selection["state_names"]
    assert mapping["action_names"] == selection["action_names"]
    assert "eval_loss=" in (tmp_path / "train.log").read_text()
    yaml_path.unlink()
    train_cli(
        [f"--config_path={checkpoint / 'train_config.json'}", "--resume=true", "--steps=3"],
        tmp_path / "resume.log",
    )
    final = (output / "checkpoints" / "last" / "pretrained_model").resolve()
    assert final != checkpoint
    assert json.loads((final / FILENAME).read_text()) == mapping
    assert fingerprint(dataset_root) == original
    # The inference adapter consumes exactly the metadata produced by training.
    from lerobot_engine.channel_mapping import ChannelMapping

    group = {
        "role": "follower",
        "msg_type": "sensor_msgs/msg/JointState",
        "joint_names": [f"j{i}" for i in range(22)],
    }
    robot = SimpleNamespace(
        _config={"joint_groups": {"follower_all": group}},
        _action_groups={
            "arm": {
                "msg_type": "trajectory_msgs/msg/JointTrajectory",
                "joint_names": [f"j{i}" for i in range(16)],
            },
            "head": {
                "msg_type": "trajectory_msgs/msg/JointTrajectory",
                "joint_names": [f"j{i}" for i in range(16, 22)],
            },
        },
    )
    config = PreTrainedConfig.from_pretrained(final)
    layout = ChannelMapping(robot, config, mapping, [])
    assert layout.action_keys == ["arm"]
    np.testing.assert_equal(layout.state((np.arange(16),)), np.arange(16))
    np.testing.assert_equal(layout.action(np.arange(16)[None, ::-1]), np.arange(16)[None, :])
    model = get_policy_class(policy).from_pretrained(final, config=config).eval()
    pre, post = make_pre_post_processors(config, pretrained_path=final)
    train_config = TrainPipelineConfig.from_pretrained(final, cli_args=[])
    train_config.resume = True
    datasets = make_train_eval_datasets(train_config)
    adapter = ChannelSelectionAdapter()
    adapted = adapter.prepare(*datasets, train_config)
    adapter.validate_policy(train_config, pre, post)
    batch = next(iter(torch.utils.data.DataLoader(adapted.train_dataset, batch_size=2)))
    for name in adapted.train_dataset.meta.camera_keys:
        batch[name] = batch[name].float() / 255
    with torch.no_grad():
        actions = post(model.predict_action_chunk(pre(batch)))
    assert actions.shape[-1] == 16 and torch.isfinite(actions).all()
    incompatible = deepcopy(config)
    incompatible.output_features["action"].shape = (3,)
    with pytest.raises(RuntimeError, match="size mismatch"):
        get_policy_class(policy).from_pretrained(final, config=incompatible)


def test_disabled_adapter_and_real_cpu_ddp(tmp_path, dataset_root):
    common = [
        "--policy.type=act",
        "--policy.device=cpu",
        "--policy.push_to_hub=false",
        "--policy.pretrained_backbone_weights=null",
        "--policy.chunk_size=2",
        "--policy.n_action_steps=1",
        "--policy.dim_model=32",
        "--policy.n_heads=4",
        "--policy.dim_feedforward=64",
        "--policy.n_encoder_layers=1",
        "--policy.n_decoder_layers=1",
        "--policy.use_vae=false",
        "--dataset.repo_id=fixture/channels",
        f"--dataset.root={dataset_root}",
        "--steps=1",
        "--batch_size=2",
        "--num_workers=0",
        "--env_eval_freq=0",
        "--save_freq=1",
    ]
    plain = tmp_path / "plain"
    train_cli(common + [f"--output_dir={plain}"], tmp_path / "plain.log")
    assert not list(plain.rglob(FILENAME))
    channels = tmp_path / "channels.yaml"
    channels.write_text("state_names: all\naction_names: [j0, j1, j2]\n")
    ddp = tmp_path / "ddp"
    train_cli(
        common
        + [
            f"--output_dir={ddp}",
            "--training_adapter.name=cyclo_channels",
            f"--training_adapter.config_path={channels}",
        ],
        tmp_path / "ddp.log",
        distributed=True,
    )
    saved = list(ddp.rglob(FILENAME))
    assert len(saved) == 1
    mapping = json.loads(saved[0].read_text())
    assert len(mapping["state_names"]) == 22 and mapping["action_names"] == ["j0", "j1", "j2"]
