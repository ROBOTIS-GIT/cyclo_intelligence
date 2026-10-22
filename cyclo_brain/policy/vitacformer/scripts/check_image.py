#!/usr/bin/env python3
"""Check the built policy with Cyclo-only mounts and no external network.

Run on the Docker host from any directory:
    python3 cyclo_brain/policy/vitacformer/scripts/check_image.py [--gpu]

The temporary container uses its own loopback Zenoh router, empty workspace
and HF cache. It cannot see the host's robot network, devices or ai_worker.
It checks real SDK message registration and both runtime services, without
loading a policy or publishing robot commands. --checkpoint additionally
runs a synthetic forward pass on a locally supplied supported checkpoint.
"""

import argparse
import json
import os
import platform
import subprocess
import time
import uuid
from pathlib import Path


def run(*args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def router():
    import zenoh

    config = zenoh.Config.from_json5('''{
        mode: "router",
        listen: {endpoints: ["tcp/127.0.0.1:7447"]},
        scouting: {multicast: {enabled: false}}
    }''')
    with zenoh.open(config):
        while True:
            time.sleep(1)


def probe(require_cuda=False):
    import torch
    import torchvision
    from main_runtime.zenoh_client import ZenohEngineCommandClient
    from robot_client.messages import (
        INFERENCE_COMMAND_REQUEST_DEF,
        INFERENCE_COMMAND_RESPONSE_DEF,
    )
    from vitacformer_engine.constants import ACTION_KEYS, JOINT_NAMES
    from vitacformer_engine.sensor_client import ViTacFormerSensorClient
    from zenoh_ros2_sdk import ROS2ServiceClient, get_message_class

    if require_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        sample = torch.ones((16, 16), device="cuda")
        if not torch.isfinite(sample @ sample).all().item():
            raise RuntimeError("CUDA matrix multiplication failed")
    message_types = (
        "sensor_msgs/msg/CompressedImage", "sensor_msgs/msg/JointState",
        "trajectory_msgs/msg/JointTrajectory", "geometry_msgs/msg/Twist",
        "robotis_interfaces/msg/HandPressures",
    )
    for message_type in message_types:
        if get_message_class(message_type) is None:
            raise RuntimeError(f"Missing message: {message_type}")
    robot = ViTacFormerSensorClient("ffw_sh5_rev1")
    try:
        if robot.action_dimension(list(ACTION_KEYS)) != len(JOINT_NAMES):
            raise RuntimeError("SH5 action layout mismatch")
        if robot._command_publishers:
            raise RuntimeError("Sensor client unexpectedly enables commands")
    finally:
        robot.close()

    client = ZenohEngineCommandClient(
        service_name="/vitacformer/engine_command", router_ip="127.0.0.1",
        router_port=7447, domain_id=30, node_name="vitacformer_image_check",
    )
    try:
        if not client.ping(timeout_s=5.0):
            raise RuntimeError("Engine ping failed")
    finally:
        client.close()
    client = ROS2ServiceClient(
        service_name="/vitacformer/inference_command",
        srv_type="interfaces/srv/InferenceCommand",
        request_definition=INFERENCE_COMMAND_REQUEST_DEF,
        response_definition=INFERENCE_COMMAND_RESPONSE_DEF,
        timeout=5.0, domain_id=30,
    )
    try:
        response = client.call(
            command=255, model_path="", embodiment_tag="", robot_type="",
            task_instruction="", publish_to_robot=False,
            action_request_mode="async", acceleration_mode="pytorch",
            acceleration_engine_path="", control_hz=0, inference_hz=0,
            chunk_align_window_s=0.0, initial_pose_sync=False,
            initial_pose_sync_duration_s=0.0,
        )
        if response is None or response.success or response.message != "Unknown command: 255":
            raise RuntimeError(f"Unexpected inference-service response: {response}")
    finally:
        client.close()
    print(json.dumps({
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "message_types": message_types, "action_dimension": len(JOINT_NAMES),
        "engine_ping": True, "inference_service": True,
        "external_network": False, "robot_commands": False,
    }), flush=True)


def check(args):
    root = Path(__file__).resolve().parents[4]
    arch = "arm64" if platform.machine() in ("aarch64", "arm64") else "amd64"
    image = args.image or f"robotis/vitacformer-zenoh:1.0.0-{arch}"
    config = json.loads(run(
        "docker", "compose", "-f", str(root / "docker/docker-compose.yml"),
        "config", "--format", "json", capture_output=True,
        env={**os.environ, "ARCH": arch},
    ).stdout)["services"]["vitacformer"]
    if config.get("depends_on"):
        raise RuntimeError("ViTacFormer must not depend on another Compose service")
    name = f"vitacformer-image-check-{uuid.uuid4().hex[:12]}"
    command = [
        "docker", "run", "-d", "--name", name, "--network", "none",
        "--ipc", "private", "--shm-size", "256m", "--entrypoint", "bash",
        "--tmpfs", "/workspace", "--tmpfs", "/root/.cache/huggingface",
        "--env", 'ZENOH_CONFIG_OVERRIDE=transport/shared_memory/enabled=false;scouting/multicast/enabled=false',
        "--mount", f"type=bind,src={Path(__file__).resolve()},dst=/image-check.py,readonly",
    ]
    if args.gpu:
        command += ["--runtime", "nvidia", "--env", "NVIDIA_VISIBLE_DEVICES=all"]
    skipped_targets = {
        "/dev", "/dev/shm", "/workspace", "/root/.cache/huggingface",
        "/etc/timezone", "/etc/localtime",
    }
    for volume in config["volumes"]:
        target = volume["target"]
        if target in skipped_targets:
            continue
        source = Path(volume["source"]).resolve()
        if not source.is_relative_to(root) or not source.is_dir():
            raise RuntimeError(f"Mount must exist inside this checkout: {source}")
        if not volume.get("read_only"):
            raise RuntimeError(f"Source mount must be read-only: {source}")
        command += ["--mount", f"type=bind,src={source},dst={target},readonly"]
    if not (root / "cyclo_brain/sdk/zenoh_ros2_sdk/zenoh_ros2_sdk/__init__.py").is_file():
        raise RuntimeError("Initialize the pinned zenoh_ros2_sdk submodule first")
    if args.checkpoint:
        command += ["--mount", f"type=bind,src={Path(args.checkpoint).resolve(strict=True)},dst=/checkpoint,readonly"]
    command += [image, "-c", "python3 /image-check.py --router & exec /init"]
    created = False
    try:
        run(*command, capture_output=True)
        created = True
        deadline = time.monotonic() + 120
        last_error = ""
        while time.monotonic() < deadline:
            try:
                probe_args = ["--probe"] + (["--gpu"] if args.gpu else [])
                result = run("docker", "exec", name, "python3", "/image-check.py",
                             *probe_args, capture_output=True, timeout=30)
            except subprocess.CalledProcessError as exc:
                last_error = exc.stdout + exc.stderr
            except subprocess.TimeoutExpired:
                raise RuntimeError("Image probe exceeded its 30-second deadline")
            else:
                print(result.stdout, end="", flush=True)
                break
            time.sleep(2)
        else:
            raise RuntimeError(f"Runtime services did not become ready: {last_error}")
        if args.checkpoint:
            run("docker", "exec", name, "python3", "-m",
                "vitacformer_engine.check_checkpoint", "/checkpoint",
                "--device", "cuda" if args.gpu else "cpu", timeout=300)
        info = json.loads(run("docker", "inspect", name, capture_output=True).stdout)[0]
        print(json.dumps({"image": image, "image_id": info["Image"],
                          "network_mode": info["HostConfig"]["NetworkMode"],
                          "mounts": [m["Destination"] for m in info["Mounts"]]}))
    except Exception:
        if created:
            run("docker", "logs", "--tail", "100", name)
        raise
    finally:
        if created:
            run("docker", "rm", "-f", name, capture_output=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--checkpoint", help="Supported run directory for synthetic inference")
    parser.add_argument("--router", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.router:
        router()
    elif options.probe:
        probe(options.gpu)
    else:
        check(options)
