import asyncio
import hashlib
import importlib.util
import json
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

APP_PATH = Path(__file__).resolve().with_name("app.py")
REPO_ROOT = APP_PATH.parents[2]

docker_stub = types.ModuleType("docker")
docker_errors_stub = types.ModuleType("docker.errors")


class DockerException(Exception):
    pass


class ImageNotFound(DockerException):
    pass


class NotFound(DockerException):
    pass


docker_stub.from_env = lambda: None
docker_errors_stub.DockerException = DockerException
docker_errors_stub.ImageNotFound = ImageNotFound
docker_errors_stub.NotFound = NotFound
sys.modules["docker"] = docker_stub
sys.modules["docker.errors"] = docker_errors_stub

original_path = list(sys.path)
sys.path = [
    path for path in sys.path
    if Path(path or ".").resolve() != REPO_ROOT
]
try:
    spec = importlib.util.spec_from_file_location("supervisor_api_app", APP_PATH)
    app = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = app
    spec.loader.exec_module(app)
finally:
    sys.path = original_path

_missing_required_mounts = app._missing_required_mounts
_mount_source_for_destination = app._mount_source_for_destination
_backend_container_image_mismatch = app._backend_container_image_mismatch
_backend_container_stale_reason = app._backend_container_stale_reason
_compose_env = app._compose_env
_host_workspace_dir = app._host_workspace_dir
_require_known_service = app._require_known_service
_validate_bt_robot_type = app._validate_bt_robot_type
_validate_robot_type = app._validate_robot_type
_write_bt_robot_type = app._write_bt_robot_type
_resolve_groot_trt_paths = app._resolve_groot_trt_paths
_trt_status = app._trt_status
_BACKENDS = app._BACKENDS
_USER_SERVICES = app._USER_SERVICES
navigation = sys.modules["supervisor_api.navigation"]
navigation_grid_cache = sys.modules["supervisor_api.navigation_grid_cache"]
_GROOT_REQUIRED_MOUNTS = app._REQUIRED_BACKEND_MOUNTS["groot"]
_LEROBOT_REQUIRED_MOUNTS = app._REQUIRED_BACKEND_MOUNTS["lerobot"]


def test_navigation_parses_binary_pgm():
    data = b"P5\n# map\n2 2\n255\n" + bytes([0, 127, 254, 255])

    assert navigation._parse_pgm(data) == (
        2,
        2,
        255,
        [0, 127, 254, 255],
    )


def test_navigation_rejects_map_path_escape():
    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        navigation._resolve_pgm_path("../../outside.pgm")


def test_navigation_validates_map_name():
    import pytest
    from fastapi import HTTPException

    assert navigation._validate_map_name("factory-1") == "factory-1"
    with pytest.raises(HTTPException):
        navigation._validate_map_name("factory; reboot")


def test_navigation_routes_are_registered():
    paths = {route.path for route in app.app.routes if hasattr(route, "path")}

    assert "/navigation/status" in paths
    assert "/navigation/start" in paths
    assert "/navigation/maps/pgm/save" in paths
    assert "/navigation/topics/ws" in paths


def test_flow_sde_ppo_routes_are_registered():
    paths = {route.path for route in app.app.routes if hasattr(route, "path")}

    assert "/flow-sde-ppo/start" in paths
    assert "/flow-sde-ppo/status" in paths
    assert "/flow-sde-ppo/stop" in paths
    assert "/flow-sde-ppo/outcome" in paths
    assert "/flow-sde-ppo/value-warmup/start" in paths
    assert "/flow-sde-ppo/value-warmup/status" in paths
    assert "/flow-sde-ppo/value-warmup/stop" in paths


def test_act_td3_critic_warmup_routes_are_registered():
    paths = {route.path for route in app.app.routes if hasattr(route, "path")}

    assert "/offline-rl/critic-warmup/start" in paths
    assert "/offline-rl/critic-warmup/status" in paths
    assert "/offline-rl/critic-warmup/stop" in paths
    assert "/offline-rl/dataset/episode-data" in paths


def test_running_flow_sde_ppo_blocks_td3_imitation_learning_and_critic_warmup():
    import pytest
    from fastapi import HTTPException

    supervisor = app._FLOW_SDE_PPO_SUPERVISOR
    with supervisor._lock:
        previous = supervisor._job
        supervisor._job = SimpleNamespace(status="running")
    try:
        td3_request = app.OfflineRLStartRequest(
            dataset_path="/not-reached",
            act_checkpoint="/not-reached",
            robot_type="ffw_sg2_rev1",
        )
        with pytest.raises(HTTPException, match="Stop Flow-SDE PPO") as td3_error:
            asyncio.run(app.offline_rl_start(td3_request))
        assert td3_error.value.status_code == 409

        il_request = app.ImitationLearningStartRequest(
            dataset_path="/not-reached",
        )
        with pytest.raises(HTTPException, match="Stop Flow-SDE PPO") as il_error:
            asyncio.run(app.imitation_learning_start(il_request))
        assert il_error.value.status_code == 409

        critic_request = app.ACTTD3CriticWarmupStartRequest(
            dataset_path="/not-reached",
            act_checkpoint="/not-reached",
            robot_type="ffw_sg2_rev1",
        )
        with pytest.raises(HTTPException, match="Stop Flow-SDE PPO") as critic_error:
            asyncio.run(app.act_td3_critic_warmup_start(critic_request))
        assert critic_error.value.status_code == 409
    finally:
        with supervisor._lock:
            supervisor._job = previous


def test_navigation_grid_data_crc32_uses_only_map_data():
    first = {"info": {"width": 2}, "data": [-1, 0, 100, 0]}
    same_data = {"info": {"width": 4}, "data": [-1, 0, 100, 0]}
    changed = {"info": {"width": 2}, "data": [-1, 0, 99, 0]}

    marker = navigation_grid_cache.occupancy_grid_data_crc32(first)
    assert navigation_grid_cache.occupancy_grid_data_crc32(same_data) == marker
    assert navigation_grid_cache.occupancy_grid_data_crc32(changed) != marker


def test_navigation_grid_cache_serializes_only_changed_data():
    cache = navigation_grid_cache.OccupancyGridCache("/map")

    cache.cache_ros_message({"info": {"width": 2}, "data": [0, 1]})
    marker, payload = cache.serialized_if_changed(None)
    assert json.loads(payload) == {
        "available": True,
        "data": {"info": {"width": 2}, "data": [0, 1]},
    }
    assert cache.serialized_if_changed(marker) == (marker, None)

    cache.cache_ros_message({"info": {"width": 99}, "data": [0, 1]})
    metadata_marker, metadata_payload = cache.serialized_if_changed(marker)
    assert metadata_marker != marker
    assert json.loads(metadata_payload)["data"]["info"]["width"] == 99

    cache.cache_ros_message({"info": {"width": 2}, "data": [0, 2]})
    changed_marker, changed_payload = cache.serialized_if_changed(metadata_marker)
    assert changed_marker != metadata_marker
    assert json.loads(changed_payload)["data"]["data"] == [0, 2]


def test_navigation_grid_websocket_sends_cached_original_topic(monkeypatch):
    cache = navigation_grid_cache.OccupancyGridCache("/map")
    cache.cache_ros_message({"info": {"width": 2}, "data": [0, 100]})
    monkeypatch.setitem(navigation_grid_cache.GRID_CACHES, "/map", cache)

    started = []
    monkeypatch.setattr(
        navigation,
        "ensure_ros_grid_subscriber_started",
        lambda: started.append(True),
    )

    class FakeWebSocket:
        def __init__(self):
            self.accepted = False
            self.messages = []

        async def accept(self):
            self.accepted = True

        async def send_text(self, payload):
            self.messages.append(json.loads(payload))

        async def receive(self):
            return {"type": "websocket.disconnect"}

    websocket = FakeWebSocket()
    asyncio.run(asyncio.wait_for(
        navigation.navigation_grid_websocket(websocket, "/map"),
        timeout=1.0,
    ))

    assert websocket.accepted is True
    assert started == [True]
    assert websocket.messages == [{
        "available": True,
        "data": {"info": {"width": 2}, "data": [0, 100]},
    }]


def test_navigation_ros_exec_environment_matches_server(monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "30")
    monkeypatch.setenv("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

    assert navigation._ros_exec_environment() == {
        "ROS_DOMAIN_ID": "30",
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
    }


def test_navigation_goal_passes_ros_environment(monkeypatch):
    captured = {}

    def fake_exec(command, *, environment=None, timeout=None):
        captured["command"] = command
        captured["environment"] = environment
        return 0, "Goal accepted"

    monkeypatch.setattr(navigation, "_exec", fake_exec)
    monkeypatch.setenv("ROS_DOMAIN_ID", "30")
    monkeypatch.setenv("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

    result = navigation.send_goal(
        navigation.NavigateGoalRequest(
            pose={
                "header": {"frame_id": "map"},
                "pose": {
                    "position": {"x": 1.0, "y": 2.0, "z": 0.0},
                    "orientation": {
                        "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0,
                    },
                },
            }
        )
    )

    assert result.ok
    assert captured["command"][:4] == [
        "bash", "--noprofile", "--norc", "-c"
    ]
    assert captured["environment"] == {
        "ROS_DOMAIN_ID": "30",
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
    }


def _container_with_mounts(*destinations):
    return SimpleNamespace(
        attrs={
            "Mounts": [
                {"Destination": destination}
                for destination in destinations
            ]
        }
    )


def test_missing_required_mounts_reports_stale_groot_container():
    container = _container_with_mounts("/legacy_model_mount/groot")

    assert _missing_required_mounts("groot", container) == list(_GROOT_REQUIRED_MOUNTS)


def test_missing_required_mounts_accepts_current_groot_container():
    container = _container_with_mounts(*_GROOT_REQUIRED_MOUNTS)

    assert _missing_required_mounts("groot", container) == []


def test_missing_required_mounts_accepts_current_lerobot_container():
    container = _container_with_mounts(*_LEROBOT_REQUIRED_MOUNTS)

    assert _missing_required_mounts("lerobot", container) == []


def test_backend_container_image_mismatch_detects_old_container_image():
    class FakeImages:
        def get(self, image):
            assert image == "robotis/groot-zenoh:1.3.4-arm64"
            return SimpleNamespace(id="sha256:new")

    container = SimpleNamespace(attrs={"Image": "sha256:old"})
    spec = {"image": "robotis/groot-zenoh:1.3.4-arm64"}

    assert _backend_container_image_mismatch(
        SimpleNamespace(images=FakeImages()),
        container,
        spec,
    )


def test_backend_container_image_mismatch_accepts_current_container_image():
    class FakeImages:
        def get(self, image):
            assert image == "robotis/groot-zenoh:1.3.4-arm64"
            return SimpleNamespace(id="sha256:new")

    container = SimpleNamespace(attrs={"Image": "sha256:new"})
    spec = {"image": "robotis/groot-zenoh:1.3.4-arm64"}

    assert not _backend_container_image_mismatch(
        SimpleNamespace(images=FakeImages()),
        container,
        spec,
    )


def test_backend_container_stale_reason_detects_workspace_mount_mismatch():
    class FakeImages:
        def get(self, image):
            assert image == "robotis/groot-zenoh:1.3.4-arm64"
            return SimpleNamespace(id="sha256:new")

    container = SimpleNamespace(
        attrs={
            "Image": "sha256:new",
            "Mounts": [
                {
                    "Destination": "/workspace",
                    "Source": "/home/robot/old_workspace",
                },
                *[
                    {"Destination": destination}
                    for destination in _GROOT_REQUIRED_MOUNTS
                    if destination != "/workspace"
                ],
            ],
        }
    )
    spec = {"image": "robotis/groot-zenoh:1.3.4-arm64"}

    assert _backend_container_stale_reason(
        "groot",
        SimpleNamespace(images=FakeImages()),
        container,
        spec,
        "/mnt/ssd/cyclo_intelligence/workspace",
    ) == "workspace_mount_mismatch"


def test_backend_container_stale_reason_accepts_repo_symlink_workspace_mount(
    monkeypatch,
    tmp_path,
):
    class FakeImages:
        def get(self, image):
            assert image == "robotis/groot-zenoh:1.3.4-arm64"
            return SimpleNamespace(id="sha256:new")

    host_repo = tmp_path / "host_repo"
    container_repo = tmp_path / "container_repo"
    ssd_workspace = tmp_path / "ssd" / "cyclo_intelligence" / "workspace"
    (host_repo / "docker").mkdir(parents=True)
    (container_repo / "docker").mkdir(parents=True)
    ssd_workspace.mkdir(parents=True)
    (container_repo / "docker" / "workspace").symlink_to(ssd_workspace)

    monkeypatch.setattr(app, "_HOST_PROJECT_DIR_CACHE", str(host_repo / "docker"))
    monkeypatch.setattr(app, "_CYCLO_REPO_MOUNT", str(container_repo))

    container = SimpleNamespace(
        attrs={
            "Image": "sha256:new",
            "Mounts": [
                {
                    "Destination": "/workspace",
                    "Source": str(host_repo / "docker" / "workspace"),
                },
                *[
                    {"Destination": destination}
                    for destination in _GROOT_REQUIRED_MOUNTS
                    if destination != "/workspace"
                ],
            ],
        }
    )
    spec = {"image": "robotis/groot-zenoh:1.3.4-arm64"}

    assert _backend_container_stale_reason(
        "groot",
        SimpleNamespace(images=FakeImages()),
        container,
        spec,
        str(ssd_workspace),
    ) is None


def test_mount_source_for_destination_resolves_workspace_host_path():
    mounts = [
        {"Destination": "/root/ros2_ws/src/cyclo_intelligence", "Source": "/repo"},
        {"Destination": "/workspace", "Source": "/mnt/ssd/cyclo_intelligence/workspace"},
    ]

    assert _mount_source_for_destination(mounts, "/workspace") == (
        "/mnt/ssd/cyclo_intelligence/workspace"
    )


def test_host_workspace_dir_prefers_actual_mount_over_legacy_env(monkeypatch):
    container = SimpleNamespace(
        attrs={
            "Mounts": [
                {
                    "Destination": "/workspace",
                    "Source": "/repo/docker/workspace",
                }
            ]
        }
    )
    client = SimpleNamespace(
        containers=SimpleNamespace(get=lambda _name: container)
    )

    monkeypatch.setenv("HOSTNAME", "self")
    monkeypatch.setenv(
        "CYCLO_WORKSPACE_DIR",
        "/mnt/ssd/cyclo_intelligence/workspace",
    )
    monkeypatch.setattr(app, "_docker_client", lambda: client)
    app._HOST_WORKSPACE_DIR_CACHE = None
    try:
        assert _host_workspace_dir() == "/repo/docker/workspace"
    finally:
        app._HOST_WORKSPACE_DIR_CACHE = None


def test_resolve_groot_trt_paths_defaults_engine_inside_model():
    model, engine = _resolve_groot_trt_paths(
        "/workspace/model/groot/example",
        "",
    )

    assert model == "/workspace/model/groot/example"
    assert engine == "/workspace/model/groot/example/dit_model_bf16.trt"


def test_trt_status_reports_ready_engine(tmp_path):
    model = tmp_path / "workspace" / "model" / "groot" / "example"
    model.mkdir(parents=True)
    engine = model / "dit_model_bf16.trt"
    engine.write_bytes(b"engine")

    status = _trt_status(str(model), str(engine))

    assert status.status == "ready"
    assert status.engine_size_bytes == len(b"engine")


def test_trt_status_reports_missing_engine(tmp_path):
    model = tmp_path / "workspace" / "model" / "groot" / "example"
    model.mkdir(parents=True)
    engine = model / "dit_model_bf16.trt"

    status = _trt_status(str(model), str(engine))

    assert status.status == "missing"


def test_trt_status_reports_stale_oom_build_from_log(tmp_path):
    model = tmp_path / "workspace" / "model" / "groot" / "example"
    model.mkdir(parents=True)
    engine = model / "dit_model_bf16.trt"
    (model / "dit_model_bf16.trt.json").write_text(
        '{"status": "building", "started_at": 1.0, "updated_at": 2.0}'
    )
    (model / "dit_model_bf16.trt.build.log").write_text(
        "=== TensorRT build exited rc=137 at 2026-06-19 06:29:02 ===\n"
    )

    status = _trt_status(str(model), str(engine))

    assert status.status == "failed"
    assert status.returncode == 137
    assert "out-of-memory" in status.message


def test_compose_uses_repo_local_workspace_mounts():
    compose = (REPO_ROOT / "docker" / "docker-compose.yml").read_text()

    assert "CYCLO_WORKSPACE_DIR" not in compose
    assert "CYCLO_HUGGINGFACE_DIR" not in compose
    assert compose.count("./workspace:/workspace") == 3
    assert compose.count("./huggingface:/root/.cache/huggingface") == 3


def test_container_helper_does_not_export_workspace_mount_overrides():
    helper = (REPO_ROOT / "docker" / "container.sh").read_text()

    assert "export CYCLO_WORKSPACE_DIR" not in helper
    assert "export CYCLO_HUGGINGFACE_DIR" not in helper
    assert "CYCLO_SSD_ROOT" not in helper
    assert "CYCLO_STORAGE_MODE" not in helper
    assert "setup_storage" not in helper
    assert "prepare_host_mounts" in helper
    assert "rsync " not in helper
    assert "rsync -aHP" not in helper


def test_bt_node_is_known_user_service():
    _require_known_service("bt_node")


def test_bt_node_robot_type_file_is_written(monkeypatch, tmp_path):
    target = tmp_path / "bt_node_robot_type"
    monkeypatch.setattr(app, "_BT_ROBOT_TYPE_FILE", str(target))

    _write_bt_robot_type("ffw_sg2_rev1")

    assert target.read_text() == "ffw_sg2_rev1\n"


def test_bt_node_robot_type_defaults_to_sg2():
    assert _validate_bt_robot_type("") == "ffw_sg2_rev1"


def test_bt_node_robot_type_rejects_other_robots():
    try:
        _validate_bt_robot_type("omy_f3m")
    except app.HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("bt_node should reject unsupported robot types")


def test_bt_node_start_defaults_to_sg2(monkeypatch, tmp_path):
    target = tmp_path / "bt_node_robot_type"
    calls = []

    async def fake_run(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(rc=0, stdout="started", stderr="")

    monkeypatch.setattr(app, "_BT_ROBOT_TYPE_FILE", str(target))
    monkeypatch.setattr(app, "_run", fake_run)

    result = asyncio.run(app.service_start("bt_node"))

    assert result.ok is True
    assert target.read_text() == "ffw_sg2_rev1\n"
    assert calls == [("s6-rc", "-u", "change", "bt_node")]


def test_bt_node_start_rejects_other_robots(monkeypatch, tmp_path):
    target = tmp_path / "bt_node_robot_type"
    calls = []

    async def fake_run(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(rc=0, stdout="started", stderr="")

    monkeypatch.setattr(app, "_BT_ROBOT_TYPE_FILE", str(target))
    monkeypatch.setattr(app, "_run", fake_run)

    try:
        asyncio.run(app.service_start(
            "bt_node",
            app.ServiceActionRequest(robot_type="omy_f3m"),
        ))
    except app.HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("bt_node should reject unsupported robot types")

    assert not target.exists()
    assert calls == []


def test_robot_type_validation_rejects_shell_metacharacters():
    try:
        _validate_robot_type("omy_f3m;echo bad")
    except app.HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("invalid robot_type should be rejected")


def test_unknown_user_service_is_rejected():
    try:
        _require_known_service("not_a_service")
    except app.HTTPException as exc:
        assert exc.status_code == 404
    else:
        raise AssertionError("unknown service should be rejected")


def test_zenoh_router_is_not_user_managed_service():
    assert "zenoh_router" not in _USER_SERVICES


def test_groot_backend_uses_current_release_image():
    assert (
        _BACKENDS["groot"]["image"]
        == f"robotis/groot-zenoh:1.3.4-{app._BACKEND_ARCH}"
    )


def test_backend_status_model_exposes_stale_image_status():
    status = app.BackendStatus(
        name="groot",
        image="robotis/groot-zenoh:1.3.4-arm64",
        image_pulled=True,
        image_status="stale",
        container_state="exited",
        raw_state="stale_image",
    )

    assert status.image_status == "stale"


def test_host_project_dir_falls_back_to_compose_container_name(monkeypatch):
    class FakeContainers:
        def __init__(self):
            self.requested = []

        def get(self, name):
            self.requested.append(name)
            if name == "cyclo_intelligence":
                return SimpleNamespace(
                    attrs={
                        "Mounts": [
                            {
                                "Destination": app._CYCLO_REPO_MOUNT,
                                "Source": "/home/rc/workspace/cyclo_intelligence",
                            }
                        ]
                    }
                )
            raise NotFound(name)

    fake_containers = FakeContainers()
    fake_client = SimpleNamespace(containers=fake_containers)

    monkeypatch.setenv("HOSTNAME", "ubuntu")
    monkeypatch.setattr(app, "_docker_client", lambda: fake_client)
    app._HOST_PROJECT_DIR_CACHE = None

    try:
        assert (
            app._host_project_dir()
            == "/home/rc/workspace/cyclo_intelligence/docker"
        )
        assert fake_containers.requested == ["ubuntu", "cyclo_intelligence"]
    finally:
        app._HOST_PROJECT_DIR_CACHE = None


def test_compose_env_uses_current_container_mounts(monkeypatch):
    class FakeContainers:
        def __init__(self):
            self.requested = []

        def get(self, name):
            self.requested.append(name)
            if name != "cyclo_intelligence":
                raise NotFound(name)
            return SimpleNamespace(
                attrs={
                    "Mounts": [
                        {
                            "Destination": "/workspace",
                            "Source": "/mnt/ssd/cyclo_intelligence/workspace",
                        },
                        {
                            "Destination": "/root/.cache/huggingface",
                            "Source": "/mnt/ssd/cyclo_intelligence/huggingface",
                        },
                    ]
                }
            )

    fake_containers = FakeContainers()
    fake_client = SimpleNamespace(containers=fake_containers)

    monkeypatch.setenv("HOSTNAME", "container-id")
    monkeypatch.delenv("CYCLO_WORKSPACE_DIR", raising=False)
    monkeypatch.delenv("CYCLO_HUGGINGFACE_DIR", raising=False)
    monkeypatch.setattr(app, "_docker_client", lambda: fake_client)
    app._HOST_WORKSPACE_DIR_CACHE = None
    app._HOST_HUGGINGFACE_DIR_CACHE = None

    try:
        env = _compose_env()
        assert (
            env["CYCLO_WORKSPACE_DIR"]
            == "/mnt/ssd/cyclo_intelligence/workspace"
        )
        assert (
            env["CYCLO_HUGGINGFACE_DIR"]
            == "/mnt/ssd/cyclo_intelligence/huggingface"
        )
        assert env["ARCH"] == app._BACKEND_ARCH
        assert fake_containers.requested == [
            "container-id",
            "cyclo_intelligence",
            "container-id",
            "cyclo_intelligence",
        ]
    finally:
        app._HOST_WORKSPACE_DIR_CACHE = None
        app._HOST_HUGGINGFACE_DIR_CACHE = None


def _offline_rl_test_layout(monkeypatch, tmp_path, episodes=50):
    dataset_root = tmp_path / "workspace" / "lerobot"
    model_root = tmp_path / "workspace" / "model" / "lerobot"
    dataset = dataset_root / "recording_v30"
    act = model_root / "base_act" / "pretrained_model"
    (dataset / "meta").mkdir(parents=True)
    act.mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0",
        "total_episodes": episodes,
        "total_frames": episodes * 10,
        "fps": 15,
        "features": {"episode_success": {"dtype": "bool"}},
    }))
    (act / "config.json").write_text('{"type":"act"}')
    (act / "model.safetensors").write_bytes(b"weights")
    (act / "policy_preprocessor.json").write_text("{}")
    (act / "policy_postprocessor.json").write_text("{}")
    monkeypatch.setattr(app, "_OFFLINE_RL_DATASET_ROOT", dataset_root)
    monkeypatch.setattr(app, "_OFFLINE_RL_MODEL_ROOT", model_root)
    monkeypatch.setattr(app, "_OFFLINE_RL_OUTPUT_ROOT", model_root / "offline_rl")
    monkeypatch.setattr(app, "_OFFLINE_RL_LOG_ROOT", tmp_path / "logs")
    monkeypatch.setattr(
        app,
        "_IMITATION_LEARNING_OUTPUT_ROOT",
        model_root / "imitation_learning",
    )
    monkeypatch.setattr(
        app,
        "_IMITATION_LEARNING_LOG_ROOT",
        tmp_path / "imitation_logs",
    )
    return dataset, act, model_root


def _offline_rl_summary_rows(count):
    return [
        app.OfflineRLDatasetEpisode(
            index=index,
            frames=10,
            outcome=(
                "failure" if index == count - 1 else "success"
            ),
            tasks=["pick jelly"],
        )
        for index in range(count)
    ]


def test_offline_rl_dataset_summary_reports_episode_labels(monkeypatch, tmp_path):
    dataset, _act, _model_root = _offline_rl_test_layout(
        monkeypatch,
        tmp_path,
        episodes=3,
    )
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.images.head"] = {"dtype": "video"}
    info_path.write_text(json.dumps(info))
    monkeypatch.setattr(
        app,
        "_offline_rl_v3_episode_rows",
        lambda _dataset, count: _offline_rl_summary_rows(count),
    )

    summary = app._offline_rl_dataset_summary(str(dataset))

    assert summary.version == "v3.0"
    assert summary.fps == 15.0
    assert summary.total_episodes == 3
    assert summary.camera_count == 1
    assert summary.success_count == 2
    assert summary.failure_count == 1
    assert summary.unlabeled_count == 0
    assert summary.success_rate == 66.67
    assert [episode.index for episode in summary.episodes] == [0, 1, 2]


def test_offline_rl_dataset_summary_rejects_non_finite_fps(monkeypatch, tmp_path):
    import pytest

    dataset, _act, _model_root = _offline_rl_test_layout(
        monkeypatch,
        tmp_path,
        episodes=1,
    )
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    monkeypatch.setattr(
        app,
        "_offline_rl_v3_episode_rows",
        lambda _dataset, _count: pytest.fail("episode rows must not be read"),
    )

    for fps in (float("nan"), float("inf"), float("-inf")):
        info["fps"] = fps
        info_path.write_text(json.dumps(info))
        with pytest.raises(app.HTTPException) as error:
            app._offline_rl_dataset_summary(str(dataset))
        assert error.value.status_code == 400
        assert "finite and positive" in error.value.detail


def test_offline_rl_episode_rows_reject_non_finite_success_stats(
    monkeypatch,
    tmp_path,
):
    import pytest

    dataset, _act, _model_root = _offline_rl_test_layout(
        monkeypatch,
        tmp_path,
        episodes=1,
    )
    v3_metadata = dataset / "meta" / "episodes" / "chunk-000"
    v3_metadata.mkdir(parents=True)
    v3_file = v3_metadata / "file-000.parquet"
    v3_file.write_bytes(b"metadata")
    current_success = [0.0]

    parquet_stub = types.ModuleType("pyarrow.parquet")
    parquet_stub.ParquetFile = lambda _path: SimpleNamespace(
        schema_arrow=SimpleNamespace(names=[
            "episode_index",
            "length",
            "stats/episode_success/mean",
        ])
    )
    parquet_stub.read_table = lambda _path, columns: SimpleNamespace(
        to_pylist=lambda: [{
            "episode_index": 0,
            "length": 10,
            "stats/episode_success/mean": current_success[0],
        }]
    )
    pyarrow_stub = types.ModuleType("pyarrow")
    pyarrow_stub.__path__ = []
    pyarrow_stub.parquet = parquet_stub
    monkeypatch.setitem(sys.modules, "pyarrow", pyarrow_stub)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet_stub)

    (dataset / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 10}) + "\n"
    )
    v21_stats = dataset / "meta" / "episodes_stats.jsonl"

    for value in (float("nan"), float("inf"), float("-inf")):
        current_success[0] = value
        with pytest.raises(app.HTTPException) as error:
            app._offline_rl_v3_episode_rows(dataset.resolve(), 1)
        assert error.value.status_code == 400
        assert "Invalid episode_success" in error.value.detail

        v21_stats.write_text(json.dumps({
            "episode_index": 0,
            "stats": {"episode_success": {"mean": [value]}},
        }) + "\n")
        with pytest.raises(app.HTTPException) as error:
            app._offline_rl_v21_episode_rows(dataset.resolve(), 1)
        assert error.value.status_code == 400
        assert "Invalid episode_success" in error.value.detail


def test_offline_rl_dataset_summary_reads_v21_episode_cards(monkeypatch, tmp_path):
    dataset, _act, _model_root = _offline_rl_test_layout(monkeypatch, tmp_path)
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["codebase_version"] = "v2.1"
    info["total_episodes"] = 2
    info["total_frames"] = 20
    info["features"]["observation.images.head"] = {"dtype": "video"}
    info_path.write_text(json.dumps(info))
    (dataset / "meta" / "episodes.jsonl").write_text(
        "\n".join([
            json.dumps({
                "episode_index": index,
                "length": 10,
                "tasks": ["pick jelly"],
            })
            for index in range(2)
        ]) + "\n"
    )
    (dataset / "meta" / "episodes_stats.jsonl").write_text(
        "\n".join([
            json.dumps({
                "episode_index": 0,
                "stats": {"episode_success": {"mean": [1.0]}},
            }),
            json.dumps({
                "episode_index": 1,
                "stats": {"episode_success": {"mean": [0.0]}},
            }),
        ]) + "\n"
    )

    summary = app._offline_rl_dataset_summary(str(dataset))

    assert summary.version == "v2.1"
    assert summary.camera_count == 1
    assert summary.success_count == 1
    assert summary.failure_count == 1
    assert summary.success_rate == 50.0
    assert [episode.frames for episode in summary.episodes] == [10, 10]
    assert [episode.tasks for episode in summary.episodes] == [
        ["pick jelly"],
        ["pick jelly"],
    ]


def test_offline_rl_dataset_summary_describes_v21_episode_media(
    monkeypatch,
    tmp_path,
):
    dataset, _act, _model_root = _offline_rl_test_layout(
        monkeypatch,
        tmp_path,
        episodes=1,
    )
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info.update({
        "codebase_version": "v2.1",
        "total_episodes": 1,
        "total_frames": 10,
        "chunks_size": 1000,
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "observation.images.rgb.cam_left_wrist": {"dtype": "video"},
            "observation.images.rgb.cam_left_head": {"dtype": "video"},
            "observation.images.rgb.cam_right_wrist": {"dtype": "video"},
        },
    })
    info_path.write_text(json.dumps(info))
    (dataset / "meta" / "episodes.jsonl").write_text(json.dumps({
        "episode_index": 0,
        "length": 10,
        "tasks": ["pick jelly"],
    }) + "\n")
    for camera_key in info["features"]:
        video = dataset / "videos" / "chunk-000" / camera_key
        video.mkdir(parents=True)
        (video / "episode_000000.mp4").write_bytes(b"video")

    summary = app._offline_rl_dataset_summary(str(dataset))

    media = summary.episodes[0].media
    assert [entry.camera_key for entry in media] == sorted(info["features"])
    assert all(not Path(entry.relative_path).is_absolute() for entry in media)
    assert [entry.relative_path for entry in media] == [
        f"videos/chunk-000/{camera_key}/episode_000000.mp4"
        for camera_key in sorted(info["features"])
    ]
    assert all(entry.from_s is None and entry.to_s is None for entry in media)


def test_offline_rl_v3_episode_media_uses_shared_video_timestamps(
    monkeypatch,
    tmp_path,
):
    dataset, _act, _model_root = _offline_rl_test_layout(
        monkeypatch,
        tmp_path,
        episodes=1,
    )
    camera_keys = [
        "observation.images.rgb.cam_left_wrist",
        "observation.images.rgb.cam_left_head",
        "observation.images.rgb.cam_right_wrist",
    ]
    info = {
        "video_path": (
            "videos/{video_key}/chunk-{chunk_index:03d}/"
            "file-{file_index:03d}.mp4"
        ),
        "features": {key: {"dtype": "video"} for key in camera_keys},
    }
    metadata = dataset / "meta" / "episodes" / "chunk-000"
    metadata.mkdir(parents=True)
    metadata_file = metadata / "file-000.parquet"
    metadata_file.write_bytes(b"metadata")
    schema_names = ["episode_index"]
    record = {"episode_index": 0}
    for camera_key in camera_keys:
        prefix = f"videos/{camera_key}"
        schema_names.extend([
            f"{prefix}/chunk_index",
            f"{prefix}/file_index",
            f"{prefix}/from_timestamp",
            f"{prefix}/to_timestamp",
        ])
        record.update({
            f"{prefix}/chunk_index": 0,
            f"{prefix}/file_index": 0,
            f"{prefix}/from_timestamp": 12.5,
            f"{prefix}/to_timestamp": 18.75,
        })
        video = dataset / "videos" / camera_key / "chunk-000"
        video.mkdir(parents=True)
        (video / "file-000.mp4").write_bytes(b"shared-video")

    parquet_stub = types.ModuleType("pyarrow.parquet")
    parquet_stub.ParquetFile = lambda _path: SimpleNamespace(
        schema_arrow=SimpleNamespace(names=schema_names)
    )
    parquet_stub.read_table = lambda _path, columns: SimpleNamespace(
        to_pylist=lambda: [{name: record[name] for name in columns}]
    )
    pyarrow_stub = types.ModuleType("pyarrow")
    pyarrow_stub.__path__ = []
    pyarrow_stub.parquet = parquet_stub
    monkeypatch.setitem(sys.modules, "pyarrow", pyarrow_stub)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet_stub)
    episodes = [app.OfflineRLDatasetEpisode(
        index=0,
        frames=94,
        outcome="success",
    )]

    media = app._offline_rl_v3_episode_media(
        dataset.resolve(),
        info,
        episodes,
    )[0]

    assert [entry.camera_key for entry in media] == sorted(camera_keys)
    assert all(entry.from_s == 12.5 and entry.to_s == 18.75 for entry in media)
    assert [entry.relative_path for entry in media] == [
        f"videos/{camera_key}/chunk-000/file-000.mp4"
        for camera_key in sorted(camera_keys)
    ]


def test_offline_rl_episode_media_rejects_path_escape(monkeypatch, tmp_path):
    dataset, _act, _model_root = _offline_rl_test_layout(
        monkeypatch,
        tmp_path,
        episodes=1,
    )

    with pytest.raises(app.HTTPException) as error:
        app._offline_rl_v3_video_path(
            dataset.resolve(),
            {"video_path": "../../outside.mp4"},
            video_key="observation.images.head",
            chunk_index=0,
            file_index=0,
        )

    assert error.value.status_code == 400
    assert "Unsafe" in error.value.detail


def _offline_rl_episode_data_info(dataset, *, version, episodes, frames):
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info.update({
        "codebase_version": version,
        "total_episodes": episodes,
        "total_frames": frames,
        "fps": 15,
        "data_path": (
            "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
            if version == "v3.0"
            else "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        ),
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [2],
                "names": ["joint_a", "joint_b"],
            },
            "action": {
                "dtype": "float32",
                "shape": [2],
                "names": ["joint_a", "joint_b"],
            },
        },
    })
    info_path.write_text(json.dumps(info))
    return info


def _offline_rl_write_episode_frames(path, episode_indices, timestamps, offset=0.0):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    state = [
        [offset + row_index, offset + row_index + 0.25]
        for row_index in range(len(episode_indices))
    ]
    action = [
        [offset + row_index + 0.5, offset + row_index + 0.75]
        for row_index in range(len(episode_indices))
    ]
    pq.write_table(pa.table({
        "episode_index": episode_indices,
        "timestamp": timestamps,
        "observation.state": state,
        "action": action,
    }), path)


def test_offline_rl_v3_episode_data_reads_only_selected_shard_slice(
    monkeypatch,
    tmp_path,
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    dataset, _act, _model_root = _offline_rl_test_layout(
        monkeypatch,
        tmp_path,
        episodes=3,
    )
    _offline_rl_episode_data_info(
        dataset,
        version="v3.0",
        episodes=3,
        frames=7,
    )
    metadata_path = dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    metadata_path.parent.mkdir(parents=True)
    pq.write_table(pa.table({
        "episode_index": [0, 1, 2],
        "length": [2, 3, 2],
        "data/chunk_index": [0, 0, 0],
        "data/file_index": [0, 0, 1],
        "dataset_from_index": [0, 2, 5],
        "dataset_to_index": [2, 5, 7],
    }), metadata_path)
    _offline_rl_write_episode_frames(
        dataset / "data" / "chunk-000" / "file-000.parquet",
        [0, 0, 1, 1, 1],
        [0.0, 0.1, 5.0, 5.1, 5.2],
    )
    _offline_rl_write_episode_frames(
        dataset / "data" / "chunk-000" / "file-001.parquet",
        [2, 2],
        [9.0, 9.1],
        offset=10.0,
    )

    episode_one = app._offline_rl_dataset_episode_data(str(dataset), 1)
    episode_two = app._offline_rl_dataset_episode_data(str(dataset), 2)

    assert episode_one.joint_names == ["joint_a", "joint_b"]
    assert episode_one.action_names == ["joint_a", "joint_b"]
    assert episode_one.joint_timestamps == pytest.approx([0.0, 0.1, 0.2])
    assert episode_one.action_timestamps == pytest.approx([0.0, 0.1, 0.2])
    assert episode_one.joint_positions == pytest.approx([
        2.0, 2.25,
        3.0, 3.25,
        4.0, 4.25,
    ])
    assert episode_one.action_values == pytest.approx([
        2.5, 2.75,
        3.5, 3.75,
        4.5, 4.75,
    ])
    assert episode_one.duration == pytest.approx(0.2)
    assert episode_two.joint_positions == pytest.approx([
        10.0, 10.25,
        11.0, 11.25,
    ])
    assert episode_two.joint_timestamps == pytest.approx([0.0, 0.1])


def test_offline_rl_v21_episode_data_uses_canonical_episode_file(
    monkeypatch,
    tmp_path,
):
    dataset, _act, _model_root = _offline_rl_test_layout(
        monkeypatch,
        tmp_path,
        episodes=1,
    )
    _offline_rl_episode_data_info(
        dataset,
        version="v2.1",
        episodes=1,
        frames=2,
    )
    (dataset / "meta" / "episodes.jsonl").write_text(json.dumps({
        "episode_index": 0,
        "length": 2,
    }) + "\n")
    _offline_rl_write_episode_frames(
        dataset / "data" / "chunk-000" / "episode_000000.parquet",
        [0, 0],
        [12.0, 12.25],
    )

    result = app._offline_rl_dataset_episode_data(str(dataset), 0)

    assert result.joint_timestamps == pytest.approx([0.0, 0.25])
    assert result.action_timestamps == pytest.approx([0.0, 0.25])
    assert result.joint_positions == pytest.approx([0.0, 0.25, 1.0, 1.25])
    assert result.action_values == pytest.approx([0.5, 0.75, 1.5, 1.75])
    assert result.duration == pytest.approx(0.25)


def test_offline_rl_episode_data_rejects_bounds_shape_and_path_escape(
    monkeypatch,
    tmp_path,
):
    dataset, _act, _model_root = _offline_rl_test_layout(
        monkeypatch,
        tmp_path,
        episodes=1,
    )
    info = _offline_rl_episode_data_info(
        dataset,
        version="v2.1",
        episodes=1,
        frames=1,
    )

    with pytest.raises(app.HTTPException) as bounds_error:
        app._offline_rl_dataset_episode_data(str(dataset), 1)
    assert bounds_error.value.status_code == 404

    info["features"]["observation.state"]["shape"] = [3]
    (dataset / "meta" / "info.json").write_text(json.dumps(info))
    with pytest.raises(app.HTTPException, match="shape/names") as shape_error:
        app._offline_rl_dataset_episode_data(str(dataset), 0)
    assert shape_error.value.status_code == 400

    info["features"]["observation.state"]["shape"] = [2]
    info["data_path"] = "../../outside.parquet"
    (dataset / "meta" / "info.json").write_text(json.dumps(info))
    (dataset / "meta" / "episodes.jsonl").write_text(json.dumps({
        "episode_index": 0,
        "length": 1,
    }) + "\n")
    with pytest.raises(app.HTTPException, match="Unsafe") as path_error:
        app._offline_rl_dataset_episode_data(str(dataset), 0)
    assert path_error.value.status_code == 400


def test_offline_rl_episode_data_rejects_nonfinite_values_and_slice_leaks(
    monkeypatch,
    tmp_path,
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    dataset, _act, _model_root = _offline_rl_test_layout(
        monkeypatch,
        tmp_path,
        episodes=1,
    )
    _offline_rl_episode_data_info(
        dataset,
        version="v3.0",
        episodes=1,
        frames=1,
    )
    metadata_path = dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    metadata_path.parent.mkdir(parents=True)
    pq.write_table(pa.table({
        "episode_index": [0],
        "length": [1],
        "data/chunk_index": [0],
        "data/file_index": [0],
        "dataset_from_index": [0],
        "dataset_to_index": [1],
    }), metadata_path)
    data_path = dataset / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True)
    pq.write_table(pa.table({
        "episode_index": [0],
        "timestamp": [0.0],
        "observation.state": [[0.0, 1.0]],
        "action": [[float("nan"), 1.0]],
    }), data_path)

    with pytest.raises(app.HTTPException, match="Invalid action value") as finite_error:
        app._offline_rl_dataset_episode_data(str(dataset), 0)
    assert finite_error.value.status_code == 400

    pq.write_table(pa.table({
        "episode_index": [9],
        "timestamp": [0.0],
        "observation.state": [[0.0, 1.0]],
        "action": [[0.0, 1.0]],
    }), data_path)
    with pytest.raises(app.HTTPException, match="frame slice leaked") as leak_error:
        app._offline_rl_dataset_episode_data(str(dataset), 0)
    assert leak_error.value.status_code == 400


def test_offline_rl_dataset_inventory_discovers_nested_datasets(monkeypatch, tmp_path):
    dataset_root = tmp_path / "workspace" / "lerobot"
    older = dataset_root / "older_lerobot_v30"
    newer = dataset_root / "RLTEST" / "newer_lerobot_v21"
    for dataset in (older, newer):
        (dataset / "meta").mkdir(parents=True)
        (dataset / "meta" / "info.json").write_text("{}")
    older_time = 1000
    newer_time = 2000
    import os
    os.utime(older / "meta" / "info.json", (older_time, older_time))
    os.utime(newer / "meta" / "info.json", (newer_time, newer_time))
    monkeypatch.setattr(app, "_OFFLINE_RL_DATASET_ROOT", dataset_root)

    def summary(path):
        dataset = Path(path)
        version = "v2.1" if dataset.name.endswith("v21") else "v3.0"
        return app.OfflineRLDatasetSummary(
            dataset_path=str(dataset),
            name=dataset.name,
            version=version,
            fps=15,
            total_episodes=1,
            total_frames=10,
            camera_count=3,
            success_count=1,
            failure_count=0,
            unlabeled_count=0,
            success_rate=100,
            episodes=[],
        )

    monkeypatch.setattr(app, "_offline_rl_dataset_summary", summary)

    inventory = app._offline_rl_dataset_inventory(str(dataset_root))

    assert inventory.root_path == str(dataset_root)
    assert [item.dataset_path for item in inventory.datasets] == [
        str(newer),
        str(older),
    ]

    endpoint_inventory = asyncio.run(app.offline_rl_datasets(str(dataset_root)))
    assert [item.dataset_path for item in endpoint_inventory.datasets] == [
        str(newer),
        str(older),
    ]


def test_offline_rl_data_epoch_reservation_is_monotonic_and_writes_provenance(
    monkeypatch,
    tmp_path,
):
    dataset_root = tmp_path / "workspace" / "lerobot"
    destination = dataset_root / "RLTEST"
    rosbag_root = tmp_path / "workspace" / "rosbag2"
    source = rosbag_root / "Task_failure_inference_MCAP"
    destination.mkdir(parents=True)
    for index, outcome in enumerate((True, False, None)):
        episode = source / str(index)
        episode.mkdir(parents=True)
        metadata = {"episode_index": index}
        if outcome is not None:
            metadata["episode_success"] = outcome
        (episode / "episode_info.json").write_text(json.dumps(metadata))

    monkeypatch.setattr(app, "_OFFLINE_RL_DATASET_ROOT", dataset_root)
    monkeypatch.setattr(app, "_OFFLINE_RL_ROSBAG_ROOT", rosbag_root)
    monkeypatch.setattr(
        app,
        "_OFFLINE_RL_DATASET_LOCK_PATH",
        tmp_path / "workspace" / ".cyclo_dataset.lock",
    )
    request = app.OfflineRLDataEpochReserveRequest(
        destination_root=str(destination),
        source_mcap=str(source),
        behavior_policy_path="/workspace/model/lerobot/base/pretrained_model",
        boundary_reason="policy_update",
        fps=15,
        formats=["v3.0"],
    )

    first = app._offline_rl_reserve_data_epoch(request)
    second = asyncio.run(app.offline_rl_reserve_data_epoch(request))

    assert first.data_epoch == 0
    assert first.epoch_name == "data_epoch_0000"
    assert first.output_root == str(destination / "data_epoch_0000")
    epoch_stat = Path(first.output_root).stat()
    destination_stat = destination.stat()
    assert (epoch_stat.st_uid, epoch_stat.st_gid) == (
        destination_stat.st_uid,
        destination_stat.st_gid,
    )
    assert first.source_mcap == str(source)
    assert first.behavior_policy_path.endswith("/base/pretrained_model")
    assert first.boundary_reason == "policy_update"
    assert first.fps == 15
    assert first.formats == ["v3.0"]
    assert first.outcome_counts.model_dump() == {
        "total": 3,
        "success": 1,
        "failure": 1,
        "unlabeled": 1,
    }
    assert first.expected_outputs == {
        "v30": str(
            destination
            / "data_epoch_0000"
            / "Task_failure_inference_MCAP_lerobot_v30"
        ),
    }
    assert first.created_at.endswith("Z")
    assert second.data_epoch == 1
    assert second.epoch_name == "data_epoch_0001"

    sidecar = destination / "data_epoch_0000" / app._OFFLINE_RL_DATA_EPOCH_FILE
    sidecar_stat = sidecar.stat()
    assert (sidecar_stat.st_uid, sidecar_stat.st_gid) == (
        destination_stat.st_uid,
        destination_stat.st_gid,
    )
    stored = json.loads(sidecar.read_text())
    assert stored["schema_version"] == 1
    assert stored["data_epoch"] == 0
    assert stored["source_mcap"] == str(source)
    assert stored["outcome_counts"]["failure"] == 1


def test_offline_rl_dataset_summary_reads_parent_data_epoch_sidecar(
    monkeypatch,
    tmp_path,
):
    dataset_root = tmp_path / "workspace" / "lerobot"
    destination = dataset_root / "RLTEST"
    rosbag_root = tmp_path / "workspace" / "rosbag2"
    source = rosbag_root / "Task_01"
    destination.mkdir(parents=True)
    (source / "0").mkdir(parents=True)
    (source / "0" / "episode_info.json").write_text(json.dumps({
        "episode_index": 0,
        "episode_success": False,
    }))
    monkeypatch.setattr(app, "_OFFLINE_RL_DATASET_ROOT", dataset_root)
    monkeypatch.setattr(app, "_OFFLINE_RL_ROSBAG_ROOT", rosbag_root)
    monkeypatch.setattr(
        app,
        "_OFFLINE_RL_DATASET_LOCK_PATH",
        tmp_path / "workspace" / ".cyclo_dataset.lock",
    )
    provenance = app._offline_rl_reserve_data_epoch(
        app.OfflineRLDataEpochReserveRequest(
            destination_root=str(destination),
            source_mcap=str(source),
            fps=10,
            formats=["v3.0"],
        )
    )
    dataset = Path(provenance.expected_outputs["v30"])
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0",
        "total_episodes": 1,
        "total_frames": 20,
        "fps": 10,
        "features": {},
    }))
    monkeypatch.setattr(
        app,
        "_offline_rl_v3_episode_rows",
        lambda _dataset, _count: [
            app.OfflineRLDatasetEpisode(
                index=0,
                frames=20,
                outcome="failure",
                tasks=[],
            )
        ],
    )

    summary = app._offline_rl_dataset_summary(str(dataset))

    assert summary.data_epoch_provenance is not None
    assert summary.data_epoch_provenance.data_epoch == 0
    assert summary.data_epoch_provenance.epoch_name == "data_epoch_0000"
    assert summary.data_epoch_provenance.outcome_counts.failure == 1


def test_offline_rl_dataset_delete_rebuilds_v21_without_reencoding(monkeypatch, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    dataset, _act, _model_root = _offline_rl_test_layout(
        monkeypatch,
        tmp_path,
        episodes=3,
    )
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info.update({
        "codebase_version": "v2.1",
        "total_episodes": 3,
        "total_frames": 6,
        "total_videos": 3,
        "total_chunks": 1,
        "chunks_size": 1000,
        "splits": {"train": "0:3"},
        "data_path": (
            "data/chunk-{episode_chunk:03d}/"
            "episode_{episode_index:06d}.parquet"
        ),
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "annotation_path": (
            "annotations/chunk-{episode_chunk:03d}/"
            "episode_{episode_index:06d}.json"
        ),
        "features": {
            "episode_success": {"dtype": "bool"},
            "observation.images.head": {"dtype": "video"},
        },
    })
    info_path.write_text(json.dumps(info))
    (dataset / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "pick jelly"}) + "\n"
    )
    (dataset / "meta" / "episodes.jsonl").write_text("".join(
        json.dumps({
            "episode_index": index,
            "length": 2,
            "tasks": ["pick jelly"],
        }) + "\n"
        for index in range(3)
    ))
    (dataset / "meta" / "episodes_stats.jsonl").write_text("".join(
        json.dumps({
            "episode_index": index,
            "stats": {
                "episode_success": {"mean": [1.0 if index != 1 else 0.0]},
                "episode_index": {
                    "min": [index], "max": [index], "mean": [float(index)],
                },
                "index": {
                    "min": [index * 2], "max": [index * 2 + 1],
                    "mean": [index * 2 + 0.5],
                },
            },
        }) + "\n"
        for index in range(3)
    ))
    data_dir = dataset / "data" / "chunk-000"
    video_dir = dataset / "videos" / "chunk-000" / "observation.images.head"
    annotation_dir = dataset / "annotations" / "chunk-000"
    data_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    for index in range(3):
        pq.write_table(pa.table({
            "episode_index": pa.array([index, index], type=pa.int64()),
            "index": pa.array([index * 2, index * 2 + 1], type=pa.int64()),
            "frame_index": pa.array([0, 1], type=pa.int64()),
        }), data_dir / f"episode_{index:06d}.parquet")
        (video_dir / f"episode_{index:06d}.mp4").write_bytes(
            f"encoded-video-{index}".encode()
        )
        (annotation_dir / f"episode_{index:06d}.json").write_text(
            json.dumps({"source": index})
        )
    (dataset / "README.md").write_text("dataset card")
    (dataset / "info.json").write_text('{"cyclo": true}')

    monkeypatch.setattr(app, "_OFFLINE_RL_DATASET_LOCK_PATH", tmp_path / "dataset.lock")
    monkeypatch.setattr(app, "_OFFLINE_RL_JOB", None)
    monkeypatch.setattr(app, "_OFFLINE_RL_DATASET_EDIT_ACTIVE", False)

    result = app._offline_rl_delete_dataset_episodes(str(dataset), [1])

    assert result.version == "v2.1"
    assert result.total_episodes == 2
    assert result.total_frames == 4
    rebuilt_info = json.loads(info_path.read_text())
    assert rebuilt_info["splits"] == {"train": "0:2"}
    assert rebuilt_info["total_videos"] == 2
    assert rebuilt_info["total_chunks"] == 1
    rebuilt_rows = [
        json.loads(line)
        for line in (dataset / "meta/episodes.jsonl").read_text().splitlines()
    ]
    assert [row["episode_index"] for row in rebuilt_rows] == [0, 1]
    assert (video_dir / "episode_000000.mp4").read_bytes() == b"encoded-video-0"
    assert (video_dir / "episode_000001.mp4").read_bytes() == b"encoded-video-2"
    second = pq.read_table(data_dir / "episode_000001.parquet")
    assert second.column("episode_index").to_pylist() == [1, 1]
    assert second.column("index").to_pylist() == [2, 3]
    assert json.loads(
        (annotation_dir / "episode_000001.json").read_text()
    ) == {"source": 2}
    assert not list(dataset.parent.glob(f".{dataset.name}.delete-*.tmp"))
    assert not list(dataset.parent.glob(f".{dataset.name}.delete-*.backup"))


def test_offline_rl_dataset_delete_v21_validation_failure_keeps_original(
    monkeypatch,
    tmp_path,
):
    import pytest

    dataset, _act, _model_root = _offline_rl_test_layout(
        monkeypatch,
        tmp_path,
        episodes=2,
    )
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info.update({
        "codebase_version": "v2.1",
        "total_episodes": 2,
        "total_frames": 2,
        "chunks_size": 1000,
        "splits": {"train": "0:2"},
        "data_path": (
            "data/chunk-{episode_chunk:03d}/"
            "episode_{episode_index:06d}.parquet"
        ),
        "features": {},
    })
    info_path.write_text(json.dumps(info))
    (dataset / "meta" / "episodes.jsonl").write_text("".join(
        json.dumps({"episode_index": index, "length": 1, "tasks": []}) + "\n"
        for index in range(2)
    ))
    original_marker = dataset / "original.txt"
    original_marker.write_text("untouched")
    monkeypatch.setattr(app, "_OFFLINE_RL_DATASET_LOCK_PATH", tmp_path / "dataset.lock")
    monkeypatch.setattr(app, "_OFFLINE_RL_JOB", None)
    monkeypatch.setattr(app, "_OFFLINE_RL_DATASET_EDIT_ACTIVE", False)

    with pytest.raises(app.HTTPException) as error:
        app._offline_rl_delete_dataset_episodes(str(dataset), [0])

    assert error.value.status_code == 400
    assert "Missing LeRobot v2.1 episode data" in error.value.detail
    assert original_marker.read_text() == "untouched"
    assert json.loads(info_path.read_text())["total_episodes"] == 2
    assert not list(dataset.parent.glob(f".{dataset.name}.delete-*.tmp"))
    assert not list(dataset.parent.glob(f".{dataset.name}.delete-*.backup"))
    assert app._OFFLINE_RL_DATASET_EDIT_ACTIVE is False


def test_offline_rl_dataset_delete_rebuilds_then_swaps(monkeypatch, tmp_path):
    dataset, _act, _model_root = _offline_rl_test_layout(
        monkeypatch,
        tmp_path,
        episodes=3,
    )
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["annotation_path"] = (
        "annotations/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.json"
    )
    info["chunks_size"] = 1000
    info_path.write_text(json.dumps(info))
    (dataset / "README.md").write_text("dataset card")
    (dataset / "info.json").write_text('{"cyclo": true}')
    annotations = dataset / "annotations" / "chunk-000"
    annotations.mkdir(parents=True)
    for index in range(3):
        (annotations / f"episode_{index:06d}.json").write_text(
            json.dumps({"source": index})
        )
    (dataset / "old-only.txt").write_text("old")

    monkeypatch.setattr(app, "_OFFLINE_RL_DATASET_LOCK_PATH", tmp_path / "dataset.lock")
    monkeypatch.setattr(app, "_OFFLINE_RL_JOB", None)
    monkeypatch.setattr(app, "_OFFLINE_RL_DATASET_EDIT_ACTIVE", False)
    monkeypatch.setattr(
        app,
        "_offline_rl_v3_episode_rows",
        lambda _dataset, count: _offline_rl_summary_rows(count),
    )

    def fake_rebuild(_source, output, _indices, expected):
        (output / "meta").mkdir(parents=True)
        rebuilt_info = dict(info)
        rebuilt_info["total_episodes"] = expected
        rebuilt_info["total_frames"] = expected * 10
        (output / "meta" / "info.json").write_text(json.dumps(rebuilt_info))
        (output / "rebuilt.txt").write_text("new")

    monkeypatch.setattr(
        app,
        "_offline_rl_run_lerobot_episode_delete",
        fake_rebuild,
    )

    result = app._offline_rl_delete_dataset_episodes(str(dataset), [1])

    assert result.total_episodes == 2
    assert not (dataset / "old-only.txt").exists()
    assert (dataset / "rebuilt.txt").read_text() == "new"
    assert (dataset / "README.md").read_text() == "dataset card"
    assert json.loads((dataset / "info.json").read_text()) == {"cyclo": True}
    assert json.loads((dataset / "meta/info.json").read_text())[
        "annotation_path"
    ] == info["annotation_path"]
    assert json.loads(
        (dataset / "annotations/chunk-000/episode_000001.json").read_text()
    ) == {"source": 2}
    assert not list(dataset.parent.glob(f".{dataset.name}.delete-*.tmp"))
    assert not list(dataset.parent.glob(f".{dataset.name}.delete-*.backup"))


def test_offline_rl_dataset_delete_failure_keeps_original(monkeypatch, tmp_path):
    import pytest

    dataset, _act, _model_root = _offline_rl_test_layout(
        monkeypatch,
        tmp_path,
        episodes=3,
    )
    marker = dataset / "original.txt"
    marker.write_text("untouched")
    monkeypatch.setattr(app, "_OFFLINE_RL_DATASET_LOCK_PATH", tmp_path / "dataset.lock")
    monkeypatch.setattr(app, "_OFFLINE_RL_JOB", None)
    monkeypatch.setattr(app, "_OFFLINE_RL_DATASET_EDIT_ACTIVE", False)
    monkeypatch.setattr(
        app,
        "_offline_rl_v3_episode_rows",
        lambda _dataset, count: _offline_rl_summary_rows(count),
    )

    def fail_rebuild(*_args):
        raise app.HTTPException(500, "synthetic rebuild failure")

    monkeypatch.setattr(
        app,
        "_offline_rl_run_lerobot_episode_delete",
        fail_rebuild,
    )

    with pytest.raises(app.HTTPException) as error:
        app._offline_rl_delete_dataset_episodes(str(dataset), [1])

    assert error.value.status_code == 500
    assert marker.read_text() == "untouched"
    assert app._OFFLINE_RL_DATASET_EDIT_ACTIVE is False


def test_offline_rl_dataset_delete_all_removes_dataset(monkeypatch, tmp_path):
    dataset, _act, _model_root = _offline_rl_test_layout(
        monkeypatch,
        tmp_path,
        episodes=2,
    )
    monkeypatch.setattr(app, "_OFFLINE_RL_DATASET_LOCK_PATH", tmp_path / "dataset.lock")
    monkeypatch.setattr(app, "_OFFLINE_RL_JOB", None)
    monkeypatch.setattr(app, "_OFFLINE_RL_DATASET_EDIT_ACTIVE", False)
    monkeypatch.setattr(
        app,
        "_offline_rl_v3_episode_rows",
        lambda _dataset, count: _offline_rl_summary_rows(count),
    )

    result = app._offline_rl_delete_dataset_episodes(str(dataset), [0, 1])

    assert result is None
    assert not dataset.exists()
    assert not list(dataset.parent.glob(f".{dataset.name}.delete-all-*.trash"))
    assert app._OFFLINE_RL_DATASET_EDIT_ACTIVE is False


def test_offline_rl_rejects_dataset_symlink_escape(monkeypatch, tmp_path):
    import pytest

    dataset, _act, _model_root = _offline_rl_test_layout(monkeypatch, tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    escaped = app._OFFLINE_RL_DATASET_ROOT / "escaped"
    escaped.symlink_to(outside, target_is_directory=True)

    with pytest.raises(app.HTTPException) as error:
        app._offline_rl_dataset(str(escaped))

    assert error.value.status_code == 400
    assert "symbolic links" in error.value.detail
    assert app._offline_rl_dataset(str(dataset))[1] == 50


def test_offline_rl_multi_dataset_request_is_ordered_and_aggregated(monkeypatch, tmp_path):
    import pytest

    first, act, _model_root = _offline_rl_test_layout(
        monkeypatch,
        tmp_path,
        episodes=2,
    )
    second = first.parent / "data_epoch_0002" / "recording_v30"
    (second / "meta").mkdir(parents=True)
    (second / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0",
        "total_episodes": 3,
        "total_frames": 30,
        "fps": 15,
        "features": {"episode_success": {"dtype": "bool"}},
    }))
    monkeypatch.setattr(
        app,
        "_offline_rl_v3_episode_rows",
        lambda current, count: [
            app.OfflineRLDatasetEpisode(
                index=index,
                frames=10,
                outcome="success" if current == first.resolve() else "failure",
                tasks=["pick jelly"],
            )
            for index in range(count)
        ],
    )

    request = app.OfflineRLStartRequest(
        dataset_path=str(first),
        dataset_paths=[str(first), str(second)],
        act_checkpoint=str(act),
        robot_type="ffw_sg2_rev1",
    )
    assert app._offline_rl_requested_dataset_paths(request) == [
        str(first),
        str(second),
    ]
    datasets, episodes, successes, failures = app._offline_rl_datasets(
        request.dataset_paths
    )
    assert datasets == [first.resolve(), second.resolve()]
    assert (episodes, successes, failures) == (5, 2, 3)

    duplicate = app.OfflineRLStartRequest(
        dataset_paths=[str(first), str(first)],
        act_checkpoint=str(act),
        robot_type="ffw_sg2_rev1",
    )
    with pytest.raises(app.HTTPException) as error:
        app._offline_rl_datasets(duplicate.dataset_paths)
    assert "duplicates" in error.value.detail

    ambiguous = app.OfflineRLStartRequest(
        dataset_path=str(second),
        dataset_paths=[str(first), str(second)],
        act_checkpoint=str(act),
        robot_type="ffw_sg2_rev1",
    )
    with pytest.raises(app.HTTPException) as error:
        app._offline_rl_requested_dataset_paths(ambiguous)
    assert "first ordered" in error.value.detail


def test_offline_rl_enforces_v30_max_and_first_round_cap(monkeypatch, tmp_path):
    import pytest

    dataset, _act, _model_root = _offline_rl_test_layout(
        monkeypatch,
        tmp_path,
        episodes=201,
    )
    with pytest.raises(app.HTTPException) as error:
        app._offline_rl_dataset(str(dataset))
    assert error.value.status_code == 400
    assert "at most 200" in error.value.detail

    (dataset / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0",
        "total_episodes": 100,
        "features": {"episode_success": {"dtype": "bool"}},
    }))
    with pytest.raises(app.HTTPException) as error:
        app._offline_rl_parent_checkpoint("", 100)
    assert "1..50" in error.value.detail
    assert app._offline_rl_parent_checkpoint("", 30) == (None, 0, 0)


def test_offline_rl_parent_accepts_inferred_next_1_to_50(monkeypatch, tmp_path):
    import pytest

    _dataset, _act, model_root = _offline_rl_test_layout(
        monkeypatch,
        tmp_path,
        episodes=100,
    )
    round_root = model_root / "offline_rl" / "round_0050"
    parent = round_root / "training_state" / "act_td3.pt"
    parent.parent.mkdir(parents=True)
    parent.write_bytes(b"checkpoint")
    (round_root / "training_manifest.json").write_text(json.dumps({
        "event": "result",
        "status": "complete",
        "episode_count": 50,
        "checkpoint_path": str(parent),
    }))

    assert app._offline_rl_parent_checkpoint(str(parent), 100) == (parent, 50, 1)
    assert app._offline_rl_parent_checkpoint(str(parent), 51) == (parent, 50, 1)
    with pytest.raises(app.HTTPException) as error:
        app._offline_rl_parent_checkpoint(str(parent), 101)
    assert "1..50" in error.value.detail


def test_offline_rl_parent_validates_its_recorded_td3_schedule(monkeypatch, tmp_path):
    import pytest

    _dataset, _act, model_root = _offline_rl_test_layout(monkeypatch, tmp_path)
    round_root = model_root / "offline_rl" / "round_custom"
    parent = round_root / "training_state" / "act_td3.pt"
    parent.parent.mkdir(parents=True)
    parent.write_bytes(b"checkpoint")
    (round_root / "training_manifest.json").write_text(json.dumps({
        "event": "result",
        "status": "complete",
        "episode_count": 30,
        "round_index": 1,
        "checkpoint_path": str(parent),
        "schedule": {
            "critic_epochs": 6,
            "actor_equivalent_epochs": 3,
        },
    }))

    assert app._offline_rl_parent_checkpoint(str(parent), 60) == (parent, 30, 1)

    manifest_path = round_root / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schedule"]["actor_equivalent_epochs"] = 2
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(app.HTTPException) as error:
        app._offline_rl_parent_checkpoint(str(parent), 60)
    assert "schedule is invalid" in error.value.detail


def test_offline_rl_parent_requires_an_exact_explicit_actor_objective(
    monkeypatch,
    tmp_path,
):
    import pytest

    _dataset, _act, model_root = _offline_rl_test_layout(monkeypatch, tmp_path)
    round_root = model_root / "offline_rl" / "round_objective"
    parent = round_root / "training_state" / "act_td3.pt"
    parent.parent.mkdir(parents=True)
    parent.write_bytes(b"checkpoint")
    manifest_path = round_root / "training_manifest.json"
    manifest = {
        "event": "result",
        "status": "complete",
        "episode_count": 30,
        "checkpoint_path": str(parent),
        "actor_trainable_groups": ["visual_backbone", "action_decoder"],
    }
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(app.HTTPException, match=r"fresh TD3 or TD3\+BC lineage"):
        app._offline_rl_parent_checkpoint(
            str(parent),
            60,
            ["visual_backbone", "action_decoder"],
            4,
            "td3_bc",
        )

    manifest["actor_objective"] = "td3_bc"
    manifest_path.write_text(json.dumps(manifest))
    # Result manifests written before the machine-readable algorithm field are
    # safe only because the exact actor objective is present.
    assert app._offline_rl_parent_checkpoint(
        str(parent),
        60,
        ["visual_backbone", "action_decoder"],
        4,
        "td3_bc",
    ) == (parent, 30, 1)

    manifest["algorithm"] = "ACT-TD3+BC cumulative replay"
    manifest_path.write_text(json.dumps(manifest))
    assert app._offline_rl_parent_checkpoint(
        str(parent),
        60,
        ["visual_backbone", "action_decoder"],
        4,
        "td3_bc",
    ) == (parent, 30, 1)

    manifest["algorithm"] = "td3"
    manifest_path.write_text(json.dumps(manifest))
    assert app._offline_rl_parent_checkpoint(
        str(parent),
        60,
        ["visual_backbone", "action_decoder"],
        4,
        "td3_bc",
    ) == (parent, 30, 1)

    manifest["algorithm"] = "ACT-TD3 cumulative replay"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(app.HTTPException, match="compatible TD3 artifact"):
        app._offline_rl_parent_checkpoint(
            str(parent),
            60,
            ["visual_backbone", "action_decoder"],
            4,
            "td3_bc",
        )

    manifest["algorithm"] = "td3"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(app.HTTPException, match="does not match"):
        app._offline_rl_parent_checkpoint(
            str(parent),
            60,
            ["visual_backbone", "action_decoder"],
            4,
            "td3",
        )


def test_offline_rl_parent_requires_matching_batch_size(monkeypatch, tmp_path):
    import pytest

    _dataset, _act, model_root = _offline_rl_test_layout(monkeypatch, tmp_path)
    round_root = model_root / "offline_rl" / "round_batch"
    parent = round_root / "training_state" / "act_td3.pt"
    parent.parent.mkdir(parents=True)
    parent.write_bytes(b"checkpoint")
    (round_root / "training_manifest.json").write_text(json.dumps({
        "event": "result",
        "status": "complete",
        "episode_count": 30,
        "checkpoint_path": str(parent),
        "batch_size": 8,
    }))

    assert app._offline_rl_parent_checkpoint(
        str(parent),
        60,
        batch_size=8,
    ) == (parent, 30, 1)
    with pytest.raises(app.HTTPException) as error:
        app._offline_rl_parent_checkpoint(str(parent), 60, batch_size=4)
    assert error.value.status_code == 400
    assert "batch_size does not match" in error.value.detail


def test_offline_rl_actor_trainability_contract_normalizes_and_validates():
    import pytest

    assert app._offline_rl_actor_trainable_groups([
        "action_decoder",
        "visual_backbone",
        "transformer_encoder",
    ]) == [
        "visual_backbone",
        "transformer_encoder",
        "action_decoder",
    ]

    invalid_contracts = (
        ([], "all-frozen"),
        (["visual_backbone", "visual_backbone"], "duplicates"),
        (["visual_backbone", "unknown"], "Unknown"),
        (["cvae_encoder"], "CVAE-only"),
    )
    for groups, message in invalid_contracts:
        with pytest.raises(app.HTTPException) as error:
            app._offline_rl_actor_trainable_groups(groups)
        assert error.value.status_code == 400
        assert message in error.value.detail

    request = app.OfflineRLStartRequest(
        dataset_path="/workspace/lerobot/data",
        act_checkpoint="/workspace/model/lerobot/base",
        robot_type="ffw_sg2_rev1",
    )
    assert request.actor_trainable_groups == list(
        app._OFFLINE_RL_ACTOR_TRAINABLE_GROUPS
    )
    assert request.algorithm == "td3"
    assert request.actor_objective == "td3_bc"
    assert app._offline_rl_objective_trainable_groups(
        "td3",
        list(app._OFFLINE_RL_ACTOR_TRAINABLE_GROUPS),
    ) == ["visual_backbone", "transformer_encoder", "action_decoder"]
    assert app._offline_rl_objective_trainable_groups(
        "td3_bc",
        list(app._OFFLINE_RL_ACTOR_TRAINABLE_GROUPS),
    ) == list(app._OFFLINE_RL_ACTOR_TRAINABLE_GROUPS)


def test_offline_rl_request_separates_algorithm_from_actor_loss_option():
    import pytest

    common = {
        "dataset_path": "/workspace/lerobot/data",
        "act_checkpoint": "/workspace/model/lerobot/base",
        "robot_type": "ffw_sg2_rev1",
    }
    pure = app.OfflineRLStartRequest(**common, actor_objective="td3")
    hybrid = app.OfflineRLStartRequest(**common, actor_objective="td3_bc")

    assert pure.algorithm == hybrid.algorithm == "td3"
    assert pure.actor_objective == "td3"
    assert hybrid.actor_objective == "td3_bc"
    with pytest.raises(app.ValidationError):
        app.OfflineRLStartRequest(**common, actor_objective="unknown")

    # The API's documented default is TD3+BC when neither field is supplied.
    assert app._offline_rl_algorithm_contract(
        app.OfflineRLStartRequest(**common)
    ) == ("td3", "td3_bc")
    # The previous API placed the loss choice in ``algorithm``; preserve both
    # old spellings without silently changing their actor objective.
    assert app._offline_rl_algorithm_contract(
        app.OfflineRLStartRequest(**common, algorithm="td3")
    ) == ("td3", "td3")
    assert app._offline_rl_algorithm_contract(
        app.OfflineRLStartRequest(**common, algorithm="td3_bc")
    ) == ("td3", "td3_bc")


def test_offline_rl_parent_requires_matching_actor_trainability(monkeypatch, tmp_path):
    import pytest

    _dataset, _act, model_root = _offline_rl_test_layout(monkeypatch, tmp_path)
    round_root = model_root / "offline_rl" / "round_contract"
    parent = round_root / "training_state" / "act_td3.pt"
    parent.parent.mkdir(parents=True)
    parent.write_bytes(b"checkpoint")
    (round_root / "training_manifest.json").write_text(json.dumps({
        "event": "result",
        "status": "complete",
        "episode_count": 30,
        "checkpoint_path": str(parent),
        "actor_trainable_groups": ["visual_backbone", "action_decoder"],
    }))

    assert app._offline_rl_parent_checkpoint(
        str(parent),
        60,
        ["visual_backbone", "action_decoder"],
    ) == (parent, 30, 1)

    with pytest.raises(app.HTTPException) as error:
        app._offline_rl_parent_checkpoint(
            str(parent),
            60,
            ["action_decoder"],
        )
    assert error.value.status_code == 400
    assert "do not match" in error.value.detail


def test_offline_rl_legacy_parent_requires_all_actor_groups(monkeypatch, tmp_path):
    import pytest

    _dataset, _act, model_root = _offline_rl_test_layout(monkeypatch, tmp_path)
    round_root = model_root / "offline_rl" / "legacy_round_contract"
    parent = round_root / "training_state" / "act_td3.pt"
    parent.parent.mkdir(parents=True)
    parent.write_bytes(b"checkpoint")
    (round_root / "training_manifest.json").write_text(json.dumps({
        "event": "result",
        "status": "complete",
        "episode_count": 30,
        "checkpoint_path": str(parent),
    }))

    assert app._offline_rl_parent_checkpoint(
        str(parent),
        60,
        list(app._OFFLINE_RL_ACTOR_TRAINABLE_GROUPS),
    ) == (parent, 30, 1)

    with pytest.raises(app.HTTPException) as error:
        app._offline_rl_parent_checkpoint(
            str(parent),
            60,
            ["visual_backbone", "action_decoder"],
        )
    assert error.value.status_code == 400
    assert "Legacy parent checkpoints" in error.value.detail

    with pytest.raises(app.HTTPException) as error:
        app._offline_rl_parent_checkpoint(
            str(parent),
            60,
            list(app._OFFLINE_RL_ACTOR_TRAINABLE_GROUPS),
            8,
        )
    assert error.value.status_code == 400
    assert "batch_size does not match" in error.value.detail


def test_offline_rl_command_is_pinned_and_offline(monkeypatch):
    monkeypatch.setattr(app, "_compose_base_cmd", lambda: ["docker", "compose"])
    job = app._OfflineRLJob(
        job_id="1234567890abcdef",
        dataset_path="/workspace/lerobot/data",
        act_checkpoint="/workspace/model/lerobot/base/pretrained_model",
        parent_checkpoint="/workspace/model/lerobot/round/training_state/act_td3.pt",
        output_dir="/workspace/model/lerobot/offline_rl/round",
        episode_count=100,
        log_path="/tmp/job.log",
        batch_size=16,
        dataset_paths=[
            "/workspace/lerobot/data_epoch_0001/data",
            "/workspace/lerobot/data_epoch_0002/data",
        ],
    )

    command = app._offline_rl_command(
        job=job,
        robot_type="ffw_sg2_rev1",
        robot_config="/orchestrator_config/ffw_sg2_rev1_config.yaml",
    )

    assert command[:4] == ["docker", "compose", "run", "--rm"]
    assert command[command.index("--pull") + 1] == "never"
    assert command[command.index("--user") + 1] == "1000:1000"
    assert command[command.index("--entrypoint") + 1] == "/lerobot/.venv/bin/python"
    assert "HOME=/tmp" in command
    assert "XDG_CACHE_HOME=/tmp/cyclo_offline_rl_cache" in command
    assert "HF_HOME=/tmp/cyclo_offline_rl_cache/huggingface" in command
    assert (
        "HF_LEROBOT_HOME=/tmp/cyclo_offline_rl_cache/huggingface/lerobot"
        in command
    )
    assert "TORCH_HOME=/tmp/cyclo_offline_rl_cache/torch" in command
    assert "TRITON_CACHE_DIR=/tmp/cyclo_offline_rl_cache/triton" in command
    runtime_environment = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--env"
    ]
    assert all("/root/.cache" not in value for value in runtime_environment)
    assert "HF_HUB_OFFLINE=1" in command
    assert "--allow-partial-round" not in command
    dataset_roots = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--dataset-root"
    ]
    assert dataset_roots == job.dataset_paths
    assert command[command.index("--batch-size") + 1] == "16"
    assert command[command.index("--actor-objective") + 1] == "td3_bc"
    status = app._offline_rl_status(job)
    assert status.algorithm == "td3"
    assert status.actor_objective == "td3_bc"
    assert status.batch_size == 16
    assert command[command.index("--critic-epochs") + 1] == "10"
    assert command[command.index("--actor-equivalent-epochs") + 1] == "5"
    trainable_groups = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--actor-trainable-group"
    ]
    assert trainable_groups == list(app._OFFLINE_RL_ACTOR_TRAINABLE_GROUPS)
    assert command[-2:] == ["--parent-checkpoint", job.parent_checkpoint]


def test_offline_rl_result_event_does_not_recurse_and_updates_progress():
    job = app._OfflineRLJob(
        job_id="job",
        dataset_path="/workspace/lerobot/data",
        act_checkpoint="/workspace/model/lerobot/base",
        parent_checkpoint="",
        output_dir="/workspace/model/lerobot/offline_rl/round",
        episode_count=50,
        log_path="/tmp/job.log",
    )
    model_path = f"{job.output_dir}/pretrained_model"

    complete = app._offline_rl_consume_event(job, {
        "event": "result",
        "algorithm": "td3",
        "actor_objective": "td3_bc",
        "status": "complete",
        "percentage": 100.0,
        "completed_epochs": 10,
        "total_epochs": 10,
        "completed_critic_updates": 320,
        "total_critic_updates": 320,
        "completed_actor_updates": 160,
        "total_actor_updates": 160,
        "critic_loss": 0.01,
        "actor_loss": 0.02,
        "eta_seconds": 0.0,
        "checkpoint_path": f"{job.output_dir}/training_state/act_td3.pt",
        "critic_source": "policy_warmup",
        "critic_checkpoint": f"{job.act_checkpoint}/critic/latest.pt",
        "model_path": model_path,
    })

    assert complete is True
    assert job.percentage == 100.0
    assert job.completed_epochs == 10
    assert job.completed_critic_updates == 320
    assert job.completed_actor_updates == 160
    assert job.model_path == model_path
    assert job.critic_source == "policy_warmup"
    assert job.critic_checkpoint == f"{job.act_checkpoint}/critic/latest.pt"
    status = app._offline_rl_status(job)
    assert status.critic_source == "policy_warmup"
    assert status.critic_checkpoint == f"{job.act_checkpoint}/critic/latest.pt"


def test_offline_rl_progress_persists_typed_loss_history_and_updates_same_step():
    job = app._OfflineRLJob(
        job_id="loss-history",
        dataset_path="/workspace/lerobot/data",
        act_checkpoint="/workspace/model/lerobot/base",
        parent_checkpoint="",
        output_dir="/workspace/model/lerobot/offline_rl/round",
        episode_count=50,
        log_path="/tmp/job.log",
    )

    app._offline_rl_consume_event(job, {
        "event": "progress",
        "completed_critic_updates": 1,
        "critic_loss": 0.8,
    })
    app._offline_rl_consume_event(job, {
        "event": "progress",
        "completed_critic_updates": 1,
        "actor_loss": -0.2,
    })
    app._offline_rl_consume_event(job, {
        "event": "progress",
        "completed_critic_updates": 2,
        "critic_loss": 0.6,
        "actor_loss": -0.3,
    })

    status = app._offline_rl_status(job)
    assert status.loss_history == [
        app.OfflineRLLossPoint(step=1, critic_loss=0.8, actor_loss=-0.2),
        app.OfflineRLLossPoint(step=2, critic_loss=0.6, actor_loss=-0.3),
    ]
    assert all(isinstance(point, app.OfflineRLLossPoint) for point in status.loss_history)


def test_offline_rl_loss_history_is_finite_monotonic_and_bounded():
    job = app._OfflineRLJob(
        job_id="bounded-loss-history",
        dataset_path="/workspace/lerobot/data",
        act_checkpoint="/workspace/model/lerobot/base",
        parent_checkpoint="",
        output_dir="/workspace/model/lerobot/offline_rl/round",
        episode_count=50,
        log_path="/tmp/job.log",
    )

    app._offline_rl_consume_event(job, {
        "event": "progress",
        "completed_critic_updates": 0,
        "critic_loss": float("nan"),
        "actor_loss": float("inf"),
        "percentage": float("nan"),
    })
    assert job.loss_history == []
    assert job.critic_loss is None
    assert job.actor_loss is None
    assert job.percentage == 0.0

    last_step = app._OFFLINE_RL_LOSS_HISTORY_POINTS + 5
    for step in range(1, last_step + 1):
        app._offline_rl_consume_event(job, {
            "event": "progress",
            "completed_critic_updates": step,
            "critic_loss": 1.0 / step,
        })
    app._offline_rl_consume_event(job, {
        "event": "progress",
        "completed_critic_updates": 0,
        "critic_loss": 99.0,
    })

    status = app._offline_rl_status(job)
    assert len(status.loss_history) == app._OFFLINE_RL_LOSS_HISTORY_POINTS
    assert status.loss_history[0].step == 6
    assert status.loss_history[-1].step == last_step
    assert [point.step for point in status.loss_history] == sorted(
        point.step for point in status.loss_history
    )
    assert all(
        point.critic_loss is None or math.isfinite(point.critic_loss)
        for point in status.loss_history
    )


def test_offline_rl_progress_persists_round_mean_metric_history():
    job = app._OfflineRLJob(
        job_id="rl-metric-history",
        dataset_path="/workspace/lerobot/data",
        act_checkpoint="/workspace/model/lerobot/base",
        parent_checkpoint="",
        output_dir="/workspace/model/lerobot/offline_rl/round",
        episode_count=50,
        log_path="/tmp/job.log",
    )
    first = {
        "rl_epoch": 1,
        "actor_loss_mean": None,
        "critic_loss_mean": 0.8,
        "replay_average_reward": 0.6,
    }
    running_second = {
        "rl_epoch": 2,
        "actor_loss_mean": None,
        "critic_loss_mean": None,
        "replay_average_reward": 0.7,
    }
    app._offline_rl_consume_event(job, {
        "event": "progress",
        "rl_metric_history": [first, running_second],
    })
    updated_second = {
        **running_second,
        "actor_loss_mean": -0.2,
        "critic_loss_mean": 0.4,
    }
    app._offline_rl_consume_event(job, {
        "event": "progress",
        "rl_metric_history": [first, updated_second],
    })

    status = app._offline_rl_status(job)
    assert status.rl_metric_history == [
        app.OfflineRLRLMetricPoint(**first),
        app.OfflineRLRLMetricPoint(**updated_second),
    ]
    assert all(
        isinstance(point, app.OfflineRLRLMetricPoint)
        for point in status.rl_metric_history
    )
    # Existing raw optimizer-update telemetry remains independent.
    assert status.loss_history == []


def test_offline_rl_metric_history_validation_fails_closed():
    import pytest

    job = app._OfflineRLJob(
        job_id="invalid-rl-metrics",
        dataset_path="/workspace/lerobot/data",
        act_checkpoint="/workspace/model/lerobot/base",
        parent_checkpoint="",
        output_dir="/workspace/model/lerobot/offline_rl/round",
        episode_count=50,
        log_path="/tmp/job.log",
    )
    base = {
        "rl_epoch": 1,
        "actor_loss_mean": 0.1,
        "critic_loss_mean": 0.2,
        "replay_average_reward": 0.5,
    }
    invalid_histories = (
        [{**base, "critic_loss_mean": float("nan")}],
        [{**base, "replay_average_reward": 1.1}],
        [base, dict(base)],
        [{key: value for key, value in base.items() if key != "actor_loss_mean"}],
    )
    for history in invalid_histories:
        with pytest.raises(ValueError, match="ACT-TD3"):
            app._offline_rl_consume_event(job, {
                "event": "progress",
                "rl_metric_history": history,
            })
    assert job.rl_metric_history == []


def test_offline_rl_critic_source_telemetry_fails_closed():
    import pytest

    job = app._OfflineRLJob(
        job_id="job",
        dataset_path="/workspace/lerobot/data",
        act_checkpoint="/workspace/model/lerobot/base",
        parent_checkpoint="/workspace/model/lerobot/round_0/training_state/act_td3.pt",
        output_dir="/workspace/model/lerobot/offline_rl/round_1",
        episode_count=50,
        log_path="/tmp/job.log",
        checkpoint_path="/workspace/model/lerobot/offline_rl/round_1/training_state/act_td3.pt",
    )
    app._offline_rl_consume_event(job, {
        "event": "manifest",
        "algorithm": "td3",
        "actor_objective": "td3_bc",
        "checkpoint": job.checkpoint_path,
        "critic_source": "parent_checkpoint",
        "critic_checkpoint": job.parent_checkpoint,
    })
    assert job.critic_source == "parent_checkpoint"
    assert job.critic_checkpoint == job.parent_checkpoint

    with pytest.raises(ValueError, match="actor objective telemetry disagrees"):
        app._offline_rl_consume_event(job, {
            "event": "manifest",
            "algorithm": "td3",
            "critic_source": "parent_checkpoint",
            "critic_checkpoint": job.parent_checkpoint,
        })

    with pytest.raises(ValueError, match="algorithm telemetry disagrees"):
        app._offline_rl_consume_event(job, {
            "event": "manifest",
            "algorithm": "td3_bc",
            "actor_objective": "td3_bc",
            "critic_source": "parent_checkpoint",
            "critic_checkpoint": job.parent_checkpoint,
        })
    with pytest.raises(ValueError, match="algorithm result disagrees"):
        app._offline_rl_consume_event(job, {
            "event": "result",
            "algorithm": "ACT-TD3 cumulative replay",
            "actor_objective": "td3_bc",
            "status": "stopped",
        })
    with pytest.raises(ValueError, match="actor objective telemetry disagrees"):
        app._offline_rl_consume_event(job, {
            "event": "manifest",
            "algorithm": "td3",
            "actor_objective": "td3",
            "critic_source": "parent_checkpoint",
            "critic_checkpoint": job.parent_checkpoint,
        })

    with pytest.raises(ValueError, match="fields are incomplete"):
        app._offline_rl_consume_event(job, {
            "event": "manifest",
            "algorithm": "td3",
            "actor_objective": "td3_bc",
            "critic_source": "random",
        })
    with pytest.raises(ValueError, match="disagrees"):
        app._offline_rl_consume_event(job, {
            "event": "manifest",
            "algorithm": "td3",
            "actor_objective": "td3_bc",
            "critic_source": "policy_warmup",
            "critic_checkpoint": "/workspace/model/lerobot/other/critic/latest.pt",
        })


def test_offline_rl_monitor_accepts_only_verified_export(monkeypatch, tmp_path):
    _dataset, _act, model_root = _offline_rl_test_layout(monkeypatch, tmp_path)
    output = model_root / "offline_rl" / "round"
    model = output / "pretrained_model"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"type":"act"}')
    (model / "model.safetensors").write_bytes(b"weights")
    checkpoint = output / "training_state" / "act_td3.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"state")
    result = json.dumps({
        "event": "result",
        "algorithm": "td3",
        "actor_objective": "td3_bc",
        "status": "complete",
        "episode_count": 50,
        "completed_epochs": 10,
        "total_epochs": 10,
        "completed_critic_updates": 20,
        "total_critic_updates": 20,
        "completed_actor_updates": 10,
        "total_actor_updates": 10,
        "percentage": 100.0,
        "critic_loss": 0.1,
        "actor_loss": 0.2,
        "eta_seconds": 0.0,
        "checkpoint_path": str(checkpoint),
        "critic_source": "random",
        "critic_checkpoint": None,
        "model_path": str(model),
    })

    class FakeProcess:
        stdout = [result + "\n"]

        @staticmethod
        def wait():
            return 0

    job = app._OfflineRLJob(
        job_id="job",
        dataset_path=str(app._OFFLINE_RL_DATASET_ROOT / "recording_v30"),
        act_checkpoint=str(_act),
        parent_checkpoint="",
        output_dir=str(output),
        episode_count=50,
        log_path=str(tmp_path / "logs" / "job.log"),
        process=FakeProcess(),
        stop_requested=True,
    )

    app._monitor_offline_rl_job(job)

    assert job.status == "completed"
    assert job.returncode == 0
    assert app._offline_rl_status(job).model_path == str(model)
    assert app._offline_rl_status(job).critic_source == "random"
    assert app._offline_rl_status(job).critic_checkpoint == ""


def test_offline_rl_monitor_confirms_cooperative_stop_and_preserves_failures(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(app, "_OFFLINE_RL_LOG_ROOT", tmp_path / "logs")
    stopped_result = json.dumps({
        "event": "result",
        "algorithm": "td3",
        "actor_objective": "td3_bc",
        "status": "stopped",
        "percentage": 25.0,
        "completed_critic_updates": 10,
        "total_critic_updates": 40,
        "eta_seconds": 60.0,
        "checkpoint_path": str(tmp_path / "stopped" / "act_td3.pt"),
        "model_path": None,
    })

    class StoppedProcess:
        stdout = [stopped_result + "\n"]

        @staticmethod
        def wait():
            return 0

    stopped_job = app._OfflineRLJob(
        job_id="stopped",
        dataset_path="/workspace/lerobot/data",
        act_checkpoint="/workspace/model/lerobot/base",
        parent_checkpoint="",
        output_dir="/workspace/model/lerobot/offline_rl/stopped",
        episode_count=50,
        log_path=str(tmp_path / "stopped.log"),
        process=StoppedProcess(),
        stop_requested=True,
    )
    app._monitor_offline_rl_job(stopped_job)

    assert stopped_job.status == "stopped"
    assert stopped_job.stop_confirmed is True
    assert stopped_job.returncode == 0
    assert stopped_job.checkpoint_path.endswith("act_td3.pt")
    assert stopped_job.eta_seconds is None

    class FailedProcess:
        stdout = []

        @staticmethod
        def wait():
            return 1

    failed_job = app._OfflineRLJob(
        job_id="failed",
        dataset_path="/workspace/lerobot/data",
        act_checkpoint="/workspace/model/lerobot/base",
        parent_checkpoint="",
        output_dir="/workspace/model/lerobot/offline_rl/failed",
        episode_count=50,
        log_path=str(tmp_path / "failed.log"),
        process=FailedProcess(),
        stop_requested=True,
    )
    app._monitor_offline_rl_job(failed_job)

    assert failed_job.status == "failed"
    assert failed_job.stop_confirmed is False


def test_offline_rl_stop_rejects_stale_id_and_signals_only_current_container(
    monkeypatch,
    tmp_path,
):
    import pytest

    class FakeProcess:
        def __init__(self):
            self.signals = []

        @staticmethod
        def poll():
            return None

        def send_signal(self, value):
            self.signals.append(value)

    class FakeContainer:
        def __init__(self):
            self.signals = []

        def kill(self, *, signal):
            self.signals.append(signal)

    process = FakeProcess()
    container = FakeContainer()
    requested_containers = []

    class FakeContainers:
        def get(self, name):
            requested_containers.append(name)
            return container

    job = app._OfflineRLJob(
        job_id="1234567890abcdef1234567890abcdef",
        dataset_path="/workspace/lerobot/data",
        act_checkpoint="/workspace/model/lerobot/base",
        parent_checkpoint="",
        output_dir="/workspace/model/lerobot/offline_rl/current",
        episode_count=50,
        log_path=str(tmp_path / "current.log"),
        process=process,
    )
    monkeypatch.setattr(app, "_OFFLINE_RL_JOB", job)
    monkeypatch.setattr(
        app,
        "_docker_client",
        lambda: SimpleNamespace(containers=FakeContainers()),
    )

    with pytest.raises(app.HTTPException) as error:
        asyncio.run(app.offline_rl_stop(app.OfflineRLStopRequest(job_id="stale")))
    assert error.value.status_code == 409
    assert requested_containers == []

    response = asyncio.run(app.offline_rl_stop(
        app.OfflineRLStopRequest(job_id=job.job_id)
    ))
    assert response.status == "running"
    assert response.message == "Stopping ACT-TD3 training"
    assert response.job_id == job.job_id
    assert requested_containers == ["cyclo_offline_rl_1234567890ab"]
    assert container.signals == ["SIGINT"]
    assert process.signals == []
    assert job.stop_requested is True

    # A repeated request is idempotent and does not signal the target twice.
    asyncio.run(app.offline_rl_stop(app.OfflineRLStopRequest(job_id=job.job_id)))
    assert container.signals == ["SIGINT"]


def test_offline_rl_stop_falls_back_to_current_compose_process(monkeypatch, tmp_path):
    class FakeProcess:
        def __init__(self):
            self.signals = []

        @staticmethod
        def poll():
            return None

        def send_signal(self, value):
            self.signals.append(value)

    class MissingContainers:
        @staticmethod
        def get(_name):
            raise app.NotFound("not created")

    process = FakeProcess()
    job = app._OfflineRLJob(
        job_id="current-job",
        dataset_path="/workspace/lerobot/data",
        act_checkpoint="/workspace/model/lerobot/base",
        parent_checkpoint="",
        output_dir="/workspace/model/lerobot/offline_rl/current",
        episode_count=50,
        log_path=str(tmp_path / "current.log"),
        process=process,
    )
    monkeypatch.setattr(app, "_OFFLINE_RL_JOB", job)
    monkeypatch.setattr(
        app,
        "_docker_client",
        lambda: SimpleNamespace(containers=MissingContainers()),
    )

    asyncio.run(app.offline_rl_stop(app.OfflineRLStopRequest(job_id=job.job_id)))

    assert process.signals == [app.signal.SIGINT]
    assert job.stop_requested is True


def test_offline_rl_stop_rolls_back_when_no_target_can_be_signalled(
    monkeypatch,
    tmp_path,
):
    import pytest

    class MissingContainers:
        @staticmethod
        def get(_name):
            raise app.NotFound("already gone")

    job = app._OfflineRLJob(
        job_id="already-exited",
        dataset_path="/workspace/lerobot/data",
        act_checkpoint="/workspace/model/lerobot/base",
        parent_checkpoint="",
        output_dir="/workspace/model/lerobot/offline_rl/exited",
        episode_count=50,
        log_path=str(tmp_path / "exited.log"),
        process=None,
    )
    monkeypatch.setattr(app, "_OFFLINE_RL_JOB", job)
    monkeypatch.setattr(
        app,
        "_docker_client",
        lambda: SimpleNamespace(containers=MissingContainers()),
    )

    with pytest.raises(app.HTTPException) as error:
        asyncio.run(app.offline_rl_stop(app.OfflineRLStopRequest(job_id=job.job_id)))

    assert error.value.status_code == 409
    assert job.stop_requested is False
    assert job.status == "running"


def test_offline_rl_start_launches_single_pinned_compose_job(monkeypatch, tmp_path):
    dataset, act, _model_root = _offline_rl_test_layout(monkeypatch, tmp_path)
    launched = {}

    class FakeProcess:
        stdout = []

        @staticmethod
        def wait():
            return 0

    def fake_popen(command, **kwargs):
        launched["command"] = command
        launched["kwargs"] = kwargs
        return FakeProcess()

    class FakeThread:
        def __init__(self, **kwargs):
            launched["thread"] = kwargs

        def start(self):
            launched["thread_started"] = True

    monkeypatch.setattr(app, "_compose_base_cmd", lambda: ["docker", "compose"])
    monkeypatch.setattr(app, "_compose_env", lambda: {"ARCH": "amd64"})
    monkeypatch.setattr(
        app,
        "_offline_rl_robot_config",
        lambda _robot: "/orchestrator_config/ffw_sg2_rev1_config.yaml",
    )
    monkeypatch.setattr(app.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(app.threading, "Thread", FakeThread)
    monkeypatch.setattr(
        app.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="1234567890abcdef1234567890abcdef"),
    )
    monkeypatch.setattr(app, "_OFFLINE_RL_JOB", None)
    monkeypatch.setattr(
        app,
        "_offline_rl_v3_episode_rows",
        lambda _dataset, count: _offline_rl_summary_rows(count),
    )

    response = asyncio.run(app.offline_rl_start(app.OfflineRLStartRequest(
        dataset_path=str(dataset),
        act_checkpoint=str(act),
        parent_checkpoint="",
        algorithm="td3",
        actor_objective="td3",
        robot_type="ffw_sg2_rev1",
        batch_size=8,
        actor_trainable_groups=["action_decoder", "visual_backbone"],
    )))

    assert response.status == "running"
    assert response.algorithm == "td3"
    assert response.actor_objective == "td3"
    assert response.episode_count == 50
    assert response.dataset_paths == [str(dataset)]
    assert response.success_count == 49
    assert response.failure_count == 1
    assert response.round_index == 1
    assert response.round_episode_count == 50
    assert response.batch_size == 8
    assert response.critic_epochs == 10
    assert response.actor_equivalent_epochs == 5
    assert response.actor_trainable_groups == ["visual_backbone", "action_decoder"]
    assert response.model_path == ""
    assert launched["thread_started"] is True
    assert launched["kwargs"]["text"] is True
    assert launched["command"][launched["command"].index("--pull") + 1] == "never"
    assert "/lerobot/.venv/bin/python" in launched["command"]
    assert launched["command"][launched["command"].index("--batch-size") + 1] == "8"
    assert launched["command"][launched["command"].index("--actor-objective") + 1] == "td3"
    assert [
        launched["command"][index + 1]
        for index, value in enumerate(launched["command"][:-1])
        if value == "--actor-trainable-group"
    ] == ["visual_backbone", "action_decoder"]


def test_offline_rl_start_rejects_concurrent_job(monkeypatch, tmp_path):
    dataset, act, _model_root = _offline_rl_test_layout(monkeypatch, tmp_path)
    running = app._OfflineRLJob(
        job_id="running",
        dataset_path=str(dataset),
        act_checkpoint=str(act),
        parent_checkpoint="",
        output_dir=str(app._OFFLINE_RL_OUTPUT_ROOT / "running"),
        episode_count=50,
        log_path=str(tmp_path / "running.log"),
    )
    monkeypatch.setattr(app, "_OFFLINE_RL_JOB", running)
    monkeypatch.setattr(app, "_compose_base_cmd", lambda: ["docker", "compose"])
    monkeypatch.setattr(app, "_compose_env", lambda: {"ARCH": "amd64"})
    monkeypatch.setattr(
        app,
        "_offline_rl_robot_config",
        lambda _robot: "/orchestrator_config/ffw_sg2_rev1_config.yaml",
    )

    import pytest
    with pytest.raises(app.HTTPException) as error:
        asyncio.run(app.offline_rl_start(app.OfflineRLStartRequest(
            dataset_path=str(dataset),
            act_checkpoint=str(act),
            parent_checkpoint="",
            algorithm="td3",
            robot_type="ffw_sg2_rev1",
        )))

    assert error.value.status_code == 409


def test_offline_rl_start_rejects_unimplemented_algorithms():
    import pytest

    request = app.OfflineRLStartRequest(
        dataset_path="/workspace/lerobot/data",
        act_checkpoint="/workspace/model/lerobot/base",
        parent_checkpoint="",
        algorithm="sac",
        robot_type="ffw_sg2_rev1",
    )
    with pytest.raises(app.HTTPException) as error:
        asyncio.run(app.offline_rl_start(request))

    assert error.value.status_code == 400
    assert "available algorithm is TD3" in error.value.detail


def test_offline_rl_legacy_td3_bc_algorithm_rejects_conflicting_loss_option():
    import pytest

    request = app.OfflineRLStartRequest(
        dataset_path="/workspace/lerobot/data",
        act_checkpoint="/workspace/model/lerobot/base",
        parent_checkpoint="",
        algorithm="td3_bc",
        actor_objective="td3",
        robot_type="ffw_sg2_rev1",
    )
    with pytest.raises(app.HTTPException) as error:
        asyncio.run(app.offline_rl_start(request))

    assert error.value.status_code == 400
    assert "conflicts with actor_objective" in error.value.detail


def test_offline_rl_start_rejects_invalid_td3_epoch_ratio():
    import pytest

    request = app.OfflineRLStartRequest(
        dataset_path="/workspace/lerobot/data",
        act_checkpoint="/workspace/model/lerobot/base",
        algorithm="td3",
        robot_type="ffw_sg2_rev1",
        critic_epochs=10,
        actor_equivalent_epochs=4,
    )
    with pytest.raises(app.HTTPException) as error:
        asyncio.run(app.offline_rl_start(request))

    assert error.value.status_code == 400
    assert "policy_update_period" in error.value.detail


def test_offline_rl_start_request_validates_batch_size():
    import pytest

    common = {
        "dataset_path": "/workspace/lerobot/data",
        "act_checkpoint": "/workspace/model/lerobot/base",
        "algorithm": "td3",
        "robot_type": "ffw_sg2_rev1",
    }
    assert app.OfflineRLStartRequest(**common).batch_size == 4
    assert app.OfflineRLStartRequest(**common, batch_size=64).batch_size == 64
    for value in (0, 65, True, 4.5, "8"):
        with pytest.raises(app.ValidationError):
            app.OfflineRLStartRequest(**common, batch_size=value)


def test_imitation_learning_selects_only_successful_episodes(monkeypatch, tmp_path):
    dataset, _act, _model_root = _offline_rl_test_layout(
        monkeypatch,
        tmp_path,
        episodes=4,
    )
    rows = [
        app.OfflineRLDatasetEpisode(
            index=index,
            frames=10,
            outcome=outcome,
            tasks=["pick jelly"],
        )
        for index, outcome in enumerate(
            ("success", "failure", "success", "unlabeled")
        )
    ]
    monkeypatch.setattr(
        app,
        "_offline_rl_v3_episode_rows",
        lambda _dataset, _count: rows,
    )

    datasets, episodes, selected, excluded = app._imitation_learning_datasets(
        [str(dataset)]
    )

    assert datasets == [dataset.resolve()]
    assert episodes == [[0, 2]]
    assert selected == 2
    assert excluded == 2


def test_imitation_learning_accepts_unlabeled_v3_demos_but_td3_does_not(
    monkeypatch,
    tmp_path,
):
    import pytest

    dataset, _act, _model_root = _offline_rl_test_layout(
        monkeypatch,
        tmp_path,
        episodes=3,
    )
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"].pop("episode_success")
    info_path.write_text(json.dumps(info))
    rows = [
        app.OfflineRLDatasetEpisode(
            index=index,
            frames=10,
            outcome="unlabeled",
            tasks=["pick jelly"],
        )
        for index in range(3)
    ]
    monkeypatch.setattr(
        app,
        "_offline_rl_v3_episode_rows",
        lambda _dataset, _count: rows,
    )

    datasets, episodes, selected, excluded = app._imitation_learning_datasets(
        [str(dataset)]
    )

    assert datasets == [dataset.resolve()]
    assert episodes == [[0, 1, 2]]
    assert selected == 3
    assert excluded == 0
    with pytest.raises(app.HTTPException) as error:
        app._offline_rl_datasets([str(dataset)])
    assert error.value.status_code == 400
    assert "missing episode_success labels" in error.value.detail


def test_imitation_learning_command_is_pinned_offline_and_multi_root(monkeypatch):
    monkeypatch.setattr(app, "_compose_base_cmd", lambda: ["docker", "compose"])
    job = app._ImitationLearningJob(
        job_id="1234567890abcdef",
        dataset_path="/workspace/lerobot/data_epoch_0001/data",
        dataset_paths=[
            "/workspace/lerobot/data_epoch_0001/data",
            "/workspace/lerobot/data_epoch_0002/data",
        ],
        success_episodes=[[0, 2], [1]],
        output_dir="/workspace/model/lerobot/imitation_learning/run",
        episode_count=3,
        excluded_episode_count=2,
        log_path="/tmp/imitation.log",
        total_steps=80_000,
        batch_size=8,
        save_freq=10_000,
        chunk_size=30,
        trainable_groups=["visual_backbone", "action_decoder"],
    )

    command = app._imitation_learning_command(job)

    assert command[:4] == ["docker", "compose", "run", "--rm"]
    assert command[command.index("--pull") + 1] == "never"
    assert command[command.index("--user") + 1] == "1000:1000"
    assert command[command.index("--entrypoint") + 1] == "/lerobot/.venv/bin/python"
    assert "HOME=/tmp" in command
    assert "HF_HUB_OFFLINE=1" in command
    assert "TRANSFORMERS_OFFLINE=1" in command
    assert "HF_DATASETS_OFFLINE=1" in command
    assert "cyclo_brain.algorithm.il.act_bc.training_cli" in command
    assert [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--dataset-root"
    ] == job.dataset_paths
    assert [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--episodes"
    ] == ["0,2", "1"]
    assert command[command.index("--steps") + 1] == "80000"
    assert command[command.index("--batch-size") + 1] == "8"
    assert command[command.index("--save-freq") + 1] == "10000"
    assert command[command.index("--chunk-size") + 1] == "30"
    assert [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--trainable-group"
    ] == ["visual_backbone", "action_decoder"]


def test_imitation_learning_multi_task_dit_contract(monkeypatch, tmp_path):
    import pytest

    _dataset, _act, _model_root = _offline_rl_test_layout(monkeypatch, tmp_path)
    request = app.ImitationLearningStartRequest(
        dataset_path="/workspace/lerobot/data",
        policy_type="multi_task_dit",
        chunk_size=16,
    )
    assert request.policy_type == "multi_task_dit"
    assert app.ImitationLearningStartRequest().policy_type == "act"
    assert app.ImitationLearningStartRequest().chunk_size == 30
    with pytest.raises(app.ValidationError):
        app.ImitationLearningStartRequest(policy_type="diffusion")

    output = app._imitation_learning_output_path(
        "1234567890abcdef",
        80_000,
        policy_type=request.policy_type,
    )
    assert output.name == "multi_task_dit_bc_steps_080000_1234567890ab"

    monkeypatch.setattr(app, "_compose_base_cmd", lambda: ["docker", "compose"])
    job = app._ImitationLearningJob(
        job_id="1234567890abcdef",
        dataset_path="/workspace/lerobot/data",
        dataset_paths=["/workspace/lerobot/data"],
        success_episodes=[[0, 1]],
        output_dir=str(output),
        episode_count=2,
        excluded_episode_count=0,
        log_path="/tmp/imitation.log",
        chunk_size=16,
        policy_type="multi_task_dit",
        task_instruction="  pick up the blue jelly bag  ",
    )
    command = app._imitation_learning_command(job)
    assert "cyclo_brain.algorithm.il.multi_task_dit.training_cli" in command
    assert "cyclo_brain.algorithm.il.act_bc.training_cli" not in command
    assert command[command.index("--chunk-size") + 1] == "16"
    assert command[command.index("--task-instruction") + 1] == (
        "pick up the blue jelly bag"
    )
    assert "HF_HUB_CACHE=/huggingface_hub" in command
    assert "HUGGINGFACE_HUB_CACHE=/huggingface_hub" in command
    assert "TRANSFORMERS_CACHE=/huggingface_hub" in command
    assert "HF_HUB_OFFLINE=1" in command
    assert "TRANSFORMERS_OFFLINE=1" in command
    assert job.message == "Starting MultiTaskDiT imitation learning"
    assert job.trainable_groups == []
    assert "--trainable-group" not in command

    app._imitation_learning_consume_event(job, {"event": "manifest"})
    status = app._imitation_learning_status(job)
    assert status.policy_type == "multi_task_dit"
    assert status.chunk_size == 16
    assert status.task_instruction == "pick up the blue jelly bag"
    assert status.message == "MultiTaskDiT imitation learning is running"


def test_imitation_learning_model_verification_requires_requested_policy_type(
    monkeypatch,
    tmp_path,
):
    _dataset, _act, model_root = _offline_rl_test_layout(monkeypatch, tmp_path)
    output = model_root / "imitation_learning" / "dit_run"
    model = output / "checkpoints" / "000100" / "pretrained_model"
    model.mkdir(parents=True)
    for name, contents in (
        ("config.json", b'{"type":"multi_task_dit"}'),
        ("model.safetensors", b"weights"),
        ("policy_preprocessor.json", b"{}"),
        ("policy_postprocessor.json", b"{}"),
    ):
        (model / name).write_bytes(contents)
    job = app._ImitationLearningJob(
        job_id="dit",
        dataset_path="/workspace/lerobot/data",
        dataset_paths=["/workspace/lerobot/data"],
        success_episodes=[[0]],
        output_dir=str(output),
        episode_count=1,
        excluded_episode_count=0,
        log_path=str(tmp_path / "imitation.log"),
        total_steps=100,
        policy_type="multi_task_dit",
        model_path=str(model),
    )

    assert app._imitation_learning_verified_model(job) is True
    (model / "config.json").write_text('{"type":"act"}')
    assert app._imitation_learning_verified_model(job) is False


def test_imitation_learning_progress_and_verified_completion(monkeypatch, tmp_path):
    _dataset, _act, model_root = _offline_rl_test_layout(monkeypatch, tmp_path)
    output = model_root / "imitation_learning" / "run"
    model = output / "checkpoints" / "000100" / "pretrained_model"
    model.mkdir(parents=True)
    for name, contents in (
        ("config.json", b'{"type":"act"}'),
        ("model.safetensors", b"weights"),
        ("policy_preprocessor.json", b"{}"),
        ("policy_postprocessor.json", b"{}"),
    ):
        (model / name).write_bytes(contents)
    training_state = model.parent / "training_state"
    training_state.mkdir()
    result = json.dumps({
        "event": "result",
        "status": "complete",
        "step": 100,
        "total_steps": 100,
        "percentage": 100.0,
        "loss": 0.3,
        "l1_loss": 0.2,
        "kld_loss": 0.01,
        "eta_seconds": 0.0,
        "checkpoint_path": str(training_state),
        "model_path": str(model),
    })

    class FakeProcess:
        stdout = [result + "\n"]

        @staticmethod
        def wait():
            return 0

    job = app._ImitationLearningJob(
        job_id="job",
        dataset_path=str(app._OFFLINE_RL_DATASET_ROOT / "recording_v30"),
        dataset_paths=[str(app._OFFLINE_RL_DATASET_ROOT / "recording_v30")],
        success_episodes=[[0]],
        output_dir=str(output),
        episode_count=1,
        excluded_episode_count=0,
        log_path=str(tmp_path / "imitation.log"),
        total_steps=100,
        process=FakeProcess(),
    )

    app._monitor_imitation_learning_job(job)

    status = app._imitation_learning_status(job)
    assert status.status == "completed"
    assert status.completed_steps == 100
    assert status.loss == 0.3
    assert status.l1_loss == 0.2
    assert status.kld_loss == 0.01
    assert status.model_path == str(model)
    assert status.checkpoint_path == str(training_state)


@pytest.mark.parametrize(
    ("policy_type", "chunk_size", "training_module", "output_prefix"),
    (
        (
            "act",
            24,
            "cyclo_brain.algorithm.il.act_bc.training_cli",
            "act_bc_steps_080000_",
        ),
        (
            "multi_task_dit",
            16,
            "cyclo_brain.algorithm.il.multi_task_dit.training_cli",
            "multi_task_dit_bc_steps_080000_",
        ),
    ),
)
def test_imitation_learning_start_launches_selected_policy_job(
    monkeypatch,
    tmp_path,
    policy_type,
    chunk_size,
    training_module,
    output_prefix,
):
    dataset, _act, _model_root = _offline_rl_test_layout(
        monkeypatch,
        tmp_path,
        episodes=3,
    )
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"].pop("episode_success")
    info_path.write_text(json.dumps(info))
    launched = {}

    class FakeProcess:
        stdout = []

        @staticmethod
        def wait():
            return 0

    def fake_popen(command, **kwargs):
        launched["command"] = command
        launched["kwargs"] = kwargs
        return FakeProcess()

    class FakeThread:
        def __init__(self, **kwargs):
            launched["thread"] = kwargs

        def start(self):
            launched["thread_started"] = True

    monkeypatch.setattr(app, "_compose_base_cmd", lambda: ["docker", "compose"])
    monkeypatch.setattr(app, "_compose_env", lambda: {"ARCH": "amd64"})
    monkeypatch.setattr(app.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(app.threading, "Thread", FakeThread)
    monkeypatch.setattr(
        app.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="1234567890abcdef1234567890abcdef"),
    )
    monkeypatch.setattr(app, "_OFFLINE_RL_JOB", None)
    monkeypatch.setattr(app, "_IMITATION_LEARNING_JOB", None)
    monkeypatch.setattr(app, "_OFFLINE_RL_DATASET_EDIT_ACTIVE", False)
    monkeypatch.setattr(
        app,
        "_offline_rl_v3_episode_rows",
        lambda _dataset, count: [
            app.OfflineRLDatasetEpisode(
                index=index,
                frames=10,
                outcome="unlabeled",
                tasks=["pick jelly"],
            )
            for index in range(count)
        ],
    )

    response = asyncio.run(app.imitation_learning_start(
        app.ImitationLearningStartRequest(
            dataset_path=str(dataset),
            dataset_paths=[str(dataset)],
            policy_type=policy_type,
            steps=80_000,
            batch_size=8,
            save_freq=10_000,
            chunk_size=chunk_size,
            task_instruction=(
                "pick up the red jelly bag"
                if policy_type == "multi_task_dit"
                else ""
            ),
            trainable_groups=(
                ["action_decoder", "visual_backbone"]
                if policy_type == "act"
                else None
            ),
        )
    ))

    assert response.status == "running"
    assert response.episode_count == 3
    assert response.excluded_episode_count == 0
    assert response.dataset_paths == [str(dataset.resolve())]
    assert response.total_steps == 80_000
    assert response.batch_size == 8
    assert response.chunk_size == chunk_size
    assert response.policy_type == policy_type
    assert response.task_instruction == (
        "pick up the red jelly bag"
        if policy_type == "multi_task_dit"
        else ""
    )
    assert response.trainable_groups == (
        ["visual_backbone", "action_decoder"]
        if policy_type == "act"
        else []
    )
    assert Path(response.output_dir).name.startswith(output_prefix)
    assert launched["thread_started"] is True
    assert launched["kwargs"]["text"] is True
    assert training_module in launched["command"]
    assert launched["command"][
        launched["command"].index("--chunk-size") + 1
    ] == str(chunk_size)
    if policy_type == "multi_task_dit":
        assert launched["command"][
            launched["command"].index("--task-instruction") + 1
        ] == "pick up the red jelly bag"
        assert "--trainable-group" not in launched["command"]
    else:
        assert "--task-instruction" not in launched["command"]
        assert [
            launched["command"][index + 1]
            for index, value in enumerate(launched["command"][:-1])
            if value == "--trainable-group"
        ] == ["visual_backbone", "action_decoder"]
    assert launched["command"][launched["command"].index("--episodes") + 1] == "0,1,2"


def test_imitation_learning_rejects_invalid_act_trainability_and_dit_groups():
    import pytest

    invalid_contracts = (
        ([], "all-frozen"),
        (["action_decoder", "action_decoder"], "duplicates"),
        (["unknown"], "Unknown"),
        (["cvae_encoder"], "CVAE-only"),
    )
    for groups, message in invalid_contracts:
        with pytest.raises(app.HTTPException) as error:
            asyncio.run(app.imitation_learning_start(
                app.ImitationLearningStartRequest(
                    dataset_path="/workspace/lerobot/data",
                    trainable_groups=groups,
                )
            ))
        assert error.value.status_code == 400
        assert message in error.value.detail

    with pytest.raises(app.HTTPException) as error:
        asyncio.run(app.imitation_learning_start(
            app.ImitationLearningStartRequest(
                dataset_path="/workspace/lerobot/data",
                policy_type="multi_task_dit",
                chunk_size=16,
                trainable_groups=["action_decoder"],
            )
        ))
    assert error.value.status_code == 400
    assert "available only for ACT" in error.value.detail


def test_imitation_learning_rejects_fixed_dit_chunk_and_running_td3():
    import pytest

    wrong_dit_chunk = app.ImitationLearningStartRequest(
        dataset_path="/workspace/lerobot/data",
        policy_type="multi_task_dit",
        chunk_size=30,
    )
    with pytest.raises(app.HTTPException) as error:
        asyncio.run(app.imitation_learning_start(wrong_dit_chunk))
    assert error.value.status_code == 400
    assert "MultiTaskDiT" in error.value.detail
    assert "chunk_size=16" in error.value.detail

    running = app._OfflineRLJob(
        job_id="td3",
        dataset_path="/workspace/lerobot/data",
        act_checkpoint="/workspace/model/lerobot/base",
        parent_checkpoint="",
        output_dir="/workspace/model/lerobot/offline_rl/current",
        episode_count=2,
        log_path="/tmp/td3.log",
    )
    original = app._OFFLINE_RL_JOB
    try:
        app._OFFLINE_RL_JOB = running
        with pytest.raises(app.HTTPException) as error:
            asyncio.run(app.imitation_learning_start(
                app.ImitationLearningStartRequest(
                    dataset_path="/workspace/lerobot/data",
                )
            ))
        assert error.value.status_code == 409
        assert "offline RL" in error.value.detail
    finally:
        app._OFFLINE_RL_JOB = original


def test_imitation_learning_stop_signals_only_matching_job(monkeypatch, tmp_path):
    import pytest

    class FakeContainer:
        def __init__(self):
            self.signals = []

        def kill(self, *, signal):
            self.signals.append(signal)

    container = FakeContainer()

    class FakeContainers:
        @staticmethod
        def get(name):
            assert name == "cyclo_imitation_learning_1234567890ab"
            return container

    job = app._ImitationLearningJob(
        job_id="1234567890abcdef1234567890abcdef",
        dataset_path="/workspace/lerobot/data",
        dataset_paths=["/workspace/lerobot/data"],
        success_episodes=[[0]],
        output_dir="/workspace/model/lerobot/imitation_learning/current",
        episode_count=1,
        excluded_episode_count=0,
        log_path=str(tmp_path / "imitation.log"),
    )
    monkeypatch.setattr(app, "_IMITATION_LEARNING_JOB", job)
    monkeypatch.setattr(
        app,
        "_docker_client",
        lambda: SimpleNamespace(containers=FakeContainers()),
    )

    with pytest.raises(app.HTTPException) as error:
        asyncio.run(app.imitation_learning_stop(
            app.ImitationLearningStopRequest(job_id="stale")
        ))
    assert error.value.status_code == 409
    assert container.signals == []

    response = asyncio.run(app.imitation_learning_stop(
        app.ImitationLearningStopRequest(job_id=job.job_id)
    ))
    assert response.status == "running"
    assert response.message == "Stopping ACT imitation learning"
    assert container.signals == ["SIGINT"]
    assert job.stop_requested is True


def _act_td3_critic_warmup_test_job(tmp_path, **overrides):
    job_id = "1234567890abcdef1234567890abcdef"
    actor = tmp_path / "workspace" / "model" / "lerobot" / "base" / "pretrained_model"
    publish_dir = actor / "critic"
    values = {
        "job_id": job_id,
        "dataset_path": "/workspace/lerobot/data_epoch_0001/data",
        "dataset_paths": [
            "/workspace/lerobot/data_epoch_0001/data",
            "/workspace/lerobot/data_epoch_0002/data",
        ],
        "act_checkpoint": str(actor),
        "checkpoint_path": str(publish_dir / "latest.pt"),
        "manifest_path": str(publish_dir / "manifest.json"),
        "run_checkpoint_path": str(publish_dir / "runs" / f"{job_id}.pt"),
        "episode_count": 5,
        "success_count": 3,
        "failure_count": 2,
        "batch_size": 8,
        "log_path": str(tmp_path / "logs" / f"critic_warmup_{job_id}.log"),
    }
    values.update(overrides)
    return app._ACTTD3CriticWarmupJob(**values)


def test_act_td3_critic_warmup_paths_are_policy_local_and_reject_symlinks(
    tmp_path,
):
    actor = tmp_path / "policy" / "pretrained_model"
    actor.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    critic_dir = actor / "critic"
    critic_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(app.HTTPException, match="symbolic link") as error:
        app._act_td3_critic_warmup_paths(actor, "a" * 32)
    assert error.value.status_code == 400

    critic_dir.unlink()
    run_checkpoint, latest, manifest = app._act_td3_critic_warmup_paths(
        actor,
        "a" * 32,
    )
    assert run_checkpoint == actor / "critic" / "runs" / f"{'a' * 32}.pt"
    assert latest == actor / "critic" / "latest.pt"
    assert manifest == actor / "critic" / "manifest.json"

    run_checkpoint.parent.mkdir(parents=True)
    run_checkpoint.write_bytes(b"existing-run")
    with pytest.raises(app.HTTPException, match="already exists") as collision:
        app._act_td3_critic_warmup_paths(actor, "a" * 32)
    assert collision.value.status_code == 409

    run_checkpoint.unlink()
    run_checkpoint.parent.rmdir()
    run_checkpoint.parent.symlink_to(outside, target_is_directory=True)
    with pytest.raises(app.HTTPException, match="runs directory") as runs_error:
        app._act_td3_critic_warmup_paths(actor, "a" * 32)
    assert runs_error.value.status_code == 400


def test_act_td3_critic_warmup_command_is_pinned_multi_root_and_publishes_to_actor(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(app, "_compose_base_cmd", lambda: ["docker", "compose"])
    job = _act_td3_critic_warmup_test_job(tmp_path)

    command = app._act_td3_critic_warmup_command(
        job=job,
        robot_type="ffw_sg2_rev1",
        robot_config="/orchestrator_config/ffw_sg2_rev1_config.yaml",
    )

    assert command[:4] == ["docker", "compose", "run", "--rm"]
    assert command[command.index("--pull") + 1] == "never"
    assert command[command.index("--user") + 1] == "1000:1000"
    assert command[command.index("--entrypoint") + 1] == "/lerobot/.venv/bin/python"
    assert "cyclo_brain.algorithm.rl.act_td3.offline_warmup_cli" in command
    assert "HF_HUB_OFFLINE=1" in command
    assert "TRANSFORMERS_OFFLINE=1" in command
    assert "HF_DATASETS_OFFLINE=1" in command
    assert [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--dataset-root"
    ] == job.dataset_paths
    assert command[command.index("--act-checkpoint") + 1] == job.act_checkpoint
    assert command[command.index("--checkpoint") + 1] == job.run_checkpoint_path
    assert command[command.index("--publish-dir") + 1] == str(
        Path(job.act_checkpoint) / "critic"
    )
    assert command[command.index("--batch-size") + 1] == "8"
    assert command[command.index("--critic-updates") + 1] == "5000"
    assert "--resume" not in command


def test_act_td3_critic_warmup_start_preserves_ordered_payload(monkeypatch, tmp_path):
    first, actor, _model_root = _offline_rl_test_layout(
        monkeypatch,
        tmp_path,
        episodes=2,
    )
    second = first.parent / "data_epoch_0002" / "recording_v30"
    (second / "meta").mkdir(parents=True)
    (second / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0",
        "total_episodes": 3,
        "total_frames": 30,
        "fps": 15,
        "features": {"episode_success": {"dtype": "bool"}},
    }))
    launched = {}

    class FakeProcess:
        stdout = []

        @staticmethod
        def wait():
            return 0

    def fake_popen(command, **kwargs):
        launched["command"] = command
        launched["kwargs"] = kwargs
        return FakeProcess()

    class FakeThread:
        def __init__(self, **kwargs):
            launched["thread"] = kwargs

        def start(self):
            launched["thread_started"] = True

    monkeypatch.setattr(app, "_compose_base_cmd", lambda: ["docker", "compose"])
    monkeypatch.setattr(app, "_compose_env", lambda: {"ARCH": "amd64"})
    monkeypatch.setattr(
        app,
        "_offline_rl_robot_config",
        lambda _robot: "/orchestrator_config/ffw_sg2_rev1_config.yaml",
    )
    monkeypatch.setattr(app, "_reject_running_flow_sde_ppo", lambda: None)
    monkeypatch.setattr(app.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(app.threading, "Thread", FakeThread)
    monkeypatch.setattr(
        app.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="1234567890abcdef1234567890abcdef"),
    )
    monkeypatch.setattr(app, "_OFFLINE_RL_JOB", None)
    monkeypatch.setattr(app, "_ACT_TD3_CRITIC_WARMUP_JOB", None)
    monkeypatch.setattr(app, "_IMITATION_LEARNING_JOB", None)
    monkeypatch.setattr(app, "_OFFLINE_RL_DATASET_EDIT_ACTIVE", False)
    monkeypatch.setattr(
        app,
        "_offline_rl_v3_episode_rows",
        lambda current, count: [
            app.OfflineRLDatasetEpisode(
                index=index,
                frames=10,
                outcome="success" if current == first.resolve() else "failure",
                tasks=["pick jelly"],
            )
            for index in range(count)
        ],
    )

    response = asyncio.run(app.act_td3_critic_warmup_start(
        app.ACTTD3CriticWarmupStartRequest(
            dataset_path=str(first),
            dataset_paths=[str(first), str(second)],
            act_checkpoint=str(actor),
            robot_type="ffw_sg2_rev1",
            batch_size=8,
            critic_updates=1200,
        )
    ))

    assert response.status == "running"
    assert response.dataset_path == str(first.resolve())
    assert response.dataset_paths == [str(first.resolve()), str(second.resolve())]
    assert response.act_checkpoint == str(actor.resolve())
    assert response.episode_count == 5
    assert response.success_count == 2
    assert response.failure_count == 3
    assert response.batch_size == 8
    assert response.total_critic_updates == 1200
    assert response.checkpoint_path == ""
    assert response.manifest_path == ""
    assert launched["thread_started"] is True
    assert launched["kwargs"]["text"] is True
    command = launched["command"]
    assert [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--dataset-root"
    ] == response.dataset_paths
    assert command[command.index("--act-checkpoint") + 1] == response.act_checkpoint
    assert command[command.index("--batch-size") + 1] == "8"
    assert command[command.index("--critic-updates") + 1] == "1200"
    assert command[command.index("--checkpoint") + 1] == str(
        actor / "critic" / "runs" / "1234567890abcdef1234567890abcdef.pt"
    )


def _write_act_td3_critic_warmup_artifact(
    job,
    *,
    checkpoint_contents=b"critic-state",
    manifest_overrides=None,
):
    checkpoint = Path(job.checkpoint_path)
    manifest_path = Path(job.manifest_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(checkpoint_contents)
    manifest = {
        "format": "cyclo_brain.act_td3_critic_manifest/v1",
        "status": "complete",
        "actor_exactly_unchanged": True,
        "completed_critic_updates": job.total_critic_updates,
        "completed_actor_updates": 0,
        "base_policy": {
            "path": job.act_checkpoint,
            "actor_sha256": "a" * 64,
        },
        "training_data": {"dataset_roots": job.dataset_paths},
        "artifact": {
            "format": "cyclo_brain.act_td3_critic/v1",
            "checkpoint_path": "latest.pt",
            "byte_count": len(checkpoint_contents),
            "sha256": hashlib.sha256(checkpoint_contents).hexdigest(),
        },
    }
    if manifest_overrides:
        manifest.update(manifest_overrides)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return checkpoint, manifest_path


def _act_td3_critic_warmup_result(job, **overrides):
    result = {
        "event": "result",
        "status": "complete",
        "completed_critic_updates": job.total_critic_updates,
        "total_critic_updates": job.total_critic_updates,
        "durable_checkpoint_updates": job.total_critic_updates,
        "percentage": 100.0,
        "critic_loss": 0.1,
        "target_mean": 0.2,
        "eta_seconds": 0.0,
        "actor_exactly_unchanged": True,
        "checkpoint_path": job.checkpoint_path,
        "manifest_path": job.manifest_path,
    }
    result.update(overrides)
    return result


def test_act_td3_critic_warmup_events_update_status_and_verified_completion(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(app, "_OFFLINE_RL_LOG_ROOT", tmp_path / "logs")
    job = _act_td3_critic_warmup_test_job(tmp_path)
    _write_act_td3_critic_warmup_artifact(job)

    assert app._act_td3_critic_warmup_consume_event(
        job,
        {"event": "manifest", "total_critic_updates": 5000},
    ) is False
    assert app._act_td3_critic_warmup_consume_event(
        job,
        {
            "event": "progress",
            "completed_critic_updates": 2500,
            "total_critic_updates": 5000,
            "durable_checkpoint_updates": 2000,
            "percentage": 50.0,
            "critic_loss": 0.3,
            "target_mean": 0.4,
            "eta_seconds": 12.5,
            "actor_exactly_unchanged": True,
        },
    ) is False
    status = app._act_td3_critic_warmup_status(job)
    assert status.completed_critic_updates == 2500
    assert status.durable_checkpoint_updates == 2000
    assert status.critic_loss == pytest.approx(0.3)
    assert status.target_mean == pytest.approx(0.4)
    assert status.actor_exactly_unchanged is True

    terminal = _act_td3_critic_warmup_result(job)

    class FakeProcess:
        stdout = [json.dumps(terminal) + "\n"]

        @staticmethod
        def wait():
            return 0

    job.process = FakeProcess()
    app._monitor_act_td3_critic_warmup_job(job)

    status = app._act_td3_critic_warmup_status(job)
    assert status.status == "completed"
    assert status.percentage == 100.0
    assert status.completed_critic_updates == 5000
    assert status.total_critic_updates == 5000
    assert status.durable_checkpoint_updates == 5000
    assert status.actor_exactly_unchanged is True
    assert status.checkpoint_path == job.checkpoint_path
    assert status.manifest_path == job.manifest_path


def test_act_td3_critic_warmup_completes_at_requested_dynamic_update_count(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(app, "_OFFLINE_RL_LOG_ROOT", tmp_path / "logs")
    job = _act_td3_critic_warmup_test_job(
        tmp_path,
        total_critic_updates=1200,
    )
    _write_act_td3_critic_warmup_artifact(job)
    terminal = _act_td3_critic_warmup_result(job)

    class FakeProcess:
        stdout = [json.dumps(terminal) + "\n"]

        @staticmethod
        def wait():
            return 0

    job.process = FakeProcess()
    app._monitor_act_td3_critic_warmup_job(job)

    status = app._act_td3_critic_warmup_status(job)
    assert status.status == "completed"
    assert status.completed_critic_updates == 1200
    assert status.total_critic_updates == 1200
    assert status.durable_checkpoint_updates == 1200


@pytest.mark.parametrize(
    ("terminal_overrides", "manifest_mutation"),
    (
        ({"completed_critic_updates": 4999}, None),
        ({"total_critic_updates": 4999}, None),
        ({"durable_checkpoint_updates": 4999}, None),
        ({"actor_exactly_unchanged": False}, None),
        ({}, "sha256"),
        ({}, "byte_count"),
        ({}, "checkpoint_path"),
    ),
)
def test_act_td3_critic_warmup_completion_fails_closed(
    monkeypatch,
    tmp_path,
    terminal_overrides,
    manifest_mutation,
):
    monkeypatch.setattr(app, "_OFFLINE_RL_LOG_ROOT", tmp_path / "logs")
    job = _act_td3_critic_warmup_test_job(tmp_path)
    _checkpoint, manifest_path = _write_act_td3_critic_warmup_artifact(job)
    if manifest_mutation:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_mutation == "sha256":
            manifest["artifact"]["sha256"] = "0" * 64
        elif manifest_mutation == "byte_count":
            manifest["artifact"]["byte_count"] += 1
        elif manifest_mutation == "checkpoint_path":
            manifest["artifact"]["checkpoint_path"] = "other.pt"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    terminal = _act_td3_critic_warmup_result(job, **terminal_overrides)

    class FakeProcess:
        stdout = [json.dumps(terminal) + "\n"]

        @staticmethod
        def wait():
            return 0

    job.process = FakeProcess()

    app._monitor_act_td3_critic_warmup_job(job)

    assert job.status == "failed"
    assert "contract" in job.message


def test_act_td3_critic_warmup_stopped_run_does_not_publish(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "_OFFLINE_RL_LOG_ROOT", tmp_path / "logs")
    job = _act_td3_critic_warmup_test_job(tmp_path, stop_requested=True)
    stopped = {
        "event": "result",
        "status": "stopped",
        "completed_critic_updates": 1250,
        "total_critic_updates": 5000,
        "durable_checkpoint_updates": 1250,
        "percentage": 25.0,
        "actor_exactly_unchanged": True,
    }

    class FakeProcess:
        stdout = [json.dumps(stopped) + "\n"]

        @staticmethod
        def wait():
            return 0

    job.process = FakeProcess()

    app._monitor_act_td3_critic_warmup_job(job)

    assert job.status == "stopped"
    assert job.stop_confirmed is True
    assert job.result_complete is False
    assert not Path(job.checkpoint_path).exists()
    assert not Path(job.manifest_path).exists()
    status = app._act_td3_critic_warmup_status(job)
    assert status.checkpoint_path == ""
    assert status.manifest_path == ""


def test_act_td3_critic_warmup_stop_rejects_stale_id_and_signals_exact_container(
    monkeypatch,
    tmp_path,
):
    class FakeContainer:
        def __init__(self):
            self.signals = []

        def kill(self, *, signal):
            self.signals.append(signal)

    container = FakeContainer()
    requested_containers = []

    class FakeContainers:
        def get(self, name):
            requested_containers.append(name)
            return container

    job = _act_td3_critic_warmup_test_job(tmp_path)
    monkeypatch.setattr(app, "_ACT_TD3_CRITIC_WARMUP_JOB", job)
    monkeypatch.setattr(
        app,
        "_docker_client",
        lambda: SimpleNamespace(containers=FakeContainers()),
    )
    async def direct_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(app.asyncio, "to_thread", direct_to_thread)

    with pytest.raises(app.HTTPException) as error:
        asyncio.run(app.act_td3_critic_warmup_stop(
            app.ACTTD3CriticWarmupStopRequest(job_id="stale")
        ))
    assert error.value.status_code == 409
    assert requested_containers == []

    status = asyncio.run(app.act_td3_critic_warmup_stop(
        app.ACTTD3CriticWarmupStopRequest(job_id=job.job_id)
    ))
    assert status.status == "running"
    assert status.message == "Stopping ACT-TD3 critic warm-up"
    assert requested_containers == ["cyclo_act_td3_critic_warmup_1234567890ab"]
    assert container.signals == ["SIGINT"]
    assert job.stop_requested is True

    asyncio.run(app.act_td3_critic_warmup_stop(
        app.ACTTD3CriticWarmupStopRequest(job_id=job.job_id)
    ))
    assert container.signals == ["SIGINT"]


def test_act_td3_critic_warmup_conflicts_are_bidirectional(monkeypatch, tmp_path):
    warmup = _act_td3_critic_warmup_test_job(tmp_path)
    monkeypatch.setattr(app, "_OFFLINE_RL_DATASET_EDIT_ACTIVE", False)
    monkeypatch.setattr(app, "_OFFLINE_RL_JOB", None)
    monkeypatch.setattr(app, "_IMITATION_LEARNING_JOB", None)
    monkeypatch.setattr(app, "_ACT_TD3_CRITIC_WARMUP_JOB", warmup)
    monkeypatch.setattr(app, "_reject_running_flow_sde_ppo", lambda: None)

    assert app._flow_sde_ppo_training_conflict() == (
        "Stop ACT-TD3 critic warm-up before starting Flow-SDE PPO"
    )

    with pytest.raises(app.HTTPException, match="critic warm-up") as rl_error:
        asyncio.run(app.offline_rl_start(app.OfflineRLStartRequest(
            dataset_path="/not-reached",
            act_checkpoint="/not-reached",
            robot_type="ffw_sg2_rev1",
        )))
    assert rl_error.value.status_code == 409

    with pytest.raises(app.HTTPException, match="critic warm-up") as il_error:
        asyncio.run(app.imitation_learning_start(app.ImitationLearningStartRequest(
            dataset_path="/not-reached",
        )))
    assert il_error.value.status_code == 409

    monkeypatch.setattr(app, "_ACT_TD3_CRITIC_WARMUP_JOB", None)
    monkeypatch.setattr(app, "_OFFLINE_RL_JOB", SimpleNamespace(status="running"))
    warmup_request = app.ACTTD3CriticWarmupStartRequest(
        dataset_path="/not-reached",
        act_checkpoint="/not-reached",
        robot_type="ffw_sg2_rev1",
    )
    with pytest.raises(app.HTTPException, match="offline RL") as reverse_rl_error:
        asyncio.run(app.act_td3_critic_warmup_start(warmup_request))
    assert reverse_rl_error.value.status_code == 409

    monkeypatch.setattr(app, "_OFFLINE_RL_JOB", None)
    monkeypatch.setattr(
        app,
        "_IMITATION_LEARNING_JOB",
        SimpleNamespace(status="running"),
    )
    with pytest.raises(app.HTTPException, match="imitation-learning") as reverse_il_error:
        asyncio.run(app.act_td3_critic_warmup_start(warmup_request))
    assert reverse_il_error.value.status_code == 409
