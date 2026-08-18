import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

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
    return dataset, act, model_root


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
    assert "HF_HUB_OFFLINE=1" in command
    assert "--allow-partial-round" not in command
    assert command[command.index("--critic-epochs") + 1] == "10"
    assert command[command.index("--actor-equivalent-epochs") + 1] == "5"
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
        "model_path": model_path,
    })

    assert complete is True
    assert job.percentage == 100.0
    assert job.completed_epochs == 10
    assert job.completed_critic_updates == 320
    assert job.completed_actor_updates == 160
    assert job.model_path == model_path


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
    )

    app._monitor_offline_rl_job(job)

    assert job.status == "completed"
    assert job.returncode == 0
    assert app._offline_rl_status(job).model_path == str(model)


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

    response = asyncio.run(app.offline_rl_start(app.OfflineRLStartRequest(
        dataset_path=str(dataset),
        act_checkpoint=str(act),
        parent_checkpoint="",
        algorithm="td3",
        robot_type="ffw_sg2_rev1",
    )))

    assert response.status == "running"
    assert response.episode_count == 50
    assert response.round_index == 1
    assert response.round_episode_count == 50
    assert response.critic_epochs == 10
    assert response.actor_equivalent_epochs == 5
    assert response.model_path == ""
    assert launched["thread_started"] is True
    assert launched["kwargs"]["text"] is True
    assert launched["command"][launched["command"].index("--pull") + 1] == "never"
    assert "/lerobot/.venv/bin/python" in launched["command"]


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
    assert "only TD3" in error.value.detail


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
