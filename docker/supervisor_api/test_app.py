import importlib.util
import asyncio
import os
import subprocess
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
_backend_compose_env = app._backend_compose_env
_host_workspace_dir = app._host_workspace_dir
_require_known_service = app._require_known_service
_resolve_groot_trt_paths = app._resolve_groot_trt_paths
_resolve_rldx_model_host_path = app._resolve_rldx_model_host_path
_trt_status = app._trt_status
_BACKENDS = app._BACKENDS
_USER_SERVICES = app._USER_SERVICES


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

    assert _missing_required_mounts("groot", container) == [
        "/workspace"
    ]


def test_missing_required_mounts_accepts_current_groot_container():
    container = _container_with_mounts(
        "/workspace",
    )

    assert _missing_required_mounts("groot", container) == []


def test_backend_container_image_mismatch_detects_old_container_image():
    class FakeImages:
        def get(self, image):
            assert image == "robotis/groot-zenoh:1.3.2-arm64"
            return SimpleNamespace(id="sha256:new")

    container = SimpleNamespace(attrs={"Image": "sha256:old"})
    spec = {"image": "robotis/groot-zenoh:1.3.2-arm64"}

    assert _backend_container_image_mismatch(
        SimpleNamespace(images=FakeImages()),
        container,
        spec,
    )


def test_backend_container_image_mismatch_accepts_current_container_image():
    class FakeImages:
        def get(self, image):
            assert image == "robotis/groot-zenoh:1.3.2-arm64"
            return SimpleNamespace(id="sha256:new")

    container = SimpleNamespace(attrs={"Image": "sha256:new"})
    spec = {"image": "robotis/groot-zenoh:1.3.2-arm64"}

    assert not _backend_container_image_mismatch(
        SimpleNamespace(images=FakeImages()),
        container,
        spec,
    )


def test_backend_container_stale_reason_detects_workspace_mount_mismatch():
    class FakeImages:
        def get(self, image):
            assert image == "robotis/groot-zenoh:1.3.2-arm64"
            return SimpleNamespace(id="sha256:new")

    container = SimpleNamespace(
        attrs={
            "Image": "sha256:new",
            "Mounts": [
                {
                    "Destination": "/workspace",
                    "Source": "/home/robot/old_workspace",
                },
            ],
        }
    )
    spec = {"image": "robotis/groot-zenoh:1.3.2-arm64"}

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
            assert image == "robotis/groot-zenoh:1.3.2-arm64"
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
            ],
        }
    )
    spec = {"image": "robotis/groot-zenoh:1.3.2-arm64"}

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
    assert compose.count("./workspace:/workspace") == 4
    assert compose.count("./huggingface:/root/.cache/huggingface") == 4


def test_container_helper_does_not_export_workspace_mount_overrides():
    helper = (REPO_ROOT / "docker" / "container.sh").read_text()
    relocate = (REPO_ROOT / "docker" / "relocate_workspace_to_ssd.sh").read_text()
    rsync_options = (
        "rsync -rltHP --omit-dir-times --no-owner --no-group --no-perms"
    )

    assert "export CYCLO_WORKSPACE_DIR" not in helper
    assert "export CYCLO_HUGGINGFACE_DIR" not in helper
    assert rsync_options in helper
    assert rsync_options in relocate
    assert "rsync -aHP" not in helper
    assert "rsync -aHP" not in relocate


def _run_storage_setup(docker_dir, ssd_root, mode="auto", extra_env=None):
    script = r"""
set -e
source <(sed -n '/^ensure_host_dir()/,/^CYCLO_AGENT_SOCKETS_DIR=/p' "$HELPER" | sed '$d')
setup_storage
"""
    env = {
        **os.environ,
        "HELPER": str(REPO_ROOT / "docker" / "container.sh"),
        "SCRIPT_DIR": str(docker_dir),
        "CYCLO_SSD_ROOT": str(ssd_root),
        "CYCLO_STORAGE_MODE": mode,
    }
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", "-c", script],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )


def test_container_helper_auto_migrates_without_overwriting_ssd(tmp_path):
    docker_dir = tmp_path / "docker"
    workspace = docker_dir / "workspace"
    huggingface = docker_dir / "huggingface"
    ssd_root = tmp_path / "ssd"
    workspace.mkdir(parents=True)
    huggingface.mkdir()
    (workspace / "local-only.txt").write_text("old local data")
    (workspace / "conflict.txt").write_text("local conflict")
    (ssd_root / "workspace").mkdir(parents=True)
    (ssd_root / "workspace" / "conflict.txt").write_text("ssd wins")

    result = _run_storage_setup(docker_dir, ssd_root)

    assert result.returncode == 0, result.stderr
    assert workspace.resolve() == ssd_root / "workspace"
    assert huggingface.resolve() == ssd_root / "huggingface"
    assert (ssd_root / "workspace" / "dataset").is_dir()
    assert (ssd_root / "workspace" / "local-only.txt").read_text() == (
        "old local data"
    )
    assert (ssd_root / "workspace" / "conflict.txt").read_text() == "ssd wins"
    backups = list(docker_dir.glob("workspace.local-before-ssd-*"))
    assert len(backups) == 1
    assert (backups[0] / "conflict.txt").read_text() == "local conflict"


def test_container_helper_local_mode_does_not_create_ssd_symlinks(tmp_path):
    docker_dir = tmp_path / "docker"
    workspace = docker_dir / "workspace"
    ssd_root = tmp_path / "ssd"
    workspace.mkdir(parents=True)
    (workspace / "local.txt").write_text("keep local")

    result = _run_storage_setup(docker_dir, ssd_root, mode="local")

    assert result.returncode == 0, result.stderr
    assert not workspace.is_symlink()
    assert (workspace / "local.txt").read_text() == "keep local"
    assert not (ssd_root / "workspace").exists()


def test_container_helper_auto_falls_back_when_ssd_root_unusable(tmp_path):
    docker_dir = tmp_path / "docker"
    workspace = docker_dir / "workspace"
    ssd_root = Path("/proc/cyclo-intelligence-test-root")
    workspace.mkdir(parents=True)
    (workspace / "local.txt").write_text("keep local")

    result = _run_storage_setup(docker_dir, ssd_root)

    assert result.returncode == 0, result.stderr
    assert not workspace.is_symlink()
    assert (workspace / "local.txt").read_text() == "keep local"


def test_container_helper_auto_ignores_unmounted_ssd_mountpoint(tmp_path):
    docker_dir = tmp_path / "docker"
    workspace = docker_dir / "workspace"
    ssd_root = tmp_path / "ssd"
    unmounted = tmp_path / "not-mounted"
    workspace.mkdir(parents=True)
    unmounted.mkdir()
    (workspace / "local.txt").write_text("keep local")

    result = _run_storage_setup(
        docker_dir,
        ssd_root,
        extra_env={"CYCLO_SSD_MOUNTPOINT": str(unmounted)},
    )

    assert result.returncode == 0, result.stderr
    assert not workspace.is_symlink()
    assert (workspace / "local.txt").read_text() == "keep local"
    assert not (ssd_root / "workspace").exists()


def test_container_helper_ssd_mode_creates_missing_ssd_root(tmp_path):
    docker_dir = tmp_path / "docker"
    workspace = docker_dir / "workspace"
    ssd_root = tmp_path / "new-ssd-root"
    workspace.mkdir(parents=True)
    (workspace / "local.txt").write_text("move me")

    result = _run_storage_setup(docker_dir, ssd_root, mode="ssd")

    assert result.returncode == 0, result.stderr
    assert workspace.resolve() == ssd_root / "workspace"
    assert (ssd_root / "workspace" / "local.txt").read_text() == "move me"


def test_container_helper_ssd_mode_fails_when_ssd_root_unusable(tmp_path):
    docker_dir = tmp_path / "docker"
    (docker_dir / "workspace").mkdir(parents=True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sudo = fake_bin / "sudo"
    fake_sudo.write_text("#!/bin/sh\nexit 1\n")
    fake_sudo.chmod(0o755)

    result = _run_storage_setup(
        docker_dir,
        Path("/proc/cyclo-intelligence-test-root"),
        mode="ssd",
        extra_env={"PATH": f"{fake_bin}:{os.environ['PATH']}"},
    )

    assert result.returncode != 0
    assert "SSD storage root is not writable" in result.stderr


def test_container_helper_ssd_mode_fails_when_mountpoint_unmounted(tmp_path):
    docker_dir = tmp_path / "docker"
    workspace = docker_dir / "workspace"
    ssd_root = tmp_path / "ssd"
    unmounted = tmp_path / "not-mounted"
    workspace.mkdir(parents=True)
    unmounted.mkdir()

    result = _run_storage_setup(
        docker_dir,
        ssd_root,
        mode="ssd",
        extra_env={"CYCLO_SSD_MOUNTPOINT": str(unmounted)},
    )

    assert result.returncode != 0
    assert "SSD storage root is not writable" in result.stderr


def test_container_helper_rejects_stale_workspace_symlink(tmp_path):
    docker_dir = tmp_path / "docker"
    docker_dir.mkdir()
    (docker_dir / "workspace").symlink_to(tmp_path / "missing-target")
    ssd_root = tmp_path / "ssd"
    ssd_root.mkdir()

    result = _run_storage_setup(docker_dir, ssd_root)

    assert result.returncode != 0
    assert "is a symlink outside" in result.stderr


def test_relocate_script_migrates_without_overwriting_ssd(tmp_path):
    repo = tmp_path / "repo"
    docker_dir = repo / "docker"
    workspace = docker_dir / "workspace"
    huggingface = docker_dir / "huggingface"
    ssd_root = tmp_path / "ssd"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_docker.write_text("#!/bin/sh\nexit 0\n")
    fake_docker.chmod(0o755)

    workspace.mkdir(parents=True)
    huggingface.mkdir()
    (workspace / "local-only.txt").write_text("local")
    (workspace / "conflict.txt").write_text("local conflict")
    (ssd_root / "workspace").mkdir(parents=True)
    (ssd_root / "workspace" / "conflict.txt").write_text("ssd wins")

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CYCLO_REPO": str(repo),
        "CYCLO_SSD_ROOT": str(ssd_root),
        "CYCLO_STORAGE_USER": str(os.getuid()),
        "CYCLO_STORAGE_GROUP": str(os.getgid()),
    }
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "docker" / "relocate_workspace_to_ssd.sh")],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert workspace.resolve() == ssd_root / "workspace"
    assert huggingface.resolve() == ssd_root / "huggingface"
    assert (ssd_root / "workspace" / "local-only.txt").read_text() == "local"
    assert (ssd_root / "workspace" / "conflict.txt").read_text() == "ssd wins"
    backups = list(docker_dir.glob("workspace.local-before-ssd-*"))
    assert len(backups) == 1
    assert (backups[0] / "conflict.txt").read_text() == "local conflict"


def test_bt_node_is_known_user_service():
    _require_known_service("bt_node")


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
        == f"robotis/groot-zenoh:1.3.2-{app._BACKEND_ARCH}"
    )


def test_rldx_server_backend_targets_policy_server_container():
    assert _BACKENDS["rldx-server"]["service"] == "rldx_policy_server"
    assert _BACKENDS["rldx-server"]["container"] == "rldx_policy_server"
    assert _BACKENDS["rldx-server"]["services"] == []


def test_rldx_server_mounts_are_required():
    container = _container_with_mounts("/opt/RLDX-1")

    assert _missing_required_mounts("rldx-server", container) == ["/model/rldx"]


def test_backend_status_model_exposes_stale_image_status():
    status = app.BackendStatus(
        name="groot",
        image="robotis/groot-zenoh:1.3.2-arm64",
        arch="arm64",
        image_pulled=True,
        image_status="stale",
        container_state="exited",
        raw_state="stale_image",
    )

    assert status.image_status == "stale"
    assert status.arch == "arm64"


def test_backend_logs_tails_and_filters_rldx_traffic(monkeypatch):
    class FakeContainer:
        attrs = {"State": {"Status": "running"}}

        def reload(self):
            return None

        def logs(self, **kwargs):
            assert kwargs["stdout"] is True
            assert kwargs["stderr"] is True
            assert kwargs["tail"] == 12
            assert kwargs["timestamps"] is True
            return (
                b"2026-06-30T00:00:00Z boot noise\n"
                b"2026-06-30T00:00:01Z [RLDX-ZMQ] <- #1 endpoint=ping\n"
                b"2026-06-30T00:00:02Z random model log\n"
                b"2026-06-30T00:00:03Z RLDX chunk T=16 D=17 remote_ms=183.2\n"
            )

    class FakeContainers:
        def get(self, name):
            assert name == "rldx_runtime"
            return FakeContainer()

    monkeypatch.setattr(
        app,
        "_docker_client",
        lambda: SimpleNamespace(containers=FakeContainers()),
    )

    result = asyncio.run(app.backend_logs("rldx", tail=3, traffic_only=True))

    assert result.container_state == "running"
    assert result.lines == [
        "2026-06-30T00:00:01Z [RLDX-ZMQ] <- #1 endpoint=ping",
        "2026-06-30T00:00:03Z RLDX chunk T=16 D=17 remote_ms=183.2",
    ]


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


def test_rldx_model_path_maps_workspace_path_to_host_mount(monkeypatch):
    monkeypatch.setattr(
        app,
        "_host_workspace_dir",
        lambda: "/mnt/ssd/cyclo_intelligence/workspace",
    )

    assert (
        _resolve_rldx_model_host_path("/workspace/model/rldx/task0047")
        == "/mnt/ssd/cyclo_intelligence/workspace/model/rldx/task0047"
    )


def test_rldx_server_compose_env_uses_requested_model_path(monkeypatch):
    monkeypatch.setattr(
        app,
        "_host_workspace_dir",
        lambda: "/mnt/ssd/cyclo_intelligence/workspace",
    )
    monkeypatch.setattr(
        app,
        "_host_huggingface_dir",
        lambda: "/mnt/ssd/cyclo_intelligence/huggingface",
    )

    env = _backend_compose_env(
        "rldx-server",
        app.BackendActionRequest(
            model_path="/workspace/model/rldx/task0047",
        ),
    )

    assert (
        env["RLDX_MODEL_PATH"]
        == "/mnt/ssd/cyclo_intelligence/workspace/model/rldx/task0047"
    )
    assert env["RLDX_SERVER_MODEL_PATH"] == "/model/rldx"
