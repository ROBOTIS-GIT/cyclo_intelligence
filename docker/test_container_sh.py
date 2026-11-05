import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_merged_compose_preserves_runtime_navigation_and_release_contracts():
    services = yaml.safe_load((REPO_ROOT / "docker/docker-compose.yml").read_text())["services"]
    main = services["cyclo_intelligence"]
    assert main["image"] == "robotis/cyclo-intelligence:1.4.4"
    assert services["lerobot"]["image"] == "robotis/lerobot-zenoh:1.4.2-${ARCH:-arm64}"
    for name, service in services.items():
        assert service["build"]["context"] == ".."
        if name != "cyclo_intelligence":
            assert service["pull_policy"] == "never"
        targets = [
            mount["target"] if isinstance(mount, dict) else mount.split(":")[1]
            for mount in service["volumes"]
        ]
        assert len(targets) == len(set(targets)), name
    for setting in (
        "CYCLO_POLICY_ROOT=/opt/cyclo/policy",
        "CYCLO_SUPERVISOR_API_COMPOSE_FILE=/opt/cyclo/docker/docker-compose.yml",
        "CYCLO_NAVIGATION_DATA_DIR=${CYCLO_NAVIGATION_DATA_DIR:-/workspace/navigation}",
        "CYCLO_BT_TREES_DIR=${CYCLO_BT_TREES_DIR:-/workspace/bt/trees}",
        "CYCLO_BT_EXAMPLE_TREES_DIR=${CYCLO_BT_EXAMPLE_TREES_DIR:-/root/ros2_ws/install/orchestrator/share/orchestrator/bt/trees}",
    ):
        assert setting in main["environment"]
    for mount in (
        "./workspace:/workspace",
        "../shared/shared/robot_configs:/orchestrator_config:ro",
        "./s6-services/common/ros2_service_run.sh:/usr/local/lib/s6-services/ros2_service_run.sh:ro",
    ):
        assert mount in main["volumes"]


def test_lerobot_arm64_does_not_restore_inherited_compat_driver_path():
    contents = (REPO_ROOT / "cyclo_brain/policy/lerobot/Dockerfile.arm64").read_text()
    overrides = [line for line in contents.splitlines() if line.startswith("ENV LD_LIBRARY_PATH=")]
    assert overrides == ["ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64"]


def test_ci_keeps_converter_catalog_runtime_and_worker_build_suites():
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text())
    commands = "\n".join(
        step.get("run", "")
        for job in workflow["jobs"].values()
        for step in job["steps"]
    )
    for suite in (
        "cyclo_data/tests/converter",
        "orchestrator/tests/data_processing/test_rosbag_to_lerobot_converter.py",
        "cyclo_brain/policy/common/catalog/tests",
        "cyclo_brain/policy/common/runtime/tests",
        "docker/test_container_sh.py",
        "docker/test_dockerfiles.py",
        "docker/supervisor_api/test_app.py",
        "docker/supervisor_api/test_worker_build.py",
    ):
        assert suite in commands


def test_python_ci_initializes_only_the_pinned_lerobot_submodule():
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["python-tests"]["steps"]
    checkout = next(step for step in steps if step.get("uses", "").startswith("actions/checkout@"))
    assert not checkout.get("with", {}).get("submodules", False)
    commands = [
        shlex.split(line)
        for step in steps for line in step.get("run", "").splitlines()
        if line.strip().startswith("git submodule ")
    ]
    assert commands == [[
        "git", "submodule", "update", "--init", "--",
        "cyclo_brain/policy/lerobot/lerobot",
    ]]


def test_python_ci_installs_torch_and_torchvision_together_from_cpu_index():
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["python-tests"]["steps"]
    commands = [
        shlex.split(line)
        for step in steps for line in step.get("run", "").splitlines()
        if "download.pytorch.org" in line
    ]
    assert len(commands) == 1
    command = commands[0]
    assert command[:4] == ["python", "-m", "pip", "install"]
    assert command[command.index("--index-url") + 1] == "https://download.pytorch.org/whl/cpu"
    assert {"torch", "torchvision"} <= set(command)


def _copy_container_script(tmp_path):
    docker_dir = tmp_path / "docker"
    docker_dir.mkdir()
    shutil.copy2(REPO_ROOT / "docker" / "container.sh", docker_dir / "container.sh")
    (docker_dir / "container.sh").chmod(0o755)
    for runtime in ("lerobot", "groot"):
        target = tmp_path / "cyclo_brain" / "policy" / runtime
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            REPO_ROOT / "cyclo_brain" / "policy" / runtime / "manifest.yaml",
            target / "manifest.yaml",
        )
    return docker_dir


def _write_start_stub(tmp_path):
    log_path = tmp_path / "docker.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    docker_stub = bin_dir / "docker"
    docker_stub.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$DOCKER_STUB_LOG\"\n"
        "if [ \"$1 $2\" = 'image inspect' ] && [ \"$IMAGE_MISSING\" = 1 ]; then exit 1; fi\n"
        "if [ \"$1\" = buildx ] && [ \"$BUILDX_MISSING\" = 1 ]; then exit 1; fi\n"
        "case \" $* \" in *' build '*) exit \"${BUILD_EXIT:-0}\" ;; esac\n"
        "case \" $* \" in\n"
        "  *' config --services '*)\n"
        "    printf '%s\\n' cyclo_intelligence lerobot groot\n"
        "    ;;\n"
        "  *' config --format json '*)\n"
        "    printf '%s\\n' '{\"services\":{\"lerobot\":{\"container_name\":\"lerobot_server\",\"image\":\"robotis/lerobot-zenoh:1.4.2-arm64\"},\"groot\":{\"container_name\":\"groot_server\",\"image\":\"robotis/groot-zenoh:1.3.5-arm64\"}}}'\n"
        "    ;;\n"
        "esac\n"
        "if [ \"$1\" = image ] && [ \"$2\" = inspect ]; then\n"
        "  printf '%s\\n' 'sha256:current'\n"
        "fi\n"
        "exit 0\n"
    )
    docker_stub.chmod(0o755)
    return log_path


def _init_lerobot_build_sources(tmp_path, runtime="lerobot"):
    """Provide real Git provenance without fetching the model submodule."""
    policy = tmp_path / "cyclo_brain/policy" / runtime
    source = policy / "upstream" if runtime != "lerobot" else policy / "lerobot"
    source.mkdir()
    (source / "uv.lock").write_text("version = 1\n")
    shutil.copytree(REPO_ROOT / "cyclo_brain/policy/lerobot/dependency_profile",
                    policy / "dependency_profile")
    for arch in ("amd64", "arm64"):
        (policy / f"Dockerfile.{arch}").write_text("FROM scratch\n")
    relative = source.relative_to(tmp_path).as_posix()
    sdk = tmp_path / "cyclo_brain/sdk/zenoh_ros2_sdk"
    sdk.mkdir(parents=True)
    (sdk / "README.md").write_text("SDK fixture\n")
    (tmp_path / ".gitmodules").write_text(
        f'[submodule "upstream"]\n\tpath = {relative}\n\turl = https://example.invalid/upstream\n'
        '[submodule "sdk"]\n\tpath = cyclo_brain/sdk/zenoh_ros2_sdk\n\turl = https://example.invalid/sdk\n'
    )

    def git(root, *args):
        return subprocess.run(
            ["git", "-C", str(root), "-c", "user.name=Test", "-c",
             "user.email=test@example.invalid", "-c", "commit.gpgsign=false", *args],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    for root in (source, sdk, tmp_path):
        git(root, "init", "-q")
        git(root, "add", ".")
        git(root, "commit", "-qm", "Build fixture")
    git(tmp_path, "submodule", "init")
    return source, git


def _write_enter_stub(tmp_path, running_container):
    log_path = tmp_path / "docker.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    docker_stub = bin_dir / "docker"
    docker_stub.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$DOCKER_STUB_LOG\"\n"
        "case \" $* \" in\n"
        "  *' config --services '*) printf '%s\\n' cyclo_intelligence lerobot groot; exit 0 ;;\n"
        "  *' config --format json '*) printf '%s\\n' '{\"services\":{\"lerobot\":{\"container_name\":\"lerobot_server\"},\"groot\":{\"container_name\":\"groot_server\"}}}'; exit 0 ;;\n"
        "esac\n"
        "if [ \"$1\" = ps ]; then\n"
        f"  printf '%s\\n' '{running_container}'\n"
        "fi\n"
        "exit 0\n"
    )
    docker_stub.chmod(0o755)
    return log_path


def _stub_env(tmp_path, log_path):
    return {
        **os.environ,
        "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
        "DOCKER_STUB_LOG": str(log_path),
        "CYCLO_AGENT_SOCKETS_DIR": str(tmp_path / "agent_sockets"),
    }


def test_build_policy_does_not_start_pull_or_remove_containers(tmp_path):
    docker_dir = _copy_container_script(tmp_path)
    _init_lerobot_build_sources(tmp_path)
    (tmp_path / "cyclo_brain/policy/lerobot/Dockerfile.amd64").write_text("FROM scratch\n")
    (tmp_path / "cyclo_brain/policy/lerobot/Dockerfile.arm64").write_text("FROM scratch\n")
    log = _write_start_stub(tmp_path)
    result = subprocess.run(
        [str(docker_dir / "container.sh"), "build-policy", "lerobot"],
        env=_stub_env(tmp_path, log), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    assert any("build lerobot" in call for call in calls)
    assert not any("pull " in call or "up -d" in call or "rm -f" in call for call in calls)


def test_start_policy_reuses_local_image_without_pull(tmp_path):
    docker_dir = _copy_container_script(tmp_path)
    log = _write_start_stub(tmp_path)
    result = subprocess.run(
        [str(docker_dir / "container.sh"), "start-groot"],
        env=_stub_env(tmp_path, log), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    assert not any("pull --" in call or " build groot" in call for call in calls)
    assert any("up -d --no-build --pull never groot" in call for call in calls)


@pytest.mark.parametrize("build_exit", [0, 7])
def test_missing_image_builds_before_start_and_build_failure_never_starts(tmp_path, build_exit):
    docker_dir = _copy_container_script(tmp_path)
    _init_lerobot_build_sources(tmp_path, "groot")
    log = _write_start_stub(tmp_path)
    result = subprocess.run(
        [str(docker_dir / "container.sh"), "start-groot"],
        env={**_stub_env(tmp_path, log), "IMAGE_MISSING": "1", "BUILD_EXIT": str(build_exit)},
        capture_output=True, text=True,
    )
    assert result.returncode == build_exit, result.stderr
    calls = log.read_text().splitlines()
    build_index = next(i for i, call in enumerate(calls) if " build groot" in call)
    starts = [i for i, call in enumerate(calls) if "up -d" in call]
    assert (len(starts) == 1 and starts[0] > build_index) if build_exit == 0 else not starts
    assert not any("pull --" in call for call in calls)


@pytest.mark.parametrize("env, message", [
    ({"BUILDX_MISSING": "1"}, "Buildx is required"),
    ({"CYCLO_EXPECTED_WORKER_IMAGE": "wrong:tag"}, "configuration differs"),
])
def test_worker_build_preflight_failure_does_not_build(tmp_path, env, message):
    docker_dir = _copy_container_script(tmp_path)
    _init_lerobot_build_sources(tmp_path, "groot")
    log = _write_start_stub(tmp_path)
    result = subprocess.run(
        [str(docker_dir / "container.sh"), "build-policy", "groot"],
        env={**_stub_env(tmp_path, log), **env}, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert message in result.stderr
    assert " build groot" not in log.read_text()


def test_build_policy_rejects_missing_architecture_dockerfile(tmp_path):
    docker_dir = _copy_container_script(tmp_path)
    log = _write_start_stub(tmp_path)
    result = subprocess.run(
        [str(docker_dir / "container.sh"), "build-policy", "groot"],
        env=_stub_env(tmp_path, log), capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "Dockerfile" in result.stderr
    assert not any(" build groot" in call for call in log.read_text().splitlines())


def test_ui_commands_use_external_non_root_node_builder():
    script = (REPO_ROOT / "docker" / "container.sh").read_text()

    assert "run_ui_npm()" not in script
    assert "run_ui_npm_external ci --legacy-peer-deps" in script
    assert "run_ui_npm_external run build" in script
    assert "run_ui_npm_in_main" not in script
    assert '--user "$(id -u):$(id -g)"' in script
    assert '-v "${dir}:/ui"' in script


def test_start_does_not_create_legacy_runtime_env_file_from_default(tmp_path):
    docker_dir = _copy_container_script(tmp_path)
    log_path = _write_start_stub(tmp_path)

    result = subprocess.run(
        [str(docker_dir / "container.sh"), "start"],
        cwd=tmp_path,
        env=_stub_env(tmp_path, log_path),
        check=True,
        text=True,
        capture_output=True,
    )

    legacy_runtime_env = docker_dir / "workspace" / "config" / "ros_zenoh.env"
    assert not legacy_runtime_env.exists()
    assert "Created ROS/Zenoh runtime env from default" not in result.stdout


def test_start_creates_repo_local_mount_directories(tmp_path):
    docker_dir = _copy_container_script(tmp_path)
    log_path = _write_start_stub(tmp_path)

    subprocess.run(
        [str(docker_dir / "container.sh"), "start"],
        cwd=tmp_path,
        env=_stub_env(tmp_path, log_path),
        check=True,
        text=True,
        capture_output=True,
    )

    assert (docker_dir / "workspace" / "dataset").is_dir()
    assert (docker_dir / "workspace" / "rosbag2").is_dir()
    assert (docker_dir / "workspace" / "lerobot").is_dir()
    assert (docker_dir / "workspace" / "model" / "lerobot").is_dir()
    assert (docker_dir / "workspace" / "model" / "groot").is_dir()
    assert (docker_dir / "huggingface").is_dir()


def test_start_creates_checkpoint_root_for_new_manifest_runtime(tmp_path):
    docker_dir = _copy_container_script(tmp_path)
    runtime = tmp_path / "cyclo_brain" / "policy" / "sample"
    runtime.mkdir(parents=True)
    (runtime / "manifest.yaml").write_text(
        "runtime:\n  checkpoint_root: /workspace/model/sample/nested\n",
        encoding="utf-8",
    )
    log_path = _write_start_stub(tmp_path)

    subprocess.run(
        [str(docker_dir / "container.sh"), "start"],
        cwd=tmp_path,
        env=_stub_env(tmp_path, log_path),
        check=True,
        text=True,
        capture_output=True,
    )

    assert (docker_dir / "workspace" / "model" / "sample" / "nested").is_dir()


def test_start_preserves_existing_legacy_runtime_env_file_without_touching(tmp_path):
    docker_dir = _copy_container_script(tmp_path)
    legacy_runtime_env = docker_dir / "workspace" / "config" / "ros_zenoh.env"
    legacy_runtime_env.parent.mkdir(parents=True)
    legacy_runtime_env.write_text("export ZENOH_ROUTER_IP=192.168.60.139\n")
    log_path = _write_start_stub(tmp_path)

    result = subprocess.run(
        [str(docker_dir / "container.sh"), "start"],
        cwd=tmp_path,
        env=_stub_env(tmp_path, log_path),
        check=True,
        text=True,
        capture_output=True,
    )

    assert legacy_runtime_env.read_text() == "export ZENOH_ROUTER_IP=192.168.60.139\n"
    assert "Created ROS/Zenoh runtime env from default" not in result.stdout


def test_enter_lerobot_uses_plain_bash(tmp_path):
    docker_dir = _copy_container_script(tmp_path)
    log_path = _write_enter_stub(tmp_path, "lerobot_server")

    subprocess.run(
        [str(docker_dir / "container.sh"), "enter-lerobot"],
        cwd=tmp_path,
        env=_stub_env(tmp_path, log_path),
        check=True,
        text=True,
        capture_output=True,
    )

    legacy_runtime_env = docker_dir / "workspace" / "config" / "ros_zenoh.env"
    docker_calls = log_path.read_text().splitlines()
    assert not legacy_runtime_env.exists()
    assert "exec -it lerobot_server bash" in docker_calls
    assert not any("bash -lc" in call for call in docker_calls)


def test_start_pulls_main_image_only(tmp_path):
    log_path = tmp_path / "docker.log"
    docker_stub = tmp_path / "docker"
    docker_stub.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$DOCKER_STUB_LOG\"\n"
        "case \" $* \" in\n"
        "  *' config --format json '*)\n"
        "    printf '%s\\n' '{\"services\":{\"lerobot\":{\"container_name\":\"lerobot_server\",\"image\":\"robotis/lerobot-zenoh:1.4.2-arm64\"},\"groot\":{\"container_name\":\"groot_server\",\"image\":\"robotis/groot-zenoh:1.3.5-arm64\"}}}'\n"
        "    ;;\n"
        "esac\n"
        "if [ \"$1\" = image ] && [ \"$2\" = inspect ]; then\n"
        "  printf '%s\\n' 'sha256:current'\n"
        "fi\n"
        "exit 0\n"
    )
    docker_stub.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "DOCKER_STUB_LOG": str(log_path),
        "CYCLO_AGENT_SOCKETS_DIR": str(tmp_path / "agent_sockets"),
    }

    subprocess.run(
        [str(REPO_ROOT / "docker" / "container.sh"), "start"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    docker_calls = log_path.read_text().splitlines()
    assert any(
        "pull --ignore-pull-failures cyclo_intelligence" in call
        for call in docker_calls
    )
    assert not any("pull --ignore-pull-failures lerobot" in call for call in docker_calls)
    assert not any("pull --ignore-pull-failures groot" in call for call in docker_calls)


def test_start_groot_build_skips_prebuilt_pull(tmp_path):
    docker_dir = _copy_container_script(tmp_path)
    _init_lerobot_build_sources(tmp_path, "groot")
    log_path = _write_start_stub(tmp_path)

    subprocess.run(
        [str(docker_dir / "container.sh"), "start-groot", "--build"],
        cwd=tmp_path,
        env=_stub_env(tmp_path, log_path),
        check=True,
        text=True,
        capture_output=True,
    )

    docker_calls = log_path.read_text().splitlines()
    assert not any("pull --ignore-pull-failures groot" in call for call in docker_calls)
    assert any("build groot" in call for call in docker_calls)
    assert any("up -d --no-build --pull never groot" in call for call in docker_calls)


def test_start_policy_uses_manifest_backed_runtime(tmp_path):
    docker_dir = _copy_container_script(tmp_path)
    _init_lerobot_build_sources(tmp_path)
    log_path = _write_start_stub(tmp_path)

    subprocess.run(
        [str(docker_dir / "container.sh"), "start-policy", "lerobot", "--build"],
        cwd=tmp_path,
        env=_stub_env(tmp_path, log_path),
        check=True,
        text=True,
        capture_output=True,
    )

    assert any(
        "up -d --no-build --pull never lerobot" in call
        for call in log_path.read_text().splitlines()
    )


def test_lerobot_build_rejects_a_checkout_different_from_the_parent_commit(tmp_path):
    docker_dir = _copy_container_script(tmp_path)
    source, git = _init_lerobot_build_sources(tmp_path)
    git(source, "commit", "--allow-empty", "-qm", "Unrecorded model revision")
    log_path = _write_start_stub(tmp_path)
    result = subprocess.run(
        [str(docker_dir / "container.sh"), "start-policy", "lerobot", "--build"],
        cwd=tmp_path, env=_stub_env(tmp_path, log_path), capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "submodules at the commits recorded by Cyclo" in result.stderr
    assert not any("up -d --build" in call for call in log_path.read_text().splitlines())


def test_start_does_not_remove_policy_container_with_stale_workspace_mount(tmp_path):
    log_path = tmp_path / "docker.log"
    docker_stub = tmp_path / "docker"
    docker_stub.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$DOCKER_STUB_LOG\"\n"
        "case \" $* \" in\n"
        "  *' config --format json '*)\n"
        "    printf '%s\\n' '{\"services\":{\"lerobot\":{\"image\":\"robotis/lerobot-zenoh:1.4.2-arm64\"},\"groot\":{\"image\":\"robotis/groot-zenoh:1.3.5-arm64\"}}}'\n"
        "    exit 0\n"
        "    ;;\n"
        "esac\n"
        "if [ \"$1\" = image ] && [ \"$2\" = inspect ]; then\n"
        "  printf '%s\\n' 'sha256:current'\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = inspect ] && [ \"$2\" = -f ] && [ \"$3\" = '{{.Image}}' ]; then\n"
        "  printf '%s\\n' 'sha256:current'\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = inspect ] && [ \"$2\" = -f ]; then\n"
        "  case \"$3\" in\n"
        "    *'.Destination \"/workspace\"'*)\n"
        "      if [ \"$4\" = lerobot_server ]; then\n"
        "        printf '%s\\n' '/old/workspace'\n"
        "      else\n"
        "        printf '%s\\n' \"$EXPECTED_WORKSPACE_DIR\"\n"
        "      fi\n"
        "      exit 0\n"
        "      ;;\n"
        "  esac\n"
        "fi\n"
        "exit 0\n"
    )
    docker_stub.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "DOCKER_STUB_LOG": str(log_path),
        "CYCLO_AGENT_SOCKETS_DIR": str(tmp_path / "agent_sockets"),
        "EXPECTED_WORKSPACE_DIR": str(REPO_ROOT / "docker" / "workspace"),
    }

    subprocess.run(
        [str(REPO_ROOT / "docker" / "container.sh"), "start"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    docker_calls = log_path.read_text().splitlines()
    assert "rm -f lerobot_server" not in docker_calls
    assert "rm -f groot_server" not in docker_calls


def test_start_lerobot_removes_stale_workspace_mount(tmp_path):
    log_path = tmp_path / "docker.log"
    docker_stub = tmp_path / "docker"
    docker_stub.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$DOCKER_STUB_LOG\"\n"
        "case \" $* \" in\n"
        "  *' config --format json '*)\n"
        "    printf '%s\\n' '{\"services\":{\"lerobot\":{\"container_name\":\"lerobot_server\",\"image\":\"robotis/lerobot-zenoh:1.4.2-arm64\"},\"groot\":{\"container_name\":\"groot_server\",\"image\":\"robotis/groot-zenoh:1.3.5-arm64\"}}}'\n"
        "    exit 0\n"
        "    ;;\n"
        "esac\n"
        "if [ \"$1\" = image ] && [ \"$2\" = inspect ]; then\n"
        "  printf '%s\\n' 'sha256:current'\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = inspect ] && [ \"$2\" = -f ] && [ \"$3\" = '{{.Image}}' ]; then\n"
        "  printf '%s\\n' 'sha256:current'\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = inspect ] && [ \"$2\" = -f ]; then\n"
        "  case \"$3\" in\n"
        "    *'.Destination \"/workspace\"'*)\n"
        "      printf '%s\\n' '/old/workspace'\n"
        "      exit 0\n"
        "      ;;\n"
        "  esac\n"
        "fi\n"
        "exit 0\n"
    )
    docker_stub.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "DOCKER_STUB_LOG": str(log_path),
        "CYCLO_AGENT_SOCKETS_DIR": str(tmp_path / "agent_sockets"),
    }

    subprocess.run(
        [str(REPO_ROOT / "docker" / "container.sh"), "start-lerobot"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    docker_calls = log_path.read_text().splitlines()
    assert not any(
        "pull --ignore-pull-failures lerobot" in call
        for call in docker_calls
    )
    assert not any("pull --ignore-pull-failures groot" in call for call in docker_calls)
    assert any(
        call == "rm -f lerobot_server"
        for call in docker_calls
    )
