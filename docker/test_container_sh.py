import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


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
        "case \" $* \" in\n"
        "  *' config --services '*)\n"
        "    printf '%s\\n' cyclo_intelligence lerobot groot\n"
        "    ;;\n"
        "  *' config --format json '*)\n"
        "    printf '%s\\n' '{\"services\":{\"lerobot\":{\"container_name\":\"lerobot_server\",\"image\":\"robotis/lerobot-zenoh:1.4.1-arm64\"},\"groot\":{\"container_name\":\"groot_server\",\"image\":\"robotis/groot-zenoh:1.3.5-arm64\"}}}'\n"
        "    ;;\n"
        "esac\n"
        "if [ \"$1\" = image ] && [ \"$2\" = inspect ]; then\n"
        "  printf '%s\\n' 'sha256:current'\n"
        "fi\n"
        "exit 0\n"
    )
    docker_stub.chmod(0o755)
    return log_path


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
        "    printf '%s\\n' '{\"services\":{\"lerobot\":{\"container_name\":\"lerobot_server\",\"image\":\"robotis/lerobot-zenoh:1.4.1-arm64\"},\"groot\":{\"container_name\":\"groot_server\",\"image\":\"robotis/groot-zenoh:1.3.5-arm64\"}}}'\n"
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
    assert any("up -d --build groot" in call for call in docker_calls)


def test_start_policy_uses_manifest_backed_runtime(tmp_path):
    docker_dir = _copy_container_script(tmp_path)
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
        "up -d --build lerobot" in call
        for call in log_path.read_text().splitlines()
    )


def test_start_does_not_remove_policy_container_with_stale_workspace_mount(tmp_path):
    log_path = tmp_path / "docker.log"
    docker_stub = tmp_path / "docker"
    docker_stub.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$DOCKER_STUB_LOG\"\n"
        "case \" $* \" in\n"
        "  *' config --format json '*)\n"
        "    printf '%s\\n' '{\"services\":{\"lerobot\":{\"image\":\"robotis/lerobot-zenoh:1.4.1-arm64\"},\"groot\":{\"image\":\"robotis/groot-zenoh:1.3.5-arm64\"}}}'\n"
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
        "    printf '%s\\n' '{\"services\":{\"lerobot\":{\"container_name\":\"lerobot_server\",\"image\":\"robotis/lerobot-zenoh:1.4.1-arm64\"},\"groot\":{\"container_name\":\"groot_server\",\"image\":\"robotis/groot-zenoh:1.3.5-arm64\"}}}'\n"
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
    assert any(
        "pull --ignore-pull-failures lerobot" in call
        for call in docker_calls
    )
    assert not any("pull --ignore-pull-failures groot" in call for call in docker_calls)
    assert any(
        call == "rm -f lerobot_server"
        for call in docker_calls
    )
