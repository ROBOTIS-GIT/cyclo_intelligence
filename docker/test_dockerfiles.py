import os
from pathlib import Path
import subprocess
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_main_dockerfiles_install_compose_v2():
    for dockerfile in (
        REPO_ROOT / "docker" / "Dockerfile.arm64",
        REPO_ROOT / "docker" / "Dockerfile.amd64",
    ):
        contents = dockerfile.read_text()

        assert "docker-compose-v2" in contents, (
            f"{dockerfile.name} must install docker-compose-v2 so "
            "supervisor_api can recreate policy containers with docker compose"
        )


def test_main_compose_mounts_shared_s6_runner():
    contents = (REPO_ROOT / "docker" / "docker-compose.yml").read_text()

    assert (
        "./s6-services/common/ros2_service_run.sh:"
        "/usr/local/lib/s6-services/ros2_service_run.sh:ro"
    ) in contents


def test_interactive_bashrc_includes_simple_ros_zenoh_block():
    dockerfiles = (
        REPO_ROOT / "docker" / "Dockerfile.arm64",
        REPO_ROOT / "docker" / "Dockerfile.amd64",
        REPO_ROOT / "cyclo_brain" / "policy" / "lerobot" / "Dockerfile.arm64",
        REPO_ROOT / "cyclo_brain" / "policy" / "lerobot" / "Dockerfile.amd64",
        REPO_ROOT / "cyclo_brain" / "policy" / "groot" / "Dockerfile.arm64",
        REPO_ROOT / "cyclo_brain" / "policy" / "groot" / "Dockerfile.amd64",
    )
    required = (
        "export ROS_DOMAIN_ID=30",
        "export RMW_IMPLEMENTATION=rmw_zenoh_cpp",
        "export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true'",
        "# export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true;mode=\\\"client\\\";connect/endpoints=[\\\"tcp/192.168.60.139:7447\\\"]'",
    )
    removed = (
        "export ZENOH_ROUTER_IP=${ZENOH_ROUTER_IP:-127.0.0.1}",
        "export ZENOH_ROUTER_PORT=${ZENOH_ROUTER_PORT:-7447}",
        "if [ \"${ZENOH_ROUTER_IP}\" = \"127.0.0.1\" ]",
        "export ZENOH_TRANSPORT_SHM_ENABLED=${ZENOH_TRANSPORT_SHM_ENABLED:-true}",
        "export ZENOH_SHM_ENABLED=${ZENOH_SHM_ENABLED:-true}",
    )

    for dockerfile in dockerfiles:
        contents = dockerfile.read_text()
        for expected in required:
            assert expected in contents, f"{dockerfile} is missing {expected}"
        for unexpected in removed:
            assert unexpected not in contents, f"{dockerfile} still has {unexpected}"


def test_dockerfiles_prepend_ros_zenoh_block_before_existing_bashrc():
    dockerfiles = (
        REPO_ROOT / "docker" / "Dockerfile.arm64",
        REPO_ROOT / "docker" / "Dockerfile.amd64",
        REPO_ROOT / "cyclo_brain" / "policy" / "lerobot" / "Dockerfile.arm64",
        REPO_ROOT / "cyclo_brain" / "policy" / "lerobot" / "Dockerfile.amd64",
        REPO_ROOT / "cyclo_brain" / "policy" / "groot" / "Dockerfile.arm64",
        REPO_ROOT / "cyclo_brain" / "policy" / "groot" / "Dockerfile.amd64",
    )

    for dockerfile in dockerfiles:
        contents = dockerfile.read_text()
        write_block_index = contents.index("> /tmp/cyclo_bashrc")
        existing_bashrc_index = contents.index("cat /root/.bashrc >> /tmp/cyclo_bashrc")
        assert write_block_index < existing_bashrc_index, (
            f"{dockerfile} must write Cyclo env before appending the base bashrc"
        )


def test_ros_zenoh_runtime_env_file_is_not_referenced_by_images_or_s6():
    paths = (
        REPO_ROOT / "docker" / "Dockerfile.arm64",
        REPO_ROOT / "docker" / "Dockerfile.amd64",
        REPO_ROOT / "cyclo_brain" / "policy" / "lerobot" / "Dockerfile.arm64",
        REPO_ROOT / "cyclo_brain" / "policy" / "lerobot" / "Dockerfile.amd64",
        REPO_ROOT / "cyclo_brain" / "policy" / "groot" / "Dockerfile.arm64",
        REPO_ROOT / "cyclo_brain" / "policy" / "groot" / "Dockerfile.amd64",
        REPO_ROOT / "docker" / "s6-services" / "common" / "ros2_service_run.sh",
        REPO_ROOT / "cyclo_brain" / "policy" / "common" / "s6-services" / "main-runtime" / "run",
        REPO_ROOT / "cyclo_brain" / "policy" / "common" / "s6-services" / "engine-process" / "run",
        REPO_ROOT / "cyclo_brain" / "policy" / "groot" / "s6-services" / "main-runtime" / "run",
        REPO_ROOT / "cyclo_brain" / "policy" / "groot" / "s6-services" / "engine-process" / "run",
    )

    for path in paths:
        contents = path.read_text()
        assert "CYCLO_ROS_ENV_FILE" not in contents, f"{path} references the old env file hook"
        assert "ros_zenoh.env" not in contents, f"{path} references the old runtime env file"


def test_s6_services_run_through_interactive_bashrc_shell():
    paths = (
        REPO_ROOT / "docker" / "s6-services" / "common" / "ros2_service_run.sh",
        REPO_ROOT / "cyclo_brain" / "policy" / "common" / "s6-services" / "main-runtime" / "run",
        REPO_ROOT / "cyclo_brain" / "policy" / "common" / "s6-services" / "engine-process" / "run",
        REPO_ROOT / "cyclo_brain" / "policy" / "groot" / "s6-services" / "main-runtime" / "run",
        REPO_ROOT / "cyclo_brain" / "policy" / "groot" / "s6-services" / "engine-process" / "run",
    )

    for path in paths:
        contents = path.read_text()
        assert "bash -ic" in contents, f"{path} does not run through an interactive bashrc shell"
        assert "source /root/.bashrc" not in contents, f"{path} manually sources /root/.bashrc"
        assert "export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-30}" in contents
        assert (
            "export ZENOH_CONFIG_OVERRIDE=${ZENOH_CONFIG_OVERRIDE:-transport/shared_memory/enabled=true}"
            in contents
        )
        assert "export ZENOH_ROUTER_IP=${ZENOH_ROUTER_IP:-127.0.0.1}" not in contents


def test_compose_does_not_override_ros_zenoh_runtime_env():
    contents = (REPO_ROOT / "docker" / "docker-compose.yml").read_text()

    removed_entries = (
        "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-30}",
        "RMW_IMPLEMENTATION=rmw_zenoh_cpp",
        "CYCLO_ROS_ENV_FILE=",
        "ZENOH_ROUTER_IP=127.0.0.1",
        "ZENOH_ROUTER_PORT=7447",
        "ZENOH_CONFIG_OVERRIDE=transport/shared_memory/enabled=true",
        "ZENOH_TRANSPORT_SHM_ENABLED=true",
        "ZENOH_SHM_ENABLED=true",
    )
    for entry in removed_entries:
        assert entry not in contents


def test_cyclo_runtime_ports_avoid_physical_ai_defaults():
    compose = (REPO_ROOT / "docker" / "docker-compose.yml").read_text()
    nginx_run = (REPO_ROOT / "docker" / "s6-services" / "nginx" / "run").read_text()
    supervisor_run = (
        REPO_ROOT / "docker" / "s6-services" / "supervisor_api" / "run"
    ).read_text()
    launch = (
        REPO_ROOT / "orchestrator" / "launch" / "orchestrator_bringup.launch.py"
    ).read_text()
    orchestrator = (
        REPO_ROOT / "orchestrator" / "orchestrator" / "orchestrator_node.py"
    ).read_text()
    ui_config = (
        REPO_ROOT / "orchestrator" / "ui" / "public" / "cyclo-config.js"
    ).read_text()

    expected_defaults = (
        "CYCLO_UI_PORT=${CYCLO_UI_PORT:-7080}",
        "CYCLO_ROSBRIDGE_PORT=${CYCLO_ROSBRIDGE_PORT:-7090}",
        "CYCLO_VIDEO_SERVER_PORT=${CYCLO_VIDEO_SERVER_PORT:-7082}",
        "CYCLO_WEB_VIDEO_SERVER_PORT=${CYCLO_WEB_VIDEO_SERVER_PORT:-7085}",
        "CYCLO_SUPERVISOR_API_PORT=${CYCLO_SUPERVISOR_API_PORT:-7100}",
    )
    for expected in expected_defaults:
        assert expected in compose

    assert 'UI_PORT="$(port_or_default ' in nginx_run
    assert "CYCLO_UI_PORT:-}\" 7080" in nginx_run
    assert "CYCLO_ROSBRIDGE_PORT:-}\" 7090" in nginx_run
    assert "CYCLO_VIDEO_SERVER_PORT:-}\" 7082" in nginx_run
    assert "CYCLO_WEB_VIDEO_SERVER_PORT:-}\" 7085" in nginx_run
    assert "CYCLO_SUPERVISOR_API_PORT:-}\" 7100" in nginx_run
    assert 'PORT="${CYCLO_SUPERVISOR_API_PORT:-7100}"' in supervisor_run
    assert "_env_int('CYCLO_ROSBRIDGE_PORT', '7090')" in launch
    assert "_env_int('CYCLO_WEB_VIDEO_SERVER_PORT', '7085')" in launch
    assert "_env_int('CYCLO_VIDEO_SERVER_PORT', '7082')" in orchestrator
    assert "uiPort: 7080" in ui_config
    assert "rosbridgePort: 7090" in ui_config
    assert "videoServerPort: 7082" in ui_config
    assert "webVideoServerPort: 7085" in ui_config
    assert "supervisorApiPort: 7100" in ui_config

    for old_default in (
        "CYCLO_UI_PORT=${CYCLO_UI_PORT:-80}",
        "CYCLO_ROSBRIDGE_PORT=${CYCLO_ROSBRIDGE_PORT:-9090}",
        "CYCLO_VIDEO_SERVER_PORT=${CYCLO_VIDEO_SERVER_PORT:-8082}",
        "CYCLO_WEB_VIDEO_SERVER_PORT=${CYCLO_WEB_VIDEO_SERVER_PORT:-8085}",
        "CYCLO_SUPERVISOR_API_PORT=${CYCLO_SUPERVISOR_API_PORT:-8100}",
    ):
        assert old_default not in compose


def test_nginx_runtime_port_rewrite_keeps_data_api_on_video_server():
    nginx_run = (REPO_ROOT / "docker" / "s6-services" / "nginx" / "run").read_text()
    nginx_conf = (REPO_ROOT / "orchestrator" / "ui" / "nginx.conf").read_text()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        conf_path = tmp_path / "default.conf"
        config_path = tmp_path / "cyclo-config.js"
        script_path = tmp_path / "run-nginx-test.sh"

        conf_path.write_text(nginx_conf)
        test_script = nginx_run.split('exec nginx -g "daemon off;"')[0]
        test_script = test_script.replace(
            "/etc/nginx/conf.d/default.conf",
            str(conf_path),
        )
        test_script = test_script.replace(
            "/usr/share/nginx/html/cyclo-config.js",
            str(config_path),
        )
        script_path.write_text(test_script)

        subprocess.run(
            ["bash", str(script_path)],
            check=True,
            text=True,
            capture_output=True,
            env={
                **os.environ,
                "CYCLO_UI_PORT": "7080",
                "CYCLO_ROSBRIDGE_PORT": "7090",
                "CYCLO_VIDEO_SERVER_PORT": "7082",
                "CYCLO_WEB_VIDEO_SERVER_PORT": "7085",
                "CYCLO_SUPERVISOR_API_PORT": "7100",
            },
        )

        rewritten = conf_path.read_text()
        assert "listen 7080;" in rewritten
        assert "location /api/" in rewritten
        assert "proxy_pass http://127.0.0.1:7100/;" in rewritten
        assert "location /data-api/" in rewritten
        assert "proxy_pass http://127.0.0.1:7082/;" in rewritten
        assert rewritten.count("proxy_pass http://127.0.0.1:7100/;") == 1
        assert rewritten.count("proxy_pass http://127.0.0.1:7082/;") == 1

        runtime_config = config_path.read_text()
        assert "rosbridgePort: 7090" in runtime_config
        assert "videoServerPort: 7082" in runtime_config


def test_policy_compose_keeps_image_defaults_in_images():
    compose = (REPO_ROOT / "docker" / "docker-compose.yml").read_text()
    duplicated_entries = (
        "ZENOH_SDK_PATH=/zenoh_sdk",
        "ROBOT_CLIENT_SDK_PATH=/robot_client_sdk",
        "ACTION_CHUNK_PROCESSING_SDK_PATH=/action_chunk_processing_sdk",
        "POLICY_BACKEND=lerobot",
        "POLICY_ENGINE_MODULE=lerobot_engine",
        "POLICY_BACKEND=groot",
        "POLICY_ENGINE_MODULE=groot_engine",
        "GROOT_TRT_ENABLED=false",
        "CONTROL_HZ=100",
        "INFERENCE_HZ=15",
        "TARGET_CHUNK_SIZE=none",
        "REFILL_MARGIN_S=0.2",
        "REFILL_LATENCY_WARMUP_SAMPLES=1",
        "REFILL_LATENCY_SAMPLE_MAX_S=2.0",
        "HF_HOME=/root/.cache/huggingface",
        "HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub",
        "TRANSFORMERS_CACHE=/root/.cache/huggingface/hub",
    )
    for entry in duplicated_entries:
        assert entry not in compose

    policy_dockerfiles = (
        REPO_ROOT / "cyclo_brain" / "policy" / "lerobot" / "Dockerfile.arm64",
        REPO_ROOT / "cyclo_brain" / "policy" / "lerobot" / "Dockerfile.amd64",
        REPO_ROOT / "cyclo_brain" / "policy" / "groot" / "Dockerfile.arm64",
        REPO_ROOT / "cyclo_brain" / "policy" / "groot" / "Dockerfile.amd64",
    )
    for dockerfile in policy_dockerfiles:
        contents = dockerfile.read_text()
        assert "ENV ZENOH_SDK_PATH=/zenoh_sdk" in contents
        assert "ENV ROBOT_CLIENT_SDK_PATH=/robot_client_sdk" in contents
        assert "ENV ACTION_CHUNK_PROCESSING_SDK_PATH=/action_chunk_processing_sdk" in contents
