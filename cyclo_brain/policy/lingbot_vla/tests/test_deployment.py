from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[4]


def test_upstream_source_edits_preserve_dependency_build_cache():
    dockerfile = (ROOT / "cyclo_brain/policy/lingbot_vla/Dockerfile.amd64").read_text()
    requirements = dockerfile.index("COPY cyclo_brain/policy/lingbot_vla/lingbot-vla-v2/requirements.txt ")
    install = dockerfile.index("-r /tmp/lingbot-requirements.txt")
    flash = dockerfile.index("flash-attn==2.8.3")
    source = dockerfile.index("COPY cyclo_brain/policy/lingbot_vla/lingbot-vla-v2/ ")
    editable = dockerfile.index("--no-deps -e /lingbot-vla-v2")
    assert requirements < install < flash < source < editable


def test_deployment_keeps_worker_model_and_command_paths_separate():
    compose = yaml.safe_load((ROOT / "docker/docker-compose.yml").read_text())
    worker = compose["services"]["lingbot_vla"]
    assert worker["container_name"] == "lingbot_vla_server"
    assert worker["restart"] == "no"
    assert worker["build"]["dockerfile"].startswith("cyclo_brain/policy/lingbot_vla/")
    assert "../cyclo_brain/policy/common/runtime:/policy_runtime:ro" in worker["volumes"]
    assert "../cyclo_brain/policy/lingbot_vla/lingbot_vla_engine:/app/lingbot_vla_engine:ro" in worker["volumes"]
    dockerfile = (ROOT / "cyclo_brain/policy/lingbot_vla/Dockerfile.amd64").read_text()
    assert "POLICY_ENGINE_MODULE=lingbot_vla_engine" in dockerfile
    assert "python -m compileall -q /lingbot-vla-v2/deploy/lingbot_vla_v2_policy.py" in dockerfile
    assert "ENTRYPOINT [\"/init\"]" in dockerfile
    for arch in ("amd64", "arm64"):
        main = (ROOT / f"docker/Dockerfile.{arch}").read_text()
        assert "COPY cyclo_brain/policy/lingbot_vla/manifest.yaml" in main
        ignore = (ROOT / f"docker/Dockerfile.{arch}.dockerignore").read_text()
        assert "!cyclo_brain/policy/lingbot_vla/manifest.yaml" in ignore
    helper = (ROOT / "docker/container.sh").read_text()
    assert "start-lingbot-vla) start_policy lingbot_vla" in helper
