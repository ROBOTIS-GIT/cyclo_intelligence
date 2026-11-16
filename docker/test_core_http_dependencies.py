"""Offline guards plus a real resolver regression using CYCLO_HTTP_WHEELHOUSE."""

import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


DOCKER_DIR = Path(__file__).resolve().parent


def http_requirements(arch):
    contents = (DOCKER_DIR / f"Dockerfile.{arch}").read_text()
    command = next(
        line for line in contents.replace("\\\n", " ").splitlines()
        if line.startswith("RUN ") and "pip install" in line and "huggingface_hub" in line
    )
    return [arg for arg in shlex.split(command)
            if arg.startswith(("huggingface_hub", "httpcore"))]


@pytest.mark.parametrize("arch", ["amd64", "arm64"])
def test_core_resolves_httpcore_together_with_hub(arch):
    assert "httpcore>=1.0.9,<2" in http_requirements(arch)


@pytest.mark.skipif(not os.environ.get("CYCLO_HTTP_WHEELHOUSE"),
                    reason="Set CYCLO_HTTP_WHEELHOUSE to downloaded regression wheels")
@pytest.mark.parametrize("arch", ["amd64", "arm64"])
def test_core_http_install_order_passes_pip_check(arch, tmp_path):
    subprocess.run([sys.executable, "-I", "-m", "venv", str(tmp_path / "venv")], check=True)
    python = str(tmp_path / "venv/bin/python")
    env = {key: value for key, value in os.environ.items() if not key.startswith("PIP_")}
    env["PIP_CONFIG_FILE"] = os.devnull

    def pip(*args):
        result = subprocess.run(
            [python, "-I", "-m", "pip", "--disable-pip-version-check", "--no-cache-dir", *args],
            env=env, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    def install(*requirements):
        pip("install", "--no-index", "--find-links",
            os.environ["CYCLO_HTTP_WHEELHOUSE"], *requirements)

    # Reproduce the inherited HTTP stack and the Hub release from the ARM log.
    install("httpx==0.28.1", "httpcore==1.0.8", "h11==0.14.0")
    install(*http_requirements(arch), "huggingface_hub==2.4.0", "httpx2==2.16.0")
    # cyclo_manager requirements observed in core-build-host.log, lines 966-992.
    install("fastapi==0.128.0", "uvicorn[standard]==0.40.0", "pydantic==2.12.5",
            "pyyaml==6.0.1", "docker==7.1.0", "requests==2.32.5",
            "requests-unixsocket==0.4.1", "httpx==0.27.2", "psutil>=5.9.0")
    install("-r", str(DOCKER_DIR / "supervisor_api/requirements.txt"))
    assert "No broken requirements found" in pip("check")
    subprocess.run(
        [python, "-I", "-c", "from huggingface_hub import HfApi; "
         "import httpx, httpx2; HfApi(); httpx.Client().close(); httpx2.Client().close()"],
        env=env, check=True,
    )
