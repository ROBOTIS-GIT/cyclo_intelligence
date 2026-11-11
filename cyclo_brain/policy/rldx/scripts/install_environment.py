"""Install upstream's Blackwell dependency set into an isolated Python 3.10 env.

Reads upstream pixi.toml, not a second hand-maintained dependency list.
FlashAttention uses its matching wheel when available. Source builds require
a CUDA 12.8 toolkit. Does not change any existing model environment.
"""

import argparse
from pathlib import Path
import subprocess
import tempfile
import tomllib


def install(repo, environment):
    config = tomllib.loads((repo / "pixi.toml").read_text())
    deps = config["feature"]["rldx"]["pypi-dependencies"]
    subprocess.run(["uv", "venv", "--python", "3.10", str(environment)], check=True)
    python = str(environment / "bin/python")
    subprocess.run(["uv", "pip", "install", "--python", python,
                    "--index-url", "https://download.pytorch.org/whl/cu128",
                    "torch==2.8.0", "torchvision==0.23.0"], check=True)
    with tempfile.TemporaryDirectory() as directory:
        requirements = Path(directory) / "requirements.txt"
        requirements.write_text("\n".join(
            name + (value if value != "*" else "")
            for name, value in deps.items() if isinstance(value, str)
        ) + "\nsetuptools\nwheel\nninja\npackaging\n")
        subprocess.run(["uv", "pip", "install", "--python", python, "-r", str(requirements)], check=True)
    subprocess.run(["uv", "pip", "install", "--python", python,
                    "flash-attn==2.8.3", "--no-build-isolation", "--no-deps"], check=True)
    subprocess.run(["uv", "pip", "install", "--python", python, "--no-deps", "-e", str(repo)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("environment", type=Path)
    args = parser.parse_args()
    if args.environment.exists():
        parser.error("Refusing to replace an existing environment")
    install(args.repo, args.environment)
