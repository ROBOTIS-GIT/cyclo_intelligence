#!/usr/bin/env python3
"""Select a working CUDA driver before importing policy dependencies.

Only the probe subprocess imports torch. The launcher can therefore try the
host driver and the image's CUDA compat driver without reusing CUDA state.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys


STATUS_DIR = Path("/run/cyclo-gpu")
SERVICES = {"engine-process": "engine_process", "main-runtime": "main_runtime"}
HOST_DRIVER_DIRS = (
    "/opt/nvidia/l4t-gpu-libs/nvgpu",
    "/usr/lib/aarch64-linux-gnu/tegra",
    "/usr/lib/aarch64-linux-gnu/nvidia",
)


def requested_device(env):
    device = env.get("CYCLO_POLICY_DEVICE", "cuda")
    if device not in ("cuda", "cpu"):
        raise ValueError("CYCLO_POLICY_DEVICE must be cuda or cpu")
    return device


def policy_device(torch):
    """Also enforce the GPU requirement when the engine is run directly."""
    device = requested_device(os.environ)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; refusing to run inference on CPU. "
            "Check the [gpu-runtime] startup diagnostics, or explicitly set "
            "CYCLO_POLICY_DEVICE=cpu for CPU execution."
        )
    return torch.device(device)


def _is_compat(path):
    return Path(path).name == "compat" or Path(path).resolve().name == "compat"


def driver_candidates(env):
    mode = env.get("CYCLO_CUDA_DRIVER", "auto")
    if mode not in ("auto", "host", "compat"):
        raise ValueError("CYCLO_CUDA_DRIVER must be auto, host or compat")
    if requested_device(env) == "cpu":
        return [("cpu", dict(env))]

    # Non-Jetson containers keep NVIDIA Container Toolkit's environment.
    if mode == "auto" and not Path("/etc/nv_tegra_release").is_file():
        return [("image", dict(env))]

    original = env.get("LD_LIBRARY_PATH", "").split(":")
    # Empty entries mean the working directory, not a library directory.
    retained = [p for p in original if p and not _is_compat(p)]
    host = [p for p in HOST_DRIVER_DIRS if (Path(p) / "libcuda.so.1").is_file()]
    compat = [
        p for p in original
        if p and _is_compat(p) and (Path(p) / "libcuda.so.1").is_file()
    ]
    default_compat = "/usr/local/cuda/compat"
    if (Path(default_compat) / "libcuda.so.1").is_file():
        compat.append(default_compat)

    candidates = []
    for profile, prefix in (("host", host), ("compat", compat)):
        if mode not in ("auto", profile) or (profile == "compat" and not prefix):
            continue
        candidate = dict(env)
        paths = list(dict.fromkeys(prefix + retained))
        if paths:
            candidate["LD_LIBRARY_PATH"] = ":".join(paths)
        else:
            candidate.pop("LD_LIBRARY_PATH", None)  # Use the system loader cache.
        candidates.append((profile, candidate))
    if not candidates:
        raise RuntimeError("No CUDA compat driver found in this image")
    return candidates


def probe():
    import torch

    device = policy_device(torch)
    with torch.inference_mode():
        matrix = torch.ones((32, 32), device=device)
        result = matrix @ matrix
        if device.type == "cuda":
            torch.cuda.synchronize()
        if not bool((result == 32).all().item()):
            raise RuntimeError("GPU probe returned an incorrect matrix product")
    driver = sorted({
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/libcuda.so" in line
    })
    return {
        "device": device.type,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "driver_libraries": driver,
    }


def select_environment(env):
    timeout = float(env.get("CYCLO_GPU_PROBE_TIMEOUT_S", "30"))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("CYCLO_GPU_PROBE_TIMEOUT_S must be finite and positive")
    failures = []
    for profile, candidate in driver_candidates(env):
        print(f"[gpu-runtime] Testing {profile} driver", file=sys.stderr, flush=True)
        try:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "probe"],
                env=candidate, capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip())
            details = json.loads(result.stdout)
            if details["device"] != requested_device(env):
                raise RuntimeError("Probe did not use the requested device")
        except (subprocess.TimeoutExpired, RuntimeError, ValueError, KeyError) as error:
            message = f"{profile}: {error}"
            failures.append(message)
            print(f"[gpu-runtime] Probe failed: {message}", file=sys.stderr, flush=True)
            continue
        details.update(profile=profile, library_path=candidate.get("LD_LIBRARY_PATH", ""))
        print(f"[gpu-runtime] Selected {json.dumps(details)}", file=sys.stderr, flush=True)
        return candidate, details
    raise RuntimeError("No working policy device; server was not started. " + " | ".join(failures))


def process_start_time(pid):
    # comm (field 2) can contain spaces and parentheses; starttime is field 22.
    fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
    if fields[0] in ("Z", "X"):
        raise ValueError("Policy process has exited")
    return fields[19]


def launch(service):
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    status = STATUS_DIR / f"{service}.json"
    status.unlink(missing_ok=True)
    env, details = select_environment(dict(os.environ))
    # exec retains this PID and start time. An s6 restart cannot reuse a stale
    # success record while a new launcher is still testing CUDA.
    details.update(pid=os.getpid(), start_time=process_start_time(os.getpid()))
    temporary = status.with_suffix(".tmp")
    temporary.write_text(json.dumps(details))
    temporary.replace(status)
    try:
        os.execve(sys.executable, [sys.executable, "-m", SERVICES[service]], env)
    except OSError:
        status.unlink(missing_ok=True)
        raise


def healthy():
    for service in SERVICES:
        try:
            record = json.loads((STATUS_DIR / f"{service}.json").read_text())
            if record["device"] not in ("cuda", "cpu"):
                return False
            if process_start_time(record["pid"]) != record["start_time"]:
                return False
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("probe")
    commands.add_parser("health")
    launcher = commands.add_parser("launch")
    launcher.add_argument("service", choices=SERVICES)
    args = parser.parse_args()
    try:
        if args.command == "probe":
            print(json.dumps(probe()))
        elif args.command == "health":
            return 0 if healthy() else 1
        else:
            launch(args.service)
    except Exception as error:
        print(f"[gpu-runtime] {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
