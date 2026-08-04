import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
INSTALLER = REPO_ROOT / "install.sh"


def _write_stub_commands(tmp_path, hostname_value="dev-pc", ssd_mounted=True):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    git_stub = bin_dir / "git"
    git_stub.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$INSTALL_TEST_LOG\"\n"
        "for dest do :; done\n"
        "mkdir -p \"$dest/docker\"\n"
        "cat > \"$dest/docker/container.sh\" <<'EOS'\n"
        "#!/bin/sh\n"
        "printf '%s\\n' \"container.sh $*\" >> \"$INSTALL_TEST_LOG\"\n"
        "EOS\n"
        "chmod +x \"$dest/docker/container.sh\"\n",
    )
    git_stub.chmod(0o755)

    hostname_stub = bin_dir / "hostname"
    hostname_stub.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' '{hostname_value}'\n",
    )
    hostname_stub.chmod(0o755)

    mountpoint_stub = bin_dir / "mountpoint"
    mountpoint_stub.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"mountpoint $*\" >> \"$INSTALL_TEST_LOG\"\n"
        "if [ \"$1\" = -q ] && [ \"$2\" = \"$CYCLO_INSTALL_SSD_ROOT\" ]; then\n"
        f"  exit {0 if ssd_mounted else 1}\n"
        "fi\n"
        "if [ \"$1\" = -q ] && [ \"$2\" = \"$INSTALL_TEST_HOME_MOUNT\" ] && [ -f \"$INSTALL_TEST_BIND_MOUNTED\" ]; then\n"
        "  exit 0\n"
        "fi\n"
        "exit 1\n",
    )
    mountpoint_stub.chmod(0o755)

    mount_stub = bin_dir / "mount"
    mount_stub.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"mount $*\" >> \"$INSTALL_TEST_LOG\"\n"
        "if [ \"$1\" = \"$INSTALL_TEST_HOME_MOUNT\" ]; then\n"
        "  touch \"$INSTALL_TEST_BIND_MOUNTED\"\n"
        "  exit 0\n"
        "fi\n"
        "exit 1\n",
    )
    mount_stub.chmod(0o755)

    sudo_stub = bin_dir / "sudo"
    sudo_stub.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"sudo $*\" >> \"$INSTALL_TEST_LOG\"\n"
        "exec \"$@\"\n",
    )
    sudo_stub.chmod(0o755)

    return bin_dir


def _run_install(tmp_path, args=None, hostname_value="dev-pc", ssd_mounted=True):
    args = args or []
    home = tmp_path / "home"
    home.mkdir()
    ssd = tmp_path / "ssd"
    ssd.mkdir()
    fstab = tmp_path / "fstab"
    fstab.write_text("# test fstab\n")
    log_path = tmp_path / "install.log"
    bin_dir = _write_stub_commands(tmp_path, hostname_value, ssd_mounted)
    bind_marker = tmp_path / "bind-mounted"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(home),
        "INSTALL_TEST_LOG": str(log_path),
        "INSTALL_TEST_HOME_MOUNT": str(home / "cyclo_intelligence"),
        "INSTALL_TEST_BIND_MOUNTED": str(bind_marker),
        "CYCLO_INSTALL_REPO_URL": "https://example.invalid/cyclo.git",
        "CYCLO_INSTALL_SSD_ROOT": str(ssd),
        "CYCLO_INSTALL_FSTAB": str(fstab),
    }

    result = subprocess.run(
        ["bash", str(INSTALLER), *args],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )
    return result, home, ssd, log_path


def test_local_mode_installs_under_home_and_prints_start_command(tmp_path):
    result, home, _ssd, log_path = _run_install(tmp_path, ["--local"])

    assert result.returncode == 0, result.stderr
    assert (home / "cyclo_intelligence").is_dir()
    assert "clone --recurse-submodules --branch main" in log_path.read_text()
    assert "mountpoint " not in log_path.read_text()
    assert "container.sh start" not in log_path.read_text()
    assert "./docker/container.sh start" in result.stdout


def test_auto_robot_hostname_installs_on_ssd_bind_mounts_home_and_does_not_start(tmp_path):
    result, home, ssd, log_path = _run_install(
        tmp_path,
        hostname_value="ffw-SNPR48A1106",
    )

    assert result.returncode == 0, result.stderr
    install_dir = ssd / "cyclo_intelligence"
    home_mount = home / "cyclo_intelligence"
    assert install_dir.is_dir()
    assert home_mount.is_dir()
    assert not home_mount.is_symlink()
    assert "container.sh start" not in log_path.read_text()
    assert "./docker/container.sh start" in result.stdout
    assert f"{install_dir} {home_mount} none bind,nofail,x-systemd.requires-mounts-for={ssd} 0 0" in (
        tmp_path / "fstab"
    ).read_text()
    log = log_path.read_text()
    assert f"mount {home_mount}" in log


def test_robot_mode_requires_mounted_ssd(tmp_path):
    result, home, ssd, _log_path = _run_install(
        tmp_path,
        ["--robot"],
        ssd_mounted=False,
    )

    assert result.returncode != 0
    assert "requires mounted SSD" in result.stderr
    assert not (ssd / "cyclo_intelligence").exists()
    assert not (home / "cyclo_intelligence").exists()


def test_robot_mode_uses_sudo_when_ssd_parent_is_not_writable(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    ssd = tmp_path / "ssd"
    ssd.mkdir()
    fstab = tmp_path / "fstab"
    fstab.write_text("# test fstab\n")
    log_path = tmp_path / "install.log"
    bind_marker = tmp_path / "bind-mounted"
    bin_dir = _write_stub_commands(
        tmp_path,
        hostname_value="ffw-SNPR48A1106",
    )
    install_dir = ssd / "cyclo_intelligence"

    mkdir_stub = bin_dir / "mkdir"
    mkdir_stub.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = -p ] && [ \"$2\" = \"$CYCLO_INSTALL_SSD_ROOT/cyclo_intelligence\" ]; then\n"
        "  printf '%s\\n' \"mkdir direct denied $*\" >> \"$INSTALL_TEST_LOG\"\n"
        "  exit 1\n"
        "fi\n"
        "exec /bin/mkdir \"$@\"\n",
    )
    mkdir_stub.chmod(0o755)

    sudo_stub = bin_dir / "sudo"
    sudo_stub.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"sudo $*\" >> \"$INSTALL_TEST_LOG\"\n"
        "if [ \"$1\" = mkdir ]; then\n"
        "  shift\n"
        "  exec /bin/mkdir \"$@\"\n"
        "fi\n"
        "if [ \"$1\" = chown ]; then\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = mount ]; then\n"
        "  shift\n"
        "  exec mount \"$@\"\n"
        "fi\n"
        "exit 1\n",
    )
    sudo_stub.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(home),
        "INSTALL_TEST_LOG": str(log_path),
        "INSTALL_TEST_HOME_MOUNT": str(home / "cyclo_intelligence"),
        "INSTALL_TEST_BIND_MOUNTED": str(bind_marker),
        "CYCLO_INSTALL_REPO_URL": "https://example.invalid/cyclo.git",
        "CYCLO_INSTALL_SSD_ROOT": str(ssd),
        "CYCLO_INSTALL_FSTAB": str(fstab),
    }

    result = subprocess.run(
        ["bash", str(INSTALLER)],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert install_dir.is_dir()
    assert (home / "cyclo_intelligence").is_dir()
    log = log_path.read_text()
    assert "mkdir direct denied -p" in log
    assert "sudo mkdir -p" in log
    assert "sudo chown" in log
    assert "sudo mount " in log
    assert "clone --recurse-submodules --branch main" in log


def test_robot_mode_does_not_duplicate_existing_fstab_bind_mount(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    ssd = tmp_path / "ssd"
    ssd.mkdir()
    install_dir = ssd / "cyclo_intelligence"
    home_mount = home / "cyclo_intelligence"
    fstab = tmp_path / "fstab"
    line = (
        f"{install_dir} {home_mount} none "
        f"bind,nofail,x-systemd.requires-mounts-for={ssd} 0 0"
    )
    fstab.write_text(f"# test fstab\n{line}\n")
    log_path = tmp_path / "install.log"
    bind_marker = tmp_path / "bind-mounted"
    bin_dir = _write_stub_commands(tmp_path, hostname_value="ffw-SNPR48A1106")
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(home),
        "INSTALL_TEST_LOG": str(log_path),
        "INSTALL_TEST_HOME_MOUNT": str(home_mount),
        "INSTALL_TEST_BIND_MOUNTED": str(bind_marker),
        "CYCLO_INSTALL_REPO_URL": "https://example.invalid/cyclo.git",
        "CYCLO_INSTALL_SSD_ROOT": str(ssd),
        "CYCLO_INSTALL_FSTAB": str(fstab),
    }

    result = subprocess.run(
        ["bash", str(INSTALLER)],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert fstab.read_text().count(line) == 1
    assert f"mount {home_mount}" in log_path.read_text()


def test_robot_mode_rejects_conflicting_fstab_target(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    ssd = tmp_path / "ssd"
    ssd.mkdir()
    home_mount = home / "cyclo_intelligence"
    fstab = tmp_path / "fstab"
    fstab.write_text(
        "# test fstab\n"
        f"/mnt/other/cyclo_intelligence {home_mount} none bind 0 0\n"
    )
    log_path = tmp_path / "install.log"
    bind_marker = tmp_path / "bind-mounted"
    bin_dir = _write_stub_commands(tmp_path, hostname_value="ffw-SNPR48A1106")
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(home),
        "INSTALL_TEST_LOG": str(log_path),
        "INSTALL_TEST_HOME_MOUNT": str(home_mount),
        "INSTALL_TEST_BIND_MOUNTED": str(bind_marker),
        "CYCLO_INSTALL_REPO_URL": "https://example.invalid/cyclo.git",
        "CYCLO_INSTALL_SSD_ROOT": str(ssd),
        "CYCLO_INSTALL_FSTAB": str(fstab),
    }

    result = subprocess.run(
        ["bash", str(INSTALLER)],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "fstab already contains a different mount" in result.stderr
    assert "mount " not in log_path.read_text()


def test_robot_mode_rejects_conflict_even_when_exact_bind_exists(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    ssd = tmp_path / "ssd"
    ssd.mkdir()
    install_dir = ssd / "cyclo_intelligence"
    home_mount = home / "cyclo_intelligence"
    exact_line = (
        f"{install_dir} {home_mount} none "
        f"bind,nofail,x-systemd.requires-mounts-for={ssd} 0 0"
    )
    fstab = tmp_path / "fstab"
    fstab.write_text(
        "# test fstab\n"
        f"{exact_line}\n"
        f"/mnt/other/cyclo_intelligence {home_mount} none bind 0 0\n"
    )
    log_path = tmp_path / "install.log"
    bind_marker = tmp_path / "bind-mounted"
    bin_dir = _write_stub_commands(tmp_path, hostname_value="ffw-SNPR48A1106")
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(home),
        "INSTALL_TEST_LOG": str(log_path),
        "INSTALL_TEST_HOME_MOUNT": str(home_mount),
        "INSTALL_TEST_BIND_MOUNTED": str(bind_marker),
        "CYCLO_INSTALL_REPO_URL": "https://example.invalid/cyclo.git",
        "CYCLO_INSTALL_SSD_ROOT": str(ssd),
        "CYCLO_INSTALL_FSTAB": str(fstab),
    }

    result = subprocess.run(
        ["bash", str(INSTALLER)],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "fstab already contains a different mount" in result.stderr
    assert "mount " not in log_path.read_text()


def test_robot_mode_preserves_fstab_line_without_trailing_newline(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    ssd = tmp_path / "ssd"
    ssd.mkdir()
    install_dir = ssd / "cyclo_intelligence"
    home_mount = home / "cyclo_intelligence"
    fstab = tmp_path / "fstab"
    fstab.write_text("# test fstab without newline")
    log_path = tmp_path / "install.log"
    bind_marker = tmp_path / "bind-mounted"
    bin_dir = _write_stub_commands(tmp_path, hostname_value="ffw-SNPR48A1106")
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(home),
        "INSTALL_TEST_LOG": str(log_path),
        "INSTALL_TEST_HOME_MOUNT": str(home_mount),
        "INSTALL_TEST_BIND_MOUNTED": str(bind_marker),
        "CYCLO_INSTALL_REPO_URL": "https://example.invalid/cyclo.git",
        "CYCLO_INSTALL_SSD_ROOT": str(ssd),
        "CYCLO_INSTALL_FSTAB": str(fstab),
    }

    result = subprocess.run(
        ["bash", str(INSTALLER)],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    expected_line = (
        f"{install_dir} {home_mount} none "
        f"bind,nofail,x-systemd.requires-mounts-for={ssd} 0 0"
    )
    assert fstab.read_text() == f"# test fstab without newline\n{expected_line}\n"


def test_robot_mode_skips_mount_when_home_mount_already_active(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    ssd = tmp_path / "ssd"
    ssd.mkdir()
    install_dir = ssd / "cyclo_intelligence"
    home_mount = home / "cyclo_intelligence"
    fstab = tmp_path / "fstab"
    line = (
        f"{install_dir} {home_mount} none "
        f"bind,nofail,x-systemd.requires-mounts-for={ssd} 0 0"
    )
    fstab.write_text(f"# test fstab\n{line}\n")
    log_path = tmp_path / "install.log"
    bind_marker = tmp_path / "bind-mounted"
    bind_marker.touch()
    bin_dir = _write_stub_commands(tmp_path, hostname_value="ffw-SNPR48A1106")
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(home),
        "INSTALL_TEST_LOG": str(log_path),
        "INSTALL_TEST_HOME_MOUNT": str(home_mount),
        "INSTALL_TEST_BIND_MOUNTED": str(bind_marker),
        "CYCLO_INSTALL_REPO_URL": "https://example.invalid/cyclo.git",
        "CYCLO_INSTALL_SSD_ROOT": str(ssd),
        "CYCLO_INSTALL_FSTAB": str(fstab),
    }

    result = subprocess.run(
        ["bash", str(INSTALLER)],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"{home_mount} is already mounted" in result.stdout
    assert f"mount {home_mount}" not in log_path.read_text()


def test_existing_home_path_safely_stops(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "cyclo_intelligence").mkdir()
    ssd = tmp_path / "ssd"
    ssd.mkdir()
    log_path = tmp_path / "install.log"
    bin_dir = _write_stub_commands(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(home),
        "INSTALL_TEST_LOG": str(log_path),
        "CYCLO_INSTALL_REPO_URL": "https://example.invalid/cyclo.git",
        "CYCLO_INSTALL_SSD_ROOT": str(ssd),
    }

    result = subprocess.run(
        ["bash", str(INSTALLER), "--local"],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert not log_path.exists()


def test_ref_option_is_passed_to_git_clone(tmp_path):
    result, _home, _ssd, log_path = _run_install(
        tmp_path,
        ["--local", "--ref", "v9.9.9"],
    )

    assert result.returncode == 0, result.stderr
    assert "clone --recurse-submodules --branch v9.9.9" in log_path.read_text()
