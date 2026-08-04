#!/usr/bin/env bash
#
# Cyclo Intelligence installer.
#
# Default behavior:
#   - hosts named ffw* install the repository under /mnt/ssd/cyclo_intelligence
#     and bind-mount it at ~/cyclo_intelligence.
#   - other hosts install under ~/cyclo_intelligence.
#   - the user starts the main container explicitly after installation.

set -euo pipefail

REPO_URL="${CYCLO_INSTALL_REPO_URL:-https://github.com/ROBOTIS-GIT/cyclo_intelligence.git}"
REF="main"
MODE="auto"
SSD_ROOT="${CYCLO_INSTALL_SSD_ROOT:-/mnt/ssd}"
FSTAB_PATH="${CYCLO_INSTALL_FSTAB:-/etc/fstab}"

usage() {
    cat <<EOF
Usage: install.sh [--robot|--local] [--ref <branch-or-tag>]

Options:
  --robot       Install under /mnt/ssd/cyclo_intelligence and bind-mount from home.
  --local       Install under ~/cyclo_intelligence.
  --ref REF     Git branch or tag to install (default: main).
  -h, --help    Show this help.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --robot)
            MODE="robot"
            shift
            ;;
        --local)
            MODE="local"
            shift
            ;;
        --ref)
            if [ "$#" -lt 2 ] || [ -z "$2" ]; then
                echo "Error: --ref requires a value." >&2
                exit 1
            fi
            REF="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Error: required command not found: $1" >&2
        exit 1
    fi
}

hostname_short() {
    hostname -s 2>/dev/null || hostname 2>/dev/null || printf '%s\n' ""
}

detect_mode() {
    local host
    host="$(hostname_short | tr '[:upper:]' '[:lower:]')"
    case "$host" in
        ffw*) printf '%s\n' "robot" ;;
        *)    printf '%s\n' "local" ;;
    esac
}

path_exists_or_symlink() {
    [ -e "$1" ] || [ -L "$1" ]
}

assert_available_path() {
    local path="$1"
    local label="$2"
    if path_exists_or_symlink "$path"; then
        echo "Error: ${label} already exists: $path" >&2
        echo "Refusing to modify an existing Cyclo checkout automatically." >&2
        echo "Update it manually with:" >&2
        echo "  cd $path && git pull --ff-only && git submodule update --init --recursive" >&2
        exit 1
    fi
}

run_privileged() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
        return
    fi
    if ! command -v sudo >/dev/null 2>&1; then
        echo "Error: sudo is required to configure the robot bind mount." >&2
        exit 1
    fi
    sudo "$@"
}

assert_ssd_ready() {
    require_command mountpoint
    if ! mountpoint -q "$SSD_ROOT"; then
        echo "Error: robot mode requires mounted SSD: $SSD_ROOT" >&2
        exit 1
    fi
    if [ ! -d "$SSD_ROOT" ]; then
        echo "Error: SSD path is not a directory: $SSD_ROOT" >&2
        exit 1
    fi
}

fstab_has_exact_bind() {
    awk -v src="$INSTALL_DIR" -v dst="$HOME_LINK" '
        $0 !~ /^[[:space:]]*#/ && $1 == src && $2 == dst && $3 == "none" {
            for (i = 4; i <= NF; i++) {
                if ($i ~ /(^|,)bind(,|$)/) {
                    found = 1
                }
            }
        }
        END { exit(found ? 0 : 1) }
    ' "$FSTAB_PATH"
}

fstab_has_conflicting_target() {
    awk -v src="$INSTALL_DIR" -v dst="$HOME_LINK" '
        $0 !~ /^[[:space:]]*#/ && $2 == dst {
            is_exact = 0
            if ($1 == src && $3 == "none") {
                for (i = 4; i <= NF; i++) {
                    if ($i ~ /(^|,)bind(,|$)/) {
                        is_exact = 1
                    }
                }
            }
            if (!is_exact) {
                conflict = 1
            }
        }
        END { exit(conflict ? 0 : 1) }
    ' "$FSTAB_PATH"
}

append_fstab_line() {
    local line="$1"
    if [ -s "$FSTAB_PATH" ] && [ -n "$(tail -c 1 "$FSTAB_PATH" 2>/dev/null)" ]; then
        if [ -w "$FSTAB_PATH" ]; then
            printf '\n' >> "$FSTAB_PATH"
        else
            printf '\n' | run_privileged tee -a "$FSTAB_PATH" >/dev/null
        fi
    fi
    if [ -w "$FSTAB_PATH" ]; then
        printf '%s\n' "$line" >> "$FSTAB_PATH"
        return
    fi
    printf '%s\n' "$line" | run_privileged tee -a "$FSTAB_PATH" >/dev/null
}

configure_robot_bind_mount() {
    local fstab_line
    if [ -L "$HOME_LINK" ]; then
        echo "[install] Removing existing symlink at ${HOME_LINK}"
        rm -f "$HOME_LINK"
    fi
    mkdir -p "$HOME_LINK"

    if [ ! -f "$FSTAB_PATH" ]; then
        echo "Error: fstab file not found: $FSTAB_PATH" >&2
        exit 1
    fi

    if fstab_has_conflicting_target; then
        echo "Error: fstab already contains a different mount for: $HOME_LINK" >&2
        echo "Refusing to modify it automatically." >&2
        exit 1
    fi

    if ! fstab_has_exact_bind; then
        fstab_line="${INSTALL_DIR} ${HOME_LINK} none bind,nofail,x-systemd.requires-mounts-for=${SSD_ROOT%/} 0 0"
        echo "[install] Adding bind mount to ${FSTAB_PATH}: ${HOME_LINK}"
        append_fstab_line "$fstab_line"
    fi

    if mountpoint -q "$HOME_LINK"; then
        echo "[install] ${HOME_LINK} is already mounted."
    else
        echo "[install] Mounting ${INSTALL_DIR} at ${HOME_LINK}"
        run_privileged mount "$HOME_LINK"
        if ! mountpoint -q "$HOME_LINK"; then
            echo "Error: bind mount did not become active: $HOME_LINK" >&2
            exit 1
        fi
    fi
}

prepare_ssd_install_dir() {
    local install_dir="$1"
    if mkdir -p "$install_dir" 2>/dev/null; then
        return
    fi
    if [ "$(id -u)" -eq 0 ]; then
        mkdir -p "$install_dir"
        return
    fi
    if ! command -v sudo >/dev/null 2>&1; then
        echo "Error: SSD path is not writable and sudo is not available: $SSD_ROOT" >&2
        exit 1
    fi

    echo "[install] Creating SSD install directory with sudo: $install_dir"
    sudo mkdir -p "$install_dir"
    sudo chown "$(id -u):$(id -g)" "$install_dir"
}

clone_repo() {
    local install_dir="$1"
    local parent_dir
    parent_dir="$(dirname "$install_dir")"
    mkdir -p "$parent_dir"

    echo "[install] Cloning ${REPO_URL} (${REF}) into ${install_dir}"
    if ! git clone --recurse-submodules --branch "$REF" "$REPO_URL" "$install_dir"; then
        echo "[install] Clone failed; removing incomplete checkout: $install_dir" >&2
        rm -rf "$install_dir"
        exit 1
    fi
}

require_command git

if [ "$MODE" = "auto" ]; then
    MODE="$(detect_mode)"
fi

HOME_LINK="${HOME}/cyclo_intelligence"
case "$MODE" in
    robot)
        INSTALL_DIR="${SSD_ROOT%/}/cyclo_intelligence"
        assert_ssd_ready
        assert_available_path "$INSTALL_DIR" "SSD install path"
        assert_available_path "$HOME_LINK" "home link path"
        prepare_ssd_install_dir "$INSTALL_DIR"
        clone_repo "$INSTALL_DIR"
        configure_robot_bind_mount
        ;;
    local)
        INSTALL_DIR="$HOME_LINK"
        assert_available_path "$INSTALL_DIR" "local install path"
        clone_repo "$INSTALL_DIR"
        ;;
    *)
        echo "Error: invalid install mode: $MODE" >&2
        exit 1
        ;;
esac

echo "[install] Installed Cyclo Intelligence at: $INSTALL_DIR"
echo "[install] Start Cyclo Intelligence manually with:"
echo "  cd ${HOME_LINK}"
echo "  ./docker/container.sh start"
