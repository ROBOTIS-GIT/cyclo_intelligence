#!/bin/bash
#
# LeRobot Submodule Update Script
# Updates cyclo_brain/policy/lerobot's huggingface/lerobot submodule to
# the latest main branch and optionally rebuilds the lerobot Docker
# image.
#
# Usage (from anywhere):
#   cyclo_brain/policy/lerobot/scripts/update_lerobot.sh [--rebuild] [--check-only]
#

set -e

# SCRIPT_DIR = cyclo_intelligence/cyclo_brain/policy/lerobot/scripts
# LEROBOT_DIR  = one level up — backend root that holds the
#                huggingface/lerobot submodule + Dockerfile.
# PROJECT_ROOT = repo root, four levels up from scripts/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

REBUILD=false
CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --rebuild) REBUILD=true; shift ;;
        --check-only) CHECK_ONLY=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

check_updates() {
    log_info "Checking for LeRobot updates..."
    
    cd "$LEROBOT_DIR"
    git fetch origin main --quiet
    
    LOCAL=$(git rev-parse HEAD)
    REMOTE=$(git rev-parse origin/main)
    
    if [ "$LOCAL" = "$REMOTE" ]; then
        log_info "LeRobot is up to date (commit: ${LOCAL:0:8})"
        return 1
    else
        log_warn "Updates available!"
        log_info "  Current: ${LOCAL:0:8}"
        log_info "  Latest:  ${REMOTE:0:8}"
        
        echo ""
        log_info "Recent changes:"
        git log --oneline HEAD..origin/main | head -10
        return 0
    fi
}

update_submodule() {
    log_info "Updating LeRobot submodule..."
    
    cd "$LEROBOT_DIR"
    
    BEFORE=$(git rev-parse HEAD)
    git checkout main
    git pull origin main
    AFTER=$(git rev-parse HEAD)
    
    cd "$PROJECT_ROOT"
    git add cyclo_brain/policy/lerobot
    
    log_info "Updated from ${BEFORE:0:8} to ${AFTER:0:8}"
}

rebuild_docker() {
    log_info "Rebuilding LeRobot Docker image..."
    
    cd "$PROJECT_ROOT/docker"
    docker-compose build lerobot
    
    log_info "Docker image rebuilt successfully"
}

check_compatibility() {
    log_info "Checking API compatibility..."
    
    BREAKING_CHANGES=()
    
    cd "$LEROBOT_DIR"
    
    if ! grep -q "class PreTrainedPolicy" src/lerobot/policies/pretrained.py 2>/dev/null; then
        BREAKING_CHANGES+=("PreTrainedPolicy class not found")
    fi
    
    if ! grep -q "def train" src/lerobot/scripts/train.py 2>/dev/null; then
        BREAKING_CHANGES+=("train script structure changed")
    fi
    
    if [ ${#BREAKING_CHANGES[@]} -gt 0 ]; then
        log_error "Potential breaking changes detected:"
        for change in "${BREAKING_CHANGES[@]}"; do
            echo "  - $change"
        done
        return 1
    fi
    
    log_info "No breaking changes detected"
    return 0
}

main() {
    log_info "LeRobot Update Script"
    echo "========================"
    
    if [ ! -d "$LEROBOT_DIR" ]; then
        log_error "LeRobot directory not found: $LEROBOT_DIR"
        exit 1
    fi
    
    if check_updates; then
        if [ "$CHECK_ONLY" = true ]; then
            log_info "Check complete. Run without --check-only to update."
            exit 0
        fi
        
        update_submodule
        
        if ! check_compatibility; then
            log_warn "Review the changes before proceeding"
        fi
        
        if [ "$REBUILD" = true ]; then
            rebuild_docker
        else
            log_info "Run with --rebuild to rebuild Docker image"
        fi
        
        log_info "Update complete!"
    else
        log_info "No updates needed"
    fi
}

main "$@"
