#!/usr/bin/env python3
"""Bench entry point for prepared ABot datasets; never performs conversion."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abot_training.runner import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("validate", "train", "reload"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    main(args.config, args.phase, args.checkpoint)
