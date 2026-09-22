"""Use the same shared runtime imports as the model Worker image."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common" / "runtime"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extensions"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lerobot" / "src"))
