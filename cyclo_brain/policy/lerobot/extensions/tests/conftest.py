from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extensions"))
sys.path.insert(0, str(ROOT / "lerobot/src"))
