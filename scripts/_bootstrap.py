"""Put the repository root on sys.path so `python scripts/NN_x.py` works from anywhere."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
