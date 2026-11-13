"""Download the Long-Horizon-Execution files at the pinned revision and verify checksums.

usage: python3 scripts/fetch_data.py
"""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.lhe_data import DATA_DIR, REVISION, EXPECTED_SHA256, sha256, verify  # noqa: E402

BASE = f"https://huggingface.co/datasets/arvindh75/Long-Horizon-Execution/resolve/{REVISION}/"

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, want in EXPECTED_SHA256.items():
        p = DATA_DIR / name
        if p.exists() and sha256(p) == want:
            print("ok (cached)", name)
            continue
        print("downloading", name)
        urllib.request.urlretrieve(BASE + name, p)
    verify()
    print("all checksums match revision", REVISION)
