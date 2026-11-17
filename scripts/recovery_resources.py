"""Hash local model, tokenizer, data and software on CPU, once per environment."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.recovery_resources import inventory


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        ap.error('Output exists; refusing overwrite')
    manifest = inventory()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f:
        json.dump(manifest, f, indent=2); f.write('\n')
    print('Saved local CPU resource hashes:', a.out)


if __name__ == '__main__':
    main()
