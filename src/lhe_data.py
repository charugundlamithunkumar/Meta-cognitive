"""Long-Horizon-Execution (Sinha et al., ICLR 2026) loader.

Vendored at data/lhe/ from HF dataset arvindh75/Long-Horizon-Execution,
revision 7e4901fbb42b88144a65782052d0c8d87a358edc (Apache-2.0).
The loader refuses to run if any checksum differs.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "lhe"
REVISION = "7e4901fbb42b88144a65782052d0c8d87a358edc"
EXPECTED_SHA256 = {
    "dictionary.json": "33d783c31d664d791a6ff08671b86775cac681768d2853987d528c30457c017e",
    "test.jsonl": "111ef6e3850368efd84175576c279aeba36ddfb20da3582c01a078bfa77620c6",
    "config.json": "0cc5df39a45c71d1044e197f660045de09922d74011efa0c688a658c72aa4656",
}

# Fixed splits (implementation-plan.md §2). Never change.
SPLITS = {
    "calibration": list(range(0, 20)),
    "test": list(range(20, 52)),
    "reserve": list(range(52, 100)),
}


@dataclass
class Turn:
    keys: tuple          # K keys shown this turn
    value: int           # sum of their true values
    truth: int           # true running total after this turn


@dataclass
class Trajectory:
    sample_id: int
    turns: list          # list[Turn]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify(data_dir=DATA_DIR):
    for name, want in EXPECTED_SHA256.items():
        got = sha256(Path(data_dir) / name)
        if got != want:
            raise RuntimeError(f"checksum mismatch for {name}: {got} != {want}")


def load_dictionary(data_dir=DATA_DIR):
    with open(Path(data_dir) / "dictionary.json") as f:
        return json.load(f)


def group_turns(keys, values, outputs, T, K):
    """Turn j covers keys [jK, (j+1)K); its target is the last running total of the group."""
    if T * K > len(keys):
        raise ValueError(f"T*K={T*K} exceeds sample length {len(keys)}")
    turns = []
    for j in range(T):
        lo, hi = j * K, (j + 1) * K
        turns.append(Turn(tuple(keys[lo:hi]), int(sum(values[lo:hi])), int(outputs[hi - 1])))
    return turns


FRESH_ID_OFFSET = 10_000  # fresh sequences get sample_id = 10_000 + seed (never clash with 0-99)


def generate_fresh(seeds, T, K=1, data_dir=DATA_DIR):
    """Fresh sequences on the same dictionary (contamination check, plan §2): keys drawn
    uniformly at random, i.i.d., which matches the published samples' key statistics."""
    import random
    d = load_dictionary(data_dir)
    names = list(d)
    out = []
    for s in seeds:
        rng = random.Random(FRESH_ID_OFFSET + s)
        keys = [rng.choice(names) for _ in range(T * K)]
        values = [d[k] for k in keys]
        outputs, tot = [], 0
        for v in values:
            tot += v
            outputs.append(tot)
        out.append(Trajectory(FRESH_ID_OFFSET + s, group_turns(keys, values, outputs, T, K)))
    return out


def load_trajectories(ids, T, K=1, data_dir=DATA_DIR, check=True):
    if check:
        verify(data_dir)
    want = set(ids)
    found = {}
    with open(Path(data_dir) / "test.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in want:
                found[r["id"]] = Trajectory(r["id"], group_turns(r["input"], r["values"], r["output"], T, K))
                if len(found) == len(want):
                    break
    missing = want - set(found)
    if missing:
        raise KeyError(f"sample ids not found: {sorted(missing)}")
    return [found[i] for i in ids]
