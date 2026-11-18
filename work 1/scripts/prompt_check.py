"""Compare agent prompt versions on calibration ids only (one model load).

usage: python3 scripts/prompt_check.py --versions 2 3 --ids 0-3 --T 40
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from scripts.run_arm import parse_ids  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--ids", default="0-3")
    ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--util", type=float, default=None)
    a = ap.parse_args()
    from src.lhe_data import SPLITS, load_trajectories, load_dictionary
    ids = parse_ids(a.ids)
    assert set(ids) <= set(SPLITS["calibration"]), "prompt checks use calibration ids only"
    cfg0 = json.loads((ROOT / "configs" / "A0.json").read_text())
    cfg0.update(T=a.T, K=a.K, ids=ids, checkpoints=[])
    if a.util:
        cfg0["util"] = a.util
    dictionary = load_dictionary()
    trajs = load_trajectories(ids, a.T, a.K)
    from transformers import AutoTokenizer
    from vllm import LLM
    model = str(ROOT / cfg0["model_dir"])
    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model, dtype="bfloat16", gpu_memory_utilization=cfg0["util"], max_model_len=cfg0["max_model_len"],
              enable_prefix_caching=True, seed=0)
    from src.agent import run
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for v in a.versions:
        cfg = dict(cfg0, prompt_version=v, arm="A0")
        out = ROOT / "runs" / f"promptcheck_v{v}_K{a.K}_T{a.T}_{stamp}"
        run(cfg, trajs, dictionary, llm, tok, out)
        rows = [json.loads(l) for l in open(out / "rows.jsonl")]
        from collections import Counter
        c = Counter(r["kind"] for r in rows)
        copy_value = sum(r["value"] == r["v"] and r["turn"] > 0 for r in rows)
        first_err = {}
        for r in rows:
            if r["kind"] != "ok" and r["sample_id"] not in first_err:
                first_err[r["sample_id"]] = r["turn"]
        print(f"\n### prompt v{v}: kinds={dict(c)} replies==key value (t>0): {copy_value}/{len(rows) - len(ids)} "
              f"first error turn per traj={first_err}", flush=True)


if __name__ == "__main__":
    main()
