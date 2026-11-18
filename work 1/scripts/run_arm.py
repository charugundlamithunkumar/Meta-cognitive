"""Run one arm from a JSON config.

usage: python3 scripts/run_arm.py --config configs/A0.json [--split calibration|test] [--ids 0-19]
                                  [--T 300] [--K 1] [--tag pilot] [--set key=value ...]
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
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # GPU 0 (RTX 6000 Ada), agreed 2026-11-13
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_ids(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--fresh", type=int, default=0, help="use N freshly generated sequences (contamination check)")
    ap.add_argument("--set", nargs="*", default=[], help="override config entries, key=value (JSON values)")
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text())
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:
            cfg[k] = v
    if a.T is not None:
        cfg["T"] = a.T
    if a.K is not None:
        cfg["K"] = a.K

    from src.lhe_data import SPLITS, load_trajectories, load_dictionary, verify, REVISION, generate_fresh
    verify()
    dictionary = load_dictionary()
    if a.fresh:
        seeds = list(range(a.fresh))
        trajs = generate_fresh(seeds, cfg["T"], cfg["K"])
        cfg["ids"] = [tr.sample_id for tr in trajs]
        cfg["fresh_seeds"] = seeds
    else:
        if a.ids:
            ids = parse_ids(a.ids)
        else:
            ids = SPLITS[a.split or cfg.get("split", "calibration")]
        if cfg.get("split_guard", True) and a.split != "test" and set(ids) & set(SPLITS["test"]):
            sys.exit("refusing: test ids requested without --split test (pre-registration guard)")
        cfg["ids"] = ids
        trajs = load_trajectories(ids, cfg["T"], cfg["K"], check=False)
    cfg["dataset_revision"] = REVISION

    from transformers import AutoTokenizer
    from vllm import LLM
    import vllm
    cfg["vllm_version"] = vllm.__version__
    cfg["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    model = str(ROOT / cfg["model_dir"])
    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model, dtype="bfloat16", gpu_memory_utilization=cfg["util"], max_model_len=cfg["max_model_len"],
              enable_prefix_caching=True, seed=0)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{cfg['arm']}_K{cfg['K']}_T{cfg['T']}" + (f"_{a.tag}" if a.tag else "") + f"_{stamp}"
    from src.agent import run
    run(cfg, trajs, dictionary, llm, tok, ROOT / "runs" / name)
    print("wrote", ROOT / "runs" / name)


if __name__ == "__main__":
    main()
