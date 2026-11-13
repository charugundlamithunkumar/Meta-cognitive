"""Smoke-test every arm in one model load (calibration ids only).

usage: python3 scripts/smoke_all.py --ids 0-1 --T 12
"""
import argparse, json, os, sys, time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from scripts.run_arm import parse_ids  # noqa: E402

ARMS = ["A0", "A0_inline", "A1a", "A1b", "A2", "A3", "A4", "D1", "D2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="0-1")
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    a = ap.parse_args()
    from src.lhe_data import SPLITS, load_trajectories, load_dictionary
    ids = parse_ids(a.ids)
    assert set(ids) <= set(SPLITS["calibration"])
    dictionary = load_dictionary()
    trajs = load_trajectories(ids, a.T, 1)
    base = json.loads((ROOT / "configs" / "A0.json").read_text())
    from transformers import AutoTokenizer
    from vllm import LLM
    model = str(ROOT / base["model_dir"])
    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model, dtype="bfloat16", gpu_memory_utilization=base["util"], max_model_len=base["max_model_len"],
              enable_prefix_caching=True, seed=0)
    from src.agent import run
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for arm in a.arms:
        cfg = json.loads((ROOT / "configs" / f"{arm}.json").read_text())
        # smoke-only settings: placeholder tau/rate, short audit interval, early checkpoints
        cfg.update(T=a.T, ids=ids, checkpoints=[5, 10], m=4,
                   tau=cfg["tau"] if cfg["tau"] is not None else -0.5,
                   rate=cfg["rate"] if cfg["rate"] is not None else 0.3)
        out = ROOT / "runs" / f"smoke_{arm}_{stamp}"
        t0 = time.time()
        run(cfg, trajs, dictionary, llm, tok, out)
        rows = [json.loads(l) for l in open(out / "rows.jsonl")]
        msg = f"### {arm}: rows={len(rows)} kinds={dict(Counter(r['kind'] for r in rows))} {time.time()-t0:.0f}s"
        g = [r["gate"] for r in rows if "gate" in r]
        if g:
            msg += f" | gate fired={sum(x['fired'] for x in g)}/{len(g)} reasons={dict(Counter(x['reason'] for x in g if x['fired']))}"
        au = [r["audit"] for r in rows if "audit" in r]
        if au:
            msg += (f" | audits={len(au)} finished={sum(x['finished'] for x in au)} actions={dict(Counter(x['action'] for x in au))}"
                    f" think_tokens={[x['think_tokens'] for x in au]}")
        pr = [r["probes"] for r in rows if "probes" in r]
        if pr:
            hc = [p for p in pr if "fore_none" in p]
            msg += f" | probes={len(pr)} ptrue_ex={pr[0].get('ptrue'):.3f} verbal_ex={pr[0].get('verbal')} hc={[(p['fore_none'], p['fore_true'], p['fore_placebo']) for p in hc]}"
        if arm == "A0_inline":
            msg += f" | inline raw ex={[r['raw'] for r in rows[:4]]}"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
