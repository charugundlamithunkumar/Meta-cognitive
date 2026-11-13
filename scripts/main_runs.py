"""Phase 4 main runs on the test set (samples 20-51), exactly as pre-registered in prereg.md.

Guards: refuses to run unless prereg.md is committed, the working tree has no uncommitted changes to
prereg.md / configs / src, and every SHA-256 listed in prereg.md matches.
usage: python3 scripts/main_runs.py [--gpu 0] [--arms A0 ...] [--resume STAMP]
"""
import argparse, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ORDER = ["A0", "A0_inline", "A1a", "A1b", "A2", "D1", "D2", "A3", "A4"]
GROUP_SIZE = 8


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def check_registration():
    if not git("ls-files", "prereg.md"):
        sys.exit("refusing: prereg.md is not committed")
    dirty = git("status", "--porcelain", "prereg.md", "configs", "src")
    if dirty:
        sys.exit(f"refusing: uncommitted changes to registered files:\n{dirty}")
    reg = re.findall(r"^([0-9a-f]{64}) (\S+)$", (ROOT / "prereg.md").read_text(), re.M)
    if not reg:
        sys.exit("refusing: no checksums found in prereg.md")
    for h, p in reg:
        got = hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
        if got != h:
            sys.exit(f"refusing: {p} differs from its registered checksum")
    return git("rev-parse", "HEAD"), git("log", "-1", "--format=%H", "--", "prereg.md")


def a1b_rate(run_dir):
    n = fired = 0
    with open(run_dir / "rows.jsonl") as f:
        for l in f:
            r = json.loads(l)
            n += 1
            fired += int(bool(r.get("gate", {}).get("fired")))
    return fired / n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--arms", nargs="*", default=ORDER)
    ap.add_argument("--resume", default=None)
    a = ap.parse_args()
    head, prereg_commit = check_registration()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    from src.lhe_data import SPLITS, load_trajectories, load_dictionary, verify, REVISION
    verify()
    d = load_dictionary()
    base = json.loads((ROOT / "configs/A0.json").read_text())
    from transformers import AutoTokenizer
    from vllm import LLM
    tok = AutoTokenizer.from_pretrained(str(ROOT / base["model_dir"]))
    llm = LLM(str(ROOT / base["model_dir"]), dtype="bfloat16", gpu_memory_utilization=base["util"],
              max_model_len=base["max_model_len"], enable_prefix_caching=True, seed=0)
    from src.agent import run
    stamp = a.resume or time.strftime("%Y%m%d-%H%M%S")
    ids = SPLITS["test"]
    print(f"### code {head[:7]} prereg {prereg_commit[:7]} stamp {stamp} test ids {ids[0]}-{ids[-1]}", flush=True)
    for arm in [x for x in ORDER if x in a.arms]:
        cfg = json.loads((ROOT / "configs" / f"{arm}.json").read_text())
        cfg.update(ids=ids, group_size=GROUP_SIZE, split="test", dataset_revision=REVISION,
                   code_commit=head, prereg_commit=prereg_commit, gpu=a.gpu)
        if arm == "A2":
            src = ROOT / "runs" / f"main_A1b_{stamp}"
            with open(src / "trajectories.jsonl") as f:
                if sum(1 for _ in f) != len(ids):
                    sys.exit("refusing A2: A1b has not finished all test trajectories")
            cfg["rate"] = a1b_rate(src)
            print(f"### A2 rate from A1b = {cfg['rate']:.4f}", flush=True)
        trajs = load_trajectories(ids, cfg["T"], cfg["K"], check=False)
        out = ROOT / "runs" / f"main_{arm}_{stamp}"
        resume = out.exists()
        t0 = time.time()
        print(f"### START {arm}{' (resume)' if resume else ''}", flush=True)
        run(cfg, trajs, d, llm, tok, out, resume=resume)
        print(f"### DONE {arm} {time.time() - t0:.0f}s", flush=True)
