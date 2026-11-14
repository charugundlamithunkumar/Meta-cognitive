"""Study 2 runs (prereg2.md): reserve ids 52-99 and fresh sequences.

  pilot (calibration ids only, no registration needed):
      python3 scripts/study2_runs.py --pilot --arms T1 A1all A5 --ids 0-3 --T 40
  main (refuses unless prereg2.md is committed, the registered files are clean and every SHA-256 matches):
      python3 scripts/study2_runs.py --gpu 0 [--arms ...] [--resume STAMP]
"""
import argparse, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ORDER = ["R_A0", "R_A1b", "A1all", "T1", "D2u", "D2b", "A5", "R_A3"]
GROUP_SIZE = 16


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def check_registration():
    if not git("ls-files", "prereg2.md"):
        sys.exit("refusing: prereg2.md is not committed")
    dirty = git("status", "--porcelain", "prereg2.md", "configs/study2", "src")
    if dirty:
        sys.exit(f"refusing: uncommitted changes to registered files:\n{dirty}")
    reg = re.findall(r"^([0-9a-f]{64}) (\S+)$", (ROOT / "prereg2.md").read_text(), re.M)
    if not reg:
        sys.exit("refusing: no checksums found in prereg2.md")
    for h, p in reg:
        if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h:
            sys.exit(f"refusing: {p} differs from its registered checksum")
    return git("rev-parse", "HEAD"), git("log", "-1", "--format=%H", "--", "prereg2.md")


def parse_ids(s):
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--arms", nargs="*", default=ORDER)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--ids", default="0-3", help="pilot only: calibration ids, e.g. 0-3")
    ap.add_argument("--T", type=int, default=40, help="pilot only")
    a = ap.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    from src.lhe_data import SPLITS, load_trajectories, load_dictionary, verify, generate_fresh, REVISION
    if a.pilot:
        ids = parse_ids(a.ids)
        assert set(ids) <= set(SPLITS["calibration"]), "pilot uses calibration ids only"
        head, reg = git("rev-parse", "HEAD"), None
        split, T_over, prefix = "calibration", a.T, "s2pilot"
    else:
        head, reg = check_registration()
        ids, split, T_over, prefix = SPLITS["reserve"], "reserve", None, "s2"
    verify()
    d = load_dictionary()
    base = json.loads((ROOT / "configs/study2/R_A0.json").read_text())
    from transformers import AutoTokenizer
    from vllm import LLM
    tok = AutoTokenizer.from_pretrained(str(ROOT / base["model_dir"]))
    llm = LLM(str(ROOT / base["model_dir"]), dtype="bfloat16", gpu_memory_utilization=base["util"],
              max_model_len=base["max_model_len"], enable_prefix_caching=True, seed=0)
    from src.agent import run
    stamp = a.resume or time.strftime("%Y%m%d-%H%M%S")
    print(f"### study2 code {head[:7]} prereg2 {(reg or 'PILOT')[:7]} stamp {stamp} split {split}", flush=True)
    for name in [x for x in ORDER if x in a.arms]:
        cfg = json.loads((ROOT / "configs" / "study2" / f"{name}.json").read_text())
        T = T_over or cfg["T"]
        fresh = cfg.pop("fresh", None)
        if fresh:
            seeds = fresh["seeds"][:len(ids)] if a.pilot else fresh["seeds"]
            trajs = generate_fresh(seeds, T, cfg["K"], bound=fresh["bound"])
            cfg.update(fresh_seeds=seeds, fresh_bound=fresh["bound"])
        else:
            trajs = load_trajectories(ids, T, cfg["K"], check=False)
        cfg.update(T=T, ids=[t.sample_id for t in trajs], group_size=GROUP_SIZE, split=split,
                   dataset_revision=REVISION, code_commit=head, prereg2_commit=reg, gpu=a.gpu)
        out = ROOT / "runs" / f"{prefix}_{name}_{stamp}"
        resume = out.exists()
        t0 = time.time()
        print(f"### START {name}{' (resume)' if resume else ''}", flush=True)
        run(cfg, trajs, d, llm, tok, out, resume=resume)
        print(f"### DONE {name} {time.time() - t0:.0f}s", flush=True)
