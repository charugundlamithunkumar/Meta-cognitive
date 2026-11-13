"""Trimmed Phase 1 pilot (user decision 2026-11-13): baseline vs one treatment arm on the same
calibration trajectories, plus trimmed contamination / inline / audit checks.
Per-trajectory saving + resume (agent.run). Refuses to start without enough free GPU memory.
GPU 0 approved again by the user on 2026-11-13.

usage: python3 scripts/pilot_trimmed.py --gpu 2 [--util 0.22] [--jobs name ...] [--resume RUN_STAMP]
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TAU = -0.13805100321769714   # Youden tau from pilot job 1 (runs/pilot_A0_K1_T300_calib_20261113-150049)
JOBS = [
    # treatment arm, paired with job 1's A0 on the same 20 calibration trajectories (first 100 turns)
    dict(name="trim_A1b_T100_calib", arm="A1b", ids=list(range(0, 20)), T=100, tau=TAU, group_size=4),
    # contamination check: fresh sequences vs job 1's first 100 turns; side-question probes off
    dict(name="trim_A0_T100_fresh", arm="A0", fresh=20, T=100, probes=False, group_size=4),
    dict(name="trim_A0inline_T30_calib", arm="A0_inline", ids=list(range(0, 4)), T=30, checkpoints=[10, 25], group_size=4),
    dict(name="trim_A3_full8192_T40", arm="A3", ids=list(range(0, 4)), T=40, think_budget=8192, group_size=4),
    dict(name="trim_A3_low4096_T40", arm="A3", ids=list(range(0, 4)), T=40, think_budget=4096,
         audit_low_effort=True, group_size=4),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--util", type=float, default=None)
    ap.add_argument("--jobs", nargs="*", default=None)
    ap.add_argument("--resume", default=None, help="timestamp of an earlier run to resume")
    a = ap.parse_args()
    q = subprocess.run(["nvidia-smi", f"--id={a.gpu}", "--query-gpu=memory.total,memory.free",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split(",")
    total, free = float(q[0]), float(q[1])
    util = a.util or round(min(0.6, (free - 1500) / total), 3)
    need = 17600 + 3000  # Granite bf16 weights + minimum KV cache, MiB
    if util * total < need:
        sys.exit(f"refusing: GPU {a.gpu} has {free:.0f} MiB free; need about {need} MiB")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    from src.lhe_data import SPLITS, load_trajectories, load_dictionary, generate_fresh, verify, REVISION
    verify()
    d = load_dictionary()
    base = json.loads((ROOT / "configs/A0.json").read_text())
    from transformers import AutoTokenizer
    from vllm import LLM
    tok = AutoTokenizer.from_pretrained(str(ROOT / base["model_dir"]))
    llm = LLM(str(ROOT / base["model_dir"]), dtype="bfloat16", gpu_memory_utilization=util,
              max_model_len=base["max_model_len"], enable_prefix_caching=True, seed=0)
    from src.agent import run
    stamp = a.resume or time.strftime("%Y%m%d-%H%M%S")
    print(f"### GPU {a.gpu} util {util} stamp {stamp}", flush=True)
    for job in JOBS:
        if a.jobs and job["name"] not in a.jobs:
            continue
        cfg = json.loads((ROOT / "configs" / f"{job['arm']}.json").read_text())
        cfg.update({k: v for k, v in job.items() if k not in ("name", "ids", "fresh")})
        cfg.update(util=util, gpu=a.gpu)
        if job.get("fresh"):
            trajs = generate_fresh(list(range(job["fresh"])), cfg["T"], cfg["K"])
        else:
            assert set(job["ids"]) <= set(SPLITS["calibration"]), "pilot uses calibration ids only"
            trajs = load_trajectories(job["ids"], cfg["T"], cfg["K"], check=False)
        cfg["ids"] = [t.sample_id for t in trajs]
        cfg["dataset_revision"] = REVISION
        out = ROOT / "runs" / f"{job['name']}_{stamp}"
        resume = bool(a.resume) and out.exists()
        t0 = time.time()
        print(f"### START {job['name']}{' (resume)' if resume else ''}", flush=True)
        run(cfg, trajs, d, llm, tok, out, resume=resume)
        print(f"### DONE {job['name']} {time.time() - t0:.0f}s", flush=True)
