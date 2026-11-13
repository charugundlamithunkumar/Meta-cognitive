"""Phase 1 pilot: several calibration jobs in one model load (implementation-plan.md §7, Phase 1).
Calibration ids and fresh sequences only; never test ids."""
import json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from src.lhe_data import SPLITS, load_trajectories, load_dictionary, generate_fresh, verify, REVISION

JOBS = [
    dict(name="pilot_A0_K1_T300_calib", arm="A0", ids=list(range(0, 20)), T=300),
    dict(name="pilot_A0_K1_T300_fresh", arm="A0", fresh=20, T=300),
    dict(name="pilot_A0inline_T30_calib", arm="A0_inline", ids=list(range(0, 4)), T=30, checkpoints=[10, 25]),
    dict(name="pilot_A3_full8192_T60", arm="A3", ids=list(range(0, 8)), T=60, think_budget=8192, tau=None),
    dict(name="pilot_A3_low4096_T60", arm="A3", ids=list(range(0, 8)), T=60, think_budget=4096, audit_low_effort=True),
]

if __name__ == "__main__":
    only = sys.argv[1:]
    verify()
    d = load_dictionary()
    base = json.loads((ROOT / "configs/A0.json").read_text())
    from transformers import AutoTokenizer
    from vllm import LLM
    tok = AutoTokenizer.from_pretrained(str(ROOT / base["model_dir"]))
    llm = LLM(str(ROOT / base["model_dir"]), dtype="bfloat16", gpu_memory_utilization=base["util"],
              max_model_len=base["max_model_len"], enable_prefix_caching=True, seed=0)
    from src.agent import run
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for job in JOBS:
        if only and job["name"] not in only:
            continue
        cfg = json.loads((ROOT / "configs" / f"{job['arm']}.json").read_text())
        cfg.update({k: v for k, v in job.items() if k not in ("name", "ids", "fresh")})
        if job.get("fresh"):
            trajs = generate_fresh(list(range(job["fresh"])), cfg["T"], cfg["K"])
            cfg["fresh_seeds"] = list(range(job["fresh"]))
        else:
            assert set(job["ids"]) <= set(SPLITS["calibration"]), "pilot uses calibration ids only"
            trajs = load_trajectories(job["ids"], cfg["T"], cfg["K"], check=False)
        cfg["ids"] = [t.sample_id for t in trajs]
        cfg["dataset_revision"] = REVISION
        t0 = time.time()
        print(f"### START {job['name']}", flush=True)
        run(cfg, trajs, d, llm, tok, ROOT / "runs" / f"{job['name']}_{stamp}")
        print(f"### DONE {job['name']} {time.time() - t0:.0f}s", flush=True)
