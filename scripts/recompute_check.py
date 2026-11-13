"""Measure re-compute accuracy in isolation, per prompt version / mode, on (total, key) cases
taken from a calibration run's fired gate turns.  usage: python3 scripts/recompute_check.py RUN_DIR [--gpu 0]"""
import argparse, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--gpu", default="0")
    ap.add_argument("--util", type=float, default=0.6); ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"; os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    from src import prompts as PR
    from src.lhe_data import load_dictionary
    from src.agent import Chat
    from src.labels import parse_reply
    rows = [json.loads(l) for l in open(Path(a.run) / "rows.jsonl")]
    cases = [(r["prev"], r["keys"], r["v"]) for r in rows if r.get("gate", {}).get("fired")][: a.n]
    d = load_dictionary()
    cfg = json.loads((ROOT / "configs/A1b.json").read_text())
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    tok = AutoTokenizer.from_pretrained(str(ROOT / cfg["model_dir"])); ch = Chat(tok)
    llm = LLM(str(ROOT / cfg["model_dir"]), dtype="bfloat16", gpu_memory_utilization=a.util,
              max_model_len=8192, enable_prefix_caching=True, seed=0)
    sysm = ch.system(PR.recompute_system(d))
    so = StructuredOutputsParams(regex=r"-?[0-9]{1,6}")
    variants = {
        "v1_nonthink": (1, False, SamplingParams(temperature=0, max_tokens=16, structured_outputs=so)),
        "v2_nonthink": (2, False, SamplingParams(temperature=0, max_tokens=16, structured_outputs=so)),
        "v2_think_low": (2, True, SamplingParams(temperature=1.0, top_p=0.95, max_tokens=1024, seed=0)),
    }
    for name, (ver, think, sp) in variants.items():
        t0 = time.time()
        ps = []
        for prev, keys, v in cases:
            u = PR.recompute_user(prev, keys, ver) + (PR.LOW_EFFORT_SUFFIX if think else "")
            ps.append({"prompt_token_ids": sysm + ch.user(u) + ch.gen(think=think)})
        outs = llm.generate(ps, sp, use_tqdm=False)
        ok = echo = unfinished = 0
        ntok = []
        for (prev, keys, v), o in zip(cases, outs):
            txt = o.outputs[0].text
            ntok.append(len(o.outputs[0].token_ids))
            if think:
                if "</think>" not in txt:
                    unfinished += 1; continue
                txt = txt.rsplit("</think>", 1)[1]
            val, _, _ = parse_reply(txt.strip())
            ok += int(val == prev + v); echo += int(val == prev)
        n = len(cases)
        print(f"### {name}: correct {ok}/{n} = {ok / n:.3f} | echoed start {echo / n:.3f} | unfinished {unfinished / n:.3f} "
              f"| mean tokens {sum(ntok) / n:.0f} | {time.time() - t0:.0f}s", flush=True)
