"""Diagnose inline-format replies on calibration ids: first-token alternatives per wording variant."""
import json, math, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from src import prompts as PR
from src.lhe_data import load_trajectories, load_dictionary
from src.agent import Chat

PROSE = (", and reply on one line with the new running total, then a space, a vertical bar '|', a space, and your "
         "confidence as an integer from 0 to 100 that the total is exactly correct. The total must always be your "
         "best integer, even when you are unsure; express doubt only through the confidence. No words or working.")

def system_variant(d, which):
    if which == "current":
        return PR.agent_system(d, 1, inline=True, version=2)
    base = PR.agent_system(d, 1, inline=False, version=2)
    head, tail = base.split(", and reply with ONLY", 1)
    ex = tail.split("\n\n", 1)[1]
    return head + PROSE + "\n\n" + ex.replace(" -> reply ", " -> new total ")

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    cfg = json.loads((ROOT / "configs/A0_inline.json").read_text())
    tok = AutoTokenizer.from_pretrained(str(ROOT / cfg["model_dir"]))
    ch = Chat(tok)
    llm = LLM(str(ROOT / cfg["model_dir"]), dtype="bfloat16", gpu_memory_utilization=cfg["util"],
              max_model_len=8192, seed=0)
    d = load_dictionary()
    trajs = load_trajectories(list(range(8)), 1, 1)
    sp = SamplingParams(temperature=0, max_tokens=12, logprobs=8, stop=["\n"])
    for which in ["current", "prose"]:
        sysm = system_variant(d, which)
        prompts = [{"prompt_token_ids": ch.system(sysm) + ch.user(PR.keys_text(tr.turns[0].keys, version=2)) + ch.gen()}
                   for tr in trajs]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        print(f"\n=== {which}")
        for tr, o in zip(trajs, outs):
            co = o.outputs[0]
            first = sorted(co.logprobs[0].values(), key=lambda x: -x.logprob)[:5]
            alts = [(x.decoded_token, round(math.exp(x.logprob), 3)) for x in first]
            print(f"sample {tr.sample_id} key={tr.turns[0].keys[0]} truth={tr.turns[0].truth} reply={co.text!r} first-token alts={alts}")
