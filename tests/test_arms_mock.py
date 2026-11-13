"""Phase 2 forced tests with a scripted fake model (no GPU).

The fake reads the real prompt (decoded with the real tokenizer) and answers by rule:
- agent turn: (its own last number, or the auditor's corrected total, or the D1/D2 state line) + key value,
  except for scripted faults on chosen keys: 'slip' (+1, low logprob) or 'malformed' ('oops');
- re-compute prompt: exact arithmetic;  audit prompt: exact verdict;  probes: fixed replies.
So every arm's bookkeeping (gates, commits, labels, corrections, state lines, HC evidence) can be checked exactly.
Run: python3 -m unittest tests.test_arms_mock -v
"""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import lhe_data  # noqa: E402

CFG0 = json.loads((ROOT / "configs" / "A0.json").read_text())
MODEL = ROOT / CFG0["model_dir"]
HAVE_TOK = (MODEL / "tokenizer_config.json").exists()

LP_OK, LP_SLIP = -0.01, -3.0


class FakeLLM:
    def __init__(self, tok, dictionary, faults):
        self.tok, self.d, self.faults = tok, dictionary, faults  # faults: {key: 'slip'|'malformed'}
        self.eos = tok.eos_token_id
        self.calls = []

    def _out(self, text, lp=LP_OK, with_lps=True):
        ids = self.tok.encode(text, add_special_tokens=False) + [self.eos]
        lps = None
        if with_lps:
            alt = self.tok.encode("7", add_special_tokens=False)[0]
            lps = []
            for i in ids:
                dct = {i: SimpleNamespace(logprob=lp)}
                if alt != i:
                    dct[alt] = SimpleNamespace(logprob=lp - 5)
                lps.append(dct)
        return SimpleNamespace(outputs=[SimpleNamespace(text=text, token_ids=ids, logprobs=lps)])

    def _yes(self, p_yes):
        import math
        y = self.tok.encode("Yes", add_special_tokens=False)[0]
        n = self.tok.encode("No", add_special_tokens=False)[0]
        lps = [{y: SimpleNamespace(logprob=math.log(p_yes)), n: SimpleNamespace(logprob=math.log(1 - p_yes))}]
        return SimpleNamespace(outputs=[SimpleNamespace(text="Yes", token_ids=[y], logprobs=lps)])

    def _abc(self):
        ids = [self.tok.encode(w, add_special_tokens=False)[0] for w in "ABC"]
        lps = [{i: SimpleNamespace(logprob=v) for i, v in zip(ids, [-3.0, -0.1, -3.0])}]
        return SimpleNamespace(outputs=[SimpleNamespace(text="B", token_ids=[ids[1]], logprobs=lps)])

    def respond(self, text):
        users = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, re.S)
        last = users[-1]
        if "Is the running total you just reported" in last:
            return self._yes(0.9)
        if "How confident are you" in last:
            return self._out("100", with_lps=False)
        if "Over your next 10 turns" in last:
            m = re.search(r"found (\d+) of them wrong", last)
            return self._out(str(100 - 10 * int(m.group(1))) if m else "100", with_lps=False)
        if "Compared with the start" in last:
            return self._abc()
        if last.startswith("Audit."):
            mv = re.search(r"verified running total after turn \d+ was (-?\d+)", last)
            V = int(mv.group(1)) if mv else 0
            keys = re.search(r"were: (.*?)\. The reported", last).group(1).split(", ")
            R = int(re.search(r"is (-?\d+)\. Recompute", last).group(1))
            tot = V + sum(self.d[k] for k in keys)
            v = "CORRECT" if tot == R else f"WRONG, total = {tot}"
            return self._out(f"checking</think>\nVERDICT: {v}", with_lps=False)
        if "Current running total:" in last:  # re-compute prompt
            prev = int(re.search(r"Current running total: (-?\d+)", last).group(1))
            key = re.search(r"Key: (\w+)", last).group(1)
            return self._out(str(prev + self.d[key]))
        # agent turn
        key = re.search(r"Key: (\w+)", last).group(1)
        ms = re.search(r"Your current running total: (-?\d+)", last)
        mn = re.search(r"should be (-?\d+)", last)
        if ms:
            base = int(ms.group(1))
        elif mn:
            base = int(mn.group(1))
        else:
            asst = re.findall(r"<\|im_start\|>assistant\n(?:<think></think>)?(.*?)<\|im_end\|>", text, re.S)
            nums = [int(a) for a in asst if re.fullmatch(r"-?\d+", a.strip())]
            base = nums[-1] if nums else 0
        f = self.faults.get(key)
        if f == "malformed":
            return self._out("oops")
        if f == "slip":
            return self._out(str(base + self.d[key] + 1), lp=LP_SLIP)
        return self._out(str(base + self.d[key]))

    def generate(self, prompts, params, use_tqdm=False):
        outs = []
        for p in prompts:
            text = self.tok.decode(p["prompt_token_ids"])
            self.calls.append(text)
            outs.append(self.respond(text))
        return outs


@unittest.skipUnless(HAVE_TOK, "model tokenizer not downloaded")
class TestArms(unittest.TestCase):
    T = 12

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(str(MODEL))
        cls.d = lhe_data.load_dictionary()
        cls.trajs = lhe_data.load_trajectories([0, 1], cls.T, 1)
        k0 = [t.keys[0] for t in cls.trajs[0].turns]
        k1 = {t.keys[0] for t in cls.trajs[1].turns}
        # a key that appears exactly once in trajectory 0 (turns 2..6) and never in trajectory 1
        cls.fault_turn = next(i for i in range(2, 7) if k0.count(k0[i]) == 1 and k0[i] not in k1)
        cls.fault_key = k0[cls.fault_turn]

    def run_arm(self, arm, faults, **over):
        from src.agent import run
        cfg = json.loads((ROOT / "configs" / f"{arm}.json").read_text())
        cfg.update({"T": self.T, "checkpoints": [], **over})
        llm = FakeLLM(self.tok, self.d, faults)
        out = Path(tempfile.mkdtemp()) / "run"
        run(cfg, self.trajs, self.d, llm, self.tok, out)
        with open(out / "rows.jsonl") as f:
            rows = [json.loads(l) for l in f]
        return rows, llm

    def traj0(self, rows):
        return sorted([r for r in rows if r["sample_id"] == 0], key=lambda r: r["turn"])

    def test_A0_slip_then_inherited(self):
        rows, _ = self.run_arm("A0", {self.fault_key: "slip"})
        r0 = self.traj0(rows)
        ft = self.fault_turn
        self.assertTrue(all(r["kind"] == "ok" for r in r0[:ft]))
        self.assertEqual(r0[ft]["kind"], "numeric_slip")
        self.assertTrue(all(r["kind"] == "inherited" for r in r0[ft + 1:]))
        self.assertAlmostEqual(r0[ft]["lp"]["lp_min"], LP_SLIP, places=5)
        self.assertTrue(all(r["kind"] == "ok" for r in rows if r["sample_id"] == 1))

    def test_A1a_format_gate_fixes_malformed(self):
        rows, _ = self.run_arm("A1a", {self.fault_key: "malformed"})
        r = self.traj0(rows)[self.fault_turn]
        self.assertEqual(r["gate"]["reason"], "format")
        self.assertEqual(r["committed"], r["truth"])
        self.assertEqual(r["hist_str"], str(r["truth"]))
        self.assertTrue(all(x["kind"] == "ok" for x in rows))

    def test_A1a_ignores_confident_slips(self):
        rows, _ = self.run_arm("A1a", {self.fault_key: "slip"})
        r = self.traj0(rows)[self.fault_turn]
        self.assertFalse(r["gate"]["fired"])
        self.assertEqual(r["kind"], "numeric_slip")

    def test_A1b_logit_gate_fixes_low_confidence_slip(self):
        rows, _ = self.run_arm("A1b", {self.fault_key: "slip"}, tau=-1.0)
        r0 = self.traj0(rows)
        self.assertEqual(r0[self.fault_turn]["gate"]["reason"], "logit")
        fired = [r for r in rows if r["gate"]["fired"]]
        self.assertEqual(len(fired), 1)  # confident correct answers never trigger
        self.assertTrue(all(r["kind"] == "ok" for r in rows))

    def test_A2_random_gate_rate(self):
        rows, _ = self.run_arm("A2", {}, rate=1.0)
        self.assertTrue(all(r["gate"]["reason"] == "random" for r in rows))
        rows, _ = self.run_arm("A2", {}, rate=0.0)
        self.assertFalse(any(r["gate"]["fired"] for r in rows))

    def _check_audit_repair(self, arm):
        m = 4
        rows, llm = self.run_arm(arm, {self.fault_key: "slip"}, m=m)
        r0 = self.traj0(rows)
        ft = self.fault_turn
        audit_turn = next(t for t in range(ft, self.T) if (t + 1) % m == 0)
        a = r0[audit_turn]["audit"]
        self.assertTrue(a["actually_wrong"])
        self.assertEqual(a["verdict"], "WRONG")
        self.assertEqual(a["action"], "correct")
        self.assertTrue(a["auditor_total_correct"])
        nxt = r0[audit_turn + 1]
        self.assertEqual(nxt["note"], [audit_turn, r0[audit_turn]["truth"]])
        self.assertEqual(nxt["prev"], r0[audit_turn]["truth"])
        self.assertTrue(all(r["kind"] == "ok" for r in r0[audit_turn + 1:]))
        self.assertTrue(all(r["kind"] == "inherited" for r in r0[ft + 1:audit_turn + 1]))
        # every audit on the clean trajectory says CORRECT and moves the verified total forward
        a1 = [r["audit"] for r in rows if r["sample_id"] == 1 and "audit" in r]
        self.assertTrue(all(x["verdict"] == "CORRECT" and x["action"] == "accept" for x in a1))
        return llm

    def test_A3_fresh_audit_repairs_chain(self):
        llm = self._check_audit_repair("A3")
        audits = [c for c in llm.calls if "Audit." in c]
        self.assertTrue(all("Key: " not in c.split("Audit.")[0] for c in audits))  # fresh context: no history

    def test_A4_in_context_audit_repairs_chain(self):
        llm = self._check_audit_repair("A4")
        audits = [c for c in llm.calls if "Audit." in c]
        self.assertTrue(all("Key: " in c.split("Audit.")[0] for c in audits))  # history precedes the audit

    def test_D1_carries_own_total_D2_carries_truth(self):
        rows, _ = self.run_arm("D1", {self.fault_key: "slip"})
        r0 = self.traj0(rows)
        ft = self.fault_turn
        self.assertEqual(r0[ft]["kind"], "numeric_slip")
        self.assertTrue(all(r["kind"] == "inherited" for r in r0[ft + 1:]))
        rows, _ = self.run_arm("D2", {self.fault_key: "slip"})
        r0 = self.traj0(rows)
        self.assertEqual(r0[ft]["kind"], "numeric_slip")
        self.assertTrue(all(r["kind"] == "ok" for r in r0[ft + 1:]))
        self.assertTrue(all(r["prev"] == (r0[i - 1]["truth"] if i else 0) for i, r in enumerate(r0)))

    def test_HC_evidence_true_and_placebo(self):
        cp = self.T - 1
        rows, llm = self.run_arm("A0", {self.fault_key: "slip"}, checkpoints=[cp])
        r0 = [r for r in rows if r["turn"] == cp]
        by = {r["sample_id"]: r for r in r0}
        n_true0 = sum(1 for r in self.traj0(rows)[max(0, cp - 9):cp + 1] if r["kind"] != "ok")
        self.assertEqual(by[0]["hc"]["n_true"], n_true0)
        self.assertEqual(by[1]["hc"]["n_true"], 0)
        self.assertEqual(by[0]["hc"]["n_placebo"], 0)      # taken from the other trajectory
        self.assertEqual(by[1]["hc"]["n_placebo"], n_true0)
        self.assertEqual(by[0]["probes"]["fore_true"], 100 - 10 * n_true0)
        self.assertEqual(by[0]["probes"]["fore_none"], 100)
        self.assertAlmostEqual(by[0]["probes"]["ptrue"], 0.9, places=6)


if __name__ == "__main__":
    unittest.main()
