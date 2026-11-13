"""Unit tests that need no GPU. Run: python3 -m unittest discover -s tests -v"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import lhe_data, labels, metrics, stats, prompts  # noqa: E402
from src.agent import parse_audit  # noqa: E402


class TestData(unittest.TestCase):
    def test_checksums_ok(self):
        lhe_data.verify()

    def test_corrupted_file_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for f in ["dictionary.json", "config.json"]:
                shutil.copy(lhe_data.DATA_DIR / f, d)
            Path(d, "test.jsonl").write_text('{"id": 0}\n')
            with self.assertRaises(RuntimeError):
                lhe_data.verify(d)

    def test_ground_truth_and_grouping(self):
        d = lhe_data.load_dictionary()
        tr1 = lhe_data.load_trajectories([0, 3], T=50, K=1)
        tr2 = lhe_data.load_trajectories([0], T=25, K=2, check=False)[0]
        for tr in tr1:
            total = 0
            for turn in tr.turns:
                self.assertEqual(turn.value, sum(d[k] for k in turn.keys))
                total += turn.value
                self.assertEqual(turn.truth, total)
        # K=2 turn j == K=1 turns 2j, 2j+1
        for j, turn in enumerate(tr2.turns):
            self.assertEqual(turn.keys, tr1[0].turns[2 * j].keys + tr1[0].turns[2 * j + 1].keys)
            self.assertEqual(turn.truth, tr1[0].turns[2 * j + 1].truth)

    def test_splits_disjoint(self):
        s = lhe_data.SPLITS
        allids = s["calibration"] + s["test"] + s["reserve"]
        self.assertEqual(sorted(allids), list(range(100)))


class TestLabels(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(labels.parse_reply("1523"), (1523, True, None))
        self.assertEqual(labels.parse_reply(" -42 "), (-42, True, None))
        self.assertEqual(labels.parse_reply("1,523"), (1523, True, None))
        self.assertEqual(labels.parse_reply("1500 + 23 = 1523"), (1523, False, None))
        self.assertEqual(labels.parse_reply("I think"), (None, False, None))
        self.assertEqual(labels.parse_reply("75 | 90", inline=True), (75, True, 90))
        self.assertEqual(labels.parse_reply("50 + 25 = 75 | 90", inline=True), (75, False, 90))
        self.assertEqual(labels.parse_reply("75", inline=True), (75, False, None))

    def test_chain(self):
        # truth: 10, 30, 25, 65 ; agent slips at turn 1 (writes 31), then inherits, then a non-answer
        vs = [10, 20, -5, 40]
        truths = [10, 30, 25, 65]
        replies = ["10", "31", "26", "oops"]
        prev, kinds = 0, []
        for v, tr, raw in zip(vs, truths, replies):
            val, wf, _ = labels.parse_reply(raw)
            kinds.append(labels.label_turn(val, wf, prev, v, tr))
            prev = labels.next_prev(prev, val)
        self.assertEqual(kinds, ["ok", "numeric_slip", "inherited", "nonanswer"])
        self.assertEqual(labels.assign_seeds(kinds), [None, 1, 1, 3])
        self.assertEqual(prev, 26)  # non-answer leaves the state unchanged

    def test_correction_mid_chain(self):
        # after a correction to the true total, a correct step is ok again and resets the seed
        kinds = ["ok", "numeric_slip", "inherited", "ok", "inherited"]
        self.assertEqual(labels.assign_seeds(kinds), [None, 1, 1, None, None])

    def test_retrieval_like(self):
        self.assertTrue(labels.retrieval_like(150, 100, 20, {20, 50}))
        self.assertFalse(labels.retrieval_like(149, 100, 20, {20, 50}))


class TestMetrics(unittest.TestCase):
    def test_auroc(self):
        rng = np.random.default_rng(0)
        y = np.r_[np.ones(50), np.zeros(50)]
        s = np.r_[rng.normal(1, 1, 50), rng.normal(0, 1, 50)]
        from sklearn.metrics import roc_auc_score
        self.assertAlmostEqual(metrics.auroc(y, s), roc_auc_score(y, s))
        self.assertTrue(np.isnan(metrics.auroc(np.r_[np.ones(50), np.zeros(5)], np.zeros(55))))

    def test_rates_and_youden(self):
        r = metrics.rates([1, 1, 0, 0], [1, 0, 0, 1])
        self.assertEqual((r["tpr"], r["fpr"], r["j"]), (0.5, 0.5, 0.0))
        t, j = metrics.youden_threshold([1, 1, 0, 0], [-0.1, -0.2, -3.0, -4.0])
        self.assertEqual(j, 1.0)
        self.assertTrue(-3.0 < t <= -0.2)

    def test_runs(self):
        k = ["ok", "x", "x", "ok", "x"]
        self.assertEqual(metrics.turns_to_first_error(k), 1)
        self.assertEqual(metrics.wrong_run_lengths(k), [2, 1])

    def test_token_stats(self):
        s = metrics.token_stats([-0.1, -2.0], [[-0.1, -2.5], [-2.0, -0.2]])
        self.assertAlmostEqual(s["lp_min"], -2.0)
        self.assertAlmostEqual(s["margin_min"], 1.8)


class TestStats(unittest.TestCase):
    def test_bootstrap(self):
        a = np.r_[np.ones(30)]
        b = np.r_[np.zeros(30)]
        r = stats.paired_bootstrap_diff(a, b)
        self.assertEqual((r["diff"], r["lo"], r["hi"]), (1.0, 1.0, 1.0))
        r2 = stats.paired_bootstrap_diff(np.arange(30.0), np.arange(30.0))
        self.assertEqual(r2["diff"], 0.0)

    def test_holm(self):
        self.assertEqual(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


class TestAuditParse(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_audit("<think>x</think>\nVERDICT: CORRECT")["verdict"], "CORRECT")
        r = parse_audit("<think>x</think>\nVERDICT: WRONG, total = -512")
        self.assertEqual((r["verdict"], r["total"]), ("WRONG", -512))
        self.assertFalse(parse_audit("<think>still thinking ... VERDICT: CORRECT")["finished"])
        self.assertFalse(parse_audit("<think>x</think> no verdict")["finished"])


class TestPrompts(unittest.TestCase):
    def test_no_numeric_confidence_example_inline(self):
        p = prompts.agent_system(lhe_data.load_dictionary(), inline=True)
        self.assertNotRegex(p.split("Dictionary")[0] + p.split("\n\n")[-1], r"\|\s*\d")

    def test_template_matches_official(self):
        """Our segment-built prompts must equal the model's official chat template exactly,
        for the model named in configs/A0.json, in both thinking and non-thinking mode."""
        cfg = json.loads((ROOT / "configs" / "A0.json").read_text())
        model = ROOT / cfg["model_dir"]
        if not (model / "tokenizer_config.json").exists():
            self.skipTest("model tokenizer not downloaded")
        from transformers import AutoTokenizer
        from src.agent import Chat
        tok = AutoTokenizer.from_pretrained(str(model))
        ch = Chat(tok)
        msgs = [dict(role="system", content="S line\nwith dictionary"), dict(role="user", content="Key: alarm"),
                dict(role="assistant", content="88"), dict(role="user", content="Key: coach"),
                dict(role="assistant", content="57"), dict(role="user", content="Key: doubt")]
        for think in (False, True):
            official = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=think)
            ours = (ch.system(msgs[0]["content"]) + ch.user("Key: alarm") + ch.asst("88") + ch.user("Key: coach")
                    + ch.asst("57") + ch.user("Key: doubt") + ch.gen(think=think))
            self.assertEqual(tok.decode(ours), official)
            self.assertEqual(ours, tok.encode(official, add_special_tokens=False))

    def test_low_effort_matches_official(self):
        cfg = json.loads((ROOT / "configs" / "A0.json").read_text())
        model = ROOT / cfg["model_dir"]
        if not (model / "tokenizer_config.json").exists():
            self.skipTest("model tokenizer not downloaded")
        from transformers import AutoTokenizer
        from src.agent import Chat
        tok = AutoTokenizer.from_pretrained(str(model))
        if "low_effort" not in (tok.chat_template or ""):
            self.skipTest("template has no low-effort mode")
        ch = Chat(tok)
        msgs = [dict(role="system", content="S"), dict(role="user", content="Audit. check")]
        official = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=True, low_effort=True)
        ours = ch.system("S") + ch.user("Audit. check" + prompts.LOW_EFFORT_SUFFIX) + ch.gen(think=True)
        self.assertEqual(tok.decode(ours), official)


class TestFresh(unittest.TestCase):
    def test_fresh_sequences(self):
        a = lhe_data.generate_fresh([0, 1], T=50)
        b = lhe_data.generate_fresh([0], T=50)
        d = lhe_data.load_dictionary()
        self.assertEqual(a[0].turns, b[0].turns)          # deterministic
        self.assertNotEqual(a[0].turns, a[1].turns)
        self.assertEqual(a[0].sample_id, lhe_data.FRESH_ID_OFFSET)
        tot = 0
        for t in a[1].turns:
            tot += sum(d[k] for k in t.keys)
            self.assertEqual(t.truth, tot)


if __name__ == "__main__":
    unittest.main()
