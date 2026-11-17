"""CPU-only M and analysis tests, including no tokenizer/model/data dependencies."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.analyze_meta import records, next_accuracy, matched_history, auc_summary, read_bundle, complete_a0
from scripts.analyze_meta_experiment import decision, calibration
from scripts.run_meta import synthetic
from src.meta_history import build_history, numeric_slip, MockTokenizer, existing_fake_class, ARMS
from src.meta_runner import MetaEngine, run_meta_arm, archive_arm
from src.metrics import auroc

ROOT = Path(__file__).resolve().parent.parent


def cfg_for(arm='L50'):
    cfg = json.loads((ROOT / 'configs' / f'M_{arm}.json').read_text())
    cfg.update(ids=[20, 21], checkpoints=[10, 25], mock=True)
    return cfg


class TestHistory(unittest.TestCase):
    def setUp(self):
        self.d, self.tr = synthetic([20, 21], 300)

    def test_exact_rate_rounding_and_nested_masks(self):
        for cp in [10, 25, 50, 100, 150, 200, 250]:
            previous = set()
            for rate in [0, .25, .5, .75, 1]:
                h = build_history(self.tr[0], cp, rate, self.d)
                wrong = {r['turn'] for r in h['turns'] if r['value'] != r['truth']}
                self.assertEqual(len(wrong), math.floor(rate * (cp + 1) + .5))
                self.assertEqual(wrong, set(h['injected_positions']))
                self.assertLessEqual(previous, wrong)
                self.assertTrue(all(r['value'] == r['truth'] + int(r['injected']) for r in h['turns']))
                previous = wrong

    def test_reuses_existing_slip_generator(self):
        for t in self.tr[0].turns[:10]:
            self.assertEqual(numeric_slip(t, self.d), t.truth + 1)

    def test_short_history_is_exact_suffix(self):
        for rate in [0, 1]:
            full = build_history(self.tr[0], 50, rate, self.d)
            short = build_history(self.tr[0], 50, rate, self.d, window=5)
            self.assertEqual(short['turns'], full['turns'][-5:])
            self.assertEqual([t['turn'] for t in short['turns']], [46, 47, 48, 49, 50])
            self.assertEqual(short['shown_length'], 5)

    def test_reproducible_and_invalid_inputs(self):
        a = build_history(self.tr[0], 25, .5, self.d)
        self.assertEqual(a, build_history(self.tr[0], 25, .5, self.d))
        self.assertNotEqual(a['injected_positions'], build_history(self.tr[1], 25, .5, self.d)['injected_positions'])
        for kwargs in [dict(rate=-.1), dict(rate=1.1), dict(rate=.5, window=0)]:
            with self.assertRaises(ValueError):
                build_history(self.tr[0], 25, dictionary=self.d, **kwargs)


class TestMetaRunner(unittest.TestCase):
    def setUp(self):
        self.d, self.tr = synthetic([20, 21], 300)
        self.tok = MockTokenizer()

    def engine(self, cfg):
        return MetaEngine(cfg, self.d, existing_fake_class()(self.tok, self.d, {}), self.tok, mock=True)

    def test_clean_accuracy_and_contaminated_inheritance(self):
        for arm, expected in [('L0', 1.), ('L100', 0.), ('S0', 1.), ('S100', 0.)]:
            cfg = cfg_for(arm)
            rows = self.engine(cfg).checkpoint(self.tr[0], 10, cfg['arm'])
            self.assertEqual(rows[0]['realized_next10'], expected)
            self.assertEqual([r['turn'] for r in rows[1:]], list(range(11, 21)))
            self.assertTrue(all(r['kind'] == ('ok' if expected else 'inherited') for r in rows[1:]))
            self.assertTrue(all('lp' in r and set(r['probes']) == {'verbal', 'ptrue'} for r in rows[1:]))

    def test_probe_forks_do_not_enter_continuation(self):
        cfg = cfg_for('S0'); engine = self.engine(cfg)
        engine.checkpoint(self.tr[0], 10, cfg['arm'])
        calls = engine.llm.calls
        # First continuation call follows the three independent checkpoint probes.
        for q in calls[:3]:
            self.assertEqual(q.count('<|im_start|>assistant\n'), 6)  # five history answers + generation
        continuation = calls[3]
        self.assertNotIn('How confident are you', continuation)
        self.assertNotIn('Over your next 10 turns', continuation)
        self.assertNotIn('Compared with the start', continuation)
        self.assertEqual(continuation.count('<|im_start|>user\n'), 6)

    def test_resume_after_crash_is_identical_and_config_guarded(self):
        cfg = cfg_for()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); ref = root / 'ref'; crashed = root / 'crashed'
            run_meta_arm(cfg, self.tr, self.engine(cfg), ref)
            engine = self.engine(cfg); original = engine.checkpoint; calls = 0
            def fail(*args):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('simulated mid-run failure')
                return original(*args)
            engine.checkpoint = fail
            with self.assertRaises(RuntimeError):
                run_meta_arm(cfg, self.tr, engine, crashed)
            # An interrupted write is ignored; a stale exported JSONL is repaired.
            (crashed / 'checkpoints/20_25.json.tmp').write_text('{partial')
            (crashed / 'rows.jsonl').write_text('{partial')
            engine = self.engine(cfg)
            run_meta_arm(cfg, self.tr, engine, crashed, resume=True)
            self.assertEqual((ref / 'rows.jsonl').read_bytes(), (crashed / 'rows.jsonl').read_bytes())
            self.assertEqual((ref / 'trajectories.jsonl').read_bytes(), (crashed / 'trajectories.jsonl').read_bytes())
            before = len(engine.llm.calls)
            run_meta_arm(cfg, self.tr, engine, crashed, resume=True)
            self.assertEqual(before, len(engine.llm.calls))
            with self.assertRaises(FileExistsError):
                run_meta_arm(cfg, self.tr, engine, crashed)
            changed = dict(cfg, seed=123)
            with self.assertRaises(ValueError):
                run_meta_arm(changed, self.tr, engine, crashed, resume=True)
            archive = root / 'M_L50.tar.gz'
            archive_arm(crashed, archive)
            archive_arm(crashed, archive)  # read-only complete resume
            stored_cfg, rr, summaries = read_bundle(archive)
            self.assertEqual(stored_cfg, cfg)
            self.assertEqual(len(rr), 44)
            self.assertEqual(len(summaries), 2)
            self.assertEqual({r['sample_id'] for r in rr}, {20, 21})

    def test_context_limit_fails_without_truncation(self):
        cfg = cfg_for(); cfg['max_model_len'] = 10
        with self.assertRaises(ValueError):
            self.engine(cfg).checkpoint(self.tr[0], 10, cfg['arm'])


class TestMetaAnalysis(unittest.TestCase):
    def fixture(self):
        by = {}
        for sid in range(4):
            by[sid] = [dict(sample_id=sid, turn=t, kind='numeric_slip' if t == sid+1 else 'ok',
                            probes=dict(verbal=20 if t > sid+1 else 100, ptrue=.5)) for t in range(15)]
        return by

    def test_history_excludes_current_error_and_future_window(self):
        by = self.fixture(); rr = records(by)
        r = next(r for r in rr if r['sample_id'] == 0 and r['turn'] == 1)
        self.assertFalse(r['prior_error']); self.assertTrue(r['inclusive_error'])
        self.assertAlmostEqual(next_accuracy(by[0], 0), .9)
        self.assertEqual(next_accuracy(by[0], 1), 1.)
        self.assertTrue(np.isnan(next_accuracy(by[0], 5)))
        self.assertTrue(np.isnan(next_accuracy([r for r in by[0] if r['turn'] != 4], 0)))

    def test_exact_matching_counts_and_sparse_ci(self):
        rr = records(self.fixture()); result = matched_history(rr, n_boot=100)
        cells = [r for r in result['cells'] if r['turn'] == 2]
        self.assertEqual([r['n'] for r in cells], [3, 1])
        self.assertEqual([r['mean'] for r in cells], [100., 20.])
        self.assertTrue(all(r['sparse'] for r in cells))
        self.assertEqual(cells[0]['lo'], 100.)
        self.assertEqual(cells[1]['hi'], 20.)
        self.assertEqual(result['error_minus_clean'], -80.)

    def test_auc_matches_existing_definition_and_brute_bootstrap(self):
        rows = []
        for sid in range(4):
            for t, (kind, score) in enumerate([('ok', .8), ('inherited', .8), ('numeric_slip', .2), ('numeric_slip', .8)]):
                rows.append(dict(sample_id=sid, kind=kind, probes=dict(verbal=score + sid*.05)))
        report, samples = auc_summary(rows, 'verbal', n_boot=100, min_per_class=1, return_samples=True)
        self.assertEqual(report['n_local'], 8); self.assertEqual(report['n_slip'], 8)
        self.assertAlmostEqual(report['point'], auroc([r['kind'] != 'numeric_slip' for r in rows],
                                                    [r['probes']['verbal'] for r in rows], min_per_class=1))
        w = np.random.default_rng(0).multinomial(4, [.25]*4, size=100)
        for weights, got in zip(w, samples):
            selected = [r for sid, count in enumerate(weights) for _ in range(count) for r in rows if r['sample_id'] == sid]
            want = auroc([r['kind'] != 'numeric_slip' for r in selected], [r['probes']['verbal'] for r in selected], min_per_class=1)
            self.assertAlmostEqual(got, want)
        self.assertTrue(np.isnan(auc_summary(rows, 'verbal', n_boot=10)['point']))  # 15/class guard
        self.assertTrue(np.isnan(auc_summary([], 'verbal', n_boot=10)['point']))

    def test_hf_known_mechanisms_and_boundary(self):
        n = 32
        length = np.full(n, -20.); zero = np.zeros(n); dose = np.full(n, 20.)
        self.assertEqual(decision(length, zero, zero, 100)['verdict'], 'length-driven')
        self.assertEqual(decision(zero, dose, dose, 100)['verdict'], 'monitoring-driven (error-sensitive reports)')
        self.assertEqual(decision(length, dose, dose, 100)['verdict'], 'mixed')
        self.assertEqual(decision(zero, zero, zero, 100)['verdict'], 'inconclusive')
        self.assertEqual(decision(np.full(n, -10.), zero, zero, 100)['verdict'], 'inconclusive')
        self.assertEqual(decision(np.full(n, 20.), zero, zero, 100)['verdict'], 'inconclusive')

    def test_calibration_known_slope_and_unidentifiable_floor(self):
        points = [dict(sample_id=sid, checkpoint=c, verbal=x, realized=.2 + .5*x + (c==25)*.1)
                  for sid in range(5) for c in [10, 25] for x in [.1, .5, .9]]
        result = calibration(points, 'verbal', n_boot=100)
        self.assertAlmostEqual(result['slope'], .5)
        self.assertAlmostEqual(result['intercept'], .2)
        self.assertAlmostEqual(result['lo'], .5)
        floor = [dict(p, realized=0.) for p in points]
        self.assertIn('unidentifiable', calibration(floor, 'verbal', 100)['status'])
        constant = [dict(p, verbal=1.) for p in points]
        self.assertIn('unidentifiable', calibration(constant, 'verbal', 100)['status'])


if __name__ == '__main__':
    unittest.main()
