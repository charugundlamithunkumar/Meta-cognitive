"""M calibration infrastructure must preserve protocol and block test overrides."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

from scripts.run_meta import context_audit, synthetic
from src.experiment_runtime import calibration_profile
from src.meta_history import MockTokenizer

ROOT = Path(__file__).resolve().parent.parent


class TestMetaRuntime(unittest.TestCase):
    def test_calibration_profile_and_context_guard(self):
        cfg = json.loads((ROOT/'configs/M_L0.json').read_text())
        cfg['checkpoints'] = [10]
        small = calibration_profile(cfg, 'gpu1-small', 'calibration')
        self.assertEqual(cfg['util'], json.loads((ROOT/'configs/M_L0.json').read_text())['util'])
        self.assertEqual(small['util'], .38)
        self.assertEqual(small['prompt_version'], cfg['prompt_version'])
        with self.assertRaises(ValueError):
            calibration_profile(cfg, 'gpu1-long', 'test')
        dictionary, trajectories = synthetic([0], 300)
        cfg['max_model_len'] = 100000
        self.assertGreater(context_audit(cfg, dictionary, trajectories, MockTokenizer()), 0)
        cfg['max_model_len'] = 10
        with self.assertRaises(ValueError):
            context_audit(cfg, dictionary, trajectories, MockTokenizer())

    def test_cli_rejects_test_profile_before_inference(self):
        r = subprocess.run([sys.executable, 'scripts/run_meta.py', '--split', 'test',
                            '--calibration-profile', 'gpu1-long'], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('calibration-only', r.stderr)


if __name__ == '__main__':
    unittest.main()
