import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.run_recovery_gpu2 import plan


class TestGPU2Plan(unittest.TestCase):
    def test_only_recovery_on_new_gpu_and_output(self):
        p = plan('gpu2_description_fixture', ROOT/'requirements.txt')
        self.assertEqual(p['task']['study'], 'HG')
        self.assertEqual(p['task_answers'], 8960)
        self.assertEqual(p['task']['inference'][p['task']['inference'].index('--gpu')+1], '2')
        self.assertNotIn('run_meta.py', ' '.join(p['task']['inference']))
        self.assertIn('gpu2_description_fixture', p['task']['run'])
        with self.assertRaises(ValueError):
            plan('../unsafe', ROOT/'requirements.txt')

    def test_description_is_read_only(self):
        tag = 'gpu2_description_fixture'
        job = ROOT/'runs'/f'main_recovery_{tag}'
        self.assertFalse(job.exists())
        r = subprocess.run([sys.executable, 'scripts/run_recovery_gpu2.py',
                            '--describe', '--tag', tag], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout)['conditions'], 896)
        self.assertFalse(job.exists())


if __name__ == '__main__':
    unittest.main()
