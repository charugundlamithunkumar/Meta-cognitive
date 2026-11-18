import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.run_recovery_gpu2 import plan, prepare_environment


class TestGPU0Plan(unittest.TestCase):
    def test_gpu0_description_and_no_output(self):
        tag = 'gpu0_description_fixture'
        job = ROOT/'runs'/f'main_recovery_{tag}'
        self.assertFalse(job.exists())
        r = subprocess.run([sys.executable, 'scripts/run_recovery_gpu0.py',
                            '--describe', '--tag', tag], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        p = json.loads(r.stdout)
        self.assertEqual(p['gpu'], '0')
        self.assertEqual(p['task']['study'], 'HG')
        self.assertEqual(p['task_answers'], 8960)
        self.assertEqual(p['amendment'], 'prereg_recovery_gpu0_amendment.md')
        self.assertEqual(p['task']['inference'][p['task']['inference'].index('--gpu')+1], '0')
        self.assertFalse(job.exists())
        with self.assertRaises(ValueError):
            plan(tag, ROOT/'requirements.txt', gpu='9')

    def test_service_path_finds_python_environment_ninja(self):
        with patch.dict(os.environ, {'PATH': '/usr/bin'}):
            tools = prepare_environment()
            self.assertEqual(Path(tools['ninja']).parent, Path(sys.executable).parent)
            self.assertTrue(tools['ninja_version'])
            self.assertEqual(len(tools['ninja_sha256']), 64)


if __name__ == '__main__':
    unittest.main()
