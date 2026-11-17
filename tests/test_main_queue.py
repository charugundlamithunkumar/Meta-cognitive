"""Main queue must stay read-only before registration and protect paired contrasts."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from scripts.run_registered_meta import commands
from scripts.meta_precision import contrasts_from_confidence
from src.meta_history import ARMS

ROOT=Path(__file__).resolve().parent.parent


class TestMainQueue(unittest.TestCase):
    def test_describe_is_read_only_and_has_two_guarded_studies(self):
        tag='cpu_queue_description_fixture'
        root=ROOT/'runs'/('main_meta_'+tag)
        self.assertFalse(root.exists())
        r=subprocess.run([sys.executable,'scripts/run_registered_meta.py','--describe','--tag',tag],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        d=json.loads(r.stdout);self.assertEqual(d['task_answers'],24640)
        self.assertEqual([t['study'] for t in d['tasks']],['HF','HG'])
        self.assertTrue(all('--split' in t['inference'] and 'test' in t['inference'] for t in d['tasks']))
        self.assertFalse(root.exists())
        with self.assertRaises(ValueError):commands('../unsafe','1',ROOT/'requirements.txt')

    def test_hf_precision_uses_registered_paired_contrasts(self):
        dose={'L0':0.,'L25':.25,'L50':.5,'L75':.75,'L100':1.,'S0':0.,'S100':1.}
        confidence={a:np.full((8,7),80-20*dose[a]) for a in ARMS}
        got=contrasts_from_confidence(confidence)
        np.testing.assert_allclose(got,np.tile([0,20,20],(8,1)))
        confidence['S0']+=30
        np.testing.assert_allclose(contrasts_from_confidence(confidence)[:,0],30)


if __name__=='__main__':unittest.main()
