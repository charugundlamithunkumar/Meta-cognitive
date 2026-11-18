import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.confirmation_common import load_samples,content_hash
from scripts.run_confirmation import registration


class TestConfirmation(unittest.TestCase):
    def test_manifest_reproduces_fresh_independent_content(self):
        m,tr=load_samples()
        self.assertEqual(m['ids'],list(range(12000,12032)))
        self.assertEqual(len({content_hash(t) for t in tr}),32)
        self.assertFalse(set(m['fresh_seeds'])&set(range(1000,1048)))
        bad=dict(m,ids=[20]*32)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.json';p.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):load_samples(p)

    def test_mock_full_families_resume_and_withheld_verdict(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'C'
            cmd=[sys.executable,'scripts/run_confirmation.py','--dry-run','--ids','12000-12001',
                 '--checkpoints','10,25','--out',str(out)]
            r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            state=json.loads((out/'status.json').read_text());self.assertEqual(state['status'],'complete')
            report=json.loads((out/'analysis/analysis.json').read_text())
            self.assertFalse(report['eligibility']['eligible'])
            self.assertTrue(report['C_HF']['verdict'].startswith('withheld'))
            self.assertTrue(report['C_HG']['verdict'].startswith('withheld'))
            self.assertEqual(len(list(out.glob('*/checkpoints/*.json'))),0)
            self.assertEqual(len(list(out.glob('*/*/checkpoints/*.json'))),44)
            before={str(p):p.read_bytes() for p in out.glob('*/*/checkpoints/*.json')}
            r=subprocess.run(cmd+['--resume'],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            self.assertEqual(before,{str(p):p.read_bytes() for p in out.glob('*/*/checkpoints/*.json')})
            first=next(out.glob('M/*/checkpoints/*.json'));first.write_text('[]')
            r=subprocess.run(cmd+['--resume'],cwd=ROOT,capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0,'Corrupt committed checkpoint must be rejected on resume')

    def test_real_collection_requires_registration_and_no_sample_override(self):
        text=(ROOT/'prereg_confirmation.md').read_text()
        if 'Status: DRAFT' in text:
            with self.assertRaises(ValueError):registration()
        r=subprocess.run([sys.executable,'scripts/run_confirmation.py','--ids','20','--out','unused'],
                         cwd=ROOT,capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0)
        self.assertIn('mock-only',r.stderr)


if __name__=='__main__':unittest.main()
