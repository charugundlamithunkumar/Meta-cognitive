"""CPU acceptance and statistical fixtures for HG; no model or GPU loads."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.run_meta import synthetic
from scripts.run_recovery import check_registration, context_audit, calibration_profile
from scripts.analyze_recovery import decision, test_contrasts, analyze_suite, bundle_suite
from scripts.recovery_precision import simulate
from src.meta_history import MockTokenizer
from src.meta_runner import archive_arm
from src.recovery_history import ARMS, CHECKPOINTS, build_recovery_history
from src.recovery_resources import inventory, validate_inventory
from src.recovery_runner import RecoveryEngine, RecoveryMock, run_recovery_arm, numeric_probe, validate_checkpoint

ROOT=Path(__file__).resolve().parent.parent


def cfg(arm='MR_A'):
    c=json.loads((ROOT/f'configs/{arm}.json').read_text())
    c.update(ids=[20,21],checkpoints=[10,25],mock=True,split='test',source_sha256={},
             resource_sha256=None,registration_commit=None,confirmatory=False)
    return c


class TestRecovery(unittest.TestCase):
    def setUp(self):
        self.d,self.tr=synthetic([20,21],300);self.tok=MockTokenizer()

    def engine(self,c):
        return RecoveryEngine(c,self.d,RecoveryMock(self.tok,self.d),self.tok,mock=True)

    def test_exact_masks_and_pairing(self):
        for cp in CHECKPOINTS:
            h={a:build_recovery_history(self.tr[0],cp,a,self.d) for a in ARMS}
            self.assertEqual(h['MR_A']['turns'],h['MR_D']['turns'])
            self.assertEqual(h['MR_B']['turns'],h['MR_C']['turns'])
            for a in ARMS:
                n=(cp+1)//2+1 if a in ['MR_B','MR_C'] else 0
                self.assertEqual(len(h[a]['injected_positions']),n)
                for r in h[a]['turns']:
                    self.assertEqual(r['value'],r['truth']+int(r['injected']))
            self.assertNotEqual(h['MR_A']['supplied_state'],h['MR_D']['supplied_state'])
            self.assertEqual(h['MR_A']['supplied_state'],h['MR_C']['supplied_state'])
        with self.assertRaises(ValueError):build_recovery_history(self.tr[0],299,'MR_A',self.d)
        with self.assertRaises(ValueError):build_recovery_history(self.tr[0],10,'A0',self.d)

    def test_first_state_labels_and_isolated_forks(self):
        for arm in ARMS:
            c=cfg(arm);e=self.engine(c);rows=e.checkpoint(self.tr[0],10,arm)
            validate_checkpoint(rows,20,10,arm,True)
            self.assertEqual(rows[0]['realized_next10'],1. if arm in ['MR_A','MR_C'] else 0.)
            self.assertEqual(rows[1]['prev'],rows[0]['history']['supplied_state'])
            task=[s for s in e.llm.calls if 'Key: ' in s.split('<|im_start|>user\n')[-1]]
            self.assertEqual(len(task),10)
            for s in task:
                self.assertEqual(s.count('Continue using this current running total:'),1)
                for probe in ['How confident are you','Over your next 10 turns','Compared with the start','Is the running total you just reported']:
                    self.assertNotIn(probe,s)
            self.assertNotIn('Continue using',e.llm.calls[0])
            self.assertIn('Continue using',e.llm.calls[1])
            self.assertIn('Compared with the start',e.llm.calls[-1])
            self.assertEqual(len([r for r in rows if r['record_type']=='continuation']),10)

    def test_missing_raw_and_probability_mass(self):
        self.assertEqual(numeric_probe('999')['value'],100)
        self.assertTrue(numeric_probe('999')['out_of_range'])
        for raw in ['oops','-1','50 percent','']:
            self.assertIsNone(numeric_probe(raw)['value'])
        e=self.engine(cfg())
        co=type('Output',(),dict(text='?',logprobs=[{}]))()
        self.assertIsNone(e.probability(co,[e.yes,e.no])['probabilities'])
        self.assertEqual(e.probability(co,[e.yes,e.no])['retained_mass'],0.)
        original=e.llm.generate
        def invalid_forecast(prompts,params,**kwargs):
            if any('Over your next 10 turns' in e.tok.decode(p['prompt_token_ids']) for p in prompts):
                return [e.llm.base._out('oops',with_lps=False)]
            return original(prompts,params,**kwargs)
        e.llm.generate=invalid_forecast
        rows=e.checkpoint(self.tr[0],10,'MR_A')
        self.assertIsNone(rows[0]['post_forecast']['value'])
        validate_checkpoint(rows,20,10,'MR_A',True)

    def test_context_guard_and_preflight_bound(self):
        c=cfg();c['max_model_len']=10
        with self.assertRaises(ValueError):self.engine(c).checkpoint(self.tr[0],10,'MR_A')
        with self.assertRaises(ValueError):context_audit(c,self.d,self.tr,self.tok)
        self.assertLess(context_audit(cfg(),self.d,self.tr,self.tok),cfg()['max_model_len'])

    def test_resume_crash_archives_and_corruption(self):
        c=cfg('MR_C')
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);ref=root/'ref';out=root/'out'
            run_recovery_arm(c,self.tr,self.engine(c),ref)
            e=self.engine(c);original=e.checkpoint;count=0
            def crash(*args):
                nonlocal count
                count+=1
                if count==2:raise RuntimeError('simulated crash')
                return original(*args)
            e.checkpoint=crash
            with self.assertRaises(RuntimeError):run_recovery_arm(c,self.tr,e,out)
            (out/'checkpoints/20_25.json.tmp').write_text('{partial')
            (out/'rows.jsonl').write_text('{partial')
            run_recovery_arm(c,self.tr,self.engine(c),out,resume=True)
            self.assertEqual((out/'rows.jsonl').read_bytes(),(ref/'rows.jsonl').read_bytes())
            run_recovery_arm(c,self.tr,None,out,resume=True)
            for field,value in [('seed',1),('resource_sha256','changed'),('source_sha256',{'x':'changed'})]:
                with self.assertRaises(ValueError):run_recovery_arm(dict(c,**{field:value}),self.tr,None,out,resume=True)
            with self.assertRaises(FileExistsError):run_recovery_arm(c,self.tr,None,out)
            archive=root/'MR_C.tar.gz';archive_arm(out,archive);old=archive.read_bytes();archive_arm(out,archive)
            self.assertEqual(old,archive.read_bytes())
            (out/'rows.jsonl').write_text('changed')
            with self.assertRaises(ValueError):archive_arm(out,archive)
            p=out/'checkpoints/20_10.json';envelope=json.loads(p.read_text());envelope['rows'][0]['realized_next10']=.5
            p.write_text(json.dumps(envelope))
            with self.assertRaises(ValueError):run_recovery_arm(c,self.tr,None,out,resume=True)

    def test_analysis_mock_missing_and_archive_inputs(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for a in ARMS:
                c=cfg(a);run_recovery_arm(c,self.tr,self.engine(c),root/a)
                archive_arm(root/a,root/f'{a}.tar.gz')
            report=analyze_suite(root,n_boot=100)
            self.assertEqual(report['primary']['verdict'],'withheld: mock or exploratory run')
            self.assertTrue(all(r['scores']['lp_min']['point'] is None for r in report['auroc']))
            self.assertEqual(report['checkpoint_summary'][0]['accuracy'],100.)
            archive_only=root/'archives';archive_only.mkdir()
            for a in ARMS:(archive_only/f'{a}.tar.gz').write_bytes((root/f'{a}.tar.gz').read_bytes())
            self.assertEqual(analyze_suite(archive_only,100)['checkpoint_summary'],report['checkpoint_summary'])
            (archive_only/'MR_D.tar.gz').unlink()
            self.assertIn('missing arms',analyze_suite(archive_only,100)['primary']['verdict'])
            p=root/'MR_A/rows.jsonl';rows=p.read_text();p.write_text(rows+rows.splitlines()[0]+'\n')
            with self.assertRaises(ValueError):bundle_suite(root)

    def test_registration_and_cli_guards(self):
        with patch('pathlib.Path.read_text', return_value='Draft registration'):
            with self.assertRaises(ValueError):check_registration()
        for args in [['--split','test','--checkpoints','10'],['--split','calibration','--ids','20'],
                     ['--resume'],['--split','test']]:
            r=subprocess.run([sys.executable,'scripts/run_recovery.py']+args,cwd=ROOT,capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0)

    def test_small_profile_only_changes_calibration_runtime(self):
        base=cfg()
        small=calibration_profile(base,'gpu1-small','calibration')
        self.assertEqual(base['util'],json.loads((ROOT/'configs/MR_A.json').read_text())['util'])
        self.assertEqual(base['max_model_len'],json.loads((ROOT/'configs/MR_A.json').read_text())['max_model_len'])
        self.assertEqual(small['util'],.38)
        self.assertEqual(small['max_model_len'],2048)
        self.assertTrue(small['engine_settings']['enforce_eager'])
        self.assertEqual(small['seed'],base['seed'])
        with self.assertRaises(ValueError):calibration_profile(base,'gpu1-small','test')
        long=calibration_profile(base,'gpu1-long','calibration')
        self.assertEqual(long['max_model_len'],6144)
        self.assertEqual(long['util'],.38)
        with self.assertRaises(ValueError):calibration_profile(base,'gpu1-long','test')

    def test_resource_manifest_detects_weight_and_software_changes(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);model=root/'models/granite-4.2-8b';data=root/'data/lhe'
            model.mkdir(parents=True);data.mkdir(parents=True)
            for name in ['config.json','tokenizer.json','tokenizer_config.json']:
                (model/name).write_text('{}')
            (model/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'w':'shard.safetensors'}}))
            shard=model/'shard.safetensors';shard.write_bytes(b'fixture weight')
            for name in ['dictionary.json','test.jsonl','config.json']:(data/name).write_text('{}')
            with patch('src.recovery_resources.verify'):
                saved=inventory(root)
                self.assertEqual(validate_inventory(saved,root),saved)
                shard.write_bytes(b'changed weight')
                with self.assertRaises(ValueError):validate_inventory(saved,root)


class TestHGStatistics(unittest.TestCase):
    def fixture(self):
        r=np.tile([80.,30.,80.,30.],(32,7,1));f=np.tile([80.,30.,50.,30.],(32,7,1))
        return r,f

    def test_all_four_and_boundaries(self):
        r,f=self.fixture();result=decision(r,f,1000)
        self.assertIn('supported',result['verdict'])
        self.assertTrue(all(t['p']>0 for t in result['tests'].values()))
        for delta in [-5,5]:
            rr=r.copy();rr[:,:,2]=rr[:,:,0]+delta
            self.assertFalse(decision(rr,f,1000)['tests']['HG2']['supported'])
        ff=f.copy();ff[:,:,0]=ff[:,:,2]+10
        self.assertFalse(decision(r,ff,1000)['tests']['HG3']['supported'])

    def test_no_recovery_floor_missing_and_overconfident_reference(self):
        r,f=self.fixture();rr=r.copy();rr[:,:,2]=30
        self.assertEqual(decision(rr,f,1000)['verdict'],'inconclusive')
        self.assertEqual(decision(r*0,f,1000)['verdict'],'floor-limited')
        ff=f.copy();ff[0,0,0]=np.nan
        self.assertIn('missing',decision(r,ff,1000)['verdict'])
        rr=r.copy();rr[:,:,[0,2]]=60;ff=f.copy();ff[:,:,0]=100;ff[:,:,2]=70
        result=decision(rr,ff,1000)
        self.assertTrue(result['tests']['HG3']['supported'])
        self.assertFalse(result['tests']['HG4']['supported'])

    def test_trajectory_bootstrap_not_checkpoint_pseudoreplication(self):
        r,f=self.fixture();f[:,:,0]+=np.arange(32)[:,None]/10
        one=decision(r[:,:1,:],f[:,:1,:],500)
        repeated=decision(r,f,500)
        self.assertEqual(one['tests'],repeated['tests'])

    def test_null_simulation(self):
        result=simulate(n=32,reps=100,n_boot=500,seed=19,sigma=2.)
        for scenario,hypothesis in [('HG1_null','HG1'),('HG2_boundary','HG2'),('HG3_boundary','HG3'),('HG4_boundary','HG4')]:
            s=result['scenarios'][scenario]
            self.assertEqual(s['valid_replicates'],100)
            self.assertLess(s['raw_rejection_rate'][hypothesis],.15)

    def test_contrast_refactor_and_large_residual_simulation(self):
        r,f=self.fixture();rr,ff=r.mean(axis=1),f.mean(axis=1)
        contrasts=np.column_stack([rr[:,2]-rr[:,1],rr[:,2]-rr[:,0],ff[:,0]-ff[:,2],rr[:,2]-ff[:,2]])
        self.assertEqual(test_contrasts(contrasts,500),decision(r,f,500)['tests'])
        residuals=np.random.default_rng(3).normal(0,30,size=(8,2,4))
        result=simulate(n=32,reps=10,n_boot=100,pilot=residuals,variance_multiplier=2)
        for s in result['scenarios'].values():
            self.assertEqual(s['valid_replicates'],10)
            self.assertEqual(s['rejected_out_of_bounds'],0)


if __name__=='__main__':unittest.main()
