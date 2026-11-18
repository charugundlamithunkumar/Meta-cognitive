"""Independent-sequence C runner. --dry-run is synthetic; --preflight never uses GPU."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from scripts.confirmation_common import settings, load_samples, source_paths
from scripts.run_meta import synthetic, context_audit as meta_context, check_meta_registration
from scripts.run_recovery import context_audit as recovery_context
from scripts.run_recovery_gpu2 import prepare_environment
from scripts.analyze_meta import read_bundle
from src.lhe_data import load_dictionary, sha256
from src.meta_history import ARMS as META_ARMS, MockTokenizer, existing_fake_class
from src.meta_runner import MetaEngine, run_meta_arm, archive_arm, atomic_text, validate_checkpoint as validate_meta
from src.recovery_history import ARMS as RECOVERY_ARMS
from src.recovery_runner import RecoveryEngine, RecoveryMock, run_recovery_arm, load_checkpoint
from src.experiment_runtime import gpu_snapshot, preserve_source
from src.recovery_resources import validate_inventory
from src.study_common import parse_ids


def registration():
    if not re.search(r'^Status: FROZEN\s*$',(ROOT/'prereg_confirmation.md').read_text(),re.M):
        raise ValueError('C is not frozen; registration and GPU approval precede real collection')
    return check_meta_registration(source_paths())


def configurations(family, ids, cps, mock, commit, hashes, resources):
    names=['M_'+a for a in META_ARMS] if family=='M' else list(RECOVERY_ARMS)
    sample_sha=sha256(ROOT/settings()['manifest'])
    result=[]
    for name in names:
        c=json.loads((ROOT/f'configs/{name}.json').read_text())
        c.update(ids=ids,checkpoints=cps,split='confirmation',mock=mock,
                 study_instance=settings()['study_instance'],confirmation_label='C_'+name,
                 sample_manifest_sha256=sample_sha,source_sha256=hashes,
                 registration_commit=commit,confirmatory=bool(commit),
                 resource_sha256=resources['sha256'] if resources else None)
        if mock:c.update(max_model_len=100000,mock_context_note='Character-token fixture only')
        result.append(c)
    return result


def validate_family(root, family, ids, cps):
    names=['M_'+a for a in META_ARMS] if family=='M' else list(RECOVERY_ARMS)
    for name in names:
        d=root/name;cfg,rows,summaries=read_bundle(d)
        if cfg['ids']!=ids or cfg['checkpoints']!=cps or sorted(s['sample_id'] for s in summaries)!=sorted(ids):
            raise ValueError('Incomplete confirmation trajectory set')
        grouped={}
        for row in rows:grouped.setdefault((row['sample_id'],row['checkpoint']),[]).append(row)
        for sid in ids:
            for cp in cps:
                if family=='M':
                    rr=json.loads((d/'checkpoints'/f'{sid}_{cp}.json').read_text())
                    validate_meta(rr,sid,cp,name)
                else:rr=load_checkpoint(d/'checkpoints'/f'{sid}_{cp}.json',sid,cp,cfg)
                if rr!=grouped.get((sid,cp)):raise ValueError('Checkpoint/export mismatch')
        archive_arm(d,root/f'{name}.tar.gz')
        for p,h in cfg['source_sha256'].items():
            if sha256(root/'source_snapshot'/p)!=h:raise ValueError('Source snapshot mismatch')


def worker(a, spec, ids, cps, commit, hashes):
    resources=None
    if a.dry_run:
        dictionary,trajectories=synthetic(ids,spec['T']);tok=MockTokenizer()
    else:
        resources=validate_inventory(json.loads((ROOT/spec['resource_manifest']).read_text()))
        _,trajectories=load_samples();dictionary=load_dictionary()
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(str(ROOT/'models/granite-4.2-8b'),local_files_only=True)
    configs=configurations(a.family,ids,cps,a.dry_run,commit,hashes,resources)
    audit=meta_context if a.family=='M' else recovery_context
    for c in configs:c['context_upper_bound']=audit(c,dictionary,trajectories,tok)
    print(json.dumps(dict(family=a.family,mock=a.dry_run,ids=ids,checkpoints=cps,
                          conditions=len(ids)*len(cps)*len(configs),
                          context_upper_bound=max(c['context_upper_bound'] for c in configs))),flush=True)
    if a.preflight:return
    manifest=dict(study_instance=spec['study_instance'],family=a.family,mock=a.dry_run,configs=configs,
                  sample_manifest_sha256=sha256(ROOT/spec['manifest']),resources=resources)
    out=a.out
    if out.exists():
        if not a.resume or json.loads((out/'manifest.json').read_text())!=manifest:
            raise ValueError('Existing family requires matching --resume')
    else:
        if a.resume:raise ValueError('Missing family resume path')
        out.mkdir(parents=True)
        atomic_text(out/'manifest.json',json.dumps(manifest,indent=2)+'\n')
    preserve_source(out,hashes)
    for c in configs:
        d=out/c['arm']
        if not d.exists():continue
        if json.loads((d/'config.json').read_text())!=c:raise ValueError('Resume arm config mismatch')
        expected={f'{sid}_{cp}.json' for sid in ids for cp in cps}
        for p in (d/'checkpoints').glob('*.json'):
            if p.name not in expected:raise ValueError('Unknown checkpoint in resume output')
            sid,cp=map(int,p.stem.split('_'))
            if a.family=='M':validate_meta(json.loads(p.read_text()),sid,cp,c['arm'])
            else:load_checkpoint(p,sid,cp,c)
    pending=any(not (out/c['arm']/'checkpoints'/f'{sid}_{cp}.json').is_file()
                for c in configs for sid in ids for cp in cps)
    llm=None
    if pending and a.dry_run:
        llm=existing_fake_class()(tok,dictionary,{}) if a.family=='M' else RecoveryMock(tok,dictionary)
    elif pending:
        while True:
            try:hardware=gpu_snapshot(spec['gpu'],configs[0]['util']);break
            except ValueError as e:
                print(str(e),flush=True);time.sleep(30)
        os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=spec['gpu'],HF_HUB_OFFLINE='1')
        from vllm import LLM
        c=configs[0]
        llm=LLM(str(ROOT/c['model_dir']),dtype='bfloat16',gpu_memory_utilization=c['util'],
                max_model_len=c['max_model_len'],enable_prefix_caching=True,seed=0,**c['engine_settings'])
        atomic_text(out/'hardware.json',json.dumps(hardware,indent=2)+'\n')
    for c in configs:
        engine_class=MetaEngine if a.family=='M' else RecoveryEngine
        engine=engine_class(c,dictionary,llm,tok,mock=a.dry_run) if llm else None
        run_arm=run_meta_arm if a.family=='M' else run_recovery_arm
        run_arm(c,trajectories,engine,out/c['arm'],resume=(out/c['arm']).exists())
        archive_arm(out/c['arm'],out/f"{c['arm']}.tar.gz")
    validate_family(out,a.family,ids,cps)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path);ap.add_argument('--family',choices=['M','MR'])
    ap.add_argument('--dry-run',action='store_true');ap.add_argument('--preflight',action='store_true')
    ap.add_argument('--describe',action='store_true');ap.add_argument('--resume',action='store_true')
    ap.add_argument('--ids');ap.add_argument('--checkpoints')
    a=ap.parse_args();spec=settings();ids=[10000+s for s in spec['fresh_seeds']];cps=spec['checkpoints']
    if (a.ids or a.checkpoints) and not a.dry_run:ap.error('Sample/checkpoint overrides are mock-only')
    if a.ids:ids=parse_ids(a.ids)
    if a.checkpoints:cps=[int(v) for v in a.checkpoints.split(',')]
    if not ids or len(set(ids))!=len(ids) or not set(ids)<=set(10000+s for s in spec['fresh_seeds']):ap.error('Invalid C ids')
    if not cps or cps!=sorted(set(cps)) or min(cps)<0 or max(cps)+10>=spec['T']:ap.error('Invalid checkpoints')
    if a.describe:
        print(json.dumps(dict(study=spec['study_instance'],gpu=spec['gpu'],ids=ids,checkpoints=cps,
                              conditions=11*len(ids)*len(cps),task_answers=110*len(ids)*len(cps),
                              families=['M','MR'],note='Read-only plan; no inference or allocation')));return
    if not a.out and not a.preflight:ap.error('--out is required')
    commit=registration() if not a.dry_run and not a.preflight else None
    hashes={p:sha256(ROOT/p) for p in source_paths()}
    if a.family:return worker(a,spec,ids,cps,commit,hashes)
    if a.preflight:
        for family in ['M','MR']:
            a.family=family;worker(a,spec,ids,cps,None,hashes)
        return
    if a.out.exists() and not a.resume:ap.error('Output exists; choose a new path or --resume')
    if a.resume and not (a.out/'status.json').exists():ap.error('Resume requires a saved parent job')
    tools=prepare_environment() if not a.dry_run else None
    signature=dict(study=spec['study_instance'],ids=ids,checkpoints=cps,mock=a.dry_run,
                   registration_commit=commit,source_sha256=hashes,
                   resource_manifest_sha256=sha256(ROOT/spec['resource_manifest']) if not a.dry_run else None,
                   runtime_tools=tools)
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads((a.out/'status.json').read_text()) if a.resume else dict(signature=signature,completed=[])
        if state['signature']!=signature:raise ValueError('Confirmation resume fingerprint mismatch')
        def save(status,**kw):
            state.update(status=status,updated=datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),pid=os.getpid(),**kw)
            atomic_text(a.out/'status.json',json.dumps(state,indent=2)+'\n')
        if state.get('status')=='complete':
            for family in ['M','MR']:validate_family(a.out/family,family,ids,cps)
            return
        save('checking');preserve_source(a.out,hashes)
        try:
            for family in ['M','MR']:
                if family in state['completed']:continue
                if commit and registration()!=commit:raise ValueError('C registration changed')
                save('inference',family=family)
                cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--family',family,'--out',str(a.out/family)]
                if a.dry_run:cmd+=['--dry-run','--ids',','.join(map(str,ids)),'--checkpoints',','.join(map(str,cps))]
                if (a.out/family).exists():cmd+=['--resume']
                with (a.out/f'{family}_inference.log').open('a') as log:
                    subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                state['completed'].append(family);save('inference')
            save('analysis')
            result=a.out/'analysis'
            if not result.exists():
                with (a.out/'analysis.log').open('a') as log:
                    subprocess.run([sys.executable,str(ROOT/'scripts/analyze_confirmation.py'),str(a.out),
                                    '--out',str(result),'--n-boot','100' if a.dry_run else '10000'],
                                   cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            if not all((result/name).is_file() for name in ['analysis.json','report.md','M_curves.png',
                       'accuracy.png','forecasts.png','hg_effects.png','secondary.png']):
                raise ValueError('Incomplete C analysis; preserve it and inspect')
            save('complete',analysis=str(result))
        except BaseException as e:save('failed',error=str(e));raise


if __name__=='__main__':main()
