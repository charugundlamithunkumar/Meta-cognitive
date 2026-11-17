"""HG runner: --dry-run is CPU synthetic; --preflight is read-only and never loads weights."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.run_meta import synthetic, check_meta_registration
from src.study_common import parse_ids
from src.lhe_data import SPLITS, REVISION, verify, load_dictionary, load_trajectories, sha256
from src.meta_history import MockTokenizer
from src.meta_runner import atomic_text, archive_arm
from src.recovery_history import ARMS, CHECKPOINTS, fingerprint, build_recovery_history
from src.recovery_runner import RecoveryEngine, RecoveryMock, run_recovery_arm, load_checkpoint
from src.recovery_prompts import state_cue
from src.recovery_resources import resource_paths, validate_inventory
from src import prompts as PR
from src.agent import Chat
from src.experiment_runtime import calibration_profile, gpu_snapshot, preserve_source


def source_files():
    return (['prereg_recovery.md', 'requirements.txt', 'scripts/run_recovery.py',
             'scripts/recovery_resources.py', 'scripts/analyze_recovery.py',
             'scripts/recovery_precision.py', 'scripts/analyze_meta.py',
             'scripts/analyze_meta_experiment.py', 'scripts/run_meta.py',
             'previous work/tests/test_arms_mock.py', 'previous work/configs/A0.json',
             'configs/recovery_calibration_gpu1.json', 'configs/recovery_calibration_gpu1_long.json',
             'scripts/run_registered_meta.py']
            + sorted(str(p.relative_to(ROOT)) for p in (ROOT/'src').glob('*.py'))
            + sorted(str(p.relative_to(ROOT)) for p in (ROOT/'tests').glob('*.py'))
            + [f'configs/{arm}.json' for arm in ARMS])


def check_registration():
    text = (ROOT/'prereg_recovery.md').read_text()
    if not re.search(r'^Status: FROZEN\s*$', text, re.M):
        raise ValueError('HG is a draft; finish calibration and freeze registration before test inference')
    return check_meta_registration(source_files())


def context_audit(cfg, dictionary, trajectories, tokenizer):
    ch = Chat(tokenizer)
    system = ch.system(PR.agent_system(dictionary, K=1, version=2))
    maximum = 0
    for tr in trajectories:
        for cp in cfg['checkpoints']:
            h = build_recovery_history(tr, cp, cfg['arm'], dictionary, cfg['seed'])
            base = list(system)
            for turn in h['turns']:
                base += ch.user(turn['user']) + ch.asst(turn['assistant'])
            cue = state_cue(h['supplied_state'])
            for q in [PR.P_FORECAST, cue+'\n'+PR.P_FORECAST]:
                maximum = max(maximum, len(base + ch.user(q) + ch.gen())+6)
            # Conservative for constrained output: enumerate all allowed integer characters.
            # Future user prompts/cue are exact; each assistant may occupy up to 16 tokens.
            length = len(base)
            assistant_overhead = len(ch.asst(''))
            for offset in range(1, 11):
                q = PR.keys_text(tr.turns[cp+offset].keys, version=2)
                if offset == 1:
                    q = cue+'\n'+q
                length += len(ch.user(q))
                maximum = max(maximum, length+len(ch.gen())+16)
                length += assistant_overhead+16
                maximum = max(maximum, length+max(len(ch.user(p)) for p in
                              [PR.P_VERBAL, PR.P_TRUE, PR.P_AWARE])+len(ch.gen())+6)
    if maximum > cfg['max_model_len']:
        raise ValueError(f'Context upper bound {maximum} exceeds {cfg["max_model_len"]}; no truncation')
    return maximum


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    ap.add_argument('--split', choices=['calibration', 'test'])
    ap.add_argument('--ids')
    ap.add_argument('--checkpoints')
    ap.add_argument('--dry-run', '--mock', dest='mock', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--resource-manifest', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--calibration-profile', choices=['gpu1-small', 'gpu1-long'])
    a = ap.parse_args()
    try:
        if not a.mock and not a.split:
            raise ValueError('Choose --split calibration or test; use --dry-run for CPU mocks')
        if len(a.arms) != len(set(a.arms)):
            raise ValueError('Duplicate arms')
        split = a.split or 'test'
        ids = parse_ids(a.ids) if a.ids else SPLITS[split]
        if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(SPLITS[split]):
            raise ValueError('Ids must be unique members of selected split')
        cps = [int(c) for c in a.checkpoints.split(',')] if a.checkpoints else CHECKPOINTS
        if not cps or cps != sorted(set(cps)) or min(cps)<0 or max(cps)+10 >= 300:
            raise ValueError('Invalid checkpoints or incomplete horizon')
        if not a.mock and split == 'test' and (ids != SPLITS['test'] or cps != CHECKPOINTS or a.arms != list(ARMS)):
            raise ValueError('Main MR requires all registered ids, checkpoints and arms')
        if a.resume and (a.out is None or not a.out.is_dir()):
            raise ValueError('--resume requires an existing --out directory')
        if a.out and a.out.exists() and not a.resume and not a.preflight:
            raise FileExistsError('Output exists; choose a new path or --resume')
        registered = check_registration() if not a.mock and split == 'test' else None
        configs = [calibration_profile(json.loads((ROOT/f'configs/{arm}.json').read_text()),
                                       a.calibration_profile, split) for arm in a.arms]
        hashes = {p: sha256(ROOT/p) for p in source_files()}
        resources = None
        if a.mock:
            dictionary, trajectories = synthetic(ids, 300)
            tok = MockTokenizer()
        else:
            resource_paths(); verify()
            if a.resource_manifest:
                resources = validate_inventory(json.loads(a.resource_manifest.read_text()))
            elif not a.preflight:
                raise ValueError('Create CPU resource hashes first; provide --resource-manifest')
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(ROOT/configs[0]['model_dir']), local_files_only=True)
            dictionary = load_dictionary()
            trajectories = load_trajectories(ids, 300, check=False)
        for cfg in configs:
            if a.mock:
                # Character tokens are a CPU fixture, not Granite tokens. Keep all
                # seven checkpoints runnable without altering the real limit.
                cfg.update(max_model_len=100000, mock_context_note='character-token fixture only')
            bound = context_audit(cfg | dict(checkpoints=cps), dictionary, trajectories, tok)
            cfg.update(ids=ids, checkpoints=cps, split=split, mock=a.mock,
                       confirmatory=bool(registered), registration_commit=registered,
                       source_sha256=hashes, dataset_revision=REVISION,
                       resource_sha256=resources['sha256'] if resources else None,
                       context_upper_bound=bound)
        print(json.dumps(dict(mode='MOCK' if a.mock else 'LOCAL', split=split, ids=ids,
                              arms=a.arms, checkpoints=cps, conditions=len(ids)*len(cps)*len(configs),
                              continuation_answers=10*len(ids)*len(cps)*len(configs),
                              context_upper_bound=max(c['context_upper_bound'] for c in configs),
                              resource_hashes_verified=resources is not None,
                              registration_commit=registered, free_disk_bytes=shutil.disk_usage(ROOT).free)))
        if a.preflight:
            print('Read-only CPU preflight complete. No inference or output writes.'); return
        stamp = datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%Y%m%d-%H%M%S')
        out = a.out or ROOT/'runs'/f'recovery_{"mock_" if a.mock else ""}{stamp}'
        manifest = dict(schema_version=1, study='MR', mock=a.mock, split=split, configs=configs,
                        resources=resources, registration_commit=registered)
        if a.resume:
            if json.loads((out/'manifest.json').read_text()) != manifest:
                raise ValueError('Resume source/config/model/data/registration mismatch')
            # Validate every completed checkpoint and config before any model allocation.
            for cfg in configs:
                d = out/cfg['arm']
                if d.exists():
                    if json.loads((d/'config.json').read_text()) != cfg:
                        raise ValueError('Arm config mismatch')
                    for sid in ids:
                        for cp in cps:
                            if (d/'checkpoints'/f'{sid}_{cp}.json').exists():
                                load_checkpoint(d/'checkpoints'/f'{sid}_{cp}.json', sid, cp, cfg)
        pending = any(not (out/c['arm']/'checkpoints'/f'{sid}_{cp}.json').exists()
                      for c in configs for sid in ids for cp in cps)
        llm = None
        if pending and a.mock:
            llm = RecoveryMock(tok, dictionary)
        elif pending:
            hardware = gpu_snapshot(a.gpu, configs[0]['util'])
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
            os.environ['HF_HUB_OFFLINE'] = '1'
            from vllm import LLM
            cfg = configs[0]
            llm = LLM(str(ROOT/cfg['model_dir']), dtype='bfloat16', gpu_memory_utilization=cfg['util'],
                      max_model_len=cfg['max_model_len'], enable_prefix_caching=True, seed=0,
                      **cfg.get('engine_settings', {}))
        out.mkdir(parents=True, exist_ok=a.resume)
        if not a.resume:
            atomic_text(out/'manifest.json', json.dumps(manifest, indent=2)+'\n')
        preserve_source(out, configs[0]['source_sha256'])
        if pending and not a.mock:
            with (out/'runtime.log').open('a') as log:
                log.write(json.dumps(dict(gpu=hardware, started=datetime.now(ZoneInfo('Asia/Kolkata')).isoformat()))+'\n')
        for cfg in configs:
            engine = RecoveryEngine(cfg, dictionary, llm, tok, mock=a.mock) if llm else None
            run_recovery_arm(cfg, trajectories, engine, out/cfg['arm'], resume=a.resume)
            if engine:
                with (out/cfg['arm']/'run.log').open('a') as log:
                    log.write(json.dumps(dict(telemetry=engine.telemetry))+'\n')
            archive_arm(out/cfg['arm'], out/f'{cfg["arm"]}.tar.gz')
        print('Saved recovery outputs:', out, '(MOCK ONLY)' if a.mock else '')
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, subprocess.CalledProcessError) as e:
        ap.error(str(e))


if __name__ == '__main__':
    main()
