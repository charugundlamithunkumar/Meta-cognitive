"""Run only the new M arms. --dry-run/--mock uses synthetic data, no model/GPU.

python3 scripts/run_meta.py --dry-run --ids 20-21 --checkpoints 10,25 --out runs/meta_mock
python3 scripts/run_meta.py --split test --out runs/meta_confirmatory
python3 scripts/run_meta.py --split test --out runs/meta_confirmatory --resume
"""
import argparse
import json
import os
import subprocess
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.study_common import parse_ids
from src.lhe_data import SPLITS, Trajectory, Turn, load_dictionary, load_trajectories, verify, sha256, REVISION
from src.meta_history import ARMS, existing_fake_class, MockTokenizer
from src.meta_runner import MetaEngine, run_meta_arm, archive_arm, validate_checkpoint, atomic_text
from src.experiment_runtime import calibration_profile, gpu_snapshot, preserve_source
from src.recovery_resources import resource_paths, validate_inventory
from src.agent import Chat
from src import prompts as PR
from src.meta_history import build_history


def context_audit(cfg, dictionary, trajectories, tokenizer):
    """Conservative real-tokenizer M bound, including ten generated replies."""
    ch = Chat(tokenizer)
    system = ch.system(PR.agent_system(dictionary, K=1, version=cfg['prompt_version']))
    rate, window = ARMS[cfg['arm'].removeprefix('M_')]
    maximum = 0
    for tr in trajectories:
        for cp in cfg['checkpoints']:
            history = build_history(tr, cp, rate, dictionary, window, cfg['seed'])
            base = list(system)
            for h in history['turns']:
                base += ch.user(h['user']) + ch.asst(h['assistant'])
            maximum = max(maximum, *(len(base+ch.user(q)+ch.gen())+6 for q in
                          [PR.P_VERBAL, PR.P_FORECAST, PR.P_AWARE]))
            length = len(base)
            overhead = len(ch.asst(''))
            for t in range(cp+1, cp+11):
                length += len(ch.user(PR.keys_text(tr.turns[t].keys, version=cfg['prompt_version'])))
                maximum = max(maximum, length+len(ch.gen())+16)
                length += overhead+16
                maximum = max(maximum, length+max(len(ch.user(q)) for q in
                              [PR.P_VERBAL, PR.P_TRUE])+len(ch.gen())+6)
    if maximum > cfg['max_model_len']:
        raise ValueError(f'Context upper bound {maximum} exceeds {cfg["max_model_len"]}; no truncation')
    return maximum


def check_meta_registration(paths):
    """The original runner's commit-before-test rule, scoped to the M materials."""
    for name in paths:
        committed = subprocess.run(['git', 'show', f'HEAD:{name}'], cwd=ROOT, capture_output=True)
        if committed.returncode or committed.stdout != (ROOT / name).read_bytes():
            raise ValueError(f'Commit M materials before a test run: {name} is uncommitted or differs from HEAD')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def synthetic(ids, T):
    dictionary = {'alpha': 3, 'bravo': -2, 'charlie': 7}
    result = []
    for sid in ids:
        total, turns = 0, []
        for t in range(T):
            key = list(dictionary)[(t + sid) % len(dictionary)]
            total += dictionary[key]
            turns.append(Turn((key,), dictionary[key], total))
        result.append(Trajectory(sid, turns))
    return dictionary, result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arms', nargs='+', choices=list(ARMS), default=list(ARMS))
    ap.add_argument('--config', type=Path, help='One configs/M_*.json; overrides --arms')
    ap.add_argument('--split', choices=['test', 'calibration'])
    ap.add_argument('--ids', help='Defaults to the chosen split (test for mock)')
    ap.add_argument('--checkpoints', help='Comma-separated checkpoints for mock/calibration smoke only')
    ap.add_argument('--dry-run', '--mock', dest='mock', action='store_true')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--calibration-profile', choices=['gpu1-small', 'gpu1-long'])
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--resource-manifest', type=Path)
    a = ap.parse_args()
    if not a.mock and a.split is None:
        ap.error('Real model runs require --split test or --split calibration')
    if a.resume and a.out is None:
        ap.error('--resume requires --out')
    ids = parse_ids(a.ids) if a.ids else SPLITS[a.split or 'test']
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(SPLITS[a.split or 'test']):
        ap.error('ids must be unique members of the selected split')
    configs = [json.loads(a.config.read_text())] if a.config else [json.loads((ROOT / 'configs' / f'M_{k}.json').read_text()) for k in a.arms]
    if len({c['arm'] for c in configs}) != len(configs):
        ap.error('Duplicate arms')
    for cfg in configs:
        if cfg.get('arm') not in ['M_' + k for k in ARMS] or cfg.get('K') != 1 or cfg.get('T') != 300:
            ap.error('Expected an M config with K=1, T=300')
        canonical = json.loads((ROOT / 'configs' / (cfg['arm'] + '.json')).read_text())
        if cfg != canonical:
            ap.error('Use the registered M config unchanged; a modified design needs a new registration')
        try:
            cfg.update(calibration_profile(cfg, a.calibration_profile, a.split or 'test'))
        except ValueError as e:
            ap.error(str(e))
        if a.mock:
            cfg.update(max_model_len=100000, mock_context_note='character-token fixture only')
        if a.checkpoints:
            if not a.mock and a.split == 'test':
                ap.error('Test checkpoints are frozen; use calibration for smoke runs')
            cfg['checkpoints'] = [int(c) for c in a.checkpoints.split(',')]
        cp = cfg['checkpoints']
        if not cp or cp != sorted(set(cp)) or min(cp) < 0 or max(cp) + 10 >= cfg['T']:
            ap.error('Invalid checkpoints / incomplete future windows')
        if not a.mock and a.split == 'test' and cp != [10, 25, 50, 100, 150, 200, 250]:
            ap.error('Test checkpoints are frozen')
        cfg.update(ids=ids, mock=a.mock, split=a.split or 'test', dataset_revision=REVISION,
                   data_source='synthetic_mock' if a.mock else 'pinned_LHE',
                   confirmatory=not a.mock and a.split == 'test' and ids == SPLITS['test'])
        files = ['prereg.md', 'src/meta_history.py', 'src/meta_runner.py', 'scripts/run_meta.py',
                 'src/agent.py', 'src/prompts.py', 'src/labels.py', 'src/lhe_data.py', 'src/metrics.py',
                 'src/stats.py', 'scripts/analyze_meta.py', 'scripts/analyze_meta_experiment.py',
                 'src/study_common.py', 'tests/test_meta.py', 'previous work/tests/test_arms_mock.py',
                 'previous work/configs/A0.json', 'src/experiment_runtime.py',
                 'src/recovery_resources.py', 'src/recovery_history.py', 'requirements.txt',
                 'configs/recovery_calibration_gpu1.json', 'configs/recovery_calibration_gpu1_long.json',
                 'tests/test_meta_runtime.py', 'tests/test_main_queue.py', 'scripts/meta_precision.py',
                 'scripts/plot_meta_experiment.py', 'scripts/run_registered_meta.py'] + [f'configs/M_{k}.json' for k in ARMS]
        cfg['source_sha256'] = {p: sha256(ROOT / p) for p in files}
    stamp = datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%Y%m%d-%H%M%S')
    out = a.out or ROOT / 'runs' / f"meta_{'mock_' if a.mock else ''}{stamp}"
    if out.exists() and not a.resume and not a.preflight:
        ap.error(f'{out} exists; use a new directory or --resume')
    if a.resume and not out.is_dir():
        ap.error('--resume directory does not exist')
    registered = None
    if not a.mock and a.split == 'test':
        if not re.search(r'^Status: FROZEN\s*$', (ROOT/'prereg.md').read_text(), re.M):
            ap.error('HF is not frozen; complete calibration/review before test inference')
        try:
            commit = check_meta_registration(files)
        except ValueError as e:
            ap.error(str(e))
        registered = commit
        print('Registered M source commit:', commit)
    resources = None
    if a.mock:
        dictionary, trajectories = synthetic(ids, configs[0]['T'])
        tok = MockTokenizer()
        print('MOCK ONLY: synthetic trajectories, scripted outputs; no empirical results.')
    else:
        if not (ROOT / 'data/lhe/test.jsonl').exists():
            ap.error('Missing data/lhe/test.jsonl; fetch it or use --dry-run')
        model = ROOT / configs[0]['model_dir']
        if not (model / 'tokenizer_config.json').exists() or not list(model.glob('*.safetensors')):
            ap.error('Missing model weights/tokenizer; use --dry-run')
        resource_paths(); verify()
        if a.resource_manifest:
            try:
                resources = validate_inventory(json.loads(a.resource_manifest.read_text()))
            except ValueError as e:
                ap.error(str(e))
        elif not a.preflight:
            ap.error('Provide --resource-manifest for real inference')
        dictionary = load_dictionary()
        trajectories = load_trajectories(ids, configs[0]['T'], check=False)
        # Check config compatibility before allocating GPU memory.
        for cfg in configs[1:]:
            if any(cfg[k] != configs[0][k] for k in ['model_dir', 'model_revision', 'util', 'max_model_len']):
                ap.error('All arms must use the same model settings')
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    for cfg in configs:
        try:
            bound = context_audit(cfg, dictionary, trajectories, tok)
        except ValueError as e:
            ap.error(str(e))
        cfg.update(context_upper_bound=bound, resource_sha256=resources['sha256'] if resources else None,
                   registration_commit=registered)
        saved = out / cfg['arm'] / 'config.json'
        if saved.exists() and json.loads(saved.read_text()) != cfg:
            ap.error(f'Resume config/source/model/data mismatch: {saved}')
        for sid in ids:
            for cp in cfg['checkpoints']:
                path = out/cfg['arm']/'checkpoints'/f'{sid}_{cp}.json'
                if path.exists():
                    validate_checkpoint(json.loads(path.read_text()), sid, cp, cfg['arm'])
    print(json.dumps(dict(mode='MOCK' if a.mock else 'LOCAL', ids=ids,
          arms=[c['arm'] for c in configs], conditions=len(configs)*len(ids)*len(configs[0]['checkpoints']),
          context_upper_bound=max(c['context_upper_bound'] for c in configs),
          resource_hashes_verified=resources is not None)))
    if a.preflight:
        print('Read-only CPU preflight complete; no weight loading or output writes.'); return
    manifest = dict(study='M', configs=configs, resources=resources, registration_commit=registered)
    if a.resume and (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text()) != manifest:
        ap.error('Resume manifest mismatch')
    pending = any(not (out / cfg['arm'] / 'checkpoints' / f'{sid}_{cp}.json').exists()
                  for cfg in configs for sid in ids for cp in cfg['checkpoints'])
    llm = None
    hardware = None
    if pending and a.mock:
        llm = existing_fake_class()(tok, dictionary, {})
    elif pending:
        try:
            hardware = gpu_snapshot(a.gpu, configs[0]['util'])
        except (ValueError, subprocess.CalledProcessError) as e:
            ap.error(str(e))
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        from vllm import LLM
        llm = LLM(str(model), dtype='bfloat16', gpu_memory_utilization=configs[0]['util'],
                  max_model_len=configs[0]['max_model_len'], enable_prefix_caching=True, seed=0,
                  **configs[0].get('engine_settings', {}))
    out.mkdir(parents=True, exist_ok=a.resume)
    if not a.resume:
        atomic_text(out/'manifest.json', json.dumps(manifest, indent=2)+'\n')
    preserve_source(out, configs[0]['source_sha256'])
    if hardware:
        with (out/'runtime.log').open('a') as log:
            log.write(json.dumps(dict(gpu=hardware, started=datetime.now(ZoneInfo('Asia/Kolkata')).isoformat()))+'\n')
    for cfg in configs:
        arm_out = out / cfg['arm']
        engine = MetaEngine(cfg, dictionary, llm, tok, mock=a.mock) if llm is not None else None
        started = datetime.now(ZoneInfo('Asia/Kolkata')).isoformat()
        t0 = time.monotonic()
        summaries = run_meta_arm(cfg, trajectories, engine, arm_out, resume=a.resume)
        with (arm_out/'run.log').open('a') as log:
            log.write(json.dumps(dict(started=started, elapsed_seconds=time.monotonic()-t0,
                                     completed_trajectories=len(summaries)))+'\n')
        if len(summaries) == len(ids):
            archive_arm(arm_out, out / f"{cfg['arm']}.tar.gz")
    print('Saved M runs and archives to', out)


if __name__ == '__main__':
    main()
