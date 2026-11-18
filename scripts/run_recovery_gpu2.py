"""Persistently collect the hardware-amended HG main study, without rerunning M."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.run_registered_meta import commands
from scripts.run_recovery import source_files, check_registration
from scripts.run_meta import check_meta_registration
from scripts.analyze_meta import read_bundle
from src.experiment_runtime import gpu_snapshot, preserve_source
from src.lhe_data import sha256, SPLITS
from src.meta_runner import atomic_text, archive_arm
from src.recovery_history import ARMS, CHECKPOINTS
from src.recovery_runner import load_checkpoint

AMENDMENT = 'prereg_recovery_gpu2_amendment.md'
EXTRA_SOURCE = [AMENDMENT, 'scripts/run_recovery_gpu2.py',
                'paper/tools/test_gpu2_job.py']
ORIGINAL_COMMIT = '1d11bc5a2a665062b3aec023f2dc18ba0c0d02a4'


def plan(tag, resource_manifest, gpu='2', amendment=AMENDMENT):
    if gpu not in {'0', '2'}:
        raise ValueError('No registered migration for this GPU')
    task = commands(tag, gpu, resource_manifest)[1]
    return dict(task=task, gpu=gpu, conditions=896, task_answers=8960,
                job=str(ROOT/'runs'/f'main_recovery_{tag}'),
                amendment=amendment, original_registration_commit=ORIGINAL_COMMIT,
                completed_M=str(ROOT/'results/meta_HF_main_20261118/meta_experiment.json'))


def prepare_environment():
    """Give service children the build tools from this Python environment."""
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', os.defpath)
    ninja = shutil.which('ninja')
    if ninja is None:
        raise FileNotFoundError('ninja is required for vLLM initialization')
    return dict(ninja=ninja, ninja_sha256=sha256(ninja),
                ninja_version=subprocess.check_output([ninja, '--version'], text=True).strip(),
                compiler=shutil.which('g++'))


def validate_outputs(root):
    root = Path(root)
    counts = {}
    for arm in ARMS:
        d = root/arm
        cfg, exported, summaries = read_bundle(d)
        if cfg['mock'] or cfg['ids'] != SPLITS['test'] or cfg['checkpoints'] != CHECKPOINTS:
            raise ValueError('Incomplete or non-main arm configuration')
        saved = []
        for sid in SPLITS['test']:
            for cp in CHECKPOINTS:
                saved.extend(load_checkpoint(d/'checkpoints'/f'{sid}_{cp}.json', sid, cp, cfg))
        if saved != exported or sorted(s['sample_id'] for s in summaries) != SPLITS['test']:
            raise ValueError('Checkpoint/export/completion mismatch')
        archive = root/f'{arm}.tar.gz'
        if not archive.is_file():
            raise ValueError('Missing archive')
        archive_arm(d, archive)  # Existing helper verifies instead of overwriting.
        for name, expected in cfg['source_sha256'].items():
            if sha256(root/'source_snapshot'/name) != expected:
                raise ValueError('Generation source snapshot mismatch')
        counts[arm] = dict(conditions=len(saved)//11, task_answers=len(saved)//11*10,
                           archive_sha256=sha256(archive))
    return counts


def main(*, gpu='2', amendment=AMENDMENT, additional_sources=()):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tag', default=f'gpu{gpu}_main_20261118')
    ap.add_argument('--resource-manifest', type=Path,
                    default=ROOT/'results/recovery_resources_complete_20261117.json')
    ap.add_argument('--describe', action='store_true')
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    p = plan(a.tag, a.resource_manifest, gpu, amendment)
    if a.describe:
        print(json.dumps(p, indent=2)); return
    paths = source_files() + EXTRA_SOURCE + [amendment] + list(additional_sources)
    commit = check_meta_registration(paths)
    if check_registration() != commit:
        raise ValueError('HG registration commit mismatch')
    tools = prepare_environment()
    task = p['task']; job = Path(p['job'])
    signature = dict(plan=p, commit=commit, resource_manifest_sha256=sha256(a.resource_manifest),
                     completed_M_sha256=sha256(p['completed_M']),
                     runtime_tools=tools,
                     source_sha256={name:sha256(ROOT/name) for name in paths})
    if job.exists() and not a.resume:
        raise FileExistsError('Job exists; use --resume')
    if a.resume and not (job/'status.json').is_file():
        raise ValueError('Resume requires a saved job')
    if not job.exists() and (Path(task['run']).exists() or Path(task['analysis']).exists()):
        raise FileExistsError('Output exists without this job record')
    job.mkdir(parents=True, exist_ok=True)
    with (job/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((job/'status.json').read_text()) if a.resume else dict(signature=signature)
        if state['signature'] != signature:
            raise ValueError('Resume fingerprint mismatch')
        def save(status, **extra):
            state.update(status=status, pid=os.getpid(),
                         updated=datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(), **extra)
            atomic_text(job/'status.json', json.dumps(state, indent=2)+'\n')
            print(json.dumps(dict(status=status, **extra)), flush=True)
        try:
            if state.get('status') == 'complete':
                return
            save('checking')
            preserve_source(job, signature['source_sha256'])
            run = Path(task['run'])
            complete = all((run/arm/'checkpoints'/f'{sid}_{cp}.json').is_file()
                           for arm in ARMS for sid in SPLITS['test'] for cp in CHECKPOINTS)
            if not complete:
                utilization = json.loads((ROOT/'configs/MR_A.json').read_text())['util']
                while True:
                    if check_meta_registration(paths) != commit:
                        raise ValueError('Source or commit changed during wait')
                    try:
                        hardware = gpu_snapshot(gpu, utilization); break
                    except ValueError as e:
                        save('waiting_for_gpu', reason=str(e)); time.sleep(30)
                save('inference', hardware=hardware)
                cmd = task['inference'] + (['--resume'] if run.exists() else [])
                with (job/'HG_inference.log').open('a') as log:
                    subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            save('validating')
            counts = validate_outputs(run)
            if check_meta_registration(paths) != commit:
                raise ValueError('Source/commit changed before analysis')
            analysis = Path(task['analysis'])
            if not analysis.exists():
                save('analysis')
                with (job/'HG_analysis.log').open('a') as log:
                    subprocess.run(task['analyze'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            required = ['analysis.json', 'report.md', 'accuracy.png', 'forecasts.png',
                        'hg_effects.png', 'secondary.png']
            if not all((analysis/name).is_file() for name in required):
                raise ValueError('Incomplete analysis output; preserve and inspect before resuming')
            result = json.loads((analysis/'analysis.json').read_text())
            save('complete', validated_counts=counts, primary=result['primary'],
                 hardware_amendment=amendment)
        except BaseException as e:
            save('failed', error=str(e)); raise


if __name__ == '__main__':
    main()
