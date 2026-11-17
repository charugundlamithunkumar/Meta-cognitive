"""Run the frozen M then MR main studies sequentially, saving persistent job status.

--describe is read-only. Actual inference requires both registrations frozen
and all relevant files committed. This script never freezes a protocol itself.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.meta_runner import atomic_text
from src.experiment_runtime import gpu_snapshot
from src.lhe_data import sha256


def commands(tag, gpu, resource_manifest):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', tag):
        raise ValueError('Tag may contain letters, digits, underscore and hyphen only')
    resource_manifest = Path(resource_manifest).resolve()
    tasks = []
    for study, runner, analyzer, prefix, result_file in [
        ('HF', 'run_meta.py', 'analyze_meta_experiment.py', 'meta_HF', 'meta_experiment.json'),
        ('HG', 'run_recovery.py', 'analyze_recovery.py', 'recovery_HG', 'analysis.json'),
    ]:
        run = ROOT/'runs'/f'{prefix}_{tag}'
        out = ROOT/'results'/f'{prefix}_{tag}'
        tasks.append(dict(study=study, run=str(run), analysis=str(out), result_file=result_file,
            inference=[sys.executable, str(ROOT/'scripts'/runner), '--split', 'test', '--gpu', gpu,
                       '--resource-manifest', str(resource_manifest), '--out', str(run)],
            analyze=[sys.executable, str(ROOT/'scripts'/analyzer), str(run), '--n-boot', '10000', '--out', str(out)]))
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu', default='1')
    ap.add_argument('--tag', default=datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%Y%m%d-%H%M%S'))
    ap.add_argument('--resource-manifest', type=Path, default=ROOT/'results/recovery_resources_complete_20261117.json')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--describe', action='store_true')
    a = ap.parse_args()
    try:
        tasks = commands(a.tag, a.gpu, a.resource_manifest)
        if a.describe:
            print(json.dumps(dict(tasks=tasks, conditions=2464, task_answers=24640,
                                  note='Read-only plan; no model or dataset loaded.'), indent=2)); return
        # All protocol checks precede job-directory creation and GPU access.
        for name in ['prereg.md', 'prereg_recovery.md']:
            if not re.search(r'^Status: FROZEN\s*$', (ROOT/name).read_text(), re.M):
                raise ValueError(f'{name} is not frozen; complete review before main inference')
        from scripts.run_meta import check_meta_registration
        from scripts.run_recovery import source_files
        paths = source_files()+['scripts/run_registered_meta.py', 'prereg.md',
                'scripts/meta_precision.py', 'scripts/plot_meta_experiment.py'] + [
                str(p.relative_to(ROOT)) for p in (ROOT/'configs').glob('M_*.json')]
        commit = check_meta_registration(paths)
        job = ROOT/'runs'/f'main_meta_{a.tag}'
        signature = dict(tag=a.tag, gpu=a.gpu, commit=commit,
                         resource_manifest_sha256=sha256(a.resource_manifest), tasks=tasks)
        if job.exists():
            if not a.resume:
                raise FileExistsError('Job directory exists; use --resume')
            state = json.loads((job/'status.json').read_text())
            if state['signature'] != signature:
                raise ValueError('Resume job/source/resource fingerprint mismatch')
        else:
            if a.resume:
                raise ValueError('Resume job does not exist')
            if any(Path(t['run']).exists() or Path(t['analysis']).exists() for t in tasks):
                raise FileExistsError('An output exists without this job record; choose a new tag')
            job.mkdir(parents=True)
            state = dict(signature=signature, stages={}, status='starting')
        def save(status):
            state.update(status=status, pid=os.getpid(), updated=datetime.now(ZoneInfo('Asia/Kolkata')).isoformat())
            atomic_text(job/'status.json', json.dumps(state, indent=2)+'\n')
        save('running')
        try:
            for task in tasks:
                if state['stages'].get(task['study'], {}).get('status') == 'complete':
                    if not (Path(task['analysis'])/task['result_file']).is_file():
                        raise ValueError('Completed stage is missing its result file')
                    continue
                if check_meta_registration(paths) != commit:
                    raise ValueError('Registration commit changed while the job was running')
                study = task['study']
                if state['stages'].get(study, {}).get('status') == 'complete':
                    continue
                stage = state['stages'].setdefault(study, {})
                stage.update(status='inference', started=datetime.now(ZoneInfo('Asia/Kolkata')).isoformat())
                save('running')
                # A prior child may take a few seconds to release its context.
                cfg = json.loads((ROOT/('configs/M_L0.json' if study=='HF' else 'configs/MR_A.json')).read_text())
                for attempt in range(6):
                    try:
                        gpu_snapshot(a.gpu, cfg['util']); break
                    except ValueError:
                        if attempt == 5: raise
                        time.sleep(10)
                cmd = list(task['inference'])
                if Path(task['run']).exists(): cmd.append('--resume')
                print('Starting', study, task['run'], flush=True)
                with (job/f'{study}_inference.log').open('a') as log:
                    subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                stage['status'] = 'analysis'; save('running')
                if check_meta_registration(paths) != commit:
                    raise ValueError('Registration/source changed before analysis')
                analysis = Path(task['analysis'])
                if analysis.exists():
                    if not (analysis/task['result_file']).is_file() or not any(analysis.glob('*.md')):
                        raise ValueError('Incomplete analysis output exists; preserve it and select a new output path before resume')
                else:
                    with (job/f'{study}_analysis.log').open('a') as log:
                        subprocess.run(task['analyze'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                if task['study'] == 'HF' and not (analysis/'figures').exists():
                    with (job/'HF_analysis.log').open('a') as log:
                        subprocess.run([sys.executable, str(ROOT/'scripts/plot_meta_experiment.py'),
                                        str(analysis/task['result_file']), '--out', str(analysis/'figures')],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                result = json.loads((analysis/task['result_file']).read_text())
                stage.update(status='complete', finished=datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),
                             result_status=result.get('status', result.get('primary', {}).get('verdict')))
                save('running')
                # A null or inconclusive research result never cancels the other study.
            save('complete'); print('Both studies complete:', job/'status.json', flush=True)
        except BaseException as e:
            state['error'] = str(e); save('failed'); raise
    except (ValueError, FileNotFoundError, FileExistsError, subprocess.CalledProcessError) as e:
        ap.error(str(e))


if __name__ == '__main__':
    main()
