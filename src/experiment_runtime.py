"""Shared calibration profiles and non-destructive GPU memory preflight."""
import json
from pathlib import Path
import subprocess
import hashlib

ROOT = Path(__file__).resolve().parent.parent


def preserve_source(out, hashes):
    """Keep exact generation source bytes beside a run, including on resume."""
    for name, expected in hashes.items():
        source = ROOT/name
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Source changed after preflight: {name}')
        destination = Path(out)/'source_snapshot'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != data:
                raise ValueError(f'Source snapshot mismatch: {name}')
        else:
            with destination.open('xb') as f:
                f.write(data)


def calibration_profile(cfg, name, split):
    if name is None:
        return cfg
    if split != 'calibration':
        raise ValueError('Memory profile overrides are calibration-only; main settings remain frozen')
    files = {'gpu1-small': 'recovery_calibration_gpu1.json',
             'gpu1-long': 'recovery_calibration_gpu1_long.json'}
    if name not in files:
        raise ValueError('Unknown calibration profile')
    profile = json.loads((ROOT/'configs'/files[name]).read_text())
    if name != profile['name'] or not profile['calibration_only']:
        raise ValueError('Unknown calibration profile')
    return dict(cfg, util=profile['util'], max_model_len=profile['max_model_len'],
                engine_settings=profile['engine_settings'], calibration_profile=name)


def gpu_snapshot(gpu, utilization):
    """Only called on the real inference path, immediately before allocation."""
    result = subprocess.run(['nvidia-smi', '--id='+gpu,
        '--query-gpu=index,name,driver_version,memory.total,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    values = [s.strip() for s in result.stdout.strip().split(',')]
    if len(values) != 6:
        raise ValueError('Could not identify exactly one GPU')
    total, free = float(values[3]), float(values[4])
    if free < total*utilization + 512:
        raise ValueError(f'GPU {gpu} busy: {free:.0f} MiB free; need at least {total*utilization+512:.0f}. No process stopped.')
    return dict(zip(['index','name','driver','total_MiB','free_MiB','utilization_percent'],values))
