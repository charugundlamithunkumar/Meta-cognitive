"""Audit run/config metadata only; never parse outcomes or allocate samples.

This reporting tool is outside the frozen inference source set. Run from the
repository root; existing outputs are protected against overwrite.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
MAX_CONFIG_BYTES = 4 * 1024 * 1024


def metadata(config, source, raw):
    if not isinstance(config, dict):
        raise ValueError('Config must be an object')
    ids = config.get('ids', [])
    seeds = config.get('fresh_seeds', [])
    if not isinstance(ids, list) or not all(type(x) is int for x in ids):
        raise ValueError('Expected integer id list')
    if seeds is None:
        seeds = []
    if not isinstance(seeds, list) or not all(type(x) is int for x in seeds):
        raise ValueError('Expected integer fresh-seed list')
    # Only an explicit boolean marks synthetic output. Absent/unknown flags
    # count conservatively as possible real exposure.
    mock = config.get('mock') is True
    return dict(source=source, config_sha256=hashlib.sha256(raw).hexdigest(),
                arm=config.get('arm'), split=config.get('split'), mock=mock,
                ids=ids, explicit_fresh_seeds=seeds,
                inferred_fresh_seeds=[x - 10000 for x in ids if x >= 10000])


def scan(roots, base=ROOT):
    records, errors = [], []
    for root in roots:
        if not root.is_dir():
            errors.append(dict(source=str(root), error='Missing scan root'))
            continue
        for p in sorted(root.rglob('config.json')):
            source = str(p.relative_to(base))
            try:
                if p.stat().st_size > MAX_CONFIG_BYTES:
                    raise ValueError('Config exceeds size limit')
                raw = p.read_bytes()
                records.append(metadata(json.loads(raw), source, raw))
            except (ValueError, OSError) as e:
                errors.append(dict(source=source, error=str(e)))
        for p in sorted(root.rglob('*.tar.gz')):
            source = str(p.relative_to(base))
            try:
                found = False
                # No archive extraction. Only config.json payloads are parsed.
                with tarfile.open(p, 'r:gz') as archive:
                    for member in archive:
                        if not member.isfile() or Path(member.name).name != 'config.json':
                            continue
                        found = True
                        if member.size > MAX_CONFIG_BYTES:
                            raise ValueError('Archived config exceeds size limit')
                        raw = archive.extractfile(member).read()
                        records.append(metadata(json.loads(raw), source+'::'+member.name, raw))
                if not found:
                    raise ValueError('No config.json in archive')
            except (ValueError, OSError, tarfile.TarError, EOFError) as e:
                errors.append(dict(source=source, error=str(e)))
    possible_real = [r for r in records if not r['mock']]
    ids = sorted({i for r in possible_real for i in r['ids']})
    seeds = sorted({s for r in possible_real for s in
                    r['explicit_fresh_seeds'] + r['inferred_fresh_seeds']})
    return dict(kind='METADATA ONLY; no outcome payloads parsed; no sample allocation',
                records=records, errors=errors,
                possible_real_ids=ids, possible_real_fresh_seeds=seeds,
                reserve_ids_found=sorted(set(ids) & set(range(52, 100))),
                historically_reserved_fresh_seeds_found=sorted(set(seeds) & set(range(1000, 1048))),
                historical_reservations=dict(ids=list(range(52, 100)),
                                             fresh_seeds=list(range(1000, 1048)),
                    source='previous work/implementation-plan-through-20261117.md, extension section'),
                conclusion='No independent confirmation sample allocated. Missing metadata, deleted/external runs and prior human exposure cannot be ruled out.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        ap.error('Output exists; choose a new path')
    roots = [ROOT/'runs', ROOT/'previous work/runs', ROOT/'previous work/results']
    report = scan(roots)
    report['registration_head'] = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    report['scan_roots'] = [str(p.relative_to(ROOT)) for p in roots]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['records', 'historical_reservations']}))
    print('Config records:', len(report['records']))


if __name__ == '__main__':
    main()
