"""Explicit fresh-sequence allocation for independent study C; no GPU imports."""
import json
from pathlib import Path

from src.lhe_data import generate_fresh, load_trajectories, sha256, REVISION
from src.recovery_history import fingerprint

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT/'configs/C_confirmation.json'


def settings():
    return json.loads(CONFIG.read_text())


def content_hash(trajectory):
    return fingerprint([dict(keys=list(t.keys), value=t.value, truth=t.truth)
                        for t in trajectory.turns])


def build_sample_manifest(config, audit):
    seeds = config['fresh_seeds']
    used = set(audit['possible_real_fresh_seeds'])
    reserved = set(audit['historical_reservations']['fresh_seeds'])
    if len(seeds) != 32 or len(set(seeds)) != 32 or any(type(s) is not int for s in seeds):
        raise ValueError('C requires exactly 32 unique integer seeds')
    if audit['errors'] or set(seeds) & (used | reserved):
        raise ValueError('Allocation audit is incomplete or seeds overlap prior use/reservations')
    trajectories = generate_fresh(seeds, config['T'], config['K'])
    published = load_trajectories(range(100), config['T'], config['K'], check=False)
    prior_fresh = generate_fresh(sorted(used), config['T'], config['K'])
    old_hashes = {content_hash(t) for t in published + prior_fresh}
    hashes = [content_hash(t) for t in trajectories]
    if len(set(hashes)) != len(hashes) or set(hashes) & old_hashes:
        raise ValueError('Duplicate sequence content; do not silently replace the allocation')
    body = dict(schema_version=1, study_instance=config['study_instance'],
                T=config['T'], K=config['K'], fresh_seeds=seeds,
                ids=[t.sample_id for t in trajectories],
                trajectories=[dict(sample_id=t.sample_id, seed=s, content_sha256=h)
                              for t,s,h in zip(trajectories,seeds,hashes)],
                dictionary_sha256=sha256(ROOT/'data/lhe/dictionary.json'),
                generator_sha256=sha256(ROOT/'src/lhe_data.py'),
                dataset_revision=REVISION, allocation_audit_sha256=fingerprint(audit),
                compared_published_ids=list(range(100)), compared_prior_fresh_seeds=sorted(used))
    return dict(body, sha256=fingerprint(body))


def load_samples(path=None):
    path = Path(path or ROOT/settings()['manifest'])
    manifest = json.loads(path.read_text())
    body = {k:v for k,v in manifest.items() if k!='sha256'}
    if fingerprint(body) != manifest['sha256']:
        raise ValueError('Sample manifest checksum mismatch')
    if manifest['dictionary_sha256'] != sha256(ROOT/'data/lhe/dictionary.json') or manifest['generator_sha256'] != sha256(ROOT/'src/lhe_data.py'):
        raise ValueError('Sample generator/dictionary changed')
    cfg = settings()
    if any(manifest[k] != cfg[k] for k in ['study_instance','T','K','fresh_seeds']):
        raise ValueError('Sample manifest does not match C settings')
    trajectories = generate_fresh(manifest['fresh_seeds'], manifest['T'], manifest['K'])
    if len(manifest['trajectories']) != len(trajectories) or [t.sample_id for t in trajectories] != manifest['ids'] or any(
            t.sample_id!=m['sample_id'] or s!=m['seed'] or content_hash(t)!=m['content_sha256']
            for t,s,m in zip(trajectories,manifest['fresh_seeds'],manifest['trajectories'])):
        raise ValueError('Generated sequence content differs from manifest')
    return manifest, trajectories


def source_paths():
    from scripts.run_recovery import source_files
    return sorted(set(source_files()+[
        'prereg.md','prereg_confirmation.md','configs/C_confirmation.json',
        settings()['manifest'],str(Path(settings()['manifest']).with_name(Path(settings()['manifest']).stem+'_allocation.json')),
        'scripts/confirmation_common.py',
        'scripts/prepare_confirmation.py','scripts/run_confirmation.py',
        'scripts/analyze_confirmation.py','scripts/run_recovery_gpu2.py',
        'scripts/plot_meta_experiment.py','paper/tools/test_confirmation.py']))
