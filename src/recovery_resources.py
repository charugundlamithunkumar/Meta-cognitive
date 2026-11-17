"""CPU-only local-resource inventory. No imports of torch, vLLM or transformers."""
import importlib.metadata
import json
from pathlib import Path
import platform

from .lhe_data import verify, sha256, REVISION
from .recovery_history import fingerprint

ROOT = Path(__file__).resolve().parent.parent


def resource_paths(root=ROOT):
    model = root/'models/granite-4.2-8b'
    index = model/'model.safetensors.index.json'
    shards = sorted(set(json.loads(index.read_text())['weight_map'].values()))
    if any(Path(s).name != s for s in shards):
        raise ValueError('Invalid shard filename')
    paths = [model/n for n in ['config.json', 'tokenizer.json', 'tokenizer_config.json',
                              'model.safetensors.index.json'] + shards]
    # AutoTokenizer may load an external chat template and auxiliary vocab files.
    paths += [p for pattern in ['*.json', '*.jinja', '*.model', 'merges.txt', 'chat_templates/*.jinja']
              for p in model.glob(pattern) if p not in paths]
    paths = sorted(set(paths))
    paths += [root/'data/lhe'/n for n in ['dictionary.json', 'test.jsonl', 'config.json']]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)
    return paths


def inventory(root=ROOT):
    verify(root/'data/lhe')
    files = {str(p.relative_to(root)): dict(size=p.stat().st_size, sha256=sha256(p))
             for p in resource_paths(root)}
    packages = {}
    for name in ['numpy', 'scipy', 'torch', 'vllm', 'transformers', 'matplotlib', 'statsmodels']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    body = dict(schema_version=1, dataset_revision=REVISION, files=files,
                python=platform.python_version(), packages=packages)
    return dict(body, sha256=fingerprint(body))


def validate_inventory(saved, root=ROOT):
    current = inventory(root)
    if current != saved:
        raise ValueError('Local resource/software manifest mismatch; create a new run')
    return current
