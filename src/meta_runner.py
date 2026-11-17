"""Independent checkpoint runner for M, reusing the frozen agent primitives."""
import json
import math
import os
import tarfile
from pathlib import Path
from types import SimpleNamespace

from . import prompts as PR
from .agent import Chat, answer_token_info, p_from_first_token, INT_RE
from .labels import label_turn, parse_reply, next_prev
from .meta_history import ARMS, build_history, existing_fake_class, MockTokenizer


class MetaEngine:
    def __init__(self, cfg, dictionary, llm, tokenizer, mock=False):
        self.cfg, self.dictionary, self.llm, self.tok = cfg, dictionary, llm, tokenizer
        self.ch = Chat(tokenizer)
        self.eos = {i for i in [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|im_end|>'),
                               tokenizer.convert_tokens_to_ids('<|endoftext|>')]
                    if isinstance(i, int) and i != tokenizer.unk_token_id}
        self.system = self.ch.system(PR.agent_system(dictionary, K=1, version=cfg['prompt_version']))
        if mock:
            sampling = lambda **kwargs: SimpleNamespace(**kwargs)
            structured = lambda regex: dict(structured_outputs=SimpleNamespace(regex=regex))
        else:
            from vllm import SamplingParams
            from vllm.sampling_params import StructuredOutputsParams
            sampling = SamplingParams
            structured = lambda regex: dict(structured_outputs=StructuredOutputsParams(regex=regex))
        self.answer_params = sampling(temperature=0, max_tokens=16, logprobs=20,
                                      stop_token_ids=list(self.eos), stop=['\n'], **structured(r'-?[0-9]{1,6}'))
        self.num_params = sampling(temperature=0, max_tokens=6, stop_token_ids=list(self.eos), stop=['\n'],
                                   **structured(r'[0-9]{1,3}'))
        self.lp_params = sampling(temperature=0, max_tokens=1, logprobs=20)
        # Match the original runner's token lists and normalization exactly.
        self.yes = [self.ch.enc(w)[0] for w in ['Yes', 'yes', ' Yes']]
        self.no = [self.ch.enc(w)[0] for w in ['No', 'no', ' No']]
        self.abc = [self.ch.enc(w)[0] for w in 'ABC']

    def generate(self, prompts, params):
        if any(len(p) + sp.max_tokens > self.cfg['max_model_len'] for p, sp in zip(prompts, params)):
            raise ValueError('Prompt exceeds max_model_len; refusing silent truncation')
        return self.llm.generate([{'prompt_token_ids': p} for p in prompts], params, use_tqdm=False)

    def probes(self, base, checkpoint=True):
        probes = [('verbal', PR.P_VERBAL, self.num_params)]
        if checkpoint:
            probes += [('fore_none', PR.P_FORECAST, self.num_params), ('aware', PR.P_AWARE, self.lp_params)]
        else:
            probes += [('ptrue', PR.P_TRUE, self.lp_params)]
        outputs = self.generate([base + self.ch.user(q) + self.ch.gen() for _, q, _ in probes],
                                [sp for _, _, sp in probes])
        result, raw = {}, {}
        for (key, _, _), output in zip(probes, outputs):
            co = output.outputs[0]
            raw[key] = co.text
            if key == 'ptrue':
                result[key] = p_from_first_token(co, self.yes, self.no)
            elif key == 'aware':
                lps = co.logprobs[0]
                p = [math.exp(lps[i].logprob) if i in lps else 0. for i in self.abc]
                denom = sum(p) or 1.
                result[key] = [v / denom for v in p]
            else:
                match = INT_RE.search(co.text)
                result[key] = int(match.group()) if match else None
        return result, raw

    def checkpoint(self, trajectory, checkpoint, arm):
        rate, window = ARMS[arm.removeprefix('M_')]
        history = build_history(trajectory, checkpoint, rate, self.dictionary, window, self.cfg['seed'])
        base = list(self.system)
        for h in history['turns']:
            base += self.ch.user(h['user']) + self.ch.asst(h['assistant'])
        pr, raw = self.probes(base)
        row = dict(record_type='checkpoint', arm=arm, sample_id=trajectory.sample_id, turn=checkpoint,
                   checkpoint=checkpoint, history=history, probes=pr, probe_raw=raw,
                   context_tokens=len(base), mock=self.cfg['mock'])
        rows = [row]
        prev = history['turns'][-1]['value']
        for t in range(checkpoint + 1, checkpoint + 11):
            turn = trajectory.turns[t]
            user = PR.keys_text(turn.keys, version=self.cfg['prompt_version'])
            co = self.generate([base + self.ch.user(user) + self.ch.gen()], [self.answer_params])[0].outputs[0]
            value, wf, _ = parse_reply(co.text.strip())
            info = answer_token_info(co, self.tok, self.eos, False)
            kind = label_turn(value, wf, prev, turn.value, turn.truth)
            base = base + self.ch.user(user) + self.ch.asst(co.text.strip())
            probes, probe_raw = self.probes(base, checkpoint=False)
            rows.append(dict(record_type='continuation', arm=arm, sample_id=trajectory.sample_id,
                             checkpoint=checkpoint, turn=t, offset=t-checkpoint, keys=list(turn.keys),
                             v=turn.value, truth=turn.truth, prev=prev, raw=co.text.strip(), value=value,
                             wellformed=wf, committed=value, committed_wf=wf, hist_str=co.text.strip(), kind=kind,
                             lp=info['all'], lp_pre=info['pre'], lp_fin=info['fin'],
                             lp_fin_renorm_min=info['fin_renorm_lp_min'], tok_strs=info['tok_strs'],
                             tok_lps=info['tok_lps'], gen_tokens=len(co.token_ids), probes=probes,
                             probe_raw=probe_raw, mock=self.cfg['mock']))
            prev = next_prev(prev, value)
        row['realized_next10'] = sum(r['kind'] == 'ok' for r in rows[1:]) / 10
        return rows


def atomic_text(path, text):
    """Only M-owned generated files; incomplete temporary files are safe to replace."""
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def validate_checkpoint(rows, sid, cp, arm):
    if len(rows) != 11 or rows[0].get('record_type') != 'checkpoint':
        raise ValueError('Incomplete checkpoint')
    if any(r['sample_id'] != sid or r['checkpoint'] != cp or r['arm'] != arm for r in rows):
        raise ValueError('Checkpoint identity mismatch')
    if [r['turn'] for r in rows] != list(range(cp, cp + 11)):
        raise ValueError('Checkpoint turns mismatch')
    if any(r.get('record_type') != 'continuation' for r in rows[1:]):
        raise ValueError('Invalid continuation row')
    actual = sum(r['kind'] == 'ok' for r in rows[1:]) / 10
    if rows[0]['realized_next10'] != actual:
        raise ValueError('Checkpoint accuracy mismatch')


def materialize(out, cfg):
    """Rebuild canonical JSONL from atomic completed checkpoints, without duplicates."""
    rows, summaries = [], []
    for sid in cfg['ids']:
        complete = []
        for cp in cfg['checkpoints']:
            p = out / 'checkpoints' / f'{sid}_{cp}.json'
            if p.exists():
                rr = json.loads(p.read_text())
                validate_checkpoint(rr, sid, cp, cfg['arm'])
                rows.extend(rr)
                complete.append(rr[0])
        if len(complete) == len(cfg['checkpoints']):
            summaries.append(dict(sample_id=sid, arm=cfg['arm'], T=cfg['T'],
                                  n_checkpoints=len(complete), n_generated=10*len(complete),
                                  accuracy=sum(r['realized_next10'] for r in complete) / len(complete), mock=cfg['mock']))
    atomic_text(out / 'rows.jsonl', ''.join(json.dumps(r) + '\n' for r in rows))
    atomic_text(out / 'trajectories.jsonl', ''.join(json.dumps(r) + '\n' for r in summaries))
    return summaries


def run_meta_arm(cfg, trajectories, engine, out, resume=False):
    out = Path(out)
    if not cfg['arm'].startswith('M_') or cfg['arm'][2:] not in ARMS:
        raise ValueError('Only M arms are permitted')
    if out.exists():
        if not resume:
            raise FileExistsError(f'{out}: use --resume to continue an M run')
        if json.loads((out / 'config.json').read_text()) != cfg:
            raise ValueError('Resume config/source fingerprint mismatch')
    else:
        out.mkdir(parents=True)
        (out / 'checkpoints').mkdir()
        atomic_text(out / 'config.json', json.dumps(cfg, indent=2) + '\n')
    materialize(out, cfg)  # recover a crash after checkpoint commit but before JSONL update
    with (out / 'run.log').open('a') as log:
        for tr in trajectories:
            for cp in cfg['checkpoints']:
                path = out / 'checkpoints' / f'{tr.sample_id}_{cp}.json'
                if path.exists():
                    continue
                rr = engine.checkpoint(tr, cp, cfg['arm'])
                validate_checkpoint(rr, tr.sample_id, cp, cfg['arm'])
                atomic_text(path, json.dumps(rr) + '\n')
                materialize(out, cfg)
                message = f"{cfg['arm']} saved trajectory={tr.sample_id} checkpoint={cp} mock={cfg['mock']}"
                log.write(message + '\n'); log.flush()
                print(message, flush=True)
    return materialize(out, cfg)


def archive_arm(out, destination):
    """Package in the original one-root-directory tar.gz layout, never overwrite."""
    out, destination = Path(out), Path(destination)
    if destination.exists():
        # Resume of a complete archive is read-only and must match canonical data.
        with tarfile.open(destination, 'r:gz') as tf:
            for name in ['config.json', 'rows.jsonl', 'trajectories.jsonl']:
                if tf.extractfile(f'{out.name}/{name}').read() != (out / name).read_bytes():
                    raise ValueError(f'Existing archive differs: {destination}')
        return
    temporary = destination.with_suffix('.tar.gz.tmp')
    with tarfile.open(temporary, 'w:gz') as tf:
        tf.add(out, arcname=out.name, filter=lambda m: None if m.name.endswith('.tmp') else m)
    temporary.replace(destination)
