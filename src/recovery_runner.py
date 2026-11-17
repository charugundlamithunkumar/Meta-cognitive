"""HG forks and persistence, reusing M generation parameters and atomic/archive helpers."""
import json
import math
from pathlib import Path
import re
import time

from . import prompts as PR
from .agent import answer_token_info
from .labels import label_turn, parse_reply, next_prev
from .meta_history import existing_fake_class
from .meta_runner import MetaEngine, atomic_text
from .recovery_history import ARMS, build_recovery_history, fingerprint
from .recovery_prompts import state_cue


def numeric_probe(raw):
    text = raw.strip()
    parsed = int(text) if re.fullmatch(r'[0-9]+', text) else None
    return dict(raw=raw, parsed=parsed, value=min(100, parsed) if parsed is not None else None,
                valid=parsed is not None, out_of_range=parsed is not None and parsed > 100)


class RecoveryMock:
    """Scripted cue-aware arithmetic. Its forecasts are fixtures, never measurements."""
    def __init__(self, tokenizer, dictionary):
        self.base = existing_fake_class()(tokenizer, dictionary, {})
        self.tok, self.dictionary, self.calls = tokenizer, dictionary, []

    def generate(self, prompts, params, use_tqdm=False):
        out = []
        for p in prompts:
            text = self.tok.decode(p['prompt_token_ids'])
            self.calls.append(text)
            users = re.findall(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', text, re.S)
            last = users[-1]
            cue = re.search(r'Continue using this current running total: (-?\d+)\.', last)
            key = re.search(r'Key: (\w+)', last)
            if cue and key:
                out.append(self.base._out(str(int(cue.group(1)) + self.dictionary[key.group(1)])))
            else:
                out.append(self.base.respond(text))
        return out


class RecoveryEngine(MetaEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.telemetry = dict(sequences=0, prompt_tokens=0, generated_tokens=0, seconds=0.)

    def generate(self, prompts, params):
        started = time.monotonic()
        outputs = super().generate(prompts, params)
        self.telemetry['seconds'] += time.monotonic()-started
        self.telemetry['sequences'] += len(prompts)
        self.telemetry['prompt_tokens'] += sum(map(len, prompts))
        self.telemetry['generated_tokens'] += sum(len(o.outputs[0].token_ids) for o in outputs)
        return outputs

    def numeric(self, base, question):
        tokens = base + self.ch.user(question) + self.ch.gen()
        co = self.generate([tokens], [self.num_params])[0].outputs[0]
        return dict(numeric_probe(co.text), prompt_sha256=fingerprint(tokens), context_tokens=len(tokens))

    def probability(self, co, groups):
        lps = co.logprobs[0] if co.logprobs else {}
        masses = [sum(math.exp(lps[i].logprob) for i in set(ids) if i in lps) for ids in groups]
        total = sum(masses)
        return dict(raw=co.text, retained_mass=total,
                    probabilities=[v/total for v in masses] if total else None)

    def answer_probes(self, base):
        questions = [PR.P_VERBAL, PR.P_TRUE]
        outputs = self.generate([base + self.ch.user(q) + self.ch.gen() for q in questions],
                                [self.num_params, self.lp_params])
        verbal = numeric_probe(outputs[0].outputs[0].text)
        yes = self.probability(outputs[1].outputs[0], [self.yes, self.no])
        return dict(verbal=verbal['value'], ptrue=yes['probabilities'][0] if yes['probabilities'] else None), dict(verbal=verbal, ptrue=yes)

    def checkpoint(self, trajectory, checkpoint, arm):
        history = build_recovery_history(trajectory, checkpoint, arm, self.dictionary, self.cfg['seed'])
        base = list(self.system)
        for h in history['turns']:
            base += self.ch.user(h['user']) + self.ch.asst(h['assistant'])
        cue = state_cue(history['supplied_state'])
        before = self.numeric(base, PR.P_FORECAST)
        after = self.numeric(base, cue + '\n' + PR.P_FORECAST)
        common = dict(study='MR', schema_version=1, arm=arm, sample_id=trajectory.sample_id,
                      checkpoint=checkpoint, mock=self.cfg['mock'])
        head = dict(common, record_type='checkpoint', turn=checkpoint, offset=0, history=history,
                    pre_forecast=before, post_forecast=after, context_tokens=len(base))
        rows = [head]
        prev = history['supplied_state']
        for offset in range(1, 11):
            t = checkpoint + offset
            turn = trajectory.turns[t]
            user = PR.keys_text(turn.keys, version=self.cfg['prompt_version'])
            if offset == 1:
                user = cue + '\n' + user
            tokens = base + self.ch.user(user) + self.ch.gen()
            co = self.generate([tokens], [self.answer_params])[0].outputs[0]
            value, wf, _ = parse_reply(co.text.strip())
            # A malformed string that happens to contain truth is still a nonanswer.
            kind = label_turn(value, wf, prev, turn.value, turn.truth) if wf else 'nonanswer'
            info = answer_token_info(co, self.tok, self.eos, False)
            base += self.ch.user(user) + self.ch.asst(co.text.strip())
            probes, details = self.answer_probes(base)
            rows.append(dict(common, record_type='continuation', turn=t, offset=offset,
                             keys=list(turn.keys), v=turn.value, truth=turn.truth, prev=prev,
                             raw=co.text, value=value, wellformed=wf, kind=kind,
                             committed=value, committed_wf=wf, probes=probes, probe_details=details,
                             lp=info['all'], lp_pre=info['pre'], lp_fin=info['fin'],
                             lp_fin_renorm_min=info['fin_renorm_lp_min'],
                             tok_strs=info['tok_strs'], tok_lps=info['tok_lps'],
                             context_tokens=len(tokens), prompt_sha256=fingerprint(tokens),
                             gen_tokens=len(co.token_ids)))
            prev = next_prev(prev, value)
        co = self.generate([base + self.ch.user(PR.P_AWARE) + self.ch.gen()], [self.lp_params])[0].outputs[0]
        head['post_continuation_awareness'] = self.probability(co, [[i] for i in self.abc])
        head['realized_next10'] = sum(r['kind'] == 'ok' for r in rows[1:])/10
        return rows


def validate_checkpoint(rows, sid, cp, arm, mock=None):
    if arm not in ARMS or len(rows) != 11:
        raise ValueError('Invalid MR arm or checkpoint length')
    head = rows[0]
    if head.get('record_type') != 'checkpoint' or [r.get('offset') for r in rows] != list(range(11)):
        raise ValueError('Invalid checkpoint structure')
    if any((r.get('study'), r.get('schema_version'), r.get('arm'), r.get('sample_id'), r.get('checkpoint'), r.get('turn'))
           != ('MR', 1, arm, sid, cp, cp+i) for i, r in enumerate(rows)):
        raise ValueError('Checkpoint identity mismatch')
    if len({r.get('mock') for r in rows}) != 1 or (mock is not None and head.get('mock') != mock):
        raise ValueError('Mixed mock/real checkpoint')
    h = head['history']
    if len(h['turns']) != cp+1 or h['history_sha256'] != fingerprint(h['turns']):
        raise ValueError('Corrupt history')
    if [r['turn'] for r in h['turns']] != list(range(cp+1)):
        raise ValueError('History positions mismatch')
    mask = [r['turn'] for r in h['turns'] if r['value'] != r['truth']]
    expected = math.floor(.5*cp+.5)+1 if arm in ('MR_B', 'MR_C') else 0
    if mask != h['injected_positions'] or len(mask) != expected or (expected and cp not in mask):
        raise ValueError('Invalid history injection')
    if h['truth'] != h['turns'][-1]['truth'] or h['supplied_state'] != h['truth'] + int(arm in ('MR_B', 'MR_D')):
        raise ValueError('Invalid supplied state')
    prev = h['supplied_state']
    for r in rows[1:]:
        value, wf, _ = parse_reply(r['raw'].strip())
        expected_kind = label_turn(value, wf, prev, r['v'], r['truth']) if wf else 'nonanswer'
        if (r['record_type'], r['prev'], r['value'], r['wellformed'], r['kind']) != ('continuation', prev, value, wf, expected_kind):
            raise ValueError('Continuation labels/state mismatch')
        prev = next_prev(prev, value)
    if head['realized_next10'] != sum(r['kind']=='ok' for r in rows[1:])/10:
        raise ValueError('Accuracy mismatch')
    for key in ['pre_forecast', 'post_forecast']:
        parsed = numeric_probe(head[key]['raw'])
        if any(head[key].get(k) != v for k, v in parsed.items()):
            raise ValueError('Forecast preprocessing mismatch')


def load_checkpoint(path, sid, cp, cfg):
    envelope = json.loads(Path(path).read_text())
    if envelope['sha256'] != fingerprint(envelope['rows']):
        raise ValueError('Checkpoint checksum mismatch')
    validate_checkpoint(envelope['rows'], sid, cp, cfg['arm'], cfg['mock'])
    return envelope['rows']


def materialize(out, cfg):
    rows, summaries = [], []
    for sid in cfg['ids']:
        heads = []
        for cp in cfg['checkpoints']:
            p = out/'checkpoints'/f'{sid}_{cp}.json'
            if p.exists():
                rr = load_checkpoint(p, sid, cp, cfg)
                rows.extend(rr); heads.append(rr[0])
        if len(heads) == len(cfg['checkpoints']):
            summaries.append(dict(sample_id=sid, arm=cfg['arm'], n_checkpoints=len(heads),
                                  n_generated=10*len(heads), mock=cfg['mock'],
                                  accuracy=sum(r['realized_next10'] for r in heads)/len(heads)))
    atomic_text(out/'rows.jsonl', ''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    atomic_text(out/'trajectories.jsonl', ''.join(json.dumps(r)+'\n' for r in summaries))
    return summaries


def run_recovery_arm(cfg, trajectories, engine, out, resume=False):
    out = Path(out)
    if cfg['arm'] not in ARMS:
        raise ValueError('Only MR arms allowed')
    if out.exists():
        if not resume:
            raise FileExistsError(out)
        if json.loads((out/'config.json').read_text()) != cfg:
            raise ValueError('Resume config/source/model/data mismatch')
    else:
        (out/'checkpoints').mkdir(parents=True)
        atomic_text(out/'config.json', json.dumps(cfg, indent=2)+'\n')
    materialize(out, cfg)
    with (out/'run.log').open('a') as log:
        for tr in trajectories:
            for cp in cfg['checkpoints']:
                path = out/'checkpoints'/f'{tr.sample_id}_{cp}.json'
                if path.exists():
                    continue
                rr = engine.checkpoint(tr, cp, cfg['arm'])
                validate_checkpoint(rr, tr.sample_id, cp, cfg['arm'], cfg['mock'])
                atomic_text(path, json.dumps(dict(rows=rr, sha256=fingerprint(rr)), allow_nan=False)+'\n')
                materialize(out, cfg)
                message = f"{cfg['arm']} saved id={tr.sample_id} checkpoint={cp} mock={cfg['mock']}"
                log.write(message+'\n'); log.flush(); print(message, flush=True)
    return materialize(out, cfg)
