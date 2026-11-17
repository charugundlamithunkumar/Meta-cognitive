"""Controlled paired histories for HG; never changes the underlying task truth."""
import hashlib
import json
import math
import random

from . import prompts as PR
from .meta_history import numeric_slip

ARMS = ('MR_A', 'MR_B', 'MR_C', 'MR_D')
CHECKPOINTS = [10, 25, 50, 100, 150, 200, 250]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def build_recovery_history(trajectory, checkpoint, arm, dictionary, seed=0):
    if arm not in ARMS or not 0 <= checkpoint < len(trajectory.turns) - 10:
        raise ValueError('Invalid MR arm/checkpoint or incomplete future window')
    corrupt = arm in ('MR_B', 'MR_C')
    order = list(range(checkpoint))
    random.Random(seed*1000003 + trajectory.sample_id*10007 + checkpoint).shuffle(order)
    mask = set(order[:math.floor(.5*checkpoint + .5)]) | {checkpoint} if corrupt else set()
    history = []
    for t, turn in enumerate(trajectory.turns[:checkpoint+1]):
        value = numeric_slip(turn, dictionary) if t in mask else turn.truth
        history.append(dict(turn=t, keys=list(turn.keys), truth=turn.truth, value=value,
                            injected=t in mask, user=PR.keys_text(turn.keys, version=2),
                            assistant=str(value)))
    truth = trajectory.turns[checkpoint].truth
    state = truth if arm in ('MR_A', 'MR_C') else numeric_slip(trajectory.turns[checkpoint], dictionary)
    offsets = [h['value']-h['truth'] for h in history]
    return dict(turns=history, injected_positions=sorted(mask), full_length=len(history),
                n_wrong_earlier=len(mask-{checkpoint}), requested_earlier_rate=.5 if corrupt else 0.,
                actual_earlier_rate=len(mask-{checkpoint})/checkpoint if checkpoint else None,
                actual_full_rate=len(mask)/len(history), history_sha256=fingerprint(history),
                truth=truth, supplied_state=state, state_correct=state == truth,
                state_width=len(str(abs(state))), state_magnitude=abs(state),
                state_sign=(state > 0)-(state < 0),
                correction_transitions=sum(a != b for a, b in zip(offsets, offsets[1:])))
