"""M histories and CPU adapters; the registered Study-1 files remain untouched."""
import importlib.util
import math
import random
import sys
from functools import lru_cache
from pathlib import Path

from . import prompts as PR

ROOT = Path(__file__).resolve().parent.parent
ARMS = {'L0': (0., None), 'L25': (.25, None), 'L50': (.5, None),
        'L75': (.75, None), 'L100': (1., None), 'S0': (0., 5), 'S100': (1., 5)}


class MockTokenizer:
    """Minimal character tokenizer for CPU tests; never used for real measurements."""
    eos_token_id = 0
    unk_token_id = -1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 1 for c in text]

    def decode(self, ids):
        return ''.join(chr(i - 1) for i in ids if i != 0)

    def convert_tokens_to_ids(self, token):
        return self.eos_token_id

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        result = ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            result += '<|im_start|>assistant\n' + ('<think>' if enable_thinking else '<think></think>')
        return self.encode(result) if tokenize else result


@lru_cache(maxsize=1)
def existing_fake_class():
    """Reuse the existing +1 generator directly, without copying/changing its rule.

    The only generator in this checkout lives in the forced-error test harness.
    Keep this explicit dependency until a separately registered refactor moves it.
    Loading the module does not run its unittest entry point or initialize vLLM.
    """
    spec = importlib.util.spec_from_file_location('meta_existing_fake', ROOT / 'previous work/tests/test_arms_mock.py')
    module = importlib.util.module_from_spec(spec)
    original_path = list(sys.path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
    return module.FakeLLM


def numeric_slip(turn, dictionary):
    if len(turn.keys) != 1:
        raise ValueError('The existing numeric-slip generator supports K=1 only')
    key = turn.keys[0]
    fake = existing_fake_class()(MockTokenizer(), dictionary, {key: 'slip'})
    text = (f'<|im_start|>assistant\n{turn.truth - turn.value}<|im_end|>\n'
            f'<|im_start|>user\n{PR.keys_text(turn.keys, version=2)}<|im_end|>\n')
    value = int(fake.respond(text).outputs[0].text)
    if value == turn.truth:
        raise ValueError('Numeric-slip generator returned truth')
    return value


def build_history(trajectory, checkpoint, rate, dictionary, window=None, seed=0):
    """Rate controls wrong totals in c+1 prefix turns; masks nest across doses."""
    if not 0 <= rate <= 1 or not 0 <= checkpoint < len(trajectory.turns):
        raise ValueError('Invalid rate or checkpoint')
    if window is not None and window < 1:
        raise ValueError('window must be positive')
    count = checkpoint + 1
    order = list(range(count))
    random.Random(seed * 1000003 + trajectory.sample_id * 10007 + checkpoint).shuffle(order)
    mask = set(order[:math.floor(rate * count + .5)])
    history = []
    for t, turn in enumerate(trajectory.turns[:count]):
        value = numeric_slip(turn, dictionary) if t in mask else turn.truth
        history.append(dict(turn=t, keys=list(turn.keys), truth=turn.truth, value=value, injected=t in mask,
                            user=PR.keys_text(turn.keys, version=2), assistant=str(value)))
    shown = history[-window:] if window is not None else history
    return dict(turns=shown, full_length=count, shown_length=len(shown), requested_rate=rate,
                n_wrong_full=len(mask), actual_rate_full=len(mask) / count,
                injected_positions=sorted(mask), n_wrong_shown=sum(h['injected'] for h in shown))
