"""HG's only new prompt; all other prompts are imported from prompts.py."""

STATE_CUE = 'Continue using this current running total: {s}.'


def state_cue(state):
    return STATE_CUE.format(s=state)
