"""Every prompt string in one place (implementation-plan.md §5). Frozen in prereg.md before main runs."""


def dict_block(dictionary):
    return "\n".join(f"{k}: {v}" for k, v in dictionary.items())


# Prompt versions. v1 failed the Phase 0 smoke test (the model replied with the key's value,
# not the running total). v2/v3 are compared on calibration ids only; the chosen version is
# frozen in prereg.md (see implementation-plan.md §13).
PROMPT_VERSION = {"current": 1}

EXAMPLE_V2 = ("Worked example with a DIFFERENT dictionary (pear: 7, kiwi: -12, lime: 30), total starts at 0:\n"
              "Key: pear -> reply 7   (0 + 7)\n"
              "Key: kiwi -> reply -5  (7 - 12)\n"
              "Key: lime -> reply 25  (-5 + 30)\n"
              "Key: pear -> reply 32  (25 + 7)\n"
              "Always reply with the new running TOTAL of all values so far, never the value of the key alone.")


def agent_system(dictionary, K=1, inline=False, version=None):
    v = version or PROMPT_VERSION["current"]
    each = ("Each turn I send one key. Look up its value, add it to the running total"
            if K == 1 else
            f"Each turn I send {K} keys separated by commas. Look up each value and add all of them to the running total")
    if inline and v == 1:
        fmt = (", and reply in exactly this format: <new running total> | <confidence>, where <confidence> is an "
               "integer from 0 to 100 giving how confident you are that the new running total is exactly correct. "
               "No other words or working.")
    elif inline:
        # v2: the format is described in prose. Placeholders were copied literally by Granite
        # ('<88> | 100' with <..>; '? | ?' with letters N/C; diagnosed in scripts/diag_inline.py).
        # No numeric confidence example (models copy example values; instructor's E8).
        fmt = (", and reply on one line with the new running total, then a space, a vertical bar '|', a space, "
               "and your confidence as an integer from 0 to 100 that the total is exactly correct. The total must "
               "always be your best integer, even when you are unsure; express doubt only through the confidence. "
               "No words or working.")
    else:
        fmt = ", and reply with ONLY the new running total as a single integer (no words, no working)."
        if v == 1:
            fmt += " Example: if the total is 120 and the key's value is -45, reply: 75"
    body = ("You are an execution agent. Keep a running total that starts at 0.\n\n"
            f"Dictionary (key: value):\n{dict_block(dictionary)}\n\n{each}{fmt}")
    if v >= 2:
        ex = EXAMPLE_V2 if (K == 1 and not inline) else EXAMPLE_V2.replace(" -> reply ", " -> new total ")
        body += "\n\n" + ex
    return body


def keys_text(keys, version=None):
    v = version or PROMPT_VERSION["current"]
    base = ("Key: " if len(keys) == 1 else "Keys: ") + ", ".join(keys)
    return base + (". Reply with the new running total." if v >= 3 else "")


def turn_text(keys, note=None):
    """note = (turn_index_0based, total) correction from an auditor, shown before the key."""
    if note is None:
        return keys_text(keys)
    t, n = note
    return (f"Note: an independent check found the running total after turn {t + 1} should be {n}. "
            f"Continue from that total.\n{keys_text(keys)}")


def compact_turn(total, keys):
    return f"Your current running total: {total}.\n{keys_text(keys)}"


def recompute_system(dictionary):
    return f"You are a careful calculator.\n\nDictionary (key: value):\n{dict_block(dictionary)}"


def recompute_user(prev, keys):
    return (f"Current running total: {prev}. {keys_text(keys)}. Look up the value(s), add to the total, "
            "and reply with ONLY the new running total as a single integer.")


def audit_system(dictionary):
    return f"You are a careful auditor.\n\nDictionary (key: value):\n{dict_block(dictionary)}"


def audit_user(t0, V, keys_since, t, R):
    start = ("The running total started at 0 before turn 1." if t0 < 0
             else f"The verified running total after turn {t0 + 1} was {V}.")
    return (f"Audit. {start} The keys since then, in order, were: {', '.join(keys_since)}. "
            f"The reported running total after turn {t + 1} is {R}. "
            f"Recompute the total starting from {V} using the dictionary, and check whether the reported total is "
            "exactly correct. End your reply with exactly one line: 'VERDICT: CORRECT' or "
            "'VERDICT: WRONG, total = <N>'.")


# Granite 4.2's official chat template implements low-effort thinking by appending this to the
# last user message (low_effort=True). tests/test_core.py checks it against the template.
LOW_EFFORT_SUFFIX = "\n\n{reasoning effort: low}"

P_TRUE ="Is the running total you just reported exactly correct? Answer Yes or No."
P_VERBAL = ("How confident are you, from 0 to 100, that the running total you just reported is exactly correct? "
            "Reply with only a number.")
P_FORECAST = ("Over your next 10 turns, what percentage of your running totals do you expect to be exactly correct? "
              "Reply with only a number from 0 to 100.")
P_AWARE = ("Compared with the start of this session, is your accuracy on these turns now: A) higher, "
           "B) about the same, or C) lower? Answer with a single letter.")


def evidence_line(n_wrong, n_window):
    return (f"An independent checker recomputed your last {n_window} running totals and found "
            f"{n_wrong} of them wrong. ")
