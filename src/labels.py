"""Reply parsing and the error taxonomy (implementation-plan.md §4).

Labels per turn, in priority order:
  ok           committed total == truth
  nonanswer    reply is not a well-formed integer (or 'N | C' in inline mode)
  inherited    well-formed, == prev + v, but != truth (prev was already wrong)
  numeric_slip well-formed and != prev + v (a fresh mistake at this step)
retrieval_like (exploratory, K=1 only): a numeric slip with value - prev equal to some
other dictionary value.
"""
import re

BARE_RE = re.compile(r"\s*(-?\d+)\s*")
INLINE_RE = re.compile(r"\s*(-?\d+)\s*\|\s*(\d+)\s*")
INT_RE = re.compile(r"-?\d+")


def parse_reply(raw, inline=False):
    """Return (value or None, wellformed, confidence or None).

    value = the integer just before '|' if present, else the last integer in the reply.
    """
    s = str(raw).replace(",", "")
    conf = None
    if inline:
        m = INLINE_RE.fullmatch(s)
        if m:
            return int(m.group(1)), True, int(m.group(2))
        left = s.split("|", 1)
        if len(left) == 2:
            mc = INT_RE.search(left[1])
            conf = int(mc.group()) if mc else None
        ints = INT_RE.findall(left[0])
        return (int(ints[-1]) if ints else None), False, conf
    m = BARE_RE.fullmatch(s)
    if m:
        return int(m.group(1)), True, None
    ints = INT_RE.findall(s)
    return (int(ints[-1]) if ints else None), False, None


def label_turn(value, wellformed, prev, v, truth):
    if value is not None and value == truth:
        return "ok"
    if not wellformed or value is None:
        return "nonanswer"
    if value == prev + v:
        return "inherited"
    return "numeric_slip"


def retrieval_like(value, prev, v, dict_values):
    if value is None:
        return False
    d = value - prev
    return d != v and d in dict_values


def next_prev(prev, value):
    """State the agent will build on next: its committed value if parseable, else unchanged."""
    return value if value is not None else prev


def assign_seeds(kinds):
    """For each turn, the index of the seed error (most recent nonanswer/numeric_slip that
    started the current wrong chain). None for ok turns and for seed turns themselves get
    their own index."""
    seeds, cur = [], None
    for i, k in enumerate(kinds):
        if k == "ok":
            cur = None
            seeds.append(None)
        elif k in ("nonanswer", "numeric_slip"):
            cur = i
            seeds.append(i)
        else:  # inherited
            seeds.append(cur)
    return seeds
