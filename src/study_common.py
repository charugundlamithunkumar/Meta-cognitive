"""Shared definitions extracted unchanged from the archived study runners."""

BINS = [(0, 49), (50, 99), (100, 199), (200, 299)]


def parse_ids(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out
