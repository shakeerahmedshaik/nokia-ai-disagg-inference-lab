#!/usr/bin/env python3
"""Usage: scripts/ecmp-delta.py before.txt after.txt
Turns two `scripts/fabric.sh counters` snapshots into spine interface byte deltas (MB).
Spine port ethernet-1/1 faces leaf1 (prefill side), ethernet-1/2 faces leaf2 (decode side)."""
import re
import sys


def load(path):
    out = {}
    for line in open(path):
        m = re.match(r"(spine\d)\s+(ethernet-1/\d)\s+(.*)", line)
        if m:
            out[(m.group(1), m.group(2))] = {k: int(v) for k, v in re.findall(r"([\w-]+)=(\d+)", m.group(3))}
    return out


b, a = load(sys.argv[1]), load(sys.argv[2])
print(f"{'spine':<7}{'port':<14}{'towards':<8}{'in_MB(from leaf)':>18}{'out_MB(to leaf)':>17}")
for key in sorted(a):
    d = {k: a[key][k] - b[key][k] for k in a[key]}
    towards = "leaf1" if key[1].endswith("/1") else "leaf2"
    print(f"{key[0]:<7}{key[1]:<14}{towards:<8}{d['in-octets'] / 1e6:>18.1f}{d['out-octets'] / 1e6:>17.1f}")
