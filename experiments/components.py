"""Lost Memory benchmark (Section 6.3): is each erased offer still connected, in the damaged E2O graph,
to the event it was erased from?
Usage (from the repository root): python experiments/components.py data/BPI2017-Final.csv
"""
import os as _os, sys as _sys  # run from the repository root; makes adapters/ and core/ importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys
from collections import Counter
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # sibling script
from bpi2017_lost_memory_reconstruction import load, inject
f, _ = load(sys.argv[1])
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb
for seed in [0, 1]:
    parent.clear()
    removed, sampled = inject(f, seed)
    for r, (a, o) in enumerate(zip(f["app"], f["off"])):
        objs = set(a) | set(o)
        if r in removed: objs.discard(removed[r])
        ev = ("E", r); find(ev)
        for x in objs: union(ev, x)
    same = Counter(find(("E", r)) == find(t) for r, t in removed.items())
    sizes = Counter(find(t) for t in sampled.values())
    comp = Counter(find(x) for x in list(parent))
    tsize = Counter(min(comp[find(t)], 99) for t in sampled.values())
    print(seed, "affected events whose erased offer is in the SAME component:", same[True], "of", len(removed))
    print("   component size of erased offers (offer + its Send events):", sorted(tsize.items())[:8])
