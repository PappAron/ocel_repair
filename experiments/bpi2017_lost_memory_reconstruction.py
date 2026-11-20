"""Lost Memory reconstruction on BPI 2017, in the setting of Sadeghianasl et al. (2026).

1. Rebuilds the authors' polluted logs exactly: same file (ocpa BPI2017-Final.csv),
   same activity filter (filter_infrequent_activities(ocel, 0.5)), same selection
   (lost_memory: one offer per application, 10% of applications, random.seed(seed)).
   The number of affected events per seed is checked against the authors' own code
   (results/ocip_lost_memory_seeds_0_1_2_3_4.json).
2. For every affected event (an event that lost its offer link), ranks the candidate
   offers (all 42,995 offers in the log's object set, minus those still linked to
   the event) and reports Hits@k and MRR (ties: expected rank under random
   tie-breaking).

Methods (all use only the polluted log, never the ground truth):
- cooccurrence: pairwise co-occurrence model of the DS Lab 1 paper,
  score(c) = sum_{o in O(e)} log(P(c | o) + 1e-9), statistics from the polluted log.
- orphan_time: offers that no longer co-occur with any application are "orphans";
  orphans rank first, ordered by the time distance between the affected event and
  the offer's nearest event; all other offers rank after them, ordered the same way.
- time_only: all offers ordered by the time distance to the offer's nearest event
  (ablation: how much of orphan_time comes from time alone).
- orphan_only: orphans first, all orphans tied, all other offers tied after them
  (structure only, no time: the rule counterpart of a model without timestamps).

Usage: python bpi2017_lost_memory_reconstruction.py CSV OUT_JSON [SEED ...]
"""
import os as _os, sys as _sys  # run from the repository root; makes adapters/ and core/ importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import ast
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

KEEP_THRESHOLD = 0.5
RATIO = 0.1
KS = (1, 3, 10)


def load(csv_path):
    df = pd.read_csv(csv_path, usecols=["event_id", "application", "offer",
                                         "event_activity", "event_timestamp"])
    # filter_infrequent_activities(ocel, 0.5): keep the most frequent activities
    # until the cumulative share first exceeds the threshold (authors' localocpa code)
    counts = Counter(df["event_activity"])
    total = sum(counts.values())
    keep, acc = [], 0.0
    for act, n in counts.most_common():
        keep.append(act)
        acc += n / total
        if acc > KEEP_THRESHOLD:
            break
    f = df[df["event_activity"].isin(keep)].reset_index(drop=True)
    lst = lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
    f["app"] = f["application"].map(lst)
    f["off"] = f["offer"].map(lst)
    f["t"] = pd.to_datetime(f["event_timestamp"], format="%Y/%m/%d %H:%M:%S.%f").astype("int64") / 1e9
    return f, keep


def inject(f, seed):
    """Replicates error_patterns/lost_memory.py on the filtered log."""
    first = lambda l: l[0] if l else ""
    fa, fo = f["app"].map(first), f["off"].map(first)
    corr = pd.DataFrame({"a": fa, "o": fo}).drop_duplicates().reset_index(drop=True)
    corr = corr.loc[(corr["a"] != "") & (corr["o"] != "")]
    dict_replace = dict(corr.values)
    random.seed(seed)
    keys = random.sample(list(dict_replace.keys()), round(len(dict_replace) * RATIO))
    sampled = {k: dict_replace[k] for k in keys}
    hit = pd.DataFrame({"a": fa, "o": fo, "row": range(len(f))}).merge(
        pd.DataFrame(list(sampled.items()), columns=["a", "o"]), on=["a", "o"])
    affected = sorted(hit["row"].tolist())
    removed = {r: fo.iloc[r] for r in affected}           # row -> hidden offer
    return removed, sampled


def evaluate(ranks):
    ranks = np.asarray(ranks, dtype=float)
    out = {f"hits@{k}": float(np.mean(ranks <= k)) for k in KS}
    out["mrr"] = float(np.mean(1.0 / ranks))
    out["n"] = int(len(ranks))
    return out


def expected_rank(scores, target_score):
    """1 + #strictly better + (#ties - 1) / 2, higher score = better."""
    better = int(np.sum(scores > target_score))
    ties = int(np.sum(scores == target_score))
    return 1 + better + (ties - 1) / 2


def run_seed(f, seed, all_offers, offer_index):
    removed, sampled = inject(f, seed)
    # observed (polluted) object sets per row
    obs = [set(a) | set(o) for a, o in zip(f["app"], f["off"])]
    for r, o in removed.items():
        obs[r] = obs[r] - {o}

    # ---- co-occurrence statistics from the polluted log ----
    cnt, pair = Counter(), Counter()
    for s in obs:
        for x in s:
            cnt[x] += 1
        s = sorted(s)
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                pair[s[i], s[j]] += 1
                pair[s[j], s[i]] += 1
    cond = defaultdict(dict)
    for (x, y), n in pair.items():
        cond[x][y] = n / cnt[x]

    # ---- per-offer time stamps and orphan status in the polluted log ----
    offer_times = defaultdict(list)
    offer_with_app = set()
    for s, t in zip(obs, f["t"]):
        offs = [x for x in s if x.startswith("Offer_")]
        has_app = any(x.startswith("Application_") for x in s)
        for o in offs:
            offer_times[o].append(t)
            if has_app:
                offer_with_app.add(o)
    orphan = np.array([o in offer_times and o not in offer_with_app for o in all_offers])
    # padded matrix of event times per offer (inf = no event) for vectorised distances
    width = max(len(v) for v in offer_times.values())
    T = np.full((len(all_offers), width), np.inf)
    for o, v in offer_times.items():
        T[offer_index[o], :len(v)] = v

    def nearest_dist(t_e):
        return np.min(np.abs(T - t_e), axis=1)

    ranks = {"cooccurrence": [], "orphan_only": [], "orphan_time": [], "time_only": []}
    floor = math.log(1e-9)
    for r, target in removed.items():
        observed = obs[r]
        excl = [offer_index[o] for o in observed if o in offer_index]
        ti = offer_index[target]
        # co-occurrence
        sc = np.full(len(all_offers), floor * len(observed))
        for o in observed:
            for c, p in cond[o].items():
                if c in offer_index:
                    sc[offer_index[c]] += math.log(p + 1e-9) - floor
        sc[excl] = -np.inf
        ranks["cooccurrence"].append(expected_rank(sc, sc[ti]))
        # orphan only (no time)
        s_o = np.where(orphan, 1.0, 0.0)
        s_o[excl] = -np.inf
        ranks["orphan_only"].append(expected_rank(s_o, s_o[ti]))
        # time based
        d = nearest_dist(f["t"].iloc[r])
        s_time = -d
        s_time[excl] = -np.inf
        ranks["time_only"].append(expected_rank(s_time, s_time[ti]))
        s_orph = np.where(orphan, 1e12, 0.0) - d
        s_orph[excl] = -np.inf
        ranks["orphan_time"].append(expected_rank(s_orph, s_orph[ti]))

    # stratify by number of offers of the application in the clean log
    offers_per_app = Counter()
    seen = set()
    for a, o in zip(f["app"], f["off"]):
        for x in a:
            for y in o:
                if (x, y) not in seen:
                    seen.add((x, y))
                    offers_per_app[x] += 1
    row_app = {r: f["app"].iloc[r][0] for r in removed}
    single = [i for i, r in enumerate(removed) if offers_per_app[row_app[r]] == 1]
    multi = [i for i, r in enumerate(removed) if offers_per_app[row_app[r]] > 1]

    res = {"seed": seed,
           "affected_events": len(removed),
           "deleted_offers": len(sampled),
           "deleted_offers_still_in_log": int(sum(o in offer_times for o in sampled.values())),
           "affected_by_activity": dict(Counter(f["event_activity"].iloc[list(removed)])),
           "orphan_offers": int(orphan.sum()),
           "affected_single_offer_app": len(single),
           "affected_multi_offer_app": len(multi)}
    for m, rk in ranks.items():
        res[m] = {"all": evaluate(rk),
                  "single_offer_app": evaluate([rk[i] for i in single]),
                  "multi_offer_app": evaluate([rk[i] for i in multi])}
    return res


def main():
    csv_path, out_path = sys.argv[1], sys.argv[2]
    seeds = [int(s) for s in sys.argv[3:]] or [0, 1, 2, 3, 4]
    t0 = time.time()
    f, keep = load(csv_path)
    # candidate set: every offer of the full log (object table), not only those in the filtered log
    full = pd.read_csv(csv_path, usecols=["offer"])["offer"]
    lst = lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
    all_offers = sorted({o for l in full.map(lst) for o in l})
    offer_index = {o: i for i, o in enumerate(all_offers)}
    print(f"filtered rows {len(f)}, kept activities {keep}, candidate offers {len(all_offers)}", flush=True)
    runs = []
    for s in seeds:
        ts = time.time()
        r = run_seed(f, s, all_offers, offer_index)
        r["seconds"] = round(time.time() - ts, 1)
        print(json.dumps(r), flush=True)
        runs.append(r)
    json.dump({"meta": {"rows": len(f), "kept_activities": keep,
                        "candidate_offers": len(all_offers),
                        "seconds": round(time.time() - t0, 1)},
               "runs": runs}, open(out_path, "w"), indent=2)
    print("saved", out_path)


if __name__ == "__main__":
    main()
