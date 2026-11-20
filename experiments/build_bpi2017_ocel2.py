"""Convert ocpa's BPI2017-Final.csv (flat, list-valued object columns) into an
OCEL 2.0 SQLite log readable by ocel_repair's SqliteLoader.

- Events: all 393,931 rows; activity = event_activity; time = event_timestamp.
  Event attributes = the CSV's other event_* columns (as in the CSV).
- Objects: applications (31,509) and offers (42,995).
  Application attributes: LoanGoal, ApplicationType, RequestedAmount (constant per application).
  Offer attributes: taken from the offer's 'Create offer' event (OfferedAmount, MonthlyCost,
  NumberOfTerms, FirstWithdrawalAmount, CreditScore, Accepted, Selected).
- E2O: from the 'application' and 'offer' list columns (qualifier = object type).
- O2O: offer -> application ('belongs to'), derived from the clean log
  (every offer co-occurs with exactly one application).

Usage: python build_bpi2017_ocel2.py CSV_IN SQLITE_OUT
"""
import os as _os, sys as _sys  # run from the repository root; makes adapters/ and core/ importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import ast
import os
import re
import sqlite3
import sys

import pandas as pd

SRC, DST = sys.argv[1], sys.argv[2]
if os.path.exists(DST):
    sys.exit(f"{DST} exists; refusing to overwrite")

df = pd.read_csv(SRC)
lst = lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
df["_app"] = df["application"].map(lst)
df["_off"] = df["offer"].map(lst)
iso = lambda s: s.replace("/", "-") if isinstance(s, str) else s
df["_time"] = df["event_timestamp"].map(iso)
df["_eid"] = df["event_id"].astype(str)
type_map = lambda name: re.sub(r"[^A-Za-z0-9]", "", name.title())

drop = {"application", "offer", "event_activity", "event_timestamp", "event_None",
        "event_Unnamed: 0", "event_id", "event_CaseID"}
ev_attr = [c for c in df.columns if c.startswith("event_") and c not in drop]
rename = {c: c[len("event_"):] for c in ev_attr}

con = sqlite3.connect(DST)
cur = con.cursor()

# events
pd.DataFrame({"ocel_id": df["_eid"], "ocel_type": df["event_activity"]}).to_sql("event", con, index=False)
acts = sorted(df["event_activity"].unique())
pd.DataFrame({"ocel_type": acts, "ocel_type_map": [type_map(a) for a in acts]}).to_sql("event_map_type", con, index=False)
for a in acts:
    sub = df[df["event_activity"] == a]
    t = pd.DataFrame({"ocel_id": sub["_eid"], "ocel_time": sub["_time"]})
    t = pd.concat([t, sub[ev_attr].rename(columns=rename)], axis=1)
    name = f"event_{type_map(a)}"
    t.to_sql(name, con, index=False)
    cur.execute(f'CREATE UNIQUE INDEX "ix_{name}" ON "{name}"(ocel_id)')

# E2O
e2o = []
for eid, apps, offs in zip(df["_eid"], df["_app"], df["_off"]):
    e2o += [(eid, a, "application") for a in apps]
    e2o += [(eid, o, "offer") for o in offs]
pd.DataFrame(e2o, columns=["ocel_event_id", "ocel_object_id", "ocel_qualifier"]).to_sql("event_object", con, index=False)

# objects: applications
app_rows = df[df["_app"].map(len) > 0].copy()
app_rows["_a"] = app_rows["_app"].map(lambda l: l[0])
first_app = app_rows.sort_values("_time").groupby("_a", sort=False).first()
apps = pd.DataFrame({"ocel_id": first_app.index, "ocel_time": first_app["_time"].values,
                     "ocel_changed_field": None,
                     "LoanGoal": first_app["event_LoanGoal"].values,
                     "ApplicationType": first_app["event_ApplicationType"].values,
                     "RequestedAmount": first_app["event_RequestedAmount"].values})
# offers: attributes from the 'Create offer' event
co = df[(df["event_activity"] == "Create offer") & (df["_off"].map(len) == 1)].copy()
co["_o"] = co["_off"].map(lambda l: l[0])
co = co.drop_duplicates("_o")
offer_cols = ["OfferedAmount", "MonthlyCost", "NumberOfTerms", "FirstWithdrawalAmount",
              "CreditScore", "Accepted", "Selected"]
offs = pd.DataFrame({"ocel_id": co["_o"].values, "ocel_time": co["_time"].values,
                     "ocel_changed_field": None})
for c in offer_cols:
    offs[c] = co[f"event_{c}"].values
all_offers = sorted(set(o for l in df["_off"] for o in l))
missing = sorted(set(all_offers) - set(offs["ocel_id"]))
if missing:
    print("offers without a single-offer 'Create offer' event:", len(missing))
    offs = pd.concat([offs, pd.DataFrame({"ocel_id": missing})], ignore_index=True)

pd.DataFrame({"ocel_id": list(apps["ocel_id"]) + list(offs["ocel_id"]),
              "ocel_type": ["application"] * len(apps) + ["offer"] * len(offs)}).to_sql("object", con, index=False)
pd.DataFrame({"ocel_type": ["application", "offer"], "ocel_type_map": ["Application", "Offer"]}).to_sql("object_map_type", con, index=False)
apps.to_sql("object_Application", con, index=False)
offs.to_sql("object_Offer", con, index=False)
cur.execute('CREATE INDEX "ix_object_Application" ON "object_Application"(ocel_id)')
cur.execute('CREATE INDEX "ix_object_Offer" ON "object_Offer"(ocel_id)')

# O2O offer -> application
o2a = {}
for apps_l, offs_l in zip(df["_app"], df["_off"]):
    for o in offs_l:
        for a in apps_l:
            o2a.setdefault(o, set()).add(a)
bad = {o: s for o, s in o2a.items() if len(s) != 1}
if bad:
    sys.exit(f"{len(bad)} offers linked to != 1 application")
pd.DataFrame([(o, next(iter(s)), "belongs to") for o, s in sorted(o2a.items())],
             columns=["ocel_source_id", "ocel_target_id", "ocel_qualifier"]).to_sql("object_object", con, index=False)
con.commit()

for t in ["event", "object", "event_object", "object_object"]:
    print(t, cur.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0])
con.close()
print("written", DST)
