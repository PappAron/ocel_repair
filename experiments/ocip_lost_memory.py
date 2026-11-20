"""Re-run the Lost Memory experiment of Sadeghianasl et al. (2026) with the
authors' own code (AnomalyDetection_OCIP), unchanged, but with fixed seeds.

Reports (a) the authors' metric exactly as in their notebook cell 7.B and
(b) the same detector scored against the injection labels written by their
lost_memory() function. Also scores a one-line activity rule.

Usage: python ocip_lost_memory.py SEED [SEED ...]
"""
import os as _os, sys as _sys  # run from the repository root; makes adapters/ and core/ importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json
import os
import random
import sys
import time
import warnings

warnings.filterwarnings(action="ignore")  # as in the authors' notebook

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "AnomalyDetection_OCIP")
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

from error_patterns.lost_memory import lost_memory
from localocpa.algo.util.filtering.log.activity_filtering import filter_infrequent_activities
from localocpa.objects.log.importer.csv import factory as ocel_import_factory
from utils.helpers import calculate_classification_metrics


def binary_metrics(pred, label):
    pred = np.asarray(pred, dtype=bool)
    label = np.asarray(label, dtype=bool)
    tp = int((pred & label).sum()); fp = int((pred & ~label).sum())
    fn = int((~pred & label).sum()); tn = int((~pred & ~label).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Precision": precision, "Recall": recall, "F1": f1}


def main(seeds):
    t0 = time.time()
    # ---- notebook cell 2 (verbatim) ----
    filename = "sample-logs/bpi2017/BPI2017-Final.csv"
    object_types = ["application", "offer"]
    parameters = {"obj_names": object_types,
                  "val_names": [],
                  "act_name": "event_activity",
                  "time_name": "event_timestamp",
                  "sep": ","}
    print("Import started", flush=True)
    ocel = ocel_import_factory.apply(file_path=filename, parameters=parameters)
    print("Import done", round(time.time() - t0, 1), "s", flush=True)
    filtered_ocel = filter_infrequent_activities(ocel, 0.5)
    print("Filtered rows:", len(filtered_ocel.log.log), flush=True)

    # ---- notebook cell 7.B, part 1 (verbatim): train on the clean log ----
    temp_df = filtered_ocel.log.log
    x_col = ['event_activity', 'event_LoanGoal', 'event_ApplicationType', 'event_RequestedAmount', 'event_org:resource']
    y_train = temp_df['offer'].apply(len).astype(str).tolist()
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(temp_df[x_col])
    encoded_x = encoder.transform(temp_df[x_col])
    X_train = pd.DataFrame(encoded_x, columns=encoder.get_feature_names_out(x_col))
    model = LinearRegression()
    model.fit(X_train, y_train)

    # activities that carry an offer in the clean filtered log (for the one-line rule)
    offer_activities = sorted(set(temp_df.loc[temp_df['offer'].apply(len) > 0, 'event_activity']))

    results = []
    for seed in seeds:
        ts = time.time()
        random.seed(seed)  # the authors' code uses the unseeded global `random`
        # ---- notebook cell 7.A (verbatim) ----
        OCEL_polluted = lost_memory(filtered_ocel, ratio=0.1, related_objects=['application', 'offer'])

        # ---- notebook cell 7.B, part 2 (verbatim) ----
        test_df = OCEL_polluted.log.log
        y_test = test_df['offer'].apply(len).tolist()
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder.fit(test_df[x_col])
        encoded_x = encoder.transform(test_df[x_col])
        X_test = pd.DataFrame(encoded_x, columns=encoder.get_feature_names_out(x_col))
        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.5)

        # ---- notebook cell 7.B, part 3 (verbatim): the authors' metric ----
        res = pd.DataFrame({'pred': y_pred, 'label': y_test})
        theirs = calculate_classification_metrics(res, pred_col='pred', label_col='label')
        theirs = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in theirs.items()}

        # ---- added: the same detector scored against the injection labels ----
        injected = (test_df['label'].astype(str) != '').to_numpy()
        if injected.sum() == 0:
            raise SystemExit("lost_memory() injected nothing. Its chained assignment "
                             "(result[col].loc[mask] = '') has no effect under pandas >= 3 "
                             "(copy-on-write). Install pandas < 3, e.g. pandas==2.2.3. "
                             "Current pandas: " + pd.__version__)
        has_offer = np.asarray(y_test) > 0
        flag_regression = np.asarray(y_pred, dtype=bool) & ~has_offer  # should have an offer, but has none
        flag_rule = test_df['event_activity'].isin(offer_activities).to_numpy() & ~has_offer
        out = {
            "seed": seed,
            "rows": int(len(test_df)),
            "injected_rows": int(injected.sum()),
            "injected_offers": int(test_df.loc[injected, 'label'].nunique()),
            "authors_metric": theirs,
            "regression_vs_injection_labels": binary_metrics(flag_regression, injected),
            "activity_rule_vs_injection_labels": binary_metrics(flag_rule, injected),
            "check_FP_equals_injected_rows": theirs["FP"] == int(injected.sum()),
            "seconds": round(time.time() - ts, 1),
        }
        print(json.dumps(out), flush=True)
        results.append(out)

    meta = {"offer_activities": offer_activities,
            "pandas": pd.__version__,
            "filtered_rows": int(len(filtered_ocel.log.log)),
            "total_seconds": round(time.time() - t0, 1)}
    path = os.path.join(HERE, "results", "ocip_lost_memory_seeds_%s.json" % "_".join(map(str, seeds)))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "runs": results}, fh, indent=2)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or [0])
