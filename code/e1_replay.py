"""Consequences of the logged E1 decisions (Bank Marketing): apply each logged drop set, train, and score on
random and chronological deployment-proxy splits. `duration` is median-filled at deployment when retained."""
import sys, json, collections, numpy as np, pandas as pd
import audit_core as A
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
OUT = Path(__file__).parent.parent / "results" / "e1_replay.csv"
b = pd.read_csv(str(__import__("pathlib").Path(__file__).parent.parent / "data" / "bank_additional_full.csv"), sep=";"); T, P = "y", "yes"
V = [json.loads(l) for l in open(str(__import__("pathlib").Path(__file__).parent.parent / "agent_runs" / "decision_logs.jsonl"))]
E1 = [x for x in V if x["exp"] == "E1_bank_duration"]
sets = sorted(set(tuple(sorted(c for c in x["columns_to_drop"] if c != T)) for x in E1))
sets = sorted(set(sets) | {("duration",)})          # add the leakage-controlled reference
which = sys.argv[1]                                  # 'temporal' or seed list e.g. 0,1,2
rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
def run(tr, va, de, split, seed):
    for s in sets:
        tr_, va_, de_ = (x.drop(columns=list(s)) for x in (tr, va, de))
        for k in A.CLASSIFIERS:
            Xt, yt, Xv, yv = A.make_xy(tr_, va_, T, P); _, _, Xd, yd = A.make_xy(tr_, de_, T, P)
            if "duration" in Xd.columns: Xd["duration"] = float(tr_["duration"].median())
            m = A.build_classifier(k); m.fit(Xt, yt); pv, pd_ = m.predict(Xv), m.predict(Xd)
            rows.append(dict(split=split, seed=seed, drop_set="|".join(s) or "(none)", clf=k,
                             val_acc=accuracy_score(yv, pv), val_f1=f1_score(yv, pv, average="macro"),
                             dep_acc=accuracy_score(yd, pd_), dep_f1=f1_score(yd, pd_, average="macro")))
    pd.DataFrame(rows).to_csv(OUT, index=False)
if which == "temporal":
    n = len(b); i1, i2 = int(.6 * n), int(.8 * n)
    run(b.iloc[:i1].reset_index(drop=True), b.iloc[i1:i2].reset_index(drop=True), b.iloc[i2:].reset_index(drop=True), "temporal", 0)
else:
    for s in map(int, which.split(",")):
        run(*A.split(b, T, s), "random", s); print("seed", s, flush=True)
print(len(sets), "sets done", which)
