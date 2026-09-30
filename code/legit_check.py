"""PIG vs DVGex for legitimate strong predictors (available at deployment) vs. a leak (unavailable)."""
import sys, json, numpy as np, pandas as pd
import audit_core as A
from sklearn.metrics import accuracy_score
cols = ['age','workclass','fnlwgt','education','education.num','marital.status','occupation','relationship','race','sex','capital.gain','capital.loss','hours.per.week','native.country','income']
d = pd.read_csv(__import__("pathlib").Path(__file__).parent.parent / "data" / "adult-all.csv", header=None, names=cols).iloc[:32561]
for c in d.select_dtypes(include="object").columns: d[c] = d[c].astype(str).str.strip()
rows = []
for s in range(3):
    tr, va, de = A.split(d, "income", s)
    for k in A.CLASSIFIERS:
        Xt, yt, Xv, yv = A.make_xy(tr, va, "income", ">50K"); _, _, Xd, yd = A.make_xy(tr, de, "income", ">50K")
        m = A.build_classifier(k); m.fit(Xt, yt); av, ad = accuracy_score(yv, m.predict(Xv)), accuracy_score(yd, m.predict(Xd))
        for feat in ["relationship", "capital.gain"]:
            tr_, va_, de_ = (x.drop(columns=[feat]) for x in (tr, va, de))
            Xt, yt, Xv, yv = A.make_xy(tr_, va_, "income", ">50K"); _, _, Xd, yd = A.make_xy(tr_, de_, "income", ">50K")
            m2 = A.build_classifier(k); m2.fit(Xt, yt); lv, ld = accuracy_score(yv, m2.predict(Xv)), accuracy_score(yd, m2.predict(Xd))
            rows.append(dict(seed=s, clf=k, feature=feat, PIG=av - lv, DVG=av - ad, DVGex=(av - ad) - (lv - ld)))
    print("seed", s, flush=True)
r = pd.DataFrame(rows); r.to_csv(__import__("pathlib").Path(__file__).parent.parent / "results" / "legit_feature_check.csv", index=False)
print(r.groupby("feature")[["PIG", "DVG", "DVGex"]].agg(["mean", "min", "max"]).round(4))
