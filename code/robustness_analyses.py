"""
Revision analyses for "Auditing Validity and Fairness Failures in LLM-Driven Data
Preparation Pipelines".  No LLM calls are made.  Every analysis holds the logged agent
decision fixed (trap retained vs. removed; flagged rows removed vs. kept) and re-draws
everything else that the original single-seed runs held fixed: the 60/20/20 split,
the trap noise, and the flagged subset.

Parts
  german    : trap-free DVG noise floor + split-robust PIG/DVG replay (German Credit)
  adult     : noise floor, split-robust PIG/DVG replay, imputation robustness,
              covariate/label-shift deployment, official UCI test release (Adult)
  deletion  : split-robust minority-deletion replay + random-positive control (Adult)
  bank      : naturally occurring leakage case study (Bank Marketing `duration`),
              random and chronological (temporal) deployment splits
Usage: python revision_analyses.py <part> [n_seeds]
"""
import sys, time, json
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

OUT = Path(__file__).parent.parent / "results"; OUT.mkdir(exist_ok=True, parents=True)
DATA = Path(__file__).parent.parent / "data"
GERMAN_CSV = str(DATA / "german_credit_data_with_risk.csv")
INTENSITIES = [0.60, 0.80, 0.95]


def acc_f1(y, p):
    return accuracy_score(y, p), f1_score(y, p, average="macro")


# ─────────────────────────────── German Credit ────────────────────────────────
def run_german(n_seeds=50):
    import german_core as G
    d = G.load_german(GERMAN_CSV, faithful=True)
    rows = []
    for s in range(n_seeds):
        tr, va, de = G.split(d, s)
        for k in G.CLASSIFIERS:
            # trap-free (= trap removed by agent)
            Xt, yt, Xv, yv = G.encode(tr, va); _, _, Xd, yd = G.encode(tr, de)
            m = G.clf(k); m.fit(Xt, yt)
            va_c = acc_f1(yv, m.predict(Xv)); de_c = acc_f1(yd, m.predict(Xd))
            # header-artefact check: drop former row index feature_0
            tr0, va0, de0 = (x.drop(columns=["feature_0"]) for x in (tr, va, de))
            Xt0, yt0, Xv0, yv0 = G.encode(tr0, va0); _, _, Xd0, yd0 = G.encode(tr0, de0)
            m0 = G.clf(k); m0.fit(Xt0, yt0)
            rows.append(dict(dataset="German", seed=s, clf=k, cond="clean", r=0.0,
                             val_acc=va_c[0], val_f1=va_c[1], dep_acc=de_c[0], dep_f1=de_c[1],
                             dep_acc_noidx=accuracy_score(yd0, m0.predict(Xd0)),
                             val_acc_noidx=accuracy_score(yv0, m0.predict(Xv0)),
                             n_val=len(va), n_dep=len(de)))
            for r in INTENSITIES:
                trt = G.add_proxy(tr, r, "repayment_score", s)
                vat = G.add_proxy(va, r, "repayment_score", s + 1)
                Xt, yt, Xv, yv = G.encode(trt, vat); _, _, Xd, yd = G.encode(trt, de)
                m = G.clf(k); m.fit(Xt, yt)
                v = acc_f1(yv, m.predict(Xv)); dd = acc_f1(yd, m.predict(Xd))
                rng = np.random.default_rng(1000 + s)
                Xp = Xd.copy(); Xp["repayment_score"] = rng.permutation(Xt["repayment_score"].values)[:len(Xd)]
                dp = acc_f1(yd, m.predict(Xp))
                rows.append(dict(dataset="German", seed=s, clf=k, cond="retained", r=r,
                                 val_acc=v[0], val_f1=v[1], dep_acc=dd[0], dep_f1=dd[1],
                                 dep_acc_perm=dp[0], lc_val_acc=va_c[0], lc_val_f1=va_c[1],
                                 n_val=len(va), n_dep=len(de)))
        print(f"german seed {s} done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "german_replay.csv", index=False)


# ─────────────────────────────── Adult helpers ────────────────────────────────
ADULT_COLS = ['age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital.status',
              'occupation', 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss',
              'hours.per.week', 'native.country', 'income']


def load_adult_both():
    """UCI Adult: adult.data (32,561 records; the file used in all experiments) and
    the separately distributed official test release adult.test (16,281 records)."""
    a = pd.read_csv(DATA / "adult-all.csv", header=None, names=ADULT_COLS)
    for c in a.select_dtypes(include="object").columns:
        a[c] = a[c].astype(str).str.strip()
    a["income"] = a["income"].str.replace(".", "", regex=False)
    return a.iloc[:32561].reset_index(drop=True), a.iloc[32561:].reset_index(drop=True)


def shifted_indices(df, kind, rng, n=3000):
    """Resample the deployment-proxy split without replacement to a shifted composition."""
    f = (df["sex"] == "Female").to_numpy(); old = (df["age"] >= 45).to_numpy()
    pos = (df["income"] == ">50K").to_numpy()
    spec = {"female_15": (f, 0.15), "female_50": (f, 0.50), "age45_20": (old, 0.20),
            "age45_60": (old, 0.60), "prev_12": (pos, 0.12), "prev_40": (pos, 0.40)}[kind]
    mask, share = spec
    a = np.flatnonzero(mask); b = np.flatnonzero(~mask)
    na = int(round(n * share))
    return np.concatenate([rng.choice(a, na, replace=False), rng.choice(b, n - na, replace=False)])


SHIFTS = ["female_15", "female_50", "age45_20", "age45_60", "prev_12", "prev_40"]


def run_adult(n_seeds=20, start=0, tag=""):
    import audit_core as A
    data, test = load_adult_both()
    T, P = "income", ">50K"
    rows = []
    for s in range(start, n_seeds):
        t0 = time.time()
        tr, va, de = A.split(data, T, s)
        rng = np.random.default_rng(500 + s)
        sh_idx = {k: shifted_indices(de, k, rng) for k in SHIFTS}
        for k in A.CLASSIFIERS:
            Xt, yt, Xv, yv = A.make_xy(tr, va, T, P)
            _, _, Xd, yd = A.make_xy(tr, de, T, P)
            _, _, Xe, ye = A.make_xy(tr, test, T, P)
            m = A.build_classifier(k); m.fit(Xt, yt)
            pv, pdp, pe = m.predict(Xv), m.predict(Xd), m.predict(Xe)
            va_c, de_c, ex_c = acc_f1(yv, pv), acc_f1(yd, pdp), acc_f1(ye, pe)
            row = dict(dataset="Adult", seed=s, clf=k, cond="clean", r=0.0, val_acc=va_c[0], val_f1=va_c[1],
                       dep_acc=de_c[0], dep_f1=de_c[1], ext_acc=ex_c[0], ext_f1=ex_c[1],
                       n_val=len(va), n_dep=len(de), n_ext=len(test))
            for kk, ii in sh_idx.items():
                row[f"shift_{kk}"] = accuracy_score(yd[ii], pdp[ii])
            rows.append(row)
            for r in INTENSITIES:
                col = "derived_income_score"
                trt = A.add_leakage(tr, T, P, r, col, s)
                vat = A.add_leakage(va, T, P, r, col, s + 100)
                Xt, yt, Xv, yv = A.make_xy(trt, vat, T, P)
                _, _, Xd, yd = A.make_xy(trt, de, T, P)      # trap absent -> 0 (training mean)
                _, _, Xe, ye = A.make_xy(trt, test, T, P)
                m = A.build_classifier(k); m.fit(Xt, yt)
                v = acc_f1(yv, m.predict(Xv)); pdp = m.predict(Xd)
                dd = acc_f1(yd, pdp); ee = acc_f1(ye, m.predict(Xe))
                Xm = Xd.copy(); Xm[col] = float(np.median(Xt[col]))          # median imputation
                Xq = Xd.copy(); Xq[col] = rng.permutation(Xt[col].values)[:len(Xd)]  # uninformative draw
                row = dict(dataset="Adult", seed=s, clf=k, cond="retained", r=r, val_acc=v[0], val_f1=v[1],
                           dep_acc=dd[0], dep_f1=dd[1], ext_acc=ee[0], ext_f1=ee[1],
                           dep_acc_median=accuracy_score(yd, m.predict(Xm)),
                           dep_acc_perm=accuracy_score(yd, m.predict(Xq)),
                           lc_val_acc=va_c[0], lc_val_f1=va_c[1],
                           n_val=len(va), n_dep=len(de), n_ext=len(test))
                for kk, ii in sh_idx.items():
                    row[f"shift_{kk}"] = accuracy_score(yd[ii], pdp[ii])
                rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / f"adult_replay{tag}.csv", index=False)
        print(f"adult seed {s} done in {time.time()-t0:.0f}s", flush=True)


# ─────────────────────────────── Deletion (3B) ────────────────────────────────
def run_deletion(n_seeds=20, start=0, tag=""):
    """Stage 3B configuration: ordinal encoding, 300-tree ensembles, LR max_iter=5000,
    trap column dropped before modelling, fairness on the deployment-proxy split."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    import german_core as G  # reuse the identical ordinal encoder

    def clf(k):
        return {"logistic": LogisticRegression(max_iter=5000, random_state=42, n_jobs=-1),
                "rf": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
                "xgboost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
                                         colsample_bytree=0.9, eval_metric="logloss", random_state=42, n_jobs=-1),
                "lightgbm": LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)}[k]

    data, test = load_adult_both()
    data = data.copy(); test = test.copy()
    data["income"] = (data["income"] == ">50K").astype(int); test["income"] = (test["income"] == ">50K").astype(int)
    G.TARGET = "income"

    def group(ev, y, p):
        f = (ev["sex"] == "Female").to_numpy(); m_ = (ev["sex"] == "Male").to_numpy()
        tf = p[f & (y == 1)].mean(); tm = p[m_ & (y == 1)].mean()
        return dict(acc=accuracy_score(y, p), prot_recall=tf, ref_recall=tm, eo=abs(tf - tm),
                    dp=abs(p[f].mean() - p[m_].mean()))

    def fit_eval(train, evs, k):
        out = {}; m = None
        for nm, ev in evs.items():
            Xt, yt, Xe, ye = G.encode(train, ev)
            if m is None: m = clf(k); m.fit(Xt, yt)
            out[nm] = group(ev.reset_index(drop=True), ye, m.predict(Xe))
        return out

    rows = []
    for s in range(start, n_seeds):
        t0 = time.time()
        tr, tmp = train_test_split(data, test_size=0.40, random_state=s, stratify=data["income"])
        va, de = train_test_split(tmp, test_size=0.50, random_state=s, stratify=tmp["income"])
        tr, de = tr.reset_index(drop=True), de.reset_index(drop=True)
        rng = np.random.default_rng(s)
        pp = np.flatnonzero(((tr["sex"] == "Female") & (tr["income"] == 1)).to_numpy())
        up = np.flatnonzero(((tr["sex"] == "Male") & (tr["income"] == 1)).to_numpy())
        allpos = np.flatnonzero((tr["income"] == 1).to_numpy())
        for k in ["logistic", "rf", "xgboost", "lightgbm"]:
            base = fit_eval(tr, {"dep": de, "ext": test}, k)
            for phi in [0.25, 0.50, 0.75]:
                n_flag = int(np.ceil(len(pp) * phi))
                variants = {"protected_positive": rng.choice(pp, n_flag, replace=False)}
                if phi == 0.75:
                    variants["random_positive"] = rng.choice(allpos, n_flag, replace=False)
                    variants["unprotected_positive"] = rng.choice(up, n_flag, replace=False)
                for scen, idx in variants.items():
                    res = fit_eval(tr.drop(index=idx).reset_index(drop=True), {"dep": de, "ext": test}, k)
                    for ev in ["dep", "ext"]:
                        b, x = base[ev], res[ev]
                        rows.append(dict(seed=s, clf=k, phi=phi, scenario=scen, eval=ev, removed=len(idx),
                                         d_acc=x["acc"] - b["acc"], d_eo=x["eo"] - b["eo"], d_dp=x["dp"] - b["dp"],
                                         d_prot_recall=x["prot_recall"] - b["prot_recall"],
                                         d_ref_recall=x["ref_recall"] - b["ref_recall"],
                                         base_acc=b["acc"], base_prot_recall=b["prot_recall"], base_eo=b["eo"]))
        pd.DataFrame(rows).to_csv(OUT / f"deletion_replay{tag}.csv", index=False)
        print(f"deletion seed {s} done in {time.time()-t0:.0f}s", flush=True)


# ────────────────────────── Bank Marketing case study ─────────────────────────
def run_bank(n_seeds=10, start=None, tag=""):
    """`duration` (last-call duration) is documented by the dataset providers as known only
    after the call, i.e. after the outcome, and to be discarded for realistic prediction.
    Records are ordered chronologically (May 2008 - Nov 2010), which permits a temporal split."""
    import audit_core as A
    b = pd.read_csv(DATA / "bank_additional_full.csv", sep=";")
    T, P, L = "y", "yes", "duration"
    rows = []

    def one(tr, va, de, split_kind, s):
        med = float(tr[L].median())
        for k in A.CLASSIFIERS:
            # leakage-controlled model (duration removed)
            trc, vac, dec = (x.drop(columns=[L]) for x in (tr, va, de))
            Xt, yt, Xv, yv = A.make_xy(trc, vac, T, P); _, _, Xd, yd = A.make_xy(trc, dec, T, P)
            m = A.build_classifier(k); m.fit(Xt, yt)
            lc_v = acc_f1(yv, m.predict(Xv)); lc_d = acc_f1(yd, m.predict(Xd))
            # model that retains duration; at deployment duration is not yet known -> training median
            Xt, yt, Xv, yv = A.make_xy(tr, va, T, P); _, _, Xd, yd = A.make_xy(tr, de, T, P)
            m = A.build_classifier(k); m.fit(Xt, yt)
            v = acc_f1(yv, m.predict(Xv))
            Xm = Xd.copy(); Xm[L] = med; d_ = acc_f1(yd, m.predict(Xm))
            d_avail = acc_f1(yd, m.predict(Xd))   # hypothetical: duration available at deployment
            rows.append(dict(split=split_kind, seed=s, clf=k, feature="duration (natural)",
                             val_acc=v[0], val_f1=v[1], lc_val_acc=lc_v[0], lc_val_f1=lc_v[1],
                             dep_acc=d_[0], dep_f1=d_[1], dep_acc_if_available=d_avail[0],
                             clean_dep_acc=lc_d[0], clean_dep_f1=lc_d[1], n_val=len(va), n_dep=len(de)))
            if split_kind == "temporal":   # synthetic trap retained under true temporal separation
                for r in INTENSITIES:
                    trt = A.add_leakage(trc, T, P, r, "engagement_score", s)
                    vat = A.add_leakage(vac, T, P, r, "engagement_score", s + 100)
                    Xt, yt, Xv, yv = A.make_xy(trt, vat, T, P); _, _, Xd, yd = A.make_xy(trt, dec, T, P)
                    m = A.build_classifier(k); m.fit(Xt, yt)
                    v = acc_f1(yv, m.predict(Xv)); d_ = acc_f1(yd, m.predict(Xd))
                    rows.append(dict(split=split_kind, seed=s, clf=k, feature=f"synthetic trap r={r}",
                                     val_acc=v[0], val_f1=v[1], lc_val_acc=lc_v[0], lc_val_f1=lc_v[1],
                                     dep_acc=d_[0], dep_f1=d_[1], clean_dep_acc=lc_d[0], clean_dep_f1=lc_d[1],
                                     n_val=len(va), n_dep=len(de)))

    n = len(b); i1, i2 = int(0.6 * n), int(0.8 * n)
    if start is None or start < 0:
        one(b.iloc[:i1].reset_index(drop=True), b.iloc[i1:i2].reset_index(drop=True),
            b.iloc[i2:].reset_index(drop=True), "temporal", 0)
        pd.DataFrame(rows).to_csv(OUT / "bank_case_temporal.csv", index=False); print("bank temporal done", flush=True)
        rows.clear()
        if start is not None: return
    for s in range(max(start or 0, 0), n_seeds):
        tr, va, de = A.split(b, T, s); one(tr, va, de, "random", s)
        pd.DataFrame(rows).to_csv(OUT / f"bank_case_random{tag}.csv", index=False); print(f"bank seed {s} done", flush=True)
    # statistical screen values for the natural leakage feature
    from sklearn.feature_selection import mutual_info_classif
    y = (b[T] == P).astype(int).to_numpy(); x = b[L].to_numpy(float)
    json.dump(dict(pearson=float(np.corrcoef(x, y)[0, 1]),
                   mi_nats=float(mutual_info_classif(x.reshape(-1, 1), y, random_state=42)[0]),
                   n=int(n), positive_rate=float(y.mean())), open(OUT / "bank_screen.json", "w"), indent=2)


if __name__ == "__main__":
    part = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else None
    fn = {"german": run_german, "adult": run_adult, "deletion": run_deletion, "bank": run_bank}[part]
    if len(sys.argv) > 3:   # resumable chunks: <part> <end_seed> <start_seed>
        st = int(sys.argv[3]); fn(n, st, f"_{st}")
    else:
        fn(n) if n else fn()
