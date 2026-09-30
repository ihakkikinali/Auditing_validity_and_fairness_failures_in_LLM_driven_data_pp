"""German Credit pipeline transcribed from llm_audit_STAGE3A_german_credit_proxy_leakage_FIXED.ipynb."""
import numpy as np, pandas as pd, warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")
TARGET = "class"   # set to "Risk" when faithful=False

def load_german(path, faithful=True):
    """faithful=True reproduces the original notebook loader exactly: because the Kaggle CSV
    has an 'Unnamed: 0' index column, the loader's header heuristic replaced all headers with
    generic names feature_0..feature_9 (feature_0 = former row index) and renamed Risk -> class.
    faithful=False gives the semantically named schema (index dropped, target Risk)."""
    d = pd.read_csv(path)
    if faithful:
        d.columns = [f"feature_{i}" for i in range(d.shape[1] - 1)] + ["class"]
        d["class"] = d["class"].astype(str).str.strip().str.lower().map({"good": 0, "bad": 1}).astype(int)
        for c in d.select_dtypes(include="object").columns:
            d[c] = d[c].astype(str).str.strip()
        return d.reset_index(drop=True)
    d.columns = [str(c).strip().replace(" ", "_").replace("-", "_").replace(".", "_") for c in d.columns]
    d = d.drop(columns=[c for c in d.columns if c.lower() in ["unnamed:_0", "unnamed_0", "index", "id"]])
    d[TARGET] = d[TARGET].astype(str).str.strip().str.lower().map({"good": 0, "bad": 1}).astype(int)
    for c in d.select_dtypes(include="object").columns:
        d[c] = d[c].astype(str).str.strip()
    return d.reset_index(drop=True)

def split(d, seed):
    tr, tmp = train_test_split(d, test_size=0.40, random_state=seed, stratify=d[TARGET])
    va, de = train_test_split(tmp, test_size=0.50, random_state=seed, stratify=tmp[TARGET])
    return tr.reset_index(drop=True), va.reset_index(drop=True), de.reset_index(drop=True)

def proxy(y, r, seed):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(float); ys = (y - y.mean()) / (y.std() + 1e-12)
    n = rng.normal(0, 1, len(y)); n -= (n @ ys) / (ys @ ys + 1e-12) * ys
    n = (n - n.mean()) / (n.std() + 1e-12)
    x = r * ys + np.sqrt(max(0, 1 - r**2)) * n
    return (x - x.mean()) / (x.std() + 1e-12)

def add_proxy(df, r, col, seed):
    df = df.reset_index(drop=True).copy(); df[col] = proxy(df[TARGET].values, r, seed); return df

def encode(train_df, test_df):
    tr = train_df.copy().reset_index(drop=True); te = test_df.copy().reset_index(drop=True)
    ytr = tr[TARGET].astype(int).values; yte = te[TARGET].astype(int).values
    Xtr = tr.drop(columns=[TARGET]); Xte = te.drop(columns=[TARGET]).reindex(columns=Xtr.columns, fill_value=np.nan)
    for c in Xtr.columns:
        if Xtr[c].dtype == "object" or str(Xtr[c].dtype).startswith(("category", "str")):
            a = Xtr[c].astype(str).fillna("__MISSING__"); b = Xte[c].astype(str).fillna("__MISSING__")
            cats = {v: i for i, v in enumerate(sorted(a.unique()))}
            Xtr[c] = a.map(cats).astype(float); Xte[c] = b.map(cats).fillna(-1).astype(float)
        else:
            Xtr[c] = pd.to_numeric(Xtr[c], errors="coerce"); Xte[c] = pd.to_numeric(Xte[c], errors="coerce")
            med = Xtr[c].median(); med = 0 if pd.isna(med) else med
            Xtr[c] = Xtr[c].fillna(med); Xte[c] = Xte[c].fillna(med)   # absent trap at deployment -> training median
    return Xtr, ytr, Xte, yte

def clf(kind, seed=42):
    return {"logistic": lambda: LogisticRegression(max_iter=2000, random_state=seed, n_jobs=-1),
            "rf": lambda: RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
            "xgboost": lambda: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
                                             colsample_bytree=0.9, eval_metric="logloss", random_state=seed, n_jobs=-1),
            "lightgbm": lambda: LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=seed, verbose=-1)}[kind]()

CLASSIFIERS = ["logistic", "rf", "xgboost", "lightgbm"]

def fit_eval(tr, evs, kind, seed=42):
    out = {}; m = None
    for name, ev in evs.items():
        Xt, yt, Xe, ye = encode(tr, ev)
        if m is None: m = clf(kind, seed); m.fit(Xt, yt)
        p = m.predict(Xe); out[name] = (accuracy_score(ye, p), f1_score(ye, p, average="macro"))
    return out
