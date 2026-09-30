"""Core audit functions, transcribed from the original Stage 2 / 3A / 3B notebooks
so that revision analyses reuse exactly the same trap generator, preprocessing,
classifiers and metric definitions."""
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings("ignore")

CLASSIFIERS = ["logistic", "rf", "xgboost", "lightgbm"]

def build_classifier(kind, seed=42):
    if kind == "logistic":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=-1)
    if kind == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=150, random_state=seed, eval_metric="logloss", n_jobs=-1)
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=150, random_state=seed, verbose=-1, n_jobs=-1)
    raise ValueError(kind)

def make_xy(train_df, eval_df, target, pos):
    y_tr = (train_df[target] == pos).astype(int).to_numpy()
    y_ev = (eval_df[target] == pos).astype(int).to_numpy()
    X_tr = pd.get_dummies(train_df.drop(columns=[target]), dummy_na=True)
    X_ev = pd.get_dummies(eval_df.drop(columns=[target]), dummy_na=True)
    X_ev = X_ev.reindex(columns=X_tr.columns, fill_value=0)   # absent trap -> 0 (= training mean of standardised trap)
    return X_tr.astype(float), y_tr, X_ev.astype(float), y_ev

def add_leakage(df, target, pos, r, col, seed):
    """Orthogonalised-noise leakage feature with finite-sample Pearson r (Algorithm 2)."""
    rng = np.random.default_rng(seed)
    base = df.copy().reset_index(drop=True)
    y = (base[target] == pos).astype(float).to_numpy()
    ys = (y - y.mean()) / (y.std() + 1e-12)
    eps = rng.normal(0, 1, len(base)); eps -= eps.mean()
    eps -= (eps @ ys) / (ys @ ys + 1e-12) * ys
    eps /= eps.std() + 1e-12
    f = r * ys + np.sqrt(max(1 - r**2, 0)) * eps
    base[col] = (f - f.mean()) / (f.std() + 1e-12)
    return base

def split(df, target, seed):
    tr, tmp = train_test_split(df, test_size=0.40, random_state=seed, stratify=df[target])
    va, de = train_test_split(tmp, test_size=0.50, random_state=seed, stratify=tmp[target])
    return tr.reset_index(drop=True), va.reset_index(drop=True), de.reset_index(drop=True)

def fit_eval(tr, evs, target, pos, kind, seed=42):
    """Fit once on tr, return {name: (acc, macroF1, preds)} for each eval frame."""
    out = {}
    clf = None
    X_tr = None
    for name, ev in evs.items():
        Xt, yt, Xe, ye = make_xy(tr, ev, target, pos)
        if clf is None:
            clf = build_classifier(kind, seed); clf.fit(Xt, yt)
        p = clf.predict(Xe)
        out[name] = (accuracy_score(ye, p), f1_score(ye, p, average="macro"), p, ye)
    return out

def load_adult(path):
    d = pd.read_csv(path)
    d.columns = d.columns.str.strip()
    d = d.apply(lambda s: s.str.strip() if s.dtype == object else s)
    d = d.dropna().reset_index(drop=True)
    d["income"] = d["income"].astype(str).str.replace(".", "", regex=False).str.strip()
    return d
