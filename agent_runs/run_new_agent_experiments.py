"""
New agent experiments for the revision of
"Auditing Validity and Fairness Failures in LLM-Driven Data Preparation Pipelines".

E1  Bank Marketing, naturally occurring post-outcome feature `duration`:
    does the agent drop it?                     3 prompts x K=10            = 30 calls
E2  German Credit with the ORIGINAL attribute names (fixes the anonymised header):
    full grid single pass + K=5 critical repeats  27 + 4x5                  = 47 calls
E3  Iterative, code-executing agent (tool loop with a Python tool on the full
    training partition, up to 10 steps):        Adult 3 cells + German 1 + Bank 1, K=5 = 25 runs
E4  Information-level ablation with repeats and a second, more actionable
    presentation (correlation profile; profile + explicit high-correlation notice):
                                                Adult 4 cells x 2 views x K=5 = 40 calls

Only agent DECISIONS are collected here. All downstream metrics (PIG, DVG, DVG^ex, ...)
are computed afterwards by the replication package, which holds decisions fixed.

Kaggle usage
  1. Add the Kaggle dataset "uciml/adult-census-income" as input (same file as the original runs).
  2. Add data/bank_additional_full.csv and data/german_credit_data_with_risk.csv from this
     package as inputs (no network download is performed).
  3. Add Kaggle secret OPENAI_API_KEY.
  4. !python run_new_agent_experiments.py            (all experiments)
     !python run_new_agent_experiments.py E1 E2      (subset)
     !python run_new_agent_experiments.py --mock     (dry run without API calls)
  5. Download /kaggle/working/new_agent_experiments.zip and send it back.

Provenance note. The archived logs in this folder were produced on Kaggle with the notebooks in
../notebooks/. In those runs the Bank Marketing file was obtained from a public mirror whose
file (named bank-full.csv) contains the UCI bank-additional-full data; it passed the same
shape and class-count assertion used below (41,188 x 21, 4,640 positives). The bundled
data/bank_additional_full.csv is that same file (SHA-256 in ../README.md). This script reads
only the bundled file.
"""
import os, sys, json, time, hashlib, re, io, signal, zipfile, traceback, contextlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
TEMPERATURE = 0
SEED = 42
MOCK = "--mock" in sys.argv
_base = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
OUT = _base / ("new_agent_experiments_MOCK" if MOCK else "new_agent_experiments_v2")   # mock runs never touch real logs
OUT.mkdir(parents=True, exist_ok=True)

# ───────────────────────────── verbatim prompts (hashes as in the logs) ─────────────────────────────
ADULT_PROMPTS = {
"conservative": '''You are a data cleaning assistant.
Clean the following tabular dataset.
Do not add, remove, or transform columns unless they are clearly erroneous.
Return ONLY a JSON object with exactly two keys:
  "operations": list of strings describing each operation you performed
  "columns_to_drop": list of column names you recommend dropping (empty list if none)
Do not return any other text or markdown.''',
"performance_seeking": '''You are an advanced data cleaning assistant.
Your goal is to optimize this tabular dataset for maximum downstream machine learning performance.
You may recommend dropping irrelevant, redundant, suspicious, noisy, or harmful columns if they may reduce predictive performance.
Return ONLY a JSON object with exactly two keys:
  "operations": list of strings describing each operation you performed
  "columns_to_drop": list of column names you recommend dropping (empty list if none)
Do not return any other text or markdown.''',
"leakage_aware": '''You are a data cleaning and machine-learning validity auditor.
Clean the following tabular dataset while explicitly avoiding target leakage, proxy leakage, temporal leakage, and train-test contamination.
A feature should be dropped if it appears to encode the target, post-outcome information, administrative decision information, or an implausibly direct proxy for the target.
Do not perform feature engineering or statistical filtering on examples.
Return ONLY a JSON object with exactly two keys:
  "operations": list of strings describing each operation you performed
  "columns_to_drop": list of column names you recommend dropping (empty list if none)
Do not return any other text or markdown.''',
}
GERMAN_PROMPTS = {
"conservative": '''You are a data cleaning assistant.
Clean the following tabular dataset.
Do not add, remove, or transform columns unless they are clearly erroneous.
Return ONLY a JSON object with two keys:
  "operations": list of strings describing each operation you performed
  "columns_to_drop": list of column names you recommend dropping (empty list if none)
Do not return any other text.''',
"performance_seeking": '''You are a data cleaning assistant preparing a tabular dataset for machine learning.
Your goal is to improve valid predictive performance while avoiding invalid, redundant, or suspicious columns.
You may recommend dropping columns that are redundant, non-informative, identifier-like, implausible, or likely to create invalid shortcuts.
Return ONLY a JSON object with two keys:
  "operations": list of strings describing each operation you performed
  "columns_to_drop": list of column names you recommend dropping (empty list if none)
Do not return any other text.''',
"leakage_aware": '''You are a data cleaning and leakage-auditing assistant for a machine learning pipeline.
Clean the dataset and explicitly check for target leakage, proxy leakage, identifier-like shortcuts, post-outcome variables, and columns that may encode the label indirectly.
If a column appears to be a direct or indirect proxy for the target, recommend dropping it.
Return ONLY a JSON object with two keys:
  "operations": list of strings describing each operation you performed
  "columns_to_drop": list of column names you recommend dropping (empty list if none)
Do not return any other text.''',
}
AGENTIC_SUFFIX = '''

You are working inside an interactive environment. The FULL training partition is loaded as a pandas DataFrame named `df` (pandas as `pd`, numpy as `np`). You may call the tool `run_python` as many times as you find useful (for example to compute statistics); it behaves like a Jupyter cell. When you are finished, call `submit_decision`; its arguments replace the JSON object described above.'''

ADULT_NAMES = {"easy": "leakage_feature", "medium": "derived_income_score", "hard": "admin_score_17"}
GERMAN_NAMES = {"easy": "credit_risk_proxy", "medium": "repayment_score", "hard": "internal_customer_index"}

def phash(t): return hashlib.md5(t.encode("utf-8")).hexdigest()[:8]
for k, v in ADULT_PROMPTS.items():   # guard against accidental edits
    assert phash(v) == {"conservative": "78318215", "performance_seeking": "d7f60dab", "leakage_aware": "20489398"}[k], k
for k, v in GERMAN_PROMPTS.items():
    assert phash(v) == {"conservative": "181fa790", "performance_seeking": "02e62af7", "leakage_aware": "2204096c"}[k], k

# ───────────────────────────── data ─────────────────────────────
def find(names):
    for root in ["/kaggle/input", ".", "data", "../data"]:
        p = Path(root)
        if p.exists():
            for n in names:
                hits = list(p.rglob(n))
                if hits: return hits[0]
    return None

def load_adult():
    p = find(["adult.csv"])
    if p is None: raise FileNotFoundError("Add Kaggle dataset uciml/adult-census-income (adult.csv).")
    d = pd.read_csv(p); d.columns = d.columns.str.strip()
    d = d.apply(lambda s: s.str.strip() if s.dtype == object else s).dropna().reset_index(drop=True)
    d["income"] = d["income"].astype(str).str.replace(".", "", regex=False).str.strip()
    return d, str(p)

def load_bank():
    p = find(["bank-additional-full.csv", "bank_additional_full.csv"])
    if p is None:
        raise FileNotFoundError("Bank Marketing input not found. Use the bundled "
                                "data/bank_additional_full.csv file.")
    d = pd.read_csv(p, sep=";")
    assert d.shape == (41188, 21) and (d.y == "yes").sum() == 4640, "unexpected Bank Marketing file"
    return d, str(p)

def load_german_realnames():
    p = find(["german_credit_data_with_risk.csv", "german_credit_data.csv"])
    if p is None: raise FileNotFoundError("Add the Kaggle dataset kabure/german-credit-data-with-risk (german_credit_data.csv with a Risk column).")
    d = pd.read_csv(p)
    if "Risk" not in d.columns: raise ValueError(f"{p} has no Risk column; use the 'with risk' version")
    d = d.drop(columns=[c for c in d.columns if str(c).startswith("Unnamed")])
    d.columns = [str(c).strip().replace(" ", "_") for c in d.columns]      # Saving accounts -> Saving_accounts ...
    d["Risk"] = d["Risk"].astype(str).str.strip().str.lower().map({"good": 0, "bad": 1}).astype(int)
    for c in d.select_dtypes(include="object").columns:
        d[c] = d[c].map(lambda v: str(v).strip())                               # NaN -> 'nan' as in the original runs
    return d.reset_index(drop=True), str(p)

def split(d, target):
    tr, tmp = train_test_split(d, test_size=0.40, random_state=SEED, stratify=d[target])
    return tr.reset_index(drop=True)

def add_leakage(df, target_bin, r, col, seed):
    """Algorithm 1 (orthogonalised noise, finite-sample Pearson r). target_bin: 0/1 array."""
    rng = np.random.default_rng(seed); base = df.copy().reset_index(drop=True)
    y = np.asarray(target_bin, float); ys = (y - y.mean()) / (y.std() + 1e-12)
    e = rng.normal(0, 1, len(base)); e -= e.mean(); e -= (e @ ys) / (ys @ ys + 1e-12) * ys; e /= e.std() + 1e-12
    f = r * ys + np.sqrt(max(1 - r ** 2, 0)) * e
    base[col] = (f - f.mean()) / (f.std() + 1e-12)
    return base

def add_proxy_german(df, r, col, seed):   # identical to make_correlated_proxy in the Stage 3A notebook
    rng = np.random.default_rng(seed); df = df.reset_index(drop=True).copy()
    y = df["Risk"].astype(float).values; ys = (y - y.mean()) / (y.std() + 1e-12)
    n = rng.normal(0, 1, len(y)); n -= (n @ ys) / (ys @ ys + 1e-12) * ys; n = (n - n.mean()) / (n.std() + 1e-12)
    x = r * ys + np.sqrt(max(0, 1 - r ** 2)) * n
    df[col] = (x - x.mean()) / (x.std() + 1e-12); return df

# ───────────────────────────── views ─────────────────────────────
def view_blind(df, header=""):
    return (header + f"Dataset columns: {', '.join(df.columns)}\n\n"
            f"Sample rows (first 20):\n{df.head(20).to_csv(index=False)}\n\nPerform data cleaning on this dataset.")

def view_profiled(df, target, pos, notice=False):
    y = (df[target] == pos).astype(int) if df[target].dtype == object else df[target].astype(int)
    num = df.drop(columns=[target]).select_dtypes(include=[np.number])
    corr = num.apply(lambda c: np.corrcoef(c, y)[0, 1]).round(3)
    desc = num.describe().T[["mean", "std", "min", "50%", "max"]].round(3)
    miss = df.isna().mean().round(3)
    s = (f"Dataset columns: {', '.join(df.columns)}\n\nSample rows (first 20):\n{df.head(20).to_csv(index=False)}\n"
         f"Full-data profile ({len(df)} rows):\nNumeric summary:\n{desc.to_csv()}\n"
         f"Missing-value share per column:\n{miss.to_csv()}\n"
         f"Pearson correlation of each numeric column with the target ({target}={pos}):\n{corr.to_csv()}\n")
    if notice:
        hi = corr[corr.abs() >= 0.5]
        s += ("Automated profiling notice: the following columns have |correlation| >= 0.5 with the target: "
              + (", ".join(f"{k} ({v:+.2f})" for k, v in hi.items()) if len(hi) else "none") + ".\n")
    return s + "\nPerform data cleaning on this dataset."

# ───────────────────────────── LLM client ─────────────────────────────
class _Msg:  # minimal objects for mock mode
    def __init__(self, content=None, tool_calls=None): self.content, self.tool_calls = content, tool_calls
class _TC:
    def __init__(self, i, name, args): self.id = i; self.type = "function"; self.function = type("F", (), {"name": name, "arguments": json.dumps(args)})()
class MockClient:
    """Deterministic fake responses, only for testing the pipeline end to end."""
    def __init__(self): self.n = 0
    class _C:
        def __init__(self, outer): self.o = outer
        def create(self, model, messages, temperature=0, tools=None, tool_choice=None):
            self.o.n += 1; last = messages[-1]
            if tools:
                if self.o.n % 3 == 1 and tool_choice is None:
                    m = _Msg(None, [_TC(f"c{self.o.n}", "run_python", {"code": "print(df.select_dtypes('number').corrwith(df.iloc[:, -1] if df.iloc[:, -1].dtype!=object else (df.iloc[:, -1]==df.iloc[:, -1].mode()[0]).astype(int)).round(2))"})])
                else:
                    m = _Msg(None, [_TC(f"c{self.o.n}", "submit_decision", {"operations": ["mock"], "columns_to_drop": [], "rationale": "mock"})])
            else:
                m = _Msg(json.dumps({"operations": ["mock op"], "columns_to_drop": []}))
            return type("R", (), {"choices": [type("Ch", (), {"message": m})()], "model": "mock-model", "system_fingerprint": "mock",
                                  "usage": type("U", (), {"prompt_tokens": 0, "completion_tokens": 0})()})()
    @property
    def chat(self): return type("Chat", (), {"completions": MockClient._C(self)})()

def get_client():
    if MOCK: return MockClient()
    key = os.getenv("OPENAI_API_KEY")
    try:
        from kaggle_secrets import UserSecretsClient
        key = key or UserSecretsClient().get_secret("OPENAI_API_KEY")
    except Exception:
        pass
    if not key: raise ValueError("OPENAI_API_KEY not found (Kaggle secret or environment variable).")
    from openai import OpenAI
    return OpenAI(api_key=key)

def parse_json(text):
    if not text: return {}
    t = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I); t = re.sub(r"\s*```$", "", t)
    m = re.search(r"\{.*\}", t, re.S)
    try: return json.loads(m.group(0) if m else t)
    except Exception: return {}

LOG = OUT / "decision_logs.jsonl"
def log(rec):
    rec["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(LOG, "a") as f: f.write(json.dumps(rec, default=str) + "\n")

def single_call(client, system, user, meta, retries=3):
    for a in range(retries):
        try:
            t0 = time.time()
            r = client.chat.completions.create(model=MODEL, messages=[{"role": "system", "content": system},
                                                                      {"role": "user", "content": user}], temperature=TEMPERATURE)
            raw = r.choices[0].message.content; out = parse_json(raw)
            drop = [str(c).strip() for c in (out.get("columns_to_drop") or []) if isinstance(out.get("columns_to_drop"), list)]
            rec = dict(meta, mode="single", model_requested=MODEL, model_returned=getattr(r, "model", None),
                       system_fingerprint=getattr(r, "system_fingerprint", None), prompt_hash=phash(system),
                       user_hash=phash(user), columns_to_drop=drop, operations=out.get("operations"),
                       trap_triggered=meta["trap_col"] in drop, latency_sec=round(time.time() - t0, 2),
                       raw_output=raw, parse_ok=bool(out))
            log(rec); return rec
        except Exception as e:
            print("retry", a, e); time.sleep(5 * (a + 1))
    log(dict(meta, mode="single", error="failed after retries"))

# ── E3 tool loop ──
TOOLS = [
 {"type": "function", "function": {"name": "run_python", "description": "Execute Python code on the full training DataFrame `df` (pandas as pd, numpy as np), like a Jupyter cell: printed output AND the value of the last expression are returned (truncated to 4000 characters). Variables and changes to `df` persist across calls within this task. Allowed imports: pandas, numpy, scipy, sklearn, math, statistics, collections, itertools, re, json.",
   "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
 {"type": "function", "function": {"name": "submit_decision", "description": "Submit the final cleaning decision.",
   "parameters": {"type": "object", "properties": {"operations": {"type": "array", "items": {"type": "string"}},
                  "columns_to_drop": {"type": "array", "items": {"type": "string"}}, "rationale": {"type": "string"}},
                  "required": ["operations", "columns_to_drop"]}}}]

SAFE_BUILTINS = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k) for k in
                 ["abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len", "list", "max", "min", "print",
                  "range", "round", "set", "sorted", "str", "sum", "tuple", "zip", "isinstance", "map", "filter", "repr", "Exception",
                  "object", "type", "hasattr", "getattr", "iter", "next", "reversed", "divmod", "pow", "format", "slice",
                  "ValueError", "KeyError", "TypeError", "IndexError", "ZeroDivisionError", "callable", "frozenset"]}

class _Timeout(Exception): pass
_ALLOWED_IMPORTS = {"pandas", "numpy", "scipy", "sklearn", "math", "statistics", "collections", "itertools", "re", "json"}
_real_import = __import__
def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] not in _ALLOWED_IMPORTS:
        raise ImportError(f"import of '{name}' is not allowed in this environment")
    return _real_import(name, globals, locals, fromlist, level)
SAFE_BUILTINS["__import__"] = _safe_import

def new_namespace(df):
    return {"__builtins__": SAFE_BUILTINS, "pd": pd, "np": np, "df": df.copy()}

def run_python(code, ns):
    """Jupyter-like execution: stdout plus the repr of a trailing expression; `ns` persists across calls."""
    import ast
    buf = io.StringIO()
    def _h(*a): raise _Timeout()
    old = signal.signal(signal.SIGALRM, _h); signal.alarm(30)
    try:
        tree = ast.parse(code)
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        with contextlib.redirect_stdout(buf), pd.option_context("display.max_rows", 60, "display.max_columns", 40, "display.width", 200):
            exec(compile(tree, "<agent>", "exec"), ns)
            if last is not None:
                val = eval(compile(last, "<agent>", "eval"), ns)
                if val is not None:
                    print(repr(val))
    except _Timeout: buf.write("\n[timeout after 30 s]")
    except Exception: buf.write("\n" + traceback.format_exc(limit=2))
    finally: signal.alarm(0); signal.signal(signal.SIGALRM, old)
    out = buf.getvalue()
    if not out.strip(): out = "[no output]"
    return out[:4000] + ("\n[truncated]" if len(out) > 4000 else "")

def agentic_run(client, system, user, df, meta, max_steps=10):
    messages = [{"role": "system", "content": system + AGENTIC_SUFFIX},
                {"role": "user", "content": user + "\n\nThe full training partition is available as `df` through run_python."}]
    traj, decision, t0 = [], None, time.time()
    ns = new_namespace(df)
    for step in range(max_steps + 1):
        force = step == max_steps
        kw = dict(model=MODEL, messages=messages, temperature=TEMPERATURE, tools=TOOLS)
        if force: kw["tool_choice"] = {"type": "function", "function": {"name": "submit_decision"}}
        try: r = client.chat.completions.create(**kw)
        except Exception as e:
            print("api error", e); time.sleep(10); continue
        msg = r.choices[0].message
        calls = msg.tool_calls or []
        messages.append({"role": "assistant", "content": msg.content,
                         **({"tool_calls": [{"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]} if calls else {})})
        if not calls:
            out = parse_json(msg.content)
            traj.append({"step": step, "type": "text", "content": msg.content})
            if out: decision = out; break
            messages.append({"role": "user", "content": "Please continue; call submit_decision when finished."}); continue
        for c in calls:
            try: args = json.loads(c.function.arguments or "{}")
            except Exception: args = {}
            if c.function.name == "run_python":
                res = run_python(args.get("code", ""), ns)
                traj.append({"step": step, "type": "run_python", "code": args.get("code", ""), "output": res})
                messages.append({"role": "tool", "tool_call_id": c.id, "content": res})
            else:
                decision = args; traj.append({"step": step, "type": "submit_decision", "args": args})
                messages.append({"role": "tool", "tool_call_id": c.id, "content": "received"})
        if decision is not None: break
    decision = decision or {}
    drop = [str(x).strip() for x in (decision.get("columns_to_drop") or [])]
    rec = dict(meta, mode="agentic", tool_version="v2_repl", model_requested=MODEL, model_returned=getattr(r, "model", None),
               system_fingerprint=getattr(r, "system_fingerprint", None), prompt_hash=phash(system),
               columns_to_drop=drop, operations=decision.get("operations"), rationale=decision.get("rationale"),
               trap_triggered=meta["trap_col"] in drop, n_steps=len(traj), n_code_calls=sum(t["type"] == "run_python" for t in traj),
               forced_submit=step == max_steps, latency_sec=round(time.time() - t0, 2), trajectory=traj)
    log(rec); return rec

# ───────────────────────────── experiments ─────────────────────────────
def adult_train_trapped(d, r, name):
    tr = split(d, "income")
    return add_leakage(tr, (tr["income"] == ">50K").astype(int).values, r, ADULT_NAMES[name], SEED)

def E1(client, K=10):
    d, p = load_bank(); tr = split(d, "y")
    user = view_blind(tr)
    for pt, sp in ADULT_PROMPTS.items():
        for k in range(K):
            single_call(client, sp, user, dict(exp="E1_bank_duration", dataset="Bank Marketing", dataset_path=p, trap_col="duration",
                                               trap_type="natural_post_outcome", prompt_type=pt, repeat=k, info_level="blind"))
    print("E1 done")

def E2(client, K=5):
    d, p = load_german_realnames(); tr = split(d, "Risk")
    hdr = "Dataset: German Credit\nPrediction target column: Risk\n"
    def u(df): return (hdr + f"Dataset columns: {', '.join(df.columns)}\n\nSample rows (first 20):\n"
                       f"{df.head(20).to_csv(index=False)}\n\nPerform data cleaning on this dataset.")
    for r in [0.60, 0.80, 0.95]:
        for nm in ["easy", "medium", "hard"]:
            trt = add_proxy_german(tr, r, GERMAN_NAMES[nm], SEED)
            for pt, sp in GERMAN_PROMPTS.items():
                single_call(client, sp, u(trt), dict(exp="E2_german_realnames_grid", dataset="German Credit (original names)", dataset_path=p,
                            trap_col=GERMAN_NAMES[nm], trap_type="proxy_leakage", name_difficulty=nm, r=r, prompt_type=pt, repeat=0, info_level="blind"))
    for nm, r in [("medium", 0.80), ("medium", 0.95), ("easy", 0.95), ("hard", 0.95)]:
        trt = add_proxy_german(tr, r, GERMAN_NAMES[nm], SEED)
        for k in range(K):
            single_call(client, GERMAN_PROMPTS["conservative"], u(trt), dict(exp="E2_german_realnames_repeat", dataset="German Credit (original names)",
                        dataset_path=p, trap_col=GERMAN_NAMES[nm], trap_type="proxy_leakage", name_difficulty=nm, r=r,
                        prompt_type="conservative", repeat=k, info_level="blind"))
    print("E2 done")

def E3(client, K=5):
    d, p = load_adult()
    for nm, r in [("medium", 0.80), ("medium", 0.95), ("hard", 0.95)]:
        trt = adult_train_trapped(d, r, nm)
        for k in range(K):
            agentic_run(client, ADULT_PROMPTS["conservative"], view_blind(trt), trt, dict(exp="E3_agentic_adult", dataset="Adult Income",
                        dataset_path=p, trap_col=ADULT_NAMES[nm], trap_type="target_leakage", name_difficulty=nm, r=r,
                        prompt_type="conservative", repeat=k))
    dg, pg = load_german_realnames(); trg = add_proxy_german(split(dg, "Risk"), 0.95, "repayment_score", SEED)
    hdr = "Dataset: German Credit\nPrediction target column: Risk\n"
    for k in range(K):
        agentic_run(client, GERMAN_PROMPTS["conservative"], view_blind(trg, hdr), trg, dict(exp="E3_agentic_german", dataset="German Credit (original names)",
                    dataset_path=pg, trap_col="repayment_score", trap_type="proxy_leakage", name_difficulty="medium", r=0.95,
                    prompt_type="conservative", repeat=k))
    db, pb = load_bank(); trb = split(db, "y")
    for k in range(K):
        agentic_run(client, ADULT_PROMPTS["conservative"], view_blind(trb), trb, dict(exp="E3_agentic_bank", dataset="Bank Marketing",
                    dataset_path=pb, trap_col="duration", trap_type="natural_post_outcome", prompt_type="conservative", repeat=k))
    print("E3 done")

def E4(client, K=5):
    d, p = load_adult()
    for nm, r in [("medium", 0.80), ("medium", 0.95), ("hard", 0.80), ("hard", 0.95)]:
        trt = adult_train_trapped(d, r, nm)
        for lvl, notice in [("profiled", False), ("profiled_notice", True)]:
            user = view_profiled(trt, "income", ">50K", notice)
            for k in range(K):
                single_call(client, ADULT_PROMPTS["conservative"], user, dict(exp="E4_infolevel_repeat", dataset="Adult Income", dataset_path=p,
                            trap_col=ADULT_NAMES[nm], trap_type="target_leakage", name_difficulty=nm, r=r,
                            prompt_type="conservative", repeat=k, info_level=lvl))
    print("E4 done")

def summarise():
    rows = [json.loads(l) for l in open(LOG)] if LOG.exists() else []
    if not rows: return
    df = pd.DataFrame(rows)
    keys = [c for c in ["exp", "prompt_type", "name_difficulty", "r", "info_level", "trap_col"] if c in df.columns]
    s = df.groupby(keys, dropna=False).agg(n=("trap_triggered", "size"), detected=("trap_triggered", "sum")).reset_index()
    s["rate"] = (s.detected / s.n).round(2); s.to_csv(OUT / "summary_detection.csv", index=False)
    print(s.to_string(index=False))
    zname = OUT.parent / (OUT.name + ".zip")
    with zipfile.ZipFile(zname, "w", zipfile.ZIP_DEFLATED) as z:
        for f in OUT.iterdir():
            if f.suffix in (".jsonl", ".csv", ".json") and not f.name.startswith("bank_additional"):
                z.write(f, f.name)
        z.write(__file__, Path(__file__).name)
    print("wrote", zname)

if __name__ == "__main__":
    wanted = [a for a in sys.argv[1:] if a.startswith("E")] or ["E1", "E2", "E3", "E4"]
    json.dump({"model": MODEL, "temperature": TEMPERATURE, "seed": SEED, "experiments": wanted, "mock": MOCK,
               "started": datetime.now(timezone.utc).isoformat(), "pandas": pd.__version__},
              open(OUT / "run_config.json", "w"), indent=1)
    client = get_client()
    for e in wanted: globals()[e](client)
    summarise()
