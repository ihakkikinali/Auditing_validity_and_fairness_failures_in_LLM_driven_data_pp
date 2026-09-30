import json, re, glob, numpy as np, pandas as pd, warnings
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
warnings.filterwarnings("ignore")
R = Path(__file__).parent.parent / "results"; T = Path(__file__).parent.parent / "tables"; OUT = {}
# ── (a) CUR on Bank decisions: score(reference = only duration removed) − score(agent-cleaned), deployment-proxy
rep = pd.read_csv(R / "e1_replay.csv")
V = [json.loads(l) for l in open(Path(__file__).parent.parent / "agent_runs" / "decision_logs.jsonl")]
E1 = [x for x in V if x["exp"] == "E1_bank_duration"]
key = lambda x: "|".join(sorted(c for c in x["columns_to_drop"] if c != "y")) or "(none)"
m = rep.groupby(["split", "seed", "drop_set"])[["dep_f1", "dep_acc"]].mean()
rows = []
for pt in ["conservative", "performance_seeking", "leakage_aware"]:
    xs = [x for x in E1 if x["prompt_type"] == pt]
    per = {}
    for (split, seed) in sorted(set((a, b) for a, b, _ in m.index)):
        ref = m.loc[(split, seed, "duration")]
        vals = [ref.dep_f1 - m.loc[(split, seed, key(x))].dep_f1 for x in xs]
        per[f"{split}{seed if split=='random' else ''}"] = float(np.mean(vals))
    callwise = [np.mean([m.loc[(s, sd, "duration")].dep_f1 - m.loc[(s, sd, key(x))].dep_f1 for (s, sd) in sorted(set((a, b) for a, b, _ in m.index)) if s == "random"]) for x in xs]
    rows.append(dict(prompt=pt, cur_random_mean=np.mean([v for k, v in per.items() if k.startswith("random")]),
                     cur_random_min=min(v for k, v in per.items() if k.startswith("random")),
                     cur_random_max=max(v for k, v in per.items() if k.startswith("random")),
                     cur_temporal=per["temporal"], share_calls_cur_gt_001=float(np.mean([c > 0.01 for c in callwise]))))
OUT["CUR_bank"] = rows
# ── (b) screener specificity on native (non-trap) features
def screens(df, target, pos, exclude=()):
    y = (df[target] == pos).astype(int).to_numpy() if df[target].dtype == object else df[target].astype(int).to_numpy()
    res = []
    for c in df.columns:
        if c == target or c in exclude: continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            r = abs(np.corrcoef(s.astype(float), y)[0, 1]); mi = mutual_info_classif(s.to_numpy(float).reshape(-1, 1), y, random_state=42)[0]
        else:
            r = np.nan; codes = s.astype("category").cat.codes.to_numpy().reshape(-1, 1)
            mi = mutual_info_classif(codes, y, discrete_features=True, random_state=42)[0]
        res.append(dict(feature=c, numeric=pd.api.types.is_numeric_dtype(s), abs_r=r, mi=mi,
                        flag_r=bool(r >= 0.20) if not np.isnan(r) else False, flag_mi=bool(mi >= 0.02)))
    return pd.DataFrame(res)
cols = ['age','workclass','fnlwgt','education','education.num','marital.status','occupation','relationship','race','sex','capital.gain','capital.loss','hours.per.week','native.country','income']
a = pd.read_csv(Path(__file__).parent.parent / "data" / "adult-all.csv", header=None, names=cols).iloc[:32561]
for c in a.select_dtypes(include="object").columns: a[c] = a[c].astype(str).str.strip()
g = pd.read_csv(Path(__file__).parent.parent / "data" / "german_credit_data_with_risk.csv").drop(columns=["Unnamed: 0"])
g.columns = [c.replace(" ", "_") for c in g.columns]; g["Saving_accounts"] = g["Saving_accounts"].fillna("nan"); g["Checking_account"] = g["Checking_account"].fillna("nan")
b = pd.read_csv(Path(__file__).parent.parent / "data" / "bank_additional_full.csv", sep=";")
scr = {"Adult Income": screens(a, "income", ">50K"), "German Credit": screens(g, "Risk", "bad"),
       "Bank Marketing": screens(b, "y", "yes", exclude=("duration",))}
OUT["screens"] = {}
with open(T / "tab_s_screen_specificity.tex", "w") as f:
    f.write("\\begin{tabular}{@{}lrrrrl@{}}\n\\toprule\nDataset & Native features & Flagged by $|r|$ & Flagged by MI & Flagged by either & Flagged features \\\\\n\\midrule\n")
    for k, d in scr.items():
        fl = d[d.flag_r | d.flag_mi]
        OUT["screens"][k] = dict(n=len(d), r=int(d.flag_r.sum()), mi=int(d.flag_mi.sum()), either=len(fl), names=fl.feature.tolist())
        names = ", ".join(n.replace("_", "\\_") for n in fl.feature)
        f.write(f"{k} & {len(d)} & {int(d.flag_r.sum())} & {int(d.flag_mi.sum())} & {len(fl)} ({100*len(fl)/len(d):.0f}\\%) & \\parbox[t]{{0.34\\linewidth}}{{\\raggedright\\code{{{names}}}}} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
# ── (c) unparseable outputs in released logs
def parse(t):
    if not t: return None
    t = re.sub(r"^```(?:json)?\s*", "", str(t).strip(), flags=re.I); t = re.sub(r"\s*```$", "", t)
    mm = re.search(r"\{.*\}", t, re.S)
    try: return json.loads(mm.group(0) if mm else t)
    except Exception: return None
cnt = {}
for fpath in glob.glob(str(Path(__file__).parent.parent / "agent_runs" / "stage_logs" / "*.json")):
    try: L = json.load(open(fpath))
    except Exception: continue
    if not isinstance(L, list) or not L or "raw_output" not in L[0]: continue
    n = len(L); bad = sum(parse(x.get("raw_output")) is None for x in L)
    trunc = sum(len(str(x.get("raw_output") or "")) >= 1000 for x in L)
    cnt[Path(fpath).name] = dict(n=n, unparseable_from_stored=bad, stored_output_truncated=trunc)
newL = V + [json.loads(l) for l in open(Path(__file__).parent.parent / "agent_runs" / "decision_logs_E3v2.jsonl")]
cnt["agent_level_new"] = dict(n=len(newL), unparseable=sum((x.get("parse_ok") is False) for x in newL if x["mode"] == "single"))
OUT["parse"] = cnt
# ── (d) DP gap from deletion replay
D = pd.read_csv(R / "deletion_replay_all.csv")
dd = D[D["eval"] == "dep"].groupby(["scenario", "phi", "seed"])[["d_dp", "d_eo"]].mean().groupby(["scenario", "phi"])
OUT["dp"] = {f"{s}_{p}": dict(d_dp=float(v.d_dp.mean()), lo=float(v.d_dp.quantile(.025)), hi=float(v.d_dp.quantile(.975))) for (s, p), v in dd}
# ── (e) shift table with DVG and DVGex at all three r
A = pd.read_csv(R / "adult_replay_all.csv")
shifts = ["dep", "ext", "female_15", "female_50", "age45_20", "age45_60", "prev_12", "prev_40"]
lab = {"female_15": "Female share 15\\%", "female_50": "Female share 50\\%", "age45_20": "Age $\\geq45$: 20\\%", "age45_60": "Age $\\geq45$: 60\\%",
       "prev_12": "Prevalence 12\\%", "prev_40": "Prevalence 40\\%", "dep": "Unshifted dep.-proxy", "ext": "Official test release"}
c = A[A.cond == "clean"].set_index(["seed", "clf"])
with open(T / "tab_shift.tex", "w") as f:
    f.write("\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n & Clean model & \\multicolumn{2}{c}{$r=0.60$} & \\multicolumn{2}{c}{$r=0.80$} & \\multicolumn{2}{c}{$r=0.95$} \\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\nDeployment-proxy set & gap & \\DVG & \\DVGex & \\DVG & \\DVGex & \\DVG & \\DVGex \\\\\n\\midrule\n")
    for s in shifts:
        col = {"dep": "dep_acc", "ext": "ext_acc"}.get(s, f"shift_{s}")
        cg = c.val_acc - c[col]; cells = [f"{cg.mean():+.3f}"]
        for rr in [0.60, 0.80, 0.95]:
            x = A[(A.cond == "retained") & (A.r == rr)].set_index(["seed", "clf"])
            gg = x.val_acc - x[col]; ex = gg - cg.reindex(gg.index)
            cells += [f"{gg.mean():.3f}", f"{ex.mean():.3f}"]
        f.write(f"{lab[s]} & " + " & ".join(cells) + " \\\\\n")
        if s == "ext": f.write("\\addlinespace\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
json.dump(OUT, open(R / "extra_numbers.json", "w"), indent=1, default=float)
print(json.dumps(OUT, indent=1, default=float)[:4000])
