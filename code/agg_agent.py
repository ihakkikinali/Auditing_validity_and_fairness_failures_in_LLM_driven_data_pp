import json, math, collections, pandas as pd, numpy as np
from pathlib import Path
T = Path(__file__).parent.parent / "tables"; R = Path(__file__).parent.parent / "results"
V = [json.loads(l) for l in open(Path(__file__).parent.parent / "agent_runs" / "decision_logs.jsonl")]
E3 = [json.loads(l) for l in open(Path(__file__).parent.parent / "agent_runs" / "decision_logs_E3v2.jsonl")]
def wil(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return max(0, c-h), min(1, c+h)
def kn(k, n): lo, hi = wil(k, n); return f"{k}/{n} {{\\scriptsize$[{lo:.2f},{hi:.2f}]$}}"
NUM = {}
# ── E1 detection + consequences
E1 = [x for x in V if x["exp"] == "E1_bank_duration"]
rep = pd.read_csv(R / "e1_replay.csv"); rep["dvg_f1"] = rep.val_f1 - rep.dep_f1
m = rep.groupby(["split", "drop_set"])[["dep_f1", "dvg_f1", "val_f1"]].mean()
fam = {}
for pt in ["conservative", "performance_seeking", "leakage_aware"]:
    xs = [x for x in E1 if x["prompt_type"] == pt]
    k = sum(x["trap_triggered"] for x in xs)
    sets = ["|".join(sorted(c for c in x["columns_to_drop"] if c != "y")) or "(none)" for x in xs]
    nd = np.mean([len([c for c in x["columns_to_drop"] if c not in ("duration", "y")]) for x in xs])
    w = collections.Counter(sets)
    def wm(split, col): return sum(m.loc[(split, s), col] * c for s, c in w.items()) / len(xs)
    fam[pt] = dict(k=k, n=len(xs), other_drops=nd, dep_f1_r=wm("random", "dep_f1"), dvg_f1_r=wm("random", "dvg_f1"),
                   dep_f1_t=wm("temporal", "dep_f1"), dvg_f1_t=wm("temporal", "dvg_f1"))
ref = dict(dep_f1_r=m.loc[("random", "duration"), "dep_f1"], dvg_f1_r=m.loc[("random", "duration"), "dvg_f1"],
           dep_f1_t=m.loc[("temporal", "duration"), "dep_f1"], dvg_f1_t=m.loc[("temporal", "duration"), "dvg_f1"])
NUM["E1"] = dict(fam=fam, ref=ref, target_dropped=sum("y" in x["columns_to_drop"] for x in E1))
lab = {"conservative": "Conservative", "performance_seeking": "Performance-seeking", "leakage_aware": "Leakage-aware (guardrail)"}
with open(T / "tab_e1.tex", "w") as f:
    f.write("\\begin{tabular}{@{}lcrrrrr@{}}\n\\toprule\n & \\code{duration} removed & Other columns & \\multicolumn{2}{c}{Random split} & \\multicolumn{2}{c}{Chronological split} \\\\\n"
            "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\nPrompt family & $k/K$ [Wilson 95\\%] & removed (mean) & Dep.\\ F1 & \\DVG-F1 & Dep.\\ F1 & \\DVG-F1 \\\\\n\\midrule\n")
    for pt, v in fam.items():
        f.write(f"{lab[pt]} & {kn(v['k'], v['n'])} & {v['other_drops']:.1f} & {v['dep_f1_r']:.3f} & {v['dvg_f1_r']:.3f} & {v['dep_f1_t']:.3f} & {v['dvg_f1_t']:.3f} \\\\\n")
    f.write(f"\\addlinespace\n\\emph{{Reference: only \\code{{duration}} removed}} & -- & 0 & {ref['dep_f1_r']:.3f} & {ref['dvg_f1_r']:.3f} & {ref['dep_f1_t']:.3f} & {ref['dvg_f1_t']:.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
# ── E2 original names (conservative) vs anonymised
E2 = [x for x in V if x["exp"].startswith("E2")]
def cnt(nm, r, pt="conservative"):
    xs = [x for x in E2 if x["prompt_type"] == pt and x["name_difficulty"] == nm and abs(x["r"] - r) < 1e-9]
    return sum(x["trap_triggered"] for x in xs), len(xs)
anon = {("medium", 0.80): (1, 6), ("medium", 0.95): (3, 26)}
NUM["E2"] = {}
with open(T / "tab_e2.tex", "w") as f:
    f.write("\\begin{tabular}{@{}llccc@{}}\n\\toprule\nTrap name & $r$ & Original names & Anonymised schema & Other prompts, original names \\\\\n\\midrule\n")
    for nm, col in [("easy", "credit\\_risk\\_proxy"), ("medium", "repayment\\_score"), ("hard", "internal\\_customer\\_index")]:
        for r in [0.60, 0.80, 0.95]:
            k, n = cnt(nm, r); a = anon.get((nm, r))
            po = [cnt(nm, r, p) for p in ["performance_seeking", "leakage_aware"]]
            NUM["E2"][f"{nm}_{r}"] = (k, n)
            f.write(f"\\code{{{col}}} & {r:.2f} & {kn(k, n)} & {kn(*a) if a else '--'} & {po[0][0]+po[1][0]}/{po[0][1]+po[1][1]} \\\\\n")
        if nm != "hard": f.write("\\addlinespace\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
NUM["E2"]["medium_total"] = [sum(cnt("medium", r)[i] for r in [0.6, 0.8, 0.95]) for i in (0, 1)]
NUM["E2"]["hard_total"] = [sum(cnt("hard", r)[i] for r in [0.6, 0.8, 0.95]) for i in (0, 1)]
NUM["E2"]["mentions_leak_hard"] = sum(("leak" in (x["raw_output"] or "").lower()) for x in E2 if x["name_difficulty"] == "hard" and x["prompt_type"] == "conservative" and x["trap_triggered"])
# ── E3 agentic vs single call
single = {("Adult", "derived_income_score", 0.80): (15, 26), ("Adult", "derived_income_score", 0.95): (0, 26),
          ("Adult", "admin_score_17", 0.95): (15, 26), ("German", "repayment_score", 0.95): (0, 6), ("Bank", "duration", None): (0, 10)}
rows = []; NUM["E3"] = {}
for (ds, col, r), sref in single.items():
    xs = [x for x in E3 if x["trap_col"] == col and (r is None or abs((x.get("r") or 0) - r) < 1e-9) and ds.lower() in x["exp"]]
    k = sum(x["trap_triggered"] for x in xs); n = len(xs)
    insp = sum(any(col in t.get("code", "") for t in x["trajectory"]) for x in xs)
    rel = sum(any(col in t.get("code", "") and ("groupby" in t["code"] or "corr" in t["code"]) for t in x["trajectory"]) for x in xs)
    calls = np.mean([x["n_code_calls"] for x in xs])
    NUM["E3"][f"{ds}|{col}|{r}"] = dict(k=k, n=n, inspected=insp, relation=rel, calls=calls, single=sref)
    rows.append((ds, col, r, sref, k, n, insp, rel, calls))
with open(T / "tab_e3.tex", "w") as f:
    f.write("\\begin{tabular}{@{}llccccr@{}}\n\\toprule\n & & Single call & \\multicolumn{3}{c}{Iterative code-executing agent} & Code calls \\\\\n"
            "\\cmidrule(lr){4-6}\nDataset & Leak (\\,$r$\\,) & removed & removed & inspected & vs.\\ target & (mean) \\\\\n\\midrule\n")
    for ds, col, r, sref, k, n, insp, rel, calls in rows:
        rr = f"{r:.2f}" if r else "natural"
        f.write(f"{ds} & \\code{{{col.replace('_', chr(92)+'_')}}} ({rr}) & {kn(*sref)} & {kn(k, n)} & {insp}/{n} & {rel}/{n} & {calls:.1f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
# ── E4
E4 = [x for x in V if x["exp"].startswith("E4")]
blind = {("medium", 0.80): (15, 26), ("medium", 0.95): (0, 26), ("hard", 0.80): (1, 6), ("hard", 0.95): (15, 26)}
NUM["E4"] = {}
with open(T / "tab_e4.tex", "w") as f:
    f.write("\\begin{tabular}{@{}llcccc@{}}\n\\toprule\n & & Blind (pooled, & \\multicolumn{3}{c}{Profiled views, $K{=}5$ each} \\\\\n\\cmidrule(lr){4-6}\n"
            "Trap name & $r$ & Table~\\ref{tab:pooled}) & Profile & Profile $+$ notice & Named, kept$^{a}$ \\\\\n\\midrule\n")
    for (nm, r), b in blind.items():
        out = {}
        for lvl in ["profiled", "profiled_notice"]:
            xs = [x for x in E4 if x["name_difficulty"] == nm and abs(x["r"] - r) < 1e-9 and x["info_level"] == lvl]
            out[lvl] = (sum(x["trap_triggered"] for x in xs), len(xs),
                        sum((x["trap_col"] in json.dumps(x.get("operations") or [])) and not x["trap_triggered"] for x in xs))
        NUM["E4"][f"{nm}_{r}"] = out
        col = "derived\\_income\\_score" if nm == "medium" else "admin\\_score\\_17"
        f.write(f"\\code{{{col}}} & {r:.2f} & {kn(*b)} & {kn(*out['profiled'][:2])} & {kn(*out['profiled_notice'][:2])} & {out['profiled'][2]+out['profiled_notice'][2]}/10 \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
json.dump(NUM, open(R / "agent_numbers.json", "w"), indent=1, default=float)
print(json.dumps(NUM, indent=1, default=float)[:3500])
