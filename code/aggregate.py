"""Aggregate revision analyses into LaTeX tables, a figure, and a JSON of every number quoted in the text."""
import json, math
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).parent.parent / "results"
TAB = Path(__file__).parent.parent / "tables"; TAB.mkdir(exist_ok=True)
FIG = Path(__file__).parent.parent / "figures"; FIG.mkdir(exist_ok=True)
NUM = {}
Z = 1.645  # one-sided 95%


def wilson(k, n, z=1.96):
    if n == 0: return (0, 1)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def pint(x):  # mean and 2.5/97.5 percentile across seeds
    x = np.asarray(x); return x.mean(), np.percentile(x, 2.5), np.percentile(x, 97.5)


def f3(x): return f"{x:.3f}".replace("-0.000", "0.000")
def sgn(x): s = f"{x:+.3f}"; return "$" + s.replace("+0.000", "0.000").replace("-0.000", "0.000") + "$"
def ci(m, lo, hi, signed=False):
    fm = (lambda v: f"{v:+.3f}") if signed else f3
    return f"${fm(m)}$ {{\\scriptsize[{fm(lo)}, {fm(hi)}]}}"


# ───────────────────────── leakage replay ─────────────────────────
A = pd.read_csv(R / "adult_replay_all.csv"); G = pd.read_csv(R / "german_replay.csv")
rows_tab = []; rows_cal = []; figdata = {}
for name, D in [("Adult Income", A), ("German Credit", G)]:
    D = D.copy(); D["dvg"] = D.val_acc - D.dep_acc
    c = D[D.cond == "clean"].copy(); r = D[D.cond == "retained"].copy()
    r["pig"] = r.val_acc - r.lc_val_acc; r["dvg_f1"] = r.val_f1 - r.dep_f1
    n_val, n_dep = int(c.n_val.iloc[0]), int(c.n_dep.iloc[0])
    # size-aware analytic SE of a validation-vs-hold-out accuracy difference
    c["se"] = np.sqrt(c.val_acc * (1 - c.val_acc) * (1 / n_val + 1 / n_dep))
    r["se"] = np.sqrt(r.lc_val_acc * (1 - r.lc_val_acc) * (1 / n_val + 1 / n_dep))
    tau = Z * c.se.mean()
    seed_c = c.groupby("seed").dvg.mean()
    key = "adult" if name.startswith("Adult") else "german"
    NUM[key] = dict(n_seeds=int(c.seed.nunique()), n_val=n_val, n_dep=n_dep,
                    clean_dvg_mean=c.dvg.mean(), clean_dvg_sd=c.dvg.std(), clean_dvg_p95=c.dvg.abs().quantile(.95),
                    clean_dvg_seedmean_sd=seed_c.std(), clean_dvg_seedmean_p95=seed_c.abs().quantile(.95),
                    clean_dvg_min=c.dvg.min(), clean_dvg_max=c.dvg.max(),
                    analytic_se=c.se.mean(), tau=tau,
                    false_alarm=(c.dvg > (Z * c.se)).mean(),
                    false_alarm_fixed001=(c.dvg > 0.01).mean())
    rows_cal.append((name, n_val, n_dep, c.dvg.std(), c.dvg.abs().quantile(.95), c.se.mean(), tau,
                     (c.dvg > Z * c.se).mean(), (c.dvg > 0.01).mean()))
    figdata[name] = dict(clean=c.dvg.values, tau=tau, ret={})
    for rr in [0.60, 0.80, 0.95]:
        x = r[r.r == rr]
        s_pig = x.groupby("seed").pig.mean(); s_dvg = x.groupby("seed").dvg.mean(); s_f1 = x.groupby("seed").dvg_f1.mean()
        power = (x.dvg > Z * x.se).mean()
        extra = {}
        if "dep_acc_perm" in x:
            extra["dvg_perm"] = (x.val_acc - x.dep_acc_perm).mean()
        if "dep_acc_median" in x and x.dep_acc_median.notna().any():
            extra["dvg_median"] = (x.val_acc - x.dep_acc_median).mean()
        NUM[key][f"r{rr}"] = dict(pig=pint(s_pig), dvg=pint(s_dvg), dvg_f1=pint(s_f1), power=power,
                                   pig_sd_clf=x.pig.std(), dvg_sd_clf=x.dvg.std(), **extra)
        rows_tab.append((name, rr, pint(s_pig), pint(s_dvg), pint(s_f1), power))
        figdata[name]["ret"][rr] = x.dvg.values
    if key == "german":
        NUM[key]["index_check"] = dict(dep_with_idx=c.dep_acc.mean(), dep_without_idx=c.dep_acc_noidx.mean(),
                                        val_with_idx=c.val_acc.mean(), val_without_idx=c.val_acc_noidx.mean())

with open(TAB / "tab_split_robust.tex", "w") as f:
    f.write("\\begin{tabular}{@{}llrrrr@{}}\n\\toprule\n"
            "Dataset & $r$ & \\PIG\\ [95\\% split interval] & \\DVG\\ [95\\% split interval] & \\DVG-F1 & Exceeds $\\tau_n$ \\\\\n\\midrule\n")
    last = None
    for name, rr, p, d, f1, pw in rows_tab:
        if last and last != name: f.write("\\addlinespace\n")
        f.write(f"{name if name != last else ''} & {rr:.2f} & {ci(*p)} & {ci(*d)} & ${f3(f1[0])}$ & {100*pw:.0f}\\% \\\\\n"); last = name
    f.write("\\bottomrule\n\\end{tabular}\n")

with open(TAB / "tab_noise_floor.tex", "w") as f:
    f.write("\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n"
            "Dataset & $n_{\\text{val}}$ & $n_{\\text{dep}}$ & SD & P95 $|\\DVG|$ & $\\widehat{\\mathrm{SE}}_n$ & $\\tau_n$ & False alarms ($\\tau_n$ / $0.01$) \\\\\n\\midrule\n")
    for name, nv, nd, sd, p95, se, tau, fa, fa01 in rows_cal:
        f.write(f"{name} & {nv:,} & {nd:,} & {sd:.4f} & {p95:.4f} & {se:.4f} & {tau:.3f} & {100*fa:.0f}\\% / {100*fa01:.0f}\\% \\\\\n".replace(",", "{,}"))
    f.write("\\bottomrule\n\\end{tabular}\n")

# per-classifier noise floor (supplement)
with open(TAB / "tab_s_noise_by_clf.tex", "w") as f:
    f.write("\\begin{tabular}{@{}llrrrr@{}}\n\\toprule\nDataset & Classifier & Mean & SD & Min & Max \\\\\n\\midrule\n")
    for name, D in [("Adult Income", A), ("German Credit", G)]:
        c = D[D.cond == "clean"].copy(); c["dvg"] = c.val_acc - c.dep_acc
        for k, x in c.groupby("clf").dvg:
            f.write(f"{name} & {k} & {x.mean():+.4f} & {x.std():.4f} & {x.min():+.4f} & {x.max():+.4f} \\\\\n")
        f.write("\\addlinespace\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# imputation robustness (supplement)
with open(TAB / "tab_s_imputation.tex", "w") as f:
    f.write("\\begin{tabular}{@{}llrrr@{}}\n\\toprule\nDataset & $r$ & Default fill (as in main runs) & Training-median fill & Uninformative draw \\\\\n\\midrule\n")
    for key, name in [("adult", "Adult Income"), ("german", "German Credit")]:
        for rr in [0.60, 0.80, 0.95]:
            d = NUM[key][f"r{rr}"]
            med = f3(d["dvg_median"]) if key == "adult" else f3(d["dvg"][0])
            f.write(f"{name if rr == 0.60 else ''} & {rr:.2f} & {f3(d['dvg'][0])} & {med} & {f3(d['dvg_perm'])} \\\\\n")
        if key == "adult": f.write("\\addlinespace\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ───────────────────────── shift and external test (Adult) ─────────────────────────
Ad = A.copy()
shifts = ["female_15", "female_50", "age45_20", "age45_60", "prev_12", "prev_40"]
lab = {"female_15": "Female share 15\\%", "female_50": "Female share 50\\%", "age45_20": "Age $\\geq45$: 20\\%",
       "age45_60": "Age $\\geq45$: 60\\%", "prev_12": "Prevalence 12\\%", "prev_40": "Prevalence 40\\%",
       "dep": "Unshifted dep.-proxy", "ext": "Official test release"}
c = Ad[Ad.cond == "clean"].set_index(["seed", "clf"]); NUM["shift"] = {}
with open(TAB / "tab_shift.tex", "w") as f:
    f.write("\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n & Clean model & \\multicolumn{2}{c}{$r=0.60$} & \\multicolumn{2}{c}{$r=0.80$} & $r=0.95$ \\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-7}\n"
            "Deployment-proxy set & gap & \\DVG & \\DVGex & \\DVG & \\DVGex & \\DVGex \\\\\n\\midrule\n")
    for s in ["dep", "ext"] + shifts:
        colname = {"dep": "dep_acc", "ext": "ext_acc"}.get(s, f"shift_{s}")
        cg = (c.val_acc - c[colname])
        cells = [f"{cg.mean():+.3f}"]; NUM["shift"][s] = {"clean_gap": cg.mean()}
        for rr in [0.60, 0.80, 0.95]:
            x = Ad[(Ad.cond == "retained") & (Ad.r == rr)].set_index(["seed", "clf"])
            g = x.val_acc - x[colname]; ex = g - cg.reindex(g.index)
            NUM["shift"][s][f"r{rr}"] = dict(dvg=g.mean(), excess=ex.mean(), excess_min=ex.groupby(level=0).mean().min())
            cells += ([f"{g.mean():.3f}", f"{ex.mean():.3f}"] if rr < 0.9 else [f"{ex.mean():.3f}"])
        f.write(f"{lab[s]} & " + " & ".join(cells) + " \\\\\n")
        if s == "ext": f.write("\\addlinespace\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ───────────────────────── deletion replay ─────────────────────────
D = pd.read_csv(R / "deletion_replay_all.csv"); NUM["deletion"] = {"n_seeds": int(D.seed.nunique()),
     "base_prot_recall": D.base_prot_recall.mean(), "base_eo": D.base_eo.mean(), "base_acc": D.base_acc.mean()}
with open(TAB / "tab_deletion_replay.tex", "w") as f:
    f.write("\\begin{tabular}{@{}llrrrr@{}}\n\\toprule\nFlagged set & $\\phi$ & Removed & $\\Delta$Acc & $\\Delta$EO gap [95\\% split int.] & $\\Delta$Prot.\\ recall [95\\% split int.] \\\\\n\\midrule\n")
    for scen, phi in [("protected_positive", .25), ("protected_positive", .5), ("protected_positive", .75),
                      ("random_positive", .75), ("unprotected_positive", .75)]:
        x = D[(D.scenario == scen) & (D.phi == phi) & (D["eval"] == "dep")]
        s = x.groupby("seed")[["d_acc", "d_eo", "d_prot_recall", "removed"]].mean()
        e = pint(s.d_eo); p = pint(s.d_prot_recall); a = s.d_acc.mean()
        xe = D[(D.scenario == scen) & (D.phi == phi) & (D["eval"] == "ext")].groupby("seed").d_prot_recall.mean()
        NUM["deletion"][f"{scen}_{phi}"] = dict(d_acc=a, d_eo=e, d_prot=p, removed=s.removed.mean(), d_prot_ext=pint(xe),
                                                 d_acc_abs_max=x.d_acc.abs().max())
        f.write(f"{scen.replace('_', '-')} & {phi:.2f} & {s.removed.mean():.0f} & {sgn(a)} & {ci(*e, signed=True)} & {ci(*p, signed=True)} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ───────────────────────── Bank case study ─────────────────────────
B = pd.concat([pd.read_csv(R / "bank_case_temporal.csv"), pd.read_csv(R / "bank_case_random_0.csv")])
B["pig"] = B.val_acc - B.lc_val_acc; B["dvg"] = B.val_acc - B.dep_acc; B["dvg_lc"] = B.lc_val_acc - B.clean_dep_acc
B["pig_f1"] = B.val_f1 - B.lc_val_f1; B["dvg_f1"] = B.val_f1 - B.dep_f1; B["dvg_lc_f1"] = B.lc_val_f1 - B.clean_dep_f1
B["ex"] = B.dvg - B.dvg_lc; B["ex_f1"] = B.dvg_f1 - B.dvg_lc_f1
NUM["bank"] = json.load(open(R / "bank_screen.json"))
bk = pd.read_csv(Path(__file__).parent.parent / "data" / "bank_additional_full.csv", sep=";"); n = len(bk)
NUM["bank"]["prevalence_temporal"] = [float((bk.y.iloc[a:b] == "yes").mean()) for a, b in [(0, int(.6*n)), (int(.6*n), int(.8*n)), (int(.8*n), n)]]
with open(TAB / "tab_bank.tex", "w") as f:
    f.write("\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\n & & \\multicolumn{3}{c}{Accuracy} & \\multicolumn{3}{c}{Macro-F1} \\\\\n"
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\nSplit & Retained feature & \\PIG & \\DVG & \\DVGex & \\PIG & \\DVG & \\DVGex \\\\\n\\midrule\n")
    for split, feat, lbl in [("random", "duration (natural)", "\\code{duration} (natural)"),
                             ("temporal", "duration (natural)", "\\code{duration} (natural)"),
                             ("temporal", "synthetic trap r=0.6", "synthetic trap, $r=0.60$"),
                             ("temporal", "synthetic trap r=0.8", "synthetic trap, $r=0.80$"),
                             ("temporal", "synthetic trap r=0.95", "synthetic trap, $r=0.95$")]:
        x = B[(B.split == split) & (B.feature == feat)]
        m = x[["pig", "dvg", "ex", "pig_f1", "dvg_f1", "ex_f1", "dvg_lc", "dvg_lc_f1"]].mean()
        NUM["bank"][f"{split}|{feat}"] = m.to_dict()
        if split == "random": NUM["bank"][f"{split}|{feat}"].update(sd_dvg=x.groupby("seed").dvg.mean().std(), sd_dvg_f1=x.groupby("seed").dvg_f1.mean().std())
        sp = "Random (10 seeds)" if split == "random" else ("Chronological" if feat.startswith("duration") else "")
        f.write(f"{sp} & {lbl} & {m.pig:.3f} & {m.dvg:.3f} & {m.ex:+.3f} & {m.pig_f1:.3f} & {m.dvg_f1:.3f} & {m.ex_f1:+.3f} \\\\\n")
        if split == "random": f.write("\\addlinespace\n")
    x = B[(B.split == "temporal") & (B.feature == "duration (natural)")]
    f.write(f"\\addlinespace\n\\multicolumn{{2}}{{@{{}}l}}{{\\emph{{Leakage-controlled model, chronological}}}} & -- & {x.dvg_lc.mean():.3f} & -- & -- & {x.dvg_lc_f1.mean():.3f} & -- \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ───────────────────────── pooled detection evidence ─────────────────────────
pool = [("Adult", "medium", 0.60, [(0, 1), (1, 5)]), ("Adult", "medium", 0.80, [(0, 1), (1, 5), (14, 20)]),
        ("Adult", "medium", 0.95, [(0, 1), (0, 5), (0, 20)]), ("Adult", "hard", 0.60, [(1, 1), (1, 5)]),
        ("Adult", "hard", 0.80, [(0, 1), (1, 5)]), ("Adult", "hard", 0.95, [(0, 1), (2, 5), (13, 20)]),
        ("German", "medium", 0.80, [(0, 1), (1, 5)]), ("German", "medium", 0.95, [(0, 1), (2, 5), (1, 20)])]
NUM["pooled"] = []
with open(TAB / "tab_pooled_detection.tex", "w") as f:
    f.write("\\begin{tabular}{@{}lllcccrl@{}}\n\\toprule\nDataset & Name & $r$ & Grid & $K{=}5$ & $K{=}20$ & Pooled & Wilson 95\\% CI \\\\\n\\midrule\n")
    last = None
    for ds, nm, rr, ev in pool:
        k = sum(a for a, _ in ev); n_ = sum(b for _, b in ev); lo, hi = wilson(k, n_)
        cells = [f"{a}/{b}" for a, b in ev] + ["--"] * (3 - len(ev))
        NUM["pooled"].append(dict(ds=ds, name=nm, r=rr, k=k, n=n_, lo=lo, hi=hi))
        if last and last != ds: f.write("\\addlinespace\n")
        f.write(f"{ds if ds != last else ''} & {nm} & {rr:.2f} & " + " & ".join(cells) + f" & {k}/{n_} & $[{lo:.2f},\\,{hi:.2f}]$ \\\\\n"); last = ds
    f.write("\\bottomrule\n\\end{tabular}\n")

# ───────────────────────── figure: noise floor vs retained ─────────────────────────
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=False)
for ax, name in zip(axes, ["Adult Income", "German Credit"]):
    d = figdata[name]
    data = [d["clean"]] + [d["ret"][rr] for rr in [0.60, 0.80, 0.95]]
    pos = [0, 1, 2, 3]
    bp = ax.boxplot(data, positions=pos, widths=0.55, patch_artist=True, showfliers=False)
    cols = ["#9e9e9e", "#f4a261", "#e76f51", "#b23a48"]
    for patch, col in zip(bp["boxes"], cols): patch.set_facecolor(col); patch.set_alpha(0.55)
    rng = np.random.default_rng(0)
    for p_, arr, col in zip(pos, data, cols):
        ax.scatter(p_ + rng.uniform(-0.18, 0.18, len(arr)), arr, s=5, color=col, alpha=0.8, zorder=3, linewidths=0)
    ax.axhline(d["tau"], color="#1b7f79", lw=1.1, ls="--"); ax.axhline(0.01, color="#555", lw=0.8, ls=":")
    ax.text(3.45, d["tau"], r"$\tau_n$", color="#1b7f79", va="bottom", ha="right", fontsize=8)
    ax.text(-0.45, 0.01, "0.01", color="#555", va="bottom", fontsize=7)
    ax.set_xticks(pos); ax.set_xticklabels(["trap\nremoved", "retained\nr=0.60", "retained\nr=0.80", "retained\nr=0.95"])
    ax.set_title(f"{name} ({NUM['adult' if name.startswith('Adult') else 'german']['n_seeds']} split seeds × 4 classifiers)", fontsize=9)
    ax.set_ylabel("DVG (accuracy)"); ax.grid(axis="y", alpha=0.3)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "fig_split_robust_dvg.pdf"); fig.savefig(FIG / "fig_split_robust_dvg.png", dpi=200)


def conv(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, tuple): return [conv(v) for v in o]
    if isinstance(o, dict): return {k: conv(v) for k, v in o.items()}
    if isinstance(o, list): return [conv(v) for v in o]
    return o
json.dump(conv(NUM), open(R / "robustness_numbers.json", "w"), indent=1)
print(json.dumps(conv(NUM), indent=1)[:6000])
