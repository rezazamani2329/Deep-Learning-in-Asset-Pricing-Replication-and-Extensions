# %% [markdown]
# # 09 — Rolling Out-of-Sample Performance: LSTM versus Transformer
# 
# ## Process
# 1. Load Code 08's saved monthly factor returns and verify exact dates, split labels, finite decimal returns, and agreement with its performance table.
# 2. Keep the **300 untouched test months, January 1992–December 2016**, for the primary matched-budget comparison. Do not retrain or select a model.
# 3. Calculate trailing **36- and 60-month** annualized Sharpe, volatility, arithmetic mean return, compounded return, and maximum drawdown. Require complete windows.
# 4. Plot cumulative wealth, full-period drawdowns, and trailing correlations. Examine when the Transformer exceeds the matched LSTM.
# 5. Check formulas with known examples and test that changing future returns cannot alter earlier rolling statistics.
# 6. Save tables, figures, and a common dated test-return file; answer the questions below and explain the handoff to Code 10.
# 
# ## Questions
# **Q1.** Does the higher full-test Transformer Sharpe from Code 08 persist across rolling windows?
# 
# **Q2.** How do volatility, cumulative performance, and drawdowns differ?
# 
# **Q3.** How closely do the two factors move together, and does their relationship change over time?
# 
# **Q4.** What do these diagnostics establish, and how do they support Code 10's economic-regime analysis?
# 
# ## Scope and conventions
# The primary pair is Code 08's **three-seed matched LSTM versus three-seed Transformer** (64/16/128 epochs). The original nine-member LSTM is a separately labelled reference only. This notebook performs rolling **evaluation**, not rolling retraining or expanding-window model selection. Both primary factors were trained before the test period.
# 
# All rolling windows use only test observations, include the current month, and require all 36/60 returns. Thus the first results appear in **December 1994 / December 1996**. These are end-of-month realized diagnostics, not predictions for that same month. Calendar dates are represented by month starts.
# 
# Return units and the author-supplied return convention are inherited from Code 08. No extra risk-free subtraction is applied. Monthly Sharpe is mean / sample standard deviation (`ddof=1`); annualized Sharpe and volatility multiply by √12. This convention does not correct for serial dependence. Rolling mean is annualized arithmetically as 12 × monthly mean; compounded window returns and geometric annualized returns are separately labelled.
# 
# Wealth is `∏(1+r)` from initial capital 1, assuming monthly rebalancing and treating gross-normalized factor returns as strategy returns. It excludes fees, financing costs, and trading frictions. Maximum drawdown is a **wealth decline from a running peak**, including initial wealth 1; it is different from consecutive losing months in Code 07. Rolling drawdowns restart wealth at 1 for each window.
# 
# Overlapping windows are dependent. Window win fractions, extremes, and correlations are descriptive; they are not independent statistical tests or evidence of universal superiority. No windows or models are selected for future use based on the test results.
# 
# 
# ## Expanded comparison — all six models
# Also verify LS, Elastic Net, FFN, original nine-member GAN–LSTM, matched three-member GAN–LSTM, and Transformer across train/validation/test and both rolling windows. Paper values are separately labelled published benchmarks; no paper monthly factor series is available for a rolling paper comparison.

# %% [markdown]
# ## 1 — Imports, repository paths, and fixed settings

# %%
from pathlib import Path
import json, hashlib, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

def find_root():
    for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (p / "notebooks" / "08_transformer_sdf.ipynb").exists() and (p / "results").exists():
            return p
    raise FileNotFoundError("Start Jupyter inside the replication repository.")
ROOT = find_root()
SOURCE = ROOT / "results" / "transformer_extension" / "factors" / "transformer_lstm_comparison.csv"
SOURCE_PERFORMANCE = ROOT / "results" / "transformer_extension" / "tables" / "ensemble_performance.csv"
OUT = ROOT / "results" / "rolling_oos"
for name in ["tables", "figures", "factors"]:
    (OUT / name).mkdir(parents=True, exist_ok=True)
WINDOWS = (36, 60)
MODELS = {"gan_lstm_matched": "LSTM (matched)", "gan_transformer": "Transformer"}
COLORS = {"gan_lstm_matched": "#2563eb", "gan_transformer": "#ea580c"}
print("Source:", SOURCE)
print("Outputs:", OUT)


# %% [markdown]
# ## 2 — Verify Code 08 outputs and isolate the test sample

# %%
data = pd.read_csv(SOURCE, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
expected = pd.date_range("1967-01-01", "2016-12-01", freq="MS")
assert pd.DatetimeIndex(data.date).equals(expected)
assert not data.date.duplicated().any()
assert data.split.tolist() == ["train"]*240 + ["validation"]*60 + ["test"]*300
all_cols = [*MODELS, "gan_lstm_code07_reference"]
assert np.isfinite(data[all_cols].to_numpy()).all()
assert (data[all_cols] > -1).all().all(), "Compounded wealth requires returns above -100%."
saved = pd.read_csv(SOURCE_PERFORMANCE)
validation_rows = []
for model in all_cols:
    for split, group in data.groupby("split"):
        r = group[model]
        sr = r.mean()/r.std(ddof=1)
        record = saved[(saved.model==model)&(saved.split==split)]
        assert len(record)==1
        assert len(r)==record.iloc[0].months
        assert np.isclose(sr, record.iloc[0].monthly_sharpe, rtol=1e-6, atol=1e-8)
        validation_rows.append({"model": model, "split": split, "months": len(r), "monthly_sharpe": sr, "verified": True})
source_checks = pd.DataFrame(validation_rows)
source_checks.to_csv(OUT / "tables" / "source_integrity.csv", index=False)
test = data[data.split=="test"].set_index("date").copy()
assert pd.DatetimeIndex(test.index).equals(pd.date_range("1992-01-01","2016-12-01",freq="MS"))
assert len(test)==300
manifest = {"source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "performance_sha256": hashlib.sha256(SOURCE_PERFORMANCE.read_bytes()).hexdigest(),
    "windows": list(WINDOWS), "test_months": 300, "test_start": "1992-01", "test_end": "2016-12",
    "primary_models": MODELS, "ddof": 1, "rolling_scope": "test only, trailing complete windows",
    "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
display(source_checks)
display(test.head())


# %% [markdown]
# ## 3 — Metric functions and formula checks

# %%
def drawdown_path(returns):
    r = np.asarray(returns, dtype=float)
    assert np.isfinite(r).all() and (r > -1).all()
    wealth = np.cumprod(1+r)
    peaks = np.maximum.accumulate(np.r_[1., wealth])[1:]
    return wealth/peaks - 1

def max_drawdown(returns):
    return float(np.min(drawdown_path(returns)))

def compounded_return(returns):
    return float(np.expm1(np.log1p(np.asarray(returns, dtype=float)).sum()))

def rolling_metrics(r, window):
    rolling = r.rolling(window, min_periods=window)
    mean = rolling.mean(); vol = rolling.std(ddof=1)
    sr = mean / vol.replace(0, np.nan)
    total = rolling.apply(compounded_return, raw=True)
    return pd.DataFrame({"mean_annualized_arithmetic": 12*mean,
        "volatility_annualized": np.sqrt(12)*vol,
        "sharpe_annualized": np.sqrt(12)*sr,
        "return_compounded_window": total,
        "return_annualized_geometric": np.expm1(np.log1p(total)*(12/window)),
        "max_drawdown_window": rolling.apply(max_drawdown, raw=True)})

np.testing.assert_allclose(drawdown_path([-.1,.2,-.25]), [-.1,0.,-.25], atol=1e-12)
assert np.isclose(max_drawdown([-.1]), -.1)  # Initial capital must be included.
assert np.isclose(compounded_return([.1,-.1]), -.01)
assert np.isclose(max_drawdown([.01,.02,.03]), 0.)
known = pd.Series([.02,-.01,.03,-.02,.04,.01])
known_stats = rolling_metrics(known, 3)
np.testing.assert_allclose(known_stats.sharpe_annualized.iloc[2], np.sqrt(12)*known.iloc[:3].mean()/known.iloc[:3].std(ddof=1))
assert known_stats.iloc[:2].isna().all().all()
flat = rolling_metrics(pd.Series([.01]*6), 3)
assert flat.sharpe_annualized.isna().all()
print("Known-example checks passed: Sharpe, compounding, complete windows, zero volatility, and drawdown including initial wealth.")


# %% [markdown]
# ## 4 — Complete trailing windows and future-perturbation checks

# %%
rolling_parts = []
for model in MODELS:
    for window in WINDOWS:
        values = rolling_metrics(test[model], window)
        assert values.sharpe_annualized.notna().sum() == 300-window+1
        assert values.iloc[:window-1].isna().all().all()
        assert np.isfinite(values.iloc[window-1:].to_numpy()).all()
        # Perturb only later months. Earlier realized diagnostics must be unchanged.
        altered = test[model].copy(); altered.iloc[150:] += .005
        causal = rolling_metrics(altered, window)
        pd.testing.assert_frame_equal(values.iloc[:150], causal.iloc[:150])
        values["model"] = model; values["window_months"] = window
        values["window_start"] = pd.Series(test.index, index=test.index).shift(window-1)
        rolling_parts.append(values.reset_index())
rolling = pd.concat(rolling_parts, ignore_index=True)
rolling.to_csv(OUT / "tables" / "rolling_metrics.csv", index=False)
display(rolling.dropna().head())
print("Verified 265 complete 36-month windows and 241 complete 60-month windows per model; future perturbations do not change past metrics.")


# %% [markdown]
# ## 5 — Rolling Sharpe and volatility comparisons

# %%
fig, axes = plt.subplots(2,2,figsize=(14,8),sharex="col")
for j,window in enumerate(WINDOWS):
    for model,label in MODELS.items():
        rows = rolling[(rolling.model==model)&(rolling.window_months==window)].set_index("date")
        axes[0,j].plot(rows.index,rows.sharpe_annualized,label=label,color=COLORS[model])
        axes[1,j].plot(rows.index,100*rows.volatility_annualized,label=label,color=COLORS[model])
    axes[0,j].axhline(0,color="grey",linewidth=.7)
    axes[0,j].set_title(f"{window}-month trailing annualized Sharpe")
    axes[0,j].set_ylabel("Annualized Sharpe")
    axes[1,j].set_title(f"{window}-month trailing annualized volatility")
    axes[1,j].set_ylabel("Volatility (%)")
    for ax in axes[:,j]: ax.legend(); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(OUT / "figures" / "rolling_sharpe_volatility.png",dpi=150); plt.show()

window_rows = []; differences = []
for window in WINDOWS:
    wide = rolling[rolling.window_months==window].pivot(index="date",columns="model",values="sharpe_annualized").dropna()
    delta = wide.gan_transformer-wide.gan_lstm_matched
    for date,value in delta.items():
        differences.append({"date": date,"window_months": window,"transformer_minus_lstm_sharpe": value})
    window_rows.append({"window_months": window,"complete_windows": len(delta),
        "first_window_end": delta.index.min(), "last_window_end": delta.index.max(),
        "transformer_higher_fraction": (delta>0).mean(), "mean_sharpe_difference": delta.mean(),
        "minimum_sharpe_difference": delta.min(), "minimum_difference_end": delta.idxmin(),
        "maximum_sharpe_difference": delta.max(), "maximum_difference_end": delta.idxmax()})
window_comparison = pd.DataFrame(window_rows)
sharpe_differences = pd.DataFrame(differences)
window_comparison.to_csv(OUT / "tables" / "window_comparison.csv", index=False)
sharpe_differences.to_csv(OUT / "tables" / "rolling_sharpe_differences.csv", index=False)
display(window_comparison)


# %% [markdown]
# ## 6 — Cumulative performance and downside risk
# We include the initial capital point immediately before the first test return, so a loss in the first month is counted correctly. Rolling maximum drawdown is measured separately inside each complete window.

# %%
paths = pd.DataFrame(index=test.index)
full_rows = []
for model,label in MODELS.items():
    r = test[model]
    wealth = (1+r).cumprod()
    dd = pd.Series(drawdown_path(r), index=r.index)
    paths[model+"_wealth"] = wealth
    paths[model+"_drawdown"] = dd
    full_rows.append({"model": model, "months": len(r), "monthly_sharpe": r.mean()/r.std(ddof=1),
        "annualized_sharpe": np.sqrt(12)*r.mean()/r.std(ddof=1),
        "annualized_volatility": np.sqrt(12)*r.std(ddof=1),
        "total_compounded_return": wealth.iloc[-1]-1,
        "annualized_geometric_return": wealth.iloc[-1]**(12/len(r))-1,
        "maximum_drawdown": dd.min(), "trough_month": dd.idxmin(),
        "terminal_wealth": wealth.iloc[-1]})
full_performance = pd.DataFrame(full_rows)
paths.to_csv(OUT / "tables" / "wealth_and_drawdown.csv",index_label="date")
full_performance.to_csv(OUT / "tables" / "full_test_performance.csv",index=False)
display(full_performance)
fig,axes = plt.subplots(2,1,figsize=(12,8),sharex=True)
for model,label in MODELS.items():
    wealth = pd.concat([pd.Series([1.],index=[pd.Timestamp("1991-12-01")]), paths[model+"_wealth"]])
    dd = pd.concat([pd.Series([0.],index=[pd.Timestamp("1991-12-01")]), paths[model+"_drawdown"]])
    axes[0].plot(wealth.index,wealth,label=label,color=COLORS[model])
    axes[1].plot(dd.index,100*dd,label=label,color=COLORS[model])
axes[0].set_title("Test-period compounded wealth (initial capital = 1)"); axes[0].set_ylabel("Wealth")
axes[1].set_title("Drawdown from running wealth peak"); axes[1].set_ylabel("Drawdown (%)")
for ax in axes:ax.legend();ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT / "figures" / "wealth_and_drawdown.png",dpi=150);plt.show()
fig,axes = plt.subplots(1,2,figsize=(14,4),sharey=True)
for ax,window in zip(axes,WINDOWS):
    for model,label in MODELS.items():
        rows = rolling[(rolling.model==model)&(rolling.window_months==window)].set_index("date")
        ax.plot(rows.index,100*rows.max_drawdown_window,label=label,color=COLORS[model])
    ax.set_title(f"{window}-month window maximum drawdown"); ax.legend(); ax.grid(alpha=.2)
axes[0].set_ylabel("Maximum drawdown (%)")
fig.tight_layout();fig.savefig(OUT / "figures" / "rolling_drawdowns.png",dpi=150);plt.show()


# %% [markdown]
# ## 7 — Rolling factor correlations
# Pearson correlations use the same complete trailing test windows. They measure co-movement, not performance superiority or independent economic identification.

# %%
correlation = pd.DataFrame(index=test.index)
for window in WINDOWS:
    values = test.gan_lstm_matched.rolling(window,min_periods=window).corr(test.gan_transformer)
    assert values.notna().sum()==300-window+1
    assert (values.dropna().abs()<=1+1e-12).all()
    correlation[f"correlation_{window}m"] = values
correlation.to_csv(OUT / "tables" / "rolling_correlations.csv",index_label="date")
full_correlation = test.gan_lstm_matched.corr(test.gan_transformer)
correlation_summary = pd.DataFrame([{"window_months":window,
    "full_test_correlation":full_correlation,
    "minimum_rolling_correlation":correlation[f"correlation_{window}m"].min(),
    "maximum_rolling_correlation":correlation[f"correlation_{window}m"].max(),
    "mean_rolling_correlation":correlation[f"correlation_{window}m"].mean()} for window in WINDOWS])
correlation_summary.to_csv(OUT / "tables" / "correlation_summary.csv",index=False)
display(correlation_summary)
fig,ax = plt.subplots(figsize=(12,4))
for window in WINDOWS:ax.plot(correlation.index,correlation[f"correlation_{window}m"],label=f"{window} months")
ax.set_ylim(-1,1); ax.axhline(0,color="grey",linewidth=.7); ax.set_title("Trailing LSTM–Transformer return correlation")
ax.set_ylabel("Pearson correlation"); ax.legend(); ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT / "figures" / "rolling_correlations.png",dpi=150);plt.show()


# %% [markdown]
# ## 8 — Save Code 10 inputs and final integrity checks

# %%
# Preserve both the primary pair and the separately labelled Code 07 reference.
test.reset_index().to_csv(OUT / "factors" / "test_factor_returns.csv",index=False)
required = ["rolling_metrics.csv","window_comparison.csv","rolling_sharpe_differences.csv",
    "full_test_performance.csv","wealth_and_drawdown.csv","rolling_correlations.csv","correlation_summary.csv","source_integrity.csv"]
for name in required:assert (OUT / "tables" / name).exists()
reloaded = pd.read_csv(OUT / "factors" / "test_factor_returns.csv",parse_dates=["date"])
pd.testing.assert_frame_equal(reloaded.set_index("date"),test,check_dtype=False,check_freq=False)
assert len(rolling)==2*2*300
assert not rolling.duplicated(["date","model","window_months"]).any()
assert len(sharpe_differences)==265+241
assert len(source_checks)==9 and source_checks.verified.all()
print("All source, calendar, formula, causality, complete-window, and saved-output checks passed.")


# %% [markdown]
# ## 9 — Verify and compare Sharpe ratios for all six models
# 
# The four original replication models are loaded from Code 07. The two matched-budget models come from Code 08. Recalculate Sharpe from monthly returns rather than trusting a displayed number. The paper has LS, EN, FFN, and GAN benchmarks; it has no Transformer result. Its reported full-period Sharpe cannot be substituted for a time-varying rolling Sharpe series.
# 
# All-model charts add broader context. The architecture comparison remains matched LSTM versus Transformer; training-budget differences remain relevant when comparing with the original GAN.

# %%
core = pd.read_csv(ROOT / "results" / "factors" / "final_all_model_factor_returns.csv",parse_dates=["date"])
assert not core.duplicated(["model","date"]).any()
core_wide = core.pivot(index="date",columns="model",values="factor_return").sort_index()
assert pd.DatetimeIndex(core_wide.index).equals(expected)
assert set(core_wide.columns)=={"LS","EN","FFN","GAN"}
all_returns = core_wide.rename(columns={"GAN":"GAN–LSTM original"}).copy()
all_returns["GAN–LSTM matched"] = data.set_index("date").gan_lstm_matched
all_returns["GAN–Transformer"] = data.set_index("date").gan_transformer
ALL_ORDER = ["LS","EN","FFN","GAN–LSTM original","GAN–LSTM matched","GAN–Transformer"]
all_returns = all_returns[ALL_ORDER]
assert np.isfinite(all_returns.to_numpy()).all()
np.testing.assert_allclose(all_returns["GAN–LSTM original"],data.gan_lstm_code07_reference,rtol=1e-6,atol=1e-8)
original_summary = pd.read_csv(ROOT / "results" / "tables" / "final_replication_performance.csv")
expected_splits = {"train":(0,240),"validation":(240,300),"test":(300,600)}
all_rows=[]
for name in ALL_ORDER:
    for split,(a,b) in expected_splits.items():
        r=all_returns[name].iloc[a:b];monthly=r.mean()/r.std(ddof=1)
        if name in ["LS","EN","FFN","GAN–LSTM original"]:
            source_model="GAN" if name=="GAN–LSTM original" else name
            row=original_summary[(original_summary.model==source_model)&(original_summary.split==split)]
        else:
            source_model="gan_lstm_matched" if name=="GAN–LSTM matched" else "gan_transformer"
            row=saved[(saved.model==source_model)&(saved.split==split)]
        assert len(row)==1 and int(row.iloc[0].months)==len(r)
        assert np.isclose(monthly,row.iloc[0].monthly_sharpe,rtol=1e-5,atol=1e-8),(name,split)
        all_rows.append({"model":name,"split":split,"months":len(r),"monthly_sharpe":monthly,
            "annualized_sharpe":np.sqrt(12)*monthly,"verified":True})
all_model_performance=pd.DataFrame(all_rows)
all_model_performance.to_csv(OUT / "tables" / "all_model_sharpe_verified.csv",index=False)
ours=all_model_performance.pivot(index="model",columns="split",values="monthly_sharpe").reindex(ALL_ORDER)[["train","validation","test"]]
ours.insert(0,"source","Our saved model returns")
paper_values=pd.read_csv(ROOT / "results" / "tables" / "final_paper_comparison.csv")
paper=paper_values.pivot(index="model",columns="split",values="paper_sharpe").reindex(["LS","EN","FFN","GAN"])[["train","validation","test"]]
paper.index=["LS (paper)","EN (paper)","FFN (paper)","GAN–LSTM (paper)"]
paper.insert(0,"source","Published benchmark recorded in Code 07")
all_sharpe_comparison=pd.concat([ours,paper])
all_sharpe_comparison.to_csv(OUT / "tables" / "all_models_vs_paper_monthly_sharpe.csv",index_label="model")
display(all_sharpe_comparison.round(4))
all_test=all_returns.iloc[300:]
all_test.to_csv(OUT / "factors" / "all_model_test_factor_returns.csv",index_label="date")
# Published benchmark points appear only where the paper reports that model.
fig,ax=plt.subplots(figsize=(12,5))
ax.bar(np.arange(6),ours.test.to_numpy(),label="Our test monthly Sharpe",color=["#64748b"]*4+["#2563eb","#ea580c"])
ax.scatter(np.arange(4),paper.test.to_numpy(),color="black",marker="D",s=55,label="Paper benchmark (original models)",zorder=3)
ax.set_xticks(np.arange(6),ALL_ORDER,rotation=15,ha="right")
ax.set_ylabel("Monthly Sharpe");ax.set_title("1992–2016: all model results and paper benchmarks")
ax.legend();ax.grid(axis="y",alpha=.2)
fig.tight_layout();fig.savefig(OUT / "figures" / "all_models_vs_paper_sharpe.png",dpi=150);plt.show()


# %% [markdown]
# ## 10 — Rolling Sharpe for all six models
# Three panels per window separate the model groups for readability. All panels use annualized Sharpe, the same 300 test months, and complete trailing windows. They contain realized model-return series; published full-period benchmarks are not rolling series.

# %%
all_rolling_parts=[]
for model in ALL_ORDER:
    for window in WINDOWS:
        values=rolling_metrics(all_test[model],window)
        assert values.sharpe_annualized.notna().sum()==300-window+1
        values["model"]=model;values["window_months"]=window
        all_rolling_parts.append(values.reset_index())
all_rolling=pd.concat(all_rolling_parts,ignore_index=True)
all_rolling.to_csv(OUT / "tables" / "all_model_rolling_metrics.csv",index=False)
fig,axes=plt.subplots(2,3,figsize=(17,9),sharex=True,sharey=True)
groups=[["LS","EN"],["FFN","GAN–LSTM original"],["GAN–LSTM matched","GAN–Transformer"]]
for i,window in enumerate(WINDOWS):
    for j,group in enumerate(groups):
        ax=axes[i,j]
        for name in group:
            rows=all_rolling[(all_rolling.model==name)&(all_rolling.window_months==window)]
            ax.plot(rows.date,rows.sharpe_annualized,label=name)
        ax.set_title(f"{window}-month rolling annualized Sharpe")
        ax.axhline(0,color="grey",linewidth=.7);ax.legend(fontsize=9);ax.grid(alpha=.2)
    axes[i,0].set_ylabel("Annualized Sharpe")
fig.tight_layout();fig.savefig(OUT / "figures" / "all_model_rolling_sharpe.png",dpi=150);plt.show()
assert len(all_model_performance)==18 and all_model_performance.verified.all()
assert len(all_rolling)==6*2*300
print("All six models verified across all three splits and both rolling windows.")


# %% [markdown]
# ## 11 — Results, answers, and relationship to Code 10
# These conclusions are generated from the executed comparisons.

# %%
perf = full_performance.set_index("model")
w36 = window_comparison.set_index("window_months").loc[36]
w60 = window_comparison.set_index("window_months").loc[60]
summary = f"""### Executed results and answers

**Q1 — Persistence across windows:** The Transformer has higher trailing annualized Sharpe in **{w36.transformer_higher_fraction:.1%}** of the **265** complete 36-month windows and **{w60.transformer_higher_fraction:.1%}** of the **241** complete 60-month windows. Mean annualized Sharpe differences (Transformer minus matched LSTM) are **{w36.mean_sharpe_difference:.3f}** and **{w60.mean_sharpe_difference:.3f}**. Full-test monthly Sharpe is **{perf.loc['gan_transformer','monthly_sharpe']:.4f}** versus **{perf.loc['gan_lstm_matched','monthly_sharpe']:.4f}**. The window comparison table identifies positive and negative periods and the dates of the largest differences; overlapping-window counts are descriptive.

**Q2 — Return and risk:** LSTM terminal wealth is **{perf.loc['gan_lstm_matched','terminal_wealth']:.3f}**, with annualized volatility **{perf.loc['gan_lstm_matched','annualized_volatility']:.2%}** and maximum drawdown **{perf.loc['gan_lstm_matched','maximum_drawdown']:.2%}**. Transformer terminal wealth is **{perf.loc['gan_transformer','terminal_wealth']:.3f}**, with annualized volatility **{perf.loc['gan_transformer','annualized_volatility']:.2%}** and maximum drawdown **{perf.loc['gan_transformer','maximum_drawdown']:.2%}**. These are hypothetical compounded gross strategy results under the stated assumptions. Sharpe superiority should be read alongside volatility, drawdown, and the rolling risk figures.

**Q3 — Co-movement:** Full-test return correlation is **{full_correlation:.4f}**. The 36-month correlation ranges from **{correlation_summary.iloc[0].minimum_rolling_correlation:.3f}** to **{correlation_summary.iloc[0].maximum_rolling_correlation:.3f}**; the 60-month range is **{correlation_summary.iloc[1].minimum_rolling_correlation:.3f}** to **{correlation_summary.iloc[1].maximum_rolling_correlation:.3f}**. The models share information but their exposure and relative performance vary over time. Correlation alone does not establish diversification benefits after costs.

**Q4 — Meaning and next code:** Code 08 established the fixed-budget architecture comparison. Code 09 describes its evolution through time without retraining, test-based tuning, or claiming independent statistical significance. Code **10** will read `results/rolling_oos/factors/test_factor_returns.csv` and join monthly recession/expansion and high/low volatility labels. It should compare the same matched pair, retain the exact 300-month test calendar, report regime sample sizes, and keep the larger-budget Code 07 reference separately labelled.

For descriptive recession analysis, use NBER labels and state that they are retrospective. For prospective volatility labels, use lagged trailing market volatility and a threshold calibrated on training data, rather than a test-sample median. Code 10 should use monthly market returns from the cached Fama–French data, not either model's own factor volatility to define market regimes. Code 09's rolling diagnostics help identify time variation; Code 10 will investigate whether that variation is associated with economic states. Neither code should choose a model using selected favourable test periods.

Saved artifacts include all trailing metrics, Sharpe differences, full-test performance, wealth and drawdown paths, correlations, four figures, input hashes, and the dated test-return handoff file.
"""
display(Markdown(summary))
(OUT / "results_and_answers.txt").write_text(summary)
print("CODE 09 COMPLETE: executed results, answers, tables, figures, and Code 10 inputs saved.")

all_summary = "\n\n### All-model Sharpe comparison (monthly)\n" + all_sharpe_comparison.to_string(float_format=lambda value: f"{value:.4f}") + "\n\nAll 18 saved model/split Sharpe ratios were verified against monthly returns. The paper benchmarks are recorded values verified against the authors' repository in the accompanying review. The Transformer exceeds the matched LSTM, but the original replicated GAN and FFN have higher full-test Sharpe. Different budgets prevent interpreting the original-versus-Transformer comparison as an isolated architecture test. Paper rolling performance is unavailable without the original monthly factor series for those models.\n"
display(Markdown("### All-model check complete\nAll six models now appear in the full-period comparison and rolling Sharpe charts. See the table in Part 9."))
(OUT / "results_and_answers.txt").write_text(summary + all_summary)


# %% [markdown]
# ### Executed results and answers
# 
# **Q1 — Persistence across windows:** The Transformer has higher trailing annualized Sharpe in **60.4%** of the **265** complete 36-month windows and **70.1%** of the **241** complete 60-month windows. Mean annualized Sharpe differences (Transformer minus matched LSTM) are **0.519** and **0.465**. Full-test monthly Sharpe is **0.4414** versus **0.3113**. The window comparison table identifies positive and negative periods and the dates of the largest differences; overlapping-window counts are descriptive.
# 
# **Q2 — Return and risk:** LSTM terminal wealth is **3.739**, with annualized volatility **5.02%** and maximum drawdown **-16.10%**. Transformer terminal wealth is **6.046**, with annualized volatility **4.80%** and maximum drawdown **-14.20%**. These are hypothetical compounded gross strategy results under the stated assumptions. Sharpe superiority should be read alongside volatility, drawdown, and the rolling risk figures.
# 
# **Q3 — Co-movement:** Full-test return correlation is **0.5799**. The 36-month correlation ranges from **-0.311** to **0.809**; the 60-month range is **0.044** to **0.760**. The models share information but their exposure and relative performance vary over time. Correlation alone does not establish diversification benefits after costs.
# 
# **Q4 — Meaning and next code:** Code 08 established the fixed-budget architecture comparison. Code 09 describes its evolution through time without retraining, test-based tuning, or claiming independent statistical significance. Code **10** will read `results/rolling_oos/factors/test_factor_returns.csv` and join monthly recession/expansion and high/low volatility labels. It should compare the same matched pair, retain the exact 300-month test calendar, report regime sample sizes, and keep the larger-budget Code 07 reference separately labelled.
# 
# For descriptive recession analysis, use NBER labels and state that they are retrospective. For prospective volatility labels, use lagged trailing market volatility and a threshold calibrated on training data, rather than a test-sample median. Code 10 should use monthly market returns from the cached Fama–French data, not either model's own factor volatility to define market regimes. Code 09's rolling diagnostics help identify time variation; Code 10 will investigate whether that variation is associated with economic states. Neither code should choose a model using selected favourable test periods.
# 
# Saved artifacts include all trailing metrics, Sharpe differences, full-test performance, wealth and drawdown paths, correlations, four figures, input hashes, and the dated test-return handoff file.
# 
# 
# ### Verified monthly Sharpe ratios
# 
# | Model | Train | Validation | Test |
# |---|---:|---:|---:|
# | LS | 2.0248 | 0.7631 | 0.4057 |
# | EN | 1.1231 | 1.0292 | 0.4108 |
# | FFN | 0.6583 | 0.8448 | 0.5686 |
# | GAN–LSTM original | 2.2189 | 1.1755 | 0.6109 |
# | GAN–LSTM matched | 1.2083 | 0.4197 | 0.3113 |
# | GAN–Transformer | 1.2837 | 0.6515 | 0.4414 |
# | LS (paper) | 1.8000 | 0.5800 | 0.4200 |
# | EN (paper) | 1.3700 | 1.1500 | 0.5000 |
# | FFN (paper) | 0.4500 | 0.4200 | 0.4400 |
# | GAN–LSTM (paper) | 2.6800 | 1.4300 | 0.7500 |
# 
# All six model series passed verification. The original replicated GAN and FFN exceed the Transformer on full-test Sharpe; the Transformer exceeds the matched-budget LSTM.
