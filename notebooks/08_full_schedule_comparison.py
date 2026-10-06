# %% [markdown]
# # 08 Full Schedule — Author-Conventions GAN: LSTM versus Transformer
# 
# ## Process
# 1. Validate the same author firm panels, 178 macro inputs, and train/validation/test calendars.
# 2. Verify causal macro states, bounded conditional instruments, weighted pricing moments, and portfolio normalization.
# 3. Train nine seeds (42–50) for each architecture using the published configuration: 256 unconditional epochs, 64 adversary steps per pass, and 1024 conditional epochs, with four full-data optimizer passes and Adam at 0.001.
# 4. Follow the authors’ validation-loss / adversary-loss / validation-Sharpe checkpoint rules. Select nothing using test results.
# 5. Average raw member weights before monthly gross normalization; report monthly Sharpe with population standard deviation, as in the authors’ utility.
# 6. Save all results and compare both new ensembles with the paper, historical replication, and shorter experiment. Codes 09–10 will use these new factors.
# 
# ## Questions
# **Q1.** Do both models satisfy causal-state and pricing-objective checks?
# 
# **Q2.** How do train, validation, and test Sharpe compare under the original training schedule and checkpoint rules?
# 
# **Q3.** Are the results stable across nine initializations, and how similar are the two factors?
# 
# **Q4.** How do the results compare with the paper and our earlier replication, and connect to rolling and regime analysis?
# 
# ## Verified sources and scope
# [Paper, Table I and Appendix C](https://arxiv.org/pdf/1904.00745) and [authors’ training repository](https://github.com/LouisChen1992/Deep_Learning_Asset_Pricing), commit `6c26b9dad01e76b214ab8f5566c42a29e99677c9`. Checked files: `config/config.json`, `run.py`, `src/model/model_GAN.py`, `src/data/data_layer.py`, and `src/utils.py`.
# 
# The LSTM implementation is a PyTorch port of the authors’ chosen configuration. The Transformer replaces both recurrent encoders while preserving the rest of the verified training/evaluation conventions. Transformer internal attention layers are our extension. Codes 09–10 are also extension analyses; the paper does not contain this Transformer experiment.
# 
# Corrections from the earlier replication: `tanh`-bounded instruments; precision weights normalized by maximum observation count and averaging across instruments; macro-input dropout; four optimizer passes; validation-selected checkpoints; raw-weight ensemble averaging; and population-standard-deviation Sharpe. The sign representation uses economic portfolio weights, so `M = 1 − F`, algebraically equivalent to the authors’ `M = 1 + ΣwR` with economic weights `−w`.
# 
# This reproduces the selected published specification, **not the paper’s entire 384-configuration hyperparameter search**. The framework/runtime and random draws differ from TensorFlow 1.12/Python 3.6, so it is not a bit-for-bit reproduction. Default `ignoreEpoch=64` is used from the authors’ executable; their README also illustrates a 32-epoch setting, which is not used here. The ambiguous Stage 2 pre-update scoring/post-update saving order is retained explicitly.
# 
# Test results from prior experiments have already been viewed. This is a transparent follow-up comparison, not a new independent holdout. Historical outputs remain preserved and labelled. Causality verifies computation, not vintage-safe macro releases. No fees, financing costs, or additional risk-free subtraction are introduced.

# %% [markdown]
# ## 1 — Environment, paths, and fixed configuration

# %%
from pathlib import Path
import sys, json, time, hashlib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

def find_root():
    for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (p / "notebooks" / "06_GAN.ipynb").exists() and (p / "results").exists():
            return p
    raise FileNotFoundError("Start Jupyter inside the replication repository.")

ROOT = find_root()
sys.path.insert(0, str(ROOT / "src" / "models"))
from transformer_gan_paper import (GAN, CausalTransformerEncoder, MacroLSTMEncoder,
    SPLITS, load_panel, pricing_loss, factor_and_scores, train_member, evaluate,
    seed_all, stats, fingerprint_files, raw_scores, ensemble_from_raw_scores)

CONFIG = {"seeds": list(range(42, 51)), "epochs": [256, 64, 1024], "learning_rate": 0.001,
    "sdf_hidden": [64, 64], "sdf_macro_dim": 4, "conditional_macro_dim": 32,
    "macro_features": 178, "firm_features": 46, "instruments": 8,
    "dropout": 0.05, "transformer_d_model": 32, "transformer_heads": 4,
    "transformer_layers": 2, "transformer_feedforward": 64,
    "sub_epoch": 4, "ignore_epoch": 64, "adversary_selection": "author_training_score_post_update_checkpoint_final_pass", "sdf_selection": "highest_validation_raw_sharpe_after_ignore_epoch", "stage1_selection": "lowest_validation_unconditional_loss_after_ignore_epoch", "sharpe_ddof": 0, "ensemble": "average_raw_weights_then_gross_normalize", "device": "cpu"}
# CPU avoids device-dependent scatter behavior; all seeds use the same device.
DEVICE = torch.device(CONFIG["device"])
torch.set_num_threads(6)
EXPERIMENT = ROOT / "results" / "paper_matched_full_schedule"
for name in ["models", "tables", "factors", "figures"]:
    (EXPERIMENT / name).mkdir(parents=True, exist_ok=True)
DATES = pd.date_range("1967-01-01", "2016-12-01", freq="MS")
LABELS = np.array(["train"]*240 + ["validation"]*60 + ["test"]*300)
print("Repository:", ROOT)
print("Experiment:", EXPERIMENT)
print(json.dumps(CONFIG, indent=2))


# %% [markdown]
# ## 2 — Load and validate the existing replication inputs
# The author datasets remain in their original local folder. Set `DATASETS_ROOT` explicitly if you move them. Only active slots are compressed; each observation retains its original persistent asset-slot index.

# %%
DATASETS_ROOT = None
candidates = [ROOT / "datasets", ROOT.parent / "datasets", ROOT.parent / "project " / "datasets"]
if DATASETS_ROOT is None:
    DATASETS_ROOT = next((p for p in candidates if (p / "char" / "Char_train.npz").exists()), None)
if DATASETS_ROOT is None:
    raise FileNotFoundError("Set DATASETS_ROOT to the author datasets folder containing char/Char_train.npz.")
MACRO_PATH = ROOT / "results" / "macro" / "macro_lstm_input.parquet"
macro_df = pd.read_parquet(MACRO_PATH).sort_values("date").reset_index(drop=True)
macro_df["date"] = pd.to_datetime(macro_df["date"]).dt.to_period("M").dt.to_timestamp()
macro_cols = [x for x in macro_df.columns if x != "date"]
assert len(macro_cols) == 178
assert pd.DatetimeIndex(macro_df["date"]).equals(DATES)
macro_np = macro_df[macro_cols].to_numpy(dtype=np.float32)
assert np.isfinite(macro_np).all()
# Code 05 already standardized with training statistics; do not standardize twice.
assert np.allclose(macro_np[:240].mean(0), 0, atol=2e-5)
assert np.allclose(macro_np[:240].std(0, ddof=0), 1, atol=2e-5)
macro = torch.tensor(macro_np, device=DEVICE).unsqueeze(0)
filenames = {"train": "Char_train.npz", "validation": "Char_valid.npz", "test": "Char_test.npz"}
panels = {}
for split, (start, end) in SPLITS.items():
    panels[split] = load_panel(DATASETS_ROOT / "char" / filenames[split], DATES[start:end], DEVICE)
assert panels["train"]["names"] == panels["validation"]["names"] == panels["test"]["names"]
inventory = pd.DataFrame([{"split": s, "months": p["T"], "slots": p["N"],
    "active_stock_months": len(p["r"]), "start": p["dates"].min(), "end": p["dates"].max()} for s,p in panels.items()])
display(inventory)
inventory.to_csv(EXPERIMENT / "tables" / "input_inventory.csv", index=False)
fingerprint = fingerprint_files([MACRO_PATH, *[DATASETS_ROOT / "char" / f for f in filenames.values()],
    ROOT / "src" / "models" / "transformer_gan_paper.py"])
manifest = {"config": CONFIG, "inputs": fingerprint, "python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__}
(EXPERIMENT / "manifest.json").write_text(json.dumps(manifest, indent=2))


# %% [markdown]
# ## 3 — Architecture and meaningful integrity tests
# Future-value perturbations and prefix checks run in evaluation mode so dropout cannot obscure causality. A small unbalanced panel verifies that vectorized moments equal explicit per-asset averages, including an inactive slot.

# %%
seed_all(42)
checks = []
for kind in ["LSTM", "Transformer"]:
    model = GAN(kind).to(DEVICE).eval()
    print(kind, "parameters:", sum(p.numel() for p in model.parameters()))
    for name, dim in [("sdf_macro_encoder", 4), ("cond_macro_encoder", 32)]:
        encoder = getattr(model, name)
        z = macro.clone(); z[:, 300:] += 100*torch.randn_like(z[:, 300:])
        with torch.no_grad():
            original = encoder(macro); changed = encoder(z); prefix = encoder(macro[:, :300])
        assert original.shape == (1,600,dim)
        torch.testing.assert_close(original[:, :300], changed[:, :300], rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(original[:, :300], prefix, rtol=1e-5, atol=2e-6)
        checks.append({"architecture": kind, "check": name + " causal/prefix", "passed": True})
    # Full objective equivalence on an unbalanced toy panel with original slot IDs.
    toy = {"T": 3, "N": 4, "x": torch.randn(5,46)*.1,
        "r": torch.tensor([.02,-.01,.03,-.04,.01]), "month": torch.tensor([0,0,1,2,2]),
        "slot": torch.tensor([0,2,0,1,2]), "counts": torch.tensor([2.,1.,2.,0.])}
    for conditional in [False,True]:
        sm = model.sdf_macro_encoder(macro[:,:3])[0]
        cm = model.cond_macro_encoder(macro[:,:3])[0]
        raw, gross, w = factor_and_scores(model,toy,sm)
        g = model.conditional_network(toy["x"],cm[toy["month"]]) if conditional else torch.ones(5,1)
        values = ((1-raw[toy["month"]])*toy["r"])[:,None]*g
        reference = torch.stack([(toy["counts"][j]/toy["counts"].max())*(values[toy["slot"]==j].mean(0).square().mean()) for j in [0,1,2]]).mean()
        actual = pricing_loss(model,toy,macro[:,:3],conditional)
        torch.testing.assert_close(actual, reference)
        model.zero_grad(set_to_none=True); actual.backward()
        for component in ([model.sdf_macro_encoder,model.sdf_network,model.cond_macro_encoder,model.conditional_network] if conditional else [model.sdf_macro_encoder,model.sdf_network]):
            assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in component.parameters())
        normalized_gross = torch.zeros(3).index_add(0,toy["month"],(w/gross[toy["month"]]).abs())
        torch.testing.assert_close(normalized_gross, torch.ones(3))
        checks.append({"architecture": kind, "check": "conditional" if conditional else "unconditional", "passed": True})
integrity = pd.DataFrame(checks)
display(integrity)
integrity.to_csv(EXPERIMENT / "tables" / "architecture_integrity.csv", index=False)
del model
print("All causality, prefix, moment, gradient, and gross-normalization tests passed.")


# %% [markdown]
# ## 4 — Joint GAN estimation with the authors’ checkpoint rules
# 
# Each SDF epoch makes four full training passes. Stage 1 restores the lowest validation unconditional-loss checkpoint; Stage 2 executes four 64-step adversary passes and restores the last pass’s strongest training-loss checkpoint; Stage 3 restores the highest validation raw-factor Sharpe checkpoint. Selection starts after the author code’s default zero-based epoch 64. Both architectures share these rules and never select on test results.
# 
# The author Stage 2 code scores before an optimizer update and saves after it; this ordering is retained and documented. Shared macro-input dropout is active during training even for a frozen network. Checkpoint reuse requires matching hashes, settings, and source. Runtime and random initializations differ from TensorFlow 1.12, so exact numerical equality is not guaranteed.

# %%
members = {}; history_parts = []; member_rows = []; ensemble_factors = {}
started = time.monotonic()
for kind in ["LSTM", "Transformer"]:
    factors = []
    score_sums = {split: np.zeros(len(panel["r"]), dtype=np.float64) for split,panel in panels.items()}
    for seed in CONFIG["seeds"]:
        model, history = train_member(kind, seed, panels["train"], macro[:,:240],
            CONFIG, EXPERIMENT / "models", fingerprint, validation_panels=panels, full_macro=macro)
        factor = evaluate(model, panels, macro)
        assert factor.shape == (600,) and np.isfinite(factor).all()
        # Serialized checkpoints must reproduce evaluated returns.
        restored = GAN(kind).to(DEVICE)
        saved = torch.load(EXPERIMENT / "models" / f"{kind.lower()}_seed_{seed}.pt", map_location=DEVICE, weights_only=False)
        restored.load_state_dict(saved["state_dict"])
        np.testing.assert_allclose(evaluate(restored, panels, macro), factor, rtol=1e-5, atol=1e-7)
        for split, scores in raw_scores(model, panels, macro).items():
            score_sums[split] += scores
        factors.append(factor); history_parts.append(history)
        for split,(a,b) in SPLITS.items():
            member_rows.append({"architecture": kind, "seed": seed, "split": split, **stats(factor[a:b])})
        pd.DataFrame({"date": DATES, "split": LABELS, "factor_return": factor}).to_csv(
            EXPERIMENT / "factors" / f"{kind.lower()}_seed_{seed}.csv", index=False)
        del model, restored
    members[kind] = np.column_stack(factors)
    ensemble_factors[kind] = ensemble_from_raw_scores(score_sums,panels,len(CONFIG["seeds"]))
member_performance = pd.DataFrame(member_rows)
history = pd.concat(history_parts, ignore_index=True)
member_performance.to_csv(EXPERIMENT / "tables" / "member_performance.csv", index=False)
history.to_csv(EXPERIMENT / "tables" / "training_history.csv", index=False)
print(f"All eighteen GAN models completed in {(time.monotonic()-started)/60:.1f} minutes.")
display(member_performance)


# %% [markdown]
# ## 5 — Ensemble performance and comparison
# 
# Average the nine **raw stock-weight vectors**, then normalize the mean vector to unit gross exposure each month, matching the authors’ ensemble implementation. This differs from averaging already-normalized member factor returns. Primary Sharpe uses the authors’ population standard deviation (`ddof=0`). The prior replication and shorter experiment are historical references, not independently verified replicas of every author-code convention.

# %%
ensembles = ensemble_factors
reference = pd.read_csv(ROOT / "results" / "factors" / "gan_ensemble_sdf.csv")
reference["date"] = pd.to_datetime(reference["date"]).dt.to_period("M").dt.to_timestamp()
reference = reference.sort_values("date")
assert pd.DatetimeIndex(reference["date"]).equals(DATES)
comparison = pd.DataFrame({"date": DATES, "split": LABELS,
    "gan_lstm_matched": ensembles["LSTM"], "gan_transformer": ensembles["Transformer"],
    "gan_lstm_code07_reference": reference["gan_ensemble_sdf"].to_numpy()})
performance_rows = []
for name in ["gan_lstm_matched", "gan_transformer", "gan_lstm_code07_reference"]:
    for split,(a,b) in SPLITS.items():
        performance_rows.append({"model": name, "split": split, **stats(comparison[name].iloc[a:b])})
performance = pd.DataFrame(performance_rows)
display(performance)
performance.to_csv(EXPERIMENT / "tables" / "ensemble_performance.csv", index=False)
comparison.to_csv(EXPERIMENT / "factors" / "transformer_lstm_comparison.csv", index=False)
pd.DataFrame({"date": DATES, "split": LABELS, "gan_transformer_sdf": ensembles["Transformer"]}).to_csv(
    ROOT / "results" / "factors" / "gan_transformer_full_schedule_sdf.csv", index=False)
correlations = pd.DataFrame([{ "split": split, "lstm_transformer_correlation":
    comparison.iloc[a:b][["gan_lstm_matched","gan_transformer"]].corr().iloc[0,1] } for split,(a,b) in SPLITS.items()])
display(correlations)
correlations.to_csv(EXPERIMENT / "tables" / "factor_correlations.csv", index=False)
stability = member_performance.groupby(["architecture","split"])["monthly_sharpe"].agg(["mean","std","min","max"]).reset_index()
display(stability)
stability.to_csv(EXPERIMENT / "tables" / "seed_stability.csv", index=False)


# %% [markdown]
# ## 6 — Training and test-period figures
# Cumulative sums below are diagnostic cumulative factor returns, not compounded wealth. Codes 09–10 will examine rolling risk and regime dependence.

# %%
fig, axes = plt.subplots(1,3,figsize=(15,4))
for stage,ax in enumerate(axes,1):
    for kind in ["LSTM","Transformer"]:
        mean = history[(history.stage==stage)&(history.architecture==kind)].groupby("epoch")["loss"].mean()
        ax.plot(mean.index,mean,label=kind)
    ax.set_title(f"Stage {stage} — mean training loss"); ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.legend()
fig.tight_layout(); fig.savefig(EXPERIMENT / "figures" / "training_losses.png", dpi=150); plt.show()
fig,axes = plt.subplots(1,2,figsize=(13,4))
performance[performance.split=="test"].set_index("model")["monthly_sharpe"].plot.bar(ax=axes[0],rot=15)
axes[0].set_title("Test monthly Sharpe (1992–2016)")
test = comparison[comparison.split=="test"].set_index("date")
test[["gan_lstm_matched","gan_transformer"]].cumsum().plot(ax=axes[1])
axes[1].set_title("Cumulative test factor returns (sum)")
fig.tight_layout();fig.savefig(EXPERIMENT / "figures" / "test_comparison.png",dpi=150);plt.show()


# %% [markdown]
# ## 7 — Results, answers, and relationship to Codes 09–10
# The following cell generates the conclusions directly from this run. It does not assume that the Transformer wins.

# %%
test_stats = performance[performance.split=="test"].set_index("model")
lstm_sr = test_stats.loc["gan_lstm_matched","monthly_sharpe"]
transformer_sr = test_stats.loc["gan_transformer","monthly_sharpe"]
difference = transformer_sr-lstm_sr
corr = correlations.loc[correlations.split=="test","lstm_transformer_correlation"].iloc[0]
seed_ranges = stability[stability.split=="test"].set_index("architecture")
direction = "higher" if difference>0 else "lower" if difference<0 else "equal"
summary = f"""### Executed full-schedule results
The matched LSTM ensemble has **{lstm_sr:.4f}** test monthly Sharpe; the Transformer ensemble has **{transformer_sr:.4f}**. The Transformer is {direction} by **{abs(difference):.4f}** in this full-schedule matched run. The historical Code 07 reference has **{test_stats.loc['gan_lstm_code07_reference','monthly_sharpe']:.4f}**.

**Q1 — Causal state construction:** Passed. Both Transformer encoders return the required 4/32-dimensional states. Future perturbations do not affect earlier states, and full-sequence outputs agree with prefix-only outputs. The objective, gradient, normalization, calendar, and checkpoint reload checks also passed.

**Q2 — Performance:** The Transformer is {direction} on the untouched 1992–2016 test period for this experiment. This is an observed comparison, not a statistical significance claim or evidence that either architecture always wins. Checkpoint selection used validation loss/Sharpe as documented; no test-based tuning or seed selection was performed.

**Q3 — Stability and similarity:** Test correlation is **{corr:.4f}**. Individual LSTM seed Sharpes range from **{seed_ranges.loc['LSTM','min']:.4f}** to **{seed_ranges.loc['LSTM','max']:.4f}**; Transformer seeds range from **{seed_ranges.loc['Transformer','min']:.4f}** to **{seed_ranges.loc['Transformer','max']:.4f}**. Nine seeds give an initial sensitivity check, not a confidence interval. The saved member table reports every seed.

**Q4 — Interpretation and next codes:** This run isolates the macro encoder architecture under a matched budget and validates the end-to-end pricing pipeline. It uses the full nine-seed training schedule; it does not prove real-time macro-data availability or universal architecture superiority. Code **09** will read `transformer_lstm_comparison.csv`, keep only `split == 'test'`, and compare 36/60-month rolling Sharpe and volatility, cumulative performance, drawdowns, and correlations. Code **10** will use the same test returns for recession/expansion and volatility regimes. Use a training-calibrated volatility threshold and lagged market volatility for prospective labels; NBER recession labels are retrospective descriptive labels. Do not optimize models or regime thresholds on test performance.

The primary pair for Codes 09–10 is `gan_lstm_matched` versus `gan_transformer`. The `gan_lstm_code07_reference` column is optional and must retain its historical-reference label. Dates are month starts; returns use decimal units; each model has 240 train, 60 validation, and 300 test observations. Preserve the same return definition and gross-normalization convention across all three notebooks.
"""
display(Markdown(summary))
(EXPERIMENT / "results_and_answers.txt").write_text(summary)
assert len(comparison)==600 and not comparison.date.duplicated().any()
assert np.isfinite(comparison.select_dtypes(include="number").to_numpy()).all()
assert integrity.passed.all()
assert len(member_performance)==54 and len(performance)==9
assert len(list((EXPERIMENT / "models").glob("*.pt"))) == 18
print("CODE 08 COMPLETE: trained models, executed answers, tables, figures, and monthly factors saved.")

# Include the published GAN benchmark and the shorter experiment in the final table.
short = pd.read_csv(ROOT / "results" / "transformer_extension" / "tables" / "ensemble_performance.csv")
# Put historical Sharpe on the same ddof=0 scale without replacing old saved tables.
short["monthly_sharpe"] *= np.sqrt(short["months"]/(short["months"]-1))
comparison_rows = []
for label, frame, name in [
    ("Full schedule LSTM", performance, "gan_lstm_matched"),
    ("Full schedule Transformer", performance, "gan_transformer"),
    ("Historical Code 07 GAN–LSTM", performance, "gan_lstm_code07_reference"),
    ("Short schedule LSTM", short, "gan_lstm_matched"),
    ("Short schedule Transformer", short, "gan_transformer")]:
    selected = frame[frame.model==name].set_index("split")["monthly_sharpe"]
    comparison_rows.append({"model": label, **selected.to_dict()})
comparison_rows.append({"model":"Paper GAN–LSTM","train":2.68,"validation":1.43,"test":.75})
full_comparison = pd.DataFrame(comparison_rows)[["model","train","validation","test"]]
display(full_comparison)
full_comparison.to_csv(EXPERIMENT / "tables" / "full_short_paper_sharpe_comparison.csv",index=False)
(EXPERIMENT / "results_and_answers.txt").write_text(summary + "\n\nMonthly Sharpe comparison\n" + full_comparison.to_string(index=False))

