# Deep Learning in Asset Pricing: Replication and Extensions

This project replicates the asset-pricing framework of Luyang Chen, Markus Pelger, and Jason Zhu and extends it with Transformer macroeconomic encoders, rolling-window evaluation, economic regime analysis, and architecture combinations.

## Team

Reza Zamani, Hrafnhildur Lif Jonsdottir, Paraj Goyal, and Elouan Bahri

UC Berkeley, Master of Financial Engineering

Professor: Ali Kakhbod

## Original paper and model

[Deep Learning in Asset Pricing](https://arxiv.org/abs/1904.00745) estimates a stochastic discount factor (SDF) using firm characteristics and macroeconomic history. LSTM encoders summarize economic conditions. A feedforward network maps firm characteristics and economic states to portfolio weights. An adversarial network constructs instruments that expose pricing errors, while the pricing network learns to reduce those errors.

The paper appeared in *Management Science*, 70(2), 714–750, DOI [10.1287/mnsc.2023.4695](https://doi.org/10.1287/mnsc.2023.4695). The [authors' implementation](https://github.com/LouisChen1992/Deep_Learning_Asset_Pricing) provides the reference conventions.

## Replication and extension notebooks

| Code | Notebook | Purpose |
|---|---|---|
| 01–06 | Existing notebooks in `notebooks/` | Data preparation, exploration, linear benchmarks, feedforward SDF, LSTM states, and adversarial model |
| 07 | [Replication results](notebooks/07_results_replication.ipynb) | Compare saved models with the original paper, including Sharpe, risk, and characteristic importance |
| 08 | [Matched reduced-schedule comparison](notebooks/08_reduced_schedule_comparison.ipynb) | Train nine LSTM and nine Transformer seeds under shared author-style conventions and a reduced training budget |
| 09 | [Rolling evaluation](notebooks/09_reduced_schedule_oos.ipynb) | Compare saved factors over complete 36- and 60-month windows |
| 10 | [Economic regimes](notebooks/10_regime_analysis.ipynb) | Compare expansion/recession and high/low market volatility |
| 11 | [Consolidated comparisons](notebooks/11_consolidated_results_comparison.ipynb) | Combine paper benchmarks, replication, architecture, seeds, risk, rolling windows, regimes, and a fixed 50/50 blend |

Matching `.py` files provide the notebook code as Python scripts. Earlier exploratory Transformer and rolling notebooks remain available. Full-schedule notebooks are preparation for a longer experiment and do **not** represent a completed full-schedule result. The saved full-schedule notebook retains an interrupted-run output for provenance.

## Current results and interpretation

The chronological samples are training 1967–1986, validation 1987–1991, and test 1992–2016. The table reports **monthly** test Sharpe. Saved local factors use population standard deviation (`ddof=0`), consistent with the authors' utility. Annualized Sharpe equals monthly Sharpe multiplied by the square root of 12.

| Portfolio | Monthly test Sharpe |
|---|---:|
| Original paper GAN, Table I | 0.750 |
| Our historical Code 07 GAN replication | 0.612 |
| New LSTM, reduced schedule | 0.459 |
| New Transformer, reduced schedule | 0.304 |
| Fixed 50/50 architecture blend | 0.463 |

The new matched experiment uses nine seeds per architecture and the shortened **16/4/48** schedule with four optimizer passes and checkpoint warm-up of 4 epochs. The reference selected specification uses **256/64/1024** and warm-up of 64. Consequently, the new experiment is an extension under a reduced budget, not an exact reproduction of the original training schedule or its hyperparameter search. Historical replication and paper results are contextual comparisons with different procedures.

Rolling windows evaluate fixed saved models; they do not retrain the models each month. Regime comparisons are descriptive, with only 26 recession months in the test sample. Overlapping windows and previously examined test results do not establish statistical significance or universal architecture superiority.

## Run and inspect

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook notebooks/11_consolidated_results_comparison.ipynb
```

For the extension pipeline, run Code 08 before Codes 09, 10, and 11. Code 08 retrains models, whereas the later codes consume saved returns and diagnostics. Inspect the reduced-schedule notebooks for the configuration before starting training. Raw datasets and model checkpoints are excluded from Git and must be supplied locally to reproduce training.

## Saved outputs

- `src/models/`: Transformer and matched adversarial-model implementations.
- `scripts/`: helpers for completing and consolidating training pipelines.
- `results/paper_matched_reduced_schedule/`: completed matched architecture experiment.
- `results/rolling_oos_reduced_schedule/`: paper comparisons and rolling evaluations.
- `results/regime_analysis/`: economic regime diagnostics.
- `results/consolidated_comparison/`: consolidated tables, figures, and direct author-factor comparisons.
- `paper/asset_pricing_extensions_draft.tex`: research-paper draft with figures and comparisons.

## Presentations

All three presentations are saved in [`presentation/`](presentation/):

1. [Original-paper presentation](presentation/01_Original_Paper.pptx)
2. [Our replication presentation](presentation/02_Our_Replication.pptx)
3. [Combined replication and extensions, Codes 08–11](presentation/03_Combined_Replication_and_Extensions_08_11.pptx)

The combined deck contains 30 slides, with 10 on previous work and 20 on new results. It includes the team and professor, editable charts and tables, and model explanations. A [PDF for review](presentation/03_Combined_Replication_and_Extensions_08_11.pdf) accompanies the editable PowerPoint.
