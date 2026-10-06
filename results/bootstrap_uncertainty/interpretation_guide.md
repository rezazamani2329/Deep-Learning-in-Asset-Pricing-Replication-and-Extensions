### Part 1 interpretation — What is being compared?

**The statistic.** Monthly Sharpe is the sample mean monthly excess return divided by its monthly standard deviation. It measures average excess return relative to variability; it is not a percentage return. For example, LSTM Sharpe 0.459 does not mean a 45.9% return. We use `ddof=0` to match the preceding comparisons.

**Read the first table.** Each row is a fixed saved portfolio evaluated on exactly the same 300 months. Original-paper GAN is the authors' actual monthly series; Code 07 is our historical replication; Extension LSTM and Transformer are the matched reduced-budget ensembles. The blend is the average of the two new portfolio returns. It is not the average of their Sharpe ratios.

**Why the checks matter.** Calendar equality prevents comparing different market periods. The blend assertion checks its construction. Reproducing the previously reported Sharpes checks that this notebook uses the intended data. These checks cannot establish that all models had identical training procedures.

**Answer:** The observed ranking is Original-paper GAN, Our replicated GAN, blend, Extension LSTM, Transformer. This is a sample ranking. The later intervals ask how uncertain its pairwise differences are.

### Part 2 interpretation — Why resample shared blocks?

**One replicate.** With 12-month blocks, we draw 25 starting months, take 12 consecutive observations from each, and concatenate them into a 300-month synthetic sample. A block beginning near December 2016 wraps to January 1992. This wrap is a bootstrap convention, not an economically observed transition.

**Why blocks?** Monthly returns can exhibit dependence over time. Independent-month sampling discards that dependence. Blocks retain it locally, although random block boundaries break longer relationships. Six, twelve, and twenty-four months test sensitivity to this choice; none is established here as optimal.

**Why pairing?** Both portfolios must experience the same sampled market months. Otherwise a comparison introduces artificial uncertainty from different market histories. Pairing preserves contemporaneous return relationships. The identical-column test confirms that identical strategies have exactly zero differences in every replicate.

**What 10,000 means.** These are resampling experiments, not new historical observations, model fits, or independent trading strategies. More replicates reduce simulation noise but cannot create more economic information than the original sample.

**Answer:** The bootstrap approximates sampling variability conditional on the observed return history and the fixed models. It does not capture the full model-development process.

### Part 3 interpretation — How to read confidence intervals

**Individual intervals.** For Extension LSTM, the monthly Sharpe estimate is 0.459 and the primary basic interval is [0.244, 0.626]. It describes uncertainty in its Sharpe under this resampling procedure. It is not a predicted range of next month's returns. The 95% label refers to the approximate repeated-sampling coverage of a procedure under its assumptions, not a 95% posterior probability about a parameter.

**Difference intervals.** Always read A minus B. LSTM minus Transformer is 0.155, with basic interval [−0.049, 0.365]. The interval includes negative values, zero, and positive values: the observed advantage is not resolved at this interval level. Do not infer the difference by inspecting whether two individual model intervals overlap.

**Pointwise versus simultaneous.** A pointwise interval addresses one comparison. Ten pointwise intervals do not collectively have a guaranteed 95% coverage rate. Our approximate simultaneous procedure calibrates the maximum absolute centered error across all ten differences. It addresses that displayed family within one block-length analysis, not all prior research decisions.

**Answer:** A higher observed Sharpe alone is insufficient to establish a clear difference. Crossing zero means unresolved direction; it does not mean the strategies are equal. A zero-excluding interval remains conditional evidence subject to the bootstrap and experimental assumptions.

### Part 4 interpretation — Reading the figures

**Model-interval figure.** Each dot is the observed monthly Sharpe. The horizontal line is the pointwise basic interval. A narrower line indicates greater estimated precision for that statistic, not necessarily a better model.

**Difference figure.** The vertical dashed line marks zero. A segment wholly to the right favors A; wholly to the left favors B. A crossing segment leaves the sign unresolved. The gray segments use the simultaneous calibration, while the dark segments use pointwise basic intervals. Their different construction can shift conclusions.

**Sensitivity figure.** Each row changes block length while retaining the same observed point estimate. The data and observed Sharpe do not change; the estimated uncertainty does. The three results are related sensitivity exercises, not independent confirmations.

**Answer:** LSTM versus Transformer and LSTM versus blend remain unresolved under basic and simultaneous intervals at all three examined block lengths. That supports cautious wording across the selected settings, not a claim that every possible resampling method would agree.

### Part 5 interpretation — Turning numbers into research statements

**Architecture:** “LSTM has higher observed monthly test Sharpe than Transformer under our matched reduced schedule, but the paired bootstrap interval includes zero.” This states both the result and its uncertainty.

**Replication:** Authors minus Code 07 is 0.138, with primary basic interval [−0.037, 0.256]. This interval does not establish equality. Different training conventions, input construction, and pricing diagnostics still matter when assessing replication.

**Blend:** LSTM minus blend is −0.004, with primary basic interval [−0.132, 0.119]. The negative estimate slightly favors the blend, but its interval leaves the direction unresolved. This does not rule out useful diversification; it limits evidence for an improvement in full-test Sharpe.

**Answer:** Code 012 adds uncertainty to the earlier point comparisons. It does not undo their observed rankings, validate new trading rules, or explain the cause of performance differences.

### Part 6 interpretation — Method agreement and disagreement

**Basic interval:** If the observed difference is d and bootstrap quantiles are q_low and q_high, the interval is [2d − q_high, 2d − q_low]. It reflects the bootstrap error around the observed value.

**Percentile interval:** This is [q_low, q_high]. It uses the bootstrap estimates directly. Basic and percentile intervals have the same width for the same draws, but can be centered differently. Therefore their zero-exclusion conclusions can differ.

**Status columns:** “A higher” means the relevant interval lies above zero; “B higher” means it lies below zero; “Unresolved” means it includes zero. These labels summarize approximate interval evidence and are not universal model rankings.

**Actual disagreement:** Authors versus Code 07 excludes zero under percentile intervals at all three block lengths, but under basic intervals at none. Report this sensitivity instead of choosing the favorable method. Authors versus Transformer and Code 07 versus Transformer exclude zero under all three methods across all three block lengths. These latter comparisons are more consistent across the examined settings, although their models have different training procedures.

**Robustness counts:** A count of 3 means all three chosen lengths; it is not three independent statistical tests confirming the hypothesis. A count of 0 means no selected interval excludes zero, not proof of equality.

**Answer:** Method sensitivity is itself a substantive result. Simultaneous intervals are calibrated differently and need not contain every basic or percentile interval.

### Part 7 interpretation — What the IID diagnostic adds

**IID benchmark:** Length-one blocks resample independent months while keeping the five portfolios paired. This removes time dependence, making it a useful diagnostic against the primary block procedure rather than our preferred time-series method.

**Read the width ratio.** A ratio above one means the 12-month basic interval is wider than its IID counterpart. For LSTM minus Transformer, the widths are approximately 0.414 versus 0.388: a ratio of 1.066. For LSTM minus blend, they are 0.250 versus 0.226: a ratio of 1.110. Both comparisons remain unresolved under both procedures.

**Histograms:** These show the distribution of resampled differences. The gold line is the observed difference and the dashed black line is zero. Values on both sides illustrate uncertainty in the direction. The fraction of draws above zero is not automatically a valid p-value or the probability that one model is truly superior.

**Answer:** Dependence treatment affects estimated precision. In these two comparisons it does not change the basic-interval conclusion. That does not prove serial dependence is absent or the IID assumption is valid.

### Part 8 interpretation — Units, standard errors, and precision

**Annualization:** LSTM monthly Sharpe 0.459 becomes approximately 1.591 after multiplication by √12. Its interval endpoints are multiplied by the same number. Positive scaling cannot change whether zero is inside an interval, so annualization cannot strengthen statistical evidence.

**Standard error:** `bootstrap_se` is the standard deviation of the resampled Sharpe estimates. It estimates variability of the statistic, not volatility of monthly returns and not variation across training seeds.

**Interval width:** Upper minus lower measures precision on the stated scale. Basic and percentile widths are equal by construction here; their centers can differ. Comparisons across block lengths assess how the assumed dependence horizon affects uncertainty. Longer blocks need not always create wider intervals.

**Answer:** Distinguish return volatility, uncertainty about Sharpe, and uncertainty across trained models. This notebook measures the second. Conventional annualization here is not an autocorrelation-adjusted long-horizon Sharpe calculation.

### Part 9 interpretation — Scope of the contribution

**What this adds:** A reproducible uncertainty assessment for five fixed portfolios and their ten pairwise Sharpe differences, with explicit dependence, interval-method, and multiplicity comparisons. It helps prevent treating every numerical ranking as an established performance difference.

**What remains conditional:** The original test period has already informed earlier comparisons. Fixed-model resampling does not account for choosing architectures, checkpoints, blend construction, or research directions after examining outcomes. Simultaneous intervals address only the ten displayed differences within a single chosen bootstrap setting.

**Economic versus statistical meaning:** A small difference can be economically interesting but imprecisely estimated. A statistically resolved gross-Sharpe difference need not survive transaction costs or investment constraints. Neither result proves that the model prices every asset correctly; pricing errors require separate diagnostics.

**Answer and next step:** Keep observed performance, uncertainty, and implementation feasibility as separate claims. Transaction-cost work requires reliable holdings and identifiers. Full matched training and a genuinely unexamined evaluation period address different limitations. Code 012 performs neither experiment.


## Rolling uncertainty — results and interpretation

- All 265 complete 36-month windows and 241 complete 60-month windows are analyzed without retraining.
- Figures show pointwise basic intervals. Simultaneous table intervals adjust for ten pairs within each window, not across dates.
- **36 months, LSTM minus Transformer:** LSTM has higher observed Sharpe in 168/265 windows. Basic intervals favor LSTM in 39, favor Transformer in 0, and leave 226 unresolved. These overlapping counts are descriptive, not independent evidence.
- **60 months, LSTM minus Transformer:** LSTM has higher observed Sharpe in 193/241 windows. Basic intervals favor LSTM in 45, favor Transformer in 0, and leave 196 unresolved. These overlapping counts are descriptive, not independent evidence.
- Larger or smaller point estimates should be interpreted with their window-specific intervals. Short windows have limited information; apparent performance changes need not be precisely estimated.
- Three-, six-, and twelve-month block comparisons at December endpoints assess local sensitivity. They do not select a model or define a trading signal.
- Labels now distinguish the original-paper GAN, our replicated GAN, the extension architectures, and the equal-weight LSTM–Transformer combination.
