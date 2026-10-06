# Code 012 — Results and answers

No models were retrained. This is return-sample uncertainty for fixed saved portfolios.

## Q1. Individual Sharpe uncertainty

- **Original-paper GAN:** monthly Sharpe 0.750; 95% basic interval [0.442, 0.973].
- **Our replicated GAN:** monthly Sharpe 0.612; 95% basic interval [0.380, 0.783].
- **Extension LSTM:** monthly Sharpe 0.459; 95% basic interval [0.244, 0.626].
- **Transformer:** monthly Sharpe 0.304; 95% basic interval [0.146, 0.439].
- **Equal-weight LSTM–Transformer combination:** monthly Sharpe 0.463; 95% basic interval [0.265, 0.612].

## Q2. Paired comparisons and multiplicity

- **Original-paper GAN minus Our replicated GAN:** 0.138; pointwise basic interval [-0.037, 0.256]; simultaneous interval [-0.148, 0.425] contains zero.
- **Extension LSTM minus Transformer:** 0.155; pointwise basic interval [-0.049, 0.365]; simultaneous interval [-0.132, 0.441] contains zero.
- **Extension LSTM minus Equal-weight LSTM–Transformer combination:** -0.004; pointwise basic interval [-0.132, 0.119]; simultaneous interval [-0.290, 0.283] contains zero.

## Q3. Block-length sensitivity

- Extension LSTM minus Transformer: pointwise basic intervals exclude zero for 0 of 3 block choices; simultaneous intervals exclude zero for 0 of 3. These choices are sensitivity analyses, not independent confirmations.
- Extension LSTM minus Equal-weight LSTM–Transformer combination: pointwise basic intervals exclude zero for 0 of 3 block choices; simultaneous intervals exclude zero for 0 of 3. These choices are sensitivity analyses, not independent confirmations.

## Q4. Blend interpretation
Read the Extension LSTM minus Equal-weight LSTM–Transformer combination interval above. The observed blend improvement is about 0.004 monthly Sharpe. An interval containing zero leaves its direction unresolved; it does not prove equivalence.

## Relationship to Codes 07–11
Code 07 supplies the replication benchmark. Code 08 supplies the fixed new architecture ensembles. Codes 09–10 describe time and state variation. Code 11 supplies the consolidated author, replication, architecture, and blend returns. Code 012 adds approximate sampling uncertainty for full-test Sharpe, without changing those models or rerunning rolling/regime analyses.

## Limits
These intervals condition on saved models and the existing dataset. They do not correct for training uncertainty, prior test inspection, transaction costs, structural changes, or mismatched training budgets. Stationarity and block-length assumptions matter. No bootstrap sign fraction is reported as a p-value.

## Additional interval-method comparisons (12-month blocks)

- **Original-paper GAN minus Our replicated GAN:** basic: Unresolved; percentile: A higher; simultaneous: Unresolved.
- **Original-paper GAN minus Extension LSTM:** basic: Unresolved; percentile: A higher; simultaneous: A higher.
- **Original-paper GAN minus Transformer:** basic: A higher; percentile: A higher; simultaneous: A higher.
- **Original-paper GAN minus Equal-weight LSTM–Transformer combination:** basic: A higher; percentile: A higher; simultaneous: A higher.
- **Our replicated GAN minus Extension LSTM:** basic: Unresolved; percentile: Unresolved; simultaneous: Unresolved.
- **Our replicated GAN minus Transformer:** basic: A higher; percentile: A higher; simultaneous: A higher.
- **Our replicated GAN minus Equal-weight LSTM–Transformer combination:** basic: A higher; percentile: Unresolved; simultaneous: Unresolved.
- **Extension LSTM minus Transformer:** basic: Unresolved; percentile: Unresolved; simultaneous: Unresolved.
- **Extension LSTM minus Equal-weight LSTM–Transformer combination:** basic: Unresolved; percentile: Unresolved; simultaneous: Unresolved.
- **Transformer minus Equal-weight LSTM–Transformer combination:** basic: B higher; percentile: B higher; simultaneous: Unresolved.

## Additional comparisons

Tables include all ten pairs across three block lengths, IID-versus-block diagnostics, monthly-versus-annualized intervals, and model precision sensitivity. IID is a dependence diagnostic only. Method disagreement must be reported rather than resolved by choosing the favorable interval. Annualization leaves zero-crossing conclusions unchanged.

## Rolling uncertainty — results and interpretation

- All 265 complete 36-month windows and 241 complete 60-month windows are analyzed without retraining.
- Figures show pointwise basic intervals. Simultaneous table intervals adjust for ten pairs within each window, not across dates.
- **36 months, LSTM minus Transformer:** LSTM has higher observed Sharpe in 168/265 windows. Basic intervals favor LSTM in 39, favor Transformer in 0, and leave 226 unresolved. These overlapping counts are descriptive, not independent evidence.
- **60 months, LSTM minus Transformer:** LSTM has higher observed Sharpe in 193/241 windows. Basic intervals favor LSTM in 45, favor Transformer in 0, and leave 196 unresolved. These overlapping counts are descriptive, not independent evidence.
- Larger or smaller point estimates should be interpreted with their window-specific intervals. Short windows have limited information; apparent performance changes need not be precisely estimated.
- Three-, six-, and twelve-month block comparisons at December endpoints assess local sensitivity. They do not select a model or define a trading signal.
- Labels now distinguish the original-paper GAN, our replicated GAN, the extension architectures, and the equal-weight LSTM–Transformer combination.