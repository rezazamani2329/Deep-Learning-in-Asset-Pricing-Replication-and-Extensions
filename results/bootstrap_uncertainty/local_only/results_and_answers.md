## Local-only comparison: results and interpretation

- **Scope:** Seven local portfolios, including linear, Elastic Net, feedforward, our replicated GAN, extension LSTM, Transformer, and their equal-weight combination. No original-paper factor enters this section.
- **Overall monthly test Sharpe ranking:** Our replicated GAN 0.612, Feedforward replication 0.570, Equal-weight LSTM–Transformer combination 0.463, Extension LSTM 0.459, Elastic Net replication 0.412, Linear replication 0.406, Transformer 0.304.
- **Matched architecture question:** Only extension LSTM and Transformer share the new reduced training setup. Historical replication models provide contextual comparisons.
- **Uncertainty:** The simultaneous full-test intervals are recalibrated for all 21 local pairs, rather than filtering the earlier ten-pair family. Basic intervals and six-/twelve-/twenty-four-month sensitivity are saved.
- **Rolling:** All 265 three-year and 241 five-year windows have paired uncertainty estimates using six-month blocks and 2,000 draws. Pointwise bands are not simultaneous across time.
- **Regimes:** Conditional monthly Sharpes use the established business-cycle and lagged-volatility labels. These are point estimates: this section does not provide regime-specific confidence intervals. Recession has only 26 months.
- **Meaning:** Rolling and regimes describe when and where each fixed model performs differently; uncertainty describes precision. They are analyses, not additional trained models.
- **Limits:** No retraining, transaction-cost adjustment, independent new holdout, or correction for prior research choices.