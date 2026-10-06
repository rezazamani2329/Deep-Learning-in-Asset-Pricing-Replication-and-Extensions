## Rolling uncertainty — results and interpretation

- All 265 complete 36-month windows and 241 complete 60-month windows are analyzed without retraining.
- Figures show pointwise basic intervals. Simultaneous table intervals adjust for ten pairs within each window, not across dates.
- **36 months, LSTM minus Transformer:** LSTM has higher observed Sharpe in 168/265 windows. Basic intervals favor LSTM in 39, favor Transformer in 0, and leave 226 unresolved. These overlapping counts are descriptive, not independent evidence.
- **60 months, LSTM minus Transformer:** LSTM has higher observed Sharpe in 193/241 windows. Basic intervals favor LSTM in 45, favor Transformer in 0, and leave 196 unresolved. These overlapping counts are descriptive, not independent evidence.
- Larger or smaller point estimates should be interpreted with their window-specific intervals. Short windows have limited information; apparent performance changes need not be precisely estimated.
- Three-, six-, and twelve-month block comparisons at December endpoints assess local sensitivity. They do not select a model or define a trading signal.
- Labels now distinguish the original-paper GAN, our replicated GAN, the extension architectures, and the equal-weight LSTM–Transformer combination.