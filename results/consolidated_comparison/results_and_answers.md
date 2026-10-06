## Results and answers

**Q1 — Paper:** Best local test model is **GAN–LSTM original**, monthly Sharpe **0.612**, compared with the paper GAN benchmark **0.750**. The earlier GAN replication achieves **0.612**. Training budgets, framework, and tuning differ, so gaps cannot be assigned to one cause.

**Q2 — Architecture:** New LSTM test Sharpe is **0.459** and Transformer is **0.304**. The difference (Transformer minus LSTM) is **-0.155**. This reduced-budget comparison favors LSTM; it does not predict the ordering after full training.

**Q3 — Stability:** LSTM has higher mean rolling Sharpe for both 36- and 60-month windows, and higher conditional Sharpe in all four regime groups. Seed pairs show initialization sensitivity on the same market history. Overlapping windows and shared market observations are not independent replications. Recession results have only 26 observations.

**Q4 — Explanation:** The return/risk tables separate mean returns from volatility. Ensemble Sharpe is not the average member Sharpe because raw weights are averaged before normalization. Paper full-period Sharpe is not a paper rolling or regime benchmark. Use the three diagnostic sections to explain where differences occur without claiming causality.

**Relationship to preceding codes:** Code 07 supplies the original replication, Code 08 supplies the matched architecture experiment, Code 09 supplies time variation, and Code 10 supplies economic-state associations. Code 11 brings them into one reviewable report. Full training, original-author return comparisons, and transaction-cost analysis require separate experiments.
