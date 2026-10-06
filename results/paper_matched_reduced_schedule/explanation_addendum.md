### How to interpret this comparison

The Transformer exceeds its same-numbered LSTM seed on test Sharpe in **4 of 9 seed pairs**. Pairing seed numbers is a descriptive convenience: different architectures consume random draws differently. Seed dispersion measures initialization sensitivity on one market history, not independent market evidence or a confidence interval.

The ensemble Sharpe is calculated from averaged raw stock weights, followed by monthly gross normalization. It is **not the average of the nine individual Sharpe ratios**. Diversification across members can improve the ensemble even when individual members perform weakly.

The new LSTM test Sharpe is 0.459; Transformer is 0.304; historical Code 07 is 0.612; published GAN is 0.750. Historical and paper comparisons mix training budgets and implementation conventions. Only the two new architectures share this experiment's budget.

The reduced 16/4/48 schedule and four-epoch checkpoint warm-up differ from the author schedule. A gap to the paper can reflect training, implementation, or tuning differences; this experiment does not identify their separate effects. Lower training Sharpe is consistent with incomplete optimization, but is not proof of its cause.

**Connection to Code 09:** fixed saved ensemble returns become rolling diagnostics. **Connection to Code 10:** the same returns are grouped by economic conditions; neither analysis retrains or selects the model.
