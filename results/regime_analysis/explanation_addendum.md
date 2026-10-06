### What regime Sharpe means

Each conditional Sharpe uses only the monthly returns carrying that regime label. It divides their mean by their standard deviation (ddof=0). These months can be separated in time: they are not a continuous holding period. Conditional annualized Sharpe is monthly Sharpe × √12 for scale comparison, not a simulated regime-timing return.

The return/risk table and figure help explain whether a Sharpe gap accompanies weaker average returns, greater volatility, or both. Positive-month frequency adds another view, but a model can win frequently and still lose money through a few large losses. The minimum monthly return describes one observed extreme, not an estimated future risk limit.

Business-cycle labels divide the 300 test months into 274 expansion months and only 26 recession months. Volatility labels form a separate partition; business-cycle and volatility rows must not be added together. Recession estimates have a much smaller sample, so differences should not be treated as equally precise across groups.

NBER recession labels are retrospective. Volatility labels use lagged market volatility and a training-calibrated threshold. Neither model's own volatility determines the market regimes. Joint regime tables provide additional context but can have even smaller samples.

The paper's 0.750 full-test GAN Sharpe is an unconditional benchmark, not a recession or high-volatility Sharpe. A valid paper-regime comparison requires the authors' monthly return series and identical labels. These results explain where our reduced-schedule models differ; they do not establish causality or a trading rule.
