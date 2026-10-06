### Why rolling and full-period Sharpe differ

For each window, monthly Sharpe is mean monthly return divided by monthly standard deviation (ddof=0). Annualized Sharpe multiplies it by √12. A 36-month window contains the latest 36 monthly returns; a 60-month window contains the latest 60. The first complete endpoints are December 1994 and December 1996.

Averaging window Sharpe ratios is not the same calculation as dividing the full-period mean by its full-period volatility. Windows overlap and repeatedly reuse returns, while each window has a different volatility denominator. Therefore a higher average rolling Sharpe does not establish a better full-period investable strategy.

The distribution figure shows consistency: a curve farther right tends to have higher Sharpe. The end-year table groups windows by their ending year, not by independent annual returns. Longer windows smooth changes and may conceal shorter stress episodes.

The saved models are fixed: this is **rolling evaluation, not rolling retraining**. A rolling retraining study would require repeated training using only information available before each evaluation date. The paper's published full-test Sharpe cannot be drawn as its historical rolling path without the corresponding return series.

**Connection to Code 10:** rolling differences motivate a descriptive check of recession and volatility groups. We keep all test months and fixed labels rather than selecting favorable windows.
