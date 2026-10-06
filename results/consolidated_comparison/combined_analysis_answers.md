## Combined-analysis interpretation

The fixed 50/50 return blend has full-test monthly Sharpe **0.463**, compared with LSTM **0.459** and Transformer **0.304**. The blend averages the two saved portfolio returns with constant weights; no blend weights are selected by test performance. It is an additional descriptive illustration created after examining the test outcomes, not a newly validated out-of-sample strategy or a new trained model. It differs from the within-architecture raw-stock-weight ensemble and has no additional gross-exposure renormalization.

The combined table groups each completed rolling window by the regime in its **ending month**. For example, a 36-month window ending in a recession can include expansion months. These means describe trailing performance observed at different endpoint states; they are not Sharpes computed solely from recession returns. The earlier conditional-regime table answers the latter question.

Each business-cycle partition and volatility partition is separate. Window counts overlap and are not independent observations. The endpoint classification is descriptive; NBER recession dates are retrospective. Volatility labels are lagged and training calibrated. No regime selects a different model or triggers a trading switch.

Together the analyses answer four different questions: architecture comparison asks which fixed model performs better overall; rolling evaluation asks when performance changes; conditional regimes ask how returns differ across labeled months; rolling-by-regime asks what trailing performance looks like at different economic endpoints. The fixed blend asks whether simple return averaging adds descriptive diversification value.
