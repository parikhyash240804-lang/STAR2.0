"""
Walk-forward testing of the strategy that actually ships, with Indian costs and
honest statistics.

The legacy backtest is the single strongest argument for this rebuild. Verified
defects (docs/AUDIT_legacy.md): it tested a close cousin of the live strategy
rather than the strategy (M8-01); the universe was today's hand-picked list
sliced backwards, so it only ever held survivors (M8-02); the VaR breach rate was
always NaN (M8-03); costs after the first window were understated by exactly half
(M8-05); weights never drifted between rebalances (M8-09); today's risk-free rate
was used in every historical optimisation (M8-10); and there was no statistical
inference at all - no standard error, no comparison against equal weight, no
adjustment for how many configurations had been tried (M8-06).
"""
