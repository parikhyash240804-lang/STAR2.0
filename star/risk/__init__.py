"""
Risk measurement and the tests that check whether the risk model was right.

Every figure here belongs to a stated horizon and is computed from the actual
distribution at that horizon. Nothing is scaled from one horizon to another by
sqrt(t) - that shortcut is legacy defect M6-02, where a one-year loss was
labelled "1-day" and then multiplied by sqrt(252), showing a 10% loss as 159%
of capital.
"""
