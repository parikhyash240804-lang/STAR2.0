"""
Portfolio construction. Each method is its own module; constraints are shared.

Nothing here clips or renormalises a solved weight vector: a constraint is either
given to the solver or it is not a constraint. The legacy enforced its caps by
clipping and then renormalising, which restores the budget by breaking the cap it
had just imposed (M3-02/M3-06).
"""
