"""
Stress testing: what happens in the states that matter, not the average state.

Three complementary questions, one module each:

* `historical` - what this portfolio would have done in named episodes that
  really happened, using the actual returns of the actual holdings.
* `hypothetical` - what happens if *this* moves by *that*, with the rest of the
  book responding through its measured co-movement rather than being frozen.
* `reverse` - the question a risk committee actually asks: what is the smallest
  plausible move that costs us the amount we said we could not afford?

Legacy defects addressed (docs/AUDIT_legacy.md): a configured market shock that
was displayed but never used, so the label and the number disagreed (M6-05); a
uniform 1.3x beta multiplier with a -80% floor that decided the answer for
high-beta names (M6-04); and windows only partly covered by data accepted with
as few as five rows and no disclosure.
"""
