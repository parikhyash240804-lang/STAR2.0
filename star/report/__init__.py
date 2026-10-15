"""
Turning the engine's output into something a person acts on.

Two audiences, two artefacts, one set of objects behind both (see
docs/PERSONAS.md):

* `guidance.py` - rules that read the report model and emit recommendations. They
  may **not** compute new statistics, so every claim traces back to a field
  something else produced and tested.
* `brief.py` - the HTML brief for the decision-maker. Offline, instant, and
  deliberately incomplete: anything that does not change a decision is left out.
* `excel/` - the workbook for the analyst. Deliberately complete, and the archive
  that HTML cannot be.
"""
