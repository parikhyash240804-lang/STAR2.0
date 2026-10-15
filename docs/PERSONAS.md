# Who each output is for

Two artefacts, two audiences, two entirely different jobs. Every layout argument
gets settled by asking which persona the item serves — and if it serves neither,
it does not ship.

---

## Persona A — the decision-maker (HTML brief)

**Meera, Managing Director.** Runs the money, does not build models. Comfortable
with Zerodha Kite, Groww, a broker's terminal. Opens the brief on a laptop,
sometimes a phone, usually with ten minutes before a meeting. May not have
internet where she reads it.

**What she is trying to do, in order:**

1. Decide whether to approve the proposed rebalance.
2. Check the portfolio is inside its risk limits.
3. Decide whether to add or withdraw capital.

**What she needs to see:**

- One recommendation, in a sentence, at the top. Not a dashboard of tiles.
- What changed since last time, and why.
- The outcome range over a horizon **she picks**: 1 day, 1 week, 1 month, 3, 6,
  12 months — one click, instant, with a plain-language confidence badge.
- Scenario buttons: *2008 repeat*, *COVID crash*, *rates +100bp*, *Financials −20%*.
  One click. The whole page updates.
- Hovering any chart point gives the exact numbers for that point.
- Which limits are breached, and by how much.
- Which three holdings carry most of the risk.
- Where the recommendation might be wrong.

**What she must never see:** a covariance matrix, factor loadings, shrinkage
intensities, GARCH parameters, solver iteration counts, p-values as raw numbers,
Greek letters, information coefficients, or any table longer than her screen.
Not because she could not understand them — because they are not the decision,
and putting them on the page buries the part that is.

**Non-negotiables:** opens offline from a file, every toggle responds in the same
frame, works on a phone, prints to something sane if she hits Ctrl-P.

---

## Persona B — the analyst (Excel workbook)

**Arjun, Investment Analyst.** Builds his own models, lives in Excel, is paid to
be sceptical. He is the one who will ask *why* the recommendation says what it
says, and he needs enough to reconstruct it.

**What he is trying to do:**

1. Audit the numbers behind Meera's brief.
2. Pull the portfolio history into his own models.
3. Keep a permanent, dated record — the HTML brief is a snapshot; the workbook is
   the archive.
4. Run his own what-ifs on the raw inputs.

**What he needs:** everything. The covariance matrix, per-holding risk
contributions, the full horizon term structure day by day, every scenario's
per-stock impact, cost breakdowns to the paisa, model diagnostics with their
p-values, the provenance of every input, and the complete weight and trade
history. As numbers in cells, never as formatted text, so his formulas can read
them.

**The archive job matters most.** HTML cannot accumulate history across runs; the
workbook can, and it is where a year of decisions lives. Sheets are dated and
append-only.

---

## Where each thing goes

| Content | HTML | Excel |
|---|---|---|
| The recommendation, in a sentence | **Yes, first** | Yes, as a record |
| What changed since last run | **Yes** | Yes |
| Outcome range by horizon (1D…1Y) | **Yes, toggleable** | Full day-by-day table |
| Confidence badge on each horizon | **Yes** | With the underlying test results |
| Scenario impact | **Yes, one click** | Per-holding, every scenario |
| Limit breaches | **Yes, traffic lights** | With exact headroom per limit |
| Top 3 risk contributors | **Yes** | All holdings, ranked |
| Proposed trades | Summary only | **Full order list, own page** |
| Costs | One number | **Itemised per scrip per leg** |
| Backtest verdict | One sentence + caveat | **Full statistics with p-values** |
| Covariance matrix | **Never** | **Yes** |
| Correlation monitor | Trend only | Full matrix and history |
| Model diagnostics, calibration tests | Badge only | **Yes, all of it** |
| Volatility model parameters | **Never** | **Yes** |
| Provenance of every input | A line of text | **Yes, per field** |
| Run history across dates | Not possible | **Yes — the archive** |

**The rule:** if it does not change a decision Meera is about to make, it belongs
in the workbook.

---

## Confidence badges — what they mean

Every forecast horizon carries one, and the rule is stated rather than invented:

| Badge | Shown when |
|---|---|
| **Good** | horizon ≤ 3 months, the risk model passed its calibration tests, and the tail has enough independent observations to support the quantile |
| **Fair** | one of those fails — typically a 6–12 month horizon, where the sample holds few independent periods |
| **Weak** | beyond 2 years (extrapolation from 5 years of data), or the risk model failed its breach test, or the regime is unsettled |

Hovering a badge does not just name the condition that triggered it: it opens the
**whole basis**, one line per check, each with the figure that decided it — how
far ahead the statement reaches and how many independent periods of that length
the sample actually holds, whether the risk model passed its breach test, whether
the market regime is settled, how much of the variation in stock movement the
volatility model explains and how much of that is reachable at all, and how the
simulated futures were drawn. A reader who disagrees can point at the line.

No score out of 100 — there is no honest way to compute one, and a made-up number
is worse than a word.

---

## Reading the brief without a finance degree

Meera is a decision-maker, not a quant, and a page she cannot read is a page she
overrides. Three rules follow, and they are tested rather than trusted to review:

- **Every term carries its own definition.** A small superscript `i` beside each
  label opens a plain-English explanation on hover, or on tap. The definitions
  live in the template, not in the data, because they are identical on every run
  — and because the payload is held to a no-jargon rule that this text would
  otherwise break. A marker that opens nothing fails the build.
- **Every number carries its scale.** "Sharpe 0.79" means nothing without knowing
  what a good one looks like, so the page says it: below 0 the risk was not
  rewarded, 0–0.5 weak, 0.5–1 respectable, above 1 strong. And when the error bar
  crosses zero, the page says that having no edge at all still fits the record.
- **The caveats are collected, not deleted.** Assumptions, data notes and
  provenance sit in a single closed dropdown at the foot of the page. Every one
  of them survives; a page that opens with fourteen of them reads as a disclaimer
  rather than as a recommendation.

Immediately above that dropdown is **the numbers that matter** — portfolio value,
the most likely outcome and a bad run in rupees as well as percent, the deepest
fall to plan for against the deepest actually suffered, the chance of ending
down, the worst scenario tested, and the certain cost of acting. Every entry is
read back out of the payload the charts above already use, so the strip cannot
drift from them.

---

## The words on the page — settled

The owner read a draft and said: *"a lot of things you said are in a language too
difficult to understand. not good, if they dont add value remove them, those who
add value use correct terms but simple english."* That is now three rules, and
`test_no_recommendation_speaks_in_jargon` enforces the first one.

**1. No jargon in anything the reader sees.** Banned outright from headlines,
detail, evidence and falsifiers: *Sharpe difference, estimation error,
perturbation, Euler, Kupiec, CSCV, deflated, in-sample, out-of-sample, go/no-go,
active tilts, basis points, standard error, information coefficient, the book,
indicative.* Anything named after the person who invented it belongs in the
workbook. The one field allowed to name a method is `confidence`, because it goes
to the workbook and is **never rendered on the page** — shipping invisible jargon
inside a file meant for a non-technical reader is all cost and no benefit.

**2. Correct terms survive if they carry meaning and the page defines them.**
*Sharpe ratio, drawdown, volatility, turnover, dividend* all stay, each with its
hover definition. Replacing them with vague paraphrase would cost precision and
buy nothing.

**3. One message, one box.** Three separate recommendations — the Sharpe gap, the
minimum track record, the trial-and-error adjustment — all said *"the evidence is
too weak to act on"* in three different vocabularies. Three weak tellings read as
noise. They are now `SKILL-01`, one recommendation with the three measurements as
supporting lines. **No measurement was dropped to get there**, and a test asserts
each one is still present. Footer notes that repeated a recommendation word for
word are gone; provenance is one sentence about the assumed return, which is the
only part of it that changes a decision.

**Rupees are written the way an Indian reader reads them** — 36.91 lakh, 1.23
crore — not 3,691,029.

### Two defects this audit found

Both had shipped, and neither was a wording problem:

- **`REGIME-01` said the opposite of what it meant.** Under the headline *"Calm
  regime: 91% probability"* it printed *"expected duration 18 days"*. That figure
  is the expected length of a **stressed** spell, so it read as "the calm lasts 18
  days". Now stated as *"when stress does arrive, spells have typically lasted
  about 18 days"*.
- **`BENCH-01` claimed a number it had not measured.** The dividend handicap was
  the literal `0.012` typed into `tools/make_brief.py`, while the rule labelled it
  *"measured against a total-return proxy"*. `star/data/benchmark.py` can actually
  measure it from NIFTYBEES, and does now: **1.14%**, calibrated over five years.
  The rule states honestly whether the figure was measured or assumed. This is the
  same failure as the legacy's hard-coded "HONEST RISK SUMMARY" (M6-07) — in our
  own code, three phases after we removed it from theirs.

---

## Technical analysis: workbook only — settled

RSI, MACD and Bollinger bands answer *"is today a reasonable moment to place this
order"*. That is Arjun's question at the dealing desk, not a board decision, and
putting a chart pattern next to an allocation invites the allocation to be
second-guessed on it. They live on the **workbook's trades page**, with their
measured predictive value shown beside them rather than assumed, and they stay
out of the optimiser entirely.

---

## What this means for the build

- **P7 (HTML)** is the product. It gets the design effort, the interaction work
  and the polish. Reference: Kite's restraint — numbers as the interface, one
  accent colour, red and green reserved for direction, no gradients or card
  stacks, instant response.
- **P8 (Excel)** is the archive and the audit trail. It gets completeness and
  correct types — every figure a real number with a format, never a string — plus
  the print-ready decision pages for anyone who still wants paper.
- Both read from the **same objects**, so they cannot disagree. The brief shows a
  subset; the workbook shows all of it.
