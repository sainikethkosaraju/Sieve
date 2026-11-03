# Sieve v2 pre-registration: reading comprehension with unanswerable questions (SQuAD 2.0)

Frozen at the commit that adds this file, before any real v2 generation exists. The frozen part is
everything above `## Deviations`; the harness hashes it and refuses to run, analyse or report if it changes.
Changes go in the dated Deviations section. v1 (`docs/PREREGISTRATION.md`, `docs/RESULTS.md`) is not edited.

## Why v2 exists
v1 found that the certified cascade cost more per correct answer than the large model alone, because the small
tier could certify only about 4% of TriviaQA queries. With cost ratio r = cost_L / cost_S (3.1 in v1) the cascade is
cheaper only if the small tier answers more than a fraction 1/r of queries, and its saving is at most c - 1/r. The
pitch's "same accuracy at a fraction of the compute" therefore needs a task where the small model can answer a
large share at the certified error level. Reading comprehension (the answer is in the passage) is the natural
candidate, and SQuAD 2.0 adds unanswerable questions that test abstention.

Prior expectation, stated before the run and not used as a criterion: roughly a one-in-three chance of a cost win.
A 1.5B model may confidently invent answers to unanswerable questions, which would keep its certified coverage low.

## Question
On SQuAD 2.0, does the certified cascade (small tier answering when certified, otherwise escalating, otherwise
abstaining) achieve a lower cost per correct answer than the large model alone at the same certified error level?

## Configuration (frozen)
* Same backend, gate, certification (fixed-sequence Learn-then-Test), statistics and cost accounting as v1:
  llama.cpp commit `fac3a894fc04901c3e5abaff457e024c6b52e04f`, greedy decoding, stop at the first newline,
  alpha = delta = 0.10, paired bootstrap with 10,000 resamples, cost = wall time of every attempted generation
  including wasted attempts. Energy is not claimed.
* Small tier: Qwen2.5-1.5B-Instruct Q4_K_M. Large tier: Qwen2.5-7B-Instruct Q4_K_M. Both Apache-2.0, as in v1.
* Prompt: `squad-v1-fewshot` (passage, question, three worked examples including one `unanswerable`), template hash
  in `src/sieve/prompting.py`. Generation limit: 96 tokens (see Deviations 1).
* Data: SQuAD 2.0 validation (`rajpurkar/squad_v2`, revision `3ffb306f725f7d2ce8394bc1873b24868140c412`,
  CC BY-SA 4.0), 6,000 items sampled with seed 20261108, 50 hash-locked re-splits into A/B/C = 2,000/2,000/2,000.
  The published answerable/unanswerable mixture is kept. Gold for an unanswerable item is the string `unanswerable`.
* Grading: normalised exact match against the gold strings (one grader scores both kinds). "Wrong" means graded
  wrong by this strict grader, as in v1.
* Cells: `S_q4km` and `L_q4km` only, interleaved ABAB in 4 blocks. No quantization ladder, no self-consistency,
  no shift sets. A 100-item x 3-run determinism check on the small tier.
* The threshold path, grid and primary-signal selection rule are those coded in `sieve.eval` and `sieve.calibrate`
  at the commit that freezes this document; they are not tuned after any v2 output exists.

## Hypotheses and decision rule (frozen)
* **V2-V (validity).** Evaluated by the existing H1 code on the 50 SQuAD re-splits: the share of re-splits whose
  test risk exceeds alpha. Passing means the report prints SUPPORTED, or INCONCLUSIVE with a "SUPPORTED-WEAK"
  reason (point estimate <= 15% but its 95% interval reaches above 15%). Failing means point estimate > 15%.
* **V2-P (primary, cost).** On the canonical split 0 test set, from `comparisons.noninferiority_vs_L_only` in the
  analysis output: (a) `non_inferior` is true (correct-answer rate within the +-0.02 margin of the large tier at
  matched thresholds), AND (b) the 95% paired-bootstrap interval of `cost_per_correct_diff` (cascade minus
  large-tier-only, in seconds) has its upper bound below 0, AND (c) the point saving
  `(L_only_cost_per_correct - gate_cost_per_correct) / L_only_cost_per_correct` is at least 10%.
* **Decision.** v2 is a **success** if and only if V2-V passes and V2-P holds. Anything else, including a result
  that holds only on some other split, dataset slice or threshold, is a **negative** and is published as one.
* **Descriptive, no verdict:** coverage and accuracy among answered versus the large tier alone (as in
  `docs/RESULTS.md`), wrong answers as a share of all queries, per-stratum risk (answerable versus unanswerable),
  the small tier's certified coverage c against the measured break-even cost_S / cost_L, and cost per correct
  averaged over the 50 re-splits. The generic report also prints rows H2 and H4 to H7; they are not part of v2 and
  carry no meaning here.

## Forking-path rules
* No change to models, prompt, generation limit, alpha, delta, grid or success rule after any real v2 output exists.
* v2 generation runs once. If the run fails for infrastructure reasons (crash, quota, a bug in analysis code), the
  cached generations are reused for re-analysis and the event is recorded. If a headline cell fails the stop-rate
  sanity gate (< 95% clean stops), the run is invalid, the cause is fixed, and the full run is repeated and recorded.
  Generations are never repeated in order to improve a result.
* Both outcomes are added to `docs/RESULTS.md` next to v1, and the README is updated to match whichever it is.

## Known limits of the design
Evaluation only: the server's calibrated template does not yet support passages, so v2 results do not claim a
served artifact. One model pair and one hardware type. SQuAD and the base models share web text, so in-distribution
optimism applies. Exact match is strict for span answers. Cross-template transfer and contamination probes are not
run.

## Deviations
1. **Generation limit 32 -> 96 tokens (before the freeze, informed by a canary).** A 150-item canary on different
   items (seed 1) showed 6% of the small tier's generations truncated at 32 tokens because SQuAD answers are
   sometimes long copied spans, and truncated outputs fail closed. The canary's data is discarded and was used
   only to check plumbing and prompt behaviour. It showed both tiers answer `unanswerable` when appropriate.
