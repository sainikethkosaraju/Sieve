# Phase A implementation notes (ML Engineer + Model Engineer + MLOps seats)

Scope: scaffold, `types`, `signals`, `calibrate`, `gate`, tests. No backends, server, harness or
real-model runs exist yet; nothing here is a measured result about any LLM.

## What was built
- `pyproject.toml` (dist `sieve-llm`, package `sieve`, src layout, py>=3.11, deps numpy+scipy,
  extras server/eval/dev), ruff + mypy strict + pytest config, `Makefile`, CI (py3.11/3.12),
  README, CITATION.cff, SECURITY.md.
- `types.py`: Completion/TokenLP (per-token logprob + top-k), SamplingParams (with `key()`
  for calibration compatibility), BackendInfo (model id/hash, quant, KV, build, template hash),
  Signals, Action/ReasonCode/DefectCode, Decision, TierCalibration, CalibrationArtifact.
- `signals/`: max-prob, first-token margin/entropy, mean/min/sum logprob, mean margin, mean
  truncated entropy, self-consistency (agreement, cluster entropy; normalised-string + alias
  clustering). `validate_completion` enumerates 13 defect classes; defective input yields an
  all-`None` Signals carrying the defect codes.
- `calibrate/`: metrics (risk-coverage with exact expected tie handling, AURC, AUGRC,
  equal-mass ECE, Brier, AUROC, abstention precision/recall, both hallucination rates),
  temperature/Platt/isotonic, `ltt.py` (fixed-sequence and Bonferroni LTT on the composed
  cascade, exact binomial / Clopper-Pearson and Hoeffding-Bentkus p-values, explicit EMPTY
  outcome with feasibility floor, separate cost-minimising pick), `store.py` (canonical JSON,
  SHA-256 content hash verified on load), `shift.py` (KS / PSI detectors).
- `gate/`: pure `decide(signals, artifact, tier)`; `Gate` class verifies the artifact once.

## Deviations from the design, and why
1. **Gate signature.** Architect: `Gate.decide(sig, art, tier) -> Decision` with
   `action in {answer_small, escalate, answer_large, abstain}`. Implemented the task's
   `Decision{ANSWER, ESCALATE, ABSTAIN}` plus `tier` and `ReasonCode`; the router maps
   (tier, action) to the four architect actions.
2. **Artifact shape.** Architect had a flat artifact with `lam`/`abstain_lam`. Implemented
   `tiers: (TierCalibration, TierCalibration)`, each carrying its own identity, sampling,
   signal and threshold, because the two tiers are different models and D-008 requires
   per-model identity checks. Threshold `None` means "never answer at this tier" (JSON has no inf).
3. **P-values.** The ML review mentions Hoeffding-Bentkus/WSR on the loss `(w - alpha)*a`. For
   a fixed policy, answered items are i.i.d. given their count, so the exact binomial test on
   (n_answered, k_wrong) is valid and tighter; it is the default. HB is implemented
   (`pvalue="hb"`), and for the unconditional definition `n` is the full table size. WSR is not implemented.
4. **Path construction.** "Most conservative first" is not a total order on a 2-D grid. `make_path`
   orders by coverage on a separate split A, drops points under `min_coverage` (default 0.05)
   and, by default, keeps only the A-Pareto frontier so risk is near-monotone along the path.
   Without this the walk stops at the first sparse or noisy point. Validity needs only that the
   path is fixed before testing; the choice of defaults affects power, not validity.
5. **Calibration-time defects.** NaN confidence in a calibration table is treated as -inf (tier
   never answers it), mirroring the gate's fail-closed behaviour, so deployed and certified
   behaviour coincide. A test pins `evaluate_policy` to `gate.decide`.
6. **Truncated entropy** is computed over the top-k list renormalised to sum 1, flagged in the
   docstring as flatness-dependent; it is not full-vocabulary entropy.
7. `CHOSEN_ONLY` / `SAMPLE_ONLY` scores: the gate ignores `TOP_LOGPROBS_MISSING` for scores that
   need no top-k, and ignores logprob defects for consistency-keyed tiers. Everything else
   fails closed. This is a (documented) relaxation of "any defect => fail closed".
8. Streaming/early-abort prefix calibration, learned combiners and the answer-span *extraction*
   from raw text are out of Phase A; `Completion.answer_start` takes the span as input.

## Verification (see the final report for exact outputs)
- Simulation of the guarantee: 3 configurations x 500 redraws plus an end-to-end 500-redraw run;
  assertion `freq <= delta + 3*sqrt(delta(1-delta)/R)` = 0.140 for delta = 0.10, R = 500
  (3-sigma Monte Carlo allowance; seeds fixed, so deterministic). Observed frequencies were far
  below delta, i.e. the procedure is conservative.
- Shift test: calibrate on a calibrated world, deploy where P(correct) is 0.12 lower; the
  certified policy violates alpha in essentially all replicates, so the failure is detectable.
- Hypothesis property tests use random seeds per run; they were stable in 4 consecutive runs but
  can in principle surface a new counterexample.

## Known gaps / risks
- Simulations use *independent*, perfectly calibrated tiers on Beta-distributed scores. Real
  tier errors are correlated and confidences are not calibrated; the guarantee does not need
  either property, but power and coverage numbers here say nothing about real models.
- Exchangeability, grader noise (certificate is for graded-wrong), and contamination are not
  testable in code; the artifact records `grader_id` only.
- The path/grid defaults (13x13 plus NEVER, min_coverage 0.05, frontier) are untuned engineering
  choices and must be frozen (and pre-registered) before the T4 run; tuning on split B voids the certificate.
- Hash is integrity, not authenticity (no signature).
- `Backend` Protocol is declared but not implemented; the llama-server logprob self-test
  (architect decision 3) and logprob-nondeterminism epsilon checks (red-team kill test 2b) need
  backends and are not covered here. Alias clustering has no LLM/NLI fallback.
- Python 3.11 was not available locally (tests ran on 3.13); CI covers 3.11/3.12.
