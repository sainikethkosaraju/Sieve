# Independent audit of the evaluation harness and statistics

Seat: AI Evaluation Engineer (independent auditor; did not write the code). Date: 2026-11-03.
Scope: `src/sieve/eval/`, `src/sieve/calibrate/`, `src/sieve/signals/`, `tests/eval/`, `docs/PREREGISTRATION.md`,
`docs/evaluation.md`, `docs/reproduce.md`, Phase A / B2 notes, `results/mock/`, `benchmarks/t4_main.yaml`.
Audited baseline: commit `95ea59d` (code under test), with a look at the moving HEAD (`e2981fa`, which adds
`src/sieve/eval/providers/`). Read-only on `src/` and `tests/`. Nothing was committed.
All scratch scripts are in `/tmp/claude-0/-home-user-Sieve/296c653e-9125-5937-bc60-c353283a3b5e/scratchpad/`.

Nothing here is a measurement of a real LLM. No real model output exists in the repo; every numeric statement below
about the harness was computed on the repo's synthetic mock, on synthetic simulations I wrote, or on hand-written
test cases.

---------------------------------------------------------------------------------------------------------------------

## 0. Verdict in one paragraph

The statistical core is correct. Every metric, bound, test and the fixed-sequence LTT procedure reproduces to
floating-point precision from my independent implementations, and the full split-0 certification of the mock
reproduces exactly from the raw generation cache. The risks are not in the arithmetic. They are (1) one run-blocking
integration defect (no stop sequence, so real completions will hit `finish=length`, which the signal layer treats as
fatal), (2) a pre-registration that is looser than the code (many operational choices are unrecorded, the "frozen"
pin guards only the document text, and the B-once / C-hash gates are advisory), (3) a grader that is strict enough
to make "graded wrong" a poor proxy for "factually wrong", and (4) several hypothesis criteria that are weak or
vacuous as operationalised (H3, H4 power, H5, H6 threshold asymmetry). The committed mock results are stale
relative to HEAD.

---------------------------------------------------------------------------------------------------------------------

## 1. Independent re-derivation (own implementations, no repo imports for the reference value)

Scripts: `indep_metrics.py`, `indep_pipeline.py`, `split_check.py`, `h1_sim.py`, `h1_pool.py`.

| Quantity | My reference implementation | Max abs. diff vs repo |
|---|---|---|
| AUROC (ties = 1/2), n<=7 (300 cases) and n=400 (20 cases) | O(n^2) pairwise brute force | 0 |
| Risk-coverage curve points | pure-python cumulative per distinct score | 0 |
| AURC, AUGRC (tie handling) | exact expectation over ALL tie-break permutations, n<=7, 300 cases | 2.2e-16 / 1.1e-16 |
| AURC, n=40, heavy ties | Monte Carlo, 20,000 random tie-breaks | 1.8e-4 (MC s.e. 2.0e-4, consistent) |
| ECE equal-mass, 15 bins (n=15,37,100,1999,2000) | own chunking and sorting | 6.9e-17 |
| Brier | direct | 0 |
| Hallucination rates, both definitions, abstention precision | counting | 0 |
| Clopper-Pearson one-sided upper (9 (k,n) cases) | bisection on exact binomial cdf in pure math | 5.6e-17 |
| Clopper-Pearson two-sided (`cp_interval`) | own bisection | 1.1e-16 |
| Hoeffding-Bentkus p-value (6 cases) | formula from the LTT paper, own `h1` | 5.6e-16 |
| Fixed-sequence LTT, hand-solved toy (alpha=delta=0.2, paths of n=10/15/20 answered) | hand: p = 0.8^10 = 0.107374, 0.167126, 0.411449 | 5.6e-17; certified set = {(10,0),(15,1)}; first failure stops the walk; Bonferroni certifies 0 (hand: 0.107 > 0.2/3); reversed path certifies 0 (hand: 0.411 > 0.2) |
| Paired percentile bootstrap (mean diff, ratio statistic) | python loop over the same index matrix | 0 |
| Batch AUROC per resample / batch ECE per resample | brute force per resample | 0 / 5.6e-16 |
| McNemar exact two-sided | `math.comb` | 7.8e-16 |
| TOST paired t | scipy t cdf, own formula | 5.1e-17 |
| Split construction, all 50 re-splits | own `sha256(seed|k|id)` ranking over 6000 synthetic ids | 0 differing ids; sizes 2000/2000/2000; disjoint and exhaustive |
| Split-file check on the mock workdir (50 splits) | same | 0 differing |
| End-to-end split-0 certification of the mock, 7 methods | parsed `generations/*.jsonl` myself, own signals (max_prob, mean_logprob, mean_margin, renormalised top-k entropy), own EM grader, own grid/path/Pareto/fixed-sequence LTT/cost pick | grid size, path length, number certified, t1, t2, n_answered_B, k_wrong_B identical for gate:primary, 4 gate baselines, S_only, L_only (signal values agree to 5e-17) |

Statistical observations from the simulations (own data model, repo `make_path` / `fixed_sequence_ltt` /
`evaluate_policy`, n=2000/2000/2000, alpha=delta=0.10):

* True policy risk exceeds alpha in 3.8% (400 draws; cheapest-on-A pick) and 4.7% (300 draws; max-coverage pick) of
  certified draws. The guarantee holds (<= 10%) with room to spare.
* The event H1 actually tests (empirical risk on a 2000-item C exceeds alpha) occurs in 7.9% and 6.5%: about twice
  the true-violation rate because of C sampling noise, still far below the 15% pass line. 8% of draws (33/400) gave an
  empty certified set even in this easy model (the fixed-sequence walk halts at the first noisy failure).
* Dependence between the 50 re-splits (C_j and C_k overlap by 1/3; B_j is 1/3 inside A_k): over 40 simulated pools of
  6000, the s.d. of the violation frequency was 1.09x the iid-binomial value. The independence assumption behind the
  reported Clopper-Pearson CI on the violation frequency is therefore only mildly optimistic.

Conclusion: no defect in any numerical routine. The conditional-binomial argument for the selective-risk p-value is
sound, and the implementation matches it.

---------------------------------------------------------------------------------------------------------------------

## 2. Findings (prioritised)

### CRITICAL

**C1. No stop sequence: real completions will be `finish=length`, which the signal layer treats as a fatal defect.**
* Component: `src/sieve/backends/llamaserver.py:208-224` (`_payload` has no `stop`), `src/sieve/types.py:66-81`
  (`SamplingParams` has no `stop`), `src/sieve/eval/prompts.py:16-38` (few-shot template ending `Answer:`),
  `benchmarks/t4_main.yaml` (`max_tokens: 32`), `src/sieve/signals/logprob.py:90-120` (`TRUNCATED` is not exempt in
  `logprob_signals`' `fatal` list).
* Evidence: `results/evidence/t4_llamacpp_smoke_2026-11-03.json`: raw `/completion` on the real 3B and 7B never
  stopped on its own (`gen200 predicted_n = 200`; 24-token runs ended on the limit, answer followed by
  "\n\nThe capital of France is ..."). `_fake_backend.py:196` always emits `finish="stop"`, so the mock cannot see this.
  I built a completion of the shape a few-shot Qwen will produce (" Paris\n\nQuestion: ...") and ran the repo's
  `compute_signals`: with `finish="stop"` -> `max_prob 0.951, mean_logprob -0.05`; with `finish="length"` ->
  every signal `None`, defects `['TRUNCATED']` (`trunc_demo.py`).
* Consequence: unless the model emits EOS right after the answer on almost every item, every item fails closed, all
  methods have zero coverage, and the experiment reports "empty feasible set" as if it were a scientific negative.
  Even where signals survive, `mean_logprob`, `mean_margin`, entropy are computed over ~30 tokens of hallucinated
  follow-on questions, and wall-time cost per item is ~32 decoded tokens regardless of answer length, so latency
  cost is an artefact of `max_tokens`.
* Status: confirmed defect at `95ea59d`. HEAD (`e2981fa`) adds `src/sieve/eval/providers/llamacpp.py` with
  `stop: ["\n"]` and `trim_to_text`. That is the right direction but: (a) it is not in the frozen pre-registration or a
  Deviation; (b) it has only been tested against a fake server (`tests/integration/fake_llama_server.py`), never a real
  llama-server with the few-shot prompt; (c) `trim_to_text` leaves tokens untrimmed when a token straddles the stop
  boundary (e.g. Qwen's `".\n"` token after an answer ending in a period), producing `UTF8_GAP`, which is fatal, so
  those items silently fail closed (selection bias towards answers without trailing punctuation); (d) `stop` lives in
  `cfg.raw["backend"]`, not in `CellSpec.identity()` or `cell_id`, so a cache produced under a different stop setting
  would be reused.
* Fix: put `stop` into `SamplingParams` / `key()` / `cell_id`; run a 50-item real-server canary for each of S and L
  (few-shot QA and MCQ) and assert `finish=="stop"` for >= 95%, `joined == text`, and `len(tokens)==n_gen`; record the
  canary output in `results/evidence/`; add a Deviation entry. Alternatively exempt `TRUNCATED` only when the answer
  span is complete (not recommended).
* Verification: canary + `fail_closed_rate` in `accuracy.csv` must be < 2% before any headline cell is run.

### HIGH

**H1. Operational choices the frozen protocol leaves open, and post-freeze changes, are not recorded as Deviations.**
`docs/PREREGISTRATION.md` still says "Deviations: None yet." Line-by-line comparison (section 3) shows the items below.
The most consequential:
* Prompt template changed after the freeze (`qa-v1` -> `qa-v2-fewshot`, commit 40fbaf1; also MCQ few-shot). The claim
  in the pre-registration depends on "prompt" being fixed, but the prompt text is not in the document at all. (No
  Deviation recorded.)
* "Grid and monotone threshold path fixed before split B is read": the code builds a path from split A as the
  A-Pareto frontier ordered by A-coverage (12 quantile levels + NEVER, 5% coverage floor; `certify.py:46-50`,
  `ltt.py:make_path`). It is not a monotone threshold path (Phase A note 4 says this explicitly; "path/grid defaults
  ... must be frozen (and pre-registered) before the T4 run" was never done). Validity is unaffected; power and
  reproducibility are.
* Additional arms/definitions not in the document: S KV q8_0 / q4_0 arms; primary signal chosen per re-split among 4
  candidates; H1 denominator = certified splits only; H5 "before violation" = alarm needs no labels; H3 "matched" =
  +-0.02 non-inferiority on accuracy-on-answered and A-matched thresholds; TOST applied to dAUROC and dECE only;
  energy via `nvidia-smi -lms 100` not NVML; MMLU-Redux restricted to `error_type=="ok"`; TriviaQA from `validation`;
  SimpleQA judge prompt is not the official one.
* `hardware_info()` (`pipeline.py:~150`) writes the literal string `"gpu": "see nvidia-smi"`; the pre-registration
  promises results record GPU, driver, build and model hashes. The llama.cpp commit is recorded only via the new
  provider's `BackendInfo.build`.
* Fix: append a dated Deviations block before the first headline cell that lists each item (the hash covers only text
  above `## Deviations`, so this is allowed). Record real GPU name, driver, CUDA, llama.cpp commit, GGUF SHA-256 in
  the manifest.

**H2. The B-once and C-hash gates are advisory; "B used once" cannot be asserted from the artefacts.**
Demonstrated on a copy of the mock workdir (`leak_test.py`):
1. Second `open_b` in the same workdir is refused (good); a foreign capability is refused (good).
2. `CERTIFICATION_CAPABILITY` is a module global importable by anything (`splits.py:59`); `store._read(k)["B"]` returns
   B with no ledger entry (`b_state` stays unchanged).
3. Deleting `split_access_ledger.jsonl`, or using a fresh workdir with the same config, re-opens B with no trace
   (same `index_hash`, same splits).
4. `certify_or_load` accepts a hand-edited `certs/split_NNN.json` (thresholds changed, `bundle_hash` kept):
   it only checks that the stored `bundle_hash` is in the ledger (`certify.py:certify_or_load`); the hash is never
   recomputed from the stored content.
5. `open_c` has no once-only rule and accepts any hash read from the plaintext ledger; C is reopened freely.
6. Check-then-append on the ledger is not atomic (two processes can both read B).
7. Most important for forking paths: because analysis is CPU-only on the cache, anyone can change `Q_LEVELS`,
   `MIN_COVERAGE_A`, `PRIMARY_CANDIDATES`, the path rule or the template after seeing real H1/H3 results and re-run
   against the same B and C in a new workdir. The ledger records neither the analysis code version nor the config.
No code path was found where B/C labels feed an A-stage choice within one split (primary signal, calibrators, router,
grids, path, cost pick all use `store.a_ids(k)` only; `certify.py:150-168` verified). The weakness is procedural.
Fix: write analysis `git` commit, config hash, and dirty flag into the ledger and each certification; recompute and
verify the bundle hash on load; refuse `analyse` if the tree is dirty or the commit differs from the one named in
the pre-registered analysis plan; commit `certs/*.json` and the ledger to git (append-only) before C is first
evaluated; add a file lock.

**H3. The report's "refuses if the pre-registration changed" guard is narrower than advertised.**
Tested with `prereg_test.py` (calls `hypotheses.build_report` against edited copies):
* Refused: changing alpha/delta/H1 threshold text (regex `CRITERIA_PATTERNS`); any edit to the frozen body not
  accompanied by new pins (hash mismatch); inserting a `## Deviations` heading mid-file; removing baseline text.
* Accepted (by design): appending a Deviation; trailing-whitespace and CRLF-only changes.
* Accepted (gap): editing a non-coded frozen number (e.g. `B=2000` -> `B=1000`) **and** updating both pins
  (`benchmarks/t4_main.yaml` and manifest) passes. The pin is not anchored to the freeze commit `ca3ed07`.
* Accepted (gap): `cfg.alpha`, `cfg.delta`, `n_splits`, `n_boot`, `equivalence_margin`, `sc_k`, `window_size` are never
  compared with the pre-registration; the report accepted alpha=0.2, delta=0.3, n_boot=100, n_splits=5, margin=0.1,
  k=3, window=50 against the unchanged document (7/7 accepted). Analysis also does not compare `cfg.config_hash()`
  with the run manifest's.
* Fix: in `build_report`, assert config values equal the parsed pre-registration numbers (`CRITERIA` plus
  alpha/delta/sizes/n_splits/n_boot/k/margin); compare `config_hash` to the manifest; add the expected frozen hash as
  a constant verified against `git show ca3ed07:docs/PREREGISTRATION.md`.

**H4. Committed mock results are stale versus HEAD, and the mock is seed-fragile.**
`results/mock/` was generated before commit 40fbaf1 changed `prompts.py`. The fake backend is seeded by the template
hash, so a fresh `python -m sieve.eval mock` on HEAD gives different numbers (`methods.csv`, all tables, figures
differ). Verdicts differ: committed H2 SUPPORTED, H3 SUPPORTED; fresh run H2 NOT SUPPORTED, H3 INCONCLUSIVE
(two fresh runs and a `PYTHONHASHSEED=1` run agree with each other, so the harness is deterministic). The committed
page shows `commit 7733433...+dirty`, a WIP commit that is not the code that produced the tables at HEAD.
`docs/reproduce.md` §3 ("byte for byte") and the Phase B2 notes ("last run: H2 SUPPORTED ...") no longer hold.
At HEAD, `tests/eval/test_pipeline.py::test_unavailable_judge_is_reported_not_substituted` fails (asserts the H5 reason
contains "unavailable"; the new seed gives "NQ-open not run, ungraded or nothing answered"), while the notes claim
466 passed. A test that depends on a synthetic outcome of a seeded fake is brittle.
Fix: regenerate `results/mock/`, make the test assert on structure rather than on a seed-dependent branch, and add a CI
check that regenerating the mock reproduces the committed files.

### MEDIUM

**M1. Grader validity: exact match is strict, and its false-wrong rate is not bounded by the design.**
I wrote 46 hand-labelled cases (36 free-form, 10 MMLU) covering aliases, numbers, dates, verbose sentences, partial
names, refusals, multi-answer outputs and letters (`grader_cases.py`). These are adversarially chosen, not a sample of
model behaviour, so the rates are not population estimates.
* `em-alias-v1`, free-form: 23/36 agree; 13 false-wrong, 0 false-right. The false-wrongs: "The capital of France is
  Paris.", "It is Paris", "Paris, France", "In 1945", "8 legs", "8,848", "June 6th, 1944", "6/6/1944", "Kennedy",
  "President Kennedy", "da Vinci", "Nile river", "Will Shakespeare".
* `contains-alias-v1` (sensitivity grader): 28/36; 6 false-wrong, 2 false-right ("Jupiter or Saturn", "Saturn, Jupiter").
* `mmlu-letter-v1`: 9/10; misses "I think it's B" (counted invalid).
* A continuation after the answer ("Paris\n\nQuestion: ...") is handled correctly because only the first line is graded.
* Interaction with verbosity: the few-shot prompt ("short phrase only") should make most outputs terse, but the first
  token of a verbose answer (" The", " It") is graded and scored as if it were the answer, so false-wrongs also corrupt
  AUROC and the certified risk (certified wrt the grader, which the pre-registration states, but "graded wrong" then
  mixes format failures with knowledge failures). The false-wrong rate is expected to differ between 3B and 7B
  (verbosity), which would bias H3. The real size of the effect is unmeasured; only the 200-item audit can measure it.
* The audit is drawn from TriviaQA split A only (`pipeline.py:audit_sheet`). NQ-open (lowercase, no question mark,
  multi-answer lists) has a different EM strictness, so H5 ("risk rises on NQ-open") confounds model shift with grader
  shift. Kill-test 6 passes on the point estimate of overall error <= 3% (`audit.py:passes_kill_test`), not on its upper
  limit; with 50 items per stratum the Clopper-Pearson half-width is ~12 points.
* Fix: add an NQ-open audit (>= 100 items); decide kill-test 6 on the upper bound; normalise numerals, ordinals,
  thousands separators and leading prepositions; report `contains` alongside EM in the main accuracy table; log
  the first-token opener rate per tier as a verbosity proxy.

**M2. H5 as operationalised cannot support its stated reading.**
`hypotheses.py:h5`, `analysis.py:drift_windows`, `sections.py:shift_section`: (a) "alarm fires before violation" is
replaced by the overall alarm rate across all shifted windows (no per-window ordering with respect to violation);
(b) no false-alarm control on in-distribution C windows, so an alarm rate of 80% is uninterpretable; (c) 2000 NQ items
give 10 windows and 1000 SimpleQA items give 5, so 0.8 is 8/10 and 4/5; (d) grader shift confound (M1);
(e) NQ-open/TriviaQA overlap is counted (`pipeline._overlap`), not removed.
Fix: report the false-alarm rate on iid C windows at the same p threshold; use overlapping windows or more windows;
define the pre-violation statement per stream.

**M3. H4 is underpowered by construction.** `sections.sc_section` evaluates AUROC only on the ~333 SC items that fall in
C of re-split 0 (1000-item subset x 1/3). Expected paired-CI half-width is ~0.05, so "NOT SUPPORTED (upper < 0.03)"
is nearly unreachable and "SUPPORTED" needs a large gain. AUROC needs no threshold, so all 1000 SC items minus the
split-0 A items (primary signal choice) could be used. Also the SC method is certified on ~1/6 the data of the other
methods, so its `methods.csv` row is not comparable.

**M4. H3 is partly vacuous and partly true by construction.** The non-inferiority test is on accuracy-on-answered, but
both the gate and L_only are certified at alpha on B, so selective accuracy is pinned near >= 0.90 for both; the real
trade is coverage, which enters only through cost-per-correct. The random and query routers answer S-routed items
without any confidence check, so "gate beats router on accuracy-on-answered" is expected for any signal with AUROC
> 0.5. The query router is TF-IDF + L2-LR fit on 2000 questions (documented, but "RouteLLM-style" overstates it and it
is a weak opponent). `vs_*_certified` comparisons exist but are not in the H3 verdict. Fix: add
coverage-at-certified-risk non-inferiority and a router with its own abstention.

**M5. H1/H6/H2 threshold asymmetries.** H1 passes at <= 15% violating splits but H6 calls "recalibrated violates" at
> 10% (`cfg.delta`). With a per-split empirical violation probability of 8% (what my simulation produces for a valid
procedure), P(freq > 0.10 of 50) = 0.23 per config, so across 5-6 quantization configs a spurious "violates" is
likely; with 4% it is 0.014. H2 SUPPORTED takes the best of three candidate signals with no multiplicity control and
NOT SUPPORTED is decided on point estimates. H2-H7 all rest on one canonical re-split (split 0), with bootstrap CIs
conditional on its thresholds.

**M6. Per-stratum risk is vacuous for the main dataset, and the mock hides it.** `parse_triviaqa` assigns no stratum
(`datasets.py`), so the main per-stratum table has a single blank stratum on real data; only SimpleQA topic and
MMLU subject have strata. The mock generator assigns synthetic topics, so the mock shows per-stratum output that real
data will not. No length, answer-type or popularity strata.

**M7. Cost/energy accounting issues.**
* Pre-registered L KV q8_0 arm: cell `L_q4km_kvq8` is defined (4 datasets, expensive) but no pair or analysis uses it, and
  it is not in an ABAB group. S KV arms are generated but are not pre-registered.
* `peak_vram_mib = max(memory.used)` over all GPUs in an ABAB window where all cells of the group are resident
  (`power.py:summarise_window`), so "peak VRAM per cell" includes co-resident models and other processes.
* Idle baseline is measured once, before any model is loaded (`cli.py:104`); the smoke evidence shows GPU draw at
  ~9.8 W idle and ~27-38 W in other states, so `joules_net` includes a state-dependent offset. nvidia-smi `power.draw` on
  T4 refreshes about once per second; `-lms 100` oversamples stale values (the smoke samples are ~0.55 s apart).
  `throttled` uses mask `0xFC`, which includes the SW power-cap bit that a T4 hits under load, so it will usually be
  true. Pre-registration says NVML 10 Hz.
* Wasted attempts are charged (verified in `data.py`), model load time is reported separately (verified in
  `runner.py`), gates charge S on every item (verified in `methods.policy_item_costs`). Per-item S and L times come from
  separate isolated runs, not from a real cascade, and `cache_prompt` is on, so time depends on cache state.

**M8. Mock output labelling.** Only `index.html` and `hypotheses.json` carry the synthetic banner. `tables/*.csv` and
`tables.md` have no marker and use real-sounding labels (`Q4_K_M`, `triviaqa`, `L_q4km`, accuracy 0.900), so a copied
table is indistinguishable from a result. Figures carry a footer; the Pareto figure puts accuracy-on-answered against
cost with no coverage axis and draws the uncertified test-set sweep without saying so, so a gate that answers less
looks better. Fix: add a `synthetic` column or filename prefix to every CSV/Markdown table, and put coverage on the
Pareto figure.

**M9. Contamination and overlap not handled.** TriviaQA/NQ-open/MMLU are public and likely in Qwen2.5 pre-training; there is no
contamination probe. De-duplication is by exact normalised question text inside TriviaQA only; near duplicates are not
detected; the few-shot examples ("chemical symbol for gold", "Romeo and Juliet", "Kyoto", "largest planet") are not
excluded from the eval items; A/B/C are disjoint per split (verified) but each item is in B or A of other splits
(B_j is 1/3 inside A_k, verified).

### LOW

* L1. SimpleQA `not_attempted` is "not wrong" for risk but `PolicyEval.correct_answered = answered & ~wrong` counts it as
  correct, so `accuracy_on_answered = 1 - risk` includes abstentions as correct while AUROC treats them as incorrect
  (`data.py:~145`, `evaluate.py:46`). Matters only once a judge is configured.
* L2. `bootstrap()` drops NaN replicates (zero-denominator resamples), which biases CIs when few items are answered.
* L3. H2 candidate `first_margin` is not among `PRIMARY_CANDIDATES`, so the primary gate can never be it.
* L4. `ece_equal_mass` splits ties by stable order (documented); value depends on input order for tied scores.
* L5. `analysis.run_analysis` pools per-stratum counts across overlapping re-splits as if independent.

---------------------------------------------------------------------------------------------------------------------

## 3. Code vs frozen pre-registration (line by line)

| Pre-registration item | Code | Match |
|---|---|---|
| alpha = delta = 0.10 | `t4_main.yaml`; config value never compared with the document | yes in config, unchecked (H3 finding) |
| 6000 TriviaQA, fixed seed; A/B/C = 2000/2000/2000 | `datasets.PINNED` (n=6000, seed 20261103), `splits.derive_split` strict sum check | yes |
| 50 hash-locked re-splits | `build_splits`, `SplitStore.verify` re-derives all | yes (re-derived independently) |
| Shift sets NQ-open 2000, SimpleQA Verified 1000, MMLU-Redux 1500 | yes; MMLU keeps only `error_type=="ok"`; TriviaQA from `validation` | open choices, not recorded |
| Grader: normalised alias EM; SimpleQA three-label judge (judge != S, L) | EM yes; judge default `unavailable` (H5 INCONCLUSIVE), prompt is own wording | partial by design |
| 200-item human audit per grader, e1/e2 with CIs, sensitivity-adjusted alpha | `audit.py`; per grader means EM only; no NQ/SimpleQA audit | partial |
| Baselines: S only, L only, 4 gates, random router at matched escalation, oracle router, oracle abstainer, query-only router retrained on A, SC k=5 on 1000 | all present (`methods.py`, `evaluate.py`); S_only/L_only are single-tier gates certified at alpha; "entropy" = truncated top-20 mean entropy; "margin" = mean margin; random/query routers' S answers unchecked | yes with definitional choices |
| "Same certification procedure" | `certify.py` for all, except oracles (descriptive) and SC (on ~1/6 of data) | yes |
| Grid and monotone threshold path fixed before B | A-quantile grid + A-Pareto frontier ordered by A coverage | **deviation** (not monotone; not pre-registered) |
| Bootstrap 10,000, paired by item, percentile; McNemar at matched coverage; TOST +-0.02 for quantization | `stats.py`; one canonical split; TOST on dAUROC/dECE only; McNemar on answered-and-correct | yes with choices |
| Nondeterminism 300 items x 3 runs | `plan_units` (1 base + 2 repeats), S cell only | yes |
| H1 (<=15% pass, >25% fail) | `prereg.CRITERIA` + `h1()`; denominator = certified splits; <10 certified = INCONCLUSIVE | yes with choice |
| H2 (>= 0.02 AUROC, paired CI excludes 0; null = all within +-0.02) | `h2()`; best of 3 candidates, null by point estimate | yes with choices |
| H3 | `h3()`: -0.02 non-inferiority on accuracy-on-answered; cost CI < 0; beats random and query at A-matched thresholds | operational, partly vacuous (M4) |
| H4 (k=5, >= 0.03 AUROC) | `h4()` on ~333 items of C0 | yes, underpowered (M3) |
| H5 (risk > alpha on NQ, more on SimpleQA, alarm >= 80% before violation) | `h5()`; alarm rate overall, no negative control | operational, weak (M2) |
| H6 (f16 threshold violates, recalibrated does not) | `h6()`; delta=0.10 not 0.15 | yes with asymmetry (M5) |
| H7 (MMLU-Redux AUROC < 0.6) | `h7()`: SUPPORTED requires CI upper < 0.6 | stricter than the text, fine |
| Config: 3B-Instruct Q4_K_M S; ladder f16/Q8_0/Q3_K_M/Q2_K; 7B Q4_K_M L f16 KV; extra L KV q8_0; n_probs >= 20; greedy; llama-server at recorded commit | cells present; **L KV arm unanalysed**; S KV arms extra; commit now pinned in HEAD yaml; Qwen Instruct served through raw completion | partial (M7) |
| Cost: wall time of every attempt incl. wasted S and L re-gen; load time separate | verified in `data.py`, `runner.py`, `methods.py` | yes |
| Energy: NVML 10 Hz, idle-subtracted, ABAB, GPU only | nvidia-smi `-lms 100`; idle once before load; `abab_order` plain ABAB (no counterbalance); GPU only | partial (M7) |
| VRAM peak per cell | max over all GPUs and co-resident cells | invalid under ABAB co-residency (M7) |
| Hash recorded in every result; changes only as dated Deviations | text hash yes; code/config/commit not tied | partial (H2, H3) |

Operational choices the document leaves open (all should become dated Deviations): which split is canonical
(re-split 0); H3 "matches" (+-0.02); "matched coverage" (A-matched thresholds); H1 denominator; H5 "before violation"
and drift reference (split A scores, 200 non-overlapping random windows, KS p<0.05); H6 violation threshold; TOST
metrics and bootstrap level (90%); primary-signal rule; grid/path/coverage floor/frontier; dataset subsets and
filters; prompt template; max_tokens 32; stop rule; judge prompt; query-router family; SC temperature 0.7/top_p 0.95;
ABAB blocks (4); idle seconds (30).

---------------------------------------------------------------------------------------------------------------------

## 4. Leakage check

* Within a split: `choose_primary`, calibrators (`PlattScaler`, TF-IDF router fit), grids, path, cost pick, audit sheet
  and drift reference all read `store.a_ids(k)` only (verified at `certify.py:150-168`, `analysis.py:run_analysis`,
  `pipeline.audit_sheet`). `open_b` is called only from `certify_split`; a test greps for other callers. B is used
  for p-values only. No B/C-tuned quantity feeds an A-stage choice.
* Across splits: each item is in B (or C) of ~1/3 of splits and in A of ~1/3 (measured), which is inherent to the
  re-split design; certifications are per split and do not carry state. The 50 re-splits are not independent draws
  of the population (CI assumption discussed in section 1: mild).
* Gating strength: advisory only (H2). Items files hold gold answers, the cache holds every item's output, and the
  ledger/certs are editable.
* Train/test overlap: de-duplication is exact-normalised within TriviaQA only; cross-dataset duplicates are counted not
  removed; the few-shot examples are not excluded; no paraphrase/near-duplicate or pre-training contamination check.
  The overlap count was not testable here (the Hub payloads are unreachable from the sandbox, as the B2 notes say).

---------------------------------------------------------------------------------------------------------------------

## 5. What a skeptical ML reviewer would attack, and what is missing

1. "You certify against a grader, not the truth." No grader audit on shift sets, no NQ audit, point-estimate kill test.
2. "Your signal is measured over formatting, not knowledge." Opener tokens, no length-bias stratification (answer
   length vs mean_logprob), unverified stop handling.
3. "Contamination." No probe (e.g. masked-answer or perturbed-question accuracy; n-gram overlap with public corpora).
4. "Cross-template transfer." Not implemented (kill test 4); prompt changed once already.
5. "Quantization effect vs noise." No null control (kill test 11): repeat-build or alternate-kernel control, or at least
   a Q4_K_S/Q4_K_M pair, is needed before attributing a delta to bit width; greedy decoding is deterministic so the
   nondeterminism subset says nothing about build-to-build variance.
6. "One split drives the verdicts." H2-H7 use re-split 0; no across-split pooling of effect sizes.
7. "Strawman baselines." TF-IDF query router on 2000 questions; routers have no S-side abstention; self-consistency on
   1/6 the data.
8. "Per-stratum risk." Absent for TriviaQA (no strata), so no evidence the guarantee holds by question type, length or
   popularity; marginal coverage is all that is certified.
9. "Energy numbers." No wall meter; 1 Hz power refresh; single idle baseline; co-resident VRAM.
10. "Fixed-sequence power." 8% empty sets in an easy synthetic model; real weaker signals may fail more often.
11. Missing: coverage/yield non-inferiority at matched certified risk; false-alarm rate for the drift detector; per-tier
    e1/e2; a real-server canary; frozen analysis-code commit.

---------------------------------------------------------------------------------------------------------------------

## 6. Summary table

| ID | Severity | Component | Status | Headline-affecting |
|---|---|---|---|---|
| C1 | Critical | stop sequence / `TRUNCATED` fatal / cost = 32 tokens | confirmed defect (mitigation in HEAD unverified on a real server) | yes (all of it) |
| H1 | High | Deviations missing; prompt changed post-freeze; path not monotone; hardware not recorded | confirmed | yes (protocol integrity) |
| H2 | High | B-once / C-hash gates advisory; no code-version binding; cert tamper accepted | confirmed (demonstrated) | yes (claim "B used once") |
| H3 | High | prereg refusal guard misses config values and non-coded edits | confirmed (demonstrated) | yes (claim "refuses if changed") |
| H4 | High | stale committed mock; HEAD test red; docs claim reproducibility | confirmed | no (presentation), yes (credibility) |
| M1 | Medium | grader strictness, audit design (no NQ audit, point-estimate rule) | confirmed on hand cases; population rate unmeasured | yes (H1, H3, H5 interpretation) |
| M2 | Medium | H5 operationalisation | confirmed | yes (H5) |
| M3 | Medium | H4 power (333 items) | confirmed | yes (H4) |
| M4 | Medium | H3 vacuity / strawman routers | confirmed | yes (H3) |
| M5 | Medium | H6 threshold asymmetry, H2 multiplicity, single split | confirmed | yes (H2, H6) |
| M6 | Medium | per-stratum risk vacuous on real data, mock hides it | confirmed | no |
| M7 | Medium | L KV arm unanalysed; VRAM per cell; idle baseline; nvidia-smi rate | confirmed | energy claims only |
| M8 | Medium | mock tables lack synthetic marker; Pareto omits coverage | confirmed | no |
| M9 | Medium | contamination/overlap/few-shot overlap | risk | possibly (accuracy level, H5) |
| L1-L5 | Low | see list | confirmed | no |

### Headline-affecting issues (explicit list)

1. C1: with no stop sequence every real item can fail closed; no valid headline exists until a real-server canary passes.
2. H1/H2/H3: protocol integrity claims ("frozen", "B used once", "refuses if changed") are weaker than documented; a
   forking-paths path exists through CPU re-analysis against the same B and C.
3. M1: "risk <= alpha" is risk of being graded wrong by a strict EM grader, which will count verbose and formatted
   correct answers as wrong; its size differs by tier and dataset and is only partly audited. This affects H1 (what
   is certified), H3 (tier-dependent bias), H5 (grader shift), H7.
4. M3/M4/M5/M2: H3 and H5 can be "supported" through weak criteria; H4 cannot realistically reach NOT SUPPORTED; H6
   has an asymmetric threshold; H2 is best-of-three on one split.
5. M7: any "X% of the energy/compute" statement is not supported by the pre-registered measurement design as
   implemented (single idle baseline, 1 Hz source, co-resident VRAM, L KV arm unanalysed).

Things verified as correct and safe to rely on: every metric and test routine; fixed-sequence LTT and its
conditional-binomial validity (simulated true-violation 4-5% at delta 10%); split derivation and integrity checks;
cost charging of wasted attempts; the paired-bootstrap design; the report's synthetic banner on `index.html`.
