# Phase B2 implementation notes (Data Engineer + AI Evaluation Engineer + MLOps seats)

Scope: `src/sieve/eval/`, `benchmarks/`, `tests/eval/`, `docs/{reproduce,evaluation}.md`, `results/`.
Phase A packages and the backends/router/server/observe/Dockerfile/kaggle trees were not touched.
Nothing here is a measurement of a real LLM: the only runs that exist are on the fake backend
with synthetic data, and every artefact says so (`meta.synthetic_label`, report banner, figure
footers, hypothesis text, `energy.jsonl` state `synthetic`).

## What was built

* **Datasets** (`datasets.py`): loaders for TriviaQA rc.nocontext, NQ-open, SimpleQA Verified,
  MMLU-Redux 2.0 over Hugging Face `datasets` with pinned 40-hex revisions and licences; order
  independent seeded sampling; question de-duplication; hash-verified item files; counts of
  cross-dataset duplicate questions.
* **Splits** (`splits.py`): 50 hash-locked re-splits derived from a seed, split files plus an
  index whose hashes are re-derived on load; B opens once through a capability only
  `certify.py` holds and an append-only ledger; C opens only with a certification hash.
* **Grading** (`grading.py`, `audit.py`): alias exact match (primary), alias containment
  (sensitivity), MMLU letter parse, a pluggable 3-label SimpleQA judge with an explicit
  `unavailable` state, a judge cache, F1-style summary; 200-item blind audit sheet generator and
  analyser (e1, e2, Clopper-Pearson, population-weighted error, sensitivity bound, kill-test 6 rule).
* **Runner and cache** (`runner.py`, `cache.py`, `power.py`): append-only hashed JSONL per cell,
  every attempt recorded with wall time, resumable after SIGKILL, idempotent; nvidia-smi 10 Hz
  sampler, trapezoid joules, idle subtraction, ABAB scheduler, explicit `no_gpu` state.
* **Analysis** (`data.py`, `methods.py`, `certify.py`, `evaluate.py`, `analysis.py`,
  `sections.py`, `stats.py`, `hypotheses.py`, `prereg.py`): signals via Phase A `signals`,
  all pre-registered baselines, Phase A fixed-sequence LTT, 50-re-split validity, paired
  bootstrap / McNemar / TOST, per-stratum risk, non-inferiority and cost per correct including
  wasted attempts, H1-H7 verdicts with the numbers, negative-outcome list, prereg pin.
* **Reporting** (`report.py`): CSV + Markdown tables, four matplotlib figures with provenance
  footers, one self-contained accessible HTML page; schema in `results/schema/`.
* **CLI and configs**: `python -m sieve.eval {prepare-data,run,analyse,report,audit-sheet,
  audit-analyse,hypotheses,mock}`; `benchmarks/t4_main.yaml` (frozen protocol, pin included),
  `t4_fallback_1p5b.yaml`, `t4_smoke.yaml`; `benchmarks/reproduce_mock.sh`.
* **Committed artefact**: `results/mock/` (synthetic report page, tables, figures).

## Deviations and decisions the Director should know about

1. **Pre-registration operational details.** The document leaves some details open; the choices
   are listed in `docs/evaluation.md` (H3 "matches" = ±0.02 non-inferiority; matched-coverage
   comparison via A-matched thresholds; H5 "before violation" = alarm needs no labels; drift
   reference = split A; canonical split = re-split 0 for bootstraps; H6 violation = > delta of
   re-splits). None edits a criterion; record any as a dated Deviation if wanted.
2. **Frozen-text hash covers only the part above `## Deviations`**, so dated Deviations can be
   appended without breaking the pin, while any edit to the frozen text breaks run, analysis and
   report. `t4_*.yaml` pin the current hash (`4c3ca8b5...`).
3. **MMLU-Redux 2.0** (57 subjects, 5700 rows) is used with `error_type == "ok"` rows only;
   TriviaQA is sampled from `validation`. Neither is stated in the pre-registration.
4. **SimpleQA judge prompt is our own wording**, not the official autorater prompt; NOT_ATTEMPTED
   is not counted as wrong. Default judge is `unavailable` (SimpleQA ungraded, H5 INCONCLUSIVE).
5. **Query-only router uses TF-IDF + logistic regression**, no embedding model.
6. **Mock sizes are smaller** than the protocol (1200/450 TriviaQA items, 50 re-splits of 400);
   the real configs use the pre-registered sizes.
7. **Quantization arms**: ladder (f16, Q8_0, Q3_K_M, Q2_K) from the pre-registration plus S KV
   q8_0/q4_0 and an L KV q8_0 arm from the research review, all on TriviaQA only. This is more
   generation than the quota may allow; trim cells before the run, not after.
8. **Cost pick on A.** Path, grids, primary-signal choice and cost pick use only split A, per
   the ML review; B is used only for the LTT p-values.

## Verification (true outputs, 2026-11-03, Python 3.13)

* `ruff check src tests`: All checks passed. `ruff format --check src tests`: 103 files already
  formatted. `mypy` (strict, repo config): Success, no issues in 103 source files.
* `pytest` whole repo: **466 passed, 1 skipped** in about 77 s (the skip is the opt-in live Hub
  test). 137 of these are in `tests/eval/` (137 passed, 1 skipped), 329 are the existing tests.
* Mock pipeline (`python -m sieve.eval mock`, full size): about 20 s; mock-run verdicts are
  synthetic and not to be quoted (last run: H1 SUPPORTED, H2 SUPPORTED, H3 SUPPORTED, H4 NOT
  SUPPORTED, H5 NOT SUPPORTED, H6 INCONCLUSIVE, H7 NOT SUPPORTED).
* Hard-kill test: the mock CLI is killed with `os._exit(137)` after 150 generations, resumed,
  and its cache equals an uninterrupted run line for line (excluding wall time/hash).
* Determinism test: two full small mock runs give identical `results.json` (minus commit),
  tables, hypothesis text and byte-identical generation shards.
* Real Hub data was **not** loaded (Xet CDN blocked by the sandbox proxy); schemas were checked
  against cards and, for SimpleQA Verified, against the CSV header.
* Real backends, real power sampling on a GPU and a real T4 run were not exercised.

## Known gaps

* Provider for llama-server is a hook (`backend.provider: module:factory`), unverified against
  the other team's code. `pyproject.toml` (not owned here) needs `pyyaml` in the `eval` extra; the
  YAML config tests skip nothing but the loader raises a clear error without it. The Makefile
  `reproduce-mock` recipe should call `benchmarks/reproduce_mock.sh`.
* Not implemented: cross-template transfer (kill test 4), quantization null control (11),
  length-bias stratification, contamination probe, epsilon pass/fail for logprob repeat drift
  (numbers reported only), embedding router, official SimpleQA autorater prompt, wall-meter energy
  cross-check (impossible on Kaggle).
* The B-once rule is accident and misuse *detection*, not an OS sandbox; items and cache files
  are readable by anything with file access.
* Mock simulations show validity only for exchangeable synthetic data; they say nothing about
  real models, grader error, contamination or non-exchangeable traffic.
* Statistical power: at n_B = 2000 and alpha = 0.10 certification can be empty or low coverage
  (a pre-declared negative outcome, reported as such).

## Red-Team kill-test mapping

See the table in `docs/evaluation.md`: covered or partly covered are 1 (H1), 2 (cache and
nondeterminism numbers only), 5, 6, 10 (CV, no wall meter) and the cross-dataset half of 4;
3, 7-9 and 12 belong to Gate/Server teams; 11 and cross-template 4 are open.
