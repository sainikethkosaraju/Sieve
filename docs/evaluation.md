# Evaluation harness (Phase B2)

Code: `src/sieve/eval/`. Configs: `benchmarks/*.yaml`. Tests: `tests/eval/`. Protocol: the frozen
`docs/PREREGISTRATION.md`, which this harness implements and never edits. Nothing here is a
measurement of any real LLM until a T4 run has been made; the offline mock is **synthetic** and
labelled so in every output.

## Data flow

```
prepare-data  items/*.jsonl (+ .meta.json hash)   splits/split_NNN.json + splits_index.json
run           generations/<cell>.jsonl (append-only, hashed)  cells/*.json  energy.jsonl  manifest.json
analyse       certs/split_NNN.json   analysis/results.json   analysis/hypotheses.{json,txt}
report        report/{index.html,tables/*.csv,tables/tables.md,figures/*.png}
audit-sheet   audit/audit_sheet.csv (blind)  audit/audit_key.json     audit-analyse -> e1, e2
```

Only `run` calls a model. Signals, grading, calibration, baselines, statistics and plots are
recomputed from the cache on CPU (architect decision 4), so Kaggle quota is spent once.

## Datasets (pinned, 2026-11-03)

| Name | Hub repo / config / split | Revision | Licence (card) | n | Grader |
|---|---|---|---|---|---|
| TriviaQA rc.nocontext | `mandarjoshi/trivia_qa` / `rc.nocontext` / validation | `0f7faf33...` | unknown (card lists none) | 6000 (A/B/C 2000 each) | em-alias-v1 |
| NQ-open | `google-research-datasets/nq_open` / `nq_open` / validation | `5dd9790a...` | CC-BY-SA-3.0 | 2000 | em-alias-v1 |
| SimpleQA Verified | `google/simpleqa-verified` / `simpleqa_verified.csv` (split `eval`) | `0dc97e0d...` | MIT | 1000 | 3-label judge |
| MMLU-Redux 2.0 | `edinburgh-dawg/mmlu-redux-2.0` / 57 subject configs / test | `372ea425...` | CC-BY-4.0 | 1500 | mmlu-letter-v1 |

Full 40-hex revisions are in `src/sieve/eval/datasets.py::PINNED`. Fields used: TriviaQA
`question`, `question_id`, `answer.{value,aliases}`; NQ-open `question`, `answer` (list);
SimpleQA Verified `original_index`, `problem`, `answer`, `topic`; MMLU-Redux `question`,
`choices`, `answer` (index), `error_type`.

**Verification status.** Repo ids, revisions, licences, configs and field names were checked
against the Hub API and dataset cards; the SimpleQA Verified CSV header was read directly.
The Parquet/Arrow payloads of the other three sit on the Hub's Xet CDN, which this sandbox's
proxy does not reach, so row-level loading of TriviaQA, NQ-open and MMLU-Redux has **not** been
executed here. The parsers are unit-tested on rows shaped like the documented schema, and
`SIEVE_NETWORK_TESTS=1 pytest tests/eval/test_datasets.py` runs a live check where the Hub is
reachable. Run `prepare-data` once on Kaggle and read its summary before trusting counts.

Choices the pre-registration leaves open (all recorded in `items/datasets_manifest.json`):
TriviaQA is sampled from `validation` (the `test` split has no answers); duplicate questions
(normalised text) are dropped before sampling; MMLU-Redux keeps only rows annotated
`error_type == "ok"` (the loader raises if a subject has none); the number of cross-dataset
duplicate questions between TriviaQA and each shift set is counted and reported. Sampling is
order independent: the `n` items with the smallest `sha256(seed|item_id)`.

## Splits and the leakage guard

`splits.py`. Re-split `k` ranks the 6000 ids by `sha256(split_seed|k|id)` and cuts A/B/C. Each
`split_NNN.json` carries per-part hashes; `splits_index.json` carries every hash plus its own.
`SplitStore.load` recomputes all 50 splits from the seed and compares: an edited file, a
hand-picked "lucky" split with consistent hashes, or a changed index each raise
`SplitIntegrityError`.

Access rules, enforced by API and an append-only ledger (`split_access_ledger.jsonl`):

* `a_ids(k)` is free (calibrators, router training, grids, cost estimation, audit sheet).
* `open_b(k, capability=...)` needs a capability object that only `certify.py` holds, and works
  once per split. A second call, from any process, raises `SplitBAlreadyReadError`.
* `open_c(k, cert_hash=...)` needs the hash of a certification produced from that split's B.
* The runner sees `PromptItem` (no gold answers); labels live in `Item` and are looked up by id
  lists that come from the store.
* A test greps the package so that no other module calls `open_b` or names the capability.

### B-once / C-hash is a DETECTION control (tripwire), not access control

It makes accidents and casual double-looks fail loudly and leaves an auditable trail. It does
not stop a determined or careless insider. Hardening in this wave: every ledger append is atomic
(temp file + fsync + `os.replace`, under a file lock, check-then-append inside the lock); each
event records the code version (`SIEVE_GIT_COMMIT` or git HEAD, `+dirty` if modified) and the
config hash; `open_c` is once-only per certification (a second open from another process raises
`SplitCAlreadyReadError`; analysis re-runs pass `reopen=True`, recorded as `reopen_c`);
`certify_or_load` recomputes the bundle hash (thresholds, counts and embedded artifacts) and
refuses a stored certification that was edited.

Known bypasses (all remain): (a) `CERTIFICATION_CAPABILITY` is an importable module constant;
(b) deleting `split_access_ledger.jsonl` (or using a fresh workdir with the same config) re-enables
`open_b` with no durable trace; (c) `SplitStore._read(k)["B"]` and the plain `split_NNN.json` files
read B and C with no ledger event; (d) a forged `certified` ledger line unlocks `open_c`; (e) the
ledger is not hash-chained and its head is not recorded in the run manifest; (f) analysis is
CPU-only on the cache, so someone can change `Q_LEVELS`, the path rule or the template after seeing
H1/H3 and re-run in a new workdir; the recorded code version and config hash make that visible, not
impossible. Do not describe this as enforcement. Re-analysis does not re-read B: certifications
are stored in `certs/` and reused after the hash checks. If a process dies between `open_b` and
writing the certification, the split is `opened_uncertified` and must be treated as spent (record
a Deviation); the harness will not re-open it.

## Grading

* `em-alias-v1` (primary): `extract_answer` (first non-empty line, strip "The answer is" /
  "Answer:", trailing period) then TriviaQA-style normalisation (lowercase, accents, punctuation
  to space, articles dropped) and set membership against gold and all aliases. Verbose
  sentences are graded wrong; that is the false-wrong source the audit measures.
* `contains-alias-v1` (sensitivity only): whole-word alias containment in the first line.
* `mmlu-letter-v1`: letter parse (`B`, `(C) text`, `D.`, `The answer is B.`; `A cat is...` does
  not parse); unparseable replies count as wrong.
* SimpleQA: `Judge` protocol with labels CORRECT / INCORRECT / NOT_ATTEMPTED. The default is
  `UnavailableJudge`: items come back `Grade(None, "unavailable")`, are excluded from every
  risk number, the dataset is listed under `meta.ungraded_datasets`, and H5 cannot reach
  SUPPORTED. `BackendJudge` serves a judge through any `Backend` and refuses a model id equal to
  S or L; an unparseable verdict raises instead of becoming "incorrect". The judge prompt is our
  own wording, **not** the official SimpleQA / SimpleQA Verified autorater prompt. NOT_ATTEMPTED
  is not counted as wrong for risk (reported separately with the F1-style summary).
* Audit (`audit.py`): 200 items from split A, equal allocation over (tier x auto label),
  blind shuffled sheet plus a key. Analyser: false-wrong e1 = P(human correct | auto wrong),
  false-right e2 = P(human wrong | auto correct), exact Clopper-Pearson CIs per tier and pooled,
  population-weighted overall error with a Jeffreys interval, and a sensitivity bound
  `r(1-e1_lo) + (1-r) e2_hi` on the true wrong-answer rate (a diagnostic, never the guarantee).
  Kill test 6 passes only with overall error <= 3% and tier gap <= 2 pp, and is undecidable
  while rows are unlabelled.

## Runner, cache, cost accounting

`runner.py`, `cache.py`. One shard per cell (model, quant, KV, sampling, template). Each line is
one *attempted* generation with text, per-token logprobs and top-k, timings (runner wall time and
backend-reported), backend identity, config hash, pre-registration hash and a SHA-256 of its own
content. Non-finite logprobs are stored as strings so they survive and are caught as defects
downstream. Failed attempts are kept (`ok=false`) and their wall time is charged: an item's
cost is the sum of all its attempts; `wasted_time_share` is reported per cell.

Resumability: a key is done once a `final` line exists. On open, a torn last line from a killed
process is dropped; corruption anywhere else raises. Re-running generates nothing already
done (tested with call counts), and a process killed with `os._exit(137)` mid-run resumes to a
cache identical to an uninterrupted run (tested through the CLI). Retry seeds differ per
attempt; greedy outputs do not depend on them.

Model load/swap time is written to `cells/*.json` and is never added to per-item time.

Energy (`power.py`): `nvidia-smi -lms 100`, trapezoid integration of summed GPU power over host
monotonic timestamps, idle baseline subtracted, peak VRAM per window, throttle flag, ABAB block
interleaving (`abab_groups` in the config; all cells of a group resident at once). Without a GPU
the state is `no_gpu` and no numbers exist, never zeros. The analysis reports the coefficient of
variation of per-block power (Red-Team kill test 10 wants < 10%) and states that there is no wall
meter to cross-check, so joules are relative only. Energy is GPU-only.

Cost used in the frontier and cost-per-correct: per-item wall seconds from the cache
(`cost_unit: seconds`) or those seconds times the cell's measured net watts (`joules`).
Gates run S first, so every item pays S (the wasted S attempt on an escalated item is charged);
pre-routing baselines (random, query-only) run only the chosen tier; `S_only` and `L_only` run
one tier; self-consistency additionally pays the k sample generations on every item. Cost per
correct answer = total policy cost / number of answered-and-correct items.

## Methods, certification, baselines

All methods go through `certify.py` identically: threshold grid = distinct split-A quantiles
(12 levels from 0 to 0.95) plus "never"; `make_path` (Phase A) builds the fixed LTT path from A
(coverage >= 5%, Pareto frontier); B is opened once; `fixed_sequence_ltt` on the composed
S -> L -> abstain loss, selective risk, exact binomial p-values, `alpha = delta = 0.10`; among
certified points the cheapest per answered query **on A** is chosen; the artifact is built with
`build_artifact` and carries the B table hash. The primary signal is the highest S-tier AUROC on
A among max_prob, mean_logprob, neg_mean_entropy, mean_margin, chosen per re-split.

| Method | Definition |
|---|---|
| S_only / L_only | one tier gated by the primary signal; the other tier never runs |
| gate:max_softmax / mean_logprob / entropy / margin | same signal on both tiers (max_prob, mean_logprob, neg_mean_entropy, mean_margin) |
| gate:primary | primary signal on both tiers (headline gate) |
| random_router | S score = seeded hash of item id (uninformative), L uses the primary signal, pre-routed |
| query_router | TF-IDF (word uni+bigrams) + L2 logistic regression fit on A to predict S-correct; "RouteLLM-style", retrained, no embedding model was assumed available |
| self_consistency | S score = agreement of k=5 samples at T=0.7, only on the 1000-item subset; pays k extra decodes |
| oracle_router, oracle_abstainer | use labels; descriptive upper bounds, never certified or deployable |

"Random router at matched escalation rate" and the query router are compared to the gate at
matched coverage by choosing, on A, the uncertified (t1, t2) whose coverage and escalation rate
are closest to the gate's A operating point, then evaluating on C (paired). Their own
certified operating points are reported too. SimpleQA, NQ-open and MMLU-Redux are test-only:
certified thresholds from split k are applied unchanged.

## Statistics

Percentile bootstrap, 10,000 resamples (config `n_boot`), one resample shared by both methods
of a comparison (paired by item). Differences: accuracy on answered, cost per correct,
coverage. McNemar (exact binomial) on answered-and-correct. TOST for quantization: ΔAUROC and
ΔECE (equal-mass, 15 bins, Platt fit on A) with the 90% paired-bootstrap CI inside ±0.02;
`tost_paired_t` is also provided. Violation frequencies carry exact CIs. Per-stratum risk is
pooled over re-splits and flagged `low_n` below 30 answered items.

## Hypothesis rules as coded

`hypotheses.py` transcribes the pre-registered criteria; their numbers are in
`prereg.CRITERIA` and a test pins them to the document text. The report refuses to run if the
frozen body of the pre-registration (everything above `## Deviations`) differs from the hash
in the config or in the run manifest. Operational details the document leaves open are listed
here so that the Director can record any of them as a dated Deviation; none relaxes a criterion.

| ID | SUPPORTED | NOT SUPPORTED | Otherwise |
|---|---|---|---|
| H1 | violation frequency <= 15% AND CP95 upper bound <= 15% | point frequency > 25% | INCONCLUSIVE (also if < 10 certified re-splits; see SUPPORTED-WEAK below) |
| H2 | mean_logprob or a margin signal beats max_prob by >= 0.02 with paired CI above 0 | every single-token signal within ±0.02 | INCONCLUSIVE |
| H3 | non-inferior to L-only on the coverage-aware correct-answer rate (answered and correct per item; lower CI >= -0.02), cost per correct CI below 0, beats matched random and query routers (CI > 0) | random-router CI includes 0 or below, or cost advantage lost (point estimate >= 0) | INCONCLUSIVE |
| H4 | ΔAUROC >= 0.03 with CI above 0 (frontier benefit also needs k x cost_S < cost_L, reported) | CI upper bound < 0.03 | INCONCLUSIVE |
| H5 | NQ-open risk CI lower bound > alpha, SimpleQA risk point > NQ risk, drift alarm >= 80% | NQ risk CI upper <= alpha, or alarm < 80% | INCONCLUSIVE (always when no judge) |
| H6 | some config: f16-threshold violates alpha in > delta of re-splits and recalibrated does not | all configs equivalent (TOST) and transferred thresholds hold (<= delta) | INCONCLUSIVE (partial support in the numbers; needs >= 10 certified splits per frequency) |
| H7 | MMLU-Redux AUROC CI upper < 0.6 | CI lower >= 0.6 | INCONCLUSIVE |

Rules added after the red-team and eval audits (they operationalise, never edit, the frozen text):

* H1 SUPPORTED-WEAK: if the point frequency is <= 15% but the Clopper-Pearson 95% upper bound is
  above 15% (or missing/NaN), the verdict is reported as INCONCLUSIVE with `why` starting
  "SUPPORTED-WEAK". The 50 re-splits are resamples of one pool and not independent trials.
* H3 non-inferiority is on the correct-answer rate (answered and correct, over all items), so
  answering fewer items cannot buy non-inferiority; accuracy-on-answered at differing coverage is
  still reported (`acc_on_answered_non_inferior`) but not decisive. Every `lo > 0` / `hi < 0` test
  treats NaN as failure.
* H6 needs >= 10 certified splits for both the transferred and the recalibrated frequency, a
  transferred CP95 lower bound > delta, and a recalibrated frequency <= delta that is not credibly
  above delta; otherwise INCONCLUSIVE.
* The report refuses unless the config hash equals the run manifest's hash, and the loader
  refuses generation records produced under a different config hash.
* Self-consistency agreement is the share of samples agreeing with the served (greedy) answer and
  clusters use only the samples (no gold aliases).
* The sanity report (`python -m sieve.eval.providers.sanity`) exits non-zero if a tier has a
  `finish_reason == stop` rate below 95% or a token/text mismatch rate above 2% (C1 canary).
* Kill test 6 passes on the CP95 upper bound of the overall error (<= 3%), not the point estimate.

Open details and the choice made: the canonical split is re-split 0 for every bootstrap
(H1 uses all splits); "matches" in H3 is a ±0.02 non-inferiority margin (also the red-team's
example; config `noninferiority_margin`); H3 "matched coverage" is the A-matched point described
above; H5 "before violation" means the alarm needs scores only, so it is available before any
label; the drift reference is split-A scores (B must not be reread) and windows are 200
non-overlapping random windows; H6 "violates" uses the research review's definition
(> delta of re-splits), with delta = 0.10 rather than H1's 15% allowance; H2 candidates are
`mean_logprob`, `first_margin`, `mean_margin`.

## Red-Team kill-test mapping

| # | Test | Status in this harness |
|---|---|---|
| 1 | guarantee coverage | H1 over the pre-registered 50 re-splits (Red-Team asked for 200: raise `splits.n_splits`; splits are cheap, generation is not); per quant via H6 tables |
| 2 | fail-closed logprobs | defect enumeration is Phase A; here NaN/-inf/missing tokens round-trip through the cache and become `None` signals; repeat-run flip rate and max logprob drift reported (`nondeterminism`), no epsilon verdict |
| 3 | calibration-manifest mismatch | artifacts are keyed to cell identities; enforcing refusal is Gate/server work |
| 4 | cross-template / cross-dataset transfer | cross-dataset: shift section; cross-template: **not implemented** |
| 5 | non-inferiority | `noninferiority_vs_L_only` with energy/latency cost incl. wasted attempts |
| 6 | grader audit | `audit.py` (sheet + analyser + pass rule) |
| 7-9, 12 | server, privacy, cancellation, adversarial | out of scope (Backends/Server teams) |
| 10 | energy validity | ABAB blocks, idle subtraction, CV across blocks; no wall meter |
| 11 | quant vs noise null control | **not implemented** (needs a repeat-build cell) |

## Known gaps

No length-bias stratification, contamination probe or cross-template transfer; null control for
quantization; embedding-based query router; the real backend provider (written against the
`Backend` Protocol and the fake, never run against llama-server); the SimpleQA judge model
choice; the official SimpleQA autorater prompt. Mock numbers say nothing about real models.
