# Red-Team code audit of the built Sieve code (Phase A, B1, B2)

Seat: Independent Red-Team Engineer. Date: 2026-11-03. Code: working tree at HEAD 95ea59d (plus other seats' uncommitted edits in `benchmarks/`, `kaggle/`, `src/sieve/eval/providers/`, `tests/integration/`, which I did not review).
Method: read `signals, calibrate, gate, backends, router, server, observe, eval`; executed attacks against a real uvicorn instance (mock backends, `sieve.server.cli serve`) and in-process; wrote independent re-implementations of headline functions. No source or test file was edited. Scratch scripts: `/tmp/claude-0/-home-user-Sieve/296c653e-9125-5937-bc60-c353283a3b5e/scratchpad/rt/` (`atk1.py` .. `atk11.py`, `stat1.py`, `stat2.py`, `mkcfg.py`).
Limits: no real model, GPU or llama.cpp was available; anything needing a real LLM is marked UNCONFIRMED. Mock-backend results say nothing about real-model confidence.

## 1. Kill tests (my 12) and what I executed

| # | Test | Executed? | Result |
|---|---|---|---|
| 1 | Guarantee coverage | Own simulation, 1000 replicates, tied discrete scores, n_A=n_B=600, adversarial "worst certified policy" check against a 400k population | PASS on synthetic data: any-certified-policy true risk > alpha in 2.1% of replicates (limit 10%). Not run per quant / per dataset (needs real data). |
| 2 | Fail-closed logprobs | Executed: 12 mock defects x (tier 1 only, both tiers) through the full app | PASS (no defective completion ANSWERed; tier-1-only defects escalate, both-tier defects abstain). Real llama.cpp bugs #31451/#29456/#31206: UNCONFIRMED. Nondeterminism under concurrent batching is never re-tested at runtime (F-16). |
| 3 | Manifest mismatch | Executed runtime identity swap; MockTransport `/props` swap | PARTIAL FAIL: preflight detects a swap only after `refresh()`; poll gap up to `health_interval_s`; sampling-mismatch check is vacuous (F-09, F-11). |
| 4 | Cross-template / cross-dataset | Not executed (cross-template not implemented, per B2 notes) | NOT RUN |
| 5 | Non-inferiority | Code-read of H3 | WEAK: criterion not coverage-aware (F-07) |
| 6 | Grader audit | Code-read of `audit.py` | WEAK: pass rule uses point estimate (F-17) |
| 7 | SSRF / header injection | Executed on live server: 11 override body fields, `model: http://169.254.169.254/`, 8 override headers, CRLF/NUL/oversize/unicode request ids, raw-socket NUL header | PASS. Unknown body fields 400; override headers 400; nothing forwarded upstream; request ids sanitised; NUL header rejected by h11. |
| 8 | Canary privacy | Executed: canary prompt, user id, API key, streamed and non-streamed; grepped audit JSONL, `/metrics`, server log | PASS (0 occurrences). With `log_content=true` redaction misses several secret classes (F-14). |
| 9 | Cancellation and load | Executed: 100 TCP clients aborting mid-request (40 concurrent), 120-way flood, 2 MB body, 100k-deep JSON, 250 KB nested JSON, slow body, 800 half-open header connections | PASS except header-level slowloris (F-10). Aborts: inflight/queued gauges return to 0, audit shows `outcome=cancelled` with cost booked. Flood: 12x200, 108x429, no 5xx. Slow body: 408 at 10.0 s. `/slots` orphan check on a real llama-server: UNCONFIRMED. |
| 10 | Energy validity | Not executed (no GPU) | NOT RUN |
| 11 | Quant null control | Not implemented (B2 notes) | NOT RUN |
| 12 | Adversarial confidence | Executed: client `temperature/top_p/seed/stop/penalties` never reach backend; `logit_bias/tools/response_format/grammar/n_probs/cache_prompt` rejected; special-token role forging defanged (6 variants); `max_tokens=1` fails closed. Injected-prompt answered-wrong rate: UNCONFIRMED (needs real model) | Parameter side PASS. Structural risk F-12. |

## 2. Findings (ordered by severity)

### F-01 HIGH: the eval harness and the server cannot interoperate (certified artifacts are unservable; prompt and answer-span differ)
- Evidence: `src/sieve/router/prompt.py:46` (`ChatTemplate.hash`, 32 hex chars of a chatml/plain descriptor) is the `template_hash` every server backend gets (`src/sieve/server/runtime.py:make_backend`). The eval harness stamps `sieve.eval.prompts.template_hash()` (`src/sieve/eval/prompts.py:43`, 64 hex) into the cells and therefore into every artifact (`certify.py:_artifact` via `data.cells[...].backend`). Repro: `ChatTemplate("chatml","").hash` vs `prompts.template_hash()` -> lengths 32 vs 64, never equal; there is no config knob to change it.
- Also: eval generates from a raw few-shot prompt and sets `answer_start` with `find_answer_start` (`runner.py:122`); the server renders the client messages into chatml and hardcodes `answer_start=0` (`backends/llamaserver.py:267`, `backends/mock.py:183`). Calibration and deployment compute the signal on different prompts and different spans.
- Observed: any eval-certified artifact makes `/readyz` report `identity_matches_artifact=false`; every request escalates/abstains (fail closed, so no safety breach, but the product claim "certified cascade served by `sieve serve`" is unsupported). No test serves an eval-produced artifact.
- Fix: one shared prompt renderer and answer-span function used by both runner and router; template hash computed from it; end-to-end test: run mock eval -> load artifact in server -> expect `ready: true` and gate/offline-policy agreement.
- Verify: `tests/integration/` test that boots the app with the artifact from `results/mock`.

### F-02 HIGH: self-consistency signal leaks gold labels at calibration (H4 / "sc" method inflated)
- Evidence: `src/sieve/eval/data.py:96` builds `aliases = build_alias_table([item.answers])` from the GOLD answers and passes it to `compute_signals`, which merges sampled answers that match any gold alias into one cluster. Deployment (`router/core.py:_evaluate`) has no alias table.
- Repro (`atk4.py` T1): gold `["Paris","City of Light"]`, samples Paris/City of Light/Paris/City of Light/Paris -> agreement 1.0 with the table, 0.6 at deployment. Wrong samples: 0.2 both ways. The signal is higher exactly when the answer is correct.
- Impact: H4 AUROC and the certified "self_consistency" row are optimistic and do not transfer; H4 can print SUPPORTED wrongly.
- Fix: no gold-derived aliases in the signal; use label-free clustering (normalised string only, or a label-free alias source). Add a test asserting `data.py` signals are invariant to `item.answers`.

### F-03 MEDIUM-HIGH: self-consistency score ignores the answer actually served
- Evidence: `signals/__init__.py:42` and `signals/consistency.py:96-108`: agreement is computed over the k samples only; the greedy (served) text is never compared to the clusters.
- Repro (`atk4.py` T2): served text "Madrid", five samples all "Paris" -> agreement 1.0, gate ANSWERs "Madrid" at threshold 0.9.
- Fix: score = share of samples agreeing with the served answer (or include the greedy answer in the clustering and require membership in the top cluster); recalibrate.

### F-04 MEDIUM: a stored certification is trusted without re-verification
- Evidence: `eval/certify.py:285-293` (`certify_or_load`) checks only that `existing.bundle_hash` is in the ledger; the loaded thresholds and counts are never re-hashed against it.
- Repro (`atk7.py`): edit `t1` to -50 and `k_wrong_B` to 0 in `certs/split_000.json` -> accepted, analysis proceeds on the edited thresholds.
- Fix: recompute `sha256_obj({k, primary, certs(core), index})` on load and compare; refuse on mismatch.

### F-05 MEDIUM: B-once / C-hash discipline is a tripwire, not a guard (documented, but it is a headline claim)
- Evidence and repro (`atk6.py`): (a) `CERTIFICATION_CAPABILITY` is a public module constant; any code can import it and call `open_b` once. (b) deleting `split_access_ledger.jsonl` re-enables `open_b`. (c) `SplitStore._read(k)["B"]` and the plain `split_NNN.json` files read B with no ledger event. (d) same for C. (e) appending a forged `certified` ledger line unlocks `open_c`. The static test only greps for `open_b` call sites.
- Impact: nothing in code prevents or durably evidences a second look at B. Acceptable only if described as accident detection (the B2 notes do say so); the README/pre-registration must not present it as enforcement.
- Fix: hash-chain the ledger (each line includes previous line hash), record the final ledger head in the run manifest and in the committed results, make `_read` private-by-name unreachable via a function-local closure, and have `analyse` fail if the chain, the number of `open_b` per split (exactly 1) or the certification hashes disagree.

### F-06 MEDIUM: H6 can print SUPPORTED on one certified split
- Evidence: `eval/hypotheses.py:292` (`if ft > cfg.delta and fr <= cfg.delta`) has no minimum number of certified splits (H1 has `>=10`).
- Repro (`atk9.py`): recalibrated `violation_freq=0.0` from `n_certified=1` of 50, transferred 0.5 -> `H6 SUPPORTED`. Also a transferred frequency from 3 splits counts.
- Fix: require n_certified >= 10 for both frequencies (else INCONCLUSIVE); compare using CP bounds, not point values.

### F-07 MEDIUM: H3 "matches L-only accuracy-on-answered" is satisfiable by answering less; NaN passes the random-router test
- Evidence: `eval/sections.py:203` non-inferiority uses accuracy-on-answered of the gate vs certified L-only at their own, different coverages; the pre-registered wording has the same flaw (docs/PREREGISTRATION.md H3 row). A gate that answers the easiest 30% beats L-only answering 70% on this metric by construction; coverage is never part of H3. Kill test 5 asked for correct-answer-rate (or coverage-matched) non-inferiority.
- Plus `hypotheses.py:147` `if r["lo"] <= 0` is False for NaN, so a NaN random-router CI counts as "beats random router". Repro (`atk8.py`): NaN CI -> `H3 SUPPORTED`. Needs an all-NaN bootstrap (zero answered in every resample), so low likelihood.
- Fix: add a coverage-aware co-primary (answered-and-correct rate or accuracy at matched coverage, with margin); treat non-finite bounds as failure in every comparison (`not (lo > 0)`).

### F-08 MEDIUM: H1 / H2 verdict logic is looser than it looks
- H1 (`hypotheses.py:72`) tests the point violation frequency against 15% (> delta=10%) and ignores the printed CP interval. Repro (`atk8.py`): 7/50 violations, CP95 [0.058, 0.267], upper bound above the "NOT SUPPORTED" line of 25% -> `SUPPORTED`; 10 certified of 50 splits (40 empty) also SUPPORTED.
- The 50 re-splits are resamples of one 6000-item pool, so they are strongly dependent; 50 replicates are not 50 independent trials of the guarantee (which is over new draws). The frequency is also conditional on certification, whereas the guarantee is unconditional (in my simulation unconditional 2.1% corresponds to roughly 10% conditional), so H1 can falsely refute and cannot falsely confirm much beyond pool luck.
- H2 (`hypotheses.py:104`): `any()` over three candidate signals, each at 95% CI, no multiplicity adjustment; repro: one of three at +0.021 with lo > 0 -> SUPPORTED.
- Fix: base H1 on the CP upper bound (SUPPORTED only if upper <= a pre-set value), add a note that re-splits are not independent; apply Holm over H2 candidates.

### F-09 MEDIUM: backend identity is polled, not verified per request
- Evidence: `backends/llamaserver.py:141` (`info` reads cached `_props_model_file/_build`), `:188` (`refresh` is the only place `/props` is read), `server/runtime.py:_maintain` (default every 30 s), `router/core.py:182` (identity captured before the call).
- Repro (`atk11.py`): upstream swapped to another build/model; after `generate()` the identity is unchanged; only `refresh()` flips it. A swap between two polls is served under the old identity for up to `health_interval_s`. `model_hash` is operator-declared; `expected_model_file` compares basename only (`/other/dir/qwen-q4.gguf` accepted). Mock: `sm.info` replaced during a generation -> answered at tier 1 (`atk5.py`).
- Fix: refresh `/props` on a short TTL inside `generate` (or hash a `/props` fingerprint into every response), compare identity after the call as well as before, and consider checking file size/mtime or an operator-supplied digest.

### F-10 MEDIUM: header-level slowloris holds connections indefinitely; docs claim otherwise
- Evidence: `server/cli.py:37-45` sets only `timeout_keep_alive` (applies after a response) and `h11_max_incomplete_event_size`. `docs/server.md:130` says header slowloris is left to those settings.
- Repro (`atk3.py`): 800 connections sending a partial request line stayed open after 12 s (server stayed responsive; limit is the file-descriptor limit, 20000 here). No timeout ever fires.
- Fix: `uvicorn` `timeout_graceful`/a header-read timeout (e.g. run behind a proxy that enforces it, or add a connection-level timeout), cap connections (`limit_concurrency`), and correct the doc.

### F-11 LOW-MEDIUM: sampling-mismatch protection is vacuous; client `max_tokens` is outside the identity
- Evidence: `llamaserver.py` returns `sampling=params` (the requested params, not what the server applied); the gate compares that echo to the artifact, which always matches unless the router itself is misconfigured. Server-side launch flags (`--repeat-penalty`, DRY, XTC, samplers order) are never observed. `max_tokens` is excluded from `SamplingParams.key()` yet a client can set it (`schemas.py:max_tokens`), changing which outputs are TRUNCATED defects relative to calibration (`atk10.py`: backend saw max_tokens 3). Effect on real logprobs: UNCONFIRMED.
- Fix: read applied sampler settings from `/props` `default_generation_settings` and compare; pin max_tokens to the calibrated value or include it in the key.

### F-12 LOW (structural risk, UNCONFIRMED without a real model): confidence is computed over the whole generation / first token
- `answer_start=0` in the server path (see F-01); evidence D-016 (7B opens with " The", p=0.78). A client that induces a fluent confident preamble raises `max_prob`/`mean_logprob` independent of correctness. The calibration guarantee assumes exchangeable traffic, so this is outside the certificate, but there is no server-side detection beyond the drift alarm. Needs a real-model injection test (kill test 12 second half).

### F-13 LOW: empty or whitespace-only answers are ANSWERed
- Evidence: `signals/logprob.py:validate_completion` flags EMPTY_ANSWER only when there are no tokens. Repro (`atk4.py` T3): `text=""` (or `"   \n"`) with one confident token -> defects `()`, decision ANSWER. Statistically harmless (graded wrong at calibration) but a user-visible failure. Fix: treat whitespace-only text as EMPTY_ANSWER.

### F-14 LOW: opt-in content logging redaction misses common secrets
- `observe/redact.py`: with `log_content=true` SSNs, phone numbers, AWS access keys, GitHub tokens and `password=` strings pass through (repro inline in session: only email, 13-19 digit numbers, `sk-`-style and Bearer were redacted). Default (off) is clean. Fix: broaden patterns or state clearly that redaction is best-effort.

### F-15 LOW: escalation guard is global and gameable (availability only)
- `router/guard.py:36`. Repro (`atk5.py` T5): rate 0.5, window 20, 30 hard prompts -> 11 `escalation_shed` abstentions; any concurrent legitimate hard request is shed too. `max_tokens=1` forces small+large calls and abstain per request (2 attempts burned, `atk10.py`). Fail-closed, so no accuracy impact. Fix: per-key budgets or document as global.

### F-16 LOW (UNCONFIRMED): logprob determinism checked only at startup, sequentially
- `backends/selftest.py` runs idle and sequential (3 repeats, eps 1e-3). Issue #31206-style nondeterminism appears under batched concurrent decoding and would not be seen; calibration data was also generated sequentially. Needs a real llama-server load test.

### F-17 LOW: kill-test-6 pass rule uses point estimates
- `eval/audit.py:166`: `overall_error["all"][0] <= 0.03` and `tier_gap <= 0.02` ignore the CI printed next to them; with 200 labels a 3% point estimate has a CP upper bound near 6%. Fix: compare the upper bound.

### F-18 LOW: quadratic regex in grading
- `eval/grading.py:69` `_STATED`: `answer\s*(?:is|:)?\s*` has two adjacent `\s*`. Repro (`atk4.py`): "answer"+n spaces+"x": 8k -> 0.58 s, 16k -> 2.77 s, 32k -> 10.5 s. Reachable only through model output (offline eval), so low. Fix: collapse to `answer\s*(?:is|:)?\s*` -> `answer(?:\s*(?:is|:))?\s*` or cap input length.

### F-19 LOW: artifact hash is integrity, not authenticity; validator does not check internal consistency
- `calibrate/store.py:117-140`. Repro (`atk4.py` T8): edit threshold to -1e300 and recompute `content_hash` -> loads; `alpha=0.5`, `k_wrong_cal=99/100`, `risk_upper_bound=0.99` with a recomputed hash also loads. Stale hash, edited identity and NaN threshold are rejected. `expected_artifact_id` is optional and unset by default. Fix: validate `risk_upper_bound <= alpha`, bounds sanity, require a pin (or signature) outside loopback mode.

### F-20 LOW: cache record hashes are self-referential
- `eval/cache.py:171`: each line embeds `sha256(body)`; edit-and-rehash passes, no chain or MAC. Detects corruption, not tampering (code-read, not executed).

### F-21 LOW: test suite claim not reproduced
- B2 notes claim 466 passed, 1 skipped. Current tree: `tests/eval/test_pipeline.py::test_unavailable_judge_is_reported_not_substituted` fails deterministically (expects "unavailable" in the H5 `why`, gets "NQ-open not run, ungraded or nothing answered"). Not caused by uncommitted edits I could identify.

### F-22 INFO
- `/healthz` and `/readyz` are unauthenticated and disclose `artifact_id`, `config_pin`, backend health detail (`app.py`).
- `grading.is_wrong` counts SimpleQA `not_attempted` as not wrong by default (`grading.py:238`): risk on SimpleQA is understated relative to a "no wrong answer" reading.
- Audit file has no rotation (`observe/audit.py`); JSON-encode failures only increment an unexported `errors` counter.
- Cost accounting: server ledger charges small, large, samples, errors and cancelled attempts (verified in audit records); error/cancelled attempts record 0 tokens (understated wasted tokens) and queue wait is not charged. Eval costs include failed attempts via `total_wall_s`. No omission affecting the headline found.
- Unsafe deserialization: none (`yaml.safe_load`, `np.load(allow_pickle=False)`, no pickle/eval/shell). Path traversal: no request-controlled path exists; all paths are operator config (N/A).

## 3. Independent re-implementations (all agree)

Script `stat1.py`; each compared against code on randomised inputs:
1. `ltt.binom_pvalue` vs explicit pmf sum: max err 4.9e-15. `ltt.cp_upper` vs bisection: 1.2e-15. `min_certifiable_alpha`: (1-a)^n = 0.1 exactly.
2. `fixed_sequence_ltt` vs my own walk (loop over items, NaN->-inf, tie via `>=`): 0 mismatches over 400 random tables x 2 alphas, including NaN confidences, ties and NEVER thresholds. No off-by-one at the `p <= delta` boundary.
3. `metrics.aurc` vs exhaustive average over all tie-break permutations (n <= 7): max err 3.3e-16.
4. `metrics.auroc` vs pair counting with ties 1/2: err 0. `ece_equal_mass` vs independent binning: 0.
5. `stats.batch_auroc` and `batch_ece` vs per-resample scalar functions: 0 and 3.9e-16.
6. `stats.mcnemar` vs `scipy.stats.binomtest`: 0. `tost_paired_t` vs two scipy one-sided t tests: 3.3e-16.
7. `hb_pvalue` vs direct formula: 1.7e-14.
Statistical machinery is correct; the defects are in what is fed to it (F-02, F-03), how verdicts are decided (F-06..F-08), and the access discipline (F-04, F-05).

## 4. Summary table

| ID | Sev | Component | file:line | Status |
|---|---|---|---|---|
| F-01 | High | eval <-> server integration (template hash, prompt, answer span) | router/prompt.py:46; eval/prompts.py:43; backends/llamaserver.py:267; eval/runner.py:122 | Confirmed |
| F-02 | High | SC signal uses gold aliases | eval/data.py:96 | Confirmed |
| F-03 | Med-High | SC agreement ignores served answer | signals/__init__.py:42; signals/consistency.py:96 | Confirmed |
| F-04 | Medium | stored certification not re-verified | eval/certify.py:285 | Confirmed |
| F-05 | Medium | B-once / C-hash bypassable (6 ways) | eval/splits.py:62,100-200 | Confirmed (documented as tripwire) |
| F-06 | Medium | H6 SUPPORTED with 1 certified split | eval/hypotheses.py:292 | Confirmed |
| F-07 | Medium | H3 not coverage-aware; NaN passes | eval/sections.py:203; hypotheses.py:147 | Confirmed (NaN path needs rare input) |
| F-08 | Medium | H1 ignores CI, dependent re-splits; H2 no multiplicity | eval/hypotheses.py:72,104 | Confirmed |
| F-09 | Medium | identity polled, basename-only file check | backends/llamaserver.py:141,188; server/runtime.py | Confirmed |
| F-10 | Medium | header slowloris, doc claim false | server/cli.py:37 | Confirmed |
| F-11 | Low-Med | sampling echo vacuous; max_tokens outside identity | backends/llamaserver.py; server/schemas.py | Confirmed (effect UNCONFIRMED) |
| F-12 | Low | whole-generation / first-token confidence, prompt-induced | llamaserver.py:267 | UNCONFIRMED (needs real model) |
| F-13 | Low | empty/whitespace answer ANSWERed | signals/logprob.py | Confirmed |
| F-14 | Low | redaction gaps (log_content on) | observe/redact.py | Confirmed |
| F-15 | Low | guard global / amplification | router/guard.py:36 | Confirmed |
| F-16 | Low | determinism not monitored under load | backends/selftest.py | UNCONFIRMED |
| F-17 | Low | kill-test-6 rule on point estimate | eval/audit.py:166 | Confirmed |
| F-18 | Low | quadratic regex | eval/grading.py:69 | Confirmed |
| F-19 | Low | artifact hash not authenticity; no consistency check | calibrate/store.py:117 | Confirmed |
| F-20 | Low | cache hash not chained | eval/cache.py:171 | Code-read only |
| F-21 | Low | 1 test fails at HEAD; "466 passed" not reproduced | tests/eval/test_pipeline.py:164 | Confirmed |
| F-22 | Info | unauth probes, not_attempted, audit rotation, cost minor notes | various | Confirmed |

Passed with evidence: SSRF/URL override, header injection, CRLF/NUL request ids, malformed/oversized/deep JSON, slow body, client-disconnect cancellation and cost booking, concurrency flood (429, no 5xx), canary privacy across audit/metrics/log, special-token defanging, client sampling stripping, fail-closed defect matrix, unsafe deserialization, statistical primitives (8 re-implementations).

## 5. Release implications
- Blocks the headline "certified cascade served end to end": F-01.
- Blocks any H4 / self-consistency claim: F-02, F-03.
- Blocks quoting H3, H6 verdicts without the fixes in F-06, F-07 (and H1/H2 wording per F-08).
- Does not block a loopback demo server: F-10 needs a proxy for anything exposed.
