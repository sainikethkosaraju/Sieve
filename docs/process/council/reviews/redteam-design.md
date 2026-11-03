# Red-Team design review: attack on the Sieve plan (pre-build)

Seat: Independent Red-Team Engineer. Sources read: PLAN.md v0.2, landscape report, novelty_verification.md (first part). No other reviews read. Nothing below was executed; all items are risks or proposed tests, not confirmed defects.

## 1. Claims most likely false, overstated or unfalsifiable

| # | Claim | Why it is weak | Evidence that settles it |
|---|---|---|---|
| C1 | "Matches large-model accuracy at X% of compute" | "Matches" is unfalsifiable without a margin. On TriviaQA/PopQA an 8B beats a 1-3B by 15-30 pp on tail facts; a cascade can only match if the large tier handles most hard items, so energy saving is bounded by the escalation rate. Latency also includes the wasted small attempt. | Pre-register a non-inferiority margin (e.g. accuracy within 2 pp, paired bootstrap lower CI bound) and report energy per *correct answer* including wasted attempts. Falsified if the margin fails at any coverage that saves >=20% energy. |
| C2 | "Risk guarantee: hallucination <= alpha" | Conformal/risk control is marginal and needs exchangeability; the plan itself admits shift breaks it. With n_cal of a few hundred and alpha near 5%, the CI on the threshold is wide, and "hallucination rate" is ambiguous (answered-and-wrong vs error-among-answered). Also the grader's false-negative rate becomes part of the guarantee. | Report empirical violation frequency over >=200 random cal/test re-splits (should be <= delta), per-domain and per-quant. State which definition. Fail if violations exceed delta or only the pooled mean is shown. |
| C3 | "Quantization shifts the gate" | Likely true but trivial and confounded: quant changes accuracy, entropy and logprob scale together. Contested priors (2405.00632 vs 2508.16785). Without effect size vs seed/sampling noise, any difference is "found". Must separate shift in AUROC (ranking) from shift in threshold (calibration). | Same questions, same prompts, deterministic decoding, >=3 quants (Q8, Q5_K_M, Q4_K_M, Q3/Q2) plus KV q8/q4; report AUROC, ECE and threshold at fixed alpha with paired bootstrap CIs, and a null control (same model re-run on a different llama.cpp build/seed). |
| C4 | "Streaming-compatible gating" | Confidence is known only after generation; buffering defeats streaming, and abort-and-escalate leaks partial output. Likely reduces to "buffer then stream". | Measure TTFT with and without gate; show early-token probe AUROC vs full-sequence AUROC. Falsified if early-token AUROC < ~0.6 or TTFT penalty ~ full generation time. |
| C5 | "Fraction of compute/energy" on Kaggle | 2x T4 shared hosts, no wall meter, NVML counters coarse; T4 lacks FP8/bf16 and has 16 GB, so conclusions may not transfer to "consumer GPU". | One wall-meter-validated config elsewhere, idle-subtracted, repeated runs with CV reported. Drop "consumer GPU" generalisation if only T4. |
| C6 | Novelty (calibrated cascades, abstention, conformal) | Plan already concedes. Residual novelty rests on weak-evidence "open" items (KV-quant, streaming). | Dated search log; engage 2610.23249 on its failure case (MMLU-ProX-Lite). |

## 2. Gate failure scenarios and tests to write

1. **Length bias.** Sum logprob falls with length; mean logprob rewards long hedging answers; max-prob on first token favours short answers. Test: stratify AUROC by answer length bucket; require AUROC within 0.05 of pooled, and verify the gate does not correlate with length beyond the correlation of correctness (partial correlation test).
2. **Prompt-format effects.** System prompt, chat template, few-shot, "answer briefly", or the instruct template changing logit distributions. Test: 5 paraphrased prompt templates on the same 1k questions; threshold calibrated on template A applied to B-E; report realised risk. Fail if realised risk > alpha + 2 pp on any.
3. **Calibration drift / shift.** Calibrate on TriviaQA, deploy on PopQA/NQ/domain questions; also unanswerable and adversarial prompts (false-premise). Test: cross-dataset risk table; drift alarm must fire (detection power measured) before violation exceeds 2x alpha.
4. **Quantization/KV.** Test as C3, plus assert a threshold file is refused when model hash/quant/build/KV type differs from the calibration manifest (silent reuse is the likeliest real-world failure).
5. **Temperature/sampling.** Logprobs reported are pre- or post-temperature/top-k/top-p depending on backend (vLLM note in report). Test: run T in {0, 0.2, 0.7, 1.0}; verify gate logprobs equal the raw-logit softmax, not the adjusted one; calibrate with T=0 and refuse/recalibrate if user sends T>0. Self-consistency at T>0 changes cost accounting.
6. **Tokenizer / logprob bugs.** Issues #31451 (multi-token UTF-8 logprobs dropped), #29456 (uncapped n_probs), #31206 (nondeterministic /v1/completions logprobs), #12968. Tests: (a) answers with CJK/emoji/accents: assert len(logprobs) == len(tokens), else fail closed (abstain/escalate); (b) same request 20x with fixed seed: logprob variance must be below epsilon, otherwise gate is nondeterministic and the guarantee is void; (c) top_logprobs sums <= 1 and top-1 equals sampled token at T=0; (d) NaN/-inf/missing logprobs handled; (e) n_probs=1e9 request does not OOM. Gate must **fail closed**.
7. **Adversarial prompts.** Confidently-wrong prompts (popular misconceptions), prompt injection ("respond with the token X with certainty"), forced single-token answers that inflate max-prob. Test: injected prefix that makes the small model emit a high-probability wrong answer; measure answered-wrong rate. Also a gate-gaming check: the proxy must not let clients set confidence-relevant parameters (logit_bias, temperature, max_tokens=1, grammar) that distort the gate; either strip or recalibrate.
8. **Large-tier wrongness.** Escalating items the large model gets wrong without abstaining breaks the risk claim (Signed Rescue point). Test: risk computed on final answers across both tiers.
9. **Multi-answer / free-form.** Token confidence of the *answer span* vs the whole generation (preamble "Sure, the answer is" has p~1). Test: answer-span extraction vs whole-sequence gate; fail if whole-sequence AUROC is used while span AUROC is much lower.

## 3. Server-level attacks and failures

- **Malformed requests:** invalid JSON, wrong types, negative/NaN max_tokens, n>1, stream with unsupported fields, tool_calls, unknown model, duplicate keys, 10k-element messages array, deeply nested JSON. Expect 4xx in OpenAI error schema, never 500/traceback or hang.
- **Huge prompts:** body-size cap (e.g. 1 MB default), token-count cap before forwarding, context overflow returns clean error; cascade doubles the prefill so cap applies per tier.
- **Slow clients / slowloris:** header and body read timeouts, per-connection write timeout when streaming, bounded buffers (buffered-start gating holds whole generations in memory).
- **Streaming cancellation:** client disconnect must cancel the upstream generation on both tiers (otherwise GPU keeps burning, which also corrupts energy numbers); test no orphaned slots in llama-server /slots after 100 aborted streams; cancel during escalation handoff.
- **Resource exhaustion:** concurrency limit and queue with backpressure; two-model VRAM co-residency under concurrent requests (OOM or swap thrash); per-request wall-clock budget; escalation amplification (every request costing small+large = 2x DoS multiplier). Load test at 50 concurrent with mixed sizes; server must shed load, not crash.
- **SSRF via backend URLs:** backend URLs come from config; if any API/header/model-name field can select or override a backend, an attacker reaches cloud metadata (169.254.169.254), localhost admin ports, file://. Require static allowlist from config only, reject redirects, disable per-request base_url. Test: request with `X-Backend`/`model: http://169.254.169.254/` must be rejected.
- **Header injection / forwarding:** do not forward client Authorization, Cookie or hop-by-hop headers upstream; strip CRLF in any value echoed into headers (e.g. request-id, model names); log-injection through newlines in prompts into audit JSONL.
- **Secrets and privacy in logs:** Observe audit trail stores "confidence, tier, calibration hash"; the temptation is to store prompts/answers. Default must be no prompt content, hashed IDs only, opt-in redaction. Test: run with a canary string in the prompt and API key; grep all logs, metrics labels, traces, exception messages and crash dumps; must find zero. Prometheus label cardinality from user-controlled values (model name, prompt) is also a memory DoS.
- **Auth:** default bind 127.0.0.1; if 0.0.0.0, require API key; constant-time compare; metrics endpoint not public.
- **Supply chain:** pinned model hashes (GGUF from HF), no `trust_remote_code`, pickle-free calibration files (use JSON, signed manifest).

## 4. Benchmark integrity risks

- **Contamination.** TriviaQA/NQ/PopQA are in pretraining of every Llama/Qwen/Gemma; small-model "knowledge" may be memorisation, inflating confidence-accuracy coupling and not generalising to fresh queries. Mitigate: add post-cutoff questions (SimpleQA Verified, newly written set), report a contamination probe (verbatim completion of question stems), and don't generalise beyond "public QA".
- **Grader false negatives.** Exact match with aliases under-credits correct verbose answers, so "hallucination rate" is inflated and inconsistently across tiers (large models are more verbose). SimpleQA grader had 15/56 false negatives. Require human audit of >=200 items stratified across tiers, with grader error rate reported per tier and its CI propagated into the guarantee. LLM judge from the same family as a tier is self-preference-biased.
- **Cherry-picked thresholds / test-set tuning.** Threshold, signal choice (entropy vs margin vs mean-logprob), and prompt are researcher degrees of freedom. Pre-register the signal, alpha grid, and splits in a committed file before touching test; lock a test split never used for any tuning; report all signals tried (forking paths count).
- **Small n.** At n=1000, CI approx +-3.1 pp; risk at coverage 20% rests on ~200 items. Use paired question-level bootstrap and report n at each operating point; don't draw Pareto curves with overlapping CIs.
- **Cal/test leakage.** Same question paraphrases/duplicates across splits (PopQA entity overlap); split by entity, dedupe.
- **Energy artefacts on shared Kaggle.** Co-tenancy, thermal throttling, clock boosting, varying host CPU draw, NVML update period (~100 ms) and counter repeats; GPU-only joules ignore CPU/RAM (CPU-offloaded layers in llama.cpp shift energy off the measured device, flattering the large tier). Measure idle baseline per session, interleave configs (ABAB) rather than sequential, log clocks and temperature, >=5 repeats, report CV; flag runs with throttle reasons. Warm vs cold model load counted consistently. Kaggle sessions can be killed; results must be resumable and cached.
- **Metric confusion.** Two "hallucination rate" definitions, ECE binning choices, AURC flaws. Name both, use equal-mass bins.
- **Closest-work engagement.** 2610.23249 failing on MMLU-ProX-Lite is a warning that the gate may fail off easy QA; include one non-memorised, reasoning-style set or state the scope limit.

## 5. Prioritised kill tests (each pass/fail; a failure blocks release of the stated claim)

| P | Test | Pass condition |
|---|---|---|
| 1 | Guarantee coverage: 200 random cal/test re-splits, per quant, per dataset | Fraction of splits with realised risk > alpha is <= delta (+ binomial slack) |
| 2 | Fail-closed logprobs: missing/NaN/truncated/mismatched-length logprobs, UTF-8 multi-token answers, nondeterministic repeats | Zero answered outputs without valid confidence; repeat variance < pre-set epsilon |
| 3 | Calibration-manifest mismatch: wrong quant, KV type, build, model hash, temperature | Server refuses or forces escalate/abstain, 100% of mismatches |
| 4 | Cross-template and cross-dataset transfer | Realised risk <= alpha + 2 pp, or drift alarm fires before 2x alpha |
| 5 | Non-inferiority: cascade vs large-only, paired bootstrap on locked test split | Lower CI bound of (cascade acc - large acc) >= -margin at an operating point with >=20% energy saving incl. wasted attempts; else headline reworded |
| 6 | Grader audit: 200 human-labelled items | Grader error <= 3% overall and no tier differs by > 2 pp; otherwise report with corrected bounds |
| 7 | SSRF / header injection suite | No request reaches a non-configured host; no client credentials forwarded; no CRLF reflected |
| 8 | Canary privacy: canary prompt, API key, across all sinks | Zero occurrences in logs, metrics, traces, errors |
| 9 | Cancellation and load: 100 aborted streams, 50 concurrent mixed, 1 MB / over-context bodies, slowloris | No orphaned generation slots, no OOM/500, bounded memory, clean 4xx/429 |
| 10 | Energy validity: ABAB repeats, idle-subtracted, one wall-meter cross-check | CV < 10%, NVML vs wall within 15%, else report as relative only |
| 11 | Quant effect vs noise: null control (same quant, different build/seed) | Quant-induced threshold shift exceeds null-control shift with non-overlapping CI; else claim "shifts the gate" dropped |
| 12 | Adversarial confidence: injected/forced-answer prompts and client-set logit_bias/temperature | Answered-wrong rate stays <= alpha, or parameters stripped and documented |

## Summary
1. Headline "matches large model" is unfalsifiable until a non-inferiority margin and energy-per-correct-answer including wasted attempts are pre-registered.
2. The risk guarantee is the weakest link: marginal, exchangeability-dependent, grader-error-dependent; test with 200 re-splits and cross-dataset/template transfer.
3. Gate must fail closed on logprob defects (#31451, #29456, #31206), temperature/sampling mismatches, and calibration-manifest mismatches (quant, KV, build).
4. Server risks to test first: SSRF/backend override, cancellation leaks that also corrupt energy data, escalation DoS amplification, and prompt/secret leakage via audit logs and metric labels.
5. Benchmark integrity: memorised TriviaQA/PopQA, verbose-answer grader false negatives, forking-path thresholds, and Kaggle T4 energy noise need locked splits, a 200-item human audit, and ABAB repeats.
6. Twelve kill tests above are prioritised; tests 1-3 and 5 block the headline, 7-9 block any server release.
