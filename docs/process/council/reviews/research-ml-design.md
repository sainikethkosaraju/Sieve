# Sieve v1.0 scientific core: independent design opinion (Principal AI Researcher + ML Engineer)

Basis: PLAN.md, the gap report, novelty_verification.md (VERIFIED primary reads), confidence_signals_calibration.md and benchmarks_eval_methodology.md (SNIPPET-ONLY, treat as unverified). No experiments were run; everything below is design, not result.

## 0. Claim the repo may state
"Given calibration data exchangeable with test traffic, a fixed model/quant/prompt/decoding configuration, and the stated grader, Sieve returns a threshold pair such that, with probability >= 1-delta over the draw of the calibration set, the grader-judged error rate among answered queries is <= alpha." That is all.
It is not per-query, not conditional on topic or difficulty, not about true factuality (only grader-judged), not robust to shift, not valid after changing quant, KV type, prompt or sampler, and not "no hallucinations". Conformal Cascade (2607.25018) already owns the MCQ cascade guarantee. Our new content is the open-ended, abstaining, noisy-grader, measured-hardware setting, and in that setting the guarantee is weaker, not stronger.

## 1. Signals, calibration, gate

Signals from llama-server (`n_probs=K`, temp 0, K>=20; set `post_sampling_probs` false or verify raw logprobs, since #31451/#29456/#31206 concern exactly this): per generated token t, p_t (chosen), top-K probs. Answer span = tokens after any fixed answer prefix (strip "The answer is"; otherwise first-token signals measure the preamble).
- Sequence: mean logprob (length-normalised), min logprob, sum logprob, answer-span entropy mean, truncated top-K entropy (flag as truncated, renormalised), margin = mean(top1 - top2 logprob).
- First-token: p_1, margin_1, H_1 over the first answer token. Cheap, streaming-compatible, probably weak for multi-token answers (a hypothesis, not a fact).
- Self-consistency (optional): k in {1,5,10} samples at T=0.7, normalise answers (lowercase, strip articles/punctuation), cluster by normalised string-match then alias match (no LLM-judge clustering in v1; NLI clustering is an ablation). Signal = largest-cluster share, cluster entropy. Cost = k decodes; must appear in the frontier.
- Combination: logistic regression on 3-6 signals vs a single best signal; report both.
Calibration (fit on split A only): temperature scaling / Platt on a scalar score, isotonic on score -> P(correct). Report ECE (equal-mass bins, 15), Brier, AUROC, AURC. Key honesty point: the risk guarantee depends only on threshold selection on split B; calibration affects ranking only if a learned combiner is used, and otherwise buys interpretability and cross-tier comparability (S vs L scores). Do not claim calibration "provides" the guarantee.

## 2. Threshold selection (exact algorithm)

Controlled statistic (default): selective risk R = P(wrong | answered) <= alpha. Rewrite as E[(w - alpha)·a] <= 0 with w = grader-wrong, a = answered-indicator. Variable is bounded in [-alpha, 1-alpha], so Hoeffding-Bentkus or WSR bounds apply. Unconditional risk P(wrong, answered) <= alpha is also offered (plain conformal risk control, monotone in thresholds, simpler, more conservative). Both reported; the conditional one is the user-facing default.

```
Inputs: full calibration table for split B: for each item i, S answer/conf s_i, L answer/conf l_i, wrong flags wS_i, wL_i
        (both models run on ALL items, so any policy can be evaluated offline)
        alpha, delta, grids T1 (S thresholds), T2 (L thresholds, +inf = never answer at L)
policy(t1,t2): if s>=t1 -> answer S; elif l>=t2 -> answer L (escalated); else abstain
loss_i(t1,t2) = (w_final_i - alpha) * a_final_i
H0(t1,t2): E[loss] > 0   (policy violates alpha)
p(t1,t2) = HB or WSR p-value for H0 from {loss_i}
order the grid from most conservative to least (t1 high, t2 high first; strictness-monotone path);
fixed-sequence test: walk the path, reject H0 at level delta each step, stop at first non-rejection
  (LTT, FWER <= delta; avoids the 2-D Bonferroni cost; the path must be fixed before seeing p-values)
valid set V = rejected points
choose in V the point minimising estimated cost (escalation rate, latency, J) measured on split A' (NOT B)
output (t1*,t2*), certificate {alpha, delta, n_B, n_answered_B, bound, config hash, grader id, split hash}
```
Cascade composition: loss is evaluated on the pipeline outcome, so there is no K·alpha union bound (CC's penalty). L's calibration is valid only for the pipeline, because L's score on S-rejected items is a selected subset; that is why L is run on all calibration items and the joint policy is tested. Fixed-sequence needs a 1-D monotone path; a 2-D frontier needs Bonferroni over the grid (state m in the certificate). Lower bound on feasibility: with zero errors among n_ans answered, the smallest certifiable alpha is about ln(1/delta)/n_ans; with n_B=2000 and delta=0.1 expect usable alpha>=0.03 at moderate coverage. Slack at alpha=0.1, n_ans~800 is roughly +-0.03.

What breaks:
- Noisy grader: certificate is for graded-wrong. If grader has false-"wrong" rate e1 and false-"right" rate e2, true risk can be off by about e2 in the dangerous direction. Mitigation: human-audit 200 stratified items, report e1, e2 with CIs, report an adjusted worst-case alpha (alpha + e2·coverage-ratio) as a sensitivity bound, never as the guarantee.
- Distribution shift: no guarantee. The drift alarm (KS/PSI on the confidence histogram, plus shadow-mode graded sample) is detection, with its own false-negative rate; a covariate shift in x that preserves the score histogram is invisible.
- Marginal only: a 3% average can hide 30% on a topic stratum. Report per-stratum risk; any Mondrian/class-conditional claim needs per-stratum n.
- Selection after the fact: if signal, prompt, alpha or grid are tuned on B, the certificate is void. Freeze before B; B is used once.
- Nondeterminism: GPU/llama.cpp batching changes logprobs slightly; exchangeability holds in distribution but repeated-run flip rate must be reported.
- Sampling signals (k samples) add randomness but stay valid, since the score is a fixed randomized function.

## 3. Evaluation protocol

Model pair (default): small 3B Q4_K_M (Llama-3.2-3B-Instruct or Qwen2.5-3B), large 8B Q4_K_M. Both fit on 2xT4 (one per GPU, which dodges co-residency; measure the single-GPU co-resident variant separately).
Datasets (short-form, deterministic grading where possible):
- TriviaQA rc.nocontext, 6000 sampled by seed: A=2000 (calibrators, router training, cost), B=2000 (LTT), C=2000 (test). Primary in-distribution.
- NQ-open 2000 test only (mild shift); SimpleQA Verified 1000 test only (hard shift; small models answer few correctly, expect near-total abstention at low alpha, which is an honest result); MMLU-Redux 1500 MCQ as a stress set, because 2610.23249 reports routing AUROC ~0.51 there and we must engage it.
- Budget: ~11.5k queries x 2 models single decode; at ~1-2 s/query this is <10 GPU-hours per config. Self-consistency k=5 on a 1000-item subset only. Keep within Kaggle's weekly quota by caching all raw responses and logprobs (replays are free).
Grading: normalised-alias exact match (TriviaQA/NQ aliases) as primary; SimpleQA 3-label judge prompt on SimpleQA only, with a judge model different from S and L; human audit of 200 items per grader. Report accuracy, risk-given-answered, coverage, F1-style SimpleQA metric.
Seeds/resampling: 50 random re-splits of the 6000 into A/B/C. Validation of the guarantee is empirical violation frequency: fraction of re-splits where test risk on C exceeds alpha; it should be <= delta (+ binomial noise on 50 runs, so report CI). Greedy decoding; 3 repeated runs of a 300-item subset for nondeterminism.
Statistics: percentile bootstrap over test items (10k), paired by item for all method comparisons (same items, same coverage-matched operating points), report differences with CIs, not just p-values; McNemar for matched-coverage accuracy; no claim from overlapping marginal CIs.
Baselines (all under the same LTT procedure, same alpha, delta): S only, L only, max-softmax gate, mean-logprob gate, entropy, margin, random router at matched escalation rate, oracle router (escalate iff S wrong and L right; upper bound), oracle abstainer, query-only router in the RouteLLM style (embedding + logistic regression on split A predicting S-correct; the released RouteLLM weights are trained on chat preference data, so retrain; label this "RouteLLM-style", not RouteLLM), and self-consistency gate.

## 4. Quantization experiment

Grid for S (3B): weights {f16, Q8_0, Q4_K_M, Q3_K_M or Q2_K} x KV {f16, q8_0, q4_0} = 12 configs (quantized KV needs flash-attention; Turing/T4 is supported but verify and log speed). L fixed Q4_K_M/f16 KV, then L KV q8_0 as a single extra. Same GGUF source and imatrix across quants. Add a long-context condition (about 1.5-2k-token few-shot or retrieved-passage prompt): KV-quant error scales with context, so with ~100-token QA prompts the expected effect is tiny, and a null there says nothing about long contexts. Measure per config: accuracy, AUROC/AURC, ECE, risk at fixed threshold fitted on f16, risk after per-config recalibration, coverage at alpha, VRAM, tok/s, J/query (nvidia-smi power integrated at 10 Hz, idle-subtracted).
Confounds: quantization changes accuracy and ranking quality, not only scale, so compare AUROC/AURC and matched-coverage risk, not raw threshold behaviour; truncated top-K entropy differs by distribution flatness; imatrix/data leakage; speed changes with kernel choice; Q2/Q3 may break answer formatting (grader-visible, count as wrong, report).
Hypothesis "quantization shifts the gate": operationalise as threshold fitted on f16 and applied to config q violates alpha (test risk CI upper bound > alpha, in >delta of re-splits) while recalibrated q does not. Falsified if, for all q, (a) paired-bootstrap ΔAUROC and ΔECE vs f16 lie within pre-registered equivalence margins (±0.02, TOST) and (b) transferred-threshold risk stays <= alpha at rate >= 1-delta. Partial support is also possible and must be reported as such (ranking preserved, scale shifted: recalibration of threshold only suffices).

## 5. Pre-registered hypotheses (alpha=0.10, delta=0.10 unless noted)
- H1 (validity): across 50 re-splits, risk(C) > alpha in <=15% of splits (delta + noise allowance) on TriviaQA. Negative: >25%, which means a bug or non-exchangeability.
- H2 (signal): mean-logprob or margin AUROC (correct vs wrong) on C exceeds max-softmax by >=0.02 (paired CI excludes 0). Negative: all single token signals within ±0.02; then say "signal choice is irrelevant" and stop mining.
- H3 (value of cascade): gated cascade at certified alpha reaches L-only graded accuracy-on-answered with lower cost than L-only; and beats random router and query-only router at matched coverage (paired CI > 0). Negative: gate indistinguishable from random router, or cost advantage lost once the wasted-S-attempt cost is counted.
- H4 (self-consistency): k=5 adds >=0.03 AUROC over best token signal; but the cost-adjusted frontier improves only if k·cost_S < cost_L. Negative result is likely and publishable.
- H5 (shift): calibrated on TriviaQA, risk on NQ-open exceeds alpha (CI), and on SimpleQA exceeds it by a larger margin; the drift alarm fires on >=80% of shifted windows of 200 queries before violation. Negative: guarantee holds under mild shift, or alarm blind (false-negative).
- H6 (quantization): §4. H7 (hard sets): on MMLU-Redux, gate AUROC <0.6, matching 2610.23249; report as limitation, not failure.
Negative results look like: feasible set V empty (no threshold certifies alpha), near-zero coverage, bounds too loose at n_B, CIs spanning zero. Publish those with equal prominence.

## 6. Five likeliest ways the headline is wrong
1. Grader error: alias exact match counts correct paraphrases as wrong (overstates risk, and the certificate is then loose), the LLM judge is lenient on near-misses (understates it); small models' verbose answers interact badly with both. Mitigate via human audit and sensitivity bounds.
2. Contamination and in-distribution optimism: TriviaQA/NQ are likely in pretraining of both models, so confidence on them is unrepresentative; the guarantee holds for that traffic only and would be violated on a real user mix.
3. Analyst degrees of freedom: choosing signal, prompt, alpha, grid or "best" Pareto point after viewing C or B voids the certificate; the "A/B/C once" discipline is the control, enforce it in code (hash-locked splits).
4. Cost accounting: energy/latency on T4 with power capping, warm caches, nvidia-smi sampling, wasted S attempts on escalated queries, model load/swap omitted. A frontier that hides these overstates savings; the "X% of the energy" sentence is the most likely to be wrong.
5. Reading the marginal guarantee as stronger than it is: average risk 0.10 can conceal topic strata at 0.3 or answerable/unanswerable mix effects, and a pair where L is barely better than S makes any router look good or bad trivially. Always report per-stratum risk and the oracle gap.

## Summary
1. Claim only: PAC-style, grader-judged, marginal, in-distribution selective-risk bound for a frozen configuration.
2. Gate selects (t1,t2) by fixed-sequence LTT on full-pipeline loss (w-alpha)·answered, no union bound; cost chosen on a separate split.
3. Eval: TriviaQA 2000/2000/2000 + NQ/SimpleQA Verified/MMLU-Redux shift sets, 50 re-splits, paired bootstrap, oracle/random/RouteLLM-style/max-softmax baselines.
4. Quantization: 12 S configs plus long-context condition; falsified by TOST equivalence and surviving transferred thresholds.
5. Seven pre-registered hypotheses with explicit negative outcomes; empty feasible set and chance-level routers are publishable.
6. Top risks: grader noise, contamination, analyst freedom, cost accounting, marginal-guarantee overreading.
