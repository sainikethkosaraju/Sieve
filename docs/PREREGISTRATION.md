# Sieve v1.0 pre-registration

Frozen before any headline experiment on the Kaggle T4s. The commit that adds this file is
the freeze point; the harness records this file's hash in every result. Changes after the
freeze are allowed only as dated **Deviations** at the bottom, never by silent edit.
Derived from `docs/council/reviews/research-ml-design.md` (D-007, D-011).

## Claim that may be stated
Given calibration data exchangeable with test traffic, a fixed model/quantization/KV-type/
prompt/decoding configuration and the stated grader, Sieve returns thresholds such that,
with probability >= 1-delta over the calibration draw, the grader-judged error rate among
answered queries is <= alpha. Nothing stronger: not per-query, not about true factuality,
not robust to shift, not valid after changing any frozen element.

## Configuration (frozen)
- Small tier S: Qwen2.5-3B-Instruct (GGUF; weights f16, Q8_0, Q4_K_M, Q3_K_M/Q2_K ladder).
  Fallback if the ladder is unavailable: Qwen2.5-1.5B-Instruct (ladder verified to exist).
- Large tier L: Qwen2.5-7B-Instruct Q4_K_M, f16 KV (extra arm: KV q8_0).
- Backend: llama-server at a recorded llama.cpp commit; greedy decoding (temperature 0);
  logprobs via `n_probs` >= 20, validated by the fail-closed startup self-test.
- alpha = 0.10, delta = 0.10 unless stated. Grid and monotone threshold path fixed before split B is read.
- Hardware: Kaggle Tesla T4 (15 GB). Results record GPU, driver, build and model hashes.

## Data and splits (frozen)
- TriviaQA rc.nocontext: 6000 items sampled with fixed seed; A=2000 (calibrators, router
  training, cost estimation), B=2000 (threshold certification, used once), C=2000 (test).
- Shift/stress, test-only: NQ-open 2000; SimpleQA Verified 1000; MMLU-Redux 1500.
- Validation: 50 random re-splits of the 6000 into A/B/C with hash-locked split files.
- Grading: normalised-alias exact match primary; SimpleQA three-label judge (judge model
  differs from S and L). 200-item human audit per grader; report false-wrong e1 and
  false-right e2 with CIs and a sensitivity-adjusted alpha (never presented as the guarantee).

## Baselines (all under the same certification procedure)
S only, L only, max-softmax gate, mean-logprob gate, entropy gate, margin gate, random
router at matched escalation rate, oracle router, oracle abstainer, query-only router
("RouteLLM-style", retrained on split A), self-consistency gate (k=5, 1000-item subset).

## Statistics
Percentile bootstrap over test items (10,000), paired by item; differences reported with
CIs; McNemar at matched coverage; equivalence margin +-0.02 (TOST) for quantization
comparisons. No claim from overlapping marginal CIs. Repeat-run nondeterminism on 300 items x 3 runs.

## Hypotheses and pre-declared negative outcomes
| ID | Hypothesis | Negative / null outcome (reported with equal prominence) |
|---|---|---|
| H1 validity | Across 50 re-splits, test risk on C exceeds alpha in <= 15% of splits | > 25%: bug or non-exchangeability |
| H2 signal | Mean-logprob or margin AUROC exceeds max-softmax by >= 0.02 (paired CI excludes 0) | All single token signals within +-0.02: signal choice irrelevant |
| H3 cascade value | Certified cascade matches L-only accuracy-on-answered at lower total cost incl. wasted S attempts, and beats random and query-only routers at matched coverage | Indistinguishable from random router, or advantage vanishes with full cost accounting |
| H4 self-consistency | k=5 adds >= 0.03 AUROC over best token signal; helps the frontier only if k x cost_S < cost_L | No gain (likely) |
| H5 shift | Risk exceeds alpha on NQ-open and by more on SimpleQA; drift alarm fires on >= 80% of shifted 200-query windows before violation | Guarantee holds under shift, or alarm blind |
| H6 quantization | A threshold fitted on f16 violates alpha on some quantized config while recalibrated thresholds do not | For all configs dAUROC and dECE within +-0.02 and transferred thresholds hold at >= 1-delta |
| H7 hard sets | On MMLU-Redux gate AUROC < 0.6 (cf. arXiv 2610.23249) | Reported as limitation either way |

Also declared negative outcomes: empty feasible threshold set, near-zero coverage, bounds
too loose at n_B = 2000, bootstrap CIs spanning zero.

## Cost accounting (frozen)
Latency: per-item wall time of every attempted generation, including wasted S attempts and
any L re-generation; model load/swap time reported separately. Energy: NVML power sampled
at 10 Hz, integrated, idle-subtracted, ABAB-interleaved cells; GPU energy only (CPU/RAPL
not available on Kaggle). VRAM: peak nvidia-smi per cell. Any "X% of the compute/energy"
statement must come from these measurements.

## Deviations
### 2026-11-03 (before any headline experiment; no real generation had been produced)
1. **Small tier: Qwen2.5-3B-Instruct replaced by Qwen2.5-1.5B-Instruct.** Reason: licence. The 3B is under the Qwen research licence (non-commercial); the 1.5B and 7B are Apache-2.0, matching Sieve's MIT licence. The full ladder (f16, Q8_0, Q4_K_M, Q3_K_M, Q2_K) exists for the 1.5B. This is the fallback the pre-registration already named, now primary.
2. **Large tier file:** bartowski's single-file Qwen2.5-7B-Instruct Q4_K_M (an importance-matrix quant), instead of Qwen's split files. SHA-256 is verified against the Hub at download.
3. **Prompt:** a fixed few-shot QA prompt (`qa-v2-fewshot`) with an `Answer:` prefix replaces the earlier raw prompt (`qa-v1`), because the T4 smoke test showed the 7B opening with a preamble (D-016). Decoding is greedy, generation stops at the first newline, `-np 1 -c 4096 -ngl 99`, flash-attention on for quantised KV, `cache_prompt` off.
4. **Threshold path:** the fixed-sequence path is the split-A Pareto frontier of (threshold pair) points ordered by split-A coverage, with points below 5% A-coverage dropped, fixed before split B is read. This is a specific choice within the LTT framework; validity does not depend on it, power does.
5. **Operational choices the document left open** (listed in `docs/evaluation.md`): H3 "matches" read as a +-0.02 non-inferiority margin; "matched coverage" uses split-A-derived thresholds; H5 "before violation" means the label-free drift alarm fires; re-split 0 is the canonical split for bootstraps; MMLU-Redux uses `error_type == "ok"` rows only; the SimpleQA judge prompt is our own wording; the query-only router is TF-IDF plus logistic regression.

### 2026-11-03 (documentation only)
6. `docs/council/` was moved to `docs/process/council/` for presentation. The path in the header of this file
   was deliberately not edited so the frozen text and its hash stay identical.
7. The real run produced two findings that are not deviations but are recorded here for transparency: the
   4-bit KV-cache arm and the Q2_K arm failed the stop-rate sanity check (reported as ablation findings, see
   `docs/RESULTS.md`), and SimpleQA could not be graded because no independent judge model was available.
