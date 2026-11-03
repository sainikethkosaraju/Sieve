# Benchmarks, Metrics and Evaluation Methodology for Hallucination, Abstention and Calibration (for Sieve)

Research caveat: arxiv.org was not reachable by WebFetch in this session (DNS failure), so all findings come from WebSearch snippets and secondary pages; no primary paper was read in full. Items from model memory with no fetched source are listed under Gaps as UNVERIFIED and must be checked before use. Date of research: 2026-11-03.

## 1. Datasets and benchmarks: suitability, licensing, size, flaws, saturation and contamination

### Takeaway
For abstention/hallucination, the newest community-accepted anchors found are AbstentionBench (NeurIPS 2025, 20 datasets) and SimpleQA Verified (1,000 prompts, F1 headline, answered-and-wrong style metrics). MMLU has a documented ~6.5% error rate (MMLU-Redux), so use MMLU-Redux/MMLU-Pro rather than raw MMLU. Licensing for several classic sets was not resolved from reliable sources.

### Cited Findings
- AbstentionBench (Meta FAIR; Kirichenko, Ibrahim, Chaudhuri, Bell; arXiv June 2025, NeurIPS 2025) spans 20 datasets covering unknown answers, underspecification, false premises, subjective interpretations and outdated information; one version reports over 35k unanswerable questions. Code at github.com/facebookresearch/AbstentionBench — [arXiv 2506.09038](https://arxiv.org/pdf/2506.09038); [HF papers](https://huggingface.co/papers/2506.09038)
- AbstentionBench headline results: reasoning fine-tuning degrades abstention by 24% on average, even in math/science; scaling does not help; a carefully written system prompt boosts abstention but does not fix the underlying inability to reason about uncertainty; frontier LLMs struggle in all scenarios except questions with unknown answers — [arXiv 2506.09038](https://arxiv.org/pdf/2506.09038) (via search summary; not read in full)
- SimpleQA Verified (Google DeepMind, arXiv 2509.07968, v2 March 2026): 1,000 prompts filtered from OpenAI SimpleQA; fixes noisy/incorrect labels, topical bias and redundancy via deduplication, topic/answer-type rebalancing and source reconciliation; removed about 77% of original questions; Gemini 2.5 Pro F1 55.6; CC BY 4.0 paper — [arXiv 2509.07968](https://arxiv.org/abs/2509.07968); [v2](https://arxiv.org/html/2509.07968v2)
- Original SimpleQA (OpenAI, arXiv 2411.04368) grades each answer correct / incorrect / not attempted with a prompted ChatGPT classifier; reports overall correct, correct-given-attempted, and F-score (harmonic mean of the two). In the original release, 15 of 56 manually inspected "incorrect" labels were autograder false negatives — [OpenAI](https://openai.com/index/introducing-simpleqa/); [arXiv 2411.04368](https://arxiv.org/html/2411.04368v1) (via search summary)
- A Korean SimpleQA adaptation found the autorater overly permissive in assigning NOT_ATTEMPTED to outputs that should be INCORRECT (false refusals, irrelevant answers, degeneration) — a grader pitfall that directly inflates apparent abstention quality — [KoSimpleQA](https://arxiv.org/pdf/2510.18368)
- SimpleQA Verified keeps the 3-label scheme and F1 headline but improved the autorater prompt for extra detail, approximate numbers and hedging — [parallel.ai explainer](https://parallel.ai/articles/simpleqa-benchmark.md) (secondary)
- HalluLens notes SimpleQA is a factuality benchmark adaptable to hallucination measurement (inverse of correct-given-attempted); HalluLens is itself a hallucination benchmark (intrinsic and extrinsic) — [HalluLens](https://arxiv.org/pdf/2504.17550)
- MMLU: MMLU-Redux (Gema et al., NAACL 2025) estimates 6.49% of MMLU questions contain errors, 57% of the Virology subset analysed; re-annotated subset of 5,700 questions across 57 subjects (v1 described 3,000 across 30 subsets); re-evaluation changes model rankings/scores — [Are We Done with MMLU?](https://arxiv.org/pdf/2406.04127); [v2 html](https://arxiv.org/html/2406.04127v2)
- Licensing (conflicting, secondary sources only): TriviaQA Apache 2.0; Natural Questions listed as CC BY-SA 3.0 in two papers but MIT in one; SQuAD 2.0 listed CC BY-SA 4.0 in one source and CC0 on a Kaggle mirror. PopQA: 14K entity-centric QA pairs from Wikidata, license not found — [UKP-SQUARE appendix](https://arxiv.org/pdf/2203.13693); [ParallelSearch](https://arxiv.org/pdf/2508.09303); [Rethinking On-policy Optimization](https://arxiv.org/pdf/2510.17139); [Kaggle mirror](https://baselight.app/u/kaggle/dataset/thedevastator_squad2_0_a_challenge_for_question_answering_syst)
- Newer 2026 items (all from secondary aggregators, abstracts only, UNVERIFIED): PhantomBench (60K+ plausible non-existent terms, 21 models, hallucination rate = answering instead of abstaining, up to 87.3% for some attributes) [emergentmind 2606.11105](https://www.emergentmind.com/papers/2606.11105); "Two Axes of LLM Abstention" splits abstention into answer correctness vs question answerability, uses SelfAware [alphaxiv 2607.08456](https://www.alphaxiv.org/abs/2607.08456); "The Missing 'I Don't Know'" (Sept 2026) argues leaderboards should report abstention rate and calibration [alphaxiv 2609.17686](https://www.alphaxiv.org/abs/2609.17686); KGHaluBench (knowledge-graph hallucination benchmark, Feb 2026) [arXiv 2602.19643](https://arxiv.org/html/2602.19643v1); PARALLAX argues 4 of 6 widely cited hallucination-detection datasets leak ground truth into prompts [themodelwire](https://themodelwire.com/article/parallax-separating-genuine-hallucination-detection-from-benchmark-construction-01KRYYS7J96HJ3GWYGKMB16WA5)

### Inferences
- Sieve needs both a "knows/doesn't know" factual set (SimpleQA Verified, TriviaQA, PopQA) where abstention is learnable from parametric knowledge, and a broader abstention set (AbstentionBench subsets) to test generalisation beyond unknown-answer cases.
- Because AbstentionBench shows reasoning models abstain worse, any Sieve router that routes to reasoning models should be evaluated on abstention explicitly, not only accuracy.
- SimpleQA-style graders can be gamed through over-permissive NOT_ATTEMPTED labelling, so grader audit (human-labelled sample) should be part of Sieve's protocol.
- Contamination: SimpleQA Verified/AbstentionBench are recent and adversarially built against frontier models, so likely less contaminated than TriviaQA/NQ/MMLU, which are old and web-scraped (inference; no source found quantifying contamination).

### Gaps
- Not verified by any fetched source (model memory only, treat as UNVERIFIED): TruthfulQA (817 questions, Apache 2.0, known label noise and saturation, MC1/MC2 gameable); SQuAD 2.0 (~50k unanswerable questions, extractive context given so not parametric-knowledge test); SelfAware (~1,032 unanswerable + ~2,337 answerable); HaluEval (~35k samples, ChatGPT-generated hallucinations, weak ecological validity); FActScore (biography generation, atomic-fact precision, expensive); TriviaQA ~95k Q/A pairs; NQ ~300k train; MMLU-Pro 12k questions 10 options; PopQA license; KnowOrNot benchmark details; exact AbstentionBench dataset list, judge and licence per dataset.
- No source found quantifying contamination or saturation for each dataset.
- Need to check Hugging Face dataset cards directly for licences.

## 2. Correctness grading: exact match vs LLM-as-judge vs normalised matching

### Takeaway
For short-form factual QA, the accepted approach is a three-way prompted LLM grader (correct / incorrect / not attempted) with a fixed grader prompt, audited against human labels; exact match alone undercounts correct paraphrases. Grader permissiveness about "not attempted" is a documented failure mode.

### Cited Findings
- SimpleQA uses prompted-LLM three-way grading and manual audit found grader false negatives (15 of 56 examined "incorrect") — [OpenAI SimpleQA](https://openai.com/index/introducing-simpleqa/)
- SimpleQA Verified revised the grader prompt specifically to handle extra detail, approximate numbers and hedging — [parallel.ai](https://parallel.ai/articles/simpleqa-benchmark.md)
- Autoraters over-assign NOT_ATTEMPTED to false refusals, irrelevant answers and degenerate text — [KoSimpleQA](https://arxiv.org/pdf/2510.18368)
- lm-evaluation-harness paper argues evaluation details (prompting, normalisation) materially change results and motivates versioned tasks — [Lessons from the Trenches](https://arxiv.org/abs/2405.14782v1); [EleutherAI blog](https://www.eleuther.ai/papers-blog/lessons-from-the-trenches-on-reproducible-evaluation-of-language-models)

### Inferences
- Recommended practice: normalised EM/alias match as a cheap first pass (TriviaQA/PopQA have alias lists), then an LLM judge with a pinned model version and frozen prompt for unmatched answers, with a human-labelled audit sample (e.g. a few hundred items) reporting judge-human agreement/kappa. Use a judge from a different family than the evaluated models to limit self-preference (inference).
- Abstention detection should be a separate, explicit rule (structured output field or refusal classifier) rather than relying solely on the judge.

### Gaps
- No fetched source on judge self-preference bias magnitude, kappa thresholds, or sample size for judge audits (UNVERIFIED from memory).
- AbstentionBench's judge model and validation not confirmed.

## 3. Metrics: accuracy, selective accuracy, risk-coverage, AURC, AUROC, ECE, Brier, hallucination rate, abstention P/R, cost-normalised

### Takeaway
Report an operating-point triple (coverage, selective accuracy / hallucination rate among answered, abstention precision/recall) plus one threshold-free summary. AURC has documented flaws; AUGRC (NeurIPS 2024) is the recommended replacement alongside AUROC for confidence ranking. SimpleQA's correct / not-attempted / incorrect + F-score is the closest accepted LLM hallucination convention.

### Cited Findings
- Traub et al. (NeurIPS 2024) argue multi-threshold selective-classification metrics fail five requirements; AURC overweights high-confidence failures and breaks monotonicity/interpretability; e-AURC still depends on overall accuracy; propose AUGRC (average risk of undetected failures), which changed metric rankings on 5 of 6 datasets across 13 confidence scorers; fixed-threshold metrics (selective risk at given coverage) are useful at one operating point but incomplete — [arXiv 2407.01032](https://arxiv.org/pdf/2407.01032); [NeurIPS page](https://proceedings.neurips.cc/paper_files/paper/2024/hash/047c84ec50bd8ea29349b996fc64af4b-Abstract.html) (general classification, not LLM-specific)
- SimpleQA defines correct, correct-given-attempted (correct / not-declined) and F-score as harmonic mean — [OpenAI](https://openai.com/index/introducing-simpleqa/)
- HalluLens frames hallucination as the inverse of correct-given-attempted and notes this measures refusal behaviour — [HalluLens](https://arxiv.org/pdf/2504.17550)
- PhantomBench defines hallucination rate as how often a model answers rather than abstains (for unanswerable/non-existent entities) — [emergentmind](https://www.emergentmind.com/papers/2606.11105) (unverified secondary)
- 2026 literature stresses calibration and selective classification are distinct goals and should both be measured; "Missing I Don't Know" proposes mandatory abstention-rate and calibration reporting — [alphaxiv 2609.17686](https://www.alphaxiv.org/abs/2609.17686); [Two Axes](https://www.alphaxiv.org/abs/2607.08456) (unverified secondary)

### Inferences
- Define for Sieve: hallucination rate = P(answered and wrong) over all questions (unconditional) AND error rate among answered (1 - correct-given-attempted); report both, naming them distinctly, since papers conflate them.
- Abstention precision = fraction of abstentions on questions the model would have gotten wrong (or truly unanswerable); recall = fraction of would-be-wrong/unanswerable questions abstained on. Requires an always-answer reference run to define "would have been wrong".
- ECE: report with fixed equal-mass bins plus Brier and reliability diagrams, label ECE as binning-sensitive; do not use ECE as sole calibration claim (known problems — see Gaps).
- Cost-normalised: cost per correct answer, cost per answered-correct at fixed hallucination budget, and the cost-quality curve (Section 5).

### Gaps
- UNVERIFIED (memory only): ECE problems (bin-count sensitivity, bias at small n, only top-label, ignores sharpness; debiased/adaptive-bin variants); Brier decomposition; Geifman & El-Yaniv risk-coverage origin; Kadavath et al. "LMs (mostly) know what they know", Kuhn et al. semantic entropy and AUROC reporting convention. None fetched.
- How top 2025-26 papers actually tabulate these metrics not verified beyond SimpleQA/AbstentionBench summaries (AbstentionBench's exact metric — abstention recall / F1 — not confirmed).

## 4. Statistics and tooling

### Takeaway
Treat questions as samples from a population: report standard errors/CIs, use paired differences between systems on the same questions, and plan sample size by power. Build on an existing harness for tasks/versioning if possible, but a thin custom harness for abstention/routing metrics is likely needed.

### Cited Findings
- Miller, "Adding Error Bars to Evals" (Anthropic, arXiv 2411.00640): treat eval questions as draws from a larger population to give standard errors/CIs; recommends paired question-level differences when comparing models; gives formulas for power/eval size planning; clustered standard errors for grouped questions — [arXiv 2411.00640](https://arxiv.org/pdf/2411.00640); [errorbars package](https://pypi.org/project/errorbars/)
- lm-evaluation-harness: open-source, unified interface, task versioning for reproducibility — [Lessons from the Trenches](https://arxiv.org/abs/2405.14782v1); [EleutherAI](https://www.eleuther.ai/projects/large-language-model-evaluation)
- Secondary (unverified) claims: Inspect AI originates from UK AISI, defaults reported as temperature 0 for vLLM; Lighteval is HF's engine behind the Open LLM Leaderboard and has no default temperature — [instagit comparison](https://instagit.com/huggingface/skills/difference-inspect-ai-lighteval-model-evaluation)

### Inferences
- With p~0.5 accuracy, standard error ≈ sqrt(0.25/n): n=1,000 gives ±~3.1pp at 95%, n=10,000 gives ±~1pp (arithmetic, not sourced). SimpleQA Verified at 1,000 questions can therefore only resolve ~3-4pp unpaired differences; paired tests tighten this.
- Selective metrics at low coverage use fewer answered items, so CIs must be bootstrapped over questions (resample whole question units, recompute threshold-dependent metrics), with ≥1,000 resamples.
- Inspect AI (task/solver/scorer abstractions, logs) suits agentic/multi-model API-based pipelines and LLM-graded scoring; lm-eval-harness suits log-likelihood/MC tasks on local models. Sieve's router-over-API-models and custom abstention scoring points to Inspect or a thin custom harness with cached per-item outputs (inference; not verified against official docs).

### Gaps
- Official docs for Inspect, lighteval, OpenCompass were not fetched; feature claims unverified. OpenCompass not covered at all.
- Bootstrap vs CLT guidance (e.g. Bowyer et al. "Don't use the CLT in LLM evals") and seed-count guidance for stochastic decoding not sourced.

## 5. Routing papers: RouterBench and RouteLLM cost-quality metrics and plotting

### Takeaway
RouterBench plots a cost-quality curve and summarises it with AIQ (normalised area under the curve over a cost range), with Zero Router and Oracle baselines; RouteLLM uses PGR/APGR and CPT. Both are single-threshold-sweep, so Sieve's abstention-aware routing needs an extension (quality that penalises wrong answers).

### Cited Findings
- RouterBench (Hu et al., ICML 2024): >405k pre-generated inference outcomes across representative LLMs so routers can be tested without inference; AIQ = ∫ q(c) dc / (c_max − c_min) with q(c) the performance achievable at cost c; baselines Zero Router and Oracle Router (Oracle picks best-performing and cheapest LLM); KNN/MLP routers did not significantly beat Zero Router in the paper — [arXiv 2403.12031](https://arxiv.org/html/2403.12031v2); [Martian blog](https://www.plushcap.com/content/martian/blog/martian-introducing-routerbench); code [github.com/withmartian/routerbench](https://github.com/withmartian/routerbench) (via search summary; not read in full)
- RouteLLM (Ong et al., 2024): APGR = average performance gap recovered between weak and strong model across cost constraints (higher better); CPT(x%) = fraction of calls to the strong model needed to reach x% of the performance gap (lower better), e.g. CPT(50%) ≈ 37%; headline up to 3.66x cost savings on MT-Bench at 95% of GPT-4 quality (secondary write-ups) — [LMSYS blog](https://lmsys.org/blog/2024-07-01-routellm/); [Shekhar Gulati summary](https://shekhargulati.com/2024/07/09/routellm-paper/); [Zilliz](https://zilliz.com/learn/routellm-open-source-framework-for-navigate-cost-quality-trade-offs-in-llm-deployment)

### Inferences
- Accepted plot: x = cost (or % strong-model calls), y = quality, one curve per router from sweeping a threshold, with random-mixing and oracle reference lines and a convex-hull/non-decreasing frontier; for Sieve, add a second panel y = hallucination rate (answered-and-wrong) vs cost, and a third x = coverage.
- Report AIQ-style area with a common cost range across all systems and bootstrap CIs on area.
- Fair comparison to RouterBench/RouteLLM baselines suggests releasing per-item cached outputs for every model so curves are recomputable offline.

### Gaps
- Exact RouterBench convex-hull construction and RouteLLM PGR formula text not read from primary papers (arXiv unreachable).
- No sourced treatment of routers that include an abstain action.

## 6. Reproducibility standards and one-command patterns

### Takeaway
Little was retrieved from primary sources on checklists; recommendations below are largely inference and need verification.

### Cited Findings
- lm-eval-harness paper motivates versioned tasks and shared evaluation code because small implementation differences change scores — [Lessons from the Trenches](https://arxiv.org/abs/2405.14782v1)
- Nondeterminism of models/APIs affects run stability across all harnesses — [PromptLayer glossary](https://www.promptlayer.com/glossary/eval-harness) (weak source)

### Inferences
- Pin: dataset revision hash (HF commit), model identifiers with dated API snapshots, judge model+prompt hash, decoding params, seed, harness version; cache all raw generations and judge outputs; provide a `make reproduce` / single script that regenerates every table and figure from cached outputs, and a separate `make rerun` that re-queries models.
- For API models that deprecate, release cached outputs so results remain recomputable.

### Gaps
- NeurIPS/ICML reproducibility checklist contents, ML Reproducibility Challenge practices, and one-command patterns not sourced in this session (UNVERIFIED). Fetch the NeurIPS Paper Checklist and ICML reproducibility guidance directly.

## 7. Recommended benchmark suite and metric set for Sieve

### Takeaway
Use a small, tiered suite: SimpleQA Verified + PopQA/TriviaQA (parametric factual, where abstention is learnable), AbstentionBench (generalisation of abstention), MMLU-Redux or MMLU-Pro (multiple choice cost-quality routing and calibration), optionally SelfAware and a 2026 benchmark such as PhantomBench once verified. Report operating-point metrics plus AUGRC/AUROC, Brier/ECE and cost-quality curves with bootstrap CIs.

### Cited Findings
- Component justification is in Sections 1, 3, 5: SimpleQA Verified (cleaned, F1 headline), AbstentionBench (NeurIPS 2025, 20 datasets), MMLU-Redux (error-corrected), AUGRC (fixes AURC flaws), RouterBench AIQ and RouteLLM APGR/CPT (routing), Miller error bars (statistics) — see sources above.

### Inferences
- Suite: (1) SimpleQA Verified (1,000); (2) PopQA and TriviaQA subsample (several thousand, alias-based EM + judge) for tighter CIs; (3) AbstentionBench full or stratified; (4) MMLU-Redux/MMLU-Pro for calibration with MC (verify licence first); (5) hold-out split for tuning Sieve thresholds so no tuning on test sets.
- Metrics: coverage, selective accuracy, hallucination rate (unconditional and conditional), abstention precision/recall/F1, AUGRC and AUROC for correctness, Brier and ECE (equal-mass bins) with reliability diagrams, cost per correct answer, AIQ-style area and CPT/APGR-style numbers against always-weak, always-strong, random-mix and oracle baselines.
- Stats: question-level bootstrap (≥1,000 resamples) and paired differences vs baselines; report n and CIs in every table; multiple seeds only where decoding is stochastic.
- Harness: build on Inspect AI or lm-eval-harness for task loading and logging if feature check passes, with custom scorers and cached outputs; otherwise a thin custom harness (decision depends on unverified tooling checks).

### Gaps
- Final choices depend on verifying licences, AbstentionBench's exact metrics/judge, and tooling docs. Contamination status of each benchmark for the specific models Sieve routes among is unknown.
