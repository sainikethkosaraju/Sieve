# Confidence signals, calibration, and abstention for LLMs (answer / escalate / abstain)

Method note: about 14 tool calls, search snippets only. WebFetch to arxiv.org failed (host not in the session's allowed domains), so no full texts were read. Every claim below comes from search-result summaries and is NOT verified against the full papers unless stated. Several 2026 arXiv IDs (26xx.xxxxx) come from search results only; treat them as unverified preprints.

## Token-level signals (max-softmax, entropy, logprob margins, length-normalized likelihood); effect of RLHF and quantization

### Takeaway
Token-probability signals are the cheapest option, but RLHF-tuned models are reported to be poorly calibrated on raw token probabilities. Quantization (4-bit GPTQ) lowers confidence in the true label, most for samples the full model was already unsure about. I found no study that combines RLHF, quantization, and calibration, so that combination is an open gap.

### Cited Findings
- Tian et al. (EMNLP 2023), "Just Ask for Calibration": for ChatGPT, GPT-4 and Claude on TriviaQA, SciQ and TruthfulQA, verbalized confidence was usually better calibrated than conditional token probabilities, often cutting ECE by about half. Details beyond the abstract are unverified. — [ACL Anthology 2023.emnlp-main.330](https://preview.aclanthology.org/setup/2023.emnlp-main.330); [arXiv 2305.14975](https://arxiv.org/abs/2305.14975v2)
- Proskurina et al. (NAACL 2024 Findings), "When Quantization Affects Confidence of Large Language Models?": GPTQ 4-bit quantization decreases confidence on the true label, and the effect falls disproportionately on samples where the full model had low confidence. They tested base models; code is at upunaprosk/quantized-lm-confidence. — [arXiv 2405.00632](https://arxiv.org/pdf/2405.00632)
- "Mind the Confidence Gap" (arXiv 2502.11028, 2025): notes that RLHF can amplify overconfidence, but the extent and conditions are largely unexplored. It finds miscalibration varies with model scale and input structure (SimpleQA, distractor answer options). — [arXiv 2502.11028](https://arxiv.org/pdf/2502.11028)
- LM-Polygraph toolkit and benchmark (Vashurin, Fadeeva et al., TACL 2025) include the token baselines (max/mean token entropy, sequence probability), Monte Carlo sequence entropy, semantic entropy and verbalized uncertainty. They cover 11 tasks, including selective QA, translation, summarization and claim-level fact-checking. For fact-checking, Claim-Conditioned Probability (CCP) performed strongly. — [TACL 2025.tacl-1.11](https://aclanthology.org/2025.tacl-1.11.pdf); [arXiv 2406.15627](https://ar5iv.labs.arxiv.org/html/2406.15627); [PyPI lm-polygraph](https://pypi.org/project/lm-polygraph)
- Semantic Energy (arXiv 2508.14496, 2025) argues post-softmax probabilities lose information. It proposes logit-based "energy" plus semantic clustering and reports it beats semantic entropy where semantic entropy fails (author claim; unverified). — [arXiv 2508.14496](https://arxiv.org/html/2508.14496v2)

### Inferences
- Token signals need per-model, per-quantization recalibration (temperature scaling or isotonic regression on a held-out set). Raw thresholds will not transfer between an FP16 model and its GGUF 4-bit version.
- I did not find numbers on length-normalized likelihood versus entropy versus margin, so the ranking among them is unknown from this research.

### Gaps
- Head-to-head numbers (AUROC, ECE) for max-softmax, entropy, margin and length-normalized likelihood on modern instruct models.
- Joint RLHF plus quantization calibration study.
- LM-Polygraph's overall conclusion on whether simple baselines are competitive: could not verify. The search summary did not state it, and the full paper could not be fetched.

## Sampling-based signals (self-consistency, semantic entropy, SEPs, cheaper variants)

### Takeaway
Semantic entropy (Farquhar et al., Nature 2024) detects confabulations by clustering sampled answers by meaning. It costs roughly 5-10x a single generation. Semantic Entropy Probes (Kossen et al., 2024) approximate it from one forward pass with a linear probe, trading some accuracy for near-zero overhead.

### Cited Findings
- Farquhar, Kossen, Kuhn, Gal, "Detecting hallucinations in large language models using semantic entropy", Nature 630, 625-630 (June 2024), DOI 10.1038/s41586-024-07421-0. Entropy is computed over meaning clusters rather than word sequences. It needs no task-specific data. — [PMC11186750](https://pmc.ncbi.nlm.nih.gov/articles/PMC11186750); [OATML blog](https://oatml.cs.ox.ac.uk/blog/2024/06/19/detecting_hallucinations_2024.html)
- Cost per the authors' blog: about 10x the compute of a plain answer. The Oxford press release says "several times" more. — [OATML blog](https://oatml.cs.ox.ac.uk/blog/2024/06/19/detecting_hallucinations_2024.html); [Oxford news](https://www.ox.ac.uk/news/2024-06-20-major-research-hallucinating-generative-models-advances-reliability-artificial)
- Stated failure mode: it covers confabulations (arbitrary wrong answers) only. It can fail when clustering merges answers of different specificity (a full date versus only a year). Farquhar: it addresses "only part of the story". — same sources
- Kossen et al., "Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs" (arXiv 2406.15927, 2024): linear probes on hidden states from a single generation predict semantic entropy. Mid-to-late layers work best. They can predict it before generation (useful for pre-answer refusal). They generalize better than accuracy probes to unseen tasks, while accuracy probes win in-distribution. They do not beat full sampling-based semantic entropy; the advantage is cost. — [arXiv 2406.15927](https://arxiv.org/pdf/2406.15927)
- Cheaper or alternative estimators, taken from search summaries only: "Beyond Semantic Entropy" (nearest-neighbour entropy, no explicit clustering, May 2025); Bayesian estimators that work with few samples; adaptive sampling for semantic entropy (March 2026); Semantic Energy (above). — [Semantic Energy listing and related hits](https://arxiv.org/html/2508.14496v2)
- A small controlled finance study (ECLIPSE, Dec 2025) reported a semantic-entropy-only baseline at AUC 0.50 on synthetic hallucinations. This is a small, narrow dataset and should not be generalized. — search-result summary; no stable URL checked

### Inferences
- For an edge/local deployment, k-sample semantic entropy multiplies latency and energy by about k. It fits an "escalate" tier better than an always-on gate, and SEP-style probes are the cheaper substitute if hidden states are available.

### Gaps
- Exact AUROC values per model and dataset for SE versus SEP versus self-consistency (not retrieved).
- Whether discrete semantic entropy without an NLI model works well (no source found).

## Hidden-state probes (Kadavath P(IK), SAPLMA, INSIDE/EigenScore, LLM-Check)

### Takeaway
Probes on mid-layer hidden states are cheap and often strong in-distribution. Transfer across domains and models is weaker and is reported unevenly. The newest head-to-head claim is a single unreviewed preprint.

### Cited Findings
- Kadavath et al. (Anthropic, 2022), "Language Models (Mostly) Know What They Know": P(True) self-evaluation is encouraging and calibrated and improves with scale. P(IK) is a trained value head predicting whether the model knows the answer without seeing one. It partially generalizes across tasks but its calibration transfers poorly to new tasks. P(IK) rises when relevant context is provided. — [arXiv 2207.05221](https://arxiv.org/pdf/2207.05221); [Anthropic](https://www.anthropic.com/index/language-models-mostly-know-what-they-know)
- INSIDE (Chen et al., ICLR 2024): EigenScore uses eigenvalues of the covariance of sampled-response embeddings, plus test-time feature clipping (about 1-3% AUROC gain). It requires multiple samples. — [arXiv 2402.03744](https://arxiv.org/pdf/2402.03744)
- LLM-Check (Sriramanan et al., NeurIPS 2024): single-response detection from hidden states, attention maps and output probabilities, with an auxiliary proxy LLM for black-box settings. Authors claim speedups of up to 45x and 450x over baselines. — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3c1e1fdf305195cd620c118aaa9717ad-Abstract.html)
- SAPLMA-style probes (classifier on last-token hidden state) are a common baseline. — [MultiHaluDet, arXiv 2605.24919](https://arxiv.org/pdf/2605.24919)
- Preprint "Hallucination Is Linearly Decodable from Mid-Layer Hidden States in Quantized LLMs" (arXiv 2606.02628, 30 May 2026): a linear probe on one mid layer reaches 0.904-1.000 AUROC on held-out splits, while sampling-based methods such as EigenScore stayed at or below 0.541. This is a single non-peer-reviewed study using a paired true/false statement protocol, not generation-based evaluation; the gap is protocol-dependent. — [arXiv 2606.02628](https://arxiv.org/pdf/2606.02628)
- SEP paper: probes transfer better than accuracy probes to unseen tasks. — [arXiv 2406.15927](https://arxiv.org/pdf/2406.15927)

### Inferences
- A probe trained per model, per quantization, on in-domain labelled correct/incorrect data is a realistic cheap signal. Plan on retraining it for each model and domain, and verify out-of-domain before trusting it.

### Gaps
- Cross-model probe transfer: no solid source found.
- Independent replication of the 2026 preprint's near-perfect AUROCs.
- No direct LLM-Check versus SAPLMA versus INSIDE comparison found.

## Verbalized confidence and P(True)

### Takeaway
Verbalized confidence beats raw token probabilities for RLHF API models in Tian et al.'s results, but it is overconfident in general, and reasoning models are inconsistent.

### Cited Findings
- Xiong et al. (arXiv 2306.13063): vanilla verbalized confidence shows significant overconfidence and poor failure prediction. — [arXiv 2306.13063](https://arxiv.org/pdf/2306.13063)
- Mei et al. (arXiv 2506.18183): reasoning models are typically overconfident, often above 85% confidence even on wrong answers. Confidence worsens with longer reasoning. Introspection helped o3-Mini and R1 but made Claude 3.7 Sonnet less calibrated. — [arXiv 2506.18183](https://arxiv.org/abs/2506.18183v3)
- Zeng et al. (arXiv 2504.06564): SFT reasoning models often show degraded calibration versus instruct models; RL reasoning models match or exceed them. — [arXiv 2504.06564](https://arxiv.org/abs/2504.06564v1)
- Conflicting: a paper (arXiv 2505.14489, "Reasoning Models Better Express Their Confidence") reports better calibration in 33 of 36 settings; seen via a secondary summary only. — [summary](https://www.themoonlight.io/es/review/reasoning-models-better-express-their-confidence)
- A 2026 mechanistic paper (arXiv 2603.25052) reports that reasoning while verbalizing disrupts the verbalized-confidence direction (unverified). — [arXiv 2603.25052](https://arxiv.org/pdf/2603.25052)
- Tian et al.: listing several candidate answers before stating confidence noticeably improves verbalized calibration. — [arXiv 2305.14975](https://arxiv.org/abs/2305.14975v2)

### Inferences
- Small local models were not covered by these results. Verbalized confidence on small quantized models is likely unreliable (my hypothesis; no source found).

### Gaps
- No data found on verbalized confidence for small (under 8B) or quantized models.

## Calibration methods and guarantees (temperature scaling, isotonic, conformal, selective-prediction metrics)

### Takeaway
Conformal factuality offers distribution-free marginal guarantees by filtering sub-claims, at the cost of reduced output content. Post-hoc recalibration and the selective-prediction metrics are standard, but I found no LLM-specific source in this pass.

### Cited Findings
- Mohri and Hashimoto, "Language Models with Conformal Factuality Guarantees" (ICML 2024, PMLR 235): decomposes outputs into sub-claims, scores them, and removes less certain ones. It gives marginally valid coverage with no distributional assumptions. On QA and reasoning with GPT-4 it achieved 80-90% correctness guarantees while mostly preserving utility. — [PMLR](https://proceedings.mlr.press/v235/mohri24a.html); [arXiv 2402.10978](https://arxiv.org/html/2402.10978v1)
- Follow-ups: "Conformal Language Model Reasoning with Coherent Factuality" (Rubin-Toles et al., 2025, arXiv 2505.17126); "Conformal Linguistic Calibration" (Jiang, Liu, Van Durme). — [arXiv 2505.17126](https://arxiv.org/pdf/2505.17126)
- Surveys: Shorinwa et al., ACM Computing Surveys 2025 (arXiv 2412.05563); Liu et al., KDD 2025 (arXiv 2503.15850); Xia et al., ACL Findings 2025 (arXiv 2503.00172); Abbasli et al. (arXiv 2504.18346); Geng et al. (arXiv 2311.08298). Descriptions come from abstracts and snippets. — [2412.05563](https://arxiv.org/pdf/2412.05563); [2503.15850](https://arxiv.org/html/2503.15850v2); [2503.00172](https://arxiv.org/abs/2503.00172v2); [2504.18346](https://arxiv.org/abs/2504.18346); [2311.08298](https://alphaxiv.org/abs/2311.08298)

### Inferences
- Conformal guarantees are marginal (average over inputs), not per-domain or per-query. They hold only if the calibration set is exchangeable with deployment data, so distribution shift breaks them.

### Gaps
- Risk-controlling prediction sets, isotonic versus temperature scaling comparisons for LLMs, and the AURC/risk-coverage metric literature were not searched. No cited source here.

## Abstention methods and evaluation pitfalls

### Takeaway
R-Tuning trains refusal from the gap between a model's knowledge and the tuning data. I did not find sources on evaluation pitfalls.

### Cited Findings
- Zhang, Diao et al., "R-Tuning: Instructing Large Language Models to Say 'I Don't Know'" (NAACL 2024, long, pp. 7113-7139; arXiv 2311.09677). It builds refusal-aware training data from questions the model cannot answer. Authors report better accuracy on answered questions, fewer answers to unknown ones, refusal that extends out-of-domain as a "meta-skill", and better calibration. Code: shizhediao/R-Tuning. — [ACL Anthology](https://aclanthology.org/2024.naacl-long.394); [GitHub](https://github.com/shizhediao/r-tuning)

### Inferences
- Trained refusal changes the model weights and needs a labelled known/unknown split per model; a probe or logprob gate on a frozen model is the less invasive alternative.

### Gaps
- SQuAD 2.0-style unanswerables, over-refusal, threshold choice, benchmark leakage, and coverage-risk pitfalls: no sources found.

## What can be computed from llama.cpp or vLLM versus HF Transformers

### Takeaway
Token logprobs are available from both servers. Hidden-state probes and EigenScore-like methods need hidden-state access, which is clearly offered only by HF Transformers (and vLLM's pooling path, which is a separate mode). llama.cpp hidden-state/embedding support was not verified.

### Cited Findings
- vLLM returns logprobs, not raw logits, by default, for generated tokens, with prompt logprobs as a separate request option. Returned logprobs may not reflect the final adjusted sampling distribution. Raw logits need patching the sampler. — [vLLM forum](https://discuss.vllm.ai/t/how-to-obtain-the-logits-of-llm/847); [vLLM V1 user guide](https://docs.vllm.ai/en/v0.8.4/_sources/getting_started/v1_user_guide.md); [vLLM outputs docs](https://docs.vllm.ai/en/latest/api/vllm/outputs/)
- vLLM exposes hidden states through its pooling/embedding path (`PoolingOutput`, `EmbeddingOutput`), not through the normal generation output. — [vLLM outputs docs](https://docs.vllm.ai/en/latest/api/vllm/outputs/)
- llama.cpp server `/completion` supports `n_probs` (top-N token probabilities per generated token, in `completion_probabilities`), and `/chat/completions` accepts `logprobs` and `top_logprobs`. Commit #10783 made logprobs OpenAI-compatible; check the current README for exact behavior. — [llama.cpp commit](https://cdn04132025.gitlink.org.cn/replica/llama.cpp/commit/57bb2c40cd94c5a09f5210ed8264cc93b21c4b7e)

### Inferences
- Feasible from servers: max-prob, entropy over top-N, margin (top-1 minus top-2), mean and length-normalized logprob, k-sample consistency and semantic entropy (via k requests plus an NLI/embedding step), verbalized confidence, P(True) (a second call).
- Needs HF Transformers (or custom engine hooks): SAPLMA/SEP probes, EigenScore, LLM-Check (hidden states and attention maps). The top-N limit means full-vocabulary entropy is not exact from llama.cpp.

### Gaps
- llama.cpp embeddings/hidden-state API: not found. Check the `ggml-org/llama.cpp` `tools/server/README.md` directly.
- Whether the GGUF-quantized model's hidden states give probes as good as FP16 (only the unreviewed 2026 preprint touches quantized models).

## Open problems and failure modes

### Takeaway
Main documented failures: overconfidence on wrong answers, especially in RLHF and reasoning models. Semantic entropy detects only confabulations. Probe calibration does not transfer to new tasks. Conformal guarantees depend on exchangeability.

### Cited Findings
- See above: verbalized overconfidence ([Mei et al.](https://arxiv.org/abs/2506.18183v3), [Xiong et al.](https://arxiv.org/pdf/2306.13063)); P(IK) calibration transfers poorly ([Kadavath et al.](https://arxiv.org/pdf/2207.05221)); semantic entropy scope and clustering errors ([OATML blog](https://oatml.cs.ox.ac.uk/blog/2024/06/19/detecting_hallucinations_2024.html)); quantization shifts confidence ([Proskurina et al.](https://arxiv.org/pdf/2405.00632)).

### Inferences
- A model can be consistently wrong (same wrong answer on every sample), which sampling-agreement methods score as confident. This follows from the method's design, though I did not find a source quantifying it.
- Calibration (ECE) and discrimination (AUROC) are separate: a gate needs good ranking; thresholds then need recalibration per deployment.

### Gaps
- Distribution-shift evidence for cheap signals, and calibration-versus-discrimination analyses: not found in this pass. A follow-up pass with working arxiv.org access would be needed for the numeric benchmarks.
