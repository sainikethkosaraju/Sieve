# Prior work: LLM cascades, routing, selective/abstention-aware inference (for Sieve novelty assessment)

Research caveats: arxiv.org was NOT reachable from WebFetch (DNS blocked in this session), so no paper was read in full. Every 2025-2026 claim below comes from WebSearch result summaries (abstract/snippet level) and is marked "snippet-level". Items marked [from memory] are from background knowledge, not verified this session. A follow-up with arxiv.org allowed is strongly recommended before any novelty claim is made publicly.

## 1. Key papers on cascades and routers (signals, evaluation, reporting)

### Takeaway
The foundational router/cascade literature (FrugalGPT, AutoMix, Hybrid LLM, RouteLLM, RouterBench) is well established and almost entirely measures cost in API dollars or call fractions, with quality as accuracy. Signals are learned scorers, self-verification, or preference-trained routers rather than calibrated risk guarantees. Only a few of these were re-confirmed this session.

### Cited Findings
- FrugalGPT (2023): cascade that learns which LLM combinations to use per query to cut cost and improve accuracy — [Not-Diamond awesome list](https://github.com/Not-Diamond/awesome-ai-model-routing)
- AutoMix (2024): routes to larger LMs based on approximate correctness of the small LM's output (self-verification) — [Not-Diamond awesome list](https://github.com/Not-Diamond/awesome-ai-model-routing)
- Hybrid LLM (2024): router assigns query to small or large model by predicted difficulty and a desired quality level — [Not-Diamond awesome list](https://github.com/Not-Diamond/awesome-ai-model-routing)
- RouteLLM (2024): strong-vs-weak routers trained on human preference data with augmentation — [Not-Diamond awesome list](https://github.com/Not-Diamond/awesome-ai-model-routing); [write-up](https://shekhargulati.com/2024/07/09/routellm-paper/)
- RouterBench (2023): routing benchmark with a "willingness to pay" quality-cost trade-off (identification only moderately clear in the source) — [Not-Diamond awesome list](https://github.com/Not-Diamond/awesome-ai-model-routing); related-work summary at [qeios](https://qeios.com/read/NS4GU3/pdf)
- MixLLM (2025): dynamic routing among mixed LLMs — [arXiv 2502.18482](https://arxiv.org/html/2502.18482v1)
- Practitioner comparison of RouteLLM, FrugalGPT, embedding classifiers and verifier cascades; router overhead roughly 2-8 ms (embedding classifier), 15-35 ms (GPU cross-encoder), 150-450 ms (small-LLM router); cascades suffer tail-latency amplification. Blog, figures unverified — [llms.blog](https://www.llms.blog/posts/model-routing-and-cascades-in-production-comparing-routellm-frugalgpt-embedding-classifiers-and-verifier-cascades)

### Inferences
- Classic routers optimize a quality-vs-dollar frontier; none of the confirmed items treat abstention as a first-class output.

### Gaps
- RouterEval and speculative cascades (Google, 2024/25 [from memory]) returned nothing in search; venues for all of the above (FrugalGPT TMLR, AutoMix NeurIPS 2024, Hybrid LLM ICLR 2024, RouteLLM ICLR 2025 [all from memory, unverified]) were not confirmed. Mixture-of-Thought cascades (Yue et al., ICLR 2024 [from memory]) not found.

## 2. Routing combined with abstention / selective prediction / deferral

### Takeaway
This is the crowded part. At least five 2025-2026 works already combine cascades with abstention, calibration, conformal guarantees or human deferral. A "calibrated, abstention-aware cascade" per se is NOT novel.

### Cited Findings
- Cost-Saving LLM Cascades with Early Abstention (2025): small models abstain on queries they predict larger downstream models would also abstain on; reported ~4.1% absolute average abstention-rate rise (snippet-level) — [arXiv 2502.09054](https://arxiv.org/pdf/2502.09054)
- Efficiently Deploying LLMs with Controlled Risk (HCMA, 2024): hierarchical chain, token probabilities for routing and abstention, improved Platt scaling; finds multi-level abstention beneficial (snippet-level) — [arXiv 2410.02173](https://arxiv.org/pdf/2410.02173)
- UCCI: Calibrated Uncertainty for Cost-Optimal LLM Cascade Routing (May 2026): token-margin aggregation plus isotonic regression, ECE 0.03 on their workload, cost-constrained routing (snippet-level; method details not seen) — [arXiv 2605.18796](https://arxiv.org/pdf/2605.18796)
- Conformal Cascade: Distribution-Free Accuracy Guarantees for Multi-Tier LLM Inference (arXiv 2607.25018, ~Jul 2026): uses conformal prediction-set size as deferral signal; finite-sample accuracy guarantee; argues confidence thresholds are miscalibrated and must be tuned per model pair/domain (snippet-level) — [arXiv 2607.25018](https://arxiv.org/abs/2607.25018)
- Signed Rescue Routing: Harm-Aware Cascades for Efficient LLM Inference (arXiv 2609.07786, ~Sep 2026): predicts signed benefit of escalation with separate "rescue" and "harm" heads on cheap runtime features of the small model (snippet-level; no results seen) — [arXiv 2609.07786](https://arxiv.org/pdf/2609.07786)
- ReDAct: Uncertainty-Aware Deferral for LLM Agents (Apr 2026): small model default, calibrated uncertainty threshold defers to large; reported ~15% large-model calls (snippet-level) — [arXiv 2604.07036](https://arxiv.org/html/2604.07036v1)
- Cascaded Language Models for Cost-Effective Human-AI Decision-Making (NeurIPS 2025): base model, large model, human expert with explicit deferral and abstention policies — [NeurIPS PDF](https://papers.neurips.cc/paper_files/paper/2025/file/10e0c427408ccc6e073d9464e2280f89-Paper-Conference.pdf); [arXiv 2506.11887](https://arxiv.org/html/2506.11887v2)
- ICML 2026 paper on type-aware selective deferral / cascade routing for structured attribute generation with heterogeneous confidence — [ICML virtual page](https://icml.cc/virtual/2026/75219) (title partially seen; authors not captured)
- R-Tuning (Zhang et al., NAACL 2024): refusal-aware instruction tuning teaches "I don't know"; refusal is a task-agnostic meta-skill — [ACL Anthology](https://aclanthology.org/2024.naacl-long.394); [arXiv 2311.09677](https://arxiv.org/pdf/2311.09677)
- Related: "When Efficiency Backfires: Cascading LLMs Trigger Cascade Failure under Adversarial Attack" (2026) — [arXiv 2605.17288](https://arxiv.org/pdf/2605.17288) (only title seen)

### Inferences
- Calibration (UCCI, HCMA), conformal guarantees (Conformal Cascade), abstention (Early Abstention, HCMA, Cascaded Human-AI) and risk-aware escalation (Signed Rescue) each exist separately or in pairs.

### Gaps
- Did not find or verify selective-classification-for-LLM papers (Geifman/El-Yaniv, Kamath 2020 [from memory]), conformal abstention for LLMs, or the learning-to-defer lineage (Madras 2018, Mozannar and Sontag 2020 [from memory]); search returned only R-Tuning. Full texts not read, so whether any of these report risk-coverage curves, AURC, or selective-risk at fixed coverage is unknown.

## 3. Cost as latency / VRAM / energy on consumer hardware

### Takeaway
Hardware-grounded cost exists in edge-inference and energy-routing work, but I found no cascade paper that jointly does calibrated abstention and consumer-GPU latency/VRAM accounting. This looks like the more open axis, with the caveat of limited search depth.

### Cited Findings
- Measured Joules, Learned Routes (arXiv 2609.23085, ~Sep 2026): router trained on measured GPU energy; lower energy than baselines at comparable accuracy; routing overhead unevaluated (snippet-level) — [arXiv 2609.23085](https://arxiv.org/pdf/2609.23085)
- Edge-Cascade (GitHub project): laptop cascade Intel NPU -> NVIDIA GPU (RTX 5070 Ti, 14B via Ollama) -> Claude cloud, gated by an objective code verifier; anecdotal personal project, not a paper — [repo](https://github.com/danthemanvsqz/Edge-Cascade)
- EdgeReasoning: 1.5B model ~7x better energy/token than 14B on edge GPUs — [arXiv 2511.01866](https://arxiv.org/pdf/2511.01866)
- Camel: energy-aware LLM inference on resource-constrained devices — [arXiv 2508.09173](https://arxiv.org/pdf/2508.09173) (only title seen)
- Cloud to Edge benchmarking on single-board computers; up to 40x energy efficiency for accelerators vs CPU (snippet-level) — [arXiv 2604.24785](https://arxiv.org/html/2604.24785v1)
- SpecEdge: consumer-GPU drafting with server verification, 1.91x cost reduction (speculative decoding, not cascade) — [arXiv 2505.17052](https://arxiv.org/pdf/2505.17052)
- Idle consumer GPUs: H100 ~3x more energy-efficient per token than RTX 4090 — [ACM](https://dl.acm.org/doi/pdf/10.1145/3775043.3775047)

### Inferences
- Energy/latency-aware routing exists (Measured Joules) but without abstention or calibration guarantees as far as snippets show.

### Gaps
- No evidence either way on VRAM co-residency / model-swap cost in cascades. Not verified whether UCCI, Conformal Cascade or Signed Rescue use local hardware (only snippets seen).

## 4. Is the exact combination already done (2025-2026)?

### Takeaway
Not exactly as stated, per what I could see, but the nearest items (UCCI, Conformal Cascade, Early Abstention, HCMA, Signed Rescue) cover most of the algorithmic ingredients, and Edge-Cascade covers the local-hardware framing informally. Confidence is moderate-low because no full texts were read and arXiv was unreachable; latest items found are Sep 2026 (Signed Rescue, Measured Joules); nothing from Oct-Nov 2026 surfaced.

### Cited Findings
- See sections 2 and 3 for sources; no single source found combining calibration + abstention + small-to-large + consumer-hardware cost measurement.

### Inferences
- Verdict: novelty rests on the integration and the evaluation protocol (risk-coverage plus measured latency/VRAM/energy on consumer GPUs with a reproducible serving stack), not on the cascade/calibration/abstention ideas individually.

### Gaps
- Unchecked: Semantic Scholar citation graphs, GitHub repos, newest Oct-Nov 2026 arXiv listings, OpenReview ICLR 2027 submissions.

## 5. Most threatening overlaps and how Sieve could differ

### Takeaway
Five works to read in full first; Sieve differentiation should be built on hardware-measured cost and joint risk-coverage-cost evaluation.

### Cited Findings and Inferences
1. Conformal Cascade ([2607.25018](https://arxiv.org/abs/2607.25018)): distribution-free accuracy guarantee in multi-tier cascade. Threat: guarantees plus deferral. Differ: check whether it has an abstain action and non-dollar cost; Sieve could add abstention as a first-class terminal action and report measured hardware cost.
2. UCCI ([2605.18796](https://arxiv.org/pdf/2605.18796)): calibrated cost-optimal cascade routing. Threat: calibration claim (ECE 0.03) and cost-constrained routing. Differ: cost unit and whether abstention exists (unseen).
3. Cost-Saving LLM Cascades with Early Abstention ([2502.09054](https://arxiv.org/pdf/2502.09054)) and HCMA ([2410.02173](https://arxiv.org/pdf/2410.02173)): direct abstention-in-cascade precedents. Differ: calibration under shift, risk-coverage reporting, consumer hardware.
4. Signed Rescue Routing ([2609.07786](https://arxiv.org/pdf/2609.07786)): addresses large-model failure (harm) which a naive escalate-on-uncertainty cascade ignores; Sieve should account for large-model errors and abstain when neither model is trusted.
5. Measured Joules / Edge-Cascade ([2609.23085](https://arxiv.org/pdf/2609.23085), [repo](https://github.com/danthemanvsqz/Edge-Cascade)): hardware-measured cost precedent. Differ: they lack calibrated abstention (as far as seen).

### Gaps
- All differentiators are hypotheses until the five papers are read in full.
