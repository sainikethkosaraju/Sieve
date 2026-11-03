# Sieve novelty verification (2026-11-03)

## Method and provenance

- WebFetch could NOT reach arxiv.org, alphaxiv or ar5iv (DNS failure, "ENOTFOUND"). arxiv.org is reachable through the session proxy with `curl` from Bash. All arXiv material below was read that way: abstract pages (`/abs/<id>`) and full PDFs (`/pdf/<id>`, text-extracted with pdftotext). I grepped the full text for signals, hardware, quantization, code, abstention and streaming, and read the relevant passages and tables. I did not read every page of every paper.
- `export.arxiv.org` (the API) does not resolve. `arxiv.org/list/cs.CL/recent` returned HTTP 400 and `arxiv.org/a/<author>` returned 404. The only discovery channel that worked was `arxiv.org/search/?query=...` (HTML), sorted by newest announcement. Discovery therefore consists of about 15 keyword queries and is NOT exhaustive. The streaming query returned nothing relevant.
- GitHub API access to danthemanvsqz/Edge-Cascade is blocked in this session. I read its README through raw.githubusercontent.com (VERIFIED). Stars and commit history are not verified.
- Labels: VERIFIED = primary source (abstract page and PDF) read this session. ABSTRACT-ONLY = abstract page read, full text not read.

## 1. ID resolution: all seven IDs are real and match their descriptions

| ID | Resolves to | Authors | Submitted |
|---|---|---|---|
| 2607.25018 | Conformal Cascade: Distribution-Free Accuracy Guarantees for Multi-Tier LLM Inference | Dou, Lian, Li | 2026-07-27 |
| 2605.18796 | UCCI: Calibrated Uncertainty for Cost-Optimal LLM Cascade Routing | Varun Kotte (sole author) | 2026-05-11 |
| 2502.09054 | Cost-Saving LLM Cascades with Early Abstention | Zellinger, Liu, Thomson | 2025-02-13 (v2 2025-03-29) |
| 2410.02173 | Efficiently Deploying LLMs with Controlled Risk (HCMA) | Zellinger, Thomson | 2024-10-03 |
| 2609.07786 | Signed Rescue Routing: Harm-Aware Cascades for Efficient LLM Inference | Wang, Li, Song, Chen, Song, Liu | 2026-09-07 |
| 2609.23085 | Measured Joules, Learned Routes: Learning to Route for Energy-Efficient LLM Serving | Siddiqui, Rojas, Yang, Cui, Shi, Chen | 2026-09-19 |
| 2606.02628 | Hallucination Is Linearly Decodable from Mid-Layer Hidden States in Quantized LLMs | Aiersilan (sole author) | 2026-05-30 |

No ID was wrong or hallucinated. The earlier draft report's descriptions are broadly accurate. Corrections to it:
- 2609.07786's abstract still contains literal "TBD" placeholders, but the body has real results (Table 1).
- 2609.23085 reports about 23% lower mean answer energy.
- 2606.02628 is a hidden-state probing paper that happens to use NF4-quantized models. It is not a quantization study (details below).

## 2. Per-paper findings (all VERIFIED, abstract plus full text)

### 2607.25018 Conformal Cascade (CC)
- **What it does.** Conformal prediction-set size is the deferral rule: accept at a tier if the calibrated set collapses to one answer, otherwise defer. It gives a finite-sample accuracy guarantee of at least 1 - K*alpha through a per-tier union bound, tightening to 1 - alpha under a "selection-preservation" condition the authors say is not implied by their results. The guarantee is marginal.
- **Signals.** Sampling-frequency nonconformity from N=16 samples per tier at temperature 0.7. A logprob variant, s = 1 - p(a|x), is in an appendix and only discussed as an alternative where logprobs are available. It is black-box, with no training.
- **Datasets and scope.** 18 multiple-choice benchmarks, 4 open-weight families (Llama, Gemma, Ministral, Phi), two-tier only, 30/70 calibration/test splits. Open-ended generation needs an answer-clustering step that the paper "leaves for future work".
- **Abstention.** None. Cost is abstract (c1=1, c2=2.7). The paper is cascade-only with no abstain action.
- **Hardware.** Four B200 GPUs through vLLM. No latency, VRAM or energy measurements.
- **Quantization or KV-cache.** None. The only KV mention is a caveat that shared KV-cache prefixes could break independence.
- **Code.** No code link found. It only says a per-benchmark Pareto visualisation is released.

### 2605.18796 UCCI
- **What it does.** Token-level margin uncertainty is mapped to a per-query error probability with isotonic regression. The escalation threshold comes from constrained cost minimization. It proves cost-optimality of threshold policies under three assumptions and O(n^-1/3) sample complexity for ECE.
- **Signals and data.** Token margin (top-1 minus top-2), scored locally. Beats entropy thresholding, split-conformal routing and a FrugalGPT-style learned threshold. Workload is a production NER task of 75,000 queries over 6 entity types, not QA.
- **Results.** 4B and 12B instruction-tuned models via vLLM 0.4.2 on H100 80GB. Cost reduction of 31% (95% CI 27-35%) at micro-F1 0.91; ECE 0.12 to 0.03.
- **Cost and hardware.** Cost is measured end-to-end latency in H100 milliseconds (47.2 ms for 4B, 142.3 ms for 12B, averaged over 100 queries with cold cache). It explicitly does NOT measure energy, throughput under batching, or VRAM. Limitations name energy and dollar cost as future work.
- **Abstention.** None. Abstention appears only in related-work text. The paper cites 2502.09054 as concurrent work.
- **Streaming.** Only as a limitation: "In streaming deployments with distribution shift, online or continual recalibration would be needed." This is distribution-shift recalibration, not token-level streaming gating.
- **Quantization or KV-cache.** None.
- **Code.** Released: https://github.com/varunkotte6/ucci (stated in the paper). Not inspected, because GitHub access is blocked.
- **Risk guarantee.** Calibrated cost-optimality, not a distribution-free risk bound.

### 2502.09054 Early Abstention in cascades
- **What it does.** Adds a per-tier abstention threshold below the deferral threshold, so small models can abstain "early" and avoid running expensive models. Averaged over six benchmarks it lowers test loss by 2.2%, with a 4.1% higher abstention rate, 13.0% lower cost and 5.0% lower error rate. (The earlier report's "4.1% absolute gain" was a loose description. The 4.1% is an increase in abstention rate traded for those gains.)
- **Signals.** Raw max token probability over valid answer choices plus calibration. Black-box.
- **Data.** GSM8K, MedMCQA, MMLU, TriviaQA, TruthfulQA, XSum. Llama3 1B to 405B chain, GPT-4o-mini to GPT-4o, and Qwen2.5 32B Coder to 72B.
- **Cost and hardware.** Dollar cost, with a latency proxy. Hosted APIs. No GPU, VRAM or energy.
- **Quantization or KV-cache.** None. No code link found.

### 2410.02173 HCMA (Efficiently Deploying LLMs with Controlled Risk)
- **What it does.** Hierarchical chains with multi-level abstention. Each tier either answers, delegates or rejects, based on calibrated token-probability confidence. Calibration is a logistic regression on a nonlinear feature transform that needs only 50-100 labelled examples and cuts ECE by 50% versus naive Platt scaling. It reports error-abstention-cost curves, which is the closest existing analogue to Sieve's risk-coverage-cost frontier.
- **Data and cost.** MMLU (headline: 30% lower error for Llama3 405B at 20% abstention), TruthfulQA, Llama3 1B/3B/8B/70B/405B. Dollar cost, plus API latency measured from California against Fireworks (7% dollar and 6% latency improvement from early abstention).
- **Guarantee.** No formal guarantee. Risk control is empirical through calibration and thresholds.
- **Hardware, quantization, KV-cache, code.** None found, and no hardware measurements.

### 2609.07786 Signed Rescue Routing (SRR)
- **What it does.** A budgeted router predicts two events separately, "rescue" (large fixes small) and "harm" (large breaks a correct answer), and ranks requests by the difference. It claims this signed gain is the Bayes-optimal score under a fixed escalation budget. There is a lightweight two-head MLP router on the small model's output statistics (margin and entropy among them).
- **Setup.** Qwen3-4B and Qwen3-8B, BF16, PyTorch SDPA, A100s, 5,165 examples (MMLU, HellaSwag, ARC-Challenge). Scores come from next-token logits over four answer letters.
- **Results.** Held-out AUACC 0.6811 for SRR versus 0.6716 for an error predictor; at 50% escalation 69.4%, above the 68.9% large-model endpoint.
- **Latency.** 12.8 ms (4B) and 19.6 ms (8B) per example in a batched setup. No VRAM or energy.
- **Abstention, quantization, KV-cache.** None. No code link found.
- **Note.** Its harm/rescue idea is relevant to Sieve's "abstain when neither tier is trusted" design, but SRR itself has no abstain action.

### 2609.23085 Measured Joules, Learned Routes
- **What it does.** A small LLM router (Qwen2.5-1.5B-Instruct, SFT then GRPO) picks one model from a pool of 0.5B-32B models (Qwen2.5 0.5B-7B, Llama 3.1 8B, Gemma2-2B, DeepSeek-R1 8B/32B). Training uses an offline "tournament" that logs each model's correctness, latency, power and energy for every training question.
- **Hardware.** Models served through Ollama on A100-SXM4-80GB. Power is polled through NVML (pynvml) every 200 ms with an idle baseline subtracted. This is the strongest measured-energy precedent for Sieve.
- **Results.** Seven benchmarks; about 23% lower mean answer energy (1692 J versus the baseline) at comparable accuracy, and a sharp accuracy-energy "phase transition" near 60% accuracy.
- **Abstention, calibration, confidence signals, risk guarantee.** None. It is a pre-answer query router. It does not read the model's own confidence.
- **Quantization or KV-cache.** None. No VRAM numbers.
- **Code.** "We will open-source" only. No link yet.

### 2606.02628 Mid-layer linear probes in quantized models
- **What it does.** Linear and MLP probes on per-layer hidden states detect hallucination, compared with INSIDE EigenScore, self-consistency and attention entropy.
- **Models and data.** Llama-3.1-8B, Mistral-7B and Qwen2.5-7B, all loaded in 4-bit NF4. TruthfulQA, HaluEval-QA, FEVER and a synthetic set. Mid-layer probe AUROC 0.904-1.000, versus at most 0.541 for sampling methods.
- **Caveat.** The authors themselves say the gap reflects a "structural mismatch between paired-label evaluation and the information these methods access". The comparison is protocol-dependent.
- **Quantization.** It uses 4-bit models but has NO full-precision baseline ("we do not include a direct full-precision baseline"), so it measures nothing about how quantization shifts confidence. It is not a quantization study.
- **Hardware.** One RTX 5060 8 GB, 4-8 hours total. Lists quantized weight VRAM per model. No latency or energy.
- **Cascade, abstention, KV-cache.** None.
- **Code.** Released on GitHub (link truncated in the extracted text; not inspected). It needs HF Transformers hidden-state access, which conflicts with the llama-server proxy design.

## 3. New papers found in fresh arXiv searches (Oct-Nov 2026)

| ID | Title | Date | Status |
|---|---|---|---|
| 2610.23249 | Who Pays for Efficient Inference? Confidence-Based Small-to-Large LLM Routing Across Tasks and Languages (Hossen et al.) | 2026-10-16 | **VERIFIED** (abstract and PDF) |
| 2608.03854 | When Calibration Depends on the Scoring Rule: Quantized Biomedical LLM Classification (Rasmussen, Qin) | 2026-08 (listed announced 2026-09-20) | ABSTRACT-ONLY |
| 2610.38210 | GAAC: Lossless KV Compression for Bandwidth-Efficient LLM Inference on Personal Devices | 2026-10-27 | title only |
| 2610.39853 | KV Cache Quantization for Diffusion Language Models | 2026-10-29 | title only |
| 2610.34329 | SHIFT-JEV: Certified Abstention and Gain-Gated Continual Adaptation for LLM-Based Decision Models under Distribution Shift | 2026-10-25 | title only |
| 2604.12262 | CascadeDebate: Multi-Agent Deliberation for Cost-Aware LLM Cascades | 2026-04 | title only |
| 2602.15391 | Improving LLM Reliability through Hybrid Abstention and Adaptive Detection | 2026-02 | title only |
| 2609.35831 | Adaptive Entropy-Based Routing for Hybrid Retrieval and Long-Context LMs | 2026-09 | title only |

**2610.23249 is the most relevant new find.**
- It is a systematic evaluation of confidence-based small-to-large routing (max next-token probability over MCQ choices), covering 3 families (Qwen3 1.7B/4B, Gemma 3 1B/4B, Llama 3.2 1B/3B), 4 languages, 3 tasks and 42,288 predictions.
- Result: confidence works on Belebele (AUC 0.612) and SIB-200 (AUC 0.705), but is near chance on MMLU-ProX-Lite (AUC 0.510), where small models stay confident while wrong. Global thresholds create language escalation gaps of up to 30 points; equal-quota routing removes them.
- It measures wall time, throughput and GPU energy (NVML every 50 ms) on two Tesla T4s (a datacenter card), at FP16 with some FP32. It reports 27.5-58.2% time savings and 26.3-62.6% energy savings from routing.
- It has no abstention, no risk guarantee, no quantization study, no VRAM measurement and no streaming. Code is linked ("available at GitHub"; URL not captured).
- Implication: measured energy and latency for confidence routing is no longer unprecedented, and its "confidence fails on knowledge/reasoning" finding is a result Sieve must engage with.

**2608.03854 (abstract only).**
- Studies Mistral-7B variants at FP16, INT8 and INT4 on medical sentence classification and finds calibration depends on the scoring rule (summed versus mean-token log-likelihood).
- Shows weight-quantization effects on calibration are already being studied on consumer hardware, but there is no routing, no KV-cache and no cascade.

**Searches that returned nothing relevant** (limited and not exhaustive): KV-cache quantization combined with calibration or confidence, conformal risk control for LLM routing, streaming or early-abstention gating, logprob gates for local LLMs. Absence here is weak evidence only.

## 4. Edge-Cascade (danthemanvsqz/Edge-Cascade, README VERIFIED)

- It is a local-first coding workflow on an Intel Core Ultra and NVIDIA laptop.
  - Tier 1 is an Intel NPU running qwen2.5-coder-1.5b INT4 through OpenVINO.
  - Tier 2 is an RTX 5070 Ti running qwen2.5-coder:14b through Ollama.
  - Tier 3 is the Claude Code CLI session as orchestrator.
  - Tier 4 is the Anthropic API, credit-guarded and off by default.
- Escalation is gated by a deterministic verifier (AST and sandboxed execution of code), NOT model confidence. Local failures escalate after retries.
- It has no logprob signals, calibration, abstention, risk guarantee, quantization study or latency/VRAM/energy measurements. It is MIT licensed, with CI and tests.
- Verdict: no overlap with Sieve's contributions. It is a different design (verifier-gated, coding, MCP tools). Stars are not verified.

## 5. Overlap verdict per Sieve contribution

| # | Claim | Verdict | Closest prior work and what remains open |
|---|---|---|---|
| 1 | Measured risk-coverage-cost frontier with real latency, VRAM and energy on consumer GPUs | **Partly overlapping** | HCMA and Early Abstention give error-abstention-cost curves, with dollar and API-latency cost and no hardware. UCCI (H100 ms), SRR (A100 ms), Measured Joules (A100 NVML joules via Ollama) and 2610.23249 (T4 joules and wall time) each measure real hardware cost for routing, none with abstention, and none on consumer GPUs. Nobody measures VRAM or co-residency and swap cost. Open: a combined abstention + calibrated risk + measured latency/VRAM/energy frontier on consumer GPUs. The "measured cost" half alone is no longer novel. |
| 2 | How weight quantization and KV-cache quantization shift the confidence gate | **Weights: partly overlapping. KV-cache: open (no hit found)** | 2606.02628 uses NF4 but has no FP baseline and no gate. 2608.03854 (abstract) compares FP16/INT8/INT4 calibration on classification with no routing. Earlier snippet-level quantization-calibration papers (2405.00632, 2508.16785, 2607.10855) were not re-read. No paper found links KV-cache quantization to calibration or routing; the Oct 2026 KV-quant papers I saw are efficiency or compression papers by title. No paper studies GGUF K-quants. Open: effect on gate thresholds, risk-coverage and guarantee validity, plus KV-cache quant (absence is weak evidence given the search limits). |
| 3 | Streaming-compatible gating | **Open (weak evidence)** | All verified cascades gate after a full answer or a single next-token MCQ score. UCCI's only "streaming" mention concerns distribution-shift recalibration. My streaming/early-exit searches returned no relevant arXiv hits, but the search channel is limited, so confirm with a manual Scholar/OpenReview scan. Classic early-exit and speculative cascades (cited in UCCI) are model-internal, not proxy-level token gating. |
| 4 | Packaged, runnable OpenAI-compatible tool over llama-server/Ollama | **Open** | UCCI has calibration scripts, 2606.02628 and 2610.23249 have research code, and Measured Joules uses Ollama with an unreleased router. None is an OpenAI-compatible proxy with abstention. Edge-Cascade is a different design. Not rechecked this session: GitHub topic scan and the README claims for RouteLLM and vLLM Semantic Router. |

Related note on the "conformal-style risk guarantee": Conformal Cascade scoops the generic idea of conformal guarantees for LLM cascades, but only for multiple choice, with a union bound across tiers, N=16 sampling and no abstention. It explicitly leaves open-ended generation to future work. A guarantee that covers abstention and open-ended generation is therefore still unclaimed, but it is also the hard part. The earlier report's framing, "do not claim the algorithm", is confirmed.

## 6. Overall verdict and reframing

- **Does (1)-(4) still stand?** Not as an unqualified novelty claim.
  - (1) Is narrowed. The measured-hardware-cost angle has four precedents (UCCI, SRR, Measured Joules, 2610.23249), all on datacenter GPUs (A100/H100/T4) and none with abstention. It survives only as "abstention plus risk control plus latency, VRAM and energy on consumer GPUs".
  - (2) Stands for KV-cache and for the effect on gate thresholds, risk-coverage and guarantee validity. It is weaker for weights alone, because quantization-calibration studies exist.
  - (3) Appears open but rests on weak search evidence.
  - (4) Open.
- **Recommended reframing.** Lead with an empirical-systems claim, not an algorithmic one: "What a calibrated, abstaining local cascade really costs and how reliable it stays across quantization levels, on consumer hardware."
  - Primary contribution: a measurement study plus open tool. The study covers the risk-coverage-cost frontier with latency, VRAM and joules, and gate drift under weight and KV-cache quantization (GGUF K-quants, q8_0 and q4_0 KV), with per-quantization recalibration as the practical recipe.
  - Secondary: streaming gating (abort or escalate mid-generation) and generation-time abstention, explicitly labelled as an extension beyond CC's MCQ scope.
  - Engage 2610.23249's finding that confidence routing fails on knowledge/reasoning tasks (AUC 0.51 on MMLU-ProX-Lite) and say honestly where the gate works and where it does not.
  - Cite and compare against UCCI (margin plus isotonic), HCMA/Early Abstention (abstain tiers), CC (conformal MCQ), SRR (harm-aware) and Measured Joules/2610.23249 (energy).
  - Do not claim: calibrated cascades, abstention in cascades, conformal cascade guarantees, or measured-energy routing as new.
  - Treat hidden-state probes (2606.02628) as an optional offline baseline, not a core contribution, since the paper's headline protocol is paired true/false and it needs Transformers.
- **Still to check before submitting.**
  - Inspect the released code (UCCI repo and the 2610.23249 and 2606.02628 repos).
  - Run a manual Semantic Scholar, Google Scholar and ICLR/OpenReview scan for KV-cache-quantization-plus-calibration and streaming gating, since arXiv listing pages and the API were unreachable.
  - Re-read 2608.03854 in full.
