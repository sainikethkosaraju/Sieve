# Competing open-source LLM serving, gateway, and routing projects (as of 2026-11-03)

Method note: direct GitHub API access (gh/curl) was blocked in this environment, so star counts come from WebFetch of GitHub repo pages (captured 2026-11-03; page-summarizer tool, so the Releases and last-commit fields were usually NOT captured). Release cadence, feature claims not on the fetched pages, and community-channel claims are not verified here and are listed under Gaps.

## Serving engines: features, stars, activity

### Takeaway
The big engines (llama.cpp, vLLM, Ollama, SGLang) are all very large and active, and all expose OpenAI-compatible servers. Their feature details (logprobs, KV quant, hidden states) could not be verified from the fetched pages. The engines are infrastructure; none ships routing or abstention policy.

### Cited Findings
- llama.cpp: ~133.2k stars; OpenAI-compatible server (llama-server), built-in web UI; 1.5-bit to 8-bit weight quantization; CPU+GPU hybrid; CUDA, Vulkan, SYCL, Metal backends — [GitHub ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (fetched 2026-11-03)
- vLLM: ~96.2k stars, ~24.3k forks; OpenAI-compatible API plus Anthropic Messages API and gRPC; speculative decoding methods n-gram, suffix, EAGLE, DFlash; FP8/INT8/GPTQ/AWQ quantization listed — [GitHub vllm-project/vllm](https://github.com/vllm-project/vllm) (2026-11-03)
- Ollama: ~185.7k stars — [GitHub ollama/ollama](https://github.com/ollama/ollama) (2026-11-03). Page gave no API/logprob detail.
- SGLang: ~38.4k stars, ~10.1k forks, Apache-2.0; targets agentic workloads, large-scale serving, RL rollouts; hardware: NVIDIA, AMD, TPU, Intel, Apple Silicon, Ascend; speculative decoding trained via separate SpecForge project — [GitHub sgl-project/sglang](https://github.com/sgl-project/sglang) (2026-11-03)
- mistral.rs: ~7.1k stars (aggregator, last push 2026-04-15) vs "over 7,600" (directory); figures conflict, third-party — [gittrend](https://gittrend.io/repo/EricLBuehler/mistral.rs), [everydev](https://www.everydev.ai/tools/mistral-rs/llms.txt)
- TensorRT-LLM: ~13.8k stars, last push 2026-06-07 per a third-party index (low reliability) — [goodfirstissue.org](https://www.goodfirstissue.org/pt-BR/NVIDIA/TensorRT-LLM)

### Inferences
- Ollama and llama.cpp dominate consumer-hardware mindshare by star count; Sieve's local-model target is almost certainly one of these (llama-server or Ollama) as a backend, so Sieve is a layer above them rather than a competitor.
- Star order: Ollama > llama.cpp > vLLM > SGLang > TensorRT-LLM > mistral.rs.

### Gaps
- Not verified: TGI, LM Studio (closed-source app, not OSS), MLC-LLM, ExLlamaV2/V3, LocalAI stars; latest release tags/dates for all engines; per-engine logprob, hidden-state, KV-cache-quantization support (from training knowledge only: llama.cpp supports quantized KV cache and logprobs; vLLM supports logprobs; hidden-state exposure is limited everywhere. Treat as unverified). TGI is believed to be in maintenance mode — unverified.

## Routers/gateways: what they route on

### Takeaway
Existing routers route on prompt features or static/operational policy (cost, load, provider health), not on the answering model's own post-hoc confidence. None of the fetched pages mention abstention.

### Cited Findings
- RouteLLM (lm-sys): ~5.7k stars, 450 forks, Apache-2.0, 29 open issues/16 open PRs, 175 commits; no releases listed. Routes on the prompt: a router (matrix factorization `mf` recommended; also `sw_ranking`, `bert`, `causal_llm`, `random`) predicts probability the strong model wins; above a calibrated threshold goes to strong model. Claims up to 85% cost reduction at 95% of GPT-4 quality on MT-Bench. OpenAI-compatible server/client — [GitHub lm-sys/RouteLLM](https://github.com/lm-sys/RouteLLM) (2026-11-03). Last-commit date not captured; activity level unconfirmed.
- vLLM Semantic Router: ~6.4k stars, 1.1k forks; "signal-driven" routing under a single `vllm-sr/auto` model ID; routes on task fit, compute availability, location constraints, quality/latency/cost priorities; features include hallucination detection (LettuceDetect v2), category-aware semantic cache, guardrails, Kubernetes; v0.4 "Hermes" dated 2026-09-24; no confidence/abstention/cascade described — [GitHub vllm-project/semantic-router](https://github.com/vllm-project/semantic-router) (2026-11-03)
- LiteLLM: ~62.6k stars, 13.0k forks; Router with model groups, retry/fallback across deployments, load balancing, and a heuristic complexity router (SIMPLE/MEDIUM -> `fast`, COMPLEX/REASONING -> `capable`); no confidence or cascade features — [GitHub BerriAI/litellm](https://github.com/BerriAI/litellm) (2026-11-03)
- Arch/Plano (katanemo/archgw): ~7.1k stars, 494 forks; routes by model name, alias, or preferences; a 4B routing model dispatches to agents from YAML descriptions; guardrails, OTel tracing (project now branded Plano) — [GitHub katanemo/archgw](https://github.com/katanemo/archgw) (2026-11-03)
- Portkey gateway: ~13.3k stars, 1.4k forks; fallbacks on configurable errors, retries up to 5 with exponential backoff, weighted load balancing, conditional routing, 40-50+ guardrails; "Gateway 2.0" pre-release merging into OSS — [GitHub Portkey-AI/gateway](https://github.com/Portkey-AI/gateway) (2026-11-03)

### Inferences
- Routing signal taxonomy: (a) prompt-predicted difficulty/preference (RouteLLM, Semantic Router, LiteLLM complexity router, Arch), (b) operational (Portkey, LiteLLM fallbacks: errors/latency/cost), (c) output guardrails (Portkey, Semantic Router hallucination detection). Model-internal confidence (logprobs/entropy) as an escalation trigger is absent from all of them.
- Guardrail-triggered retry (Portkey) is the nearest mechanism to post-hoc escalation, but it is rule/text-based, not calibrated confidence.

### Gaps
- Martian and NotDiamond: not fetched; believed closed/commercial APIs (Not Diamond has some OSS e.g. RoRF) — unverified.
- Release cadence of every router not captured.

## Does any project do post-hoc confidence-based escalation + abstention on local models?

### Takeaway
No mature, popular project found. Only academic-code repos and small hobby/middleware repos touch the idea, none with meaningful adoption.

### Cited Findings
- fanconic/cascaded-llms: code for the NeurIPS 2025 paper on a base model -> larger model -> human with abstention/deferral; 1 star, 4 forks, 68 commits; uses Llama/Qwen small models; needs an A100 80GB to regenerate; no license shown — [GitHub](https://github.com/fanconic/cascaded-llms), paper [arXiv 2506.11887](https://arxiv.org/html/2506.11887v3)
- ashutoshrana/confidence-escalation: framework-agnostic confidence-gated escalation middleware for agents (logprob, verbalized, tool-risk scoring; LangChain/CrewAI/AutoGen/ADK handlers); targets agent actions, not model-to-model cascades; star count not captured — [GitHub](https://github.com/ashutoshrana/confidence-escalation)
- Anupbhagat77/LLM-Model-Cascade: ACCEPT/ESCALATE with hosted Gemini Flash Lite plus GPT-4o-mini judge; cloud only — [GitHub](https://github.com/Anupbhagat77/LLM-Model-Cascade)
- Cheese-Singh/LLM-Cascade: weak-to-strong cascade for code gen — [GitHub](https://github.com/Cheese-Singh/LLM-Cascade)
- binzeli/hallucinationControl (I-CALM): prompt-only verbal-confidence abstention, no cascade — [GitHub](https://github.com/binzeli/hallucinationControl)
- Research: "Cost-Saving LLM Cascades with Early Abstention" — [arXiv 2502.09054](https://arxiv.org/pdf/2502.09054); curated list [qiqitao77/Large-Model-Uncertainty](https://github.com/qiqitao77/Large-Model-Uncertainty)

### Inferences
- The space is a genuine whitespace: research exists (cascades, selective prediction), but there is no packaged, OpenAI-compatible, locally runnable tool. This is based on one extended search, not an exhaustive topic crawl (llm-router, llm-cascade, selective-prediction topics were not enumerated).

### Gaps
- GitHub topic pages not enumerated; star counts of small repos not captured; possibly missed projects (e.g. FrugalGPT code, AutoMix, other cascade repos).

## Star counts, cadence, and what made successful ones take off

### Takeaway
Only star counts were gathered (above). Evidence on launch tactics was not collected.

### Cited Findings
- RouteLLM's README leads with a quantified headline (85% cost cut, 95% GPT-4 quality) and a drop-in OpenAI client — [GitHub lm-sys/RouteLLM](https://github.com/lm-sys/RouteLLM)
- vLLM Semantic Router shows rapid growth (~6.4k stars) with frequent named releases and model drops (v0.4 2026-09-24; Decision/Vela models 2026-10-14) and rides the vLLM brand — [GitHub](https://github.com/vllm-project/semantic-router)

### Inferences
- Pattern: successful projects pair a single quantified benchmark claim with drop-in OpenAI compatibility; ecosystem-backed routers (lm-sys, vLLM) get distribution from the parent brand, which a solo Sieve lacks.

### Gaps
- Release cadence, Discord/Slack channels, launch posts, HN/Reddit threads: not researched (no sources fetched).

## Gaps Sieve could fill

### Takeaway
Uncovered niche: calibrated, output-confidence-based escalation and explicit abstention for local models, behind an OpenAI-compatible endpoint.

### Inferences
- Missing in all surveyed projects: (1) routing on the local model's own logprobs/entropy after generation, (2) abstention ("I don't know"/refuse/defer to human) as a first-class outcome, (3) threshold calibration against a coverage/risk curve, (4) consumer-hardware (llama.cpp/Ollama) focus; RouteLLM-style routers are pre-hoc and cloud-weak/strong oriented.
- Honest risks: logprob exposure is uneven across engines (unverified); prompt-based routers cost zero extra generation while post-hoc escalation pays for the failed local attempt; vLLM Semantic Router's hallucination detection could expand toward this space.

### Gaps
- No user-demand evidence (HN/Reddit/issues) gathered for this niche.
