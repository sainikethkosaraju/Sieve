# Efficient LLM inference and benchmarking on consumer hardware (as of 2026-11-03)

Research limits: docs.vllm.ai and arxiv.org were not fetchable from this session (network allowlist), so vLLM docs and paper claims come from search-result summaries only, not full-page reads. Many web sources are SEO blogs; flagged below. No hands-on measurements were made.

## Speculative decoding: llama.cpp and vLLM state

### Takeaway
llama.cpp's official docs now list a broad `--spec-type` family (draft model, EAGLE-3, DFlash, DSpark, MTP, several n-gram modes), and vLLM supports EAGLE/EAGLE-3, MTP, draft model, n-gram, suffix and others. I found no reliable consumer-GPU speedup tables from the projects themselves; speedup depends on acceptance rate.

### Cited Findings
- llama.cpp `--spec-type` values: none, draft-simple, draft-eagle3, draft-dflash, draft-dspark, draft-mtp, ngram-cache, ngram-simple, ngram-map-k, ngram-map-k4v (experimental), ngram-mod; types can be combined comma-separated (e.g. `ngram-mod,ngram-map-k4v`); draftless takes precedence over a draft model — [llama.cpp speculative.md](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md)
- EAGLE-3 in llama.cpp needs a draft converted with `--target-model-dir`; SpecForge and vLLM/AngelSlim formats supported; supported-draft list tracked in issue #18039 — same source
- DSpark currently supports only Qwen3 backbones (Gemma4 planned); DFlash/DSpark `--spec-draft-n-max` is clamped to the trained block size; EAGLE-3/DFlash/DSpark drafts are target-specific — same source
- Defaults: `--spec-draft-n-max` 3, `--spec-draft-p-min` 0.00; ngram-mod uses ~16 MB hash pool shared across all server slots; ngram-mod notes: MoE models need long drafts, small n not recommended — same source
- The docs give no performance numbers; they point to a SPEED-Bench client in `tools/server/bench/speed-bench` measuring throughput, latency and acceptance against llama-server. They also provide `--spec-synth-rates`/`--spec-synth-len` for synthetic acceptance (benchmark only, invalid output) — same source
- Fixed seed: CPU vs backend sampling can still pick different tokens (floating-point); use greedy for exact matching — same source
- vLLM main docs: model-based methods (EAGLE, MTP, draft models, PARD, MLP) give the best latency reduction; n-gram and suffix decoding give modest gains without extra load at peak traffic; MTP is best when the target has native MTP and reuses the target model; configured via `--speculative-config` JSON — [vLLM docs (search summary only, page not fetched)](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
- Old llama.cpp flags `--draft`, `--draft-n`, `--draft-max` reportedly now error in favour of `--spec-draft-n-max` — [openclawdc blog via search](https://openclawdc.com/blog/speculative-decoding-local-llm-worth-it/) (third-party, verify)
- Third-party claim: speculation pays off above ~60-70% acceptance; one PyTorch/MPS experiment saw only ~0.95x at 87.3% acceptance because per-round overhead dominated — [dev.to](https://dev.to/cyprus09/is-speculative-decodings-speedup-a-hardware-problem-or-a-model-problem-7k6) (unverified, single experiment)
- Issue #12968: llama-server reportedly 10-15% slower than llama-speculative-simple even in a best case of repeated content (status unknown) — [issue 12968](https://github.com/ggml-org/llama.cpp/issues/12968)
- PR #17808 contains a TODO for a single draft context shared across slots and batches drafting across sequences — [PR 17808](https://github.com/ggml-org/llama.cpp/pull/17808/files)

### Inferences
- Because upstream publishes no consumer-GPU speedup numbers, Sieve could add a reproducible acceptance-rate + tokens/s harness (reusing SPEED-Bench and synthetic-acceptance flags) rather than building its own.
- Speculation helps single-stream latency but likely shrinks as batch size grows (general principle; not verified here against these sources).

### Gaps
- No verified consumer-GPU speedups (3090/4090/5090, 8-16 GB) for EAGLE-3, DFlash, MTP in llama.cpp or vLLM.
- Quantized-draft acceptance-rate effects: no source found.
- vLLM per-method caveats on batching and exact current version not read (docs unreachable). Date/version of llama.cpp build for the doc above not stated.

## KV cache quantization

### Takeaway
q8_0 KV is widely reported as near-lossless and q4_0 as riskier at long context, but evidence is mostly anecdotal blogs. vLLM's own FP8 KV write-up is the strongest primary source and shows a serious long-context regression that was fixed.

### Cited Findings
- vLLM blog: on a 128k needle-in-a-haystack test on Hopper, FP8 KV accuracy fell from 91% (BF16) to 13% due to imprecise FP32 accumulation in tensor cores; a two-level accumulation fix restored 89%; FP8 "preserves near-baseline accuracy" on validated paths — [vLLM blog 2026-04-22](https://vllm.ai/blog/2026-04-22-fp8-kvcache)
- llama.cpp issue: quantized K or V with flash attention was ~1/3 slower for pp and tg on Metal (Phi-3) — [issue 8918](https://github.com/ggml-org/llama.cpp/issues/8918) (old; may be outdated)
- Build caveat: without `-DGGML_CUDA_FA_ALL_QUANTS=ON`, quantized KV with FA can reportedly fall back to CPU attention — [issue 24485 via search](https://github.com/ggml-org/llama.cpp/issues/24485) (summary only, verify)
- Community guides: Q8 KV roughly halves KV VRAM with tiny perplexity cost; Q4 K acceptable, Q4 V hurts; Ollama FAQ says q8_0 no noticeable impact, q4_0 small-to-medium loss growing with context — [llmhardware.io](https://llmhardware.io/guides/kv-cache-quantization-ollama-llama-cpp), [techplained](https://www.techplained.com/kv-cache-quantization) (unsourced figures; low confidence)
- A DeepSeek V4 Flash test: q8_0 KV left perplexity almost unchanged but changed token choices (512-token test) — [dev.to](https://dev.to/breachprotocol/quantizing-v4-flashs-kv-cache-in-llamacpp-changes-which-tokens-it-picks-38np) (single blog)
- Research: 4-bit KIVI near lossless, 2-bit degrades, especially long-context reasoning; a third-party vLLM-KIVI plugin README reports ~2.59% PPL increase for 4-bit KIVI vs 0.23% for vLLM FP8 (single sample) — search summaries of [KVarN](https://arxiv.org/pdf/2606.03458), [Kitty](https://arxiv.org/pdf/2511.18643) (not read in full)

### Inferences
- Perplexity alone misses decision changes (token choice flips); Sieve's confidence/calibration angle could measure KV-quant effects on calibration, which I found no one reporting.

### Gaps
- No same-model comparison of FP8 vs KIVI vs KVQuant vs llama.cpp q8_0/q4_0 on a task suite.
- Whether llama.cpp currently requires FA for quantized V (docs unclear per one guide). KVQuant evidence minimal.

## Weight quantization formats

### Takeaway
FP8 is the accuracy-safe choice where hardware supports it; at 4-bit, GGUF K/I-quants and AWQ are roughly comparable with GPTQ often trailing, but rankings are model- and task-dependent. NVFP4 evidence is vendor-only.

### Cited Findings
- Jarvis Labs (Jan 2026) Qwen2.5-32B HumanEval: FP16 56.1%, AWQ and GGUF 51.8%, GPTQ 46% — [theaiengineer / aggregator summaries](https://theaiengineer.substack.com/p/quantization-in-practice-gptq-vs) (secondary; exact origin unverified)
- 24GB A10G test: Q4_K_M 84.8% GSM8K strict, IQ4_XS 86.4%, GPTQ 81.8%; FP8 on Qwen3.5-9B no GSM8K loss and 1.76x faster than BF16 — [computingforgeeks](https://computingforgeeks.com/gguf-vs-awq-vs-gptq/) (blog; date claims unverified)
- Q4_K_M is ~4.8 bits/weight; IQ4_XS reportedly beats it on KL divergence and top-token agreement while smaller — same source
- NVIDIA claims <=1% deviation FP8 vs NVFP4 on DeepSeek-R1-0528 across seven benchmarks — [wz-it.com](https://wz-it.com/en/knowledge/ki/llm-quantization/) (vendor claim relayed by blog)

### Inferences
- Report KL divergence/top-1 agreement plus task accuracy, not only perplexity, in Sieve's quantization comparisons.

### Gaps
- No primary-source (paper/official) 2026 comparison of newer formats (MXFP4, NVFP4, etc.) on small 1-8B models; no calibration-set sensitivity data found.

## Quantization and calibration/confidence

### Takeaway
Papers disagree: GPTQ 4-bit lowers confidence in true labels (concentrated on already-uncertain samples), while a 2025 study finds no substantial calibration change. Direction depends on model, method and metric.

### Cited Findings
- Proskurina et al., "When Quantization Affects Confidence of Large Language Models?" (Findings NAACL 2024): GPTQ 4-bit decreases confidence on true labels, varying by model; affects samples where full model was already low-confidence; code released — [arXiv 2405.00632](https://arxiv.org/pdf/2405.00632), code [github](https://github.com/upunaprosk/quantized-lm-confidence) (from search summary)
- "Interpreting the Effects of Quantization on LLMs" (2025): 4-/8-bit show no substantial change in confidence/calibration (Adaptive Calibration Error), 4-bit slightly less confident in most cases — [arXiv 2508.16785](https://arxiv.org/pdf/2508.16785)
- A TMLR (07/2026) paper reports a reliability peak at 4-bit and best reliability/efficiency trade-off, using ECE and Brier — [alphaxiv 2607.10855](https://www.alphaxiv.org/abs/2607.10855.md) (not verified; single preprint-level claim)

### Inferences
- A gap for Sieve: calibration of GGUF K-quants, FP8 and quantized KV is unaddressed in what I found (papers cover GPTQ/bitsandbytes-style); metric choice (ECE vs ACE vs Brier) changes conclusions.

### Gaps
- Full papers not read (arXiv blocked); no study of GGUF/llama.cpp calibration or KV-quant calibration found.

## Benchmarking methodology (vLLM vs llama.cpp)

### Takeaway
No authoritative joint methodology exists; the practical guidance is to compare server-to-server under identical load, match precision, warm up and repeat.

### Cited Findings
- llama-bench measures single-run pp/tg; `vllm bench serve` measures concurrent load with TTFT/TPOT/ITL percentiles; not directly comparable, use llama-server under same concurrency — [markaicode guide](https://markaicode.com/benchmarks/vllm-vs-llamacpp-performance/) (third-party)
- Mismatched precision (Q4_K_M vs fp16/fp8) skews results; use warm-up, median of several runs — same source
- Community port vllm.cpp reports single-stream results only and states concurrent serving is unmeasured — [mudler/vllm.cpp](https://github.com/mudler/vllm.cpp)
- llama.cpp's speculative path has its own SPEED-Bench client and synthetic acceptance mode — [speculative.md](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md)

### Inferences
- Sieve should fix: prompt set, output length, sampling (greedy), concurrency, warm-up, quant-equivalence, and log build flags/commit; MLPerf-style scenarios (single-stream/server) were not researched.

### Gaps
- MLPerf Inference LLM rules, vLLM benchmark script specifics, and reproducibility studies not retrieved.

## Energy measurement

### Takeaway
NVML power sampling and RAPL are usable but have known sampling and scope caveats; validate against a wall meter. CodeCarbon is convenient but coarse.

### Cited Findings
- "Part-time Power Measurements: nvidia-smi's Lack of Attention": nvidia-smi power readings checked against an external meter; proposed practices reduce energy error by up to 35% — [arXiv 2312.02741](https://arxiv.org/html/2312.02741v3) (search summary)
- 2026 HotCarbon paper (secondary report): NVML cumulative energy counters often return repeated values at read intervals <100 ms; advice is to integrate power instead — [university news page](https://mucdai.hm.edu/aktuelles/newsdetail/news_detailseite_528704.de.html), [retit.de](https://www.retit.de/paper-measurement-frequency-energy-readings-intel-rapl-nvidia-nvml-hotcarbon-2026/) (secondary; unverified)
- RAPL correlates with external meters but depends on driver support, update timing and domain errors; reads whole-socket not per-process; DRAM estimates can overestimate — [Price of Prompting](https://arxiv.org/pdf/2407.16893), [review](https://arxiv.org/pdf/2306.08323)
- CodeCarbon polls NVML for GPU and RAPL for CPU, falls back to TDP-based models; one NAS study found its RAM estimate (750 W) far above real (12 W) and total error 10.3% vs wall meter (bounds +8.9 to -3.6%) — [arXiv 2505.15631](https://arxiv.org/pdf/2505.15631) (summary)
- Zeus user thread: ~2.6% energy variation across runs vs 0.85% time variation on a P100, called normal by maintainer — [Zeus discussion 50](https://github.com/ml-energy/zeus/discussions/50) (anecdotal)

### Inferences
- Use idle-baseline subtraction, long enough runs, power-integration with >=10 Hz sampling, and report variance; Zeus internals (NVML energy counter vs sampling) not verified.

### Gaps
- Zeus docs not read; Apple Silicon (powermetrics) energy not researched; wall-meter validation numbers for consumer GPUs absent.

## Consumer hardware envelopes and model fit

### Takeaway
Rule of thumb from low-quality sources: ~7-8B at Q4 on 8 GB, ~9-14B on 12 GB, up to ~20B (gpt-oss 20B) on 16 GB. Sources conflict and are SEO-heavy.

### Cited Findings
- 8GB: ~7-8B at Q4; Gemma 4 12B listed as tight (another source says ~6 GB at Q4, conflicting); 12GB: gemma4:12b, qwen3.5:9b; 16GB: gpt-oss:20b, gemma4:12b, qwen3.5:9b; Phi-4 14B ~9 GB at Q4_K_M; Phi-4 Mini 3.8B and Qwen2.5 3B under 3 GB — [Ollama cheat sheet](https://computingforgeeks.com/ollama-models-cheat-sheet/), [hardware guide](https://www.kunalganglani.com/blog/running-local-llms-2026-hardware-setup-guide.md) (low-quality, conflicting)
- vLLM docs mention Gemma 4 assistant checkpoints handled as Gemma 4 MTP speculators — [vLLM docs via search](https://docs.vllm.ai/en/latest/features/speculative_decoding/)

### Inferences
- Verify exact GGUF file sizes plus KV cache on model cards; context length dominates the fit on 8 GB.

### Gaps
- No data on 24 GB, Apple Silicon unified-memory envelopes, CPU-only tokens/s, SmolLM3, or small Qwen sizes (1B-4B).

## Open bugs and upstream PR opportunities

### Takeaway
Candidates exist but each needs confirmation of current status before work.

### Cited Findings
- Server-vs-standalone speculative throughput gap ([issue 12968](https://github.com/ggml-org/llama.cpp/issues/12968)); shared draft context across slots TODO ([PR 17808](https://github.com/ggml-org/llama.cpp/pull/17808/files)); EAGLE-3 draft support list ([#18039](https://github.com/ggml-org/llama.cpp/issues/18039), cited in docs); DSpark Gemma4 support planned; quantized-KV + FA build fallback ([issue 24485](https://github.com/ggml-org/llama.cpp/issues/24485)); KV q8_0 changing token choices on certain models (blog above)

### Inferences
- Plausible Sieve contributions: acceptance/calibration-aware benchmark tooling, a documented KV-quant quality matrix, a warning when FA falls back to CPU.

### Gaps
- Open/closed status of each issue not verified; no systematic issue triage done.
