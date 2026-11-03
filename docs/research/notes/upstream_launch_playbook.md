# Upstream contribution, publication, and launch playbook for Sieve (as of 2026-11-03)

Coverage caveat: info.arxiv.org, docs.vllm.ai and news.ycombinator.com were not reachable from the research sandbox (network allowlist), so arXiv endorsement rules and Show HN rules rest on search snippets/secondary sources. Only ~15 tool calls were used; first-time-contributor case studies and Reddit/HN star-trajectory data are thin (see Gaps).

## 1. Contribution guidelines, AI policies, merge culture (llama.cpp, vLLM, transformers, lm-eval)

### Takeaway
All three projects now allow AI assistance only with disclosure and full human responsibility; llama.cpp is strictest (undisclosed AI = permanent ban; AI-written PR descriptions/posts prohibited) and transformers asks first-time contributors not to use code agents at all. Sieve's PRs should be small, issue-first, hand-written in prose, with a regression test.

### Cited Findings
- llama.cpp: "AI-generated code is allowed. You are 100% responsible for every line"; "Undisclosed AI usage may result in your account being permanently banned"; must disclose AI use; must be able to explain every line; "strictly prohibited to use AI to write your posts" (bug reports, feature requests, PR descriptions, replies). — [llama.cpp CONTRIBUTING.md](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/CONTRIBUTING.md)
- llama.cpp: new contributors limited to 1 open PR; no trivial fixes (typos/formatting); features must start as an issue; niche features may be redirected to examples/tools; bug-fix PRs need a reproducible issue plus a regression test failing before/passing after; new public API/CLI faces a higher bar; run full CI locally, check perplexity/perf with llama-perplexity/llama-bench; ggml op changes need test-backend-ops on 2+ backends. — [llama.cpp CONTRIBUTING.md](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/CONTRIBUTING.md)
- llama.cpp merge culture: squash-merge, title format `<module> : <title> (#issue)`; a "merge ready" label allows fast merges; PRs may be closed for duplication, policy violations, or poor architectural fit. No CLA/DCO mentioned in the fetched content. — same source
- vLLM: no "pure agent" PRs; submitter must review all lines, validate end-to-end, run tests; disclose AI use in PR description; mark commits with trailers like `Co-authored-by:`. DCO required (`Signed-off-by` on each commit, `git commit -s`). — [vLLM contributing README](https://raw.githubusercontent.com/vllm-project/vllm/main/docs/contributing/README.md)
- vLLM PR rules: title tags like `[Bugfix][Scheduler]`, `[Doc]`, `[Misc]`; include unit+integration tests; update docs in same PR for user-visible changes; >500 LOC needs an RFC issue first; reviewer assigned, status every 2-3 days, ping after 7 days; non-write contributors capped at 6 open PRs; new commits do not auto-run CI (`/ci run` by maintainers); pre-commit required. — same source
- transformers: first-time contributors asked not to use code agents to create issues/PRs; autonomous agents asked not to open PRs/issues; agent-looking PRs may be closed without review; AI-assisted work allowed if coordinated, scoped, verified, disclosed; discuss before opening a PR; minimize diff; avoid busywork PRs; run `make fix-repo`, `utils/tests_fetcher.py`, CI green before review request; "Good First Issue" and "Good Second Issue" labels. — [transformers CONTRIBUTING.md](https://raw.githubusercontent.com/huggingface/transformers/main/CONTRIBUTING.md)
- Tension noted in transformers doc: typo fixes are discouraged as busywork but the Docs section welcomes them (fetch summary flagged this; verify with maintainers).

### Inferences
- A solo builder with agent-assisted workflow must write PR bodies/issue text by hand for llama.cpp (explicit ban) and avoid agent-authored PRs to transformers as a first-time contributor; for vLLM disclose and add trailers.
- Fastest path to a merge: llama.cpp (no CLA, 1-PR limit, small focused fixes with test), or lm-eval (task/metric additions). vLLM and transformers are higher-volume, slower to review.

### Gaps
- AGENTS.md / CLAUDE.md contents of each repo not read. HF lm-eval-specific CONTRIBUTING and CLA status not fetched. Typical time-to-merge statistics not obtained.

## 2. Concrete open issues relevant to Sieve (queried live via GitHub API on 2026-11-03)

### Takeaway
There are live, recently-updated logprobs bugs in llama-server and entropy-adjacent feature requests; lm-eval has an open calibration-metric request. Good-first-issue pools exist but few map directly to Sieve; the best fits are logprobs bug fixes in llama-server and an RMS calibration error metric for lm-eval.

### Cited Findings
llama.cpp (ggml-org/llama.cpp):
- 15 open "good first issue" issues, e.g. [#17611 wrong default threads in llama-bench](https://github.com/ggml-org/llama.cpp/issues/17611) (good first issue, performance; updated 2026-11-02), [#10732 json_schema structured output in server](https://github.com/ggml-org/llama.cpp/issues/10732), [#13523 tutorials list](https://github.com/ggml-org/llama.cpp/issues/13523) (docs-type, help wanted).
- Logprobs: [#31451 llama-server silently drops recorded logprobs/token IDs for multi-token UTF-8 characters](https://github.com/ggml-org/llama.cpp/issues/31451) (2026-10-31, bug-unconfirmed); [#29456 llama-server does not cap logprobs/top_logprobs (n_probs)](https://github.com/ggml-org/llama.cpp/issues/29456) (stale label); [#31206 /v1/completions logprobs nondeterministic](https://github.com/ggml-org/llama.cpp/issues/31206).
- Entropy: [#29875 early-exit speculative drafting via Shannon entropy](https://github.com/ggml-org/llama.cpp/issues/29875), [#27821 entropy-gated speculative decoding (research)](https://github.com/ggml-org/llama.cpp/issues/27821), [#29022 fast tool gating via prefill logit slicing](https://github.com/ggml-org/llama.cpp/issues/29022).
- Metrics: [#30123 server: flush metrics more frequently](https://github.com/ggml-org/llama.cpp/issues/30123) (only 2 matching open issues for prometheus+metrics; no existing request for entropy/confidence metric found).
vLLM (vllm-project/vllm):
- 17 open good-first-issue items, mostly core refactors/model support (e.g. [#37753 Unify MoE Oracles](https://github.com/vllm-project/vllm/issues/37753), [#39479 torch.compile config hashing follow-ups](https://github.com/vllm-project/vllm/issues/39479), [#32335 extract KV-cache update from attention backends](https://github.com/vllm-project/vllm/issues/32335)). Not Sieve-aligned; high-skill.
- 224 open issues mention logprobs, e.g. [#64584 decode logprobs drift after prefix-cache hit](https://github.com/vllm-project/vllm/issues/64584), [#27433 batch invariance](https://github.com/vllm-project/vllm/issues/27433); [#65250 extract_hidden_states throughput loss](https://github.com/vllm-project/vllm/issues/65250). Mostly deep numerics bugs.
lm-evaluation-harness (EleutherAI/lm-evaluation-harness, note: not huggingface/):
- [#2189 Adding RMS Calibration Error as a Metric](https://github.com/EleutherAI/lm-evaluation-harness/issues/2189) (updated 2026-10-26, unlabeled) — directly relevant to Sieve calibration utilities.
- [#3831 paired significance test; unpaired z-test applied to paired data](https://github.com/EleutherAI/lm-evaluation-harness/issues/3831) (2026-07-27); [#1613 better test coverage for models](https://github.com/EleutherAI/lm-evaluation-harness/issues/1613) and [#1883 add regression testing](https://github.com/EleutherAI/lm-evaluation-harness/issues/1883) (good first issue, help wanted); 15 good-first-issue total.
transformers: label query for "Good First Issue" returned 0 via search (label naming/quoting uncertain); not verified.

### Inferences
- Highest-probability merge candidates for a Sieve-aligned contributor: (1) fix #31451 or #29456 in llama-server (small C++ server change + test; fits "bug-fix + regression test" rule); (2) lm-eval #2189 calibration metric (Python, self-contained, reusable utility); (3) docs/tutorial entries (#13523) – low value, may be seen as low effort.
- A server-side entropy/confidence metric would be a *new* public API: llama.cpp says higher bar, so open an issue first and expect pushback.

### Gaps
- Did not check KV-cache-quant docs gaps, benchmark script bugs beyond #17611, vLLM Prometheus metric requests, or transformers issues. No recent-PR merge latency measured.

## 3. First-time contributors with reusable-utility PRs merged

### Takeaway
Not researched with evidence; no case studies were retrieved.

### Cited Findings
- None collected.

### Inferences
- Policy documents above imply the pattern that gets merged: issue-first, tiny diff, reproducible script, tests, disclosure.

### Gaps
- No first-time-contributor examples found; suggest sampling merged PRs from authors with 0 prior commits via `gh search prs --merged` in a follow-up.

## 4. Venues, deadlines, arXiv endorsement

### Takeaway
Most 2026 uncertainty/hallucination workshops have closed; the best realistic targets are ICLR 2027 workshops (suggested contribution deadline 1 Feb 2027, notifications by 26 Feb 2027) and later ICML/ACL/COLM 2027 workshop calls (not yet posted). MLSys 2027 main deadline (30 Oct 2026) has just passed.

### Cited Findings
- ICLR 2027: workshop organizer application deadline 9 Oct 2026 (passed); suggested workshop contribution deadline 1 Feb 2027; mandatory acceptance notifications 26 Feb 2027; workshops reported 29-30 Apr (sources conflict on venue: San Francisco/Moscone vs. Brazil from an aggregator). — [ICLR 2027 call for workshops / search results](https://iclr.cc/Conferences/2027/CallForWorkshops); [dates](https://iclr.cc/Conferences/2027/Dates)
- MLSys 2027: paper deadline 30 Oct 2026 2:00 PM PDT (one aggregator says 20:00 UTC; conflict); conference 21-25 Jun 2027, Bellevue WA; notifications 28 Feb 2027. — [mlsys.org](https://mlsys.org), [mldeadlines](https://mldeadlines.com/conference/mlsys-2027/)
- Passed: UncertaiNLP @ EMNLP 2026 (deadline 7 Aug, event 29 Oct, Budapest) — [ACL portal](https://www.aclweb.org/portal/content/3rd-workshop-uncertainty-aware-nlp-emnlp-2026); Industrial NLP @ INLG 2026 (1 Aug) — [CFP](https://groups.google.com/g/ML-news/c/kqUJa7tB8gM); UnCV @ ECCV 2026 (30 Jun) — [CFP](https://groups.google.com/g/ML-news/c/-7Pl6M1TJ7A); EIML @ ICML 2026 (20 Apr).
- arXiv: first-time submitters to a category need an endorsement from an established author in that category; institutional email can help; one positive endorsement suffices; endorsement is per-category. — [arXiv endorsement help (via search snippet)](https://info.arxiv.org/help/endorsement.html), [endorsement math](https://info.arxiv.org/help/endorsement-math.html)
- Many people seek endorsers on HF forums (e.g. [HF forum thread](https://discuss.huggingface.co/t/looking-for-arxiv-endorsement-cs-lg-first-submission/180634)); posting codes publicly is risky.

### Inferences
- With ~14 weeks (to ~mid-Feb 2027) the feasible workshop target is an ICLR 2027 workshop around 1 Feb 2027 (each workshop sets its own deadline; verify) or a short/non-archival track. OpenReview submission requires complete profiles.
- Endorsement route for a recent graduate: use university email/alumni advisor, co-author with an established author, or ask authors of papers you cite (unverified practice).

### Gaps
- Specific ICLR 2027 workshop list (reliability/uncertainty/efficient inference) not retrieved; ICML/ACL/COLM/NeurIPS 2027 workshop CFPs not yet posted; TMLR rolling policy not checked; official arXiv text for automatic endorsement criteria not read.

## 5. Launch playbook

### Takeaway
Evidence is mostly self-reported postmortems; the only controlled analysis found says Hacker News posting timing matters more than the "Show HN" tag. Treat 300 stars as plausible only with a real demo, benchmarks, and multi-channel launch.

### Cited Findings
- 2025 arXiv study of HN launches: posting time is a key factor (optimal hours worth hundreds of extra stars); the Show HN tag shows no statistical advantage after controls. — [arXiv 2511.04453](https://arxiv.org/abs/2511.04453v1)
- Infisical: HN was a major growth channel; 3,000 stars in 2 months. — [Infisical blog](https://infisical.com/blog/how-we-got-3000-github-stars-in-2-months.md)
- Postmortem: 160 stars over 1,000+ days then 3,000 in ~a month after Product Hunt → GitHub Trending. — [ExpertBeacon](https://expertbeacon.com/how-i-took-an-open-source-project-from-0-to-3000-stars-a-data-driven-post-mortem/)
- Other reports: [Freek 0→1,000 stars in 3 months](https://freek.dev/704-how-i-got-from-0-to-1-000-stars-on-github-in-three-months-with-my-open-source-side-project); [Rybbit 5k stars](https://rybbit.com/blog/5k-stars).
- r/LocalLLaMA (third-party wiki mirror): ~1/10th self-promotion guideline, disclose affiliation, no engagement farming, karma minimums (10 site / 5 subreddit). — [wiki mirror](https://r.datuan.dev/r/LocalLLaMA/wiki/index) (verify live rules)
- Buying stars is detected and penalized (marketing guide, low reliability). — [oss launch skill listing](https://skillselum.com/skills/gingiris-1031/gingiris-skills/gingiris-opensource)

### Inferences
- Plan: README with one-GIF demo + benchmark table + 3-line install; launch Tues-Thu morning US Eastern (unverified); stay in comments all day; r/LocalLLaMA post framed as a finding (e.g., entropy/confidence results on llama.cpp quant levels), not an ad; build a post-history on the subreddit first given karma/1-in-10 norms.
- Failure modes (inferred, not sourced): no demo, claims without baselines, posting on a cold account, launching before install works.

### Gaps
- Show HN official rules not fetched; no data on r/LocalLLaMA post outcomes; no median-star-trajectory datasets for LLM tooling found.

## 6. Realistic probability assessment (judgment, not sourced)

### Takeaway
Estimates for a solo builder in 14 weeks, based on policies and venue timing above; treat as unvalidated priors.

### Inferences
- >=1 merged PR in llama.cpp/lm-eval: ~60-70% if started this month with a small bug fix or the calibration metric; effort ~20-40 hrs incl. review cycles. 3 merged PRs: ~15%. vLLM/transformers merge: lower (~30%) because of volume and first-time agent restrictions.
- Workshop paper accepted: submission feasible by 1 Feb 2027 (ICLR workshops); acceptance rate for workshops often 50-70% (unsourced), so ~35-45% overall given need for solid experiments and endorsement; effort 80-120 hrs.
- 300 stars: ~25-35% without a breakout post; HN front page or a strong r/LocalLLaMA hit changes it to likely, but hit probability per attempt is low (~10-20%, unsourced). Several shots (HN, Reddit, upstream-issue visibility) raise odds.

### Gaps
- No base-rate data from sources for any of these probabilities.
