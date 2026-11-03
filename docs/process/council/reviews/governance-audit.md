# Governance and security audit for public release (D-018)

Seat: AI Governance and Security Specialist. Date: 2026-11-03. Scope: working tree at HEAD `eb42ec7` (branch `claude/great-babbage-a6srla`, 20 commits; `main` = `origin/main` = `3979ca0`, the initial commit only) plus all git history. Read-only on `src/` and `tests/`; no commits made. Tests/linters were NOT executed by this audit; no claim here depends on them.

Status vocabulary: confirmed defect / risk / improvement.

## Verdict
No secrets or personal data found in tree or history. The repo is NOT yet presentable as a public recruiter-facing repo: the README is stale and contradicts the code, SECURITY.md is stale, history is 20 commits, 19 authored by "Claude", including 7 `wip:` commits, unverified AI-generated research notes sit at top level, the 3B model in the pre-registered config is under a non-commercial licence that no document in the README mentions, and there is no AI-use disclosure. All fixable without architectural change. History rewriting needs owner approval (see P1-3).

## 1. Secrets and personal data

Commands run (all against `git rev-list --all`, 19 commits at scan time; HEAD `eb42ec7`, a later wip commit touching only `src/sieve/eval/prompts.py`, was re-scanned for token patterns separately: clean):
`git grep -n -I -E '<pattern>' $(git rev-list --all)` for: `KGAT_`, `ghp_`, `github_pat`, `hf_[A-Za-z0-9]{20,}`, `sk-[A-Za-z0-9]{20,}`, `AKIA...`, `BEGIN (RSA|OPENSSH|PRIVATE)`, `xox[bp]-`, `api[_-]?key`, `password`, `secret`, `token`; emails; phone-like numbers; `/home/|/Users/|C:\\|/root/|/tmp/`; `ksainiketh|sainiketh|niketh|kosaraju`; `kaggle\.com`. Also `git log --all -- kaggle_out '*.runlog' '*.gguf'` (empty), largest-blob listing (largest 323 KB, `results/mock/index.html`; no model files), and grep of `results/evidence/*.json` for env/URL/IP (only `127.0.0.1`).

| Finding | Evidence | Status | Severity |
|---|---|---|---|
| No tokens/keys/KGAT_/HF/GitHub/AWS secrets in any commit | zero hits for all token patterns; `key`/`secret` hits are code identifiers (`app.py:85`, `config.py:103`) and docs | no defect | - |
| Email addresses are test fixtures only | `tests/server/test_load_security.py:424,514` (`alice@example.com`, fake `sk-abcdefgh12345678`), `tests/observe/test_observe.py:73` | no defect | - |
| No phone numbers (two regex hits are a DOI and a test card number) | `confidence_signals_calibration.md:32`, `test_observe.py:82` | no defect | - |
| No private absolute paths. Only `/tmp/sieve-mock` (docs) and `/kaggle/working/...` (evidence JSON, public Kaggle image path) | `docs/reproduce.md:8` | no defect | - |
| `kaggle_out/` (run logs, kernel outputs) is git-ignored and was never committed | `.gitignore:231`; `git log --all -- kaggle_out` empty | no defect | - |
| No Kaggle output URLs (kaggleusercontent.com) or dataset URLs in history; only the API base | `kaggle/kaggle_run.sh:8` | no defect | - |
| Owner identity visible: full name (LICENSE:3, CITATION.cff:8-9, CHARTER.md:58), GitHub login in CITATION.cff:11 and initial commit email `64412849+sainikethkosaraju@users.noreply.github.com` (GitHub noreply, good) | `git show 3979ca0` | no defect (intended for a public portfolio) | - |
| Kaggle username `ksainiketh` hard-coded as default | `kaggle/kaggle_run.sh:7` (`${KAGGLE_USER:-ksainiketh}`), `DECISIONS.md:8` (`ksainiketh/sieve-smoke`), `results/evidence/t4_llamacpp_smoke_2026-11-03.json:678` (`"kernel": "ksainiketh/sieve-llamacpp-t4-smoke"`) | improvement | Low |
| Kernels were pushed `isPrivate: True` (`kaggle_run.sh:15`); no other Kaggle datasets of the owner are referenced anywhere | grep | no defect | - |
| Local-only untracked files are ignored: `.venv`, `.hypothesis`, `kaggle/__pycache__` (ignored by `.gitignore:2`). `src/sieve/eval/providers/` is untracked (new work), not a leak | `git status` | info | - |

Kaggle username assessment: it is not a secret and is derivable from the public GitHub identity, so there is no privacy harm. Parametrise anyway for hygiene and reuse: make `kaggle_run.sh` require `KAGGLE_USER` (`: "${KAGGLE_USER:?set KAGGLE_USER}"`), and in `DECISIONS.md` and the evidence JSON replace with `<kaggle-user>/sieve-...` only if the owner prefers anonymity; the JSON is raw evidence, so I recommend leaving the provenance field as is and noting that kernels are private and unresolvable. Needs a one-line owner choice (Low).

Residual caveat: scanning is regex-based. Run `gitleaks detect --log-opts="--all"` (or the GitHub MCP `run_secret_scanning`) once before flipping to promoted/public; I did not run gitleaks (not installed).

## 2. Licences

### 2.1 Repository and dependencies
- `LICENSE`: MIT, "Copyright (c) 2026 Sai Niketh Kosaraju". `pyproject.toml:9` `license = { file = "LICENSE" }`, CITATION.cff `license: MIT`. Consistent. Improvement: use SPDX `license = "MIT"` + `license-files`, add trove classifiers.
- Core runtime deps: numpy (BSD-3 + 0BSD/MIT/Zlib/CC0), scipy (BSD-3). Server extra: fastapi MIT, uvicorn BSD-3, httpx BSD-3, prometheus-client Apache-2.0/BSD-2, pydantic MIT, pydantic-settings MIT, pyyaml MIT. Eval extra: datasets Apache-2.0, matplotlib (PSF-based), pandas BSD-3. Dev: pytest MIT, pytest-cov MIT, hypothesis **MPL-2.0** (dev-only, not redistributed, no issue), ruff MIT, mypy MIT, types-PyYAML Apache-2.0. Source: `importlib.metadata` in the local `.venv`. All MIT-compatible. No copyleft in runtime path.
- Docker compose pulls `grafana/grafana:11.3.0` (AGPL-3.0) and `prom/prometheus:v2.55.1` (Apache-2.0) as separate containers, not redistributed in this repo. Provisioned dashboard JSON is original. Improvement: one line in docs/server.md noting Grafana's AGPL status.
- llama.cpp (built on Kaggle, not vendored): MIT (`raw.githubusercontent.com/ggml-org/llama.cpp/master/LICENSE`, fetched). No llama.cpp code is copied into the repo (only the HTTP API is used).

### 2.2 Models (HF API `https://huggingface.co/api/models/<id>`, fetched today)
| Model | Licence per card | Used where |
|---|---|---|
| Qwen/Qwen2.5-7B-Instruct and `-GGUF` | Apache-2.0 | large tier (`PREREGISTRATION.md`, `kaggle/llamacpp_t4_smoke.py:19` uses `bartowski/Qwen2.5-7B-Instruct-GGUF`, card Apache-2.0, third-party quantisation of Qwen) |
| Qwen/Qwen2.5-1.5B-Instruct and `-GGUF` | Apache-2.0 | smoke test, fallback config |
| **Qwen/Qwen2.5-3B-Instruct and `-GGUF`** | **`other` / `qwen-research`** | **frozen small tier S in `PREREGISTRATION.md:16` and `deploy/sieve.example.yaml`** |

Verified from the licence text (`.../Qwen2.5-3B-Instruct/raw/main/LICENSE`): "Non-Commercial" means "for research or evaluation purposes only" (line 16); commercial use requires a licence request (line 20); redistribution requires retaining the notice "Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved." in a Notice file (line 26) and marking modified files (line 25); using outputs to build a distributed AI model requires "Built with Qwen"/"Improved using Qwen" (line 31).
- Confirmed defect (documentation): `docs/reproduce.md:28-29` mentions the 3B licence, but the README, `deploy/sieve.example.yaml` and the Docker quick start default to 3B with no warning. Anyone following the deploy example commercially would breach the licence. Severity Medium. Fix: README "Models and licences" table; comment in example YAML; NOTICE.
- The repo does not redistribute weights, GGUFs or generations (`.dockerignore` excludes `*.gguf`, `*.jsonl`; no model files in history). Results containing only aggregate metrics are not "Materials". Keep it that way: never commit generation caches or model outputs under results/ (reproduce.md already says "never the generation cache"; make it a CI check, below).
- Not done: I did not hash-verify GGUF files (not present).

### 2.3 Datasets (HF API, fetched today; pinned SHAs in `src/sieve/eval/datasets.py` match the Hub `sha` values returned: TriviaQA `0f7faf33...`, NQ-open `5dd9790a...`, SimpleQA Verified `0dc97e0d...`, MMLU-Redux-2.0 `372ea425...`)
| Dataset | Card licence | Implication |
|---|---|---|
| TriviaQA (`mandarjoshi/trivia_qa`) | `unknown` (card lists none) | Code already says "check upstream terms before redistribution" (`datasets.py:105`). Do not redistribute items or per-item outputs. Cite Joshi et al. 2017 |
| NQ-open (`google-research-datasets/nq_open`) | CC-BY-SA-3.0 | Share-alike and attribution if items (questions/answers) are redistributed. Aggregate metrics are fine; per-item tables are not |
| SimpleQA Verified (`google/simpleqa-verified`) | MIT | Attribution to Google DeepMind (arXiv 2509.07968) |
| MMLU-Redux 2.0 (`edinburgh-dawg/mmlu-redux-2.0`, also 1.0 is CC-BY-4.0) | CC-BY-4.0 | Attribution required. Underlying MMLU items carry their own provenance (not resolved here) |
| `openai/simpleqa` (original) | not reachable via API (gated) | Not used. `basicv8vc/SimpleQA` mirror is MIT-tagged but unofficial; do not use |

- Redistribution rule for `results/`: aggregate statistics, figures and tables are permitted; item text, gold answers, model generations and the item-level `splits` are NOT (NQ-open SA, TriviaQA unknown). Currently compliant: `results/mock` is synthetic (fake backend), `results/evidence` is a smoke test on one hard-coded prompt ("capital of France"). Check `results/schema/results.schema.json` stays free of question/answer text (my grep of the schema for question/prompt/text/gold/answer returned nothing).
- The SimpleQA judge prompt is "own wording, not the official autorater prompt" (`docs/evaluation.md:89`): good, avoids copying OpenAI's text.

### 2.4 Copied text in docs/research notes
- Notes are paraphrases with links; no block quotes (`grep -c '^> '` = 0 in all seven notes). I saw no verbatim passages beyond titles and short phrases. Not a copyright concern. Improvement: one report cites a gitlink.org.cn mirror of llama.cpp (`reports/...gap analysis.md` line 21, `cdn04132025.gitlink.org.cn`); replace with github.com/ggml-org/llama.cpp or drop (looks unprofessional and is an unofficial mirror).
- Attribution of the Qwen notice and datasets: see NOTICE below.

### 2.5 NOTICE / attribution file needed
Add `NOTICE` (or `docs/THIRD_PARTY.md`) containing: Qwen RESEARCH LICENSE notice string exactly as in LICENSE line 26 (for 3B-based docs/configs), "Qwen2.5 is Apache-2.0" for 7B/1.5B, dataset attributions (NQ-open CC-BY-SA-3.0 Google; MMLU-Redux CC-BY-4.0 Gema et al.; SimpleQA Verified MIT, Haas et al.; TriviaQA Joshi et al.), llama.cpp MIT (ggml authors), Grafana/Prometheus runtime images. Medium/Low.

## 3. Honesty and claims

README leads with Status (good: "There are no performance, accuracy, energy or latency results yet") but is stale and wrong in the other direction.

| # | Location | Problem | Status | Sev |
|---|---|---|---|---|
| 3.1 | `README.md:11-26` | Says "Early development (Phase A)... Not built yet: backends, the server, audit/metrics, the evaluation harness". All of these exist in `src/sieve/{backends,server,observe,eval}` and `docs/`. A recruiter will see the README contradict the tree. | confirmed defect | High |
| 3.2 | `README.md` | No "proven vs not proven" section as D-018 requires. Real evidence that exists: a single-run T4 smoke test (`results/evidence/t4_llamacpp_smoke_2026-11-03.json`): llama.cpp `fac3a894` builds with CUDA 13 `-DGGML_CUDA_NO_VMM=ON`, logprobs returned, KV q8_0 runs. Nothing about accuracy/calibration/guarantee on real models. | confirmed defect | High |
| 3.3 | `README.md:47` | `make reproduce-mock` described as "simulation of the risk guarantee, ~40 s". The Makefile target also runs `benchmarks/reproduce_mock.sh` which writes `results/mock/` (and overwrites the committed copy). | improvement | Low |
| 3.4 | `results/mock/**` | Synthetic numbers can be mistaken for real: `index.html` and `hypotheses.json` carry the SYNTHETIC banner, but the CSVs and `tables.md` do NOT (`grep -il synthetic results/mock/tables/*` empty) and show model-like cells such as `L_q4km, nq_open, accuracy 0.8433` and `hypotheses.csv: H1 validity SUPPORTED`. The figures' footers say SYNTHETIC per `report.py:4`, but PNGs travel alone. Also provenance `commit 7733433...+dirty` and the directory name `mock` is the only cue. | confirmed defect | High |
| 3.5 | `docs/PLAN.md:24-27` | "Flagship result (the headline)... Target sentence: 'Matches the large model's accuracy at X% of the energy'" and "first measurement of quantization's effect": aspirational, partly refuted by own novelty notes (weight-quant "partly covered"). Labelled "to be earned", but "first" language is risky. | risk | Medium |
| 3.6 | `docs/PLAN.md:66` | Heading "Novelty status (verified 2026-11-03)" while top line (3) says "Novelty claims are unverified until the papers are read" and notes are snippet-based. Self-contradictory. I spot-checked arXiv ID resolution only: 8 of 72 distinct IDs return HTTP 200 (`curl https://arxiv.org/abs/<id>`); not all checked, and resolution does not verify what the note says the paper claims. | risk | Medium |
| 3.7 | `docs/PLAN.md:18` | "auditability is what enterprises and Palantir-style buyers care about": names a company; job-hunting subtext on the front. Remove. | confirmed defect (hygiene) | Medium |
| 3.8 | `docs/PLAN.md:38-50` | Weeks/staging, "Open decisions 1-3" (hardware, model pair, name check) are now decided or stale; "Name check for 'Sieve'": `pypi.org/pypi/sieve` exists (unrelated XML tool); `sieve-llm` is free on PyPI (404). Fine, but the PLAN still lists it open. | improvement | Low |
| 3.9 | `docs/server.md:96` | "the cap bounds worst-case cost (2x) under adversarial hard prompts without breaking the guarantee": a design argument, not measured; presented as fact. Also "fails closed" language throughout is design intent backed by tests I did not run. | risk | Low |
| 3.10 | `docs/council/DECISIONS.md:5` D-005 | "Novelty reframed ... KV-cache-quantization vs gate is the lead angle", owner-approved column "pending owner awareness"; claim "open" rests on weak evidence per the notes. Keep out of README. | risk | Low |
| 3.11 | `SECURITY.md:22-23` | "Server-side concerns... are release gates for the server component, which is not yet implemented." Stale: server exists. | confirmed defect | Medium |
| 3.12 | `CITATION.cff:14-15` | "No empirical results are claimed yet": accurate in substance, but `date-released: 2026-11-03` for a `0.1.0.dev0` is a release date for something unreleased; cite-me files on an unreleased dev repo invite citing a non-existent paper. | improvement | Low |
| 3.13 | `docs/council/DECISIONS.md:15-20` | Cites "Red-Team's 12 kill tests are release gates" without any status; I saw no record of which pass. | risk | Low |
| 3.14 | `reports/...gap analysis.md:3` | Contains planning ambitions ("300+ stars in about 14 weeks, solo", "over-ambitious") and "Do not claim" content. Internal candour, not suitable as front-page report. Legend `[Fetched]/[Snippet]/[Inference]/[Memory]` is good. | risk | Medium |
| 3.15 | `README.md:29-39` guarantee statement | Accurate, appropriately hedged (not per-query, not under shift...). Keep verbatim, but note the guarantee is currently validated only by simulation, not on real models. | ok | - |

Positive: PREREGISTRATION is frozen with a hash and a Deviations log; the harness refuses to proceed if it changes.

## 4. Privacy defaults of the server

Evidence read: `src/sieve/observe/audit.py`, `redact.py`, `src/sieve/server/config.py`, `docs/observe.md`, `docs/server.md`, `deploy/*`.
- Prompts/responses are NOT logged by default: `config.py:79` `log_content: bool = False`; `AuditRecord.to_json` drops `prompt`/`response` when None (`audit.py:60-62`); `apply_content_policy` writes content only if `log_content` (`audit.py:84-95`). `redact: bool = True` (`config.py:80`) applies email, key/bearer-like and 13-19 digit redactors when content logging is enabled. Good.
- Users: client-supplied `user` stored only as an HMAC-SHA256 prefix; salt is per-process unless `user_hash_salt` set (`audit.py:70-80`, `config.py:82`). With the random per-process salt the hash is unlinkable across restarts; with a fixed salt it is pseudonymous (personal data under GDPR Art. 4(5)).
- Audit file created with mode 0600 (`audit.py` `os.open(..., 0o600)`). Metric labels are bounded (`observe/metrics.py:6`). Unhandled exceptions log only the type (`docs/server.md:98`; `app.py:210`).
- Binding: default `host = 127.0.0.1` (`config.py:100`); non-loopback requires API keys (`config.py:137-138`), min key length 8 (`config.py:140`), `metrics_public=False` by default. Compose publishes ports on 127.0.0.1 only; `read_only`, `cap_drop: ALL`, `no-new-privileges` (`deploy/docker-compose.yml`).
- Gaps (improvements):
  1. No documented retention: audit JSONL has no rotation or retention (`docs/server.md:132` "does not rotate files"). Add retention guidance (e.g. 30 days logrotate) and a "do not enable log_content on personal data without legal basis" warning.
  2. Redaction is regex-only (emails/keys/long digit strings); names, addresses, phone numbers and short IDs pass through when `log_content` is enabled. State this as best effort (docs/observe.md:17 already disclaims "a privacy guarantee"; keep).
  3. No GDPR notes: add a short `docs/privacy.md`: controller/processor roles for operators, what is logged by default (metadata, signal values, hashed user, prompt length), pseudonymisation caveat, retention, right-to-erasure approach (delete audit lines by `request_id`/`user_hash` only if fixed salt), no telemetry sent by Sieve (verify before claiming: grep found no outbound calls except configured backends; `trust_env=False`).
  4. `prompt_chars` and latency are logged: low-sensitivity metadata, mention it.
  5. Uvicorn access logs print request paths (`docs/server.md:133`); fine, documented.
  6. Eval caches (`*.jsonl`) contain dataset questions and model generations; keep git-ignored and out of Docker (already in `.dockerignore`). Add `*.jsonl` and `**/cache/` to `.gitignore` explicitly (currently only `.dockerignore` lists `*.jsonl`).
Severity of section: Low (defaults are strong).

## 5. Supply chain and CI

| Item | Evidence | Status | Sev |
|---|---|---|---|
| Dependencies unpinned, no lockfile, no upper bounds | `pyproject.toml:13-26`: `fastapi`, `uvicorn`, `httpx`, `pydantic`, `datasets`, ... with no version at all; only `numpy>=1.26`, `scipy>=1.11` | improvement (risk for reproducibility/supply chain) | Medium |
| CI installs from PyPI with floating versions | `.github/workflows/ci.yml` `pip install -e ".[dev]"` | risk | Medium |
| GitHub Actions pinned by mutable tag, not SHA | `actions/checkout@v4`, `actions/setup-python@v5` (`ci.yml`) | improvement | Low |
| No `permissions:` block (default token may be read-write on older repos); triggers `push` + `pull_request` | `ci.yml` 21 lines | improvement | Medium |
| No Dependabot/Renovate, no `pip-audit`, no secret scanning config, no CODEOWNERS | `.github/` contains only `ci.yml` | improvement | Medium |
| No typosquat-prone or new/obscure dependencies: all well-known; `types-PyYAML` is the legitimate typeshed stub | `importlib.metadata` check | no defect | - |
| Install scripts: none. No `setup.py`; build backend hatchling (`pyproject.toml:1-3`); no `postinstall`; Makefile/shell scripts reviewed (`Makefile`, `benchmarks/reproduce_mock.sh`, `kaggle/kaggle_run.sh`) execute nothing remote | read | no defect | - |
| `kaggle/llamacpp_t4_smoke.py` does `git clone` of llama.cpp and downloads GGUFs from HF at runtime without hash check (commit recorded in output) | `kaggle/llamacpp_t4_smoke.py:18-19` | risk | Low |
| Dockerfile: multi-stage, slim base, non-root uid 10001, HEALTHCHECK, no build tools in final image, `--no-cache-dir`: good. Gaps: base image tag not digest-pinned (`python:3.12-slim`), `pip install build` unpinned, runtime deps unpinned (follows pyproject), no `.dockerignore` for `.venv`... (it does exclude `.venv`) | `Dockerfile:3,8,17` | improvement | Low |
| Compose images pinned to exact versions (prometheus v2.55.1, grafana 11.3.0), Grafana sign-up/anonymous disabled, admin password required via env | `deploy/docker-compose.yml` | no defect | - |
| `SECURITY.md`: has private reporting channel and 7-day ack, scope notes. Stale server line (3.11); missing supported-versions table, disclosure policy, mention of the API-key, SSRF allow-list, and `log_content` risks; ensure GitHub "private vulnerability reporting" is actually enabled in repo settings (cannot verify from here; if disabled the policy is a dead end) | `SECURITY.md` | confirmed defect (stale) | Medium |
| `CITATION.cff`: valid cff 1.2.0 fields; repository URL matches `git remote` (`sainikethkosaraju/Sieve`); no ORCID (optional); see 3.12 | `CITATION.cff` | improvement | Low |
| `.claude/agents/*.md` committed | 15 files | info (fine, but see section 7) | Low |

Proposed CI hardening (all new files outside src/tests): add `permissions: contents: read`, SHA-pin actions, `pip install` with `--require-hashes` or a `constraints.txt` generated by `pip-compile`, a `pip-audit` job, a "no large/forbidden files" job rejecting `*.gguf`, `*.jsonl`, `kaggle_out/` and any file >1 MB, and a gitleaks step.

## 6. AI-use disclosure and upstream policy

Facts in repo: all 19 post-initial commits are authored and committed by `Claude <noreply@anthropic.com>` (`git log --format=%an | uniq -c`: 19 Claude, 1 owner) and carry `Co-Authored-By: Claude Sonnet 5.5` and a `Claude-Session:` link trailer. The README contains no AI disclosure. The only human-authored commit is GitHub's "Initial commit". So the public history currently credits essentially all work to the assistant while no README sentence explains the human role. That is both an honesty gap (disclosure is a stated D-018 gate and an upstream requirement) and a recruiter-legibility gap (owner's contribution invisible).

Upstream policy (from `research_notes/.../upstream_launch_playbook.md`, which is snippet-level; confirm against each project's current CONTRIBUTING/AI policy before filing): llama.cpp prohibits AI-written PR descriptions/comments and requires disclosure of AI-written code; vLLM requires DCO sign-off and disclosure; transformers asks first-time contributors not to use code agents. Consequences: (a) Claude must not draft upstream PR text; the owner writes and submits (already stated in CHARTER.md:71-72 and PLAN.md); (b) `Signed-off-by` must be added only by the human who certifies the DCO; Claude must never add it.

Draft honest README wording (the owner should edit it into their own voice; a human-written sentence is better for the llama.cpp context):

> ## How this was built
> Sieve was designed and directed by Sai Niketh Kosaraju. Most of the code, tests and documents in this repository were drafted with Claude (Anthropic) coding agents, run in a structured review process (an AI "council" with separate author and reviewer roles; see `docs/process/`). I set the scope, made the decisions recorded in `docs/process/DECISIONS.md`, and [am responsible for / have reviewed] the published results. Research notes under `docs/research/` were produced by AI-assisted web search and are unverified (snippet-level). Commits made by the agent carry `Co-Authored-By: Claude` trailers. Nothing in this repository has been submitted upstream; any future pull requests to third-party projects (llama.cpp, vLLM, ...) will be written and submitted by me personally and will follow each project's AI-assistance policy.

(The bracket must be filled by the owner truthfully; do not assert review that has not happened.)

Draft commit trailer for assistant-authored commits (already in use; keep):
```
Co-Authored-By: Claude <noreply@anthropic.com>
Assisted-by: Claude Code (model: claude-sonnet-5-5)   # optional, explicit
```
For upstream PRs by the owner (human writes message; disclosure line only where the project requires it, in the owner's words), e.g.: `AI assistance: Claude was used to draft part of the code in this PR; I reviewed, tested and take responsibility for it. The PR description is my own.` plus `Signed-off-by: <owner real name> <email>` added by the owner only for vLLM/DCO projects. For llama.cpp, do not use AI for the text at all.

Authorship check (D-018): `git log --format='%an <%ae>'` shows `Claude <noreply@anthropic.com>` for 19/20 commits. Owner decision needed on whether to keep as is (honest, recommended) or set the owner as author with the Co-Authored-By trailer retained (also honest, better contribution graph). Do not strip or hide the trailers.

## 7. Recruiter-facing hygiene

First-read problems, in the order a visitor encounters them:
1. README stale (3.1) and has no AI disclosure, no model-licence warning, no results link.
2. Root directory shows `reports/`, `research_notes/` (7 AI-generated, snippet-based, partly UNVERIFIED notes), `kaggle/`, `kaggle_out/` (ignored, local only), `.claude/agents/` (15 seat definitions) next to the product code. D-018 requires moving or bannering these.
3. `docs/` front page leads with `PLAN.md` (v0.2 "ambitious edition", Palantir mention, weeks, star targets) and `council/` (process).
4. "15-seat council" language: `docs/council/CHARTER.md:3`, `DECISIONS.md:5` (D-001), README line 73 pointing at council docs. Tone: serious engineers will read this as theatre unless framed as an AI-assisted review process. Not a defect to hide, but keep it off the README beyond one disclosure sentence and a link.
5. History: `git log` shows 6 `wip:` commits (`0523380`, `11a050f`, `6a42292`, `098c921`, `6a7c392`, `7733433`) plus HEAD `eb42ec7 wip: Phase C integration and independent audits in progress`, and "Phase B in progress" (`9618473`). Per D-002 the plan is to merge to `main` directly; if merged as-is, main shows these.
6. Stale TODO-like markers: `grep` for TODO/FIXME/XXX/HACK found none in code or docs. `REPLACE_WITH_*` placeholders in `deploy/sieve.example.yaml` are intentional. "Phase A/B/C" labels and "Deviations: None yet." in `hypotheses.json` are fine.
7. Broken links: my relative-link check on README and docs/*.md found no missing targets. After moving files (below), all links must be rechecked (add a link-check CI job). `reports/Sieve landscape and gap analysis.md` and the note directory contain spaces in names; rename to kebab-case.
8. Wording to remove or rewrite: "Palantir-style buyers" (`PLAN.md:18`); "300+ stars" (`reports/...:3`) and star/HN/Product-Hunt growth tactics (`upstream_launch_playbook.md:88-96`); "adoption engine"/"get cited and starred" (`PLAN.md:17`); "ambitious edition".
9. `results/mock/` sits at `results/` top level beside `results/evidence/`; rename to `results/synthetic-demo/` and add `results/README.md`.
10. `docs/reproduce.md:19-21`: says the Makefile "is not owned by this seat" and instructs to edit it: leftover agent-process text. Edit.

Recommended layout:
```
README.md                 status (proven / not proven), AI disclosure, licences table, quickstart
NOTICE                    third-party attributions
SECURITY.md  CITATION.cff  LICENSE
docs/
  guide/                  server.md  observe.md  evaluation.md  reproduce.md  privacy.md (new)
  design/                 PREREGISTRATION.md  ROADMAP.md (cleaned PLAN.md)  architecture notes (reviews/architect-design.md etc.)
  research/               README.md (banner: AI-assisted, snippet-based, unverified)  + the 7 notes + gap analysis report
  process/                CHARTER.md  DECISIONS.md  reviews/ (all phase notes, audits)
  evidence/               t4-smoke.md (human-readable summary linking to results/evidence)
results/
  README.md               "only results/evidence/* are real measurements"
  evidence/               real, raw, provenance-stamped
  synthetic-demo/         pipeline demo, every file labelled SYNTHETIC
.claude/agents/           keep (or move under docs/process/ref); not linked from README
```
`docs/research/README.md` banner text: "These notes were produced by AI-assisted web search in one session on 2026-11-03. Most 2025-2026 claims are from search-result snippets or abstracts, not full-text reads. They are inputs to planning, not citations. Verify before relying on them."

## 8. Prioritised fix list

P0 (before making the repo promoted/merged to main)
1. README rewrite: lead with "Proven / Not proven" (proven: library + server + harness implemented and unit/simulation-tested on mock backends; one T4 smoke run; NOT proven: any accuracy/calibration/guarantee/cost result on a real model, quantisation effects, energy); remove "Phase A / not built yet". Owner: Director + Researcher. Verify: re-read against `ls src/sieve`.
2. AI-use disclosure paragraph in README (section 6 text; owner edits). Verify: grep README for "Claude".
3. Label `results/mock`: add SYNTHETIC columns/notes to CSVs and tables.md or move to `results/synthetic-demo/` with a README; add `# SYNTHETIC` first line to each CSV header comment or a `SYNTHETIC.txt`. Verify: `grep -L SYNTHETIC` over the directory is empty.
4. Models and licences section + NOTICE: Qwen2.5-3B research licence (non-commercial), 7B/1.5B Apache-2.0, dataset licences (section 2). Add comment in `deploy/sieve.example.yaml`.
5. SECURITY.md: remove "not yet implemented", add supported versions, scope (API keys, SSRF allow-list, log_content); confirm private vulnerability reporting is enabled in GitHub settings (owner).
6. Move `research_notes/` and `reports/` under `docs/research/` with banner (D-018); rename without spaces; fix the gitlink mirror URL; update `docs/PLAN.md` references. Move `docs/council/` to `docs/process/`.
7. Edit `docs/PLAN.md`: remove Palantir, star targets, "(verified ...)" heading contradiction; rename to ROADMAP and mark status.

P1
- P1-1 Remove "first measurement"/"headline" language from PLAN until evidenced.
- P1-2 CI: `permissions: contents: read`, SHA-pin actions, `pip-audit`, forbidden-file/size check, link check, gitleaks; add Dependabot for pip, actions, docker.
- P1-3 History cleanliness (OWNER APPROVAL REQUIRED, destructive): preferred = squash-merge the feature branch into `main` (one or a few well-named commits), keeping the branch and history in a tag/archive; alternative = leave history (honest, shows process). Do not force-push without explicit approval. Neither exposes secrets (none found), so this is cosmetic.
- P1-4 Decide authorship display (section 6).
- P1-5 Dependency pins: `constraints.txt` or lockfile; lower bounds + upper major bounds in pyproject; Dockerfile base by digest.
- P1-6 `docs/privacy.md` with retention and GDPR notes (section 4).
- P1-7 Add `*.jsonl`, `*.gguf`, `**/cache/` to `.gitignore`.

P2
- Parametrise `KAGGLE_USER`; SPDX licence metadata and classifiers in pyproject; CITATION.cff date and abstract refresh, optional ORCID; Grafana AGPL note; fix `reproduce.md` agent-process text; GitHub topics, description, social preview; run gitleaks once; add `docs/evidence/t4-smoke.md`.

## 9. Summary table

| Area | Result | Worst finding | Severity |
|---|---|---|---|
| Secrets / tokens in tree and history | None found | - | none |
| Personal data / private paths | None; owner name/GitHub login intentional; `ksainiketh` in 3 places | parametrise Kaggle user | Low |
| Code licence | MIT, consistent | classifiers/SPDX | Low |
| Dependency licences | All MIT-compatible (hypothesis MPL-2.0 dev only) | none | none |
| Model licences | 7B/1.5B Apache-2.0; 3B research/non-commercial | 3B is the default small tier, undocumented in README/deploy | Medium |
| Dataset licences | TriviaQA unknown; NQ-open CC-BY-SA-3.0; SimpleQA Verified MIT; MMLU-Redux CC-BY-4.0 | no item-level redistribution; need NOTICE | Medium |
| Claims / honesty | README stale and wrong; mock CSVs unlabelled; PLAN novelty contradiction | README says server/harness "not built" | High |
| Server privacy defaults | Strong (no content by default, hashed user, loopback, 0600) | no retention/GDPR doc | Low |
| Supply chain / CI | No install scripts; Dockerfile good | unpinned deps, no `permissions:`, tag-pinned actions, no audit | Medium |
| SECURITY.md / CITATION.cff | Present; SECURITY stale | "server not yet implemented" | Medium |
| AI-use disclosure | Trailers present; README silent | no disclosure | High |
| Recruiter hygiene | Many process docs on the front; 7 wip commits; Palantir mention | layout and wording | Medium |

Not verified by me: that tests/lint pass; that GitHub private vulnerability reporting is enabled; the content of each cited arXiv paper (only 8 of 72 IDs resolution-checked, all 200); upstream AI policies (taken from the repo's snippet-level notes); MMLU base-item provenance; gitleaks scan.
