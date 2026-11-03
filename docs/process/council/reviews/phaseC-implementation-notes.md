# Phase C implementation notes (MLOps + Backend + AI Evaluation seats)

Scope: `src/sieve/eval/providers/`, `tests/integration/`, `kaggle/`, `docs/kaggle.md`,
`benchmarks/*.yaml`. Every number below comes from a Kaggle log under `kaggle_out/<slug>/log.txt`
(git-ignored; cited by slug) or from a local command whose output is quoted. **Status: the
end-to-end real run is NOT yet proven.** Three kernels were used (the cap); two real defects were
found and the second fix has not yet been re-run on a T4.

## Built
* `providers/server_manager.py`: start / health-wait / stop of one `llama-server` (free port,
  `CUDA_VISIBLE_DEVICES` pinning, log capture with argv header, fail-fast with log tail, process-group
  SIGTERM/SIGKILL, atexit).
* `providers/llamacpp.py`: `make_provider` (`backend.provider`), B1 backend wrapper, `stop: ["\n"]`,
  `trim_to_text`, fail-closed self-test in `open`, build/model-file identity checks, `BackendInfo`
  (build commit + server `build_info` + fa flag + ngl + stop rule, model sha256, quant, KV, template hash),
  sidecars `<cell>.server.json` and `<cell>.timings.jsonl`, loop-safe client reuse across `asyncio.run`.
* `providers/sanity.py`: post-run report (extraction, grade, defects, latency, tok/s, samples).
* `kaggle/experiment_driver.py` + `build_experiment_script.py` -> `run_experiment.py` (single file, repo
  embedded as deterministic base64 tgz, 226 KB), `decode_log_bundle.py`, `kaggle_run.sh`
  (`KAGGLE_USER` required, `KERNEL_DATA_SOURCES`, per-run output dir, download failures only logged),
  `diagnose_completion.py` (raw-response dump), `benchmarks/t4_integration.yaml`.
* `docs/kaggle.md`. Tests: `tests/integration/` (47 tests: real-schema fixtures, process manager,
  provider, runner through a stand-in server incl. resume, driver build/bundle/resume/dry run).

## Bugs found, with evidence
1. **Parser vs real schema (task 1): no bug.** `parse_completion_probabilities` maps the real
   `completion_probabilities` (nested `top_logprobs`) and chat `logprobs.content` entries; the
   self-test accepts them (`tests/integration/test_real_schema.py`). Fixtures are the real first-token
   entries from `results/evidence/t4_llamacpp_smoke_2026-11-03.json`; the envelope fields are rebuilt.
2. **Kaggle input mount not executable** (run a, `sieve-t4-integration-a`): `PermissionError` on
   `/kaggle/input/notebooks/ksainiketh/sieve-llamacpp-t4-smoke/llama-server`. Fix: test a private copy,
   chmod it; building is now opt-in (`--allow-build`). Run b log: `version: 0.8.0-dev (build 1, commit
   fac3a89)`, `/props build_info = b1-fac3a89`.
3. **Stop-string semantics of llama-server (run b self-test failed closed)**: `count_matches_tokens`
   and `utf8_reassembles` failed on the real server. Diagnostic kernel `sieve-diag-completion` (raw
   responses): with `stop:["\n"]` the matching token's entry is missing from `completion_probabilities`
   while `tokens_predicted` counts it (7 vs 6 entries; 4 vs 3), and when the stop token is `".\n"`
   `content` keeps the `"."` that has no entry. Without a stop string the counts match. Fix: 
   `trim_to_text(stopped_by_word=True)` cuts the text back to the reported tokens and drops the unreported
   stop token from `n_gen`; any other mismatch stays a defect. Verified only against the stand-in server
   (modes `drop`/`straddle`/`keep`) and unit tests; **not yet re-run on a T4**.
4. Integration edits outside my directories: `src/sieve/eval/prompts.py` (few-shot templates,
   `TEMPLATE_VERSION qa-v2-fewshot`, D-016), `src/sieve/eval/_util.py` (`git_commit` honours
   `SIEVE_GIT_COMMIT`; the embedded repo has no `.git`), `tests/eval/test_pipeline.py` (one assertion
   accepts the second valid reason text; the prompt change altered the synthetic mock outcome).
   `results/mock/` was generated with the old template and should be regenerated
   (`benchmarks/reproduce_mock.sh`) before release; I did not touch it.

## What each Kaggle run proved (2x T4, driver 580.178.04, Python 3.13)
| slug | driver time | proved | failed |
|---|---|---|---|
| `sieve-t4-integration-a` | 5:20 | install of the embedded repo (14 s); GGUF download + sha256 equal to Hub LFS oid (3B Q4_K_M 2.10 GB in 17 s, 7B Q4_K_M 4.68 GB in 37 s); dataset probe: TriviaQA columns `answer, entity_pages, question, question_id, question_source, search_results`, NQ-open `answer, question`, SimpleQA Verified `answer, answer_type, multi_step, original_index, problem, requires_reasoning, topic, urls`, MMLU-Redux `answer, choices, correct_answer, error_type, potential_reason, question, source`; **real `prepare-data` from the Hub for all four datasets at the pinned revisions** (n = 150/50/50/50, cross-dataset overlap 0/0/0, 190 s); sanity/emit/log bundle path | cached binary unusable (item 2) |
| `sieve-t4-integration-b` | 2:21 | resume: previous kernel's workdir restored, `prepare` skipped (0 s); cached binary accepted; both GGUFs verified again; server started on GPU 0 in 2.3 s; real calls 0.12 to 0.36 s for 7 to 12 tokens (`timings.jsonl`) | self-test (item 3): cell not run, correctly |
| `sieve-diag-completion` | about 1.5 | raw schema of stop vs no-stop responses | n/a |

Not yet measured on real data: accuracy, answer-extraction/grading on real outputs, per-item latency
over a workload, VRAM, power, certification at n_B = 50. Do not quote any number from run a/b as a
result. (Also learned: kernel output files were downloadable from the sandbox in these runs.)

Quota: three GPU kernels, roughly 9 to 10 minutes of wall time in total (my estimate from the logs; the
account's remaining weekly quota is not visible to me).

## Deviations needing a dated entry in PREREGISTRATION.md (not made by me)
* L GGUF is bartowski's single-file Q4_K_M (importance-matrix quant, same file as D-014), not Qwen's
  split Q4_K_M named in `reproduce.md`; the pre-registration says only "Qwen2.5-7B-Instruct Q4_K_M".
* S is the 3B ladder: all five quants exist on the Hub (checked 2026-11-03 via the API); no fallback needed.
* Fixed few-shot prompt template (`qa-v2-fewshot`) and `stop: ["\n"]`: answer span = first line; the
  dropped stop-token logprob is not part of any signal. Server flags: `-np 1 -c 4096 -ngl 99`,
  flash attention default `auto` (forced `on` for quantised KV), `cache_prompt=false`.
* Few-shot examples are generic trivia, not benchmark items (contamination not formally checked).

## Remaining gaps
* One more T4 run of `t4_integration` is needed to prove items 3 and the measurement goals above. Command:
  `export KAGGLE_USER=ksainiketh KERNEL_DATA_SOURCES="ksainiketh/sieve-llamacpp-t4-smoke ksainiketh/sieve-t4-integration-b"`
  then `kaggle/kaggle_run.sh kaggle/run_experiment.py sieve-t4-integration-c` (script already rebuilt with
  the fix; expect about 10 to 15 min). I stopped at the three-kernel cap.
* Unknown until then: whether the EOS token has an entry; the rate of `.\n`-style straddles; the flash-attn
  log line (`log_flash_attn` was null); VRAM and power windows span both GPUs when S and L are resident.
* t4_main needs a long resumable session plan; the 8 ablation cells are untested on a GPU.
* Driver prints a log of about 20 KB for the failed runs; a full run's log size limit is unknown.
* `kaggle/run_experiment.py` embeds the sources: rebuild it (`python kaggle/build_experiment_script.py
  --config <cfg>`) after any change; a test fails when it is stale (it is currently built for t4_integration).

## Exact checks (2026-11-03, after the last edit)
`ruff check src tests`: All checks passed. `ruff format --check src tests`: 115 files already formatted.
`mypy`: Success, no issues found in 115 source files. `pytest`: 511 passed, 3 skipped in 105.86 s.
Commands: `.venv/bin/python -m ruff check src tests`, `... ruff format --check src tests`, `... mypy`, `... pytest`.
