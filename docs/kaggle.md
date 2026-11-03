# Running the experiment on Kaggle T4s

Everything needed to run the harness (`docs/evaluation.md`) on Kaggle's 2x Tesla T4 lives in
`kaggle/`. One self-contained script is pushed as a private kernel; it installs the repo, uses the
cached compiled `llama-server`, downloads the GGUFs, runs `prepare-data` / `run` / `analyse` /
`report`, and prints the results into the kernel log (Kaggle output files are served from
`kaggleusercontent.com`, which the Sieve sandbox cannot reach, D-015).

| File | Role |
|---|---|
| `kaggle/experiment_driver.py` | the driver (template; edit this) |
| `kaggle/build_experiment_script.py` | embeds the repo + run parameters -> `kaggle/run_experiment.py` |
| `kaggle/run_experiment.py` | GENERATED single file that is pushed (never edit by hand) |
| `kaggle/kaggle_run.sh` | push, poll, fetch the log (REST, auth injected by the environment) |
| `kaggle/decode_log_bundle.py` | rebuild the result bundle from the log |
| `kaggle/llamacpp_t4_smoke.py` | the original llama.cpp build/serve smoke test (kept as evidence) |

## 1. Build the script

```sh
python kaggle/build_experiment_script.py --config t4_integration --max-minutes 40   # ~220 KB output
```

The builder tars `src/ benchmarks/ pyproject.toml docs/PREREGISTRATION.md README.md LICENSE`
deterministically (sorted, zeroed times), base64-embeds it, and bakes in `config`, `stages`,
`cells`, `max_minutes`, the pinned llama.cpp commit and the git state. Same sources give the same
bytes. `tests/integration/test_kaggle_driver.py::test_committed_script_embeds_current_sources`
fails if `kaggle/run_experiment.py` is older than `src/` or `benchmarks/`: always rebuild before a push.

Useful flags: `--config t4_main` (headline), `--cells S_q4km,L_q4km` (subset), `--stages ...`
(subset of `bootstrap,binary,models,probe,prepare,run,analyse,report,sanity,emit`),
`--workdir-name` (default `sieve_work_<config>`), `--max-minutes`, `--reserve-minutes`.
At run time the environment can override `SIEVE_CONFIG`, `SIEVE_MAX_MINUTES`, `SIEVE_STAGES`, `SIEVE_CELLS`.

## 2. Push, wait, read

```sh
export KAGGLE_USER=<your-kaggle-username>
export KERNEL_DATA_SOURCES="$KAGGLE_USER/sieve-llamacpp-t4-smoke"      # cached compiled llama-server
kaggle/kaggle_run.sh kaggle/run_experiment.py sieve-t4-integration-a    # blocks until the kernel ends
python kaggle/decode_log_bundle.py kaggle_out/<run-dir>/log.txt --out kaggle_out/<run-dir>/bundle
```

* `kaggle_run.sh` needs `KAGGLE_USER`; `KERNEL_DATA_SOURCES` / `DATASET_DATA_SOURCES` are
  space/comma separated `owner/slug` lists; the output directory defaults to a fresh
  `kaggle_out/<slug>-<UTC time>`; `MAX_WAIT_MIN` (default 120) stops polling but not the kernel.
  Output-file downloads are attempted and any failure is only logged (expected from the sandbox).
* The driver prints `[+MM:SS]` progress lines, `SIEVE_STAGE_BEGIN/END`, a heartbeat every 60 s
  (GPU memory, utilisation, power, cache size) and, at the very end (also after failures):
  * `SIEVE_SUMMARY_BEGIN ... SIEVE_SUMMARY_END`: Markdown (problems, model hashes, dataset
    manifest, per cell/dataset accuracy, defects, latency, tokens/s, energy windows, real output
    samples, the hypothesis text);
  * `SIEVE_BUNDLE_BEGIN name=... chunks=N sha256=...` then `SIEVE_CHUNK <name> i/N <base64>` lines
    then `SIEVE_BUNDLE_END`: a tar.gz (capped near 500 KB) with `sanity.json`, `manifest.json`,
    `cells/*.json`, `energy.jsonl`, `analysis/*`, `report/tables/*`, server sidecars and log tails.
    `decode_log_bundle.py` checks chunk count and sha256 and extracts it.
* The full generation cache stays in the kernel's output (`/kaggle/working/<workdir_name>/`); it is
  the resume state and cannot be downloaded from the sandbox. Large artefacts need
  `kaggleusercontent.com` allowed (D-015).

## 3. What the script does

1. **bootstrap**: extract the embedded repo to `/tmp/sieve_repo`, `pip install --no-deps -e`
   (the Kaggle numpy/scipy stack is left alone), install only genuinely missing eval deps
   (`datasets matplotlib pandas pyyaml httpx scipy numpy`), restore a previous run's workdir.
2. **binary**: use `/kaggle/input/**/llama-server` from the attached kernel (checks `--version`,
   copies it to `/tmp`, requires the pinned commit in the version string); otherwise build
   llama.cpp at the pinned commit with `-DGGML_CUDA_NO_VMM=ON` (about 30 min, D-013).
3. **models**: for the selected cells, read the Hub file listing, download each GGUF (all shards
   of split files) with `curl -C -`, check size and **sha256 against the Hub's LFS oid**; the
   hash is cached next to the file and becomes `BackendInfo.model_hash`. Models go to `/kaggle/temp`
   or `/tmp`, never into the saved output.
4. **probe**: stream two rows of each pinned dataset and log the raw column names, so schema
   problems are visible even if the real load fails.
5. **prepare / run / analyse / report**: `python -m sieve.eval ...` with the provider
   `sieve.eval.providers.llamacpp:make_provider` (below). `run` uses the nvidia-smi power meter.
   A deadline (`max_minutes - reserve_minutes`) sends SIGINT to `run`; servers are stopped and the cache
   is intact. Analysis is skipped if the run is incomplete.
6. **sanity**: `sieve.eval.providers.sanity` summarises the cache (extraction, grading, defects,
   latency) with a deterministic sample of real outputs. **emit**: summary + bundle.

## 4. Resuming

Kaggle sessions end (12 h limit, weekly GPU quota, crashes). Re-run the same command with the
finished or crashed kernel's slug added as a data source:

```sh
KERNEL_DATA_SOURCES="$KAGGLE_USER/sieve-llamacpp-t4-smoke $KAGGLE_USER/sieve-t4-main-a" \
  kaggle/kaggle_run.sh kaggle/run_experiment.py sieve-t4-main-b
```

The driver finds `sieve_work_<config>/` (with its `manifest.json`) under `/kaggle/input/*/`, copies
it to `/kaggle/working/`, skips `prepare-data` (the split index is frozen) and the runner skips every
finished generation (a torn last line is dropped). Use the **same builder parameters**: the manifest
refuses a changed config, pre-registration or template hash. The new kernel must use a different slug
(or the old kernel is overwritten). Models are re-downloaded (they are not in the output).

## 5. The llama.cpp provider (`src/sieve/eval/providers/`)

* `server_manager.py`: one `llama-server` per cell on a free loopback port; GPU pinned with
  `CUDA_VISIBLE_DEVICES` (default: S on GPU 0, L on GPU 1 when two GPUs exist, override with
  `backend.gpu` or `SIEVE_GPU_S/L`); stdout/stderr in `server_logs/<cell>.server.log` with the argv on
  the first line; health wait that fails fast with the log tail; SIGTERM/SIGKILL of the process group,
  also at interpreter exit.
* `llamacpp.py`: B1's `LlamaServerBackend` over that server, prompts raw to `/completion`, greedy,
  `n_probs = 20`, `stop: ["\n"]` (the answer span is the first line; the prompt is the fixed
  few-shot template of `sieve.eval.prompts`, D-016), stop-trimmed trailing tokens removed
  (`trim_to_text`), the fail-closed logprob self-test at `open` (a failing cell does not run), and an
  identity check of the served build and model file. `BackendInfo.build` =
  `llama.cpp@<commit>|<server build_info>|fa=<flag>|ngl=<n>|stop=<json>`; `model_hash` = sha256 of the
  GGUF (shards combined); `template_hash` = `prompts.template_hash()`.
* Sidecars next to the logs: `<cell>.server.json` (argv, GPU, port, `/props`, flash-attn line
  from the server log, self-test results) and `<cell>.timings.jsonl` (per call: wall,
  `prompt_ms`, `predicted_ms`, `cache_n`...). Per-attempt wall time is also in every cache record.

## 6. Known limits

* The log is the only return channel; a very large log may be truncated by Kaggle. Keep bundles
  small; read `driver_summary.md` first.
* A bundle never contains the generation cache; analysis outputs inside it are those of the run.
* Kernel runtime is limited by Kaggle (12 h per session, weekly GPU hours). Plan the headline run in
  resumable sessions; the main config is far larger than the integration run.
* `nvidia-smi` power is GPU-only and sampled at 10 Hz; with S and L resident on different GPUs the
  windows sum both. VRAM is the peak over the window (see the notes for what it covered).
* The Hub's Xet CDN is reachable from Kaggle but not from the Sieve sandbox; only the kernel can
  prove a real dataset load.
* Everything in the integration config (`benchmarks/t4_integration.yaml`) is a plumbing test with
  tiny splits. Its numbers are not results.
