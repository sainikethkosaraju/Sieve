"""Sieve Kaggle experiment driver (TEMPLATE; the pushed script is generated from it).

Do not edit ``kaggle/run_experiment.py`` by hand: edit this file and run
``python kaggle/build_experiment_script.py`` (it embeds the repository as a base64 tar.gz and
bakes the run parameters in). Everything below the placeholders is stdlib-only until the repo has
been installed.

Stages (``PARAMS["stages"]``): bootstrap, binary, models, probe, prepare, run, analyse, report,
sanity, emit. The workdir lives in /kaggle/working/<workdir_name> and is the resume state: attach a
previous run's kernel as a data source and the driver copies its workdir back before continuing
(the generation cache skips finished work). Nothing secret is read or printed; HF downloads are
anonymous. Results come back through the log between SIEVE_* markers (kaggleusercontent.com is not
reachable from the Sieve sandbox); decode them with ``kaggle/decode_log_bundle.py``.
"""

from __future__ import annotations

import base64
import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

_PARAMS_JSON = "@@PARAMS@@"
_EMBEDDED_B64 = "@@EMBED@@"

T0 = time.time()
KAGGLE = os.path.isdir("/kaggle/working")
WORK = Path("/kaggle/working") if KAGGLE else Path.cwd() / "kaggle_working"
REPO = Path("/tmp/sieve_repo")
INPUT_ROOT = Path("/kaggle/input")
CHUNK = 3000
STAGE_ORDER = (
    "bootstrap",
    "binary",
    "models",
    "probe",
    "prepare",
    "run",
    "analyse",
    "report",
    "sanity",
    "emit",
)
SHARD = re.compile(r"^(?P<stem>.+)-(?P<i>\d{5})-of-(?P<n>\d{5})\.gguf$")
SUMMARY: dict[str, Any] = {"stages": {}, "problems": []}


def log(msg: str) -> None:
    el = int(time.time() - T0)
    print(f"[+{el // 60:02d}:{el % 60:02d}] {msg}", flush=True)


def problem(msg: str) -> None:
    SUMMARY["problems"].append(msg)
    log(f"PROBLEM: {msg}")


# --- parameters ----------------------------------------------------------------------------


def load_params() -> dict[str, Any]:
    if _PARAMS_JSON.startswith("@@"):
        raise SystemExit("template not built: run python kaggle/build_experiment_script.py")
    p: dict[str, Any] = json.loads(_PARAMS_JSON)
    env = os.environ
    if env.get("SIEVE_CONFIG"):
        p["config"] = env["SIEVE_CONFIG"]
    if env.get("SIEVE_MAX_MINUTES"):
        p["max_minutes"] = float(env["SIEVE_MAX_MINUTES"])
    if env.get("SIEVE_STAGES"):
        p["stages"] = env["SIEVE_STAGES"].split(",")
    if env.get("SIEVE_CELLS"):
        p["cells"] = env["SIEVE_CELLS"].split(",")
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            p["config"] = argv[i + 1]
    p.setdefault("workdir_name", "sieve_work_" + str(p["config"]))
    return p


# --- subprocess helpers --------------------------------------------------------------------


def run_cmd(
    cmd: list[str],
    *,
    tag: str,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    deadline: float | None = None,
    interrupt_grace_s: float = 120.0,
    quiet: bool = False,
) -> tuple[int, bool, str]:
    """Run, streaming prefixed lines to the log. Returns (rc, interrupted_by_deadline, tail)."""
    e = dict(os.environ)
    if env:
        e.update(env)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=e,
        cwd=str(cwd) if cwd else None,
        start_new_session=True,
    )
    tail: list[str] = []

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            tail.append(line)
            del tail[:-60]
            if not quiet:
                print(f"[{tag}] {line}", flush=True)

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    interrupted = False
    t_int = 0.0
    while proc.poll() is None:
        now = time.time()
        if deadline is not None and now > deadline and not interrupted:
            log(f"DEADLINE reached: sending SIGINT to {tag}")
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGINT)
            interrupted, t_int = True, now
        elif interrupted and now - t_int > interrupt_grace_s:
            log(f"{tag} ignored SIGINT for {interrupt_grace_s:.0f}s: SIGKILL")
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
        time.sleep(1.0)
    th.join(timeout=10)
    return int(proc.returncode), interrupted, "\n".join(tail)


def sh_out(cmd: str, timeout: float = 60) -> str:
    try:
        p = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.SubprocessError as e:
        return f"<{type(e).__name__}>"
    return (p.stdout + p.stderr).strip()


class Heartbeat(threading.Thread):
    def __init__(self, workdir: Path, every_s: float = 60.0) -> None:
        super().__init__(daemon=True)
        self.workdir, self.every_s, self.stop_flag = workdir, every_s, False

    def run(self) -> None:
        while not self.stop_flag:
            time.sleep(self.every_s)
            gpu = sh_out(
                "nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw "
                "--format=csv,noheader,nounits",
                timeout=20,
            ).replace("\n", " | ")
            gen = self.workdir / "generations"
            mb = sum(f.stat().st_size for f in gen.glob("*.jsonl")) / 1e6 if gen.exists() else 0.0
            log(f"HEARTBEAT gpu[idx,memMiB,util%,W]={gpu} generations_cache={mb:.1f}MB")


# --- stages --------------------------------------------------------------------------------


def stage_bootstrap(p: dict[str, Any]) -> None:
    raw = base64.b64decode(re.sub(r"\s+", "", _EMBEDDED_B64))
    if REPO.exists():
        shutil.rmtree(REPO)
    REPO.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        tf.extractall(REPO, filter="data")
    log(
        f"extracted repo ({len(raw)} bytes tgz) to {REPO}: {sorted(x.name for x in REPO.iterdir())}"
    )
    SUMMARY["repo_commit"] = p.get("repo_commit")
    sys.path.insert(0, str(REPO / "src"))  # importable in this process before/without pip
    log("env: " + sh_out("python --version").replace("\n", " "))
    log("gpus: " + sh_out("nvidia-smi -L").replace("\n", " ; "))
    log(
        "driver: "
        + sh_out(
            "nvidia-smi --query-gpu=driver_version,memory.total --format=csv,noheader"
        ).replace("\n", " ; ")
    )
    log(
        "disk: "
        + sh_out("df -h /kaggle/working /tmp /kaggle/temp 2>/dev/null").replace("\n", " ; ")
    )
    # Install without resolving dependencies (the Kaggle image has a tuned numpy/scipy stack),
    # then add only what is genuinely missing for the eval extra.
    if os.environ.get("SIEVE_SKIP_PIP"):  # local dry runs must not touch the active environment
        log("SIEVE_SKIP_PIP set: not installing")
        rc = 0
        tail = ""
    else:
        rc, _, tail = run_cmd(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(REPO)], tag="pip"
        )
    if rc != 0:
        problem(f"pip install -e failed (rc={rc}); falling back to PYTHONPATH. tail: {tail[-300:]}")
    need = {
        "numpy": "numpy",
        "scipy": "scipy",
        "httpx": "httpx",
        "yaml": "pyyaml",
        "datasets": "datasets",
        "matplotlib": "matplotlib",
        "pandas": "pandas",
    }
    missing = [pkg for mod, pkg in need.items() if importlib.util.find_spec(mod) is None]
    if missing and not os.environ.get("SIEVE_SKIP_PIP"):
        log(f"installing missing eval deps: {missing}")
        rc, _, tail = run_cmd([sys.executable, "-m", "pip", "install", "-q", *missing], tag="pip")
        if rc != 0:
            problem(f"pip install of {missing} failed: {tail[-300:]}")
    importlib.invalidate_caches()
    versions = sh_out(
        f'{sys.executable} -c "import numpy,scipy,httpx,yaml,datasets,pandas;'
        "print('numpy',numpy.__version__,'scipy',scipy.__version__,'httpx',httpx.__version__,"
        "'datasets',datasets.__version__,'pandas',pandas.__version__)\""
    )
    log("versions: " + versions)
    SUMMARY["versions"] = versions


def child_env(p: dict[str, Any], extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        "PYTHONPATH": str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/tmp/hf_home",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "SIEVE_GIT_COMMIT": str(p.get("repo_commit") or "unknown"),
    }
    if "SIEVE_LLAMA_SERVER" in os.environ:
        env["SIEVE_LLAMA_SERVER"] = os.environ["SIEVE_LLAMA_SERVER"]
    if extra:
        env.update(extra)
    return env


def models_dir() -> Path:
    for cand in (Path("/kaggle/temp"), Path("/tmp")):
        if cand.is_dir() and os.access(cand, os.W_OK):
            d = cand / "sieve_models"
            d.mkdir(parents=True, exist_ok=True)
            return d
    raise SystemExit("no writable scratch directory for models")


def find_files(name: str, roots: list[str], max_depth: int = 5) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        base = root.rstrip("/").count("/")
        for dp, dns, fns in os.walk(root):
            if dp.count("/") - base >= max_depth:
                dns[:] = []
            if name in fns:
                out.append(Path(dp) / name)
    return out


def build_llama_server(commit: str, dest: Path) -> bool:
    """Fallback: build llama-server at ``commit`` for sm_75 (D-013: needs GGML_CUDA_NO_VMM=ON)."""
    src = Path("/tmp/llama.cpp")
    shutil.rmtree(src, ignore_errors=True)
    src.mkdir(parents=True)
    steps = [
        ["git", "init", "-q"],
        ["git", "remote", "add", "origin", "https://github.com/ggml-org/llama.cpp"],
        ["git", "fetch", "-q", "--depth", "1", "origin", commit],
        ["git", "checkout", "-q", "FETCH_HEAD"],
    ]
    for s in steps:
        rc, _, tail = run_cmd(s, tag="git", cwd=src)
        if rc != 0:
            problem(f"llama.cpp checkout failed: {tail[-300:]}")
            return False
    cfg = (
        "cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DBUILD_SHARED_LIBS=OFF "
        "-DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_NO_VMM=ON "
        "-DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs"
    )
    for cmd in (cfg, "cmake --build build --target llama-server -j4"):
        rc, _, tail = run_cmd(["bash", "-c", cmd], tag="cmake", cwd=src, quiet=True)
        if rc != 0:
            problem(f"llama.cpp build step failed: {tail[-500:]}")
            return False
    built = src / "build" / "bin" / "llama-server"
    if not built.exists():
        return False
    shutil.copy(built, dest)
    dest.chmod(0o755)
    return True


def stage_binary(p: dict[str, Any]) -> None:
    commit = str(p["llama_cpp_commit"])
    preset = os.environ.get("SIEVE_LLAMA_SERVER")
    if preset and os.access(preset, os.X_OK):
        log(f"using preset SIEVE_LLAMA_SERVER={preset}")
        SUMMARY["binary_source"] = f"preset: {preset}"
        return
    hint = str(p.get("llama_slug_hint", "sieve-llamacpp"))
    cands = find_files("llama-server", [str(INPUT_ROOT)])
    cands.sort(key=lambda c: (hint not in str(c), str(c)))
    log(f"llama-server candidates from data sources: {[str(c) for c in cands]}")
    chosen: Path | None = None
    for i, c in enumerate(cands):
        # /kaggle/input is read-only and may lack the exec bit: test a private copy
        local_try = Path(f"/tmp/llama-server-cand{i}")
        try:
            shutil.copy(c, local_try)
            local_try.chmod(0o755)
            rc, _, tail = run_cmd([str(local_try), "--version"], tag="llama-server", quiet=True)
        except OSError as e:
            log(f"{c}: unusable ({type(e).__name__}: {e})")
            continue
        log(f"{c} --version rc={rc}: {tail[-200:]!r}")
        if rc == 0:
            chosen = local_try
            break
    if chosen is None:
        if not p.get("allow_build"):
            raise RuntimeError(
                "no usable cached llama-server in /kaggle/input and building is disabled "
                "(rebuild the script with --allow-build to compile it, about 30 min)"
            )
        dest = WORK / "llama-server"
        log(f"no usable cached llama-server; building {commit} (about 30 min)")
        if not build_llama_server(commit, dest):
            raise RuntimeError("no usable llama-server")
        chosen = dest
        SUMMARY["binary_source"] = "built"
    else:
        SUMMARY["binary_source"] = f"data source: {chosen}"
    # run from /tmp so that exec permissions survive read-only /kaggle/input mounts
    local = Path("/tmp/llama-server")
    shutil.copy(chosen, local)
    local.chmod(0o755)
    os.environ["SIEVE_LLAMA_SERVER"] = str(local)
    rc, _, tail = run_cmd([str(local), "--version"], tag="llama-server", quiet=True)
    SUMMARY["llama_server_version"] = tail[-300:]
    short = commit[:7]
    if short not in tail:
        problem(f"llama-server --version does not mention pinned commit {short}: {tail[-200:]!r}")
    log(f"using {local} (from {chosen}); version: {tail[-160:]!r}")


def plan_downloads(cfg: dict[str, Any], only: list[str] | None) -> list[tuple[str, str]]:
    """``(hf_repo, filename)`` pairs (all shards of a split GGUF) for the selected cells."""
    seen: dict[tuple[str, str], None] = {}
    for c in cfg["cells"]:
        if only and c["name"] not in only:
            continue
        spec = str(c.get("extra", {}).get("gguf", ""))
        if ":" not in spec:
            continue
        repo, fn = spec.split(":", 1)
        m = SHARD.match(fn)
        if m:
            n = int(m.group("n"))
            for i in range(1, n + 1):
                seen[(repo, f"{m.group('stem')}-{i:05d}-of-{n:05d}.gguf")] = None
        else:
            seen[(repo, fn)] = None
    return list(seen)


def load_cfg(p: dict[str, Any]) -> dict[str, Any]:
    import yaml

    path = REPO / "benchmarks" / f"{p['config']}.yaml"
    d: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return d


def hf_tree(repo: str) -> dict[str, dict[str, Any]]:
    url = f"https://huggingface.co/api/models/{repo}/tree/main"
    with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310
        rows = json.loads(r.read())
    return {x["path"]: x for x in rows if x.get("type") == "file"}


def stage_models(p: dict[str, Any]) -> None:
    from sieve.eval.providers.llamacpp import file_sha256

    cfg = load_cfg(p)
    md = models_dir()
    os.environ["SIEVE_MODELS_DIR"] = str(md)
    trees: dict[str, dict[str, dict[str, Any]]] = {}
    report: list[dict[str, Any]] = []
    for repo, fn in plan_downloads(cfg, p.get("cells")):
        if repo not in trees:
            trees[repo] = hf_tree(repo)
        meta = trees[repo].get(fn)
        if meta is None:
            raise RuntimeError(f"{repo}/{fn} not found on the Hub")
        size = int(meta["size"])
        oid = str((meta.get("lfs") or {}).get("oid", ""))
        dest = md / fn
        if dest.exists() and dest.stat().st_size == size:
            log(f"model present: {fn}")
        else:
            free = shutil.disk_usage(md).free
            if free < size + (2 << 30):
                raise RuntimeError(
                    f"not enough disk for {fn}: free={free >> 20}MiB need={size >> 20}MiB"
                )
            url = f"https://huggingface.co/{repo}/resolve/main/{fn}"
            t = time.time()
            rc, _, tail = run_cmd(
                [
                    "curl",
                    "-sSL",
                    "--fail",
                    "--retry",
                    "5",
                    "--retry-delay",
                    "5",
                    "-C",
                    "-",
                    "-o",
                    str(dest),
                    url,
                ],
                tag="curl",
            )
            if rc != 0:
                raise RuntimeError(f"download of {fn} failed rc={rc}: {tail[-200:]}")
            log(f"downloaded {fn} {size / 1e9:.2f} GB in {time.time() - t:.0f}s")
        got = dest.stat().st_size
        digest = file_sha256(dest)
        ok = got == size and (not oid or digest == oid)
        report.append(
            {"repo": repo, "file": fn, "size": got, "sha256": digest, "hf_oid": oid, "verified": ok}
        )
        log(f"verify {fn}: size_ok={got == size} sha256_matches_hub={digest == oid}")
        if not ok:
            raise RuntimeError(f"integrity check failed for {fn}")
    SUMMARY["models"] = report


def stage_probe(p: dict[str, Any]) -> None:
    """Print the raw Hub schema of each dataset (first rows, streamed) before the real load."""
    code = r"""
import itertools, json, sys
from datasets import load_dataset
from sieve.eval.datasets import PINNED, MMLU_SUBJECTS
out = {}
for name, spec in PINNED.items():
    try:
        cfgname = spec.config
        if name == "mmlu_redux":
            cfgname = MMLU_SUBJECTS[1]
        args = [spec.repo] + ([cfgname] if cfgname else [])
        ds = load_dataset(*args, split=spec.split, revision=spec.revision, streaming=True)
        rows = list(itertools.islice(ds, 2))
        out[name] = {"columns": sorted(rows[0].keys()), "row0": {k: str(v)[:160] for k, v in rows[0].items()}}
    except Exception as e:
        out[name] = {"error": repr(e)[:400]}
print("SIEVE_PROBE " + json.dumps(out))
"""
    rc, _, tail = run_cmd(
        [sys.executable, "-c", code], tag="probe", env=child_env(p), quiet=True
    )
    line = next((x for x in tail.splitlines() if x.startswith("SIEVE_PROBE ")), "")
    if not line:
        problem(f"dataset probe produced no result (rc={rc}): {tail[-400:]}")
        return
    probe = json.loads(line[len("SIEVE_PROBE ") :])
    SUMMARY["probe"] = probe
    for name, v in probe.items():
        if "error" in v:
            problem(f"dataset probe {name}: {v['error']}")
        else:
            log(f"probe {name}: columns={v['columns']}")


def cfg_path(p: dict[str, Any]) -> str:
    return str(REPO / "benchmarks" / f"{p['config']}.yaml")


def restore_previous(workdir: Path, name: str) -> None:
    """Copy a previous run's workdir (kernel output attached as a data source) back for resume."""
    if workdir.exists() and (workdir / "manifest.json").exists():
        log(f"workdir already present at {workdir}")
        return
    roots = [str(x) for x in INPUT_ROOT.glob("*")] if INPUT_ROOT.exists() else []
    best: Path | None = None
    for r in roots:
        for dp, dns, _ in os.walk(r):
            if dp[len(r) :].count("/") >= 3:
                dns[:] = []
            if Path(dp).name == name and (Path(dp) / "manifest.json").exists():
                best = Path(dp)
    if best is None:
        log(f"no previous workdir named {name!r} under {INPUT_ROOT}: starting fresh")
        return
    shutil.copytree(best, workdir, dirs_exist_ok=True)
    n_lines = (
        sum(sum(1 for _ in f.open("rb")) for f in (workdir / "generations").glob("*.jsonl"))
        if (workdir / "generations").exists()
        else 0
    )
    SUMMARY["resumed_from"] = str(best)
    log(f"RESUME: restored {best} -> {workdir}; cached generation lines: {n_lines}")


def stage_prepare(p: dict[str, Any], workdir: Path) -> None:
    if (workdir / "splits" / "splits_index.json").exists() and (
        workdir / "items" / "datasets_manifest.json"
    ).exists():
        log("prepare-data already done (resume): skipping")
    else:
        rc, _, tail = run_cmd(
            [
                sys.executable,
                "-m",
                "sieve.eval",
                "prepare-data",
                "--config",
                cfg_path(p),
                "--workdir",
                str(workdir),
            ]
            + (["--synthetic"] if os.environ.get("SIEVE_SYNTHETIC_DATA") else []),
            tag="prepare",
            env=child_env(p),
            cwd=REPO,
        )
        if rc != 0:
            raise RuntimeError(f"prepare-data failed rc={rc}: {tail[-600:]}")
    man = json.loads((workdir / "items" / "datasets_manifest.json").read_text(encoding="utf-8"))
    SUMMARY["datasets_manifest"] = man
    log("datasets_manifest: " + json.dumps(man, sort_keys=True)[:2500])


def stage_run(p: dict[str, Any], workdir: Path, deadline: float) -> bool:
    cmd = [
        sys.executable,
        "-m",
        "sieve.eval",
        "run",
        "--config",
        cfg_path(p),
        "--workdir",
        str(workdir),
    ]
    for c in p.get("cells") or []:
        cmd += ["--cell", c]
    env = child_env(
        p,
        {
            "SIEVE_SERVER_LOG_DIR": str(workdir / "server_logs"),
            "SIEVE_MODELS_DIR": os.environ.get("SIEVE_MODELS_DIR", str(models_dir())),
        },
    )
    rc, interrupted, tail = run_cmd(cmd, tag="run", env=env, cwd=REPO, deadline=deadline)
    SUMMARY["run_rc"] = rc
    SUMMARY["run_interrupted_by_deadline"] = interrupted
    if rc != 0:
        problem(f"run exited rc={rc} interrupted={interrupted}; tail: {tail[-500:]}")
        return False
    return True


def stage_analyse(p: dict[str, Any], workdir: Path) -> bool:
    cmd = [
        sys.executable,
        "-m",
        "sieve.eval",
        "analyse",
        "--config",
        cfg_path(p),
        "--workdir",
        str(workdir),
    ]
    rc, _, tail = run_cmd(cmd, tag="analyse", env=child_env(p), cwd=REPO)
    if rc != 0:
        problem(f"analyse failed rc={rc}: {tail[-800:]}")
    return rc == 0


def stage_report(p: dict[str, Any], workdir: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "sieve.eval",
        "report",
        "--config",
        cfg_path(p),
        "--workdir",
        str(workdir),
    ]
    rc, _, tail = run_cmd(cmd, tag="report", env=child_env(p), cwd=REPO, quiet=True)
    if rc != 0:
        problem(f"report failed rc={rc}: {tail[-600:]}")


def stage_sanity(p: dict[str, Any], workdir: Path) -> None:
    out = workdir / "sanity.json"
    cmd = [
        sys.executable, "-m", "sieve.eval.providers.sanity", "--config", cfg_path(p),
        "--workdir", str(workdir), "--per-dataset", str(p.get("sanity_per_dataset", 6)), "--out", str(out),
    ]  # fmt: skip
    rc, _, tail = run_cmd(cmd, tag="sanity", env=child_env(p), cwd=REPO, quiet=True)
    if rc != 0:
        problem(f"sanity report failed rc={rc}: {tail[-600:]}")
        return
    SUMMARY["sanity"] = json.loads(out.read_text(encoding="utf-8"))


# --- result emission -----------------------------------------------------------------------


def summary_markdown(p: dict[str, Any], workdir: Path) -> str:
    L: list[str] = [f"# Sieve Kaggle run: {p['config']}", ""]
    L.append(
        f"elapsed {int(time.time() - T0)} s; repo commit {p.get('repo_commit')}; stages {SUMMARY['stages']}"
    )
    for k in (
        "binary_source",
        "llama_server_version",
        "versions",
        "resumed_from",
        "run_rc",
        "run_interrupted_by_deadline",
    ):
        if k in SUMMARY:
            L.append(f"- {k}: {str(SUMMARY[k]).strip()[:300]}")
    if SUMMARY["problems"]:
        L += ["", "## Problems"] + [f"- {x[:400]}" for x in SUMMARY["problems"]]
    if "models" in SUMMARY:
        L += [
            "",
            "## Models (sha256 verified against the Hub)",
            "| file | size | verified |",
            "|---|---|---|",
        ]
        L += [f"| {m['file']} | {m['size']} | {m['verified']} |" for m in SUMMARY["models"]]
    man = SUMMARY.get("datasets_manifest")
    if man:
        L += ["", "## Datasets manifest", "```", json.dumps(man, sort_keys=True)[:3000], "```"]
    san = SUMMARY.get("sanity")
    if san:
        L += [
            "",
            "## Per cell and dataset",
            "| cell | dataset | n | acc | empty | defects | p50 wall s | tok/s p50 | first-tok p |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for cn, c in san["cells"].items():
            for dn, d in c["datasets"].items():
                L.append(
                    f"| {cn} | {dn} | {d['n']} | {d['accuracy']} | {d['empty_answer']} | {d['defects']} | "
                    f"{d['wall_s']['p50']} | {d['gen_tokens_per_s_p50']} | {d['first_token_prob_mean']} |"
                )
        L += [
            "",
            "## Servers",
            "```",
            json.dumps(san.get("server", {}), sort_keys=True)[:2500],
            "```",
        ]
        L += ["", "## Energy / VRAM (nvidia-smi windows)", "```"]
        L += [json.dumps(e, sort_keys=True) for e in san.get("energy", [])[:12]] + ["```"]
        L += ["", "## Output samples (real generations)"]
        for cn, c in san["cells"].items():
            for dn, d in c["datasets"].items():
                for s in d["sample"][:4]:
                    L.append(
                        f"- {cn}/{dn} Q={s['question'][:90]!r} gold={s['gold'][:2]} raw={s['raw_text'][:70]!r} "
                        f"ext={s['extracted'][:50]!r} grade={s['grade']} p1={s['first_token_prob']} wall={s['wall_s']}"
                    )
    hyp = workdir / "analysis" / "hypotheses.txt"
    if hyp.exists():
        L += [
            "",
            "## Hypotheses (tiny-n; not the pre-registered result)",
            "```",
            hyp.read_text(encoding="utf-8")[:6000],
            "```",
        ]
    return "\n".join(L)


def collect_bundle_files(workdir: Path) -> list[Path]:
    rel = [
        "driver_summary.md", "sanity.json", "manifest.json", "analysis/hypotheses.txt",
        "analysis/hypotheses.json", "items/datasets_manifest.json", "splits/splits_index.json",
        "energy.jsonl", "report/tables/tables.md",
    ]  # fmt: skip
    files = [workdir / r for r in rel]
    files += sorted((workdir / "cells").glob("*.json"))
    files += sorted((workdir / "server_logs").glob("*.server.json"))
    files += sorted((workdir / "server_logs").glob("*.server.log"))
    files += sorted((workdir / "report" / "tables").glob("*.csv"))
    files.append(workdir / "analysis" / "results.json")
    return [f for f in files if f.exists()]


def make_bundle(files: list[Path], base: Path, cap_bytes: int = 500_000) -> bytes:
    """tar.gz of ``files`` (relative to ``base``); later files are dropped if the cap is hit."""
    kept: list[Path] = []
    for f in files:
        trial = kept + [f]
        if len(_tgz(trial, base)) <= cap_bytes:
            kept = trial
        else:
            log(f"bundle cap: dropping {f.name} ({f.stat().st_size} bytes)")
    return _tgz(kept, base)


def _tgz(files: list[Path], base: Path) -> bytes:
    bio = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=bio, mode="wb", mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w") as tf,
    ):
        for f in files:
            data = f.read_bytes()
            if f.suffix == ".log":
                data = data[-20000:]
            ti = tarfile.TarInfo(str(f.relative_to(base)))
            ti.size, ti.mtime = len(data), 0
            tf.addfile(ti, io.BytesIO(data))
    return bio.getvalue()


def emit_bundle(name: str, blob: bytes, out=print) -> None:  # type: ignore[no-untyped-def]
    b64 = base64.b64encode(blob).decode()
    chunks = [b64[i : i + CHUNK] for i in range(0, len(b64), CHUNK)] or [""]
    sha = hashlib.sha256(b64.encode()).hexdigest()
    out(f"SIEVE_BUNDLE_BEGIN name={name} chunks={len(chunks)} b64_chars={len(b64)} sha256={sha}")
    for i, c in enumerate(chunks, 1):
        out(f"SIEVE_CHUNK {name} {i:04d}/{len(chunks):04d} {c}")
    out(f"SIEVE_BUNDLE_END name={name}")


def stage_emit(p: dict[str, Any], workdir: Path) -> None:
    md = summary_markdown(p, workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "driver_summary.md").write_text(md, encoding="utf-8")
    print("SIEVE_SUMMARY_BEGIN", flush=True)
    for ln in md.splitlines():
        print(ln, flush=True)
    print("SIEVE_SUMMARY_END", flush=True)
    blob = make_bundle(collect_bundle_files(workdir), workdir)
    emit_bundle(str(p["config"]), blob)
    log(f"emitted bundle: {len(blob)} bytes tgz")


# --- main ----------------------------------------------------------------------------------


def main() -> int:
    p = load_params()
    stages = [s for s in STAGE_ORDER if s in set(p["stages"])]
    workdir = WORK / str(p["workdir_name"])
    deadline = T0 + float(p["max_minutes"]) * 60.0
    run_deadline = deadline - float(p.get("reserve_minutes", 8)) * 60.0
    WORK.mkdir(parents=True, exist_ok=True)
    log(
        f"SIEVE DRIVER start: config={p['config']} stages={stages} max_minutes={p['max_minutes']} workdir={workdir}"
    )
    hb = Heartbeat(workdir)
    hb.start()
    complete = True
    try:
        for st in stages:
            t = time.time()
            log(f"SIEVE_STAGE_BEGIN {st}")
            ok = True
            try:
                if st == "bootstrap":
                    stage_bootstrap(p)
                    restore_previous(workdir, str(p["workdir_name"]))
                elif st == "binary":
                    stage_binary(p)
                elif st == "models":
                    stage_models(p)
                elif st == "probe":
                    stage_probe(p)
                elif st == "prepare":
                    stage_prepare(p, workdir)
                elif st == "run":
                    bad = [x for x in ("binary", "models", "prepare") if x in stages and not SUMMARY["stages"].get(x, {}).get("ok", True)]
                    if bad:
                        raise RuntimeError(f"not running: prerequisite stage(s) failed: {bad}")
                    ok = stage_run(p, workdir, run_deadline)
                    complete = complete and ok
                elif st == "analyse":
                    if complete:
                        ok = stage_analyse(p, workdir)
                    else:
                        log(
                            "skipping analyse: run incomplete (resume by re-pushing with this kernel as a data source)"
                        )
                elif st == "report":
                    if complete and (workdir / "analysis" / "results.json").exists():
                        stage_report(p, workdir)
                elif st == "sanity":
                    stage_sanity(p, workdir)
                elif st == "emit":
                    pass
            except Exception as e:
                ok = False
                complete = complete and st not in (
                    "bootstrap",
                    "binary",
                    "models",
                    "prepare",
                    "run",
                )
                problem(f"stage {st} raised {type(e).__name__}: {str(e)[:600]}")
            SUMMARY["stages"][st] = {"ok": ok, "s": round(time.time() - t, 1)}
            log(f"SIEVE_STAGE_END {st} ok={ok} {time.time() - t:.0f}s")
            if st in ("bootstrap",) and not ok:
                break
    finally:
        hb.stop_flag = True
        try:
            stage_emit(p, workdir)
        except Exception as e:
            log(f"emit failed: {type(e).__name__}: {e}")
    log("DONE " + json.dumps({"problems": len(SUMMARY["problems"]), "stages": SUMMARY["stages"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
