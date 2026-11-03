"""Sieve Kaggle smoke test: build llama.cpp (CUDA, sm_75) and serve a real GGUF on a T4.

Proves, with raw evidence written to /kaggle/working/smoke_results.json:
  - llama.cpp CUDA build works on the Kaggle image (commit recorded)
  - a small and a large GGUF load on a T4
  - llama-server returns per-token logprobs on /completion (n_probs) and /v1/chat/completions
  - quantized KV cache (q8_0) + flash attention runs, or the exact failure message
  - tokens/s, VRAM, power samples
Stdlib only. Everything is logged; failures are recorded, never hidden.
"""
import json, os, subprocess, threading, time, urllib.request, urllib.error, shutil

W = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
SRC = os.path.join(W, "llama.cpp")
BIN = os.path.join(W, "llama-server")
RES = {"steps": {}, "errors": []}
MODELS = {
    "small_q4km": ("https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf", "small.gguf"),
    "large_q4km": ("https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf", "large.gguf"),
}


def sh(cmd, timeout=None, cwd=None):
    t = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    return {"cmd": cmd, "rc": p.returncode, "sec": round(time.time() - t, 1),
            "tail": (p.stdout + p.stderr)[-1500:]}


def save():
    json.dump(RES, open(os.path.join(W, "smoke_results.json"), "w"), indent=2)


def step(name, fn):
    t = time.time()
    try:
        RES["steps"][name] = fn()
    except Exception as e:  # recorded, not hidden
        RES["steps"][name] = {"error": repr(e)}
        RES["errors"].append(name)
    RES["steps"][name]["elapsed_s"] = round(time.time() - t, 1)
    save()
    print(name, "->", json.dumps(RES["steps"][name])[:3000], flush=True)


def gpu_info():
    return sh("nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader")


def build():
    if os.path.exists(BIN):
        return {"cached": True}
    r = [sh(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {SRC}", timeout=600)]
    r.append(sh(f"git -C {SRC} rev-parse HEAD"))
    RES["llama_cpp_commit"] = r[-1]["tail"].strip()
    RES["cuda_env"] = sh("nvcc --version | tail -2; ls -d /usr/local/cuda* ; ls /usr/local/cuda/lib64/stubs 2>&1 | head -5")
    base = ("cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 "
            "-DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release")
    attempts = [
        ("stubs_path", base + " -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs"),
        ("no_vmm", base + " -DGGML_CUDA_NO_VMM=ON -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs"),
    ]
    RES["cmake_attempts"] = []
    for name, cfg in attempts:
        sh("rm -rf build", cwd=SRC)
        c = sh(cfg, cwd=SRC, timeout=900)
        RES["cmake_attempts"].append({"name": name, "rc": c["rc"], "tail": c["tail"][-700:]})
        if c["rc"] == 0:
            RES["cmake_config_used"] = name
            break
    r.append(c)
    if c["rc"] == 0:
        r.append(sh("cmake --build build --target llama-server -j4", cwd=SRC, timeout=6000))
    src_bin = os.path.join(SRC, "build", "bin", "llama-server")
    if os.path.exists(src_bin):
        shutil.copy(src_bin, BIN)
        os.chmod(BIN, 0o755)
    return {"binary": os.path.exists(BIN), "log": [x["tail"][-300:] for x in r], "rcs": [x["rc"] for x in r]}


def download():
    out = {}
    for k, (url, fn) in MODELS.items():
        p = os.path.join(W, fn)
        r = sh(f"curl -sSL --fail -o {p} '{url}'", timeout=3000)
        out[k] = {"rc": r["rc"], "bytes": os.path.getsize(p) if os.path.exists(p) else 0, "tail": r["tail"][-200:]}
    return out


class Power(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self.stop = [], False

    def run(self):
        while not self.stop:
            r = sh("nvidia-smi --query-gpu=index,power.draw,memory.used,utilization.gpu --format=csv,noheader,nounits")
            self.samples.append((round(time.time(), 2), r["tail"].strip().replace("\n", " | ")))
            time.sleep(0.5)


def http(url, payload=None, timeout=120):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def serve_and_probe(model_file, extra_args, tag, port):
    log_path = os.path.join(W, f"server_{tag}.log")
    cmd = [BIN, "-m", os.path.join(W, model_file), "-ngl", "99", "-c", "4096", "--port", str(port),
           "--host", "127.0.0.1"] + extra_args
    lf = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    res = {"cmd": " ".join(cmd[1:])}
    pw = Power(); pw.start()
    try:
        ok = False
        for _ in range(240):
            if proc.poll() is not None:
                break
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).status == 200:
                    ok = True
                    break
            except Exception:
                time.sleep(1)
        res["healthy"] = ok
        if not ok:
            res["server_log_tail"] = open(log_path).read()[-1500:]
            return res
        q = "Question: What is the capital of France?\nAnswer:"
        t = time.time()
        c = http(f"http://127.0.0.1:{port}/completion",
                 {"prompt": q, "n_predict": 24, "temperature": 0, "n_probs": 5, "cache_prompt": False})
        res["completion_wall_s"] = round(time.time() - t, 3)
        res["completion_text"] = c.get("content")
        res["timings"] = c.get("timings")
        probs = c.get("completion_probabilities") or []
        res["n_prob_entries"] = len(probs)
        res["first_prob_entry"] = probs[0] if probs else None
        t = time.time()
        ch = http(f"http://127.0.0.1:{port}/v1/chat/completions",
                  {"messages": [{"role": "user", "content": "Capital of France? One word."}],
                   "max_tokens": 8, "temperature": 0, "logprobs": True, "top_logprobs": 5})
        res["chat_wall_s"] = round(time.time() - t, 3)
        lp = (ch["choices"][0].get("logprobs") or {}).get("content")
        res["chat_text"] = ch["choices"][0]["message"]["content"]
        res["chat_logprob_tokens"] = len(lp) if lp else 0
        res["chat_first_logprob"] = lp[0] if lp else None
        # greedy determinism: same prompt twice must give identical text + logprob
        c2 = http(f"http://127.0.0.1:{port}/completion",
                  {"prompt": q, "n_predict": 24, "temperature": 0, "n_probs": 5, "cache_prompt": False})
        res["greedy_deterministic_text"] = c2.get("content") == c.get("content")
        # throughput over a longer generation
        t = time.time()
        g = http(f"http://127.0.0.1:{port}/completion",
                 {"prompt": "Write a long story about a lighthouse keeper.", "n_predict": 200, "temperature": 0,
                  "cache_prompt": False})
        res["gen200_wall_s"] = round(time.time() - t, 3)
        res["gen200_timings"] = g.get("timings")
    except Exception as e:
        res["error"] = repr(e)
        res["server_log_tail"] = open(log_path).read()[-1500:]
    finally:
        pw.stop = True
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
        res["power_samples_n"] = len(pw.samples)
        res["power_samples_head"] = pw.samples[:3]
        res["power_samples_peak"] = max(pw.samples, key=lambda s: s[1], default=None)
    return res


RES["gpu"] = gpu_info()
step("build", build)
step("download", download)
if os.path.exists(BIN):
    step("small_default", lambda: serve_and_probe("small.gguf", [], "small_default", 8081))
    step("small_kvq8_fa", lambda: serve_and_probe("small.gguf", ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"], "small_kvq8", 8082))
    step("small_kvq8_fa_oldsyntax", lambda: serve_and_probe("small.gguf", ["-fa", "-ctk", "q8_0", "-ctv", "q8_0"], "small_kvq8_old", 8083)
         if not RES["steps"].get("small_kvq8_fa", {}).get("healthy") else {"skipped": "new syntax worked"})
    step("large_default", lambda: serve_and_probe("large.gguf", [], "large_default", 8084))
for f in ("small.gguf", "large.gguf"):
    p = os.path.join(W, f)
    if os.path.exists(p):
        os.remove(p)  # keep output dir small; binary + logs are the artifacts
shutil.rmtree(SRC, ignore_errors=True)
save()
# Kaggle output files are not downloadable from the Sieve sandbox (kaggleusercontent.com is blocked),
# so the full results JSON is also emitted into the log, which the API returns.
print("RESULT_JSON_BEGIN")
print(json.dumps(RES))
print("RESULT_JSON_END")
print("DONE", json.dumps({"errors": RES["errors"]}))
