"""Diagnostic kernel: dump raw llama-server /completion responses (stop vs no stop, UTF-8).

Standalone (stdlib only). Prints RAW_BEGIN/RAW_END JSON. Used to debug the integration run
(count_matches_tokens failed on the real server, Phase C notes)."""
import glob, json, os, shutil, subprocess, time, urllib.request

os.makedirs("/tmp/m", exist_ok=True)
src = glob.glob("/kaggle/input/**/llama-server", recursive=True)[0]
shutil.copy(src, "/tmp/ls"); os.chmod("/tmp/ls", 0o755)
m = "/tmp/m/s.gguf"
subprocess.run(["curl", "-sSL", "--fail", "-o", m, "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"], check=True)
log = open("/tmp/srv.log", "w")
p = subprocess.Popen(["/tmp/ls", "-m", m, "-ngl", "99", "-c", "2048", "--port", "8199", "--host", "127.0.0.1", "-np", "1"], stdout=log, stderr=subprocess.STDOUT)
for _ in range(240):
    try:
        if urllib.request.urlopen("http://127.0.0.1:8199/health", timeout=2).status == 200: break
    except Exception: time.sleep(1)

def post(body):
    r = urllib.request.Request("http://127.0.0.1:8199/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=120).read())

def slim(d):
    cp = d.get("completion_probabilities")
    out = {k: v for k, v in d.items() if k not in ("completion_probabilities", "generation_settings", "prompt")}
    out["keys"] = sorted(d.keys())
    out["n_cp"] = None if cp is None else len(cp)
    out["cp_tokens"] = None if cp is None else [(e.get("token"), round(e.get("logprob", 0), 3), len(e.get("top_logprobs", []))) for e in cp]
    out["cp0_keys"] = None if not cp else sorted(cp[0].keys())
    return out

res = {}
Q = "Question: What is the capital of France?\nAnswer:"
U = "Repeat exactly: naïve café 日本語 🙂\nRepeat exactly: naïve café 日本語 🙂\n"
FS = "Answer each question with a short phrase only, without explanation.\n\nQuestion: What is the chemical symbol for gold?\nAnswer: Au\n\nQuestion: In which country is the city of Kyoto?\nAnswer: Japan\n\nQuestion: Who painted the Mona Lisa?\nAnswer:"
for name, prompt, extra in [("q_nostop", Q, {}), ("q_stop", Q, {"stop": ["\n"]}), ("fs_stop", FS, {"stop": ["\n"]}), ("fs_nostop", FS, {}), ("utf8_stop", U, {"stop": ["\n"]}), ("utf8_nostop", U, {})]:
    try:
        res[name] = slim(post({"prompt": prompt, "n_predict": 12, "temperature": 0, "n_probs": 20, "cache_prompt": False, "stream": False, **extra}))
    except Exception as e:
        res[name] = {"error": repr(e)}
res["props_keys"] = sorted(json.loads(urllib.request.urlopen("http://127.0.0.1:8199/props").read()).keys())
p.terminate()
print("RAW_BEGIN"); print(json.dumps(res, ensure_ascii=False, indent=1)); print("RAW_END")
