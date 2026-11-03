"""Stand-in ``llama-server`` for process-manager and provider tests (stdlib only, no GPU).

It accepts the real CLI flags the provider passes (``-m -ngl -c --host --port -np -fa -ctk
-ctv``), serves ``/health`` ``/props`` ``/completion`` with the REAL response schema (nested
``top_logprobs``, ``tokens_predicted``, ``timings``) and honours ``stop``. Behaviour switches
(environment): ``FAKE_LLAMA_LOAD_S`` seconds until healthy, ``FAKE_LLAMA_DIE`` exit code at
start, ``FAKE_LLAMA_STOP_MODE`` (``drop`` default = real llama.cpp,
``straddle``, ``keep``), ``FAKE_LLAMA_BUILD`` the ``build_info`` string,
``FAKE_LLAMA_NAN`` make chosen logprobs NaN (a logprob defect), ``FAKE_LLAMA_NONDET`` jitter.
Not a model: answers are canned from a hash of the prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

ARGS: argparse.Namespace
STATE: dict[str, Any] = {"calls": 0}
WORDS = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]


def answer_for(prompt: str) -> str:
    if "Repeat exactly" in prompt:
        return " naïve café 日本語 🙂"
    if "capital of France" in prompt:
        return " Paris"
    h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
    return " " + WORDS[h % len(WORDS)]


def pieces(text: str) -> list[str]:
    out: list[str] = []
    for w in text.split(" "):
        if w:
            out.append(" " + w if out or text.startswith(" ") else w)
    return out or [text]


def entry(tok: str, lp: float, top: list[tuple[str, float]]) -> dict[str, Any]:
    b = list(tok.encode())
    return {
        "id": abs(hash(tok)) % 150000,
        "token": tok,
        "bytes": b,
        "logprob": lp,
        "top_logprobs": [
            {"id": abs(hash(t)) % 150000, "token": t, "bytes": list(t.encode()), "logprob": x}
            for t, x in top
        ],
    }


def build_probs(toks: list[str], n_probs: int) -> list[dict[str, Any]]:
    out = []
    for i, tok in enumerate(toks):
        lp = -0.05 - 0.01 * i
        if os.environ.get("FAKE_LLAMA_NAN"):
            lp = math.nan
        alts = [(f" alt{j}", -8.0 - 0.2 * j) for j in range(max(n_probs - 1, 0))]
        top = [(tok, lp), *alts][: max(n_probs, 1)]
        out.append(entry(tok, lp, top))
    return out


def complete(body: dict[str, Any]) -> dict[str, Any]:
    """Models llama.cpp @ fac3a89 (measured on a T4): a matched stop string is counted in
    ``tokens_predicted`` but its entry is missing from ``completion_probabilities``; with
    FAKE_LLAMA_STOP_MODE=straddle the stop token is ``".\\n"`` so ``content`` keeps a ``"."``
    that has no entry; ``keep`` keeps the stop token's entry (other servers)."""
    STATE["calls"] += 1
    prompt = str(body["prompt"])
    n_predict = int(body.get("n_predict", 32))
    n_probs = int(body.get("n_probs", 0))
    stop = list(body.get("stop", []))
    mode = os.environ.get("FAKE_LLAMA_STOP_MODE", "drop")
    toks = pieces(answer_for(prompt))
    stop_type = "eos"
    tail: list[str] = []
    content = "".join(toks)
    reported = list(toks)
    predicted = len(toks)
    if "\n" in stop:
        stop_type = "word"
        if mode == "straddle":
            predicted = len(toks) + 1  # ".\n" token, dropped from the probs list
            content = "".join(toks) + "."
        else:
            predicted = len(toks) + 1
            if mode == "keep":
                reported = [*toks, "\n"]
    if len(toks) > n_predict:
        toks = toks[:n_predict]
        reported = list(toks)
        content = "".join(toks)
        predicted = len(toks)
        stop_type = "limit"
    del tail
    probs = build_probs(reported, n_probs) if n_probs > 0 else []
    t0 = time.perf_counter()
    time.sleep(0.002)
    ms = (time.perf_counter() - t0) * 1000
    return {
        "index": 0,
        "content": content,
        "tokens": [],
        "id_slot": 0,
        "stop": True,
        "model": os.path.basename(ARGS.m),
        "tokens_predicted": predicted,
        "tokens_evaluated": max(1, len(prompt) // 4),
        "stop_type": stop_type,
        "stopping_word": "\n" if stop_type == "word" else "",
        "truncated": False,
        "completion_probabilities": probs,
        "timings": {
            "cache_n": 0,
            "prompt_n": max(1, len(prompt) // 4),
            "prompt_ms": ms,
            "predicted_n": predicted,
            "predicted_ms": ms * 2,
        },
    }


class H(BaseHTTPRequestHandler):
    def log_message(self, *_: object) -> None:
        pass

    def _send(self, code: int, obj: Any) -> None:
        raw = json.dumps(obj, allow_nan=True).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/health":
            if time.time() < STATE["ready_at"]:
                self._send(503, {"error": {"message": "Loading model", "code": 503}})
            else:
                self._send(200, {"status": "ok"})
        elif self.path == "/props":
            self._send(
                200,
                {
                    "build_info": os.environ.get("FAKE_LLAMA_BUILD", "b9999-fac3a89"),
                    "model_path": ARGS.m,
                    "total_slots": ARGS.np,
                    "fake_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "fake_argv": sys.argv[1:],
                },
            )
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/completion":
            self._send(200, complete(body))
        else:
            self._send(404, {"error": "not found"})


def main() -> None:
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", required=True)
    ap.add_argument("-ngl", default="0")
    ap.add_argument("-c", default="0")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("-np", type=int, default=1)
    ap.add_argument("-fa", default="auto")
    ap.add_argument("-ctk", default="f16")
    ap.add_argument("-ctv", default="f16")
    ARGS = ap.parse_args()
    die = os.environ.get("FAKE_LLAMA_DIE")
    if die:
        print("fake llama-server: simulated load failure", flush=True)
        sys.exit(int(die))
    if not os.path.exists(ARGS.m):
        print("fake llama-server: model not found", flush=True)
        sys.exit(2)
    STATE["ready_at"] = time.time() + float(os.environ.get("FAKE_LLAMA_LOAD_S", "0"))
    print("fake llama-server: flash_attn = enabled", flush=True)
    ThreadingHTTPServer((ARGS.host, ARGS.port), H).serve_forever()


if __name__ == "__main__":
    main()
