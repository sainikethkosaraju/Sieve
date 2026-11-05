# Provenance of the serving-demo evidence

Kaggle kernel `<kaggle-user>/sieve-serve-demo-2` (2026-11-04, 2x Tesla T4, 77 s for the demo stage). An earlier launch
(`...-demo-1`) failed on a missing library (`pydantic-settings`) before serving anything.
Artifact: the one the v1 harness certified on split 0 (`gate:primary`, alpha = delta = 0.10, artifact id
8009f7d54c507d93), served by the deployed server application over ASGI in-process, with the harness's own
backends (Qwen2.5-1.5B and 7B Q4_K_M, llama.cpp fac3a89). 300 fresh TriviaQA validation questions that the
experiment's 6,000-item sample never contained. Aggregates only: no item text.
