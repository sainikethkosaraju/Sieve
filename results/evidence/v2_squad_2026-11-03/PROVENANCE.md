# Provenance of the v2 evidence

Kaggle kernel `<kaggle-user>/sieve-v2-squad-main-2` (2026-11-03, 2x Tesla T4; generation 68 min). An earlier launch
(`...-main-1`) failed after 32 s because the script builder did not embed the v2 pre-registration; it produced no
generations. Config `benchmarks/t4_squad.yaml`, pre-registration `docs/PREREGISTRATION_V2.md` (frozen at commit
bd276d1 before any real v2 generation). llama.cpp commit fac3a894fc04901c3e5abaff457e024c6b52e04f.
Models: Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct Q4_K_M (Apache-2.0). Data: SQuAD 2.0 validation (CC BY-SA 4.0),
6,000 items. Aggregates only: item text, passages, gold answers and generations are not redistributed.
The generic hypothesis report refused to run (its guard compares v1 coded criteria with the v1 document), so no
hypotheses.json exists; the v2 verdict applies the frozen v2 rule to analysis/results.json.
