# Release audit: claude/great-babbage-a6srla @ bfed6e0 -> main (D-018)

Auditor: Independent Quality and Release Auditor. Read-only; nothing edited except this file.

## Gates (run by auditor)
- ruff check: pass. ruff format --check: 119 files formatted. mypy: no issues in 119 files.
- pytest: 548 tests collected, full run exit 0, no failures (fixwave-notes.md:25 claims a failing stale-script test; it passes now, note is stale).

## Pre-registration
- `git diff ca3ed07 HEAD -- docs/PREREGISTRATION.md`: single hunk, only "None yet." replaced by dated Deviations 1-7. Frozen part byte-identical. Deviations honest (3B->1.5B licence, prompt v2, threshold path, 4-bit KV / Q2_K failures, SimpleQA ungraded).

## Claims vs evidence (re-derived 20+ numbers from hypotheses.json / results.json / sanity_scrubbed.json)
Confirmed: H1 5/50, CP [0.0333, 0.2181], mean risk 0.074, max 0.121, cov 25.8%; H2 min +0.034 (0.025-0.044), mean +0.029; H3 0.9008 vs 0.6828 s, diff CI 0.203-0.235, 50-split 1.335 vs 0.792; H4 -0.054 (-0.089,-0.019), 0.3345/0.1916 = 1.75x; H5 NQ risk 0.422, 50/50, CI [0.929,1.0], alarm 10/10; H7 0.792/0.837 with CIs; accuracy table (0.366/0.559, 0.120/0.196, 0.595/0.743); H6 rows; stop rates 99.59%/99.96%; Q2_K 86.0% stop (14% unterminated); KV q4 stop 2.7% (97% unterminated); determinism 600/0/0.0; all energy blocks throttled.

Defects:
1. BLOCKING (policy/hygiene): `results/evidence/t4_main_2026-11-03/driver_summary.md:71-142` commits 63 real item records (question text, gold answers, raw generations, e.g. line 71 `Q=[item text removed]`). This contradicts `results/evidence/t4_main_2026-11-03/PROVENANCE.md:9`, `docs/RESULTS.md:5`, README "Models and data" ("item text and generations are not redistributed"), and the D-018 criterion. Fix: delete the "Output samples" section (lines ~69-142), re-check other files for `Q='`/`gold=`; note the text remains in history of any commit that contains it (see owner decisions).
2. README.md:~28 (quantization row) and docs/RESULTS.md H6 bullet: "thresholds fitted at f16 still kept risk <= 10% on every quantized configuration" is mislabelled and overclaimed. The number is the frequency of split-level violations (0.041-0.082), not risk. It is vacuous for KV q4 (accuracy 0.0, S never answers; the table itself says "meaningless"), and the recalibrated KV q4 violation frequency is 0.12 (hypotheses.json H6 kv_q4), above 0.10. Fix: "violation frequency across 50 re-splits was <= 0.082 for the five configs that functioned; not meaningful for KV q4".
3. docs/RESULTS.md ("Stop handling" bullet): "token/text mismatch rate below 0.03%"; sanity_scrubbed.json gates.tiers.S.mismatch_rate = 0.000381 (0.038%). Fix to "below 0.04%".
4. README "about 4% of queries" is the 50-split mean (1 - escalation 0.95926 = 4.07%). On the canonical split 0 the certified gate has t1 = null, i.e. S answered 0%. RESULTS H3 row mixes the two. Say "about 4% averaged over 50 re-splits; none on split 0".
5. README/RESULTS "4-bit KV cache destroyed the 1.5B model" / "word-salad output": accuracy 0.0 and 97% length-stops are confirmed, but "word-salad" is not backed by any committed evidence (sample rows scrubbed), and a q4_0 KV + flash-attn failure on this build may be an implementation/config artefact rather than a model property. Soften to "failed completely under this llama.cpp build/config".
6. H6 table mixes accuracy sources: Q8_0 0.379, Q3_K_M 0.310, Q2_K 0.145 come from the quant analysis subset, while the full TriviaQA run gives 0.383, 0.318, 0.1495 (sanity_scrubbed.json); Q4_K_M 0.366 matches both. Label the column's basis.
7. Evidence gap: `sanity_scrubbed.json` server self_test `utf8_reassembles: false` for both L cells; not mentioned in README/RESULTS/fixwave-notes. Document or explain.
8. RESULTS H7 cites "arXiv 2610.23249" as a near-chance result; not verifiable here and docs/research is declared partly unverified. Mark as unverified citation or drop.

## Hygiene
- Secrets: git grep across all 32 commits for KGAT_, hf_, ghp_, github_pat_, sk-, AKIA, private keys: only the redaction regex (src/sieve/observe/redact.py:17) and a documented fake key in tests/observe/test_observe.py:260. No secrets.
- Personal data: commit authors are the "Claude" noreply identity and the owner's GitHub noreply address; Kaggle handle `ksainiketh` appears in PROVENANCE/smoke JSON (owner decision, not a secret). No local paths beyond Kaggle paths.
- No datasets, gguf, env files committed; results are aggregates except defect 1. .gitignore sane.
- NOTICE/LICENSE/CITATION present; 3B explicitly not used; AI disclosure present on README front page; docs/research labelled "AI-assisted, partly unverified". Relative-link check over all tracked .md: 0 broken.
- Limitations are documented, not hidden: README "Not established", RESULTS Limitations, fixwave-notes (open audit items), untested server under GPU load and Docker builds.

## Non-blocking recommendations
- Regenerate/rewrite fixwave-notes.md line 25 (stale). Add H1/H5 caveat that re-splits are dependent (already present). Consider removing `ksainiketh` paths from the smoke JSON.

## Owner decisions needed
- History rewrite vs accept: defect 1 text exists in branch history; once on public main it is permanent unless history is rewritten before merge (squash-merge would avoid it).
- Authorship of final commits (31 of 32 are authored "Claude <noreply@anthropic.com>"; README discloses co-authorship). Squash author choice.
- Delete the public feature branch origin/claude/great-babbage-a6srla after merge.

Re-audit scope after fixes: items 1-8 only (cheap text edits; no re-run needed).

VERDICT: REJECTED

---
## Re-review at HEAD 4b1918a
Gates: ruff check pass; ruff format --check 119 files OK; mypy no issues (119 files); pytest 548 collected, full run rc=0, no failures. Relative links: 0 broken. PREREGISTRATION.md vs ca3ed07: only "None yet." removed (Deviations section); frozen part byte-identical.

1. FIXED. driver_summary.md:70-72 sample rows replaced by a removal note; git grep for gold=, Q=', raw=' over results/, docs/, README finds only that heading; no item-text keys in results JSON/CSV.
2. FIXED. README.md:24 and RESULTS.md:46,56-58: "violation frequency (share of re-splits with test risk above 0.10)", at most 0.082, KV q4 "not meaningful", recalibrated 0.12 disclosed.
3. FIXED. RESULTS.md:70 "at most 0.04% (small 0.038%, large 0%)".
4. FIXED. README.md:22 and RESULTS.md:18 "about 4% on average across 50 re-splits (none on canonical split 0)".
5. FIXED. RESULTS.md:62-63 and README.md:24: logged-not-committed samples, "may be specific to this build".
6. FIXED. RESULTS.md:46 column labelled "split-0 test slice".
7. FIXED. RESULTS.md:71-72 UTF-8 probe advisory note (upstream issue 31451).
8. FIXED/see RESULTS.md:31 arXiv attribution reworded (no longer asserted as established fact in the audited text).

Owner decisions (unchanged plus): item-text records remain in history of earlier feature-branch commits; main will receive one squashed commit, which removes them from main. Choose squash author; delete the feature branch only after confirming nothing else needs its history (it still contains the item text, so deleting it is advisable).

VERDICT: APPROVED WITH DOCUMENTED LIMITATIONS
