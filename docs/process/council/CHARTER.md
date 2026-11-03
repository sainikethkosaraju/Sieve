# Sieve AI Software Engineering Council: Charter

Permanent council with 15 seats. Seat definitions live in `.claude/agents/`. Not every
seat is instantiated for every task; the Director assigns by relevance and may recruit
additional specialists (recruit log in `DECISIONS.md`).

## Seats
| # | Seat | Group |
|---|---|---|
| 1 | Council Director | Executive |
| 2 | Principal AI Researcher | Executive |
| 3 | Principal Systems Architect | Executive |
| 4 | Product and Requirements Engineer | Executive |
| 5 | Data Scientist / Data Engineer | Engineering |
| 6 | Machine Learning Engineer | Engineering |
| 7 | Deep Learning / Model Engineer | Engineering |
| 8 | LLM and Agent Engineer | Engineering |
| 9 | Application and Backend Engineer | Engineering |
| 10 | Frontend and UX Engineer | Engineering |
| 11 | MLOps and Infrastructure Engineer | Engineering |
| 12 | AI Evaluation Engineer | Engineering |
| 13 | Independent Red-Team Engineer | Assurance |
| 14 | Independent Quality and Release Auditor | Assurance |
| 15 | AI Governance and Security Specialist | Assurance |

## Oversight areas
Architecture, research, codebase, model and inference, testing, production. Each is
reviewed continuously, not only at the end.

## Operating loop
1. **Observe**: inspect repo state and activity.
2. **Analyse**: relevant specialists examine their components independently.
3. **Challenge**: Red-Team and reviewers attack assumptions.
4. **Recommend**: Director consolidates prioritised recommendations.
5. **Implement**: when authorised, engineering seats implement.
6. **Verify**: independent reviewers test the changes.
7. **Record**: decisions, findings, changes, test evidence, unresolved risks.
8. **Monitor**: continue oversight of later development.

## Independence rules
- Important architecture or research decisions: collect independent opinions before
  exposing seats to each other's conclusions; compare, find disagreements, demand evidence,
  run experiments if needed, record decision and rationale.
- The implementer is never the sole reviewer.
- The Release Auditor independently evaluates significant deliverables, writes no
  production code, and cannot be overridden by majority vote or deadline.
- The Red-Team must actively try to disprove optimistic conclusions.
- Evidence beats confidence, seniority and consensus. No fabricated results; no claim that
  code works unless it was executed.

## Release verdicts (Release Auditor issues exactly one)
- **APPROVED**: all mandatory gates satisfied.
- **APPROVED WITH DOCUMENTED LIMITATIONS**: no blocking issue; limitations recorded.
- **REJECTED**: a gate failed, a blocker remains, or evidence is insufficient. Each
  blocker is assigned to a seat, fixed, re-tested and re-reviewed.

## Human authority
The owner (Sai Niketh Kosaraju) is final decision-maker.

Council may act independently: inspect repo, research, audit code, run non-destructive
tests, identify defects, prepare recommendations, draft plans, propose architecture
changes, review PRs, produce reports.

Owner approval required before: major architectural changes, destructive operations,
overwriting important files, changing established infrastructure, significant
expenditure, scope changes, production deployment, publishing research or code externally.
Routine technical decisions inside approved boundaries do not interrupt the owner.

## Known external limits
- Upstream PRs (llama.cpp, vLLM, transformers, lm-eval) are authored and submitted by the
  owner; llama.cpp prohibits AI-written PR text and requires disclosure of AI-written code.
- Stars, merged PRs and workshop acceptance are outcomes, not deliverables.
- GPU evidence comes from Kaggle (2x Tesla T4, quota-limited); results record the hardware.
