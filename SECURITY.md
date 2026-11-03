# Security policy

Sieve is pre-release software (0.1.0.dev0). Only the latest commit on the default branch
is supported.

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub's "Report a vulnerability"
(private security advisory) feature on this repository, not in a public issue. Include a
description, affected version or commit, and reproduction steps if you have them. Expect an
acknowledgement within 7 days; this is a volunteer-maintained project and no fixed fix
timeline is promised.

## Scope and design stance

- The library packages (`signals`, `calibrate`, `gate`) perform no network IO.
- The gate fails closed: logprob defects, sampling mismatches and calibration-artifact
  identity mismatches never produce an `ANSWER`.
- Calibration artifacts are JSON with a SHA-256 content hash, verified on load (no pickle).
  The hash detects accidental or naive tampering; it is not a signature and does not
  authenticate the artifact's origin.
- The server (`sieve.server`) implements: an allow-listed backend URL (no client-supplied backend URLs),
  header and request-id sanitising, size and deadline limits with 429 backpressure, cancellation on client
  disconnect, and privacy-by-default logging (prompts, responses and API keys are not written to the audit log,
  metrics or server log unless content logging is explicitly enabled). These were exercised by an adversarial
  audit against a mock backend (`docs/process/council/reviews/redteam-code-audit.md`).
- Known gaps (see `docs/process/council/reviews/fixwave-notes.md`): header-level slowloris protection is not
  provided by the server settings, so run it behind a reverse proxy anywhere other than loopback; backend identity
  is re-checked periodically rather than per request; artifacts are integrity-hashed, not signed. It has not been
  load-tested against a real GPU backend.
- Report vulnerabilities through GitHub's "Report a vulnerability" (private security advisory) on this
  repository if it is enabled; otherwise open an issue asking for a private channel, without exploit details.
