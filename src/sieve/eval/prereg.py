"""Pre-registration pinning.

``docs/PREREGISTRATION.md`` is frozen above its ``## Deviations`` heading. The SHA-256 of that
frozen body is recorded in every run manifest and in the experiment config; the hypothesis
report refuses to run when it differs from either. Dated Deviations appended below the
heading do not change the hash but are printed with every report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sieve.eval._util import repo_root, sha256_text

DEVIATIONS_HEADING = "## Deviations"
DEFAULT_PATH = Path("docs/PREREGISTRATION.md")


class PreregistrationChangedError(RuntimeError):
    """The frozen pre-registration differs from the hash the data was collected under."""


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


def split_prereg(text: str) -> tuple[str, str]:
    """(frozen body, deviations section). The heading starts the deviations part."""
    idx = text.find("\n" + DEVIATIONS_HEADING)
    if idx < 0:
        return text, ""
    return text[: idx + 1], text[idx + 1 :]


def _normalise(s: str) -> str:
    return "\n".join(line.rstrip() for line in s.replace("\r\n", "\n").strip().split("\n"))


def frozen_hash(path: str | Path) -> str:
    body, _ = split_prereg(resolve(path).read_text(encoding="utf-8"))
    return sha256_text(_normalise(body))


def deviations_text(path: str | Path) -> str:
    _, dev = split_prereg(resolve(path).read_text(encoding="utf-8"))
    return dev.strip()


def verify_unchanged(path: str | Path, *expected: str | None) -> str:
    """Return the current frozen hash; raise if it differs from any non-None ``expected``."""
    current = frozen_hash(path)
    for exp in expected:
        if exp is not None and exp != current:
            raise PreregistrationChangedError(
                f"pre-registration changed: frozen-body hash is {current[:16]}..., "
                f"expected {exp[:16]}...; record any change as a dated Deviation below "
                f"'{DEVIATIONS_HEADING}' instead of editing the frozen text"
            )
    return current


@dataclass(frozen=True)
class Criteria:
    """Decision thresholds copied from the pre-registration (see ``CRITERIA_PATTERNS``)."""

    h1_pass_max: float = 0.15
    h1_fail_min: float = 0.25
    h2_delta_auroc: float = 0.02
    h4_delta_auroc: float = 0.03
    h4_k: int = 5
    h5_alarm_rate: float = 0.80
    h5_window: int = 200
    h6_margin: float = 0.02
    h7_auroc: float = 0.6


CRITERIA = Criteria()

#: Regex that must match the pre-registration text for each coded number (drift guard).
CRITERIA_PATTERNS: dict[str, str] = {
    "h1_pass_max": r"<=\s*15%",
    "h1_fail_min": r">\s*25%",
    "h2_delta_auroc": r"by\s*>=\s*0\.02",
    "h4_delta_auroc": r">=\s*0\.03\s*AUROC",
    "h4_k": r"k=5",
    "h5_alarm_rate": r">=\s*80%",
    "h5_window": r"200-query",
    "h6_margin": r"\+-0\.02",
    "h7_auroc": r"AUROC\s*<\s*0\.6",
    "alpha": r"alpha\s*=\s*0\.10",
    "delta": r"delta\s*=\s*0\.10",
    "bootstrap": r"\(10,000\)",
}


def criteria_missing(path: str | Path) -> list[str]:
    """Names of coded criteria that no longer appear in the frozen pre-registration text."""
    body, _ = split_prereg(resolve(path).read_text(encoding="utf-8"))
    return [k for k, pat in CRITERIA_PATTERNS.items() if not re.search(pat, body)]
