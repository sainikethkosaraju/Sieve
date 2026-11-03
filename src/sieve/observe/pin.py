"""Config / artifact version pinning (replaces canary & rollback, decision D-010).

``config_pin`` is the SHA-256 of the canonical JSON of the effective, secret-free
settings. An operator can set ``expected_config_pin`` / ``expected_artifact_id`` in the
deployment manifest; a mismatch at startup refuses to serve. The pin is stamped into
every audit record and exported as ``sieve_build_info``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


class PinMismatchError(RuntimeError):
    pass


def config_pin(settings: Mapping[str, Any]) -> str:
    body = json.dumps(settings, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def check_pins(
    *,
    config: str,
    artifact_id: str,
    expected_config: str | None,
    expected_artifact: str | None,
) -> None:
    if expected_config is not None and expected_config != config:
        raise PinMismatchError(f"config pin mismatch (have {config}, expected {expected_config})")
    if expected_artifact is not None and expected_artifact != artifact_id:
        raise PinMismatchError(
            f"artifact pin mismatch (have {artifact_id}, expected {expected_artifact})"
        )
