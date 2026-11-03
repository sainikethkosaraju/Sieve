"""F-01 / D-019: an artifact certified through the eval path is servable by the router."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sieve import prompting
from sieve.calibrate.store import loads_artifact
from sieve.eval import prompts
from sieve.router import ChatMessage, ChatTemplate
from sieve.server import create_app
from sieve.types import CalibrationArtifact
from tests.server.harness import bs, make_settings, running

pytestmark = pytest.mark.slow


def _certified_artifact(wd: Path) -> CalibrationArtifact:
    cert = json.loads((wd / "certs" / "split_000.json").read_text())
    for methods in cert["certs"].values():
        for c in methods.values():
            if c.get("artifact") is not None and c["outcome"] == "feasible":
                return loads_artifact(json.dumps(c["artifact"]))
    pytest.skip("synthetic split 0 certified no deployable artifact")


def _app(art: CalibrationArtifact, template: str) -> Any:
    def side(i: int) -> Any:
        ident = art.tiers[i].identity
        return bs(ident.model_id).model_copy(
            update={
                "model_hash": ident.model_hash,
                "quant": ident.quant,
                "kv_quant": ident.kv_quant,
                "build": ident.build,
            }
        )

    settings = make_settings(small=side(0), large=side(1), chat_template=template)
    return create_app(settings, artifact=art)


def test_eval_and_server_share_prompt_and_hash() -> None:
    assert prompts.template_hash() == prompting.template_hash() == ChatTemplate().hash
    assert len(prompting.template_hash()) == 64
    rendered = ChatTemplate().render([ChatMessage("user", "Who painted the Mona Lisa?")])
    assert rendered == prompting.render_calibration_prompt("Who painted the Mona Lisa?")


@pytest.mark.anyio
async def test_eval_certified_artifact_is_accepted_by_server(
    mock_run: tuple[Path, dict[str, object]],
) -> None:
    art = _certified_artifact(mock_run[0])
    assert art.tiers[0].identity.template_hash == prompting.template_hash()
    async with running(_app(art, "calibrated")) as c:
        r = await c.get("/readyz")
        assert r.status_code == 200, r.text
        assert r.json()["tiers"]["1"]["identity_matches_artifact"] is True


@pytest.mark.anyio
async def test_server_rejects_artifact_when_template_differs(
    mock_run: tuple[Path, dict[str, object]],
) -> None:
    art = _certified_artifact(mock_run[0])
    async with running(_app(art, "chatml")) as c:
        r = await c.get("/readyz")
        assert r.status_code == 503
        assert r.json()["tiers"]["1"]["identity_matches_artifact"] is False
