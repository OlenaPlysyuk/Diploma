import json
import os
from pathlib import Path

import pytest
import requests

from pipeline.llm_clients import call_lmstudio_chat, call_openai_chat, parse_openai_judge_json


pytestmark = pytest.mark.integration


def _integration_enabled() -> bool:
    return os.environ.get("RUN_LLM_TESTS") == "1"


def _lmstudio_model(host: str) -> str:
    explicit = os.environ.get("LM_STUDIO_MODEL", "").strip()
    if explicit:
        return explicit

    try:
        response = requests.get(host.rstrip("/") + "/v1/models", timeout=5)
    except requests.RequestException as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("sandbox cannot access localhost; run this test from your terminal")
        raise
    response.raise_for_status()
    data = response.json()
    models = data.get("data") or []
    if not models:
        pytest.skip("LM Studio is running, but /v1/models returned no loaded models.")
    return str(models[0]["id"])


@pytest.mark.skipif(not _integration_enabled(), reason="set RUN_LLM_TESTS=1 to run live LLM tests")
def test_lmstudio_chat_completion_smoke_test():
    host = os.environ.get("LM_STUDIO_HOST", "http://localhost:1234")
    model = _lmstudio_model(host)

    text = call_lmstudio_chat(
        host=host,
        model=model,
        system="Reply with exactly: PONG. No markdown.",
        user="ping",
        temperature=0.0,
        max_tokens=16,
        timeout_sec=30,
        retries=0,
        debug_label="pytest_lmstudio",
    )

    assert isinstance(text, str)


@pytest.mark.skipif(not _integration_enabled(), reason="set RUN_LLM_TESTS=1 to run live LLM tests")
def test_openai_judge_json_smoke_test(tmp_path: Path):
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        pytest.skip("OPENAI_API_KEY is not set.")

    model = os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4.1-mini")
    system = "You are a strict JSON judge. Return JSON only."
    user = (
        "Return exactly one JSON object with keys score and verdict. "
        "score must be 10 and verdict must be ok."
    )

    raw = call_openai_chat(
        api_key=api_key,
        model=model,
        system=system,
        user=user,
        temperature=0.0,
        max_tokens=80,
        timeout_sec=60,
        retries=0,
        debug_label="pytest_openai_judge",
    )
    parsed = parse_openai_judge_json(
        raw,
        api_key=api_key,
        model=model,
        system=system,
        user=user,
        temperature=0.0,
        max_tokens=80,
        out_debug_path=tmp_path / "judge_parse_failed.txt",
    )

    assert set(parsed) >= {"score", "verdict"}
    assert str(parsed["verdict"]).lower() == "ok"
    assert float(parsed["score"]) == 10.0
    assert json.dumps(parsed)
