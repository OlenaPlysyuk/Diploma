from datetime import datetime

from pipeline.json_utils import (
    _list,
    _s,
    compact_judge_feedback_for_prompt,
    compact_strategy_for_prompt,
    extract_first_json_object,
    parse_date_ymd,
    safe_json_loads,
    shrink_text,
    strip_code_fences,
)


def test_basic_value_helpers():
    assert _s(None) == ""
    assert _s("  value  ") == "value"
    assert _list([1, 2]) == [1, 2]
    assert _list("not-list") == []


def test_parse_date_ymd_valid_and_invalid():
    assert parse_date_ymd("2024-02-03") == datetime(2024, 2, 3)
    assert parse_date_ymd("bad-date") == datetime.min


def test_strip_code_fences_and_extract_first_json_object():
    fenced = '```json\n{"a": 1}\n```'
    assert strip_code_fences(fenced) == '{"a": 1}'

    mixed = '<think>{"ignored": true}</think> prefix {"a": {"b": 2}} suffix'
    assert extract_first_json_object(mixed) == '{"a": {"b": 2}}'


def test_safe_json_loads_handles_direct_extracted_and_invalid_values():
    assert safe_json_loads('{"ok": true}') == {"ok": True}
    assert safe_json_loads('prefix {"ok": true} suffix') == {"ok": True}
    assert safe_json_loads("[1, 2, 3]") is None
    assert safe_json_loads("no json here") is None


def test_shrink_and_compact_prompt_helpers():
    text = "x" * 300
    assert shrink_text(text, 220).startswith("x")
    assert "[truncated for prompt size]" in shrink_text(text, 220)
    assert shrink_text("short", 80) == "short"

    judge = {
        "score": 8,
        "verdict": "ok",
        "issues": [1, 2, 3, 4],
        "recommendations": ["a", "b", "c", "d", "e"],
        "revision_brief": {"change": ["tighten claims"]},
    }
    compact_judge = compact_judge_feedback_for_prompt(judge)
    assert '"score": 8' in compact_judge
    assert '"revision_brief"' in compact_judge
    assert compact_judge_feedback_for_prompt(None) == "{}"

    compact_strategy = compact_strategy_for_prompt({"product": {"asin": "A1"}})
    assert '"asin": "A1"' in compact_strategy
    assert compact_strategy_for_prompt(None) == "{}"
