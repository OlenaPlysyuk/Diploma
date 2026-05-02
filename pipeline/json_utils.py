import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _s(v: Any) -> str:
    return "" if v is None else str(v).strip()


def _list(v: Any) -> List[Any]:
    return v if isinstance(v, list) else []


def parse_date_ymd(s: str) -> datetime:
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except Exception:
        return datetime.min


def strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def extract_first_json_object(text: str) -> str:
    t = strip_code_fences(text).replace("\r\n", "\n")
    t = re.sub(r"<think>.*?</think>\s*", "", t, flags=re.S)

    first_brace = t.find("{")
    if first_brace == -1:
        return ""
    t = t[first_brace:]

    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(t):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[: i + 1]

    return ""


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(strip_code_fences(text))
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    extracted = extract_first_json_object(text)
    if not extracted:
        return None

    try:
        obj = json.loads(extracted)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


def shrink_text(text: str, limit: int) -> str:
    text = _s(text)
    if limit <= 0 or len(text) <= limit:
        return text
    head = max(limit - 160, 0)
    return text[:head] + "\n...[truncated for prompt size]...\n"


def compact_judge_feedback_for_prompt(judge: Optional[Dict[str, Any]], max_chars: int = 4000) -> str:
    if not isinstance(judge, dict):
        return "{}"

    compact = {
        "score": judge.get("score"),
        "verdict": judge.get("verdict"),
        "issues": _list(judge.get("issues"))[:3],
        "recommendations": _list(judge.get("recommendations"))[:4],
        "revision_brief": judge.get("revision_brief", {}),
    }
    return shrink_text(json.dumps(compact, ensure_ascii=False), max_chars)


def compact_strategy_for_prompt(strategy: Optional[Dict[str, Any]], max_chars: int = 6500) -> str:
    if not isinstance(strategy, dict):
        return "{}"
    return shrink_text(json.dumps(strategy, ensure_ascii=False), max_chars)
