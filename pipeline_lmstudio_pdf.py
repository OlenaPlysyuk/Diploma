# Project: Diploma_jsontocsv (code)
# File: pipeline_lmstudio_pdf.py

import argparse
import csv
import hashlib
import json
import os
import random
import re
import time
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.exceptions import ReadTimeout, ConnectionError as ReqConnectionError

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, Table, TableStyle
)

# Optional unicode font support (better UA text in PDF)
try:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    _HAS_TT = True
except Exception:
    _HAS_TT = False


# ----------------------------
# Utils
# ----------------------------

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
    """Remove ```json ... ``` wrappers (if any)."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def extract_first_json_object(text: str) -> str:
    """
    Extract the FIRST complete JSON object from mixed model output.
    Handles:
      - <think> ... </think> blocks
      - extra text before/after JSON
      - code fences
    Returns "" if can't find a complete object (truncated).
    """
    t = strip_code_fences(text).replace("\r\n", "\n")

    # Remove closed think blocks
    t = re.sub(r"<think>.*?</think>\s*", "", t, flags=re.S)

    # If output starts with an unclosed <think>..., just take from first "{"
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

    # Truncated JSON
    return ""


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """Try parsing JSON directly; if fails, extract first JSON object and parse that."""
    # 1) direct
    try:
        obj = json.loads(strip_code_fences(text))
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # 2) robust extract
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


def _pdf_unescape_text(text: str) -> str:
    text = text.replace(r"\(", "(").replace(r"\)", ")").replace(r"\n", "\n").replace(r"\r", "\r")
    text = text.replace(r"\t", "\t").replace(r"\/", "/").replace(r"\\", "\\")
    return text


def _extract_pdf_text_from_bytes(data: bytes) -> str:
    chunks: List[str] = []

    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, flags=re.S):
        stream = match.group(1)
        try:
            decoded = zlib.decompress(stream)
        except Exception:
            continue

        try:
            text = decoded.decode("latin-1", errors="ignore")
        except Exception:
            continue

        for txt in re.findall(r"\((.*?)(?<!\\)\)\s*Tj", text, flags=re.S):
            cleaned = _pdf_unescape_text(txt).strip()
            if cleaned:
                chunks.append(cleaned)

        for arr in re.findall(r"\[(.*?)\]\s*TJ", text, flags=re.S):
            parts = re.findall(r"\((.*?)(?<!\\)\)", arr, flags=re.S)
            joined = "".join(_pdf_unescape_text(part) for part in parts).strip()
            if joined:
                chunks.append(joined)

    return "\n".join(chunks)


def load_structure_example(example_path: str) -> Tuple[str, str]:
    if not example_path:
        return "", ""

    path = Path(example_path)
    if not path.exists():
        raise FileNotFoundError(f"Structure example file not found: {path}")

    if path.suffix.lower() == ".pdf":
        content = _extract_pdf_text_from_bytes(path.read_bytes()).strip()
    else:
        content = path.read_text(encoding="utf-8", errors="replace").strip()

    if not content:
        return "", ""

    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return content, digest


def inspect_structure_example(example_path: str) -> Dict[str, Any]:
    content, digest = load_structure_example(example_path)
    preview = shrink_text(content.replace("\r\n", "\n"), 400)
    return {
        "ok": bool(content.strip()),
        "hash": digest,
        "chars": len(content),
        "preview": preview,
    }


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


# ----------------------------
# LM Studio client (OpenAI-like)
# ----------------------------

def call_lmstudio_chat(
    host: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int = 180,
    retries: int = 1,
    debug_label: str = ""
) -> str:
    url = host.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_err = None
    for attempt in range(retries + 1):
        try:
            if debug_label:
                print(
                    f"   → LMStudio API [{debug_label}] attempt {attempt+1}/{retries+1} | "
                    f"prompt_chars={len(system)+len(user)} | max_tokens={max_tokens}"
                )

            r = requests.post(url, json=payload, timeout=timeout_sec)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]

        except ReadTimeout as e:
            last_err = e
            if debug_label:
                print(f"   ✖ ReadTimeout [{debug_label}] after {timeout_sec}s")
            time.sleep(0.6)
        except ReqConnectionError as e:
            last_err = e
            if debug_label:
                print(f"   ✖ ConnectionError [{debug_label}] cannot reach {url}")
            time.sleep(0.6)
        except Exception as e:
            last_err = e
            if debug_label:
                print(f"   ✖ Error [{debug_label}]: {repr(e)}")
            time.sleep(0.6)

    raise RuntimeError(f"LM Studio request failed ({debug_label}): {last_err}")


# ----------------------------
# OpenAI judge client (Chat Completions)
# ----------------------------

def call_openai_chat(
    api_key: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int = 180,
    retries: int = 1,
    debug_label: str = ""
) -> str:
    """
    Uses OpenAI Chat Completions API to match the same response shape:
    choices[0].message.content
    """
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Export it or pass --openai-api-key.")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_err = None
    for attempt in range(retries + 1):
        try:
            if debug_label:
                print(
                    f"   → OpenAI API [{debug_label}] attempt {attempt+1}/{retries+1} | "
                    f"model={model} | prompt_chars={len(system)+len(user)} | max_tokens={max_tokens}"
                )

            r = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]

        except ReadTimeout as e:
            last_err = e
            if debug_label:
                print(f"   ✖ OpenAI ReadTimeout [{debug_label}] after {timeout_sec}s")
            time.sleep(0.8)
        except Exception as e:
            last_err = e
            if debug_label:
                print(f"   ✖ OpenAI Error [{debug_label}]: {repr(e)}")
            time.sleep(0.8)

    raise RuntimeError(f"OpenAI request failed ({debug_label}): {last_err}")


def parse_openai_judge_json(
    judge_raw: str,
    *,
    api_key: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    out_debug_path: Path,
) -> Dict[str, Any]:
    judge_json = safe_json_loads(judge_raw)
    if judge_json is not None:
        return judge_json

    # A common failure mode is truncation from too-small max_tokens. Retry once with a larger budget.
    retry_tokens = max(max_tokens * 2, 2200)
    print(f"   ↻ OpenAI judge JSON parse failed; retrying with max_tokens={retry_tokens}")
    retry_raw = call_openai_chat(
        api_key=api_key,
        model=model,
        system=system,
        user=user,
        temperature=temperature,
        max_tokens=retry_tokens,
        timeout_sec=240,
        retries=1,
        debug_label="judge_retry_json",
    )
    judge_json = safe_json_loads(retry_raw)
    if judge_json is not None:
        return judge_json

    save_text(out_debug_path, retry_raw)
    raise ValueError(f"OpenAI judge returned non-JSON. Saved debug for ASIN={out_debug_path.stem.split('_judge_')[0]}")


def health_check(host: str, model: str) -> None:
    """
    Quick check that server/model responds.
    NOTE: some tool-use models may return <think> or get cut by max_tokens.
    This is only a connectivity check.
    """
    print("🔌 Checking LM Studio:", host)
    txt = call_lmstudio_chat(
        host=host,
        model=model,
        system="Reply with exactly: OK. No <think>. No markdown.",
        user="OK",
        temperature=0.0,
        max_tokens=32,
        timeout_sec=30,
        retries=0,
        debug_label="health_check",
    )
    cleaned = strip_code_fences(txt).strip().replace("\n", " ")
    print("✅ LM Studio replied:", cleaned[:120])


def ensure_valid_json(
    host: str,
    model: str,
    raw_text: str,
    out_debug_path: Path,
    *,
    regen_system: Optional[str] = None,
    regen_user: Optional[str] = None,
    regen_max_tokens: int = 1400
) -> Dict[str, Any]:
    """
    1) Try parse JSON
    2) If invalid: save raw output -> try RE-GENERATE once (better than repair for truncation)
    3) If still invalid: try JSON repair
    """
    obj = safe_json_loads(raw_text)
    if obj is not None:
        return obj

    save_text(out_debug_path, raw_text)

    # 2) Re-generate once (effective when JSON is truncated / has <think>)
    if regen_system and regen_user:
        regen_system2 = (
            regen_system
            + "\n\nIMPORTANT:\n"
              "- Output JSON ONLY.\n"
              "- No <think> blocks.\n"
              "- No markdown.\n"
              "- Ensure the JSON object is complete and closed.\n"
        )
        regen_raw = call_lmstudio_chat(
            host=host,
            model=model,
            system=regen_system2,
            user=regen_user,
            temperature=0.2,
            max_tokens=regen_max_tokens,
            timeout_sec=240,
            retries=1,
            debug_label="regen_json",
        )
        obj2 = safe_json_loads(regen_raw)
        if obj2 is not None:
            return obj2
        save_text(out_debug_path.with_suffix(".regen_failed.txt"), regen_raw)

    # 3) Repair as last resort
    repair_system = "You are a JSON repair tool. Return ONLY valid JSON. No extra text."
    repair_user = (
        "Output ONE valid JSON object only.\n"
        "If the input is truncated, reconstruct missing parts according to the schema.\n\n"
        "TEXT:\n" + strip_code_fences(raw_text)[:12000]
    )
    fixed = call_lmstudio_chat(
        host=host,
        model=model,
        system=repair_system,
        user=repair_user,
        temperature=0.0,
        max_tokens=regen_max_tokens,
        timeout_sec=240,
        retries=1,
        debug_label="json_repair",
    )

    obj3 = safe_json_loads(fixed)
    if obj3 is None:
        save_text(out_debug_path.with_suffix(".repair_failed.txt"), fixed)
        raise ValueError(f"Could not repair JSON. Saved raw to: {out_debug_path}")
    return obj3


# ----------------------------
# Prompts
# ----------------------------

# IMPORTANT: we keep descriptions only as context; evidence_quotes must come from reviews.
GEN_SYSTEM = """You are a senior marketing strategist.
Create a practical marketing strategy using ONLY evidence from customer reviews.

IMPORTANT:
- Output JSON ONLY.
- No <think> blocks.
- No markdown.
- No commentary. Only one JSON object.

Return ONLY valid JSON. No extra text.

JSON schema (must follow exactly):
{
  "product": {"asin": string, "title": string, "brand": string, "category": string},
  "insights": [
    {"type": "strength|weakness|use_case|audience|objection",
     "statement": string,
     "evidence_quotes": [string, string]}
  ],
  "positioning": {"value_proposition": string, "target_audience": string, "key_differentiators": [string]},
  "messaging": {"primary_message": string, "supporting_messages": [string], "tone": string},
  "channels": [{"channel": string, "why": string, "content_ideas": [string]}],
  "offers": [{"offer": string, "why": string}],
  "risks": [{"risk": string, "mitigation": string}],
  "kpis": [{"metric": string, "target": string, "measurement": string}],
  "assumptions": [string]
}

Rules:
- Use only info present in the provided reviews as evidence.
- Each insight must include 1–2 short exact quotes copied from reviews (evidence_quotes).
- Product Description (if provided) is context ONLY. Do NOT use it as evidence or quotes.
- Keep it concise and actionable.
"""

JUDGE_SYSTEM = """You are a strict marketing strategy reviewer.
Evaluate the strategy against the reviews. Return ONLY valid JSON. No extra text.

Scoring rubric (0-10):
- Evidence grounding (0-3)
- Actionability (0-2)
- Coverage (0-2)
- Consistency (0-2)
- Format (0-1)

Return JSON:
{
  "score": number,
  "verdict": "ok" | "not_ok",
  "issues": [{"severity":"low|medium|high","problem":string,"example":string}],
  "recommendations": [string],
  "revision_brief": {
    "keep": [string],
    "change": [string],
    "remove": [string]
  }
}

Rules:
- Any unsupported claim => high severity issue.
- Keep the response compact.
- Maximum 3 issues.
- Maximum 4 recommendations.
- Each item in revision_brief must be short and actionable.
- If verdict is ok, revision_brief may be empty arrays.
"""

REVISION_USER_TEMPLATE = """You must rewrite the marketing strategy JSON to address the judge feedback.

Product:
ASIN: {asin}
Title: {title}
Brand: {brand}
Category: {category}
Description: {description}

Latest reviews (newest first):
{reviews_block}

Previous strategy JSON:
{prev_strategy_json}

Judge feedback JSON:
{judge_json}

Optional structure example:
{structure_example_block}

Hard rules:
- Output JSON ONLY (no markdown, no <think>).
- Follow the schema exactly.
- Remove/replace any claim that isn't supported by review evidence quotes.
- Every insight must include 1–2 exact short quotes copied from reviews.
- Description is context ONLY. Do NOT use it as evidence or quotes.
- If an optional structure example is provided, use it only as a formatting/reference guide. Do not copy claims from it.
- Apply the judge feedback concisely, focusing on revision_brief, issues, and recommendations.

Now output the corrected strategy JSON only.
"""

# Structured imports override the legacy in-file helpers above.
from pipeline.json_utils import (
    _list,
    _s,
    compact_judge_feedback_for_prompt,
    compact_strategy_for_prompt,
    parse_date_ymd,
)
from pipeline.llm_clients import (
    call_lmstudio_chat,
    call_openai_chat,
    ensure_valid_json,
    health_check,
    parse_openai_judge_json,
)
from pipeline.prompts import GEN_SYSTEM, JUDGE_SYSTEM, REVISION_USER_TEMPLATE
from pipeline.template_utils import inspect_structure_example, load_structure_example


# ----------------------------
# Review block builder
# ----------------------------

def build_reviews_block(dfp: pd.DataFrame, k: int, max_chars: int) -> str:
    """Take K newest reviews (by review_date) and compact them."""
    dfp2 = dfp.copy()
    dfp2["__dt"] = dfp2["review_date"].apply(lambda x: parse_date_ymd(_s(x)))
    dfp2 = dfp2.sort_values("__dt", ascending=False).head(k)

    parts: List[str] = []
    total = 0
    for _, r in dfp2.iterrows():
        date = _s(r.get("review_date"))
        rating = _s(r.get("rating"))
        text = _s(r.get("review_text")).replace("\n", " ")
        if not text:
            continue

        chunk = f"- ({date}) rating={rating}: {text}"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)

    return "\n".join(parts)


# ----------------------------
# PDF Export
# ----------------------------

def try_register_unicode_font() -> Optional[str]:
    if not _HAS_TT:
        return None

    candidates = [
        "/System/Library/Fonts/Supplemental/DejaVuSans.ttf",
        "/Library/Fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode MS.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode MS.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                font_name = "UniFont"
                pdfmetrics.registerFont(TTFont(font_name, p))
                return font_name
            except Exception:
                continue
    return None


def export_strategy_pdf(strategy: Dict[str, Any], pdf_path: Path, judge: Optional[Dict[str, Any]] = None) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="Marketing Strategy"
    )

    styles = getSampleStyleSheet()
    h1 = styles["Title"]
    h2 = styles["Heading2"]
    h3 = styles["Heading3"]
    body = styles["BodyText"]

    uni_font = try_register_unicode_font()
    if uni_font:
        for st in (h1, h2, h3, body):
            st.fontName = uni_font

    story: List[Any] = []

    product = strategy.get("product", {}) if isinstance(strategy.get("product"), dict) else {}
    asin = _s(product.get("asin"))
    title = _s(product.get("title"))
    brand = _s(product.get("brand"))
    category = _s(product.get("category"))

    story.append(Paragraph("Marketing Strategy (Generated from Reviews)", h1))
    story.append(Spacer(1, 8))

    table_data = [
        ["ASIN", asin],
        ["Title", title],
        ["Brand", brand],
        ["Category", category],
    ]
    t = Table(table_data, colWidths=[3.2 * cm, 12.8 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.black),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    if isinstance(judge, dict):
        score = _s(judge.get("score"))
        verdict = _s(judge.get("verdict"))
        story.append(Paragraph("Quality Check", h2))
        story.append(Paragraph(f"<b>Score:</b> {score} / 10 &nbsp;&nbsp; <b>Verdict:</b> {verdict}", body))
        story.append(Spacer(1, 8))

    # Insights
    story.append(Paragraph("Insights from Reviews", h2))
    insights = _list(strategy.get("insights"))
    if insights:
        items = []
        for ins in insights[:25]:
            if not isinstance(ins, dict):
                continue
            itype = _s(ins.get("type"))
            stmt = _s(ins.get("statement"))
            quotes = [_s(q) for q in _list(ins.get("evidence_quotes")) if _s(q)]
            quote_str = ""
            if quotes:
                quote_str = "<br/>".join([f"“{q}”" for q in quotes[:2]])
                quote_str = f"<br/><i>Evidence:</i><br/>{quote_str}"
            items.append(ListItem(Paragraph(f"<b>{itype}:</b> {stmt}{quote_str}", body), leftIndent=12))
        story.append(ListFlowable(items, bulletType="bullet"))
    else:
        story.append(Paragraph("No insights provided.", body))
    story.append(Spacer(1, 10))

    # Positioning
    story.append(Paragraph("Positioning", h2))
    pos = strategy.get("positioning", {}) if isinstance(strategy.get("positioning"), dict) else {}
    story.append(Paragraph(f"<b>Value proposition:</b> {_s(pos.get('value_proposition'))}", body))
    story.append(Paragraph(f"<b>Target audience:</b> {_s(pos.get('target_audience'))}", body))
    diffs = [_s(x) for x in _list(pos.get("key_differentiators")) if _s(x)]
    if diffs:
        story.append(Paragraph("<b>Key differentiators:</b>", body))
        story.append(ListFlowable([ListItem(Paragraph(d, body), leftIndent=12) for d in diffs[:15]], bulletType="bullet"))
    story.append(Spacer(1, 10))

    # Messaging
    story.append(Paragraph("Messaging", h2))
    msg = strategy.get("messaging", {}) if isinstance(strategy.get("messaging"), dict) else {}
    story.append(Paragraph(f"<b>Primary message:</b> {_s(msg.get('primary_message'))}", body))
    story.append(Paragraph(f"<b>Tone:</b> {_s(msg.get('tone'))}", body))
    supp = [_s(x) for x in _list(msg.get("supporting_messages")) if _s(x)]
    if supp:
        story.append(Paragraph("<b>Supporting messages:</b>", body))
        story.append(ListFlowable([ListItem(Paragraph(s, body), leftIndent=12) for s in supp[:15]], bulletType="bullet"))
    story.append(Spacer(1, 10))

    # Channels
    story.append(Paragraph("Channels & Content Ideas", h2))
    channels = _list(strategy.get("channels"))
    if channels:
        for ch in channels[:12]:
            if not isinstance(ch, dict):
                continue
            story.append(Paragraph(f"<b>{_s(ch.get('channel'))}</b>", h3))
            story.append(Paragraph(_s(ch.get("why")), body))
            ideas = [_s(x) for x in _list(ch.get("content_ideas")) if _s(x)]
            if ideas:
                story.append(ListFlowable([ListItem(Paragraph(i, body), leftIndent=12) for i in ideas[:12]], bulletType="bullet"))
            story.append(Spacer(1, 6))
    else:
        story.append(Paragraph("No channels provided.", body))
    story.append(Spacer(1, 10))

    # Offers
    story.append(Paragraph("Offers", h2))
    offers = _list(strategy.get("offers"))
    if offers:
        items = []
        for o in offers[:12]:
            if not isinstance(o, dict):
                continue
            items.append(ListItem(Paragraph(f"<b>{_s(o.get('offer'))}</b> — {_s(o.get('why'))}", body), leftIndent=12))
        story.append(ListFlowable(items, bulletType="bullet"))
    else:
        story.append(Paragraph("No offers provided.", body))
    story.append(Spacer(1, 10))

    # Risks
    story.append(Paragraph("Risks & Mitigation", h2))
    risks = _list(strategy.get("risks"))
    if risks:
        items = []
        for r in risks[:15]:
            if not isinstance(r, dict):
                continue
            items.append(ListItem(Paragraph(f"<b>{_s(r.get('risk'))}</b> — {_s(r.get('mitigation'))}", body), leftIndent=12))
        story.append(ListFlowable(items, bulletType="bullet"))
    else:
        story.append(Paragraph("No risks provided.", body))
    story.append(Spacer(1, 10))

    # KPIs
    story.append(Paragraph("KPIs", h2))
    kpis = _list(strategy.get("kpis"))
    if kpis:
        items = []
        for k in kpis[:15]:
            if not isinstance(k, dict):
                continue
            items.append(ListItem(Paragraph(
                f"<b>{_s(k.get('metric'))}</b> — target: {_s(k.get('target'))}; measurement: {_s(k.get('measurement'))}",
                body
            ), leftIndent=12))
        story.append(ListFlowable(items, bulletType="bullet"))
    else:
        story.append(Paragraph("No KPIs provided.", body))
    story.append(Spacer(1, 10))

    # Assumptions
    story.append(Paragraph("Assumptions", h2))
    ass = [_s(x) for x in _list(strategy.get("assumptions")) if _s(x)]
    if ass:
        story.append(ListFlowable([ListItem(Paragraph(a, body), leftIndent=12) for a in ass[:15]], bulletType="bullet"))
    else:
        story.append(Paragraph("No assumptions provided.", body))

    doc.build(story)


# ----------------------------
# Main
# ----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to final cleaned CSV (5..10 reviews per ASIN)")
    p.add_argument("--out-dir", required=True, help="Output directory (JSON + PDF)")

    # NEW: process only this ASIN
    p.add_argument("--asin", default="", help="Process only this ASIN (optional)")

    # Generator (LM Studio)
    p.add_argument("--model", required=True, help="LM Studio model identifier (e.g., qwen3-4b-mlx)")
    p.add_argument("--host", default="http://localhost:1234", help="LM Studio server host")

    # Judge (OpenAI or LM Studio)
    p.add_argument("--judge-provider", choices=["openai", "lmstudio"], default="openai",
                   help="Where to run the judge model (default: openai)")
    p.add_argument("--judge-model", default="gpt-4.1-mini",
                   help="Judge model id (default: gpt-4.1-mini)")
    p.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY", ""),
                   help="OpenAI API key (or set OPENAI_API_KEY env var)")

    # Loop control
    p.add_argument("--target-score", type=float, default=9.0, help="Stop when score >= target (default: 9.0)")
    p.add_argument("--max-rounds", type=int, default=4, help="Max gen-judge iterations per ASIN (default: 4)")

    # Defaults tuned for Qwen3-4B
    p.add_argument("--k", type=int, default=6, help="Newest reviews per ASIN (default: 6)")
    p.add_argument("--max-products", type=int, default=5, help="Max ASINs to process (default: 5)")
    p.add_argument("--sample-asins", type=int, default=0, help="Randomly sample N ASINs (0 = no sampling)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    p.add_argument("--reviews-max-chars", type=int, default=2400, help="Max chars of reviews block (default: 2400)")
    p.add_argument("--structure-example-max-chars", type=int, default=1800, help="Max chars from optional structure example")
    p.add_argument("--revision-strategy-max-chars", type=int, default=4500, help="Max chars from previous strategy in revision prompt")
    p.add_argument("--revision-judge-max-chars", type=int, default=1800, help="Max chars from judge feedback in revision prompt")

    p.add_argument("--gen-temp", type=float, default=0.5, help="Generator temperature")
    p.add_argument("--judge-temp", type=float, default=0.1, help="Judge temperature")
    p.add_argument("--gen-max-tokens", type=int, default=1000, help="Max tokens for generator response")
    p.add_argument("--judge-max-tokens", type=int, default=900, help="Max tokens for judge response")
    p.add_argument("--structure-example-file", default="", help="Optional file with an example strategy structure")
    args = p.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # 0) Health check for LM Studio generator
    try:
        health_check(args.host, args.model)
    except Exception as e:
        print("❌ Health-check failed. LM Studio server/model not responding.")
        print("Error:", repr(e))
        print("Fix: LM Studio -> Server -> Start. Confirm port and model identifier.")
        return

    # Basic OpenAI key validation (only if judge-provider=openai)
    if args.judge_provider == "openai" and not args.openai_api_key:
        raise RuntimeError("judge-provider=openai but OPENAI_API_KEY is missing. Export it or pass --openai-api-key.")

    structure_example_text, structure_example_hash = load_structure_example(args.structure_example_file)
    structure_example_text = shrink_text(structure_example_text, args.structure_example_max_chars)
    structure_example_block = (
        "Use this example as a structure/style reference only. Do not reuse its product facts.\n"
        f"{structure_example_text}\n"
        if structure_example_text else
        "No structure example provided."
    )

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    required_cols = {"asin", "review_date", "review_text", "rating", "product_title", "product_category"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Normalize
    df["asin"] = df["asin"].astype(str).str.strip()
    df["review_date"] = df["review_date"].astype(str).str.strip()
    df["review_text"] = df["review_text"].astype(str)

    asins = df["asin"].unique().tolist()

    # NEW: if specific ASIN requested -> override selection
    if args.asin:
        target = args.asin.strip()
        if target not in set(asins):
            raise ValueError(f"ASIN not found in CSV: {target}")
        asins = [target]
    else:
        # Optional random sampling
        if args.sample_asins and 0 < args.sample_asins < len(asins):
            random.seed(args.seed)
            asins = random.sample(asins, args.sample_asins)

        # Cap
        asins = asins[: args.max_products]

    index_rows: List[Dict[str, Any]] = []

    for i, asin in enumerate(asins, start=1):
        dfp = df[df["asin"] == asin]
        if dfp.empty:
            continue

        title = _s(dfp["product_title"].iloc[0])
        brand = _s(dfp["product_brand"].iloc[0]) if "product_brand" in dfp.columns else ""
        category = _s(dfp["product_category"].iloc[0])

        # description context (optional)
        description = ""
        if "product_description_filled" in dfp.columns:
            description = _s(dfp["product_description_filled"].iloc[0])
        elif "product_description" in dfp.columns:
            description = _s(dfp["product_description"].iloc[0])

        # IMPORTANT: effective k (do not exceed available rows)
        k_eff = min(int(args.k), int(len(dfp)))
        reviews_block = build_reviews_block(dfp, k=k_eff, max_chars=args.reviews_max_chars)

        print(f"\n[{i}/{len(asins)}] ASIN={asin}")
        print(f"   Title: {title[:90]}")
        print(f"   Reviews used: k={k_eff} (requested {args.k}) | reviews_block_chars={len(reviews_block)}")

        # --- iterative loop: generator <-> judge until score >= target-score
        strategy_json: Optional[Dict[str, Any]] = None
        judge_json: Optional[Dict[str, Any]] = None

        # Track best across rounds (highest score; prefer ok on ties)
        best_score = -1.0
        best_ok = False
        best_strategy: Optional[Dict[str, Any]] = None
        best_judge: Optional[Dict[str, Any]] = None

        for round_idx in range(1, args.max_rounds + 1):
            print(f"   🔁 Round {round_idx}/{args.max_rounds}")

            # Build generator prompt
            if strategy_json is None:
                gen_user = f"""Product:
ASIN: {asin}
Title: {title}
Brand: {brand}
Category: {category}
Description: {description}

Latest reviews (newest first):
{reviews_block}

Optional structure example:
{structure_example_block}

Output JSON only.
"""
            else:
                # revision prompt using judge feedback
                gen_user = REVISION_USER_TEMPLATE.format(
                    asin=asin,
                    title=title,
                    brand=brand,
                    category=category,
                    description=description,
                    reviews_block=reviews_block,
                    prev_strategy_json=compact_strategy_for_prompt(strategy_json, args.revision_strategy_max_chars),
                    judge_json=compact_judge_feedback_for_prompt(judge_json, args.revision_judge_max_chars),
                    structure_example_block=structure_example_block,
                )

            # 1) Generate / revise strategy (LM Studio)
            strategy_raw = call_lmstudio_chat(
                host=args.host,
                model=args.model,
                system=GEN_SYSTEM,
                user=gen_user,
                temperature=args.gen_temp if strategy_json is None else 0.25,
                max_tokens=max(args.gen_max_tokens, 1400) if strategy_json is not None else args.gen_max_tokens,
                timeout_sec=240,
                retries=1,
                debug_label="generate" if strategy_json is None else "revise",
            )

            strategy_json = ensure_valid_json(
                args.host,
                args.model,
                strategy_raw,
                out_debug_path=out_dir / f"{asin}_strategy_round{round_idx}_raw.txt",
                regen_system=GEN_SYSTEM,
                regen_user=gen_user,
                regen_max_tokens=max(args.gen_max_tokens, 1600),
            )

            # 2) Judge strategy (OpenAI or LM Studio)
            judge_user = (
                f"Product context:\nASIN: {asin}\nTitle: {title}\n\n"
                f"Reviews:\n{reviews_block}\n\n"
                f"Strategy JSON:\n{json.dumps(strategy_json, ensure_ascii=False)}\n\n"
                "Output JSON only."
            )

            if args.judge_provider == "openai":
                judge_raw = call_openai_chat(
                    api_key=args.openai_api_key,
                    model=args.judge_model,
                    system=JUDGE_SYSTEM,
                    user=judge_user,
                    temperature=args.judge_temp,
                    max_tokens=args.judge_max_tokens,
                    timeout_sec=240,
                    retries=1,
                    debug_label="judge",
                )
                judge_json = parse_openai_judge_json(
                    judge_raw,
                    api_key=args.openai_api_key,
                    model=args.judge_model,
                    system=JUDGE_SYSTEM,
                    user=judge_user,
                    temperature=args.judge_temp,
                    max_tokens=args.judge_max_tokens,
                    out_debug_path=out_dir / f"{asin}_judge_round{round_idx}_openai_parse_failed.txt",
                )
            else:
                judge_raw = call_lmstudio_chat(
                    host=args.host,
                    model=args.judge_model,
                    system=JUDGE_SYSTEM,
                    user=judge_user,
                    temperature=args.judge_temp,
                    max_tokens=args.judge_max_tokens,
                    timeout_sec=240,
                    retries=1,
                    debug_label="judge_lmstudio",
                )
                judge_json = ensure_valid_json(
                    args.host,
                    args.judge_model,
                    judge_raw,
                    out_debug_path=out_dir / f"{asin}_judge_round{round_idx}_raw.txt",
                    regen_system=JUDGE_SYSTEM,
                    regen_user=judge_user,
                    regen_max_tokens=max(args.judge_max_tokens, 1400),
                )

            # Evaluate stop condition
            score = judge_json.get("score", 0) if isinstance(judge_json, dict) else 0
            verdict = judge_json.get("verdict", "") if isinstance(judge_json, dict) else ""

            try:
                score_f = float(score)
            except Exception:
                score_f = 0.0

            verdict_norm = str(verdict).strip().lower()
            is_ok = verdict_norm == "ok"

            print(f"   🧪 Judge: score={score_f} verdict={verdict_norm}")

            # Update best-so-far (highest score; prefer ok if tied)
            if (score_f > best_score) or (score_f == best_score and is_ok and not best_ok):
                best_score = score_f
                best_ok = is_ok
                best_strategy = strategy_json
                best_judge = judge_json

            # Stop when target reached AND verdict ok
            if is_ok and score_f >= args.target_score:
                print("   🎯 Target reached.")
                break

            if round_idx == args.max_rounds:
                print("   ⚠️ Max rounds reached; saving best-scoring round.")

        # Final output should be BEST across rounds (not the last one)
        final_strategy = best_strategy or strategy_json or {}
        final_judge = best_judge or judge_json or {}

        # Save JSONs
        (out_dir / f"{asin}_final_strategy.json").write_text(
            json.dumps(final_strategy, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        (out_dir / f"{asin}_final_judge.json").write_text(
            json.dumps(final_judge, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        # Export PDF
        pdf_path = out_dir / f"{asin}_final_strategy.pdf"
        export_strategy_pdf(final_strategy, pdf_path, judge=final_judge)

        score_out = (final_judge or {}).get("score", "")
        verdict_out = (final_judge or {}).get("verdict", "")

        index_rows.append({
            "asin": asin,
            "title": title,
            "brand": brand,
            "category": category,
            "score": score_out,
            "verdict": verdict_out,
            "pdf_file": pdf_path.name,
            "structure_example_hash": structure_example_hash,
        })

        print(f"   ✅ Done | best_score={best_score} best_ok={best_ok} | PDF={pdf_path.name}")
        time.sleep(0.15)

    # Save index.csv + index.json
    index_csv = out_dir / "index.csv"
    with index_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "asin",
                "title",
                "brand",
                "category",
                "score",
                "verdict",
                "pdf_file",
                "structure_example_hash",
            ],
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(index_rows)

    (out_dir / "index.json").write_text(json.dumps(index_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n✅ Done.")
    print("Output folder:", out_dir)
    print("Index:", index_csv)


if __name__ == "__main__":
    main()
