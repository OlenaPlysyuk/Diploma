import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from requests.exceptions import ConnectionError as ReqConnectionError
from requests.exceptions import ReadTimeout

from .json_utils import safe_json_loads, save_text, strip_code_fences


def call_lmstudio_chat(
    host: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int = 180,
    retries: int = 1,
    debug_label: str = "",
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


def call_openai_chat(
    api_key: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int = 180,
    retries: int = 1,
    debug_label: str = "",
) -> str:
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
    regen_max_tokens: int = 1400,
) -> Dict[str, Any]:
    obj = safe_json_loads(raw_text)
    if obj is not None:
        return obj

    save_text(out_debug_path, raw_text)

    if regen_system and regen_user:
        regen_raw = call_lmstudio_chat(
            host=host,
            model=model,
            system=regen_system,
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
