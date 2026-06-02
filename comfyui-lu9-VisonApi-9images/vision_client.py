import json
from typing import Any, Dict, List

import requests


class VisionError(RuntimeError):
    pass


KEY_RELATED_STATUS_CODES = {401, 402, 403, 429}
GROUP_RELATED_STATUS_CODES = {404, 405, 408, 500, 502, 503, 504, 521, 522, 523, 524}

KEY_RELATED_TEXT_MARKERS = [
    "invalid api key",
    "invalid key",
    "incorrect api key",
    "unauthorized",
    "forbidden",
    "insufficient",
    "balance",
    "quota",
    "payment required",
    "billing",
    "rate limit",
    "too many requests",
    "token expired",
    "expired key",
]

GROUP_RELATED_TEXT_MARKERS = [
    "model not found",
    "unsupported model",
    "unsupported endpoint",
    "not found",
    "connection refused",
    "timed out",
    "timeout",
    "temporary failure in name resolution",
    "name or service not known",
    "dns",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
    "remote end closed",
    "ssl",
    "certificate",
    "proxyerror",
    "max retries exceeded",
]


def _normalize_chat_url(api_url: str) -> str:
    normalized = (api_url or "").strip().rstrip("/")
    if not normalized:
        raise VisionError("Please fill api_url.")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _parse_extra_params(extra_params: str) -> Dict[str, Any]:
    text = (extra_params or "").strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
    except Exception as exc:
        raise VisionError(f"extra_params must be valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise VisionError("extra_params must be a JSON object.")
    return parsed


def _make_data_url(image_bytes: bytes) -> str:
    import base64

    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _group_has_any_value(group: Dict[str, Any]) -> bool:
    if (group.get("api_url") or "").strip():
        return True
    if (group.get("model") or "").strip():
        return True
    for key in group.get("api_keys", []):
        if (key or "").strip():
            return True
    return False


def _prepare_group(group: Dict[str, Any]) -> Dict[str, Any]:
    if not _group_has_any_value(group):
        return {}

    api_url = (group.get("api_url") or "").strip()
    model = (group.get("model") or "").strip()
    key_prefix = group.get("key_prefix", "api_key")
    name = group.get("name", "group")

    if not api_url:
        raise VisionError(f"Please fill {name}_api_url." if name == "backup" else "Please fill api_url.")
    if not model:
        raise VisionError(f"Please fill {name}_model." if name == "backup" else "Please fill model.")

    candidate_keys = []
    for key_index, key in enumerate(group.get("api_keys", []), start=1):
        normalized = (key or "").strip()
        if normalized:
            candidate_keys.append(
                {
                    "key_index": key_index,
                    "key_field": f"{key_prefix}_{key_index}",
                    "api_key": normalized,
                }
            )

    if not candidate_keys:
        label = f"{key_prefix}_1"
        raise VisionError(f"Please fill {label} at least.")

    return {
        "name": name,
        "api_url": api_url,
        "endpoint": _normalize_chat_url(api_url),
        "model": model,
        "candidate_keys": candidate_keys,
    }


def _classify_http_error(status_code: int | str, detail: str) -> str:
    try:
        code = int(status_code)
    except Exception:
        code = -1

    lower_detail = (detail or "").lower()

    if code in KEY_RELATED_STATUS_CODES:
        return "key"
    if code in GROUP_RELATED_STATUS_CODES:
        return "group"
    if any(marker in lower_detail for marker in KEY_RELATED_TEXT_MARKERS):
        return "key"
    if any(marker in lower_detail for marker in GROUP_RELATED_TEXT_MARKERS):
        return "group"
    return "unknown"


def _classify_exception(exc: Exception) -> str:
    lower_detail = str(exc).lower()
    if any(marker in lower_detail for marker in GROUP_RELATED_TEXT_MARKERS):
        return "group"
    return "unknown"


def _extract_text_from_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    chunks.append(str(item["text"]))
                elif item.get("text"):
                    chunks.append(str(item["text"]))
        return "\n".join(chunk for chunk in chunks if chunk).strip()

    if content is None:
        return ""
    return str(content)


def extract_response_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise VisionError(f"Bad response payload: {json.dumps(payload, ensure_ascii=False)}")

    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise VisionError(f"Bad response payload: {json.dumps(payload, ensure_ascii=False)}")

    text = _extract_text_from_message_content(message.get("content"))
    if text:
        return text

    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()

    raise VisionError(f"Empty assistant message: {json.dumps(payload, ensure_ascii=False)}")


def run_vision_request(
    primary_group: Dict[str, Any],
    backup_group: Dict[str, Any],
    system_prompt: str,
    prompt: str,
    image_bytes_list: List[bytes],
    temperature: float,
    top_p: float,
    max_tokens: int,
    presence_penalty: float,
    frequency_penalty: float,
    seed: int,
    extra_params: str,
    timeout: int = 900,
) -> Dict[str, Any]:
    prepared_groups = []
    normalized_primary = _prepare_group(primary_group)
    if normalized_primary:
        prepared_groups.append(normalized_primary)
    normalized_backup = _prepare_group(backup_group)
    if normalized_backup:
        prepared_groups.append(normalized_backup)
    if not prepared_groups:
        raise VisionError("Please fill at least one provider group.")

    content: List[Dict[str, Any]] = []

    if (prompt or "").strip():
        content.append({"type": "text", "text": prompt.strip()})

    for index, image_bytes in enumerate(image_bytes_list, start=1):
        content.append({"type": "text", "text": f"[Image {index}]"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _make_data_url(image_bytes)},
            }
        )

    if not content:
        raise VisionError("Please fill prompt or connect at least one image.")

    messages: List[Dict[str, Any]] = []
    if (system_prompt or "").strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": content})

    errors: List[str] = []
    merged_extra_params = _parse_extra_params(extra_params)

    for group in prepared_groups:
        payload: Dict[str, Any] = {
            "model": group["model"],
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }
        if seed > 0:
            payload["seed"] = seed
        payload.update(merged_extra_params)

        for candidate in group["candidate_keys"]:
            try:
                response = requests.post(
                    group["endpoint"],
                    headers={
                        "Authorization": f"Bearer {candidate['api_key']}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                raw_payload = response.json()
                text = extract_response_text(raw_payload)
                return {
                    "endpoint": group["endpoint"],
                    "response": raw_payload,
                    "text": text,
                    "used_group_name": group["name"],
                    "used_key_index": candidate["key_index"],
                    "used_key_field": candidate["key_field"],
                    "used_model": group["model"],
                }
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else "unknown"
                detail = exc.response.text if exc.response is not None else str(exc)
                classification = _classify_http_error(status_code, detail)
                errors.append(
                    f"group={group['name']} {candidate['key_field']}: HTTP {status_code}: {detail}"
                )
                if classification == "group":
                    break
                continue
            except Exception as exc:
                classification = _classify_exception(exc)
                errors.append(f"group={group['name']} {candidate['key_field']}: {str(exc)}")
                if classification == "group":
                    break
                continue

    raise VisionError("All API keys failed. " + " | ".join(errors))
