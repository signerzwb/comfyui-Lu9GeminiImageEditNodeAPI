import json
import time
import urllib.error
import urllib.request
import base64
from typing import Any, Dict, List, Tuple

import requests

from .size_map import resolve_size


DEFAULT_BASE_URL_PLACEHOLDER = "这里填请求网址"


class GImageError(RuntimeError):
    pass


def sanitize_response_for_display(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return payload

    sanitized = json.loads(json.dumps(payload))
    data = sanitized.get("data")
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "b64_json" in item:
                b64_value = item.get("b64_json") or ""
                item["b64_json"] = f"<omitted base64 image data: {len(b64_value)} chars>"
    return sanitized


def _json_post(url: str, api_key: str, payload: dict, timeout: int = 900) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise GImageError(f"HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        raise GImageError(str(exc)) from exc


def _multipart_edits_post(
    url: str,
    api_key: str,
    fields: List[Tuple[str, str]],
    files: List[Tuple[str, str, bytes, str]],
    timeout: int = 1200,
) -> dict:
    boundary = "----Lu9GimageBoundary7MA4YWxkTrZu0gW"
    parts: List[bytes] = []

    for name, value in fields:
        parts.append(f"--{boundary}".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        parts.append(b"")
        parts.append(str(value).encode("utf-8"))

    for name, filename, content, content_type in files:
        parts.append(f"--{boundary}".encode())
        parts.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"'.encode()
        )
        parts.append(f"Content-Type: {content_type}".encode())
        parts.append(b"")
        parts.append(content)

    parts.append(f"--{boundary}--".encode())
    parts.append(b"")
    body = b"\r\n".join(parts)

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise GImageError(f"HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        raise GImageError(str(exc)) from exc


def download_image_bytes(url: str, timeout: int = 1200) -> bytes:
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "image/*,*/*;q=0.8",
            },
        )
        resp.raise_for_status()
        return resp.content
    except Exception as exc:
        raise GImageError(f"Failed to download generated image: {exc}") from exc


def extract_image_bytes_from_response_item(item: dict) -> Tuple[bytes, str, str]:
    b64_data = item.get("b64_json")
    if b64_data:
        if b64_data.startswith("data:image"):
            b64_data = b64_data.split(",", 1)[1]
        try:
            return base64.b64decode(b64_data), "", "b64_json"
        except Exception as exc:
            raise GImageError(f"Failed to decode b64_json image payload: {exc}") from exc

    image_url = item.get("url")
    if image_url:
        return download_image_bytes(image_url), image_url, "url_fallback"

    raise GImageError(f"Bad response payload: {json.dumps(item, ensure_ascii=False)}")


def select_api_key_and_model(
    config: dict,
    resolution: str,
    model: str,
    custom_mode: bool = False,
) -> Tuple[str, str]:
    if custom_mode:
        api_key = config.get("api_key_vip", "")
        selected_model = "gpt-image-2-vip" if model == "auto" else model
        return api_key, selected_model

    if model == "auto":
        if resolution == "1k":
            api_key = config.get("api_key_1k", "")
            selected_model = "gpt-image-2"
        else:
            api_key = config.get("api_key_vip", "")
            selected_model = "gpt-image-2-vip"
        return api_key, selected_model

    if resolution == "1k":
        api_key = config.get("api_key_1k", "")
    else:
        api_key = config.get("api_key_vip", "")
    return api_key, model


def _build_generation_payload(
    model: str,
    prompt: str,
    size: str,
    quality: str,
    response_format: str,
) -> dict:
    return {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": size,
        "quality": quality,
        "response_format": response_format,
    }


def _build_edit_fields(
    model: str,
    prompt: str,
    size: str,
    quality: str,
    response_format: str,
) -> List[Tuple[str, str]]:
    return [
        ("model", model),
        ("prompt", prompt),
        ("n", "1"),
        ("size", size),
        ("quality", quality),
        ("response_format", response_format),
    ]


def run_gimage_request(
    config: dict,
    prompt: str,
    resolution: str,
    aspect_ratio: str,
    output_mode: str,
    model: str,
    quality: str,
    image_inputs: List[Tuple[str, bytes]],
    mask_input: Tuple[str, bytes] | None,
    retry_count: int,
    fallback_4k_to_2k: bool,
    custom_width: int | None = None,
    custom_height: int | None = None,
    resolved_aspect_ratio: str | None = None,
) -> Dict[str, Any]:
    base_url = (config.get("base_url") or "").strip()
    if not base_url or base_url == DEFAULT_BASE_URL_PLACEHOLDER:
        raise GImageError("Please fill request URL in lu9-Gimage Config.")
    base_url = base_url.rstrip("/")
    is_edit = len(image_inputs) > 0
    endpoint = "/images/edits" if is_edit else "/images/generations"
    effective_aspect_ratio = resolved_aspect_ratio or aspect_ratio
    custom_mode = effective_aspect_ratio == "custom"
    requested_response_format = "url" if output_mode == "url_only" else "b64_json"

    rounds = retry_count + 1
    attempt_history: List[dict] = []
    last_error = None

    for round_index in range(1, rounds + 1):
        candidate_resolutions = ["custom"] if custom_mode else [resolution]
        if not custom_mode and resolution == "4k" and fallback_4k_to_2k:
            candidate_resolutions = ["4k", "2k"]

        for candidate_resolution in candidate_resolutions:
            size = None
            selected_model = model
            try:
                size = resolve_size(
                    candidate_resolution,
                    effective_aspect_ratio,
                    custom_width=custom_width,
                    custom_height=custom_height,
                )
                api_key, selected_model = select_api_key_and_model(
                    config,
                    candidate_resolution,
                    model,
                    custom_mode=custom_mode,
                )
                if not api_key:
                    raise GImageError(
                        f"Missing API key for resolution {candidate_resolution}. "
                        f"Expected {'api_key_1k' if candidate_resolution == '1k' else 'api_key_vip'}."
                    )

                if is_edit:
                    fields = _build_edit_fields(
                        selected_model,
                        prompt,
                        size,
                        quality,
                        requested_response_format,
                    )
                    files = [(name, f"{name}.png", content, "image/png") for name, content in image_inputs]
                    if mask_input is not None:
                        files.append((mask_input[0], f"{mask_input[0]}.png", mask_input[1], "image/png"))
                    response = _multipart_edits_post(base_url + endpoint, api_key, fields, files)
                else:
                    payload = _build_generation_payload(
                        selected_model,
                        prompt,
                        size,
                        quality,
                        requested_response_format,
                    )
                    response = _json_post(base_url + endpoint, api_key, payload)

                if not response.get("data"):
                    raise GImageError(f"Bad response payload: {json.dumps(response, ensure_ascii=False)}")

                item = response["data"][0]
                if output_mode == "url_only":
                    image_url = item.get("url") or ""
                    if not image_url:
                        raise GImageError(
                            "URL-only mode requested but API response did not include a usable url."
                        )
                    image_bytes = None
                    image_transport = "url_only"
                else:
                    image_bytes, image_url, image_transport = extract_image_bytes_from_response_item(item)
                attempt_history.append(
                    {
                        "round": round_index,
                        "resolution": candidate_resolution,
                        "size": size,
                        "model": selected_model,
                        "output_mode": output_mode,
                        "response_format_requested": requested_response_format,
                        "image_transport": image_transport,
                        "status": "success",
                    }
                )
                return {
                    "image_bytes": image_bytes,
                    "image_url": image_url,
                    "output_mode": output_mode,
                    "response_format_requested": requested_response_format,
                    "image_transport": image_transport,
                    "response": sanitize_response_for_display(response),
                    "final_resolution_used": candidate_resolution,
                    "final_size_used": size,
                    "final_model_used": selected_model,
                    "attempt_history": attempt_history,
                }
            except Exception as exc:
                last_error = exc
                attempt_history.append(
                    {
                        "round": round_index,
                        "resolution": candidate_resolution,
                        "size": size,
                        "model": selected_model,
                        "output_mode": output_mode,
                        "response_format_requested": requested_response_format,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                time.sleep(2)
                continue

    raise GImageError(
        json.dumps(
            {
                "message": str(last_error) if last_error else "Unknown failure",
                "attempt_history": attempt_history,
            },
            ensure_ascii=False,
        )
    )
