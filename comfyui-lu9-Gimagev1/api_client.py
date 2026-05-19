import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Tuple

from .size_map import resolve_size


DEFAULT_BASE_URL_PLACEHOLDER = "这里填请求网址"


class GImageError(RuntimeError):
    pass


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
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except Exception as exc:
        raise GImageError(f"Failed to download generated image: {exc}") from exc


def select_api_key_and_model(config: dict, resolution: str, model: str) -> Tuple[str, str]:
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


def _build_generation_payload(model: str, prompt: str, size: str, quality: str) -> dict:
    return {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": size,
        "quality": quality,
        "response_format": "url",
    }


def _build_edit_fields(model: str, prompt: str, size: str, quality: str) -> List[Tuple[str, str]]:
    return [
        ("model", model),
        ("prompt", prompt),
        ("n", "1"),
        ("size", size),
        ("quality", quality),
        ("response_format", "url"),
    ]


def run_gimage_request(
    config: dict,
    prompt: str,
    resolution: str,
    aspect_ratio: str,
    model: str,
    quality: str,
    image_inputs: List[Tuple[str, bytes]],
    mask_input: Tuple[str, bytes] | None,
    retry_count: int,
    fallback_4k_to_2k: bool,
) -> Dict[str, Any]:
    base_url = (config.get("base_url") or "").strip()
    if not base_url or base_url == DEFAULT_BASE_URL_PLACEHOLDER:
        raise GImageError("Please fill request URL in lu9-Gimage Config.")
    base_url = base_url.rstrip("/")
    is_edit = len(image_inputs) > 0
    endpoint = "/images/edits" if is_edit else "/images/generations"

    rounds = retry_count + 1
    attempt_history: List[dict] = []
    last_error = None

    for round_index in range(1, rounds + 1):
        candidate_resolutions = [resolution]
        if resolution == "4k" and fallback_4k_to_2k:
            candidate_resolutions = ["4k", "2k"]

        for candidate_resolution in candidate_resolutions:
            size = None
            selected_model = model
            try:
                size = resolve_size(candidate_resolution, aspect_ratio)
                api_key, selected_model = select_api_key_and_model(config, candidate_resolution, model)
                if not api_key:
                    raise GImageError(
                        f"Missing API key for resolution {candidate_resolution}. "
                        f"Expected {'api_key_1k' if candidate_resolution == '1k' else 'api_key_vip'}."
                    )

                if is_edit:
                    fields = _build_edit_fields(selected_model, prompt, size, quality)
                    files = [(name, f"{name}.png", content, "image/png") for name, content in image_inputs]
                    if mask_input is not None:
                        files.append((mask_input[0], f"{mask_input[0]}.png", mask_input[1], "image/png"))
                    response = _multipart_edits_post(base_url + endpoint, api_key, fields, files)
                else:
                    payload = _build_generation_payload(selected_model, prompt, size, quality)
                    response = _json_post(base_url + endpoint, api_key, payload)

                if not response.get("data") or not response["data"][0].get("url"):
                    raise GImageError(f"Bad response payload: {json.dumps(response, ensure_ascii=False)}")

                image_url = response["data"][0]["url"]
                image_bytes = download_image_bytes(image_url)
                attempt_history.append(
                    {
                        "round": round_index,
                        "resolution": candidate_resolution,
                        "size": size,
                        "model": selected_model,
                        "status": "success",
                    }
                )
                return {
                    "image_bytes": image_bytes,
                    "image_url": image_url,
                    "response": response,
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
