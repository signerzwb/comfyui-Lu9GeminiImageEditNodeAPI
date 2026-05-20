import json

from .api_client import DEFAULT_BASE_URL_PLACEHOLDER, GImageError, run_gimage_request
from .size_map import (
    ASPECT_OPTIONS,
    pick_closest_aspect_ratio,
    resolve_size,
    resolve_smartauto_request,
)
from .utils import (
    get_tensor_image_size,
    make_blank_image_tensor,
    optional_mask_to_png_bytes,
    png_bytes_to_tensor,
    tensor_to_png_bytes,
)


CONFIG_TYPE = "LU9_GIMAGE_CONFIG"


class Lu9GimageConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key_1k": ("STRING", {"default": "", "multiline": False}),
                "api_key_vip": ("STRING", {"default": "", "multiline": False}),
                "base_url": ("STRING", {"default": DEFAULT_BASE_URL_PLACEHOLDER, "multiline": False}),
            }
        }

    RETURN_TYPES = (CONFIG_TYPE,)
    RETURN_NAMES = ("config",)
    FUNCTION = "build_config"
    CATEGORY = "lu9-Gimage"

    def build_config(self, api_key_1k, api_key_vip, base_url):
        config = {
            "api_key_1k": api_key_1k.strip(),
            "api_key_vip": api_key_vip.strip(),
            "base_url": base_url.strip(),
        }
        return (config,)


class Lu9GimageUnified:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": (CONFIG_TYPE,),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "resolution": (["1k", "2k", "4k"], {"default": "1k"}),
                "aspect_ratio": (
                    sorted(
                        set(ASPECT_OPTIONS["1k"] + ASPECT_OPTIONS["2k"] + ASPECT_OPTIONS["4k"])
                        | {"auto", "smartauto", "custom"}
                    ),
                    {"default": "1:1"},
                ),
                "model": (["auto", "gpt-image-2", "gpt-image-2-vip"], {"default": "auto"}),
                "quality": (["low", "medium", "high"], {"default": "high"}),
                "retry_count": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "fallback_4k_to_2k": ("BOOLEAN", {"default": False}),
                "skip_error": ("BOOLEAN", {"default": False}),
                "custom_width": ("INT", {"default": 2432, "min": 64, "max": 3840, "step": 64}),
                "custom_height": ("INT", {"default": 3264, "min": 64, "max": 3840, "step": 64}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "response", "status_text")
    FUNCTION = "run"
    CATEGORY = "lu9-Gimage"

    def run(
        self,
        config,
        prompt,
        resolution,
        aspect_ratio,
        model,
        quality,
        retry_count,
        fallback_4k_to_2k,
        skip_error,
        custom_width,
        custom_height,
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        image6=None,
        image7=None,
        image8=None,
        image9=None,
        mask=None,
    ):
        image_inputs = []
        for image_tensor in [image1, image2, image3, image4, image5, image6, image7, image8, image9]:
            if image_tensor is not None:
                image_inputs.append(("image", tensor_to_png_bytes(image_tensor)))

        mask_input = None
        if mask is not None and len(image_inputs) > 0:
            mask_bytes = optional_mask_to_png_bytes(mask)
            if mask_bytes is not None:
                mask_input = ("mask", mask_bytes)

        resolved_aspect_ratio = aspect_ratio
        resolved_custom_width = custom_width
        resolved_custom_height = custom_height
        resolved_size_strategy = "explicit"

        if aspect_ratio in {"auto", "smartauto"}:
            if image1 is None:
                raise RuntimeError(f"aspect_ratio={aspect_ratio} requires image1 to be connected.")
            width, height = get_tensor_image_size(image1)
            if aspect_ratio == "auto":
                resolved_aspect_ratio = pick_closest_aspect_ratio(resolution, width, height)
                resolved_custom_width = None
                resolved_custom_height = None
                resolved_size_strategy = "preset_auto"
            else:
                smartauto_result = resolve_smartauto_request(resolution, width, height)
                resolved_aspect_ratio = smartauto_result["resolved_aspect_ratio"]
                resolved_custom_width = smartauto_result["custom_width"]
                resolved_custom_height = smartauto_result["custom_height"]
                resolved_size_strategy = smartauto_result["resolved_size_strategy"]
        elif aspect_ratio == "custom":
            resolved_size_strategy = "custom_manual"

        try:
            result = run_gimage_request(
                config=config,
                prompt=prompt,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                model=model,
                quality=quality,
                image_inputs=image_inputs,
                mask_input=mask_input,
                retry_count=retry_count,
                fallback_4k_to_2k=fallback_4k_to_2k,
                custom_width=resolved_custom_width,
                custom_height=resolved_custom_height,
                resolved_aspect_ratio=resolved_aspect_ratio,
            )
            output_tensor = png_bytes_to_tensor(result["image_bytes"])
            response_payload = {
                "response": result["response"],
                "final_resolution_used": result["final_resolution_used"],
                "final_size_used": result["final_size_used"],
                "final_model_used": result["final_model_used"],
                "requested_aspect_ratio": aspect_ratio,
                "resolved_aspect_ratio": resolved_aspect_ratio,
                "resolved_size_strategy": resolved_size_strategy,
                "resolved_custom_width": resolved_custom_width,
                "resolved_custom_height": resolved_custom_height,
                "attempt_history": result["attempt_history"],
            }
            status_text = (
                f"success | resolution={result['final_resolution_used']} | "
                f"size={result['final_size_used']} | aspect={resolved_aspect_ratio} | "
                f"strategy={resolved_size_strategy} | "
                f"model={result['final_model_used']}"
            )
            return (
                output_tensor,
                result["image_url"],
                json.dumps(response_payload, ensure_ascii=False),
                status_text,
            )
        except GImageError as exc:
            if skip_error:
                fallback_resolution = "custom" if resolved_aspect_ratio == "custom" else resolution
                if resolved_aspect_ratio != "custom" and resolution == "4k" and fallback_4k_to_2k:
                    try:
                        resolve_size("4k", resolved_aspect_ratio)
                    except Exception:
                        fallback_resolution = "2k"

                blank_tensor = make_blank_image_tensor()
                response_payload = {
                    "status": "error",
                    "message": f"ERROR FAILED: {str(exc)}",
                    "final_resolution_used": fallback_resolution,
                    "final_size_used": None,
                    "final_model_used": model,
                    "requested_aspect_ratio": aspect_ratio,
                    "resolved_aspect_ratio": resolved_aspect_ratio,
                    "resolved_size_strategy": resolved_size_strategy,
                    "resolved_custom_width": resolved_custom_width,
                    "resolved_custom_height": resolved_custom_height,
                }
                status_text = (
                    f"ERROR FAILED | resolution={fallback_resolution} | "
                    f"aspect={resolved_aspect_ratio} | strategy={resolved_size_strategy} | "
                    f"detail={str(exc)}"
                )
                return (
                    blank_tensor,
                    "",
                    json.dumps(response_payload, ensure_ascii=False),
                    status_text,
                )
            raise RuntimeError(str(exc)) from exc


NODE_CLASS_MAPPINGS = {
    "Lu9GimageConfig": Lu9GimageConfig,
    "Lu9GimageUnified": Lu9GimageUnified,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Lu9GimageConfig": "lu9-Gimage Config",
    "Lu9GimageUnified": "lu9-Gimage Unified",
}
