import json

from .api_client import DEFAULT_BASE_URL_PLACEHOLDER, GImageError, run_gimage_request
from .size_map import ASPECT_OPTIONS
from .utils import optional_mask_to_png_bytes, png_bytes_to_tensor, tensor_to_png_bytes


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
                    sorted(set(ASPECT_OPTIONS["1k"] + ASPECT_OPTIONS["2k"] + ASPECT_OPTIONS["4k"])),
                    {"default": "1:1"},
                ),
                "model": (["auto", "gpt-image-2", "gpt-image-2-vip"], {"default": "auto"}),
                "quality": (["low", "medium", "high"], {"default": "high"}),
                "retry_count": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "fallback_4k_to_2k": ("BOOLEAN", {"default": False}),
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
        for index, image_tensor in enumerate(
            [image1, image2, image3, image4, image5, image6, image7, image8, image9], start=1
        ):
            if image_tensor is not None:
                image_inputs.append((f"image{index}", tensor_to_png_bytes(image_tensor)))

        mask_input = None
        if mask is not None and len(image_inputs) > 0:
            mask_bytes = optional_mask_to_png_bytes(mask)
            if mask_bytes is not None:
                mask_input = ("mask", mask_bytes)

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
            )
            output_tensor = png_bytes_to_tensor(result["image_bytes"])
            response_payload = {
                "response": result["response"],
                "final_resolution_used": result["final_resolution_used"],
                "final_size_used": result["final_size_used"],
                "final_model_used": result["final_model_used"],
                "attempt_history": result["attempt_history"],
            }
            status_text = (
                f"success | resolution={result['final_resolution_used']} | "
                f"size={result['final_size_used']} | model={result['final_model_used']}"
            )
            return (
                output_tensor,
                result["image_url"],
                json.dumps(response_payload, ensure_ascii=False),
                status_text,
            )
        except GImageError as exc:
            raise RuntimeError(str(exc)) from exc


NODE_CLASS_MAPPINGS = {
    "Lu9GimageConfig": Lu9GimageConfig,
    "Lu9GimageUnified": Lu9GimageUnified,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Lu9GimageConfig": "lu9-Gimage Config",
    "Lu9GimageUnified": "lu9-Gimage Unified",
}
