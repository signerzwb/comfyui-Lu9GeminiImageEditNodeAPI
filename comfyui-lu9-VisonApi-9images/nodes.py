import json

from .utils import tensor_to_png_bytes
from .vision_client import VisionError, run_vision_request


class Lu9VisionApi9Images:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {"default": "", "multiline": False}),
                "api_key_1": ("STRING", {"default": "", "multiline": False}),
                "api_key_2": ("STRING", {"default": "", "multiline": False}),
                "api_key_3": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {"default": "", "multiline": False}),
                "backup_api_url": ("STRING", {"default": "", "multiline": False}),
                "backup_api_key_1": ("STRING", {"default": "", "multiline": False}),
                "backup_api_key_2": ("STRING", {"default": "", "multiline": False}),
                "backup_api_key_3": ("STRING", {"default": "", "multiline": False}),
                "backup_model": ("STRING", {"default": "", "multiline": False}),
                "system_prompt": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 32768, "step": 1}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "extra_params": ("STRING", {"default": "", "multiline": True}),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "skip_error": ("BOOLEAN", {"default": False}),
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
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("response", "raw_response", "status_text")
    FUNCTION = "run"
    CATEGORY = "lu9-VisionApi"

    def run(
        self,
        api_url,
        api_key_1,
        api_key_2,
        api_key_3,
        model,
        backup_api_url,
        backup_api_key_1,
        backup_api_key_2,
        backup_api_key_3,
        backup_model,
        system_prompt,
        prompt,
        temperature,
        top_p,
        max_tokens,
        presence_penalty,
        frequency_penalty,
        extra_params,
        seed,
        skip_error,
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        image6=None,
        image7=None,
        image8=None,
        image9=None,
    ):
        image_bytes_list = []
        for image_tensor in [image1, image2, image3, image4, image5, image6, image7, image8, image9]:
            if image_tensor is not None:
                image_bytes_list.append(tensor_to_png_bytes(image_tensor))

        try:
            result = run_vision_request(
                primary_group={
                    "name": "primary",
                    "api_url": api_url,
                    "model": model,
                    "api_keys": [api_key_1, api_key_2, api_key_3],
                    "key_prefix": "api_key",
                },
                backup_group={
                    "name": "backup",
                    "api_url": backup_api_url,
                    "model": backup_model,
                    "api_keys": [backup_api_key_1, backup_api_key_2, backup_api_key_3],
                    "key_prefix": "backup_api_key",
                },
                system_prompt=system_prompt,
                prompt=prompt,
                image_bytes_list=image_bytes_list,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                seed=seed,
                extra_params=extra_params,
            )
            choice = {}
            choices = result["response"].get("choices")
            if isinstance(choices, list) and choices:
                choice = choices[0]
            status_text = (
                f"success | group={result.get('used_group_name', 'primary')} | "
                f"model={result.get('used_model', model).strip()} | images={len(image_bytes_list)} | "
                f"key={result.get('used_key_field', 'api_key_1')} | "
                f"seed={seed} | finish_reason={choice.get('finish_reason', '')} | "
                f"endpoint={result['endpoint']}"
            )
            return (
                result["text"],
                json.dumps(result["response"], ensure_ascii=False),
                status_text,
            )
        except VisionError as exc:
            if skip_error:
                response_payload = {
                    "status": "error",
                    "message": f"ERROR FAILED: {str(exc)}",
                    "model": model,
                    "image_count": len(image_bytes_list),
                    "seed": seed,
                }
                status_text = (
                    f"ERROR FAILED | primary_model={model.strip()} | "
                    f"backup_model={backup_model.strip()} | images={len(image_bytes_list)} | "
                    f"seed={seed} | detail={str(exc)}"
                )
                return (
                    "",
                    json.dumps(response_payload, ensure_ascii=False),
                    status_text,
                )
            raise RuntimeError(str(exc)) from exc


NODE_CLASS_MAPPINGS = {
    "Lu9VisionApi9Images": Lu9VisionApi9Images,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Lu9VisionApi9Images": "lu9-VisionApi 9images",
}
