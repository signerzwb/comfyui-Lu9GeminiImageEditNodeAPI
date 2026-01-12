import requests
import base64
import numpy as np
import torch
from PIL import Image
import json
import io  
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry  

# ==================== 原有旧节点（你的当前可用版本，完整保留无修改） ====================
class Lu9GeminiImageEditNodeAPI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images1": ("IMAGE",),
                "edit_prompt": ("STRING", {"default": "修改图片内容"}),
                "api_key": ("STRING", {"default": "sk-你的密钥", "tooltip": "若API要求Bearer前缀，需手动加：Bearer sk-xxx"}),
                "api_url": ("STRING", {
                    "default": "http://",
                    "tooltip": "接口URL，{model}会自动替换为model_name参数"
                }),
                "aspect_ratio": ("STRING", {"default": "9:16", "choices": ["1:1", "4:3", "16:9", "9:16", "3:4", "2:3", "3:2"]}),
                "image_size": ("STRING", {"default": "2K", "choices": ["1080P", "2K", "4K", "512x512", "1024x1024"]}),
            },
            "optional": {
                "images2": ("IMAGE",),
                "images3": ("IMAGE",),
                "images4": ("IMAGE",),
                "images5": ("IMAGE",),
                "model_name": ("STRING", {"default": "gemini-1.5-pro-latest"}),
                "size_align": ("STRING", {
                    "default": "images1",
                    "choices": ["images1", "max_size", "min_size"],
                    "tooltip": "仅用于统一输入多图的尺寸（方便拼接），不影响输出尺寸"
                }),
                "add_role_field": ("BOOLEAN", {"default": True, "tooltip": "是否在contents里添加role: user字段"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "edit_image"
    CATEGORY = "Custom Nodes/Gemini Image Edit"

    # 辅助函数：统一输入图片尺寸（仅用于多图拼接）
    def unify_image_size(self, image_tensor, target_size):
        np_img = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(np_img, mode="RGB")
        pil_img_resized = pil_img.resize(target_size, Image.Resampling.LANCZOS)
        np_resized = np.array(pil_img_resized).astype(np.float32) / 255.0
        tensor_resized = torch.from_numpy(np_resized).unsqueeze(0)
        return tensor_resized

    # 辅助函数：清理Base64（移除多余前缀）
    def clean_base64(self, b64_str):
        if "data:image/" in b64_str:
            b64_str = b64_str.split(",")[-1]
        return b64_str.strip()

    def edit_image(self, 
                   images1, edit_prompt, api_key, api_url, aspect_ratio, image_size, model_name,
                   images2=None, images3=None, images4=None, images5=None, size_align="images1", add_role_field=True):
        try:
            # 1. 收集并统一输入图片尺寸（仅用于多图拼接，不影响输出）
            image_tensors = []
            for img_port in [images1, images2, images3, images4, images5]:
                if img_port is not None:
                    image_tensors.append(img_port)
            
            if len(image_tensors) == 0:
                raise ValueError("至少需要连接1张图片")
            
            # 获取所有输入图片尺寸
            all_sizes = []
            for img in image_tensors:
                H, W = img.shape[1], img.shape[2]
                all_sizes.append((W, H))
            
            # 确定输入图片的统一尺寸（仅用于拼接）
            if size_align == "images1":
                target_size = all_sizes[0]
            elif size_align == "max_size":
                target_size = (max([s[0] for s in all_sizes]), max([s[1] for s in all_sizes]))
            elif size_align == "min_size":
                target_size = (min([s[0] for s in all_sizes]), min([s[1] for s in all_sizes]))
            
            # 统一输入图片尺寸
            unified_image_tensors = []
            for img in image_tensors:
                img_H, img_W = img.shape[1], img.shape[2]
                if (img_W, img_H) != target_size:
                    unified_img = self.unify_image_size(img, target_size)
                else:
                    unified_img = img
                unified_image_tensors.append(unified_img)
            
            batch_images = torch.cat(unified_image_tensors, dim=0)

            # 2. 批量转Base64（清理多余前缀）
            img_base64_list = []
            for idx in range(batch_images.shape[0]):
                tensor_img = batch_images[idx]
                np_img = (tensor_img.cpu().numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(np_img, mode="RGB")

                buffer = io.BytesIO()
                pil_img.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                img_base64 = self.clean_base64(img_base64)
                img_base64_list.append(img_base64)

            # 3. 构造请求（强制传递aspect_ratio和image_size给API）
            final_url = api_url.format(model=model_name) if "{model}" in api_url else api_url
            
            # 处理API密钥
            if api_key and not api_key.startswith("Bearer "):
                api_key = f"Bearer {api_key}"
            
            headers = {
                "Authorization": api_key,
                "Content-Type": "application/json",
                "Accept": "application/json"
            }

            # 构造parts（Gemini官方驼峰字段）
            parts = [{"text": edit_prompt.strip()}]
            for img_base64 in img_base64_list:
                parts.append({
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": img_base64
                    }
                })

            # 构造contents
            contents_item = {"parts": parts}
            if add_role_field:
                contents_item["role"] = "user"
            
            # 强制传递aspect_ratio和image_size（关键：让API按此生成）
            generation_config = {
                "responseModalities": ["IMAGE", "TEXT"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio,  # 强制生效9:16
                    "imageSize": image_size.upper()  # 统一大小写（2k→2K，1080p→1080P）
                },
                "temperature": 0.5,
                "maxOutputTokens": 2048
            }

            payload = {
                "contents": [contents_item],
                "generationConfig": generation_config
            }

            # 打印请求体（调试用）
            print("=== 发送的请求体 ===")
            print(json.dumps(payload, ensure_ascii=False, indent=2))

            # 4. 发送请求
            session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=["POST"]
            )
            session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
            
            response = session.post(
                final_url,
                headers=headers,
                json=payload,
                timeout=120,
                verify=False  # 临时关闭SSL验证，生产环境建议开启
            )

            # 打印响应（调试用）
            print(f"=== 响应状态码 == {response.status_code} ===")
            print(f"=== 响应内容 == {response.text} ===")
            
            response.raise_for_status()
            res_data = response.json()

            # 5. 解析响应（核心修改：不缩放输出图片，保留API生成的9:16尺寸）
            if "candidates" not in res_data or len(res_data["candidates"]) == 0:
                raise ValueError("响应中无有效candidates")
            
            candidate = res_data["candidates"][0]
            if "content" not in candidate or "parts" not in candidate["content"]:
                raise ValueError("响应结构异常")
            
            edited_part = candidate["content"]["parts"][0]
            if "inlineData" not in edited_part:
                raise ValueError("响应中无图片数据")
            
            edited_base64 = self.clean_base64(edited_part["inlineData"]["data"])
            edited_img_bytes = base64.b64decode(edited_base64)
            edited_pil = Image.open(io.BytesIO(edited_img_bytes)).convert("RGB")

            # 【关键修改：删除原有的“缩放到输入图尺寸”逻辑】
            # 直接使用API按aspect_ratio生成的尺寸，不做任何缩放
            edited_np = np.array(edited_pil).astype(np.float32) / 255.0
            edited_tensor = torch.from_numpy(edited_np).unsqueeze(0)

            return (edited_tensor,)

        except Exception as e:
            error_detail = f"Lu9Gemini图片编辑节点错误：{str(e)}"
            print(error_detail)
            import traceback
            traceback.print_exc()
            # 异常兜底：返回第一张输入图
            return (images1,)

# ==================== 新增节点：带任务状态码返回（基于你的当前版本扩展） ====================
class Lu9GeminiImageEditNodeAPIWithStatus:
    @classmethod
    def INPUT_TYPES(cls):
        # 完全复制旧节点的输入参数，保留你当前的所有配置（如api_url默认http://）
        return {
            "required": {
                "images1": ("IMAGE",),
                "edit_prompt": ("STRING", {"default": "修改图片内容"}),
                "api_key": ("STRING", {"default": "sk-你的密钥", "tooltip": "若API要求Bearer前缀，需手动加：Bearer sk-xxx"}),
                "api_url": ("STRING", {
                    "default": "http://",
                    "tooltip": "接口URL，{model}会自动替换为model_name参数"
                }),
                "aspect_ratio": ("STRING", {"default": "9:16", "choices": ["1:1", "4:3", "16:9", "9:16", "3:4", "2:3", "3:2"]}),
                "image_size": ("STRING", {"default": "2K", "choices": ["1080P", "2K", "4K", "512x512", "1024x1024"]}),
            },
            "optional": {
                "images2": ("IMAGE",),
                "images3": ("IMAGE",),
                "images4": ("IMAGE",),
                "images5": ("IMAGE",),
                "model_name": ("STRING", {"default": "gemini-1.5-pro-latest"}),
                "size_align": ("STRING", {
                    "default": "images1",
                    "choices": ["images1", "max_size", "min_size"],
                    "tooltip": "仅用于统一输入多图的尺寸（方便拼接），不影响输出尺寸"
                }),
                "add_role_field": ("BOOLEAN", {"default": True, "tooltip": "是否在contents里添加role: user字段"}),
            }
        }

    # 新增INT类型状态码返回，保留原有图片输出
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("edited_image", "task_status")
    FUNCTION = "edit_image"
    CATEGORY = "Custom Nodes/Gemini Image Edit"

    # 辅助函数：统一输入图片尺寸（与你的旧节点完全一致）
    def unify_image_size(self, image_tensor, target_size):
        np_img = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(np_img, mode="RGB")
        pil_img_resized = pil_img.resize(target_size, Image.Resampling.LANCZOS)
        np_resized = np.array(pil_img_resized).astype(np.float32) / 255.0
        tensor_resized = torch.from_numpy(np_resized).unsqueeze(0)
        return tensor_resized

    # 辅助函数：清理Base64（与你的旧节点完全一致）
    def clean_base64(self, b64_str):
        if "data:image/" in b64_str:
            b64_str = b64_str.split(",")[-1]
        return b64_str.strip()

    def edit_image(self, 
                   images1, edit_prompt, api_key, api_url, aspect_ratio, image_size, model_name,
                   images2=None, images3=None, images4=None, images5=None, size_align="images1", add_role_field=True):
        # 初始化任务状态码：默认0（失败）
        task_status = 0
        try:
            # 1. 收集并统一输入图片尺寸（与你的旧节点逻辑完全一致）
            image_tensors = []
            for img_port in [images1, images2, images3, images4, images5]:
                if img_port is not None:
                    image_tensors.append(img_port)
            
            if len(image_tensors) == 0:
                raise ValueError("至少需要连接1张图片")
            
            # 获取所有输入图片尺寸
            all_sizes = []
            for img in image_tensors:
                H, W = img.shape[1], img.shape[2]
                all_sizes.append((W, H))
            
            # 确定输入图片的统一尺寸（仅用于拼接）
            if size_align == "images1":
                target_size = all_sizes[0]
            elif size_align == "max_size":
                target_size = (max([s[0] for s in all_sizes]), max([s[1] for s in all_sizes]))
            elif size_align == "min_size":
                target_size = (min([s[0] for s in all_sizes]), min([s[1] for s in all_sizes]))
            
            # 统一输入图片尺寸
            unified_image_tensors = []
            for img in image_tensors:
                img_H, img_W = img.shape[1], img.shape[2]
                if (img_W, img_H) != target_size:
                    unified_img = self.unify_image_size(img, target_size)
                else:
                    unified_img = img
                unified_image_tensors.append(unified_img)
            
            batch_images = torch.cat(unified_image_tensors, dim=0)

            # 2. 批量转Base64（清理多余前缀）
            img_base64_list = []
            for idx in range(batch_images.shape[0]):
                tensor_img = batch_images[idx]
                np_img = (tensor_img.cpu().numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(np_img, mode="RGB")

                buffer = io.BytesIO()
                pil_img.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                img_base64 = self.clean_base64(img_base64)
                img_base64_list.append(img_base64)

            # 3. 构造请求（强制传递aspect_ratio和image_size给API）
            final_url = api_url.format(model=model_name) if "{model}" in api_url else api_url
            
            # 处理API密钥（与你的旧节点逻辑完全一致）
            if api_key and not api_key.startswith("Bearer "):
                api_key = f"Bearer {api_key}"
            
            headers = {
                "Authorization": api_key,
                "Content-Type": "application/json",
                "Accept": "application/json"
            }

            # 构造parts（Gemini官方驼峰字段）
            parts = [{"text": edit_prompt.strip()}]
            for img_base64 in img_base64_list:
                parts.append({
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": img_base64
                    }
                })

            # 构造contents
            contents_item = {"parts": parts}
            if add_role_field:
                contents_item["role"] = "user"
            
            # 强制传递aspect_ratio和image_size（与你的旧节点逻辑完全一致）
            generation_config = {
                "responseModalities": ["IMAGE", "TEXT"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": image_size.upper()
                },
                "temperature": 0.5,
                "maxOutputTokens": 2048
            }

            payload = {
                "contents": [contents_item],
                "generationConfig": generation_config
            }

            # 打印请求体（调试用，带新节点标识，方便区分）
            print("=== 发送的请求体（带任务状态码节点）===")
            print(json.dumps(payload, ensure_ascii=False, indent=2))

            # 4. 发送请求（与你的旧节点逻辑完全一致）
            session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=["POST"]
            )
            session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
            
            response = session.post(
                final_url,
                headers=headers,
                json=payload,
                timeout=120,
                verify=False  # 保留你的原有配置，临时关闭SSL验证
            )

            # 打印响应（调试用，带新节点标识，方便区分）
            print(f"=== 响应状态码（带任务状态码节点）== {response.status_code} ===")
            print(f"=== 响应内容（带任务状态码节点）== {response.text} ===")
            
            response.raise_for_status()
            res_data = response.json()

            # 5. 解析响应（与你的旧节点逻辑完全一致）
            if "candidates" not in res_data or len(res_data["candidates"]) == 0:
                raise ValueError("响应中无有效candidates")
            
            candidate = res_data["candidates"][0]
            if "content" not in candidate or "parts" not in candidate["content"]:
                raise ValueError("响应结构异常")
            
            edited_part = candidate["content"]["parts"][0]
            if "inlineData" not in edited_part:
                raise ValueError("响应中无图片数据")
            
            edited_base64 = self.clean_base64(edited_part["inlineData"]["data"])
            edited_img_bytes = base64.b64decode(edited_base64)
            edited_pil = Image.open(io.BytesIO(edited_img_bytes)).convert("RGB")

            # 保留API生成的原始尺寸，不缩放（与你的旧节点逻辑完全一致）
            edited_np = np.array(edited_pil).astype(np.float32) / 255.0
            edited_tensor = torch.from_numpy(edited_np).unsqueeze(0)

            # 流程完全成功，更新状态码为1
            task_status = 1
            # 返回编辑后的图片 + 成功状态码
            return (edited_tensor, task_status)

        except Exception as e:
            error_detail = f"Lu9Gemini图片编辑节点（带状态码）错误：{str(e)}"
            print(error_detail)
            import traceback
            traceback.print_exc()
            # 异常兜底：返回第一张输入图 + 失败状态码0
            return (images1, task_status)

# ==================== 节点注册：新旧节点共存（无语法错误，保留你的旧节点映射） ====================
NODE_CLASS_MAPPINGS = {
    # 旧节点映射（你的原有配置，保留不变）
    "GeminiImageEdit_1to5Input": Lu9GeminiImageEditNodeAPI,
    # 新节点映射（追加，逗号分隔，无语法错误）
    "GeminiImageEdit_1to5Input_WithStatus": Lu9GeminiImageEditNodeAPIWithStatus
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # 旧节点显示名称（你的原有配置，保留不变）
    "GeminiImageEdit_1to5Input": " Lu9Gemini 图片编辑节点（1-5图输入）",
    # 新节点显示名称（追加，逗号分隔，方便识别）
    "GeminiImageEdit_1to5Input_WithStatus": " Lu9Gemini 图片编辑节点（1-5图输入·带任务状态码）"
}
