import requests
import base64
import numpy as np
import torch
from PIL import Image
import json
import io  
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry  

# ==================== 优化旧节点：1-10图输入 + 双URL自动选快 + 新增URL输出 + 多行prompt ====================
class Lu9GeminiImageEditNodeAPI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images1": ("IMAGE",),
                # 核心修改1：edit_prompt改为多行文本（添加multiline: True）
                "edit_prompt": ("STRING", {
                    "default": "修改图片内容",
                    "multiline": True,  # 关键：启用多行输入，支持换行和大量文本
                    "tooltip": "图片编辑提示词，支持多行输入"
                }),
                "api_key": ("STRING", {"default": "sk-你的密钥", "tooltip": "若API要求Bearer前缀，需手动加：Bearer sk-xxx"}),
                "api_url_1": ("STRING", {
                    "default": "http://",
                    "tooltip": "第一个API接口URL，{model}会自动替换为model_name参数"
                }),
                "api_url_2": ("STRING", {
                    "default": "http://",
                    "tooltip": "第二个API接口URL，{model}会自动替换为model_name参数；自动选响应更快的可用URL"
                }),
                "aspect_ratio": ("STRING", {"default": "9:16", "choices": ["1:1", "4:3", "16:9", "9:16", "3:4", "2:3", "3:2"]}),
                "image_size": ("STRING", {"default": "2K", "choices": ["1080P", "2K", "4K", "512x512", "1024x1024"]}),
            },
            "optional": {
                "images2": ("IMAGE",),
                "images3": ("IMAGE",),
                "images4": ("IMAGE",),
                "images5": ("IMAGE",),
                "images6": ("IMAGE",),
                "images7": ("IMAGE",),
                "images8": ("IMAGE",),
                "images9": ("IMAGE",),
                "images10": ("IMAGE",),
                "model_name": ("STRING", {"default": "gemini-1.5-pro-latest"}),
                "size_align": ("STRING", {
                    "default": "images1",
                    "choices": ["images1", "max_size", "min_size"],
                    "tooltip": "仅用于统一输入多图的尺寸（方便拼接），不影响输出尺寸"
                }),
                "add_role_field": ("BOOLEAN", {"default": True, "tooltip": "是否在contents里添加role: user字段"}),
                # 核心修改2：url_detect_timeout默认值改为10.0秒
                "url_detect_timeout": ("FLOAT", {
                    "default": 10.0,  # 关键：默认超时从5秒改为10秒
                    "tooltip": "URL可用性探测超时时间（秒），默认10秒，建议3-20秒"
                }),
            }
        }

    # 核心修改3：新增第4个输出项（selected_url_info），返回最终选用的URL
    RETURN_TYPES = ("IMAGE", "INT", "STRING", "STRING")
    RETURN_NAMES = ("edited_image", "task_status", "error_message", "selected_url_info")
    FUNCTION = "edit_image"
    CATEGORY = "Custom Nodes/Gemini Image Edit"

    # 辅助函数：统一输入图片尺寸
    def unify_image_size(self, image_tensor, target_size):
        np_img = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(np_img, mode="RGB")
        pil_img_resized = pil_img.resize(target_size, Image.Resampling.LANCZOS)
        np_resized = np.array(pil_img_resized).astype(np.float32) / 255.0
        tensor_resized = torch.from_numpy(np_resized).unsqueeze(0)
        return tensor_resized

    # 辅助函数：清理Base64
    def clean_base64(self, b64_str):
        if not isinstance(b64_str, str):
            return ""
        if "data:image/" in b64_str:
            b64_str = b64_str.split(",")[-1]
        return b64_str.strip()

    # 辅助函数：查找有效图片部件
    def find_valid_image_part(self, parts_list):
        valid_image_fields = ["inlineData", "imageData"]
        for part in parts_list:
            if not isinstance(part, dict):
                continue
            for field in valid_image_fields:
                if field in part and part[field]:
                    if "data" in part[field] and part[field]["data"]:
                        return part, field
        return None, None

    # 辅助函数：探测单个URL的可用性和响应耗时
    def detect_url_performance(self, url, api_key, model_name, timeout):
        """
        探测单个URL的可用性和响应耗时
        返回：(是否可用, 响应耗时(秒), 最终替换model后的URL)
        """
        try:
            # 替换URL中的{model}占位符
            final_url = url.format(model=model_name) if "{model}" in url else url
            if not final_url.startswith(("http://", "https://")):
                return False, float("inf"), final_url
            
            # 构造探测请求头（和正式请求一致）
            headers = {
                "Authorization": api_key,
                "Content-Type": "application/json",
                "Accept": "application/json"
            }

            # 创建临时session，不启用重试（探测需快速反馈）
            session = requests.Session()
            session.mount("https://", HTTPAdapter(max_retries=0))
            session.mount("http://", HTTPAdapter(max_retries=0))

            # 记录开始时间
            start_time = time.time()

            # 发送轻量OPTIONS请求（优先），若不支持则发送简化POST请求
            try:
                # 第一步：尝试OPTIONS请求（仅探测连通性，不发送完整payload）
                response = session.options(
                    final_url,
                    headers=headers,
                    timeout=timeout,
                    verify=False
                )
                # 状态码2xx/4xx均视为URL可达（4xx是权限问题，属于可用但配置错误，比5xx/超时更优）
                if response.status_code < 500:
                    elapsed_time = time.time() - start_time
                    return True, elapsed_time, final_url
                else:
                    return False, float("inf"), final_url
            except requests.exceptions.RequestException:
                # OPTIONS请求失败，尝试简化POST请求（最小payload，仅用于探测）
                try:
                    simple_payload = {"contents": [{"parts": [{"text": "ping"}]}]}
                    response = session.post(
                        final_url,
                        headers=headers,
                        json=simple_payload,
                        timeout=timeout,
                        verify=False
                    )
                    elapsed_time = time.time() - start_time
                    if response.status_code < 500:
                        return True, elapsed_time, final_url
                    else:
                        return False, float("inf"), final_url
                except Exception:
                    return False, float("inf"), final_url
        except Exception:
            return False, float("inf"), ""

    # 辅助函数：从两个URL中选择最优（更快+可用）
    def select_best_url(self, url1, url2, api_key, model_name, timeout):
        """
        对比两个URL，选择响应更快的可用URL
        返回：(最终选中的URL, 错误信息)
        """
        # 同时探测两个URL（串行，简单稳定）
        url1_available, url1_time, url1_final = self.detect_url_performance(url1, api_key, model_name, timeout)
        url2_available, url2_time, url2_final = self.detect_url_performance(url2, api_key, model_name, timeout)

        # 打印探测结果（调试用）
        print(f"=== URL探测结果 ===")
        print(f"URL1: {'可用' if url1_available else '不可用'}，耗时：{url1_time:.3f}秒")
        print(f"URL2: {'可用' if url2_available else '不可用'}，耗时：{url2_time:.3f}秒")

        # 逻辑判断：选择最优URL
        available_urls = []
        if url1_available:
            available_urls.append((url1_final, url1_time))
        if url2_available:
            available_urls.append((url2_final, url2_time))

        if not available_urls:
            # 两个URL都不可用
            error_msg = f"两个API URL均不可用！URL1：{url1_final}，URL2：{url2_final}"
            return "", error_msg
        elif len(available_urls) == 1:
            # 只有一个URL可用，直接选用
            selected_url = available_urls[0][0]
            print(f"=== 仅一个URL可用，选中：{selected_url} ===")
            return selected_url, ""
        else:
            # 两个URL都可用，选择耗时更短的
            available_urls.sort(key=lambda x: x[1])  # 按耗时升序排序
            selected_url = available_urls[0][0]
            print(f"=== 两个URL均可用，选中更快的：{selected_url}（耗时{available_urls[0][1]:.3f}秒）===")
            return selected_url, ""

    def edit_image(self, 
                   images1, edit_prompt, api_key, api_url_1, api_url_2, aspect_ratio, image_size, model_name,
                   images2=None, images3=None, images4=None, images5=None, images6=None, images7=None, images8=None, images9=None, images10=None,
                   size_align="images1", add_role_field=True, url_detect_timeout=10.0):
        # 初始化：状态码默认0（失败），错误信息默认空，选中URL默认空
        task_status = 0
        error_message = ""
        selected_url_info = ""  # 新增：初始化选中URL的返回值
        try:
            # 1. 预处理API密钥（添加Bearer前缀）
            if api_key and not api_key.startswith("Bearer "):
                api_key = f"Bearer {api_key}"

            # 2. 双URL自动选择最优（核心逻辑）
            selected_url, url_error = self.select_best_url(
                api_url_1, api_url_2, api_key, model_name, url_detect_timeout
            )
            # 赋值选中URL（用于输出）
            selected_url_info = selected_url
            if url_error:
                raise ValueError(url_error)
            if not selected_url:
                raise ValueError("未获取到可用的API URL")

            # 3. 收集并统一输入图片尺寸（1-10图）
            image_tensors = []
            for img_port in [images1, images2, images3, images4, images5, images6, images7, images8, images9, images10]:
                if img_port is not None:
                    image_tensors.append(img_port)
            
            if len(image_tensors) == 0:
                raise ValueError("至少需要连接1张图片")
            
            # 获取所有输入图片尺寸
            all_sizes = []
            for img in image_tensors:
                H, W = img.shape[1], img.shape[2]
                all_sizes.append((W, H))
            
            # 确定统一尺寸
            if size_align == "images1":
                target_size = all_sizes[0]
            elif size_align == "max_size":
                target_size = (max([s[0] for s in all_sizes]), max([s[1] for s in all_sizes]))
            elif size_align == "min_size":
                target_size = (min([s[0] for s in all_sizes]), min([s[1] for s in all_sizes]))
            
            # 统一图片尺寸
            unified_image_tensors = []
            for img in image_tensors:
                img_H, img_W = img.shape[1], img.shape[2]
                if (img_W, img_H) != target_size:
                    unified_img = self.unify_image_size(img, target_size)
                else:
                    unified_img = img
                unified_image_tensors.append(unified_img)
            
            batch_images = torch.cat(unified_image_tensors, dim=0)

            # 4. 批量转Base64（过滤空值）
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

            # 5. 构造正式请求（使用选中的最优URL）
            headers = {
                "Authorization": api_key,
                "Content-Type": "application/json",
                "Accept": "application/json"
            }

            # 构造parts
            parts = [{"text": edit_prompt.strip()}]
            for img_base64 in img_base64_list:
                if img_base64:
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
            
            # 生成配置
            generation_config = {
                "responseModalities": ["IMAGE"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": image_size.upper()
                },
                "temperature": 0.5,
                "maxOutputTokens": 8192,
                "stopSequences": []
            }

            payload = {
                "contents": [contents_item],
                "generationConfig": generation_config
            }

            # 打印正式请求信息
            print("=== 发送正式请求（选中最优URL）===")
            print(f"选中URL：{selected_url}")
            print(json.dumps(payload, ensure_ascii=False, indent=2))

            # 6. 发送正式请求（启用重试）
            session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=["POST"]
            )
            session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
            session.mount("http://", HTTPAdapter(max_retries=retry_strategy))
            
            response = session.post(
                selected_url,
                headers=headers,
                json=payload,
                timeout=120,
                verify=False
            )

            # 打印响应信息
            print(f"=== 正式请求响应状态码 == {response.status_code} ===")
            print(f"=== 正式请求响应内容 == {response.text} ===")
            
            response.raise_for_status()
            res_data = response.json()

            # 7. 解析响应
            if "candidates" not in res_data or len(res_data["candidates"]) == 0:
                raise ValueError("响应中无有效candidates")
            
            candidate = res_data["candidates"][0]
            if "content" not in candidate or "parts" not in candidate["content"] or len(candidate["content"]["parts"]) == 0:
                raise ValueError("响应结构异常，缺少有效parts数据")
            
            parts_list = candidate["content"]["parts"]
            edited_part, image_field = self.find_valid_image_part(parts_list)
            if not edited_part or not image_field:
                raise ValueError("响应中无有效图片数据，未找到非空的inlineData/imageData字段")
            
            edited_base64 = self.clean_base64(edited_part[image_field]["data"])
            if not edited_base64:
                raise ValueError("图片Base64数据为空或无效")
            
            edited_img_bytes = base64.b64decode(edited_base64)
            if not edited_img_bytes:
                raise ValueError("Base64解码后无有效图片字节数据")
            
            edited_pil = Image.open(io.BytesIO(edited_img_bytes)).convert("RGB")
            edited_np = np.array(edited_pil).astype(np.float32) / 255.0
            edited_tensor = torch.from_numpy(edited_np).unsqueeze(0)

            # 8. 任务成功
            task_status = 1
            # 核心修改：返回新增的selected_url_info
            return (edited_tensor, task_status, error_message, selected_url_info)

        except Exception as e:
            # 任务失败，记录错误信息
            error_message = f"Lu9Gemini图片编辑节点（10图版）错误：{str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            # 兜底返回：补充selected_url_info（即使失败也返回当前获取到的URL信息）
            return (images1, task_status, error_message, selected_url_info)

# ==================== 新节点：完全保留，一行不改 ====================
class Lu9GeminiImageEditNodeAPIWithStatus:
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

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("edited_image", "task_status", "error_message")
    FUNCTION = "edit_image"
    CATEGORY = "Custom Nodes/Gemini Image Edit"

    def unify_image_size(self, image_tensor, target_size):
        np_img = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(np_img, mode="RGB")
        pil_img_resized = pil_img.resize(target_size, Image.Resampling.LANCZOS)
        np_resized = np.array(pil_img_resized).astype(np.float32) / 255.0
        tensor_resized = torch.from_numpy(np_resized).unsqueeze(0)
        return tensor_resized

    def clean_base64(self, b64_str):
        if not isinstance(b64_str, str):
            return ""
        if "data:image/" in b64_str:
            b64_str = b64_str.split(",")[-1]
        return b64_str.strip()

    def find_valid_image_part(self, parts_list):
        valid_image_fields = ["inlineData", "imageData"]
        for part in parts_list:
            if not isinstance(part, dict):
                continue
            for field in valid_image_fields:
                if field in part and part[field]:
                    if "data" in part[field] and part[field]["data"]:
                        return part, field
        return None, None

    def edit_image(self, 
                   images1, edit_prompt, api_key, api_url, aspect_ratio, image_size, model_name,
                   images2=None, images3=None, images4=None, images5=None, size_align="images1", add_role_field=True):
        task_status = 0
        error_message = ""
        try:
            image_tensors = []
            for img_port in [images1, images2, images3, images4, images5]:
                if img_port is not None:
                    image_tensors.append(img_port)
            
            if len(image_tensors) == 0:
                raise ValueError("至少需要连接1张图片")
            
            all_sizes = []
            for img in image_tensors:
                H, W = img.shape[1], img.shape[2]
                all_sizes.append((W, H))
            
            if size_align == "images1":
                target_size = all_sizes[0]
            elif size_align == "max_size":
                target_size = (max([s[0] for s in all_sizes]), max([s[1] for s in all_sizes]))
            elif size_align == "min_size":
                target_size = (min([s[0] for s in all_sizes]), min([s[1] for s in all_sizes]))
            
            unified_image_tensors = []
            for img in image_tensors:
                img_H, img_W = img.shape[1], img.shape[2]
                if (img_W, img_H) != target_size:
                    unified_img = self.unify_image_size(img, target_size)
                else:
                    unified_img = img
                unified_image_tensors.append(unified_img)
            
            batch_images = torch.cat(unified_image_tensors, dim=0)

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

            final_url = api_url.format(model=model_name) if "{model}" in api_url else api_url
            
            if api_key and not api_key.startswith("Bearer "):
                api_key = f"Bearer {api_key}"
            
            headers = {
                "Authorization": api_key,
                "Content-Type": "application/json",
                "Accept": "application/json"
            }

            parts = [{"text": edit_prompt.strip()}]
            for img_base64 in img_base64_list:
                if img_base64:
                    parts.append({
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": img_base64
                        }
                    })

            contents_item = {"parts": parts}
            if add_role_field:
                contents_item["role"] = "user"
            
            generation_config = {
                "responseModalities": ["IMAGE"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": image_size.upper()
                },
                "temperature": 0.5,
                "maxOutputTokens": 8192,
                "stopSequences": []
            }

            payload = {
                "contents": [contents_item],
                "generationConfig": generation_config
            }

            print("=== 发送的请求体（带任务状态码节点）===")
            print(json.dumps(payload, ensure_ascii=False, indent=2))

            session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=["POST"]
            )
            session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
            session.mount("http://", HTTPAdapter(max_retries=retry_strategy))
            
            response = session.post(
                final_url,
                headers=headers,
                json=payload,
                timeout=120,
                verify=False
            )

            print(f"=== 响应状态码（带任务状态码节点）== {response.status_code} ===")
            print(f"=== 响应内容（带任务状态码节点）== {response.text} ===")
            
            response.raise_for_status()
            res_data = response.json()

            if "candidates" not in res_data or len(res_data["candidates"]) == 0:
                raise ValueError("响应中无有效candidates")
            
            candidate = res_data["candidates"][0]
            if "content" not in candidate or "parts" not in candidate["content"] or len(candidate["content"]["parts"]) == 0:
                raise ValueError("响应结构异常，缺少有效parts数据")
            
            parts_list = candidate["content"]["parts"]
            edited_part, image_field = self.find_valid_image_part(parts_list)
            if not edited_part or not image_field:
                raise ValueError("响应中无有效图片数据，未找到非空的inlineData/imageData字段")
            
            edited_base64 = self.clean_base64(edited_part[image_field]["data"])
            if not edited_base64:
                raise ValueError("图片Base64数据为空或无效")
            
            edited_img_bytes = base64.b64decode(edited_base64)
            if not edited_img_bytes:
                raise ValueError("Base64解码后无有效图片字节数据")
            
            edited_pil = Image.open(io.BytesIO(edited_img_bytes)).convert("RGB")
            edited_np = np.array(edited_pil).astype(np.float32) / 255.0
            edited_tensor = torch.from_numpy(edited_np).unsqueeze(0)

            task_status = 1
            return (edited_tensor, task_status, error_message)

        except Exception as e:
            error_message = f"Lu9Gemini图片编辑节点（带状态码）错误：{str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            return (images1, task_status, error_message)

# ==================== 节点注册：简化10图节点名称 ====================
NODE_CLASS_MAPPINGS = {
    "GeminiImageEdit_1to5Input": Lu9GeminiImageEditNodeAPI,
    "GeminiImageEdit_1to5Input_WithStatus": Lu9GeminiImageEditNodeAPIWithStatus
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # 核心修改4：节点名称简化为「Lu9Gemini图片编辑10图版」
    "GeminiImageEdit_1to5Input": "Lu9Gemini图片编辑10图版",
    "GeminiImageEdit_1to5Input_WithStatus": " Lu9Gemini 图片编辑节点（1-5图输入·带任务状态码+错误输出）"
}
