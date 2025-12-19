import requests
import base64
import numpy as np
import torch
from PIL import Image
import json
import io  # 提前导入io，避免局部导入报错
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry  # 新增重试依赖

# 注册节点到ComfyUI
class Lu9GeminiImageEditNodeAPI:
    # 节点基本信息
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 输入1：原始图片（ComfyUI的IMAGE类型）
                "image": ("IMAGE",),
                # 输入2：编辑指令（文本）
                "edit_prompt": ("STRING", {"default": ""}),
                # 输入3：API密钥（sk-xxx）
                "api_key": ("STRING", {"default": "sk-你的密钥"}),
                # 新增：接口URL（可自定义）
                "api_url": ("STRING", {
                    "default": "API地址",
                    "tooltip": "接口URL模板，{model_name}会自动替换为模型名称参数"
                }),
                # 输入4：图片比例（接口要求）
                "aspect_ratio": ("STRING", {"default": "9:16", "choices": ["1:1", "4:3", "16:9", "9:16"]}),
                # 输入5：图片分辨率
                "image_size": ("STRING", {"default": "2K", "choices": ["1080P", "2K", "4K"]}),
            },
            "optional": {
                # 可选：模型名称（默认固定值）
                "model_name": ("STRING", {"default": "gemini-2.5-flash-image-preview"}),
            }
        }

    # 节点输出（编辑后的图片）
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "edit_image"  # 核心执行函数
    CATEGORY = "Custom Nodes/Gemini Image Edit"  # 节点分类（自定义）

    # 核心逻辑：图片转换 + 接口请求 + 响应解析
    def edit_image(self, image, edit_prompt, api_key, api_url, aspect_ratio, image_size, model_name):
        try:
            # ---------------------- 步骤1：ComfyUI图片转Base64（PNG格式） ----------------------
            # ComfyUI的IMAGE是torch.Tensor，形状[1, H, W, C]，取值0-1 → 转PIL Image
            tensor_image = image  # [1, H, W, C]
            # 缩放至0-255，转numpy，再转PIL
            np_image = (tensor_image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(np_image, mode="RGB")
            
            # PIL转Base64（PNG格式）
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # ---------------------- 步骤2：构造接口请求参数 ----------------------
            # 替换URL模板中的{model_name}为实际模型名
            final_url = api_url.format(model_name=model_name)
            headers = {
                "Authorization": api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": img_base64
                                }
                            },
                            {
                                "text": edit_prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "responseModalities": ["IMAGE"],
                    "imageConfig": {
                        "aspectRatio": aspect_ratio,
                        "imageSize": image_size
                    }
                }
            }

            # ---------------------- 步骤3：发送POST请求（新增重试+延长超时） ----------------------
            # 配置重试策略，解决网络波动问题
            session = requests.Session()
            retry_strategy = Retry(
                total=3,  # 重试次数
                backoff_factor=1,  # 重试间隔（1s, 2s, 4s...）
                status_forcelist=[500, 502, 503, 504]  # 触发重试的状态码
            )
            session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
            
            # 发送请求（超时延长至120s，适配大图片）
            response = session.post(final_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()  # 抛出HTTP错误（如401/500）
            res_data = response.json()

            # ---------------------- 步骤4：解析响应，Base64转ComfyUI图片 ----------------------
            # 提取编辑后图片的Base64
            edited_base64 = res_data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
            # Base64解码为PIL Image
            edited_img_bytes = base64.b64decode(edited_base64)
            edited_pil = Image.open(io.BytesIO(edited_img_bytes)).convert("RGB")
            
            # 优化：强制缩放到原始图片尺寸，避免比例/尺寸不一致
            edited_pil = edited_pil.resize(pil_image.size, Image.Resampling.LANCZOS)
            
            # PIL转ComfyUI的Tensor（[1, H, W, C]，取值0-1）
            edited_np = np.array(edited_pil).astype(np.float32) / 255.0
            edited_tensor = torch.from_numpy(edited_np).unsqueeze(0)  # 增加batch维度 [1, H, W, C]

            return (edited_tensor,)

        except Exception as e:
            # 异常处理（打印详细错误，返回原始图片）
            error_detail = f"Lu9Gemini图片编辑节点错误：{str(e)}"
            print(error_detail)
            # 可选：将错误信息输出到ComfyUI控制台（更醒目）
            import traceback
            traceback.print_exc()
            return (image,)  # 失败时返回原始图片

# 注册节点到ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeminiImageEdit": Lu9GeminiImageEditNodeAPI
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageEdit": " Lu9Gemini 图片编辑节点"
}