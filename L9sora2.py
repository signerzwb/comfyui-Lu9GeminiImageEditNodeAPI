import torch
import numpy as np
import requests
import time
import io
import os
import shutil  # 用于复制本地视频文件（Python内置，无需额外安装）
import tempfile
import imageio
from PIL import Image

# ---------------------- 视频类型核心定义（兼容ComfyUI视频节点，解决所有方法缺失报错） ----------------------
class IO:
    VIDEO = "VIDEO"

class ComflyVideoAdapter:
    def __init__(self, video_url, video_path="", fps=24.0, width=1280, height=720, task_status=0):
        self.video_url = video_url  # 视频远程URL
        self.path = video_path      # 视频本地临时路径（已下载的文件路径）
        self.fps = fps              # 视频帧率
        self.width = width          # 视频宽度
        self.height = height        # 视频高度
        self.task_status = task_status  # 新增：任务状态（0=失败，1=成功）

    def __repr__(self):
        return f"ComflyVideoAdapter(url={self.video_url}, path={self.path}, status={self.task_status})"

    # ComfyUI视频节点必需：获取视频宽高
    def get_dimensions(self):
        return (self.width, self.height)

    # 兼容部分视频节点：获取帧率
    def get_fps(self):
        return self.fps

    # 兼容部分保存节点：获取本地路径
    def get_path(self):
        return self.path

    # ComfyUI保存视频节点必需：将视频写入指定输出路径（核心解决报错）
    def save_to(self, output_path, fps=None, codec=None, bitrate=None, **kwargs):
        # 新增：先判断任务状态，失败时直接返回错误信息，不抛异常
        if self.task_status == 0:
            error_info = "⚠️ 任务未成功，无法保存视频（任务状态：失败）"
            print(error_info)
            return error_info  # 返回错误信息，不中断ComfyUI流程

        # 优先使用实例自身帧率，兼容传入参数
        target_fps = fps or self.fps or 24.0

        # 核心逻辑：复制已下载的本地视频到目标输出路径（保持原质量，无需重新编码）
        if os.path.exists(self.path):
            try:
                # 创建目标输出目录（不存在则创建）
                output_dir = os.path.dirname(output_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # 复制视频文件（保留原文件元数据）
                shutil.copy2(self.path, output_path)
                print(f"✅ 视频已成功保存到指定路径：{output_path}")
                return f"✅ 视频已成功保存到：{output_path}"
            except Exception as e:
                error_info = f"❌ 保存视频失败（复制本地文件）：{str(e)}"
                print(error_info)
                return error_info  # 返回错误信息，不抛异常

        # 兜底逻辑：本地文件不存在时，从URL下载保存
        if self.video_url and not os.path.exists(self.path):
            try:
                print(f"⚠️  本地视频文件不存在，从URL下载并保存...")
                response = requests.get(self.video_url, stream=True)
                response.raise_for_status()

                # 创建目标输出目录
                output_dir = os.path.dirname(output_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # 流式写入目标文件
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                success_info = f"✅ 视频从URL下载并保存成功：{output_path}"
                print(success_info)
                return success_info
            except Exception as e:
                error_info = f"❌ 从URL下载并保存视频失败：{str(e)}"
                print(error_info)
                return error_info  # 返回错误信息，不抛异常

        # 最终兜底：无有效视频源（返回错误信息，不抛异常）
        error_info = "❌ 无法保存视频：本地视频文件不存在且无有效视频URL"
        print(error_info)
        return error_info

# ---------------------- 核心节点类（保留你原有的API请求、轮询、下载逻辑） ----------------------
class SoraVideoGenerator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_url": ("STRING", {
                    "default": "https://", 
                    "multiline": False,
                    "tooltip": "API的基础地址"
                }),
                "api_key": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "填写您的 Bearer Token (sk-...)"
                }),
                "prompt": ("STRING", {
                    "default": "A cinematic drone shot of a futuristic city...", 
                    "multiline": True, 
                    "dynamicPrompts": True
                }),
                "model": (["sora-2", "sora-2-pro"], {
                    "default": "sora-2"
                }),
                "size": ([
                    "1280x720", 
                    "720x1280", 
                    "1024x1792", 
                    "1792x1024"
                ], {
                    "default": "1280x720",
                    "tooltip": "API要求参考图必须严格匹配此分辨率"
                }),
                "seconds": (["4", "8", "12", "15"], {  # 增加15秒选项
                    "default": "4"
                }),
                # 请求超时配置项
                "request_timeout": ("INT", {
                    "default": 60,
                    "min": 10,
                    "max": 300,
                    "step": 10,
                    "tooltip": "API请求超时时间（秒），范围10-300"
                }),
                # 轮询最大时间配置项
                "polling_max_time": ("INT", {
                    "default": 1200,
                    "min": 60,
                    "max": 3600,
                    "step": 60,
                    "tooltip": "任务轮询最大超时时间（秒），范围60-3600"
                }),
                # 轮询间隔配置项
                "polling_interval": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 60,
                    "step": 1,
                    "tooltip": "任务状态查询间隔（秒），范围1-60"
                }),
            },
            "optional": {
                "input_reference": ("IMAGE", ), # 左侧图片接口
            }
        }

    # 增加error_message返回项
    RETURN_TYPES = (IO.VIDEO, "STRING", "INT", "STRING")
    RETURN_NAMES = ("video", "video_url", "task_status", "error_message")
    FUNCTION = "generate_video"
    CATEGORY = "Sora API"

    def tensor_to_pil(self, image_tensor):
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]
        i = 255. * image_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img

    def image_to_bytes(self, pil_image):
        byte_arr = io.BytesIO()
        pil_image.save(byte_arr, format='PNG')
        byte_arr.seek(0)
        return byte_arr

    def generate_video(self, base_url, api_key, prompt, model, size, seconds, request_timeout, polling_max_time, polling_interval, input_reference=None):
        # 初始化返回值
        video_adapter = ComflyVideoAdapter(video_url="", video_path="", task_status=0)
        video_url = ""
        task_status = 0 # 0=失败，1=成功
        error_message = ""  # 初始化错误信息
        target_w, target_h = 1280, 720

        try:
            if not api_key:
                raise ValueError("API Key 不能为空，请填写有效的 Bearer Token")

            # 1. 构建API请求参数
            headers = {"Authorization": f"Bearer {api_key}"}
            base_url = base_url.rstrip("/")
            create_url = f"{base_url}/v1/videos"

            # 解析分辨率
            try:
                target_w, target_h = map(int, size.split('x'))
            except:
                target_w, target_h = 1280, 720

            data = {
                "model": model,
                "prompt": prompt,
                "seconds": seconds,
                "size": size
            }

            # 处理参考图片
            files = {}
            if input_reference is not None:
                print(f"📸 正在处理参考图片...")
                pil_img = self.tensor_to_pil(input_reference)
                if pil_img.size != (target_w, target_h):
                    print(f"🔧 调整参考图片尺寸：{pil_img.size} → ({target_w}, {target_h})")
                    pil_img = pil_img.resize((target_w, target_h), Image.LANCZOS)
                img_bytes = self.image_to_bytes(pil_img)
                files["input_reference"] = ("reference.png", img_bytes, "image/png")

            # 2. 发起视频生成任务（使用配置的超时时间）
            print(f"🚀 发送请求到：{create_url}")
            response = requests.post(
                create_url, 
                headers=headers, 
                data=data, 
                files=files if files else None, 
                timeout=request_timeout
            )
            if response.status_code != 200:
                raise RuntimeError(f"API请求失败（状态码：{response.status_code}）：{response.text}")
            task_data = response.json()

            # 获取任务ID
            task_id = task_data.get("id")
            if not task_id:
                raise RuntimeError(f"无法获取任务ID，API返回：{task_data}")
            print(f"📋 任务创建成功，ID：{task_id}，正在轮询任务状态...")

            # 3. 轮询任务状态（使用配置的轮询参数）
            status_url = f"{base_url}/v1/videos/{task_id}"
            start_time = time.time()

            while True:
                # 超时判断
                if time.time() - start_time > polling_max_time:
                    raise TimeoutError(f"任务超时，超过{polling_max_time}秒未完成")
                time.sleep(polling_interval)

                try:
                    status_resp = requests.get(status_url, headers=headers, timeout=30)
                    if status_resp.status_code != 200:
                        continue
                    
                    status_data = status_resp.json()
                    status = status_data.get("status")
                    print(f"📊 任务状态：{status}")

                    # 任务完成/失败判断
                    if status in ["completed", "succeeded", "success"]:
                        break
                    elif status in ["failed", "error", "rejected"]:
                        raise RuntimeError(f"任务失败，状态：{status}，原因：{status_data.get('error', '未知错误')}")
                except Exception as e:
                    print(f"⚠️  轮询任务状态时出错：{e}")
                    if "Failed. Status" in str(e):
                        raise e
                    continue

            # 4. 获取视频下载链接
            content_url = f"{base_url}/v1/videos/{task_id}/content"
            final_video_url = None
            video_binary_data = None

            content_resp = requests.get(content_url, headers=headers, timeout=60)
            content_resp.raise_for_status()
            
            if "application/json" in content_resp.headers.get("Content-Type", ""):
                content_data = content_resp.json()
                # 提取视频URL（兼容多种返回格式）
                final_video_url = content_data.get("url") or \
                                  content_data.get("output", {}).get("url") or \
                                  content_data.get("data", {}).get("url") or \
                                  content_data.get("download_url")
                
                if not final_video_url:
                    raise RuntimeError("无法从API返回中提取视频下载URL")
                video_url = final_video_url
                print(f"🔗 获取到视频下载URL：{video_url}")
            else:
                video_binary_data = content_resp.content
                print(f"📥 直接获取到视频二进制数据")

            # 5. 下载并保存视频到本地临时目录
            output_dir = os.path.join(tempfile.gettempdir(), "comfyui_sora_output")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            file_path = os.path.join(output_dir, f"sora_{task_id}.mp4")
            
            if final_video_url:
                print(f"📥 正在从URL下载视频到本地...")
                dl_resp = requests.get(final_video_url, timeout=300)
                dl_resp.raise_for_status()
                video_binary_data = dl_resp.content

            # 写入本地文件
            with open(file_path, "wb") as f:
                f.write(video_binary_data)
            print(f"💾 视频已保存到本地临时目录：{file_path}")

            # 6. 提取视频帧率
            fps = 24.0
            try:
                reader = imageio.get_reader(file_path, 'ffmpeg')
                fps = reader.get_meta_data().get('fps', 24.0)
                reader.close()
            except:
                print(f"⚠️  无法提取视频帧率，使用默认值24.0")
                pass

            # 7. 封装视频适配器实例（传入任务状态）
            video_adapter = ComflyVideoAdapter(
                video_url=video_url,
                video_path=file_path,
                fps=fps,
                width=target_w,
                height=target_h,
                task_status=1  # 任务成功，状态设为1
            )

            # 8. 更新任务状态为成功
            task_status = 1
            print(f"🎉 视频生成任务全部完成！")

            # 返回结果
            return (video_adapter, video_url, task_status, error_message)

        except Exception as e:
            error_detail = f"❌ Sora视频生成节点错误：{str(e)}"
            error_message = error_detail  # 赋值错误信息
            print(error_detail)
            import traceback
            traceback.print_exc()
            # 异常兜底：返回合法实例，避免节点崩溃
            return (video_adapter, video_url, task_status, error_message)

# ---------------------- 节点映射配置（ComfyUI必需） ----------------------
NODE_CLASS_MAPPINGS = {
    "SoraVideoGenerator": SoraVideoGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SoraVideoGenerator": "Lu9_sora2"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
