import torch
import numpy as np
import requests
import time
import io
import os
import shutil
import tempfile
import imageio
from PIL import Image

# ---------------------- 视频类型核心定义（兼容ComfyUI视频节点） ----------------------
class IO:
    VIDEO = "VIDEO"

class ComflyVideoAdapter:
    def __init__(self, video_url, video_path="", fps=24.0, width=720, height=720, task_status=0):
        self.video_url = video_url          # 视频远程URL
        self.path = video_path              # 本地临时路径
        self.fps = fps                      # 帧率（默认24）
        self.width = width                  # 视频宽度
        self.height = height                # 视频高度
        self.task_status = task_status      # 任务状态（0=失败，1=成功）

    def __repr__(self):
        return f"ComflyVideoAdapter(url={self.video_url}, path={self.path}, status={self.task_status})"

    # ComfyUI视频节点必需：获取宽高
    def get_dimensions(self):
        return (self.width, self.height)

    # 获取帧率
    def get_fps(self):
        return self.fps

    # 获取本地路径
    def get_path(self):
        return self.path

    # 保存视频到指定路径（避免抛异常打断流程）
    def save_to(self, output_path, fps=None, codec=None, bitrate=None, **kwargs):
        if self.task_status == 0:
            error_info = "⚠️ 15秒双传图任务未成功，无法保存视频"
            print(error_info)
            return error_info

        target_fps = fps or self.fps or 24.0
        # 本地文件存在时直接复制
        if os.path.exists(self.path):
            try:
                output_dir = os.path.dirname(output_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                shutil.copy2(self.path, output_path)
                success_info = f"✅ 15秒视频已保存至：{output_path}"
                print(success_info)
                return success_info
            except Exception as e:
                error_info = f"❌ 保存视频失败：{str(e)}"
                print(error_info)
                return error_info
        # 本地文件不存在时从URL下载
        elif self.video_url:
            try:
                print(f"📥 从URL下载15秒视频...")
                response = requests.get(self.video_url, stream=True, timeout=300)
                response.raise_for_status()
                output_dir = os.path.dirname(output_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                success_info = f"✅ 15秒视频从URL下载至：{output_path}"
                print(success_info)
                return success_info
            except Exception as e:
                error_info = f"❌ 从URL下载视频失败：{str(e)}"
                print(error_info)
                return error_info
        else:
            error_info = "❌ 无有效视频源（本地路径/URL均为空）"
            print(error_info)
            return error_info

# ---------------------- 15秒双传图版核心节点（URL+直接传图合并） ----------------------
class SoraVideoGenerator15sDualUpload:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # 基础配置
                "base_url": ("STRING", {
                    "default": "https://147ai.com", 
                    "multiline": False,
                    "tooltip": "API基础地址，例如 https://147ai.com（无需末尾斜杠）"
                }),
                "api_key": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "Bearer Token（格式：sk-xxx，无需手动加Bearer前缀）"
                }),
                # 核心生成参数（15秒专属）
                "prompt": ("STRING", {
                    "default": "根据参考图生成15秒高清广告宣传片，画面流畅自然", 
                    "multiline": True, 
                    "dynamicPrompts": True,
                    "tooltip": "视频生成提示词，需符合API合规要求（禁止真人/暴力/色情）"
                }),
                "model": (
                    [
                        "sora2-landscape-15s",    # 横屏15秒（普通模式，10秒生成）
                        "sora2-portrait-15s",     # 竖屏15秒（普通模式，10秒生成）
                        "sora2-pro-landscape-hd-15s",  # 横屏15秒（Pro模式，高清，15秒生成）
                        "sora2-pro-portrait-hd-15s"    # 竖屏15秒（Pro模式，高清，15秒生成）
                    ], 
                    {
                        "default": "sora2-portrait-15s",
                        "tooltip": "15秒专属模型：landscape=横屏（电脑），portrait=竖屏（手机），pro=高清"
                    }
                ),
                # URL传图（可选，优先级低于直接传图）
                "image_url": ("STRING", {
                    "default": "https://www.baidu.com/img/PCtm_d9c8750bed0b3c7d089fa7d55720d6cf.png", 
                    "multiline": False,
                    "tooltip": "参考图片URL（支持JPG/PNG，公网可访问；若直接传图有值，此参数会被忽略）"
                }),
                # 超时/轮询配置
                "request_timeout": ("INT", {
                    "default": 60,
                    "min": 10,
                    "max": 300,
                    "step": 10,
                    "tooltip": "API请求超时时间（秒），范围10-300"
                }),
                "polling_max_time": ("INT", {
                    "default": 900,  # 15秒模型生成快，缩短最大轮询时间（15分钟）
                    "min": 60,
                    "max": 1800,
                    "step": 60,
                    "tooltip": "任务轮询最大超时时间（秒），15秒模型建议设600-900"
                }),
                "polling_interval": ("INT", {
                    "default": 3,  # 生成快，缩短轮询间隔
                    "min": 1,
                    "max": 30,
                    "step": 1,
                    "tooltip": "任务状态查询间隔（秒），15秒模型建议设3-5"
                }),
            },
            "optional": {
                # 直接传图（优先使用：有值则忽略image_url）
                "local_reference_image": ("IMAGE", ),
            }
        }

    # 返回类型：视频实例、视频URL、任务状态、错误信息
    RETURN_TYPES = (IO.VIDEO, "STRING", "INT", "STRING")
    RETURN_NAMES = ("15s_video", "15s_video_url", "15s_task_status", "15s_error_msg")
    FUNCTION = "generate_15s_video"
    # 节点分类（与原节点区分）
    CATEGORY = "Sora API / 15秒双传图版"

    # 工具方法1：ComfyUI图像张量转PIL图片
    def tensor_to_pil(self, image_tensor):
        """将ComfyUI的IMAGE张量（4维/3维）转换为PIL图片"""
        if image_tensor is None:
            return None
        # 处理批量张量（4维：[batch, C, H, W] → 取第一张）
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]
        # 张量转numpy数组（像素值从[0,1]→[0,255]）
        img_np = 255. * image_tensor.cpu().numpy()
        # clip避免超出范围，转uint8格式
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        # 转PIL（注意：numpy格式是[C, H, W]，PIL需要[H, W, C]，所以转置）
        if img_np.shape[0] in [1, 3]:  # 1通道（灰度）或3通道（RGB）
            img_np = np.transpose(img_np, (1, 2, 0))
            # 灰度图转3通道（避免API不支持单通道）
            if img_np.shape[2] == 1:
                img_np = np.repeat(img_np, 3, axis=2)
        return Image.fromarray(img_np)

    # 工具方法2：PIL图片转字节流（用于直接传图）
    def pil_to_byte_stream(self, pil_image):
        """将PIL图片转换为字节流（PNG格式，用于multipart/form-data上传）"""
        if pil_image is None:
            return None
        byte_arr = io.BytesIO()
        pil_image.save(byte_arr, format="PNG", quality=100)
        byte_arr.seek(0)  # 重置指针到开头，确保API能读取完整数据
        return byte_arr

    # 工具方法3：根据模型推断视频尺寸（用于调整直接传图的尺寸）
    def get_size_by_model(self, model_name):
        """根据模型名称推断视频分辨率（确保直接传图尺寸匹配API要求）"""
        if "landscape" in model_name:
            return (1280, 720)  # 横屏模型：宽1280，高720
        elif "portrait" in model_name:
            return (720, 1280)  # 竖屏模型：宽720，高1280
        else:
            return (720, 720)   # 默认：正方形

    def generate_15s_video(self, base_url, api_key, prompt, model, image_url, 
                          request_timeout, polling_max_time, polling_interval, local_reference_image=None):
        # 初始化返回值
        video_adapter = ComflyVideoAdapter(video_url="", video_path="", task_status=0)
        video_url = ""
        task_status = 0  # 0=失败，1=成功
        error_msg = ""
        video_width, video_height = self.get_size_by_model(model)  # 从模型获取默认尺寸

        try:
            # 1. 基础参数校验
            if not api_key:
                raise ValueError("API Key不能为空（格式：sk-xxx）")
            
            # 2. 传图优先级判断（核心逻辑：直接传图 > URL传图）
            use_local_upload = False  # 是否使用直接传图
            upload_files = None       # multipart/form-data的文件参数
            request_payload = None    # 请求体参数
            request_headers = {       # 基础请求头（Content-Type动态调整）
                "Authorization": f"Bearer {api_key.strip()}",
                "Accept": "application/json"
            }

            # 2.1 优先处理直接传图（local_reference_image有值）
            if local_reference_image is not None:
                print(f"📸 检测到直接传图，优先使用（忽略image_url参数）")
                # 张量转PIL图片
                pil_img = self.tensor_to_pil(local_reference_image)
                if pil_img is None:
                    raise ValueError("直接传图数据无效，无法转换为图片")
                
                # 调整图片尺寸（匹配模型对应的分辨率，避免API报错）
                target_size = self.get_size_by_model(model)
                if pil_img.size != target_size:
                    print(f"🔧 调整直接传图尺寸：{pil_img.size} → {target_size}（匹配模型{model}）")
                    pil_img = pil_img.resize(target_size, Image.LANCZOS)  # 高质量缩放
                
                #  PIL图片转字节流（用于multipart/form-data上传）
                img_byte_stream = self.pil_to_byte_stream(pil_img)
                if img_byte_stream is None:
                    raise RuntimeError("直接传图转换为字节流失败，无法上传")
                
                # 构建multipart/form-data参数
                use_local_upload = True
                upload_files = {
                    "input_reference": ("reference.png", img_byte_stream, "image/png")  # API必填字段：input_reference
                }
                request_payload = {
                    "prompt": prompt.strip(),
                    "model": model.strip()
                }
                # 无需手动设置Content-Type：requests会自动为files参数添加multipart/form-data头

            # 2.2 无直接传图，使用URL传图
            else:
                if not image_url.strip():
                    raise ValueError("无直接传图，且image_url为空，请至少填写一个参考图参数")
                print(f"🌐 未检测到直接传图，使用image_url：{image_url[:50]}...")
                
                # 构建application/json请求体
                request_payload = {
                    "image_url": image_url.strip(),
                    "prompt": prompt.strip(),
                    "model": model.strip()
                }
                # 设置JSON格式请求头
                request_headers["Content-Type"] = "application/json"

            # 3. 构建API请求URL
            base_url = base_url.rstrip("/")
            create_url = f"{base_url}/v1/videos"
            print(f"🚀 发起15秒视频生成请求（模型：{model}，传图方式：{'直接传图' if use_local_upload else 'URL传图'}）")
            print(f"📋 请求参数：{request_payload}")

            # 4. 发送创建任务请求（核心修复：调整data/json参数传递逻辑）
            response = requests.post(
                create_url,
                headers=request_headers,
                # 修复点1：直接传图时，文本参数通过data传递（multipart/form-data格式）
                data=request_payload if use_local_upload else None,
                # 修复点2：URL传图时，参数通过json传递（application/json格式）
                json=request_payload if not use_local_upload else None,
                files=upload_files if use_local_upload else None,
                timeout=request_timeout
            )

            # 5. 处理创建任务响应
            if response.status_code != 200:
                raise RuntimeError(f"创建任务失败（状态码：{response.status_code}），响应：{response.text}")
            task_data = response.json()
            print(f"📥 创建任务响应：{task_data}")

            # 6. 获取任务ID（API返回id字段）
            task_id = task_data.get("id")
            if not task_id:
                raise RuntimeError(f"无法获取任务ID，API返回：{task_data}")
            print(f"✅ 15秒视频任务创建成功，任务ID：{task_id}")

            # 7. 轮询任务状态（复用之前逻辑，API：GET /v1/videos/{id}）
            status_url = f"{base_url}/v1/videos/{task_id}"
            start_time = time.time()
            task_completed = False
            print(f"🔍 开始轮询任务状态（间隔{polling_interval}秒，超时{polling_max_time}秒）")

            while not task_completed:
                # 超时判断
                if time.time() - start_time > polling_max_time:
                    raise TimeoutError(f"任务超时（超过{polling_max_time}秒），任务ID：{task_id}")
                # 间隔等待
                time.sleep(polling_interval)

                # 发送状态查询请求
                try:
                    status_resp = requests.get(
                        status_url,
                        headers=request_headers,  # 复用基础请求头（Authorization+Accept）
                        timeout=30
                    )
                    if status_resp.status_code != 200:
                        print(f"⚠️ 查询状态失败（状态码：{status_resp.status_code}），重试...")
                        continue
                    status_data = status_resp.json()
                    current_status = status_data.get("status", "").lower()
                    current_progress = status_data.get("progress", 0)
                    print(f"📊 任务状态（ID：{task_id}）：{current_status}，进度：{current_progress}%")

                    # 任务失败判断
                    if current_status in ["failed", "error", "rejected"]:
                        error_reason = status_data.get("error", "未知原因")
                        raise RuntimeError(f"任务失败（状态：{current_status}），原因：{error_reason}，任务ID：{task_id}")
                    
                    # 任务成功判断（状态为completed/success 或 进度100%）
                    elif current_status in ["completed", "success"] or current_progress >= 100:
                        # 提取视频URL（API返回video_url字段）
                        video_url = status_data.get("video_url")
                        if not video_url:
                            raise RuntimeError(f"任务状态为成功，但未获取到视频URL，响应：{status_data}")
                        # 提取视频实际分辨率（覆盖默认尺寸）
                        video_size = status_data.get("size", f"{video_width}x{video_height}")
                        try:
                            video_width, video_height = map(int, video_size.split("x"))
                            print(f"📏 从API获取视频实际分辨率：{video_size}")
                        except Exception as e:
                            print(f"⚠️ 解析视频分辨率失败（{video_size}），使用默认值：{video_width}x{video_height}，错误：{str(e)}")
                        task_completed = True
                        print(f"🎉 任务完成！视频URL：{video_url[:50]}...，分辨率：{video_size}")

                except Exception as e:
                    print(f"⚠️ 轮询时出错：{str(e)}，重试...")
                    continue

            # 8. 下载视频到本地临时目录
            temp_dir = os.path.join(tempfile.gettempdir(), "comfyui_sora_15s_dual_output")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            local_video_path = os.path.join(temp_dir, f"sora_15s_{task_id}.mp4")

            try:
                print(f"📥 下载15秒视频（URL：{video_url[:50]}...）到本地：{local_video_path}")
                # 流式下载（避免大文件占用内存）
                dl_resp = requests.get(video_url, stream=True, timeout=300)
                dl_resp.raise_for_status()
                with open(local_video_path, "wb") as f:
                    for chunk in dl_resp.iter_content(chunk_size=1024*1024):  # 1MB分块下载
                        if chunk:
                            f.write(chunk)
                print(f"💾 15秒视频本地保存成功：{local_video_path}")
            except Exception as e:
                raise RuntimeError(f"下载视频失败：{str(e)}，视频URL：{video_url[:50]}...")

            # 9. 提取视频帧率（默认24，失败时用默认值）
            video_fps = 24.0
            try:
                reader = imageio.get_reader(local_video_path, 'ffmpeg')
                video_fps = reader.get_meta_data().get('fps', 24.0)
                reader.close()
                print(f"📊 提取视频帧率：{video_fps} FPS")
            except Exception as e:
                print(f"⚠️ 提取帧率失败，使用默认值24.0 FPS，错误：{str(e)}")

            # 10. 封装视频适配器（任务成功，状态设为1）
            video_adapter = ComflyVideoAdapter(
                video_url=video_url,
                video_path=local_video_path,
                fps=video_fps,
                width=video_width,
                height=video_height,
                task_status=1
            )
            # 更新任务状态
            task_status = 1
            error_msg = ""
            print(f"🏁 15秒双传图视频生成流程全部完成！")

        except Exception as e:
            # 异常处理：捕获所有错误，赋值错误信息
            error_msg = f"❌ 15秒视频生成失败：{str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            # 异常时保持返回值合法，避免节点崩溃

        # 最终返回（与RETURN_TYPES对应）
        return (video_adapter, video_url, task_status, error_msg)

# ---------------------- 节点映射配置（正确格式：类对象映射） ----------------------
NODE_CLASS_MAPPINGS = {
    "SoraVideoGenerator15sDualUpload": SoraVideoGenerator15sDualUpload  # 类名 → 类对象
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SoraVideoGenerator15sDualUpload": "Lu9_Sora2_15s_双传图版"  # 节点显示名称（区分原节点）
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]