import re
import torch
from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


class VideoGenRetryChain10StrictLazyV5_TextOnly:
    DESCRIPTION = (
        "严格1→10顺序重试（Lazy短路V5.1，视频版）：\n"
        "- 每路输入：videoN(VIDEO) + statusN(STRING)\n"
        "- status 按关键词/正则判断成功失败；空字符串可视为成功\n"
        "- 第一个成功立即输出并短路后续分支（后续采样不会被触发）\n"
        "- 输出 status_dump：汇总本次实际检查到的 status 文本（不触发未执行分支）"
    )
    CATEGORY = "自定义节点/生成重试"

    RETURN_TYPES = ("VIDEO", "INT", "INT", "STRING")
    RETURN_NAMES = ("final_video", "total_status", "selected_index", "status_dump")
    FUNCTION = "run_chain"

    _TEXT_UNSET = "__UNSET__"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "empty_string_as_success": ("BOOLEAN", {
                "default": True,
                "label": "空字符串视为成功（成功不输出、失败才输出时建议开启）"
            }),
        }
        for i in range(1, 11):
            optional[f"enable{i}"] = ("BOOLEAN", {"default": False, "label": f"启用节点{i}"})
            optional[f"video{i}"] = ("VIDEO", {"lazy": True})
            optional[f"status{i}"] = ("STRING", {"default": cls._TEXT_UNSET, "multiline": False, "lazy": True})
        return {"optional": optional}

    def _empty_video(self):
        # 空视频占位（ComfyUI里 video 实际是 tensor list / dict，给个安全占位）
        return None

    def _norm_text(self, s: str, max_len=2000) -> str:
        if s is None:
            return ""
        if len(s) > max_len:
            s = s[:max_len] + " ...[truncated]"
        return s

    def _parse_status_text(self, raw: str, empty_string_as_success: bool):
        if raw is None or raw == self._TEXT_UNSET:
            return (None, "no_status_text")

        s = self._norm_text(str(raw)).strip()

        if s == "":
            return (1, "empty_string=>success") if empty_string_as_success else (None, "empty_string=>unknown")

        low = s.lower()

        fail_keywords = [
            "error", "fail", "failed", "exception", "traceback",
            "cuda out of memory", "out of memory", "oom",
            "invalid", "nan", "inf", "abort",
            "失败", "错误", "异常", "报错", "崩溃", "中断", "显存不足",
            "unauthorized", "401", "403"
        ]
        if any(k in low for k in fail_keywords):
            return (0, f"fail_keyword: {s[:160]}")

        m = re.search(r"(\d+)\s*/\s*(\d+)\s*successful", low)
        if m:
            ok = int(m.group(1))
            total = int(m.group(2))
            if total <= 0:
                return (None, f"successful_parse(total<=0): {s[:160]}")
            if ok <= 0:
                return (0, f"{ok}/{total} successful => fail")
            if ok >= total:
                return (1, f"{ok}/{total} successful => success")
            return (1, f"{ok}/{total} successful => partial_success")

        success_keywords = [
            "success", "successful", "succeeded",
            "ok", "done", "completed", "complete",
            "finish", "finished",
            "成功", "完成", "已完成", "结束"
        ]
        if any(k in low for k in success_keywords):
            return (1, f"success_keyword: {s[:160]}")

        return (None, f"unknown_text: {s[:160]}")

    def check_lazy_status(self, **kwargs):
        needed = []
        empty_string_as_success = bool(kwargs.get("empty_string_as_success", True))

        for i in range(1, 11):
            if not kwargs.get(f"enable{i}", False):
                continue

            status_val = kwargs.get(f"status{i}", self._TEXT_UNSET)
            if status_val is None:
                needed.append(f"status{i}")
                break

            success, _ = self._parse_status_text(status_val, empty_string_as_success)

            if success == 1:
                if kwargs.get(f"video{i}", None) is None:
                    needed.append(f"video{i}")
                break

        return needed

    def run_chain(self, **kwargs):
        empty_string_as_success = bool(kwargs.get("empty_string_as_success", True))

        any_enabled = False
        selected_index = 0
        total_status = 0
        dump_lines = []

        for i in range(1, 11):
            if not kwargs.get(f"enable{i}", False):
                continue

            any_enabled = True
            status_val = kwargs.get(f"status{i}", self._TEXT_UNSET)
            video_val = kwargs.get(f"video{i}", None)

            success, reason = self._parse_status_text(status_val, empty_string_as_success)

            if status_val is None:
                show_text = "<None>"
            elif status_val == self._TEXT_UNSET:
                show_text = "<UNSET>"
            else:
                show_text = self._norm_text(str(status_val), max_len=600).replace("\n", "\\n")

            dump_lines.append(f"[节点{i}] success={success} reason={reason} status={show_text}")
            print(f"【视频重试串联器-LazyV5.1】节点{i}: success={success}, reason={reason}")

            if success == 1 and video_val is not None:
                selected_index = i
                total_status = 1
                print(f"【视频重试串联器-LazyV5.1】节点{i}成功：输出并短路后续")
                break

        if not any_enabled:
            return (self._empty_video(), 0, 0, "未启用任何节点")

        if total_status == 1:
            final_video = kwargs.get(f"video{selected_index}", None)
        else:
            final_video = self._empty_video()

        status_dump = "\n".join(dump_lines) if dump_lines else "无状态信息"
        return (final_video, total_status, selected_index, status_dump)


NODE_CLASS_MAPPINGS["VideoGenRetryChain10StrictLazyV5_TextOnly"] = VideoGenRetryChain10StrictLazyV5_TextOnly
NODE_DISPLAY_NAME_MAPPINGS["VideoGenRetryChain10StrictLazyV5_TextOnly"] = "Lu9严格顺序视频重试器（Lazy短路V5.1：文本判定）"
