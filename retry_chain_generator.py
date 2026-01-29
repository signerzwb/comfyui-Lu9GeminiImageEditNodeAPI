# ComfyUI 生成重试串联器（最多10个，严格1→10顺序，新增启用开关，彻底解决乱序）
import torch
from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

class ImageGenRetryChain10Strict:
    DESCRIPTION = "生成节点自动重试（最多10个，严格1→10顺序）：需手动开启动用开关，仅开启的节点参与判断，第一个成功输出图片，全部失败输出空图"
    CATEGORY = "自定义节点/生成重试"
    
    INPUT_TYPES = lambda: ({
        "optional": {
            # 核心新增：10个节点启用开关（布尔型，默认关闭，手动开启才参与判断）
            "enable1": ("BOOLEAN", {"default": False, "label": "启用节点1"}),
            "img1": ("IMAGE",),
            "status1": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
            
            "enable2": ("BOOLEAN", {"default": False, "label": "启用节点2"}),
            "img2": ("IMAGE",),
            "status2": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
            
            "enable3": ("BOOLEAN", {"default": False, "label": "启用节点3"}),
            "img3": ("IMAGE",),
            "status3": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
            
            "enable4": ("BOOLEAN", {"default": False, "label": "启用节点4"}),
            "img4": ("IMAGE",),
            "status4": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
            
            "enable5": ("BOOLEAN", {"default": False, "label": "启用节点5"}),
            "img5": ("IMAGE",),
            "status5": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
            
            "enable6": ("BOOLEAN", {"default": False, "label": "启用节点6"}),
            "img6": ("IMAGE",),
            "status6": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
            
            "enable7": ("BOOLEAN", {"default": False, "label": "启用节点7"}),
            "img7": ("IMAGE",),
            "status7": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
            
            "enable8": ("BOOLEAN", {"default": False, "label": "启用节点8"}),
            "img8": ("IMAGE",),
            "status8": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
            
            "enable9": ("BOOLEAN", {"default": False, "label": "启用节点9"}),
            "img9": ("IMAGE",),
            "status9": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
            
            "enable10": ("BOOLEAN", {"default": False, "label": "启用节点10"}),
            "img10": ("IMAGE",),
            "status10": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
        }
    })
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("final_image", "total_status")
    FUNCTION = "run_chain"

    def run_chain(self, 
                  # 节点1参数
                  enable1=False, img1=None, status1=0,
                  # 节点2参数
                  enable2=False, img2=None, status2=0,
                  # 节点3参数
                  enable3=False, img3=None, status3=0,
                  # 节点4参数
                  enable4=False, img4=None, status4=0,
                  # 节点5参数
                  enable5=False, img5=None, status5=0,
                  # 节点6参数
                  enable6=False, img6=None, status6=0,
                  # 节点7参数
                  enable7=False, img7=None, status7=0,
                  # 节点8参数
                  enable8=False, img8=None, status8=0,
                  # 节点9参数
                  enable9=False, img9=None, status9=0,
                  # 节点10参数
                  enable10=False, img10=None, status10=0):
        # 核心修复：严格按1→10顺序，仅「启用开关为True」的节点才参与判断
        gen_nodes = []
        # 节点1：启用则加入
        if enable1 and img1 is not None:
            gen_nodes.append(("节点1", img1, status1))
        # 节点2：启用则加入（仅在节点1未成功时执行）
        if enable2 and img2 is not None:
            gen_nodes.append(("节点2", img2, status2))
        # 节点3：启用则加入
        if enable3 and img3 is not None:
            gen_nodes.append(("节点3", img3, status3))
        # 节点4：启用则加入
        if enable4 and img4 is not None:
            gen_nodes.append(("节点4", img4, status4))
        # 节点5：启用则加入
        if enable5 and img5 is not None:
            gen_nodes.append(("节点5", img5, status5))
        # 节点6：启用则加入
        if enable6 and img6 is not None:
            gen_nodes.append(("节点6", img6, status6))
        # 节点7：启用则加入
        if enable7 and img7 is not None:
            gen_nodes.append(("节点7", img7, status7))
        # 节点8：启用则加入
        if enable8 and img8 is not None:
            gen_nodes.append(("节点8", img8, status8))
        # 节点9：启用则加入
        if enable9 and img9 is not None:
            gen_nodes.append(("节点9", img9, status9))
        # 节点10：启用则加入
        if enable10 and img10 is not None:
            gen_nodes.append(("节点10", img10, status10))

        # 若未启用任何节点，直接返回空图+失败状态
        if not gen_nodes:
            empty_img = torch.zeros((1,1,1,3), dtype=torch.float32)
            return (empty_img, 0)

        # 严格按1→10顺序执行：第一个成功则立即输出，终止后续所有节点
        final_img = None
        total_status = 0
        for node_name, img, status in gen_nodes:
            print(f"【重试串联器】开始执行{node_name}，当前状态：{status}（1成功/0失败）")
            if status == 1:
                final_img = img
                total_status = 1
                print(f"【重试串联器】{node_name}执行成功，立即输出图片，终止后续节点")
                break  # 强制终止循环，后续节点不再执行
            else:
                print(f"【重试串联器】{node_name}执行失败，继续下一个节点")

        # 所有启用的节点均失败，返回空图
        if final_img is None:
            print(f"【重试串联器】警告：所有{len(gen_nodes)}个启用节点均失败，输出空图")
            final_img = torch.zeros((1,1,1,3), dtype=torch.float32)

        return (final_img, total_status)

# 注册节点（新名称，方便识别）
NODE_CLASS_MAPPINGS["ImageGenRetryChain10Strict"] = ImageGenRetryChain10Strict
NODE_DISPLAY_NAME_MAPPINGS["ImageGenRetryChain10Strict"] = "Lu9生成节点严格顺序重试器（最多10个，带启用开关）"