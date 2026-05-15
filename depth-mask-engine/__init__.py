"""depth_mask_engine — 景深识别与多层景深区域蒙版后端

核心 API：
    depth_mask()                — 输入路径 → 输出蒙版文件
    depth_mask_from_array()     — numpy 图像 → numpy 蒙版（推荐给 CameraSim 使用）
    get_depth_map()             — 输入路径 → 深度图数组
    CameraSimDepthBridge        — 供 CameraSim 直接调用的桥接类
"""

from depth_mask_engine.depth_engine import depth_mask, depth_mask_from_array, get_depth_map
from depth_mask_engine.camerasim_bridge import CameraSimDepthBridge

__all__ = [
    "depth_mask",
    "depth_mask_from_array",
    "get_depth_map",
    "CameraSimDepthBridge",
]
