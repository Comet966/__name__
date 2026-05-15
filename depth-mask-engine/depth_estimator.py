"""depth_estimator.py — 图片推理 → 深度图"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from transformers import DepthEstimationPipeline

logger = logging.getLogger(__name__)

# ── 环境设置 ──

import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def estimate_depth(
    image: np.ndarray,
    pipeline: "DepthEstimationPipeline",
) -> np.ndarray:
    """
    对单张图片做深度估计推理。

    参数:
        image:  (H, W, 3), BGR 或 RGB, uint8
        pipeline: 已加载的 depth-estimation pipeline
    返回:
        (H, W), float32, [0, 1], 1 = 最近, 0 = 最远
    """
    if image is None or image.size == 0:
        raise ValueError("输入图片为空")

    # 确保是 RGB（OpenCV 读出来是 BGR）
    if image.ndim == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb = image

    # Pipeline 需要 PIL Image
    pil_img = Image.fromarray(rgb)

    result = pipeline(pil_img)
    depth = np.array(result["depth"], dtype=np.float32)

    # 归一化到 [0, 1]
    dmin, dmax = depth.min(), depth.max()
    if dmax > dmin:
        depth = (depth - dmin) / (dmax - dmin)
    else:
        depth = np.zeros_like(depth)

    return depth


def estimate_depth_from_path(
    image_path: str,
    pipeline: "DepthEstimationPipeline",
) -> np.ndarray:
    """
    从文件路径读取图片并推理深度图。

    参数:
        image_path: 输入图片路径
        pipeline: 已加载的 pipeline
    返回:
        (H, W), float32, [0, 1]
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    return estimate_depth(img, pipeline)
