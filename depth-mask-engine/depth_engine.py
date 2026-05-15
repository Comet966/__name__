"""depth_engine.py — 高层 API，组合模型加载 + 推理 + 蒙版生成"""

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from depth_mask_engine.depth_estimator import estimate_depth
from depth_mask_engine.mask_generator import generate_mask

logger = logging.getLogger(__name__)

# 模块级缓存：只加载一次模型
_pipeline = None


def _get_pipeline(device: str = "auto") -> "DepthEstimationPipeline":
    """获取（并缓存）模型 pipeline"""
    global _pipeline
    if _pipeline is None:
        from depth_mask_engine.model_loader import load_depth_model
        _pipeline = load_depth_model(variant="small", device=device)
    return _pipeline


def depth_mask(
    input_path: str,
    output_path: str,
    layers: int = 3,
    method: str = "quantile",
    device: str = "auto",
    verbose: bool = True,
) -> str:
    """
    输入图片 → 深度估计 → 多层蒙版 → 保存 PNG。

    参数:
        input_path:  输入图片路径
        output_path: 输出蒙版路径（PNG）
        layers:      蒙版层数（默认 3）
        method:      分割方法（"quantile" | "kmeans"）
        device:      推理设备（"auto" | "cuda" | "cpu"）
        verbose:     是否打印耗时信息
    返回:
        output_path 字符串
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"输入图片不存在: {input_path}")

    t_start = time.time()

    pipeline = _get_pipeline(device)
    img = cv2.imread(str(input_path))
    if img is None:
        raise ValueError(f"无法读取图片: {input_path}")

    t_load = time.time()
    depth = estimate_depth(img, pipeline)
    t_depth = time.time()
    mask = generate_mask(depth, layers=layers, method=method)
    t_mask = time.time()

    # 处理输出路径
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask)

    if verbose:
        logger.info(
            f"景深蒙版完成 | "
            f"加载: {t_load - t_start:.2f}s | "
            f"推理: {t_depth - t_load:.2f}s | "
            f"蒙版: {t_mask - t_depth:.2f}s | "
            f"总计: {t_mask - t_start:.2f}s | "
            f"输出: {output_path}"
        )

    return str(output_path)


def depth_mask_from_array(
    image: np.ndarray,
    layers: int = 3,
    method: str = "quantile",
    device: str = "auto",
) -> np.ndarray:
    """
    输入 numpy 图像 → 返回 numpy 蒙版（不读写文件）。

    参数:
        image:  (H, W, 3), BGR, uint8
        layers: 蒙版层数
        method: 分割方法
        device: 推理设备
    返回:
        (H, W), uint8 蒙版
    """
    if image is None or image.size == 0:
        raise ValueError("输入图片为空")

    pipeline = _get_pipeline(device)
    depth = estimate_depth(image, pipeline)
    return generate_mask(depth, layers=layers, method=method)


def get_depth_map(
    input_source: str | np.ndarray,
    device: str = "auto",
) -> np.ndarray:
    """
    仅返回深度图（不做蒙版分割）。

    参数:
        input_source: 输入图片路径 或 BGR uint8 numpy 数组
        device:       推理设备
    返回:
        (H, W), float32, [0, 1]
    """
    if isinstance(input_source, np.ndarray):
        img = input_source
    else:
        p = Path(input_source)
        if not p.exists():
            raise FileNotFoundError(f"输入图片不存在: {p}")
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"无法读取图片: {p}")

    pipeline = _get_pipeline(device)
    return estimate_depth(img, pipeline)
