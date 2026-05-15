"""model_loader.py — 加载 Depth Anything V2 模型"""

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import DepthEstimationPipeline

logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# 可用模型变体（小/中/大）
MODEL_VARIANTS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}

_default_pipeline = None


def load_depth_model(
    variant: str = "small",
    device: str = "auto",
) -> "DepthEstimationPipeline":
    """
    加载 Depth Anything V2 模型。

    参数:
        variant: "small" | "base" | "large"
        device: "auto" | "cuda" | "cpu"
    返回:
        深度估计 pipeline 对象
    """
    global _default_pipeline

    # 先检查变体是否有效
    model_id = MODEL_VARIANTS.get(variant)
    if model_id is None:
        raise ValueError(
            f"不支持的变体: {variant}，可用: {list(MODEL_VARIANTS.keys())}"
        )

    # 如果已加载过相同变体，直接返回缓存
    if _default_pipeline is not None:
        return _default_pipeline

    # 设备选择
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(
        f"加载 Depth-Anything-V2-{variant.capitalize()} (设备: {device})..."
    )

    from transformers import pipeline

    pipe = pipeline(
        task="depth-estimation",
        model=model_id,
        device=device,
    )

    _default_pipeline = pipe
    return pipe


def get_default_pipeline() -> "DepthEstimationPipeline | None":
    """获取已缓存的默认 pipeline"""
    return _default_pipeline
