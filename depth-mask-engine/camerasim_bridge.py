"""camerasim_bridge.py — CameraSim 景深桥接层

为 CameraSim 提供基于深度学习真实景深的虚化效果。
相比 CameraSim 自带的 radial gradient 虚化（只按距画面中心距离），
本模块使用深度估计模型获取每像素的实际景深信息，实现物理正确的景深模糊。
"""

import logging
from typing import Literal

import cv2
import numpy as np

from depth_mask_engine.depth_engine import depth_mask_from_array, get_depth_map as _get_depth_map_raw

logger = logging.getLogger(__name__)


# ── 工具函数 ────────────────────────────────────────

def _gaussian_kernel_size(sigma: float) -> int:
    """根据 sigma 计算合适的高斯核大小（奇数）"""
    k = int(sigma * 6 + 1)
    return k if k % 2 == 1 else k + 1


# ── 桥接类 ──────────────────────────────────────────

class CameraSimDepthBridge:
    """CameraSim 景深桥接 — 一行代码接入真实景深虚化

    用法示例：
        >>> from depth_mask_engine import CameraSimDepthBridge

        # 1. 直接获取多层蒙版
        mask = CameraSimDepthBridge.compute_depth_mask(image_bgr, layers=5)

        # 2. 带真实景深的虚化效果（替换 CameraSim 的 simulate_aperture_blur）
        result = CameraSimDepthBridge.apply_bokeh(image_bgr, aperture=2.8)

        # 3. 获取原始深度图
        depth = CameraSimDepthBridge.compute_depth_map(image_bgr)
    """

    # ── 核心方法 ──

    @staticmethod
    def compute_depth_map(image: np.ndarray) -> np.ndarray:
        """计算图像深度图

        Args:
            image: BGR uint8 numpy 数组 (H, W, 3)
        Returns:
            (H, W) float32, [0, 1], 1=近景, 0=远景
        """
        return _get_depth_map_raw(image)

    @staticmethod
    def compute_depth_mask(image: np.ndarray, layers: int = 3,
                           method: Literal["quantile", "kmeans"] = "quantile") -> np.ndarray:
        """计算多层景深区域蒙版

        Args:
            image:  BGR uint8 (H, W, 3)
            layers: 蒙版层数（例如 3 → 近景/中景/远景）
            method: 分割方法
        Returns:
            (H, W) uint8, 像素值为 0~(layers-1)，对应不同深度区域
        """
        return depth_mask_from_array(image, layers=layers, method=method)

    @staticmethod
    def apply_bokeh(
        image: np.ndarray,
        aperture: float,
        focus_distance: float = 0.5,
        depth_smooth_sigma: float = 5.0,
        max_blur_sigma: float | None = None,
    ) -> np.ndarray:
        """基于真实深度信息应用景深虚化

        使用深度学习模型估算每像素实际深度，然后按深度施加不同强度的模糊。
        对焦平面内的物体保持清晰，离焦越远模糊越强。

        相比 CameraSim 的 simulate_aperture_blur()（径向渐变，仅按距离中心距离），
        本方法能正确处理画面中同一距离上不同深度的物体。

        Args:
            image:             BGR uint8 (H, W, 3)
            aperture:          光圈 f 值（1.4 ~ 22）
                              小 f（大光圈）→ 浅景深 → 强虚化
                              大 f（小光圈）→ 深景深 → 弱虚化
            focus_distance:    对焦距离，0~1（0=最近处，1=最远处，0.5=中间）
            depth_smooth_sigma: 深度图平滑程度，越大过渡越自然
            max_blur_sigma:    最大模糊强度（不传则根据光圈自动计算）

        Returns:
            (H, W, 3) BGR uint8 — 带真实景深虚化的图像

        集成到 CameraSim 的 ImageProcessor 示例：
            def process(self, img, aperture, ev_diff, iso):
                img = CameraSimDepthBridge.apply_bokeh(img, aperture)
                img = simulate_exposure(img, ev_diff)
                img = simulate_noise(img, iso)
                return img
        """
        h, w = image.shape[:2]

        # ── 1. 获取深度图 ──
        depth = _get_depth_map_raw(image)           # (H, W) float32, [0, 1]
        depth = cv2.GaussianBlur(depth, (0, 0), depth_smooth_sigma)  # 平滑

        # ── 2. 计算离焦量 ──
        # 对焦距离处 blur_weight=0，越远越大
        blur_weight = np.abs(depth - focus_distance)

        # ── 3. 计算模糊强度 ──
        if max_blur_sigma is None:
            # 光圈 f/1.4 → ~12, f/2.8 → ~6, f/5.6 → ~3, f/11 → ~1.5
            max_blur_sigma = max(1.0, 16.0 / max(aperture, 1.0))

        # 归一化：最大离焦处施加 max_blur_sigma
        blur_map = blur_weight * max_blur_sigma      # (H, W) float32

        # ── 4. 多尺度分层模糊 ──
        # 用 5 个离散模糊级别近似连续过渡，平衡效果与性能
        levels = [(0.0, 0.5), (0.5, 1.5), (1.5, 3.0), (3.0, 6.0), (6.0, 999)]
        result = np.zeros_like(image, dtype=np.float32)
        weight_sum = np.zeros((h, w, 3), dtype=np.float32)

        for low, high in levels:
            # 当前级别的像素权重（模糊强度在 low~high 区间内）
            weight = np.clip((blur_map - low) / (high - low), 0, 1)
            sigma = (low + high) / 2

            if sigma < 0.3:
                blurred = image.astype(np.float32)
            else:
                ks = _gaussian_kernel_size(sigma)
                ks = min(ks, min(h, w) // 2 * 2 + 1)  # 不超图尺寸
                if ks >= 3:
                    blurred = cv2.GaussianBlur(image, (ks, ks), sigma).astype(np.float32)
                else:
                    blurred = image.astype(np.float32)

            w3 = np.stack([weight, weight, weight], axis=-1)
            result += blurred * w3
            weight_sum += w3

        # 归一化防止权重重叠溢出
        result = np.divide(result, weight_sum, where=weight_sum > 0)
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    # ── 便捷方法 ──

    @staticmethod
    def apply_bokeh_by_layers(
        image: np.ndarray,
        aperture: float,
        layers: int = 5,
        focus_layer: int | None = None,
    ) -> np.ndarray:
        """按离散深度层施加虚化（省资源，适合实时预览）

        将图像分为 layers 个深度层，每层施加不同强度模糊。
        focus_layer 对应的层保持清晰。

        Args:
            image:       BGR uint8 (H, W, 3)
            aperture:    光圈 f 值
            layers:      深度层数
            focus_layer: 对焦层索引（默认取中间层）

        Returns:
            BGR uint8
        """
        if focus_layer is None:
            focus_layer = layers // 2

        mask = CameraSimDepthBridge.compute_depth_mask(image, layers=layers)
        max_sigma = max(1.0, 16.0 / max(aperture, 1.0))
        h, w = image.shape[:2]

        result = np.zeros_like(image, dtype=np.float32)
        weight_sum = np.zeros((h, w, 3), dtype=np.float32)

        for layer_id in range(layers):
            # 离对焦层越远模糊越强
            dist = abs(layer_id - focus_layer)
            sigma = max_sigma * (dist / max(layers - 1, 1))

            if sigma < 0.3:
                blurred = image.astype(np.float32)
            else:
                ks = _gaussian_kernel_size(sigma)
                ks = min(ks, min(h, w) // 2 * 2 + 1)
                if ks >= 3:
                    blurred = cv2.GaussianBlur(image, (ks, ks), sigma).astype(np.float32)
                else:
                    blurred = image.astype(np.float32)

            # 软边缘权重
            weight_layer = (mask == layer_id).astype(np.float32)
            weight_layer = cv2.GaussianBlur(weight_layer, (0, 0), 3.0)
            w3 = np.stack([weight_layer] * 3, axis=-1)

            result += blurred * w3
            weight_sum += w3

        result = np.divide(result, weight_sum, where=weight_sum > 0)
        return np.clip(result, 0, 255).astype(np.uint8)
