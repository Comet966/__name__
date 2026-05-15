"""mask_generator.py — 深度图 → 多层景深区域蒙版"""

import numpy as np


def generate_mask(
    depth_map: np.ndarray,
    layers: int = 3,
    method: str = "quantile",
) -> np.ndarray:
    """
    depth_map: (H, W), float32, [0, 1], 1=最近
    layers:    输出层数（默认 3）
    method:    "quantile" | "kmeans"
    返回:      (H, W), uint8, 值 0~255 均匀分布
    """
    if depth_map.ndim != 2:
        raise ValueError(f"depth_map 必须是 2D 数组，收到 {depth_map.ndim}D")
    if depth_map.size == 0:
        raise ValueError("depth_map 不能为空")

    h, w = depth_map.shape
    depth_flat = depth_map.ravel().astype(np.float64)

    if method == "quantile":
        _validate_layers(layers)
        # 对于所有像素深度值相同的特殊情况，直接整体赋同一层
        if depth_flat.max() == depth_flat.min():
            labels = np.zeros(depth_flat.shape, dtype=np.int32)
        else:
            bin_edges = np.linspace(0, 1, layers + 1)
            # 使用分位数作为边界，确保每层像素数大致相等
            quantile_edges = np.quantile(depth_flat, bin_edges[1:-1])
            labels = np.zeros(depth_flat.shape, dtype=np.int32)
            for i, edge in enumerate(quantile_edges):
                labels[depth_flat > edge] = i + 1
    elif method == "kmeans":
        _validate_layers(layers)
        labels = _kmeans_1d(depth_flat, layers)
    else:
        raise ValueError(f"不支持的 method: {method}，可用: quantile, kmeans")

    # 将标签 0..layers-1 映射到 0..255
    if layers > 1:
        step = 255 // (layers - 1)
        mask = (labels * step).astype(np.uint8).reshape(h, w)
    else:
        mask = np.zeros((h, w), dtype=np.uint8)

    return mask


# ── 辅助函数 ──

def _validate_layers(layers: int):
    if layers < 1:
        raise ValueError(f"layers 必须 >= 1，收到 {layers}")
    if layers > 255:
        raise ValueError(f"layers 最大 255，收到 {layers}")


def _kmeans_1d(data: np.ndarray, k: int, max_iter: int = 20) -> np.ndarray:
    """对一维数组做 KMeans 聚类"""
    # 初始化：在值域内均匀选取初始中心
    vmin, vmax = data.min(), data.max()
    if vmax == vmin:
        return np.zeros(data.shape, dtype=np.int32)
    centers = np.linspace(vmin, vmax, k)

    for _ in range(max_iter):
        # 分配标签
        distances = np.abs(data[:, None] - centers[None, :])  # (N, k)
        labels = np.argmin(distances, axis=1)

        # 更新中心
        new_centers = np.array([
            data[labels == i].mean() if np.any(labels == i) else centers[i]
            for i in range(k)
        ])

        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    return labels
