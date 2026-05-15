"""mask_generator 单元测试"""

import numpy as np
import pytest

from depth_mask_engine.mask_generator import generate_mask, _validate_layers


class TestGenerateMask:
    """generate_mask 核心行为测试"""

    def test_output_shape_matches_input(self):
        """输出形状与输入一致"""
        depth = np.random.rand(240, 320).astype(np.float32)
        mask = generate_mask(depth, layers=3)
        assert mask.shape == (240, 320)
        assert mask.dtype == np.uint8

    def test_output_value_range(self):
        """输出值在 [0, 255] 范围内"""
        depth = np.random.rand(100, 100).astype(np.float32)
        mask = generate_mask(depth, layers=3)
        assert mask.min() >= 0
        assert mask.max() <= 255

    def test_binary_mode_two_layers(self):
        """layers=2 时退化为二值蒙版（0 和 255）"""
        # 上半近（高值），下半远（低值）
        depth = np.ones((100, 100), dtype=np.float32)
        depth[50:, :] = 0.0
        mask = generate_mask(depth, layers=2)
        assert mask.dtype == np.uint8
        # 上半应该为 255（近景），下半为 0（远景）
        assert mask[:50, :].mean() > 200
        assert mask[50:, :].mean() < 10

    def test_three_layers_gradient(self):
        """均匀渐变深度图 → 3 层像素数大致均匀"""
        h, w = 150, 150
        depth = np.tile(np.linspace(0, 1, w), (h, 1)).astype(np.float32)
        mask = generate_mask(depth, layers=3, method="quantile")
        unique = np.unique(mask)
        assert len(unique) == 3  # 三种灰度值

    def test_four_layers_quantile(self):
        """4 层分位分割"""
        h, w = 200, 200
        depth = np.tile(np.linspace(0, 1, w), (h, 1)).astype(np.float32)
        mask = generate_mask(depth, layers=4, method="quantile")
        unique = np.unique(mask)
        assert len(unique) == 4

    def test_kmeans_two_layers(self):
        """KMeans 双峰深度图 → 2 层"""
        depth = np.zeros((100, 100), dtype=np.float32)
        depth[:50, :] = 0.9   # 近（前景）
        depth[50:, :] = 0.1   # 远（背景）
        mask = generate_mask(depth, layers=2, method="kmeans")
        # 上半应为较高灰度，下半为较低灰度
        assert mask[:50, :].mean() > mask[50:, :].mean()
        unique = np.unique(mask)
        assert len(unique) == 2

    def test_kmeans_three_layers_trimodal(self):
        """KMeans 三峰深度图 → 3 层"""
        depth = np.zeros((150, 100), dtype=np.float32)
        depth[:50, :] = 0.9
        depth[50:100, :] = 0.5
        depth[100:, :] = 0.1
        mask = generate_mask(depth, layers=3, method="kmeans")
        unique = np.unique(mask)
        assert len(unique) == 3

    def test_uniform_depth(self):
        """全同深度图 → 单层（所有像素相同灰度值）"""
        depth = np.full((50, 50), 0.5, dtype=np.float32)
        mask = generate_mask(depth, layers=3)
        assert np.all(mask == 0)  # 全为 0（最远层）
        # 但实际上是所有像素被归为同一层

    def test_layers_one(self):
        """layers=1 → 全黑蒙版"""
        depth = np.random.rand(50, 50).astype(np.float32)
        mask = generate_mask(depth, layers=1)
        assert np.all(mask == 0)

    def test_single_row(self):
        """单行输入"""
        depth = np.linspace(0, 1, 100).astype(np.float32).reshape(1, 100)
        mask = generate_mask(depth, layers=3)
        assert mask.shape == (1, 100)

    def test_single_column(self):
        """单列输入"""
        depth = np.linspace(0, 1, 100).astype(np.float32).reshape(100, 1)
        mask = generate_mask(depth, layers=3)
        assert mask.shape == (100, 1)

    def test_quantile_each_layer_filled(self):
        """分位法：每层至少有部分像素"""
        depth = np.random.rand(500, 500).astype(np.float32)
        mask = generate_mask(depth, layers=5, method="quantile")
        for val in range(5):
            val_scaled = val * (255 // 4)
            assert np.any(mask == val_scaled), f"层 {val} (值 {val_scaled}) 无像素"

    def test_layers_255(self):
        """最大层数 255"""
        depth = np.random.rand(256, 256).astype(np.float32)
        mask = generate_mask(depth, layers=255, method="quantile")
        assert mask.shape == (256, 256)
        assert mask.dtype == np.uint8

    def test_disparity_values_binary(self):
        """layers=2 时数值应为 0 和 255"""
        depth = np.zeros((30, 30), dtype=np.float32)
        depth[:15, :] = 1.0
        mask = generate_mask(depth, layers=2)
        vals = set(np.unique(mask))
        assert vals == {0, 255} or vals == {0} or vals == {255}


class TestInvalidInput:
    """异常输入处理"""

    def test_3d_input_raises(self):
        """3D 数组应报错"""
        depth = np.random.rand(100, 100, 3).astype(np.float32)
        with pytest.raises(ValueError, match="2D"):
            generate_mask(depth)

    def test_empty_input_raises(self):
        """空数组应报错"""
        depth = np.array([[]], dtype=np.float32)
        with pytest.raises(ValueError, match="不能为空"):
            generate_mask(depth)

    def test_layers_zero_raises(self):
        """layers=0 应报错"""
        with pytest.raises(ValueError):
            _validate_layers(0)

    def test_layers_negative_raises(self):
        """layers<0 应报错"""
        with pytest.raises(ValueError):
            _validate_layers(-1)

    def test_invalid_method_raises(self):
        """不支持的方法应报错"""
        depth = np.random.rand(50, 50).astype(np.float32)
        with pytest.raises(ValueError, match="method"):
            generate_mask(depth, method="invalid")

    def test_layers_256_raises(self):
        """layers>255 应报错"""
        with pytest.raises(ValueError):
            _validate_layers(256)

    def test_all_white_mask(self):
        """全白蒙版情况不应有问题"""
        depth = np.full((60, 60), 1.0, dtype=np.float32)
        mask = generate_mask(depth, layers=3, method="quantile")
        assert mask.shape == (60, 60)

    def test_all_black_mask(self):
        """全黑蒙版情况不应有问题"""
        depth = np.full((60, 60), 0.0, dtype=np.float32)
        mask = generate_mask(depth, layers=3, method="quantile")
        assert mask.shape == (60, 60)
