"""model_loader + depth_estimator 单元测试"""

import numpy as np
import pytest

from depth_mask_engine.model_loader import (
    load_depth_model,
    get_default_pipeline,
    MODEL_VARIANTS,
)
from depth_mask_engine.depth_estimator import (
    estimate_depth,
    estimate_depth_from_path,
)


class TestModelLoader:
    """模型加载功能测试"""

    def test_module_importable(self):
        """模块可以正常导入"""
        from depth_mask_engine import model_loader
        assert model_loader is not None

    def test_model_variants_defined(self):
        """模型变体定义完整"""
        assert "small" in MODEL_VARIANTS
        assert "base" in MODEL_VARIANTS
        assert "large" in MODEL_VARIANTS

    def test_invalid_variant_raises(self):
        """不支持的变体抛出异常"""
        with pytest.raises(ValueError, match="不支持的变体"):
            load_depth_model(variant="invalid")

    @pytest.mark.slow
    def test_load_small_model(self):
        """加载 Small 模型（实际下载+缓存）"""
        pipe = load_depth_model(variant="small", device="cpu")
        assert pipe is not None
        # 检查 get_default_pipeline 有值
        assert get_default_pipeline() is pipe

    @pytest.mark.slow
    def test_load_and_cache(self):
        """重复调用返回同一 pipeline（单例）"""
        pipe1 = load_depth_model(variant="small", device="cpu")
        pipe2 = load_depth_model(variant="small", device="cpu")
        assert pipe1 is pipe2


class TestDepthEstimator:
    """深度估计推理测试"""

    @pytest.fixture(scope="class")
    def pipeline(self):
        """加载模型（实际推理用）"""
        return load_depth_model(variant="small", device="cpu")

    @pytest.mark.slow
    def test_estimate_depth_output_shape(self, pipeline):
        """推理输出形状与输入一致"""
        img = np.ones((240, 320, 3), dtype=np.uint8) * 128
        depth = estimate_depth(img, pipeline)
        assert depth.shape == (240, 320)
        assert depth.dtype == np.float32

    @pytest.mark.slow
    def test_depth_value_range(self, pipeline):
        """深度值在 [0, 1] 范围内"""
        img = np.ones((240, 320, 3), dtype=np.uint8) * 128
        depth = estimate_depth(img, pipeline)
        assert depth.min() >= 0.0
        assert depth.max() <= 1.0

    @pytest.mark.slow
    def test_depth_non_uniform_image(self, pipeline):
        """半黑半白图 → 不同区域深度值不同"""
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        img[:, :160, :] = 255  # 左半白
        depth = estimate_depth(img, pipeline)
        # 左右半区应该有不同的深度（越亮通常越近）
        left_mean = depth[:, :160].mean()
        right_mean = depth[:, 160:].mean()
        # 由于模型足够好，两边应该有差异
        assert abs(left_mean - right_mean) > 0.01

    @pytest.mark.slow
    def test_rgb_input(self, pipeline):
        """RGB 输入（非 BGR）也能正确处理"""
        rgb = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # 模拟 RGB（blue channel 不为零来判断没有做 BGR→RGB 转换）
        rgb[:, :, 0] = 0   # R=0
        rgb[:, :, 2] = 255 # B=255
        depth_rgb = estimate_depth(rgb, pipeline)
        assert depth_rgb.shape == (100, 100)

    @pytest.mark.slow
    def test_grayscale_input(self, pipeline):
        """灰度图 (H,W) 输入"""
        gray = np.ones((100, 100), dtype=np.uint8) * 128
        depth = estimate_depth(gray, pipeline)
        assert depth.shape == (100, 100)

    @pytest.mark.slow
    def test_estimate_depth_from_path(self, pipeline, tmp_path):
        """从文件路径读取并推理"""
        import cv2
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        path = str(tmp_path / "test.jpg")
        cv2.imwrite(path, img)
        depth = estimate_depth_from_path(path, pipeline)
        assert depth.shape == (100, 100)

    def test_empty_image_raises(self, pipeline):
        """空图片抛出异常"""
        with pytest.raises(ValueError, match="为空"):
            estimate_depth(np.array([]), pipeline)

    def test_none_image_raises(self, pipeline):
        """None 输入抛出异常"""
        with pytest.raises(ValueError, match="为空"):
            estimate_depth(None, pipeline)

    def test_nonexistent_path_raises(self, pipeline):
        """不存在的路径抛出异常"""
        with pytest.raises(FileNotFoundError):
            estimate_depth_from_path("/nonexistent/image.jpg", pipeline)
