"""depth_engine 集成测试"""

import numpy as np
import pytest

from depth_mask_engine.depth_engine import (
    depth_mask,
    depth_mask_from_array,
    get_depth_map,
)


class TestDepthMaskFromArray:
    """depth_mask_from_array 测试（不涉及文件 I/O）"""

    @pytest.mark.slow
    def test_output_shape_and_type(self):
        """输出形状匹配 + uint8 类型"""
        img = np.ones((240, 320, 3), dtype=np.uint8) * 128
        mask = depth_mask_from_array(img, layers=3)
        assert mask.shape == (240, 320)
        assert mask.dtype == np.uint8

    @pytest.mark.slow
    def test_output_value_range(self):
        """输出值在 [0, 255]"""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        mask = depth_mask_from_array(img, layers=3)
        assert mask.min() >= 0
        assert mask.max() <= 255

    @pytest.mark.slow
    def test_layers_parameter_works(self):
        """layers 参数生效"""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        mask = depth_mask_from_array(img, layers=5)
        unique = np.unique(mask)
        # 至少有不同灰度值（不同层）
        assert len(unique) <= 5

    @pytest.mark.slow
    def test_kmeans_method(self):
        """KMeans 方法可用"""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        mask = depth_mask_from_array(img, layers=3, method="kmeans")
        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8

    def test_empty_image_raises(self):
        """空图片抛出异常"""
        with pytest.raises(ValueError, match="为空"):
            depth_mask_from_array(np.array([]), layers=3)


class TestDepthMask:
    """depth_mask 文件 I/O 集成测试"""

    @pytest.mark.slow
    def test_output_file_exists(self, tmp_path):
        """生成蒙版文件并保存"""
        import cv2
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        in_path = str(tmp_path / "input.jpg")
        out_path = str(tmp_path / "mask.png")
        cv2.imwrite(in_path, img)

        result = depth_mask(in_path, out_path, layers=3, verbose=False)
        assert result == out_path

        # 验证输出文件存在且可读
        saved = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
        assert saved is not None
        assert saved.shape == (100, 100)

    @pytest.mark.slow
    def test_layers_and_method_parameters(self, tmp_path):
        """layers 和 method 参数生效"""
        import cv2
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        in_path = str(tmp_path / "input.jpg")
        out_path = str(tmp_path / "mask.png")
        cv2.imwrite(in_path, img)

        depth_mask(in_path, out_path, layers=4, method="quantile", verbose=False)
        saved = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
        assert saved is not None

    def test_nonexistent_input_raises(self, tmp_path):
        """不存在的输入路径抛出异常"""
        with pytest.raises(FileNotFoundError):
            depth_mask("/nonexistent/image.jpg", str(tmp_path / "mask.png"))


class TestGetDepthMap:
    """get_depth_map 测试"""

    @pytest.mark.slow
    def test_output_shape_and_range(self, tmp_path):
        """输出深度图形状匹配 + 值域 [0, 1]"""
        import cv2
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        in_path = str(tmp_path / "input.jpg")
        cv2.imwrite(in_path, img)

        depth = get_depth_map(in_path)
        assert depth.shape == (100, 100)
        assert depth.dtype == np.float32
        assert depth.min() >= 0.0
        assert depth.max() <= 1.0

    def test_nonexistent_path_raises(self):
        """不存在的路径抛出异常"""
        with pytest.raises(FileNotFoundError):
            get_depth_map("/nonexistent/image.jpg")
