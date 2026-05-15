"""CLI 单元测试"""

import pytest
from depth_mask_engine.cli import build_parser, main


class TestBuildParser:
    """参数解析测试"""

    def test_parse_required_args(self):
        """解析必填参数"""
        parser = build_parser()
        args = parser.parse_args(["-i", "input.jpg", "-o", "mask.png"])
        assert args.input == "input.jpg"
        assert args.output == "mask.png"
        assert args.layers == 3  # 默认值
        assert args.method == "quantile"  # 默认值

    def test_parse_all_args(self):
        """解析所有参数"""
        parser = build_parser()
        args = parser.parse_args([
            "--input", "input.jpg",
            "--output", "mask.png",
            "--output-depth", "depth.png",
            "--layers", "5",
            "--method", "kmeans",
            "--device", "cpu",
            "--verbose",
        ])
        assert args.input == "input.jpg"
        assert args.output == "mask.png"
        assert args.output_depth == "depth.png"
        assert args.layers == 5
        assert args.method == "kmeans"
        assert args.device == "cpu"
        assert args.verbose is True

    def test_short_flags(self):
        """短参数别名"""
        parser = build_parser()
        args = parser.parse_args(["-i", "in.jpg", "-o", "out.png", "-l", "4", "-m", "kmeans", "-v"])
        assert args.input == "in.jpg"
        assert args.output == "out.png"
        assert args.layers == 4
        assert args.method == "kmeans"
        assert args.verbose is True

    def test_missing_input_raises(self):
        """缺少必填参数时报错"""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["-o", "mask.png"])

    def test_missing_output_raises(self):
        """缺少输出参数时报错"""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["-i", "input.jpg"])

    def test_invalid_method_raises(self):
        """无效 method 参数报错"""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["-i", "a.jpg", "-o", "b.png", "-m", "invalid"])


class TestMain:
    """main 函数集成测试"""

    @pytest.mark.slow
    def test_main_normal(self, tmp_path):
        """正常执行流程"""
        import cv2
        import numpy as np
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        in_path = str(tmp_path / "input.jpg")
        out_path = str(tmp_path / "mask.png")
        cv2.imwrite(in_path, img)

        exit_code = main(["-i", in_path, "-o", out_path, "--device", "cpu"])
        assert exit_code == 0
        saved = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
        assert saved is not None

    @pytest.mark.slow
    def test_main_with_depth_output(self, tmp_path):
        """输出深度图选项"""
        import cv2
        import numpy as np
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        in_path = str(tmp_path / "input.jpg")
        out_path = str(tmp_path / "mask.png")
        depth_path = str(tmp_path / "depth.png")
        cv2.imwrite(in_path, img)

        exit_code = main([
            "-i", in_path, "-o", out_path,
            "-d", depth_path, "--device", "cpu",
        ])
        assert exit_code == 0
        assert cv2.imread(depth_path) is not None

    @pytest.mark.slow
    def test_main_with_verbose(self, tmp_path):
        """verbose 模式"""
        import cv2
        import numpy as np
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        in_path = str(tmp_path / "input.jpg")
        out_path = str(tmp_path / "mask.png")
        cv2.imwrite(in_path, img)

        exit_code = main(["-i", in_path, "-o", out_path, "-v", "--device", "cpu"])
        assert exit_code == 0

    def test_main_nonexistent_input(self, tmp_path):
        """不存在的输入文件"""
        exit_code = main(["-i", "/nonexistent.jpg", "-o", str(tmp_path / "mask.png")])
        assert exit_code != 0  # 非零退出
