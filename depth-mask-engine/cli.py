"""cli.py — 命令行入口"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("depth_mask_engine")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m depth_mask_engine.cli",
        description="Depth Mask Engine — 景深识别与多层景深区域蒙版",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入图片路径",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="输出蒙版路径（PNG）",
    )
    parser.add_argument(
        "--output-depth", "-d",
        default=None,
        help="深度图输出路径（可选，PNG 伪彩色）",
    )
    parser.add_argument(
        "--layers", "-l",
        type=int,
        default=3,
        help="蒙版层数（默认: 3）",
    )
    parser.add_argument(
        "--method", "-m",
        choices=["quantile", "kmeans"],
        default="quantile",
        help="分割方法（默认: quantile）",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="推理设备: auto, cpu, cuda（默认: auto）",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细信息",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger("depth_mask_engine").setLevel(logging.INFO)
    else:
        logging.getLogger("depth_mask_engine").setLevel(logging.WARNING)

    try:
        from depth_mask_engine.depth_engine import depth_mask, get_depth_map

        # 生成蒙版
        result = depth_mask(
            input_path=args.input,
            output_path=args.output,
            layers=args.layers,
            method=args.method,
            device=args.device,
            verbose=args.verbose,
        )
        print(f"✅ 蒙版已保存: {result}")

        # 可选输出深度图
        if args.output_depth:
            depth = get_depth_map(args.input, device=args.device)
            # 将深度图映射到伪彩色
            import cv2
            import numpy as np
            depth_8u = (depth * 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
            cv2.imwrite(args.output_depth, depth_colored)
            print(f"✅ 深度图已保存: {args.output_depth}")

        return 0

    except Exception as e:
        logger.error(f"❌ 错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
