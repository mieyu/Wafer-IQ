# 图像质量评估主入口：滑动窗口读图一次，分发给亮度、清晰度、位移偏移、畸变各模块，统一生成报告。
# 使用方法：python main.py <输入父目录> <输出父目录>
#   输入父目录下的每个子文件夹（如 BF_2_Wafer）将被逐一分析。
#   输出会自动按子文件夹名称对应到输出目录中（如 output/BF_2_Wafer/）。
#   示例：python main.py ../data ../output

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.image_quality_calculate import ImageQualityCalculator
from src.image_quality_reporter import ImageQualityReporter
from src.image_quality_data_utils import ImageQualityDataUtils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 固定参数（一般无需改动）
# ------------------------------------------------------------------
YAML_FILENAME    = "placements-BF.yml"   # YAML 配置文件名（坐标与重叠区像素数均从此解析）
STITCH_DIRECTION = "horizontal"          # 拼接方向，一般为 horizontal
NUM_WORKERS      = 8                     # 多线程分析并发数


# ------------------------------------------------------------------
# 单个子目录完整分析流程
# ------------------------------------------------------------------

def analyze_one(data_dir: Path, output_dir: Path) -> None:
    """对单个晶圆子目录执行完整的四项质量分析并输出报告。"""
    calc       = ImageQualityCalculator()
    reporter   = ImageQualityReporter(output_dir)
    data_utils = ImageQualityDataUtils()

    logger.info(f"══════════════════════════════════════════════")
    logger.info(f"开始分析：{data_dir.name}  →  {output_dir}")
    logger.info(f"══════════════════════════════════════════════")

    # ── 读图一次，滑动窗口计算所有指标 ────────────────────────────────
    df = calc.batch_calculate(
        data_dir,
        yaml_filename=YAML_FILENAME,
        stitch_direction=STITCH_DIRECTION,
        num_workers=NUM_WORKERS,
    )
    if df.empty:
        logger.warning(f"[{data_dir.name}] 未找到图像文件或 YAML 解析失败，跳过。")
        return

    # ── 亮度报告 ──────────────────────────────────────────────────────
    brightness_metrics = calc.get_brightness_metrics(df, data_dir=data_dir)
    brightness_outputs = reporter.generate_brightness_all(df, brightness_metrics)
    logger.info(f"[{data_dir.name}][亮度] 报告已生成：")
    for key, path in brightness_outputs.items():
        logger.info(f"  [{key}] {path}")

    # ── 清晰度报告 ────────────────────────────────────────────────────
    sharpness_metrics = calc.get_sharpness_metrics(df)
    grid_data = data_utils.build_grid(df, metric_names=["laplacian", "tenengrad", "fft", "brenner"])
    sharpness_outputs = reporter.generate_sharpness_all(df, sharpness_metrics, grid_data)
    logger.info(f"[{data_dir.name}][清晰度] 报告已生成：")
    for key, val in sharpness_outputs.items():
        if isinstance(val, list):
            for p in val:
                logger.info(f"  [heatmap] {p}")
        else:
            logger.info(f"  [{key}] {val}")

    # ── 位移偏移报告 ──────────────────────────────────────────────────
    shift_metrics = calc.get_shift_metrics(df)
    shift_outputs = reporter.generate_shift_all(df, shift_metrics)
    logger.info(f"[{data_dir.name}][位移偏移] 报告已生成：")
    for key, val in shift_outputs.items():
        if isinstance(val, list):
            for p in val:
                logger.info(f"  [heatmap] {p}")
        else:
            logger.info(f"  [{key}] {val}")

    # ── 畸变报告 ──────────────────────────────────────────────────────
    distortion_metrics = calc.get_distortion_metrics(df)
    distortion_outputs = reporter.generate_distortion_all(df, distortion_metrics)
    logger.info(f"[{data_dir.name}][畸变] 报告已生成：")
    for key, val in distortion_outputs.items():
        if isinstance(val, list):
            for p in val:
                logger.info(f"  [heatmap] {p}")
        else:
            logger.info(f"  [{key}] {val}")

    logger.info(f"✅ [{data_dir.name}] 分析完成！\n")


# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="晶圆图像质量批量分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python main.py ../data ../output
      → 自动扫描 data/ 下所有子文件夹，每个子文件夹对应输出到 output/<子文件夹名>/
""",
    )
    parser.add_argument("input_dir",  type=Path, help="输入父目录，包含若干晶圆子文件夹（如 data/）")
    parser.add_argument("output_dir", type=Path, help="输出父目录，各子文件夹结果按名称对应存入（如 output/）")
    args = parser.parse_args()

    input_root: Path  = args.input_dir.resolve()
    output_root: Path = args.output_dir.resolve()

    if not input_root.exists():
        logger.error(f"输入路径不存在：{input_root}")
        sys.exit(1)

    # 扫描输入父目录下所有一级子文件夹
    sub_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()])
    if not sub_dirs:
        logger.error(f"输入路径 {input_root} 下没有找到任何子文件夹！")
        sys.exit(1)

    logger.info(f"扫描到 {len(sub_dirs)} 个待处理子目录：")
    for d in sub_dirs:
        logger.info(f"  {d.name}")

    # 逐一处理每个子文件夹
    for sub_dir in sub_dirs:
        output_sub = output_root / sub_dir.name
        try:
            analyze_one(sub_dir, output_sub)
        except Exception as exc:
            logger.error(f"[{sub_dir.name}] 分析过程中出现异常，已跳过：{exc}", exc_info=True)

    logger.info(f"🎉 所有 {len(sub_dirs)} 个目录分析完成！输出位于：{output_root}")


if __name__ == "__main__":
    main()