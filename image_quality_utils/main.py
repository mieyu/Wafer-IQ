# 图像质量评估主入口：滑动窗口读图一次，分发给亮度、清晰度、位移偏移、畸变各模块，统一生成报告。
# 使用方法：直接运行 python main.py，修改下方 DATA_DIR / OUTPUT_DIR 为实际路径即可。

from __future__ import annotations

import logging
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
# 配置
# ------------------------------------------------------------------
DATA_DIR         = Path("../data/BF_2_Wafer")
OUTPUT_DIR       = Path("output")
YAML_FILENAME    = "placements-BF.yml"   # YAML 配置文件名（坐标与重叠区像素数均从此解析）
STITCH_DIRECTION = "horizontal"
NUM_WORKERS      = 8


def main() -> None:
    calc     = ImageQualityCalculator()
    reporter = ImageQualityReporter(OUTPUT_DIR)
    data_utils = ImageQualityDataUtils()

    # ── 读图一次，滑动窗口计算所有指标 ────────────────────────────────
    logger.info("开始批量分析...")
    df = calc.batch_calculate(
        DATA_DIR,
        yaml_filename=YAML_FILENAME,
        stitch_direction=STITCH_DIRECTION,
        num_workers=NUM_WORKERS,
    )
    if df.empty:
        logger.warning(f"未找到图像文件，请检查路径：{DATA_DIR}")
        return

    # ── 亮度报告 ──────────────────────────────────────────────────────
    brightness_metrics = calc.get_brightness_metrics(df, data_dir=DATA_DIR)
    brightness_outputs = reporter.generate_brightness_all(df, brightness_metrics)
    logger.info("[亮度] 报告已生成：")
    for key, path in brightness_outputs.items():
        logger.info(f"  [{key}] {path}")

    # ── 清晰度报告 ────────────────────────────────────────────────────
    sharpness_metrics = calc.get_sharpness_metrics(df)
    grid_data = data_utils.build_grid(df, metric_names=["laplacian", "tenengrad", "fft", "brenner"])
    sharpness_outputs = reporter.generate_sharpness_all(df, sharpness_metrics, grid_data)
    logger.info("[清晰度] 报告已生成：")
    for key, val in sharpness_outputs.items():
        if isinstance(val, list):
            for p in val:
                logger.info(f"  [heatmap] {p}")
        else:
            logger.info(f"  [{key}] {val}")

    # ── 位移偏移报告 ──────────────────────────────────────────────────
    shift_metrics = calc.get_shift_metrics(df)
    shift_outputs = reporter.generate_shift_all(df, shift_metrics)
    logger.info("[位移偏移] 报告已生成：")
    for key, val in shift_outputs.items():
        if isinstance(val, list):
            for p in val:
                logger.info(f"  [heatmap] {p}")
        else:
            logger.info(f"  [{key}] {val}")

    # ── 畸变报告 ──────────────────────────────────────────────────────
    distortion_metrics = calc.get_distortion_metrics(df)
    distortion_outputs = reporter.generate_distortion_all(df, distortion_metrics)
    logger.info("[畸变] 报告已生成：")
    for key, val in distortion_outputs.items():
        if isinstance(val, list):
            for p in val:
                logger.info(f"  [heatmap] {p}")
        else:
            logger.info(f"  [{key}] {val}")

    logger.info("✅ 全部分析完成！")


if __name__ == "__main__":
    main()