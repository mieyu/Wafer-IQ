"""
-------
虚拟全景清晰度分析 — 主入口脚本

直接运行后，程序会提示在终端输入路径：
    python main.py

输出（保存在指定 output 目录）：
    明细数据.csv                        每张图块的四种清晰度指标值
    评分报告.txt                        四种指标的汇总统计数据
    晶圆全景图_*.png                    4 张空间热图
    随机抽样对比示例.png                四种指标分布直方图（合并为一张）
"""

import logging
import os
from pathlib import Path

from src.grid_mapper import build_grid
from src.tile_scanner import scan_tiles
from src.visualizer import plot_heatmaps, plot_histograms

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent.parent
DATA_ROOT_DIR = Path(os.getenv("WAFER_DATA_ROOT", WORKSPACE_DIR / "data")).expanduser().resolve()
OUTPUT_ROOT_DIR = Path(os.getenv("WAFER_OUTPUT_ROOT", WORKSPACE_DIR / "output")).expanduser().resolve()

# 配置日志格式，方便查看运行进度
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 指标中文名称（用于日志文件），含综合评分
METRIC_CN_NAMES = {
    "laplacian": "Laplacian 方差",
    "tenengrad": "Tenengrad 梯度能量",
    "fft": "FFT 高频能量比",
    "brenner": "Brenner 梯度",
}


def save_summary_log(df, output_path: Path) -> None:
    """
    计算四种指标的汇总统计量，并保存为文本日志文件。

    统计内容：图块总数、均值、标准差、最小值、最大值
    """
    log_path = output_path / "评分报告.txt"

    lines = []
    lines.append("=" * 60)
    lines.append("  晶圆清晰度分析 — 汇总统计报告")
    lines.append("=" * 60)
    lines.append(f"  有效图块总数：{len(df)}")
    lines.append("")

    for metric_name, cn_name in METRIC_CN_NAMES.items():
        col = df[metric_name].dropna()
        lines.append(f"── {cn_name} ──────────────────────────")
        lines.append(f"  样本数量  : {len(col)}")
        lines.append(f"  均    值  : {col.mean():.4f}")
        lines.append(f"  标准差    : {col.std():.4f}")
        lines.append(f"  最小值    : {col.min():.4f}")
        lines.append(f"  中位数    : {col.median():.4f}")
        lines.append(f"  最大值    : {col.max():.4f}")
        lines.append("")

    # 整体评分（两项独立评分）
    lines.append("=" * 60)
    lines.append("  整体评分")
    lines.append("=" * 60)

    # 1. 绝对清晰度评分（直接根据批次 FFT 均值换算）
    if "fft" in df.columns:
        fft_mean = df["fft"].dropna().mean()
        sharpness_score = fft_mean * 100.0
        lines.append(f"  [1] 整体清晰度评分 : {sharpness_score:.2f} 分")
        lines.append(
            "      (说明：基于 FFT 高频能量占比直接映射，分值 0~100，有绝对物理意义)"
        )

    # 2. 批内均匀性评分（百分制）
    from src.sharpness_metrics import compute_uniformity_score

    uniformity_score = compute_uniformity_score(df)
    lines.append(f"  [2] 清晰度均匀性评分: {uniformity_score:.2f} 分")
    lines.append(
        "      (说明：基于变异系数 CV 换算，分值 0~100。分数越高表示批次内各个区域清晰度越均匀一致)"
    )
    lines.append("")

    log_text = "\n".join(lines)
    log_path.write_text(log_text, encoding="utf-8")
    logger.info(f"  已保存统计日志：{log_path}")


def main():
    if not DATA_ROOT_DIR.exists():
        logger.error(f"data 目录不存在：{DATA_ROOT_DIR}")
        return
    dataset_dirs = sorted(path for path in DATA_ROOT_DIR.iterdir() if path.is_dir())
    if not dataset_dirs:
        logger.error(f"未在 data 目录下找到数据集文件夹：{DATA_ROOT_DIR}")
        return

    num_workers = 8  # 并行线程数
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        output_path = OUTPUT_ROOT_DIR / f"{dataset_name}_输出" / "清晰度检测"
        data_dir = str(dataset_dir)

        logger.info(f"[{dataset_name}] [1/3] 扫描图像目录：{data_dir}")
        records = scan_tiles(data_dir, num_workers=num_workers)
        if not records:
            logger.error(f"[{dataset_name}] 未找到任何有效图块，跳过。")
            continue

        logger.info(f"[{dataset_name}] [2/3] 构建虚拟坐标网格...")
        grid_data = build_grid(records)
        df = grid_data["df"]
        x_range = f"{df['x'].min()} ~ {df['x'].max()}"
        y_range = f"{df['y'].min()} ~ {df['y'].max()}"
        logger.info(
            f"[{dataset_name}] 网格尺寸：{len(grid_data['x_labels'])} × {len(grid_data['y_labels'])}  "
            f"(x: {x_range},  y: {y_range})"
        )

        logger.info(f"[{dataset_name}] [3/3] 保存结果到：{output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        csv_path = output_path / "明细数据.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"[{dataset_name}] 已保存 CSV：{csv_path}  ({len(df)} 行)")
        save_summary_log(df, output_path)
        plot_heatmaps(grid_data, output_dir=str(output_path))
        plot_histograms(grid_data, output_dir=str(output_path))

    logger.info("✅ 全部数据集处理完成！")


if __name__ == "__main__":
    main()
