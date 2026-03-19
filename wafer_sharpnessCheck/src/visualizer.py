"""
visualizer.py
-------------
根据虚拟网格数据绘制：
  1. 四种清晰度指标的空间热图
  2. 四种清晰度指标的分布直方图

热图说明：
  - 横轴：x 坐标（图块列）
  - 纵轴：y 坐标（图块行）
  - 颜色：指标值（越亮 = 越清晰）
  - NaN 位置显示为灰色（表示该位置无图块）

直方图说明：
  - 横轴：指标分数
  - 纵轴：处于该分数段的图块数目
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ── 中文字体配置（Windows 使用 Microsoft YaHei，避免中文乱码）──────────────
matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False  # 正常显示负号

# 四个指标的中文显示名称
METRIC_DISPLAY_NAMES = {
    "laplacian": "Laplacian 方差",
    "tenengrad": "Tenengrad 梯度能量",
    "fft": "FFT 高频能量比",
    "brenner": "Brenner 梯度",
}
METRIC_FILE_NAMES = {
    "laplacian": "Laplacian方差",
    "tenengrad": "Tenengrad梯度能量",
    "fft": "FFT高频能量比",
    "brenner": "Brenner梯度",
}


def plot_heatmaps(grid_data: dict, output_dir: str) -> None:
    """
    为每个指标绘制空间热图，保存到 output_dir。

    参数：
        grid_data:  build_grid() 的返回值
        output_dir: 输出文件夹路径（不存在会自动创建）
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    x_labels = grid_data["x_labels"]
    y_labels = grid_data["y_labels"]
    grids = grid_data["grids"]

    for metric_name, grid in grids.items():
        fig, ax = plt.subplots(figsize=(10, 8))

        # NaN 位置显示为浅灰色
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="lightgray")

        im = ax.imshow(
            grid,
            cmap=cmap,
            aspect="auto",
            origin="lower",  # y 轴从下到上递增
        )

        # 坐标轴刻度：每隔一定间距显示一个，避免过密
        x_step = max(1, len(x_labels) // 15)
        y_step = max(1, len(y_labels) // 15)

        ax.set_xticks(range(0, len(x_labels), x_step))
        ax.set_xticklabels(
            [str(x_labels[i]) for i in range(0, len(x_labels), x_step)],
            rotation=45,
            ha="right",
        )
        ax.set_yticks(range(0, len(y_labels), y_step))
        ax.set_yticklabels([str(y_labels[i]) for i in range(0, len(y_labels), y_step)])

        display_name = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
        ax.set_title(f"晶圆清晰度热图 — {display_name}", fontsize=14, pad=12)
        ax.set_xlabel("X 坐标（图块列）", fontsize=11)
        ax.set_ylabel("Y 坐标（图块行）", fontsize=11)

        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label(display_name, fontsize=10)

        fig.tight_layout()
        metric_file_name = METRIC_FILE_NAMES.get(metric_name, metric_name)
        save_path = output_path / f"晶圆全景图_{metric_file_name}.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  已保存热图：{save_path}")


def plot_histograms(grid_data: dict, output_dir: str, bins: int = 50) -> None:
    """
    将四种指标的分布直方图画在同一张图（2×2 布局），保存为 histogram_all.png。

    横轴：指标分数（连续值）
    纵轴：处于该分数段的图块数目

    参数：
        grid_data:  build_grid() 的返回值
        output_dir: 输出文件夹路径
        bins:       直方图的分箱数量（默认 50）
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = grid_data["df"]
    metric_names = list(METRIC_DISPLAY_NAMES.keys())  # 保证顺序固定

    # 创建 2 行 × 2 列的子图画布
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("晶圆清晰度 — 四种指标分布直方图", fontsize=15, y=1.01)

    for ax, metric_name in zip(axes.flat, metric_names):
        values = df[metric_name].dropna().values
        display_name = METRIC_DISPLAY_NAMES[metric_name]

        # 绘制直方图
        ax.hist(values, bins=bins, color="#4C9BE8", edgecolor="white", linewidth=0.5)

        # 标注均值线
        mean_val = values.mean()
        ax.axvline(
            mean_val,
            color="#E85C4C",
            linewidth=1.5,
            linestyle="--",
            label=f"均值 = {mean_val:.2f}",
        )

        ax.set_title(display_name, fontsize=12)
        ax.set_xlabel("分数", fontsize=10)
        ax.set_ylabel("图块数量", fontsize=10)
        ax.legend(fontsize=9)

    fig.tight_layout()
    save_path = output_path / "随机抽样对比示例.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存直方图：{save_path}")
