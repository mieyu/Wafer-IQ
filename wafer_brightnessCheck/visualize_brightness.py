"""
晶圆亮度一致性可视化工具

提供可视化功能
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 设置样式
sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


class BrightnessVisualizer:
    """亮度可视化工具"""

    def __init__(self, stats_csv: str, output_dir: str = "brightness_results"):
        """
        初始化可视化工具

        Args:
            stats_csv: 统计数据CSV文件路径
            output_dir: 输出目录
        """
        self.df = pd.read_csv(stats_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 加载指标（如果存在）
        metrics_path = self.output_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)
        else:
            self.metrics = None

    def generate_visualizations(self):
        """
        生成可视化图表
        """
        # 只显示 normal 区域（非背景区域）
        if "region_type" in self.df.columns:
            normal_df = self.df[self.df["region_type"] == "normal"]
        else:
            # 如果没有region_type列，使用is_valid
            normal_df = self.df[self.df["is_valid"] == True]

        # 创建图表
        plt.figure(figsize=(20, 12))

        # 1. 亮度热图
        ax1 = plt.subplot(2, 3, 1)
        self._plot_heatmap(ax1, normal_df)

        # 2. 亮度分布直方图
        ax2 = plt.subplot(2, 3, 2)
        self._plot_histogram(ax2, normal_df)

        # 3. X方向亮度趋势
        ax3 = plt.subplot(2, 3, 3)
        self._plot_x_trend(ax3, normal_df)

        # 4. Y方向亮度趋势
        ax4 = plt.subplot(2, 3, 4)
        self._plot_y_trend(ax4, normal_df)

        # 5. 异常标注图
        ax5 = plt.subplot(2, 3, 5)
        self._plot_outliers(ax5, normal_df)

        # 6. 统计箱线图
        ax6 = plt.subplot(2, 3, 6)
        self._plot_boxplot(ax6, normal_df)

        plt.tight_layout()

        # 保存图片
        viz_path = self.output_dir / "brightness_analysis.png"
        plt.savefig(viz_path, dpi=300, bbox_inches="tight")
        print(f"可视化图表已保存至: {viz_path}")

        plt.show()

    def _plot_heatmap(self, ax, df):
        """绘制亮度热图"""
        # 创建网格
        x_coords = sorted(df["x"].unique())
        y_coords = sorted(df["y"].unique())

        grid = np.full((len(y_coords), len(x_coords)), np.nan)

        for idx, row in df.iterrows():
            xi = x_coords.index(row["x"])
            yi = y_coords.index(row["y"])
            grid[yi, xi] = row["valid_mean"]

        im = ax.imshow(
            grid, cmap="viridis", aspect="auto", interpolation="nearest", origin="lower"
        )
        ax.set_title("晶圆亮度热图", fontsize=14, fontweight="bold")
        ax.set_xlabel("X 坐标")
        ax.set_ylabel("Y 坐标")

        # 设置刻度标签（只显示部分）
        x_step = max(1, len(x_coords) // 10)
        y_step = max(1, len(y_coords) // 10)
        ax.set_xticks(range(0, len(x_coords), x_step))
        ax.set_xticklabels(
            [x_coords[i] for i in range(0, len(x_coords), x_step)], rotation=45
        )
        ax.set_yticks(range(0, len(y_coords), y_step))
        ax.set_yticklabels([y_coords[i] for i in range(0, len(y_coords), y_step)])

        plt.colorbar(im, ax=ax, label="平均亮度")

    def _plot_histogram(self, ax, df):
        """绘制亮度分布直方图"""
        brightness = df["valid_mean"]

        ax.hist(brightness, bins=50, color="skyblue", edgecolor="black", alpha=0.7)

        # 标注均值和中位数
        mean_val = brightness.mean()
        median_val = brightness.median()

        ax.axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"均值: {mean_val:.2f}",
        )
        ax.axvline(
            median_val,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"中位数: {median_val:.2f}",
        )

        ax.set_title("亮度分布直方图", fontsize=14, fontweight="bold")
        ax.set_xlabel("亮度值")
        ax.set_ylabel("频数")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_x_trend(self, ax, df):
        """绘制X方向亮度趋势"""
        x_trend = df.groupby("x")["valid_mean"].agg(["mean", "std"])

        ax.plot(x_trend.index, x_trend["mean"], marker="o", linewidth=2, markersize=4)
        ax.fill_between(
            x_trend.index,
            x_trend["mean"] - x_trend["std"],
            x_trend["mean"] + x_trend["std"],
            alpha=0.3,
        )

        ax.set_title("X 方向亮度趋势", fontsize=14, fontweight="bold")
        ax.set_xlabel("X 坐标")
        ax.set_ylabel("平均亮度")
        ax.grid(True, alpha=0.3)

    def _plot_y_trend(self, ax, df):
        """绘制Y方向亮度趋势"""
        y_trend = df.groupby("y")["valid_mean"].agg(["mean", "std"])

        ax.plot(
            y_trend.index,
            y_trend["mean"],
            marker="o",
            linewidth=2,
            markersize=4,
            color="orange",
        )
        ax.fill_between(
            y_trend.index,
            y_trend["mean"] - y_trend["std"],
            y_trend["mean"] + y_trend["std"],
            alpha=0.3,
            color="orange",
        )

        ax.set_title("Y 方向亮度趋势", fontsize=14, fontweight="bold")
        ax.set_xlabel("Y 坐标")
        ax.set_ylabel("平均亮度")
        ax.grid(True, alpha=0.3)

    def _plot_outliers(self, ax, df):
        """绘制异常标注图"""
        # 计算异常阈值
        mean_val = df["valid_mean"].mean()
        std_val = df["valid_mean"].std()

        # 分类
        normal = df[
            (df["valid_mean"] >= mean_val - std_val)
            & (df["valid_mean"] <= mean_val + 2 * std_val)
        ]
        dark = df[
            (df["valid_mean"] < mean_val - std_val)
            & (df["valid_mean"] >= mean_val - 3 * std_val)
        ]
        very_dark = df[(df["valid_mean"] < mean_val - 3 * std_val)]
        bright = df[(df["valid_mean"] > mean_val + 2 * std_val)]

        # 绘制散点
        ax.scatter(normal["x"], normal["y"], c="green", s=20, alpha=0.5, label="正常")
        ax.scatter(dark["x"], dark["y"], c="orange", s=30, alpha=0.7, label="较暗")
        ax.scatter(
            very_dark["x"], very_dark["y"], c="red", s=50, marker="X", label="异常暗"
        )
        ax.scatter(bright["x"], bright["y"], c="blue", s=50, marker="*", label="异常亮")

        ax.set_title("异常区域标注图", fontsize=14, fontweight="bold")
        ax.set_xlabel("X 坐标")
        ax.set_ylabel("Y 坐标")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_boxplot(self, ax, df):
        """绘制亮度箱线图"""
        data_to_plot = [df["valid_mean"]]

        bp = ax.boxplot(data_to_plot, labels=["亮度"], patch_artist=True)

        # 设置颜色
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")

        ax.set_title("亮度统计箱线图", fontsize=14, fontweight="bold")
        ax.set_ylabel("亮度值")
        ax.grid(True, alpha=0.3, axis="y")


def main():
    """主函数"""
    stats_csv = "brightness_1_results/brightness_stats.csv"
    output_dir = "brightness_1_results"

    viz = BrightnessVisualizer(stats_csv, output_dir)

    # 生成可视化图表
    viz.generate_visualizations()
    print("✓ 可视化图表生成完成")


if __name__ == "__main__":
    main()
