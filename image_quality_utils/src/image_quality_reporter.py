# 图像质量报告输出模块：负责将亮度、清晰度等计算结果生成 CSV / JSON / TXT / PNG 等报告文件。
# 本模块不含任何计算逻辑，只负责格式化输出，接收外部传入的 df 和 metrics。
# 使用方法：reporter = ImageQualityReporter(output_dir); reporter.generate_all(stats_df, metrics)

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 清晰度指标的中文显示名称
SHARPNESS_METRIC_CN = {
    "laplacian": "Laplacian 方差",
    "tenengrad": "Tenengrad 梯度能量",
    "fft":       "FFT 高频能量比",
    "brenner":   "Brenner 梯度",
}

class _ReportBuilder:
    WIDTH = 60

    def __init__(self, title: str) -> None:
        self._lines: list[str] = []
        self.divider()
        self._lines.append(f"  {title}")
        self.divider()

    def divider(self) -> "_ReportBuilder":
        self._lines.append("=" * self.WIDTH)
        return self

    def blank(self) -> "_ReportBuilder":
        self._lines.append("")
        return self

    def section(self, title: str) -> "_ReportBuilder":
        self._lines.append(f"  ── {title} ──")
        return self

    def field(self, label: str, value: str) -> "_ReportBuilder":
        self._lines.append(f"  {label:<18}: {value}")
        return self

    def build(self) -> str:
        return "\n".join(self._lines) + "\n"

class ImageQualityReporter:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 各检测类型的输出子目录
        self.brightness_dir  = self.output_dir / "亮度检测"
        self.sharpness_dir   = self.output_dir / "清晰度检测"
        self.shift_dir        = self.output_dir / "位置偏移检测"
        self.distortion_dir   = self.output_dir / "畸变检测"

        # 文件名常量（各子目录下统一命名）
        self.csv_name        = "明细数据.csv"
        self.metrics_json_name = "评分指标.json"
        self.report_name     = "评分报告.txt"
        self.heatmap_name    = "热力图.png"
        self.histogram_name  = "随机抽样对比示例.png"

        # 暗色主题配色常量（与 wafer_shiftCheck 保持一致）
        self._BG    = "#1a1a2e"   # 图表外背景
        self._PANEL = "#16213e"   # 子图面板背景

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False

    # ==================================================================
    # 暗色热力图通用工具（私有）
    # ==================================================================

    def _apply_dark_style(self, ax, title: str) -> None:
        """对子图应用暗色主题：背景色、标题、刻度、边框颜色。"""
        ax.set_facecolor(self._PANEL)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.tick_params(colors="#aaa", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#0f3460")

    def _draw_dark_heatmap(
        self,
        ax,
        fig,
        grid: np.ndarray,
        x_labels: list,
        y_labels: list,
        title: str,
        cmap: str = "RdBu_r",
        vmin: float | None = None,
        vmax: float | None = None,
        cbar_label: str = "",
        fmt: str = "+.2f",
        suspicious_grid: np.ndarray | None = None,
        origin: str = "lower",
    ) -> None:
        """
        用暗色主题绘制热力图，对齐 wafer_shiftCheck 风格：
          - 深色背景 / 白色文字
          - 图块数 ≤ 2000 时每格显示数值
          - suspicious_grid 为 True 的格子用紫色框高亮
          - colorbar 使用白色标签
        """
        grid_rows, grid_cols = grid.shape
        valid = grid[~np.isnan(grid)]

        # 自动对称色阶（偏移类）或使用传入范围
        if vmax is None:
            vmax = float(np.abs(valid).max()) if len(valid) else 1.0
            vmax = max(vmax, 0.1)
        if vmin is None:
            vmin = -vmax if cmap in ("RdBu_r", "RdBu") else 0.0

        im = ax.imshow(
            grid, cmap=cmap, vmin=vmin, vmax=vmax,
            aspect="auto", interpolation="nearest", origin=origin,
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(cbar_label, color="white", fontsize=9)
        cbar.ax.tick_params(colors="white")

        # 数值标注（格子少时才显示，避免太密集）
        if grid_rows * grid_cols <= 2000:
            for ri in range(grid_rows):
                for ci in range(grid_cols):
                    val = grid[ri, ci]
                    if not np.isnan(val):
                        ax.text(
                            ci, ri, format(val, fmt),
                            ha="center", va="center",
                            fontsize=6, color="black", fontweight="bold",
                        )
                    # 可疑格子紫色框
                    if suspicious_grid is not None and suspicious_grid[ri, ci]:
                        ax.add_patch(plt.Rectangle(
                            (ci - 0.5, ri - 0.5), 1, 1,
                            linewidth=2, edgecolor="#9b59b6", facecolor="none",
                        ))

        # 坐标轴刻度（自适应步长）
        step_c = max(1, grid_cols // 20)
        step_r = max(1, grid_rows // 20)
        ax.set_xticks(range(0, grid_cols, step_c))
        ax.set_xticklabels(
            [str(x_labels[i]) for i in range(0, grid_cols, step_c)],
            color="white", fontsize=7, rotation=45, ha="right",
        )
        ax.set_yticks(range(0, grid_rows, step_r))
        ax.set_yticklabels(
            [str(y_labels[i]) for i in range(0, grid_rows, step_r)],
            color="white", fontsize=7,
        )
        ax.set_xlabel("X 坐标", color="white", fontsize=9)
        ax.set_ylabel("Y 坐标", color="white", fontsize=9)
        self._apply_dark_style(ax, title)

    def _calc_figsize(self, grid_rows: int, grid_cols: int) -> tuple[float, float]:
        """根据网格大小自适应图像尺寸，与 wafer_shiftCheck 保持一致。"""
        fig_h = max(6.0, grid_rows * 0.35 + 2.0)
        fig_w = max(10.0, grid_cols * 0.35 + 2.0)
        return fig_w, fig_h

    # ==================================================================
    # 亮度报告
    # ==================================================================

    def _ensure_dir(self, d: Path) -> Path:
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_brightness_csv(self, stats_df: pd.DataFrame) -> Path:
        path = self._ensure_dir(self.brightness_dir) / self.csv_name
        stats_df.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def save_brightness_visualization(self, stats_df: pd.DataFrame, show: bool = False) -> Path:
        focus_df = (
            stats_df[stats_df["region_type"] == "normal"]
            if "region_type" in stats_df.columns
            else stats_df[stats_df["is_valid"] == True]
        )

        figure = plt.figure(figsize=(20, 12))
        axes = [figure.add_subplot(2, 3, i + 1) for i in range(6)]
        self._plot_heatmap(axes[0], focus_df)
        self._plot_histogram(axes[1], focus_df)
        self._plot_x_trend(axes[2], focus_df)
        self._plot_y_trend(axes[3], focus_df)
        self._plot_outliers(axes[4], focus_df)
        self._plot_boxplot(axes[5], focus_df)
        figure.tight_layout()

        path = self._ensure_dir(self.brightness_dir) / self.heatmap_name
        figure.savefig(path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(figure)
        return path

    def save_brightness_report(self, metrics: dict) -> Path:
        path = self._ensure_dir(self.brightness_dir) / self.report_name
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.build_brightness_report(metrics))
        return path

    def build_brightness_report(self, metrics: dict) -> str:
        builder = _ReportBuilder("晶圆亮度质量评估报告")

        # 质量评分
        builder.section("质量评分")
        builder.field("视觉得分", f"{metrics['score_visual']:.2f}")
        builder.field("一致性得分", f"{metrics['score_consistency']:.2f}")
        builder.blank()

        # 数据集信息
        builder.section("数据集信息")
        builder.field("数据目录", metrics.get("data_dir", ""))
        builder.field("图像总数", str(metrics["total_images"]))
        builder.field("纯背景图片数", str(metrics["pure_bg_count"]))
        builder.field("半背景图片数", str(metrics["half_bg_count"]))
        builder.field("正常区域图片数", str(metrics["normal_count"]))
        builder.blank()

        # 亮度汇总（非纯背景）
        builder.section("亮度汇总 (非纯背景)")
        builder.field("平均亮度", f"{metrics['avg_brightness_all']:.2f}")
        builder.field("亮度标准差", f"{metrics['std_brightness_all']:.2f}")
        builder.field("变异系数(CV)", f"{metrics['cv_all']:.2f}%")
        builder.field(
            f"高亮图片数 (>={metrics['high_brightness_threshold']})",
            str(metrics["high_brightness_count"]),
        )
        builder.field("高亮图片比例", f"{metrics['high_brightness_ratio']:.2f}%")
        builder.blank()

        # 正常区域统计
        builder.section("正常区域统计")
        builder.field("均值", f"{metrics['mean_brightness']:.2f}")
        builder.field("中位数", f"{metrics['median_brightness']:.2f}")
        builder.field("标准差", f"{metrics['std_brightness']:.2f}")
        builder.field(
            "亮度范围",
            f"[{metrics['min_brightness']:.2f}, {metrics['max_brightness']:.2f}]",
        )
        builder.field("变异系数(CV)", f"{metrics['cv_percent']:.2f}%")
        builder.blank()

        # 异常值统计
        builder.section("异常值统计")
        builder.field("过暗异常点", str(metrics["outliers_low_count"]))
        builder.field("过亮异常点", str(metrics["outliers_high_count"]))
        builder.field(
            "偏暗区域数量",
            f"{metrics['dark_regions_count']} (占比 {metrics['dark_ratio_percent']:.2f}%)",
        )

        return builder.build()

    # ==================================================================
    # 清晰度报告
    # ==================================================================

    def save_sharpness_csv(self, stats_df: pd.DataFrame) -> Path:
        path = self._ensure_dir(self.sharpness_dir) / self.csv_name
        stats_df.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def save_sharpness_heatmaps(self, grid_data: dict[str, Any], show: bool = False) -> list[Path]:
        """为每个清晰度指标各生成一张空间热图（暗色主题，含数值标注）。"""
        out_dir  = self._ensure_dir(self.sharpness_dir)
        paths: list[Path] = []
        x_labels = grid_data["x_labels"]
        y_labels = grid_data["y_labels"]
        grid_rows, grid_cols = len(y_labels), len(x_labels)
        fig_w, fig_h = self._calc_figsize(grid_rows, grid_cols)

        for metric_name, grid in grid_data["grids"].items():
            cn_name = SHARPNESS_METRIC_CN.get(metric_name, metric_name)
            valid = grid[~np.isnan(grid)]
            mean_val = float(np.mean(valid)) if len(valid) else 0.0
            n_total  = int(np.sum(~np.isnan(grid)))

            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            fig.patch.set_facecolor(self._BG)
            ax.set_facecolor(self._PANEL)

            title = (
                f"清晰度热图 — {cn_name}  |  共 {n_total} 块  |  均值 {mean_val:.4f}"
            )
            self._draw_dark_heatmap(
                ax=ax, fig=fig, grid=grid,
                x_labels=x_labels, y_labels=y_labels,
                title=title, cmap="viridis",
                vmin=float(valid.min()) if len(valid) else 0.0,
                vmax=float(valid.max()) if len(valid) else 1.0,
                cbar_label=cn_name, fmt=".2f",
            )
            fig.tight_layout()

            path = out_dir / f"热力图_{cn_name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=self._BG)
            if show:
                plt.show()
            else:
                plt.close(fig)
            paths.append(path)

        return paths

    def save_sharpness_histograms(self, stats_df: pd.DataFrame, show: bool = False) -> Path:
        """将四种清晰度指标的分布直方图绘制为 2×2 布局并保存。"""
        metric_names = list(SHARPNESS_METRIC_CN.keys())
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("晶圆清晰度 — 四种指标分布直方图", fontsize=15)

        for ax, name in zip(axes.flat, metric_names):
            if name not in stats_df.columns:
                ax.set_visible(False)
                continue
            values = stats_df[name].dropna().values
            cn_name = SHARPNESS_METRIC_CN[name]
            ax.hist(values, bins=50, color="#4C9BE8", edgecolor="white", linewidth=0.5)
            mean_val = values.mean()
            ax.axvline(mean_val, color="#E85C4C", linewidth=1.5, linestyle="--", label=f"均值 = {mean_val:.4f}")
            ax.set_title(cn_name, fontsize=12)
            ax.set_xlabel("分数", fontsize=10)
            ax.set_ylabel("图块数量", fontsize=10)
            ax.legend(fontsize=9)

        fig.tight_layout()
        path = self._ensure_dir(self.sharpness_dir) / self.histogram_name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return path

    def save_sharpness_report(self, metrics: dict) -> Path:
        path = self._ensure_dir(self.sharpness_dir) / self.report_name
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.build_sharpness_report(metrics))
        return path

    def build_sharpness_report(self, metrics: dict) -> str:
        builder = _ReportBuilder("晶圆清晰度分析 — 汇总统计报告")

        # 有效图块总数
        builder.section("数据集信息")
        builder.field("有效图块总数", str(metrics.get("total_images", 0)))
        builder.blank()

        # 各指标统计
        for name, cn_name in SHARPNESS_METRIC_CN.items():
            builder.section(cn_name)
            builder.field("样本数量", str(metrics.get("total_images", 0)))
            builder.field("均值", f"{metrics.get(f'{name}_mean', 0):.4f}")
            builder.field("标准差", f"{metrics.get(f'{name}_std', 0):.4f}")
            builder.field("中位数", f"{metrics.get(f'{name}_median', 0):.4f}")
            builder.field("最小值", f"{metrics.get(f'{name}_min', 0):.4f}")
            builder.field("最大值", f"{metrics.get(f'{name}_max', 0):.4f}")
            builder.blank()

        # 整体评分
        builder.section("整体评分")
        builder.field(
            "[1] 整体清晰度评分",
            f"{metrics.get('sharpness_score', 0):.2f} 分",
        )
        builder.field(
            "",
            "(说明：基于 FFT 高频能量占比直接映射，分值 0~100，有绝对物理意义)",
        )
        builder.field(
            "[2] 清晰度均匀性评分",
            f"{metrics.get('uniformity_score', 0):.2f} 分",
        )
        builder.field(
            "",
            "(说明：基于变异系数 CV 换算，分值 0~100。分数越高表示批次内各个区域清晰度越均匀一致)",
        )

        return builder.build()


    # ==================================================================
    # 位移偏移报告
    # ==================================================================

    def save_shift_csv(self, stats_df: pd.DataFrame) -> Path:
        cols = ["filename", "x", "y"] + [c for c in stats_df.columns if c.startswith("shift_")]
        path = self._ensure_dir(self.shift_dir) / self.csv_name
        stats_df[cols].to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def save_shift_heatmaps(self, stats_df: pd.DataFrame, show: bool = False) -> list[Path]:
        """生成 Δx、Δy 两张空间热图（暗色主题，含数值标注与 SSIM 异常格高亮）。"""
        paths: list[Path] = []
        out_dir = self._ensure_dir(self.shift_dir)

        for col, label in [("shift_dx", "X方向偏移量(Δx)"), ("shift_dy", "Y方向偏移量(Δy)")]:
            if col not in stats_df.columns:
                continue
            valid = stats_df.dropna(subset=[col, "x", "y"])
            if valid.empty:
                continue

            x_labels = sorted(valid["x"].unique().tolist())
            y_labels = sorted(valid["y"].unique().tolist())
            x_to_col = {x: i for i, x in enumerate(x_labels)}
            y_to_row = {y: i for i, y in enumerate(y_labels)}
            grid_rows, grid_cols = len(y_labels), len(x_labels)

            grid     = np.full((grid_rows, grid_cols), np.nan)
            susp_grid = np.zeros((grid_rows, grid_cols), dtype=bool)

            for _, row in valid.iterrows():
                ri, ci = y_to_row[row["y"]], x_to_col[row["x"]]
                grid[ri, ci] = row[col]
                if "shift_suspicious" in valid.columns and row.get("shift_suspicious", False):
                    susp_grid[ri, ci] = True

            mean_val = float(np.nanmean(grid))
            n_susp   = int(susp_grid.sum())
            title = (
                f"{label}  |  共 {len(valid)} 对  |  均值 {mean_val:+.3f} px"
                + (f"  |  ⚠ SSIM异常 {n_susp} 对" if n_susp > 0 else "")
            )

            fig_w, fig_h = self._calc_figsize(grid_rows, grid_cols)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            fig.patch.set_facecolor(self._BG)
            ax.set_facecolor(self._PANEL)

            self._draw_dark_heatmap(
                ax=ax, fig=fig, grid=grid,
                x_labels=x_labels, y_labels=y_labels,
                title=title, cmap="RdBu_r",
                cbar_label="px", fmt="+.2f",
                suspicious_grid=susp_grid,
            )
            fig.tight_layout()

            path = out_dir / f"热力图_{label}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=self._BG)
            if show:
                plt.show()
            else:
                plt.close(fig)
            paths.append(path)
        return paths

    def save_shift_histograms(self, stats_df: pd.DataFrame, show: bool = False) -> Path:
        """Δx / Δy 分布直方图（2×1 布局）。"""
        valid = stats_df.dropna(subset=["shift_dx", "shift_dy"])
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("晶圆位移偏移 — Δx / Δy 分布", fontsize=14)
        for ax, col, label in zip(axes, ["shift_dx", "shift_dy"], ["Δx (px)", "Δy (px)"]):
            vals = valid[col].values
            ax.hist(vals, bins=50, color="#4C9BE8", edgecolor="white", linewidth=0.5)
            ax.axvline(vals.mean(), color="#E85C4C", linewidth=1.5, linestyle="--", label=f"均值 = {vals.mean():.3f}")
            ax.set_title(label, fontsize=12)
            ax.set_xlabel("偏移量 (px)", fontsize=10)
            ax.set_ylabel("图块对数量", fontsize=10)
            ax.legend(fontsize=9)
        fig.tight_layout()
        path = self._ensure_dir(self.shift_dir) / self.histogram_name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return path

    def save_shift_report(self, metrics: dict) -> Path:
        path = self._ensure_dir(self.shift_dir) / self.report_name
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.build_shift_report(metrics))
        return path

    def build_shift_report(self, metrics: dict) -> str:
        builder = _ReportBuilder("晶圆位移偏移检测 — 汇总统计报告")

        # 数据集信息
        builder.section("数据集信息")
        builder.field("参与双图计算的图块对数", str(metrics.get("total_pairs", 0)))
        builder.blank()

        # 偏移统计
        builder.section("偏移统计")
        builder.field("平均偏移评分", f"{metrics.get('shift_score_mean', 0):.2f} 分")
        builder.field("Δx 均值", f"{metrics.get('shift_dx_mean', 0):+.3f} px")
        builder.field("Δy 均值", f"{metrics.get('shift_dy_mean', 0):+.3f} px")
        builder.field("Δx 标准差", f"{metrics.get('shift_dx_std', 0):.3f} px")
        builder.field("Δy 标准差", f"{metrics.get('shift_dy_std', 0):.3f} px")
        builder.field(
            "SSIM 异常对数",
            f"{metrics.get('shift_suspicious_count', 0)} ({metrics.get('shift_suspicious_ratio', 0):.1f}%)",
        )
        builder.blank()

        return builder.build()


    def generate_shift_all(
        self, stats_df: pd.DataFrame, metrics: dict, show: bool = False
    ) -> dict[str, Any]:
        return {
            "csv":       self.save_shift_csv(stats_df),
            "json":      self.save_metrics_json(metrics, self.shift_dir, filename="位移评分指标.json"),
            "txt":       self.save_shift_report(metrics),
            "histogram": self.save_shift_histograms(stats_df, show=show),
            "heatmaps":  self.save_shift_heatmaps(stats_df, show=show),
        }

    # ==================================================================
    # 畸变报告
    # ==================================================================

    def save_distortion_csv(self, stats_df: pd.DataFrame) -> Path:
        cols = ["filename", "x", "y"] + [c for c in stats_df.columns if c.startswith("dist_")]
        path = self._ensure_dir(self.distortion_dir) / self.csv_name
        stats_df[cols].to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def save_distortion_heatmap(self, stats_df: pd.DataFrame, show: bool = False) -> Path:
        """畸变评分空间热图（暗色主题，含数值标注）。"""
        out_dir = self._ensure_dir(self.distortion_dir)
        valid   = stats_df.dropna(subset=["dist_score", "x", "y"])

        if valid.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor(self._BG)
            ax.set_facecolor(self._PANEL)
            ax.text(0.5, 0.5, "暂无数据", ha="center", va="center",
                    color="white", fontsize=16)
            ax.set_axis_off()
        else:
            x_labels = sorted(valid["x"].unique().tolist())
            y_labels = sorted(valid["y"].unique().tolist())
            x_to_col = {x: i for i, x in enumerate(x_labels)}
            y_to_row = {y: i for i, y in enumerate(y_labels)}
            grid_rows, grid_cols = len(y_labels), len(x_labels)
            grid = np.full((grid_rows, grid_cols), np.nan)
            for _, row in valid.iterrows():
                grid[y_to_row[row["y"]], x_to_col[row["x"]]] = row["dist_score"]

            mean_val = float(np.nanmean(grid))
            n_total  = int(np.sum(~np.isnan(grid)))
            title = f"畸变评分热图  |  共 {n_total} 对  |  均值 {mean_val:.2f} 分"

            fig_w, fig_h = self._calc_figsize(grid_rows, grid_cols)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            fig.patch.set_facecolor(self._BG)
            ax.set_facecolor(self._PANEL)

            self._draw_dark_heatmap(
                ax=ax, fig=fig, grid=grid,
                x_labels=x_labels, y_labels=y_labels,
                title=title, cmap="RdYlGn",
                vmin=0.0, vmax=100.0,
                cbar_label="畸变评分", fmt=".2f",
            )

        fig.tight_layout()
        path = out_dir / self.heatmap_name
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=self._BG)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return path

    def save_distortion_histograms(self, stats_df: pd.DataFrame, show: bool = False) -> Path:
        """旋转角 / 切变量分布直方图（2×1 布局）。"""
        valid = stats_df.dropna(subset=["dist_rotation_deg", "dist_shear"])
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("晶圆畸变 — 旋转角 / 切变分布", fontsize=14)
        for ax, col, label in zip(axes, ["dist_rotation_deg", "dist_shear"], ["旋转角 (°)", "切变量"]):
            if valid.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue
            vals = valid[col].values
            ax.hist(vals, bins=50, color="#4C9BE8", edgecolor="white", linewidth=0.5)
            ax.axvline(vals.mean(), color="#E85C4C", linewidth=1.5, linestyle="--", label=f"均值 = {vals.mean():.4f}")
            ax.set_title(label, fontsize=12)
            ax.set_xlabel(label, fontsize=10)
            ax.set_ylabel("图块对数量", fontsize=10)
            ax.legend(fontsize=9)
        fig.tight_layout()
        path = self._ensure_dir(self.distortion_dir) / self.histogram_name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return path

    def save_distortion_report(self, metrics: dict) -> Path:
        path = self._ensure_dir(self.distortion_dir) / self.report_name
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.build_distortion_report(metrics))
        return path

    def build_distortion_report(self, metrics: dict) -> str:
        builder = _ReportBuilder("晶圆畸变检测 — 汇总统计报告")

        # 数据集信息
        builder.section("数据集信息")
        builder.field("参与双图计算的图块对数", str(metrics.get("total_pairs", 0)))
        builder.field(
            "SIFT 特征提取失败对数",
            f"{metrics.get('n_failed', 0)} ({metrics.get('dist_failed_ratio', 0):.1f}%)",
        )
        builder.blank()

        # 畸变统计
        builder.section("畸变统计")
        builder.field("平均畸变评分", f"{metrics.get('dist_score_mean', 0):.2f} 分")
        builder.field("旋转角均值", f"{metrics.get('dist_rotation_mean', 0):+.4f} °")
        builder.field("切变量均值", f"{metrics.get('dist_shear_mean', 0):+.4f}")
        builder.field("评分标准差", f"{metrics.get('dist_score_std', 0):.2f}")
        builder.blank()

        return builder.build()


    def generate_distortion_all(
        self, stats_df: pd.DataFrame, metrics: dict, show: bool = False
    ) -> dict[str, Any]:
        return {
            "csv":       self.save_distortion_csv(stats_df),
            "json":      self.save_metrics_json(metrics, self.distortion_dir, filename="畸变评分指标.json"),
            "txt":       self.save_distortion_report(metrics),
            "histogram": self.save_distortion_histograms(stats_df, show=show),
            "heatmap":   self.save_distortion_heatmap(stats_df, show=show),
        }

    # ==================================================================
    # 统一入口
    # ==================================================================

    def save_metrics_json(self, metrics: dict, subdir: Path, filename: str | None = None) -> Path:
        path = self._ensure_dir(subdir) / (filename or self.metrics_json_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        return path

    def generate_brightness_all(self, stats_df: pd.DataFrame, metrics: dict, show: bool = False) -> dict[str, Path]:
        return {
            "csv":  self.save_brightness_csv(stats_df),
            "json": self.save_metrics_json(metrics, self.brightness_dir),
            "txt":  self.save_brightness_report(metrics),
            "png":  self.save_brightness_visualization(stats_df, show=show),
        }

    def generate_sharpness_all(
        self, stats_df: pd.DataFrame, metrics: dict, grid_data: dict[str, Any], show: bool = False
    ) -> dict[str, Any]:
        return {
            "csv":       self.save_sharpness_csv(stats_df),
            "json":      self.save_metrics_json(metrics, self.sharpness_dir),
            "txt":       self.save_sharpness_report(metrics),
            "histogram": self.save_sharpness_histograms(stats_df, show=show),
            "heatmaps":  self.save_sharpness_heatmaps(grid_data, show=show),
        }

    # ==================================================================
    # 亮度绘图（私有）
    # ==================================================================

    def _safe_std_local(self, series: pd.Series) -> float:
        if len(series) <= 1:
            return 0.0
        value = float(series.std())
        return 0.0 if np.isnan(value) else value

    def _plot_empty(self, ax, title: str) -> None:
        ax.set_title(title)
        ax.text(0.5, 0.5, "暂无数据", ha="center", va="center")
        ax.set_axis_off()

    def _plot_heatmap(self, ax, df: pd.DataFrame) -> None:
        if df.empty:
            self._plot_empty(ax, "亮度热力图")
            return
        x_coords = sorted(df["x"].dropna().unique().tolist())
        y_coords = sorted(df["y"].dropna().unique().tolist())
        grid_rows, grid_cols = len(y_coords), len(x_coords)
        grid   = np.full((grid_rows, grid_cols), np.nan)
        x_index = {v: i for i, v in enumerate(x_coords)}
        y_index = {v: i for i, v in enumerate(y_coords)}
        for _, row in df.dropna(subset=["x", "y"]).iterrows():
            grid[y_index[row["y"]], x_index[row["x"]]] = row["valid_mean"]

        valid    = grid[~np.isnan(grid)]
        mean_val = float(np.mean(valid)) if len(valid) else 0.0
        title    = f"亮度热力图  |  均值 {mean_val:.2f}"

        # 在综合面板图内复用暗色热力图方法
        fig = ax.get_figure()
        self._draw_dark_heatmap(
            ax=ax, fig=fig, grid=grid,
            x_labels=x_coords, y_labels=y_coords,
            title=title, cmap="viridis",
            vmin=float(valid.min()) if len(valid) else 0.0,
            vmax=float(valid.max()) if len(valid) else 255.0,
            cbar_label="有效均值", fmt=".2f",
        )

    def _plot_histogram(self, ax, df: pd.DataFrame) -> None:
        if df.empty:
            self._plot_empty(ax, "亮度直方图")
            return
        b = df["valid_mean"]
        ax.hist(b, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
        ax.axvline(b.mean(),   color="red",   linestyle="--", linewidth=2, label="均值")
        ax.axvline(b.median(), color="green", linestyle="--", linewidth=2, label="中位数")
        ax.set_title("亮度直方图")
        ax.set_xlabel("亮度")
        ax.set_ylabel("数量")
        ax.legend()

    def _plot_x_trend(self, ax, df: pd.DataFrame) -> None:
        if df.empty:
            self._plot_empty(ax, "X方向亮度趋势")
            return
        trend = df.groupby("x")["valid_mean"].agg(["mean", "std"]).fillna(0.0)
        ax.plot(trend.index, trend["mean"], marker="o", linewidth=2, markersize=4)
        ax.fill_between(trend.index, trend["mean"] - trend["std"], trend["mean"] + trend["std"], alpha=0.3)
        ax.set_title("X方向亮度趋势")
        ax.set_xlabel("X")
        ax.set_ylabel("亮度")

    def _plot_y_trend(self, ax, df: pd.DataFrame) -> None:
        if df.empty:
            self._plot_empty(ax, "Y方向亮度趋势")
            return
        trend = df.groupby("y")["valid_mean"].agg(["mean", "std"]).fillna(0.0)
        ax.plot(trend.index, trend["mean"], marker="o", linewidth=2, markersize=4, color="orange")
        ax.fill_between(trend.index, trend["mean"] - trend["std"], trend["mean"] + trend["std"], alpha=0.3, color="orange")
        ax.set_title("Y方向亮度趋势")
        ax.set_xlabel("Y")
        ax.set_ylabel("亮度")

    def _plot_outliers(self, ax, df: pd.DataFrame) -> None:
        if df.empty:
            self._plot_empty(ax, "离群点分布图")
            return
        mean_val = float(df["valid_mean"].mean())
        std_val  = self._safe_std_local(df["valid_mean"])
        normal    = df[(df["valid_mean"] >= mean_val - std_val)   & (df["valid_mean"] <= mean_val + 2 * std_val)]
        dark      = df[(df["valid_mean"] <  mean_val - std_val)   & (df["valid_mean"] >= mean_val - 3 * std_val)]
        very_dark = df[ df["valid_mean"] <  mean_val - 3 * std_val]
        bright    = df[ df["valid_mean"] >  mean_val + 2 * std_val]
        ax.scatter(normal["x"],    normal["y"],    c="green",  s=20, alpha=0.5, label="正常")
        ax.scatter(dark["x"],      dark["y"],      c="orange", s=30, alpha=0.7, label="偏暗")
        ax.scatter(very_dark["x"], very_dark["y"], c="red",    s=50, marker="X", label="极暗")
        ax.scatter(bright["x"],    bright["y"],    c="blue",   s=50, marker="*", label="偏亮")
        ax.set_title("离群点分布图")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()

    def _plot_boxplot(self, ax, df: pd.DataFrame) -> None:
        if df.empty:
            self._plot_empty(ax, "亮度箱线图")
            return
        bp = ax.boxplot([df["valid_mean"]], labels=["亮度"], patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
        ax.set_title("亮度箱线图")
        ax.set_ylabel("亮度")