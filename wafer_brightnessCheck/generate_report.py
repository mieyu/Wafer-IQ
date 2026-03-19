"""
晶圆亮度质量评估报告生成器

该模块负责从CSV文件计算指标并生成报告
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent.parent
OUTPUT_ROOT_DIR = Path(os.getenv("WAFER_OUTPUT_ROOT", WORKSPACE_DIR / "output")).expanduser().resolve()


class BrightnessReportGenerator:
    """亮度报告生成器"""

    def __init__(self, stats_csv: str, output_dir: str):
        """
        初始化报告生成器

        Args:
            stats_csv: brightness_stats.csv 文件路径
            output_dir: 输出目录
        """
        self.stats_csv = Path(stats_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 读取CSV数据
        self.df = pd.read_csv(self.stats_csv)

        # 获取数据目录（从CSV文件路径推断）
        self.data_dir = str(self.stats_csv.parent)

        # metrics将在calculate_metrics()中生成
        self.metrics: Optional[Dict] = None

    def calculate_metrics(self) -> Dict:
        """
        计算量化指标

        Returns:
            包含各项指标和评分的字典
        """
        print("\n" + "=" * 60)
        print("计算亮度质量指标...")
        print("=" * 60)

        # ========== 1. 区域分类统计 ==========
        print("\n步骤1: 区域分类")

        pure_bg_count = len(self.df[self.df["region_type"] == "pure_bg"])
        half_bg_count = len(self.df[self.df["region_type"] == "half_bg"])
        normal_count = len(self.df[self.df["region_type"] == "normal"])
        total_images = len(self.df)

        print(f"  纯背景: {pure_bg_count} 张")
        print(f"  半背景: {half_bg_count} 张")
        print(f"  正常区域: {normal_count} 张")
        print(f"  总计: {total_images} 张")

        # ========== 2. 非背景区域亮度统计 ==========
        print("\n步骤2: 非背景区域亮度统计")

        # 剔除纯背景，统计所有非背景区域（包括半背景和正常区域）
        non_bg_df = self.df[self.df["region_type"] != "pure_bg"]

        if len(non_bg_df) > 0:
            avg_brightness = non_bg_df["valid_mean"].mean()
            std_brightness_all = non_bg_df["valid_mean"].std()
            cv_all = (
                (std_brightness_all / avg_brightness * 100) if avg_brightness > 0 else 0
            )
        else:
            avg_brightness = 0
            std_brightness_all = 0
            cv_all = 0

        print(f"  平均亮度: {avg_brightness:.2f}")
        print(f"  标准差: {std_brightness_all:.2f}")
        print(f"  变异系数 (CV): {cv_all:.2f}%")

        # ========== 3. 高亮度占比统计 ==========
        print("\n步骤3: 高亮度占比")

        HIGH_BRIGHTNESS_THRESHOLD = 100
        normal_df = self.df[self.df["region_type"] == "normal"]

        if len(normal_df) > 0:
            high_brightness_count = len(
                normal_df[normal_df["valid_mean"] >= HIGH_BRIGHTNESS_THRESHOLD]
            )
            high_brightness_ratio = (high_brightness_count / len(normal_df)) * 100
        else:
            high_brightness_count = 0
            high_brightness_ratio = 0

        print(
            f"  高亮度区域(>={HIGH_BRIGHTNESS_THRESHOLD}): {high_brightness_count} 张"
        )
        print(f"  高亮度占比: {high_brightness_ratio:.1f}%")

        # ========== 4. 评分计算 ==========
        print("\n步骤4: 评分计算")

        # 评分1: 肉眼直觉评分（基于高斯曲线，最高分在128）
        # 使用反常高斯曲线：对高亮度惩罚较低，对低亮度惩罚较高
        scores_visual = []
        for idx, row in non_bg_df.iterrows():
            brightness = row["valid_mean"]
            # 高斯曲线: score = 100 * exp(-((x-128)^2) / (2*sigma^2))
            # sigma控制曲线宽度，值越大曲线越平缓
            # 对高亮度（>128）使用较大的sigma，惩罚较小
            # 对低亮度（<128）使用较小的sigma，惩罚较大
            if brightness >= 128:
                sigma = 60  # 高亮度侧：曲线平缓，惩罚小
            else:
                sigma = 40  # 低亮度侧：曲线陡峭，惩罚大

            score = 100 * np.exp(-((brightness - 128) ** 2) / (2 * sigma**2))
            scores_visual.append(score)

        avg_score_visual = np.mean(scores_visual) if len(scores_visual) > 0 else 0
        print(f"  肉眼直觉评分: {avg_score_visual:.2f}")

        # 评分2: 基于平均亮度和变异系数的评分
        # 先计算全局平均亮度
        global_mean = non_bg_df["valid_mean"].mean() if len(non_bg_df) > 0 else 0

        scores_cv = []
        for idx, row in non_bg_df.iterrows():
            brightness = row["valid_mean"]
            # 计算该图片亮度相对于平均值的偏差
            deviation = abs(brightness - global_mean)
            # 偏差越小，分数越高
            # 使用指数衰减函数: score = 100 * exp(-deviation / scale)
            scale = 20  # 调整衰减速度
            score = 100 * np.exp(-deviation / scale)
            scores_cv.append(score)

        avg_score_cv = np.mean(scores_cv) if len(scores_cv) > 0 else 0
        print(f"  平均亮度一致性评分: {avg_score_cv:.2f}")

        # ========== 5. 传统统计指标（仅针对 normal 区域）==========
        normal_df = self.df[self.df["region_type"] == "normal"]

        if len(normal_df) > 0:
            mean_brightness = normal_df["valid_mean"].mean()
            std_brightness = normal_df["valid_mean"].std()
            median_brightness = normal_df["valid_mean"].median()
            min_brightness = normal_df["valid_mean"].min()
            max_brightness = normal_df["valid_mean"].max()
            cv = (std_brightness / mean_brightness) * 100 if mean_brightness > 0 else 0

            # 异常检测（仅在 normal 区域中检测）
            threshold_low = mean_brightness - 3 * std_brightness
            threshold_high = mean_brightness + 2 * std_brightness
            outliers_low = normal_df[normal_df["valid_mean"] < threshold_low]
            outliers_high = normal_df[normal_df["valid_mean"] > threshold_high]

            # 暗区检测（仅在 normal 区域中检测）
            dark_threshold = mean_brightness - std_brightness
            dark_regions = normal_df[normal_df["valid_mean"] < dark_threshold]
            dark_ratio = len(dark_regions) / len(normal_df) * 100
        else:
            mean_brightness = std_brightness = median_brightness = 0
            min_brightness = max_brightness = cv = 0
            threshold_low = threshold_high = dark_threshold = 0
            outliers_low = outliers_high = dark_regions = pd.DataFrame()
            dark_ratio = 0

        # ========== 6. 汇总指标 ==========
        metrics = {
            # 区域分类
            "total_images": total_images,
            "pure_bg_count": pure_bg_count,
            "half_bg_count": half_bg_count,
            "normal_count": normal_count,
            # 非背景区域统计（包括半背景和正常区域）
            "avg_brightness_all": float(avg_brightness),
            "std_brightness_all": float(std_brightness_all),
            "cv_all": float(cv_all),
            # 评分指标
            "score_visual": float(avg_score_visual),
            "score_consistency": float(avg_score_cv),
            # 高亮度占比
            "high_brightness_count": high_brightness_count,
            "high_brightness_ratio": float(high_brightness_ratio),
            "high_brightness_threshold": HIGH_BRIGHTNESS_THRESHOLD,
            # 传统指标
            "mean_brightness": float(mean_brightness),
            "median_brightness": float(median_brightness),
            "std_brightness": float(std_brightness),
            "min_brightness": float(min_brightness),
            "max_brightness": float(max_brightness),
            "brightness_range": float(max_brightness - min_brightness),
            "cv_percent": float(cv),
            # 异常检测
            "outliers_low_count": len(outliers_low),
            "outliers_high_count": len(outliers_high),
            "dark_regions_count": len(dark_regions),
            "dark_ratio_percent": float(dark_ratio),
            "threshold_low": float(threshold_low),
            "threshold_high": float(threshold_high),
            "dark_threshold": float(dark_threshold),
            # 数据目录（用于报告生成）
            "data_dir": self.data_dir,
        }

        # 保存指标
        self.metrics = metrics
        json_path = self.output_dir / "评分指标.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"\n量化指标已保存至: {json_path}")
        print("=" * 60)

        return metrics

    def generate_report(self):
        """
        生成文本报告
        """
        if self.metrics is None:
            raise ValueError("请先运行 calculate_metrics()")

        metrics = self.metrics

        report = f"""
{'='*60}
晶圆亮度质量评估报告
{'='*60}

{'='*60}
质量评分 (满分100分):
{'='*60}
- 肉眼直觉评分: {metrics['score_visual']:.2f} 分
- 平均亮度一致性评分: {metrics['score_consistency']:.2f} 分

{'='*60}
数据集信息:
{'='*60}
- 数据目录: {self.data_dir}
- 总图片数: {metrics['total_images']}
- 纯背景: {metrics.get('pure_bg_count', 0)} 张 (已剔除统计)
- 半背景: {metrics.get('half_bg_count', 0)} 张 (权重×0.5)
- 正常区域: {metrics.get('normal_count', 0)} 张

{'='*60}
亮度统计指标:
{'='*60}
- 平均亮度 (非背景): {metrics['avg_brightness_all']:.2f}
- 标准差 (非背景): {metrics['std_brightness_all']:.2f}
- 变异系数 CV (非背景): {metrics['cv_all']:.2f}%
- 高亮度区域 (≥{metrics.get('high_brightness_threshold', 100)}): {metrics['high_brightness_count']} 张
- 高亮度占比: {metrics['high_brightness_ratio']:.1f}%



{'='*60}
传统亮度统计指标 (仅供参考):
{'='*60}
- 平均亮度: {metrics['mean_brightness']:.2f}
- 中位数亮度: {metrics['median_brightness']:.2f}
- 标准差: {metrics['std_brightness']:.2f}
- 亮度范围: [{metrics['min_brightness']:.2f}, {metrics['max_brightness']:.2f}]
- CV (传统): {metrics['cv_percent']:.2f}%

{'='*60}
异常区域检测:
{'='*60}
- 异常暗区数量 (<μ-3σ): {metrics['outliers_low_count']}
- 异常亮区数量 (>μ+2σ): {metrics['outliers_high_count']}
- 较暗区域数量 (<μ-1σ): {metrics['dark_regions_count']} ({metrics['dark_ratio_percent']:.1f}%)
"""

        # 保存报告
        report_path = self.output_dir / "评分报告.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(report)
        print(f"报告已保存至: {report_path}")


def main():
    """主函数"""
    if not OUTPUT_ROOT_DIR.exists():
        print(f"output 目录不存在：{OUTPUT_ROOT_DIR}")
        return
    module_output_dirs = sorted(
        path / "亮度检测"
        for path in OUTPUT_ROOT_DIR.iterdir()
        if path.is_dir() and path.name.endswith("_输出")
    )
    if not module_output_dirs:
        print(f"未在 output 目录下找到 *_输出 目录：{OUTPUT_ROOT_DIR}")
        return

    for module_output_dir in module_output_dirs:
        stats_csv_path = module_output_dir / "明细数据.csv"
        if not stats_csv_path.exists():
            continue
        print(f"\n开始生成亮度报告：{module_output_dir.parent.name}")
        generator = BrightnessReportGenerator(str(stats_csv_path), str(module_output_dir))
        generator.calculate_metrics()
        print("\n✓ 指标计算完成，已生成 评分指标.json")
        generator.generate_report()
        print("✓ 报告生成完成，已生成 评分报告.txt")


if __name__ == "__main__":
    main()
