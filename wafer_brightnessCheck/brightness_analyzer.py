"""
晶圆亮度一致性评估系统 - 主分析模块

该模块用于批量分析线阵相机扫描的晶圆图像，评估亮度一致性
无需拼接即可获得全景亮度分布的量化指标和可视化结果
"""

import re
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent.parent
DATA_ROOT_DIR = Path(os.getenv("WAFER_DATA_ROOT", WORKSPACE_DIR / "data")).expanduser().resolve()
OUTPUT_ROOT_DIR = Path(os.getenv("WAFER_OUTPUT_ROOT", WORKSPACE_DIR / "output")).expanduser().resolve()

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


class BrightnessAnalyzer:
    """晶圆亮度分析器"""

    def __init__(self, data_dir: str, output_dir: str = "brightness_results"):
        """
        初始化分析器

        Args:
            data_dir: 图片所在目录
            output_dir: 结果输出目录
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.image_stats: list[Dict] = []
        self.grid_data: dict[str, float] = {}

    def parse_filename(self, filename: str) -> Optional[Tuple[int, int]]:
        """
        解析文件名获取坐标

        Args:
            filename: 文件名，格式为 0000_x_y.png

        Returns:
            (x, y) 坐标
        """
        pattern = r"0000_(\d+)_(\d+)\.png"
        match = re.match(pattern, filename)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None

    def analyze_image(self, image_path: Path) -> Optional[Dict]:
        """
        分析单张图片的亮度统计

        Args:
            image_path: 图片路径

        Returns:
            包含统计信息的字典
        """
        # 读取图片（灰度）
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            return None

        # 基本统计
        mean_brightness = np.mean(img)  # type: ignore
        std_brightness = np.std(img)  # type: ignore
        min_brightness = np.min(img)
        max_brightness = np.max(img)

        # 检测有效区域（非背景）
        # 背景阈值设为30，更严格地排除低亮度背景
        BACKGROUND_THRESHOLD = 30
        valid_mask = img > BACKGROUND_THRESHOLD
        valid_ratio = np.sum(valid_mask) / img.size

        # 有效区域的亮度统计
        if valid_ratio > 0.01:  # 至少有1%的有效像素
            valid_pixels = img[valid_mask]
            valid_mean = np.mean(valid_pixels)  # type: ignore
            valid_std = np.std(valid_pixels)  # type: ignore
        else:
            valid_mean = np.float64(0.0)
            valid_std = np.float64(0.0)

        # 解析坐标
        coords = self.parse_filename(image_path.name)
        if coords is None:
            x, y = None, None
        else:
            x, y = coords

        stats = {
            "filename": image_path.name,
            "x": x,
            "y": y,
            "mean_brightness": mean_brightness,
            "std_brightness": std_brightness,
            "min_brightness": min_brightness,
            "max_brightness": max_brightness,
            "valid_ratio": valid_ratio,
            "valid_mean": valid_mean,
            "valid_std": valid_std,
            "is_valid": valid_ratio > 0.1,  # 有效晶圆区域阈值：10%
        }

        return stats

    def batch_analyze(self, pattern: str = "*.png") -> pd.DataFrame:
        """
        批量分析图片

        Args:
            pattern: 文件名匹配模式

        Returns:
            DataFrame包含所有图片的统计信息
        """
        image_files = sorted(self.data_dir.glob(pattern))

        if len(image_files) == 0:
            print(f"未找到匹配的图片：{pattern}")
            return pd.DataFrame()

        print(f"找到 {len(image_files)} 张图片，开始分析...")

        results = []
        for img_path in tqdm(image_files, desc="分析进度"):
            stats = self.analyze_image(img_path)
            if stats is not None:
                results.append(stats)

        self.df = pd.DataFrame(results)

        # 计算坐标范围（用于边缘检测）
        self.x_min, self.x_max = self.df["x"].min(), self.df["x"].max()
        self.y_min, self.y_max = self.df["y"].min(), self.df["y"].max()
        print(f"坐标范围: X[{self.x_min}, {self.x_max}], Y[{self.y_min}, {self.y_max}]")

        # 添加区域类型（基于坐标和 valid_ratio）
        # 为基于邻域的边缘检测构建坐标索引
        self._build_coord_index()

        # 从有效区域推导暗度阈值
        valid_df = self.df[self.df["is_valid"] == True]
        if len(valid_df) > 0:
            mean_val = valid_df["valid_mean"].mean()
            std_val = valid_df["valid_mean"].std()
            self.dark_threshold = mean_val - std_val
            self.very_dark_threshold = mean_val - 3 * std_val
        else:
            self.dark_threshold = 0
            self.very_dark_threshold = 0

        # 基于边缘+暗度的区域分类
        self.df["region_type"] = self.df.apply(
            lambda row: self.classify_region_type(
                row["valid_mean"], row["x"], row["y"]
            ),
            axis=1,
        )

        # 保存统计数据
        output_path = self.output_dir / "明细数据.csv"
        self.df.to_csv(output_path, index=False)
        print(f"统计数据已保存至: {output_path}")

        return self.df

    def _build_coord_index(self) -> None:
        """为基于邻域的边缘检测构建坐标索引"""
        self.x_coords = sorted(self.df["x"].dropna().unique())
        self.y_coords = sorted(self.df["y"].dropna().unique())
        self.x_index = {x: i for i, x in enumerate(self.x_coords)}
        self.y_index = {y: i for i, y in enumerate(self.y_coords)}
        self.coord_set = set(zip(self.df["x"], self.df["y"]))

    def is_edge_position(self, x: int, y: int) -> bool:
        """如果任何邻居（上/下/左/右）缺失，则返回 True"""
        if not hasattr(self, "coord_set"):
            return False
        if x not in self.x_index or y not in self.y_index:
            return False

        ix = self.x_index[x]
        iy = self.y_index[y]

        x_prev = self.x_coords[ix - 1] if ix > 0 else None
        x_next = self.x_coords[ix + 1] if ix < len(self.x_coords) - 1 else None
        y_prev = self.y_coords[iy - 1] if iy > 0 else None
        y_next = self.y_coords[iy + 1] if iy < len(self.y_coords) - 1 else None

        missing_left = (x_prev is None) or ((x_prev, y) not in self.coord_set)
        missing_right = (x_next is None) or ((x_next, y) not in self.coord_set)
        missing_up = (y_prev is None) or ((x, y_prev) not in self.coord_set)
        missing_down = (y_next is None) or ((x, y_next) not in self.coord_set)

        return missing_left or missing_right or missing_up or missing_down

    def classify_region_type(self, valid_mean: float, x: int = 0, y: int = 0) -> str:
        """基于边缘位置和暗度阈值分类区域"""
        if not hasattr(self, "dark_threshold") or not hasattr(
            self, "very_dark_threshold"
        ):
            return "normal"

        if not self.is_edge_position(x, y):
            return "normal"

        if valid_mean < self.very_dark_threshold:
            return "pure_bg"
        if valid_mean < self.dark_threshold:
            return "half_bg"
        return "normal"


def main():
    """主函数"""
    if not DATA_ROOT_DIR.exists():
        print(f"data 目录不存在：{DATA_ROOT_DIR}")
        return
    dataset_dirs = sorted(path for path in DATA_ROOT_DIR.iterdir() if path.is_dir())
    if not dataset_dirs:
        print(f"未在 data 目录下找到数据集文件夹：{DATA_ROOT_DIR}")
        return

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        output_dir = OUTPUT_ROOT_DIR / f"{dataset_name}_输出" / "亮度检测"
        print(f"\n开始处理数据集：{dataset_name}")
        analyzer = BrightnessAnalyzer(str(dataset_dir), str(output_dir))
        analyzer.batch_analyze(pattern="0000_*.png")

    print("\n" + "=" * 60)
    print("✓ 数据采集完成！全部数据集CSV文件已生成")
    print("=" * 60)
    print("\n接下来请运行：")
    print("  1. python generate_report.py     # 计算指标和生成报告")
    print("  2. python visualize_brightness.py   # 生成可视化图表")
    print("=" * 60)


if __name__ == "__main__":
    main()
