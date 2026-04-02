# 数据层通用工具模块：提供 YAML 配置解析、坐标索引构建、边缘位置判断、DataFrame 网格映射等功能。
# 不含任何图像像素操作，不依赖 cv2。坐标来源统一为 placements-BF.yml。
# 供 calculate / reporter 等上层模块复用。
# 使用方法：from src.image_quality_data_utils import ImageQualityDataUtils

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class ImageQualityDataUtils:
    """数据层通用工具：YAML 解析、坐标索引、DataFrame 操作。不含任何图像像素计算。"""

    # ==================================================================
    # YAML 配置解析（坐标来源）
    # ==================================================================

    @staticmethod
    def load_yaml_meta(
        yaml_path: str | Path,
    ) -> dict[str, Any]:
        """解析 placements-BF.yml，返回包含重叠区域与视图文件路径的字典。

        Args:
            yaml_path (str | Path): YAML 配置文件路径。

        Returns:
            dict[str, Any]: 包含以下键的字典：
                - overlap_px (int): 重叠区域实际像素宽度。
                - views (dict): key 为 (col, row) 坐标元组，value 为图像文件 Path。
                如果文件不存在或解析失败，返回空 dict。
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            return {}
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                meta = yaml.safe_load(f)
            hex_to_double = lambda h: struct.unpack(">d", bytes.fromhex(h))[0]
            px_size = hex_to_double(meta["image_meta"]["pixel_equivalents"].split(",")[0])
            overlap_px = max(1, int(round(hex_to_double(meta["view_meta"]["overlap_width"]) / px_size)))
            data_dir = yaml_path.parent
            views: dict[tuple[int, int], Path] = {
                tuple(int(i) for i in v["index"].split(",")): data_dir / v["filename"]  # type: ignore[misc]
                for v in meta["views"]
            }
            return {"overlap_px": overlap_px, "views": views}
        except Exception as exc:
            logger.error(f"解析 YAML 配置失败 [{yaml_path}]：{exc}")
            return {}

    # ==================================================================
    # 统计工具
    # ==================================================================

    @staticmethod
    def safe_std(series: pd.Series) -> float:
        """安全地计算 DataFrame / Series 的标准差。

        防止因样本数为 0 或 1 导致的 NaN 返回，统一以 0.0 处理。

        Args:
            series (pd.Series): 需要计算标准差的 Pandas Series 数据。

        Returns:
            float: 计算出的安全标准差，遇到 NaN 默认返回 0.0。
        """
        if len(series) <= 1:
            return 0.0
        value = float(series.std())
        return 0.0 if np.isnan(value) else value

    # ==================================================================
    # 坐标索引工具
    # ==================================================================

    @staticmethod
    def build_coord_index(
        df: pd.DataFrame
    ) -> tuple[list[Any], list[Any], dict[Any, int], dict[Any, int], set[tuple[Any, Any]]]:
        """提取 DataFrame 中所有的去重坐标，并构建位置索引与集合。

        用来加速后续对相邻坐标(边缘位置)存在与否的查找判定。

        Args:
            df (pd.DataFrame): 包含列名 'x' 和 'y' 的 DataFrame 数据。

        Returns:
            tuple:
                - x_coords (list[Any]): 排序去重后的 X 坐标列表。
                - y_coords (list[Any]): 排序去重后的 Y 坐标列表。
                - x_index (dict[Any, int]): X 坐标与其在列表内索引的映射字典。
                - y_index (dict[Any, int]): Y 坐标与其在列表内索引的映射字典。
                - coord_set (set[tuple[Any, Any]]): 去重后的 (x, y) 坐标元组集合。
        """
        x_coords = sorted(df["x"].dropna().unique().tolist())
        y_coords = sorted(df["y"].dropna().unique().tolist())
        x_index = {x: idx for idx, x in enumerate(x_coords)}
        y_index = {y: idx for idx, y in enumerate(y_coords)}
        coord_set = set(zip(df["x"], df["y"]))
        return x_coords, y_coords, x_index, y_index, coord_set

    @staticmethod
    def is_edge_position(
        x: Any,
        y: Any,
        x_coords: list[Any],
        y_coords: list[Any],
        x_index: dict[Any, int],
        y_index: dict[Any, int],
        coord_set: set[tuple[Any, Any]],
    ) -> bool:
        """基于全局构建好的坐标索引信息，判断当前给定的 (x, y) 是否位于边缘位置。

        即判断当前坐标上下左右相邻的点是否缺失。

        Args:
            x (Any): 当前的 x 坐标值。
            y (Any): 当前的 y 坐标值。
            x_coords (list[Any]): 预构建的 x 坐标列表。
            y_coords (list[Any]): 预构建的 y 坐标列表。
            x_index (dict[Any, int]): 预构建的 x 坐标索引字典。
            y_index (dict[Any, int]): 预构建的 y 坐标索引字典。
            coord_set (set[tuple[Any, Any]]): 预构建的坐标点集合。

        Returns:
            bool: 处于边缘（上下左右任一相邻图块缺失）时返回 True，否则返回 False。
        """
        if pd.isna(x) or pd.isna(y):
            return False
        if x not in x_index or y not in y_index:
            return False

        ix = x_index[x]
        iy = y_index[y]

        x_prev = x_coords[ix - 1] if ix > 0 else None
        x_next = x_coords[ix + 1] if ix < len(x_coords) - 1 else None
        y_prev = y_coords[iy - 1] if iy > 0 else None
        y_next = y_coords[iy + 1] if iy < len(y_coords) - 1 else None

        missing_left  = x_prev is None or (x_prev, y) not in coord_set
        missing_right = x_next is None or (x_next, y) not in coord_set
        missing_up    = y_prev is None or (x, y_prev) not in coord_set
        missing_down  = y_next is None or (x, y_next) not in coord_set
        return missing_left or missing_right or missing_up or missing_down

    # ==================================================================
    # 网格映射
    # ==================================================================

    @staticmethod
    def build_grid(
        df: pd.DataFrame, metric_names: list[str]
    ) -> dict[str, Any]:
        """将 DataFrame 映射到二维网格，供热图绘制使用。

        Args:
            df (pd.DataFrame): 包含 x, y 及各指标列的 DataFrame。
            metric_names (list[str]): 需要构建网格的指标列名列表。

        Returns:
            dict[str, Any]: 包含以下内容的字典：
                - "df":       原始 DataFrame
                - "x_labels": 排序后的 x 坐标值列表
                - "y_labels": 排序后的 y 坐标值列表
                - "grids":    dict，键为指标名，值为 (num_y, num_x) 的 numpy 数组（缺失位置为 NaN）。
        """
        x_labels = sorted(df["x"].dropna().unique().tolist())
        y_labels = sorted(df["y"].dropna().unique().tolist())
        x_to_col = {x: i for i, x in enumerate(x_labels)}
        y_to_row = {y: i for i, y in enumerate(y_labels)}

        num_x = len(x_labels)
        num_y = len(y_labels)
        grids = {name: np.full((num_y, num_x), np.nan) for name in metric_names}

        for row in df.itertuples():
            col_idx = x_to_col[row.x]
            row_idx = y_to_row[row.y]
            for name in metric_names:
                val = getattr(row, name, None)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    grids[name][row_idx, col_idx] = val

        return {
            "df": df,
            "x_labels": x_labels,
            "y_labels": y_labels,
            "grids": grids,
        }

    # ==================================================================
    # 逻辑推断
    # ==================================================================

    @staticmethod
    def enrich_region_type(df: pd.DataFrame) -> pd.DataFrame:
        """基于所有图像的亮度统计学特征，推断每张图像所属的区域类型（正常、纯背景、半背景）。"""
        result = df.copy()
        if result.empty:
            result["region_type"] = pd.Series(dtype="object")
            return result

        # 根据有效图像的均值和标准差计算动态阈值
        valid_df = result[result["is_valid"] == True]
        mean_val = float(valid_df["valid_mean"].mean()) if len(valid_df) > 0 else 0.0
        std_val  = ImageQualityDataUtils.safe_std(valid_df["valid_mean"]) if len(valid_df) > 0 else 0.0
        dark_threshold      = mean_val - std_val
        very_dark_threshold = mean_val - 3 * std_val

        # 构建坐标索引以便判断是否是边缘位置
        x_coords, y_coords, x_index, y_index, coord_set = ImageQualityDataUtils.build_coord_index(result)

        def classify(row: pd.Series) -> str:
            # 只有在边缘的图像才可能是背景过渡区
            if not ImageQualityDataUtils.is_edge_position(
                row["x"], row["y"], x_coords, y_coords, x_index, y_index, coord_set
            ):
                return "normal"
            # 根据动态阈值判定具体类型
            if float(row["valid_mean"]) < very_dark_threshold:
                return "pure_bg"
            if float(row["valid_mean"]) < dark_threshold:
                return "half_bg"
            return "normal"

        result["region_type"] = result.apply(classify, axis=1)
        return result
