"""
grid_mapper.py
--------------
将清晰度记录列表映射到虚拟二维网格。

说明：
  - 文件名中的 (x, y) 表示图块的网格坐标
  - 本模块将这些坐标归一化为从 0 开始的索引，构建二维数组
  - 缺少图块的位置填 NaN（用于热图中显示为空白）
"""

import numpy as np
import pandas as pd

# 四个清晰度指标
METRIC_NAMES = ["laplacian", "tenengrad", "fft", "brenner"]


def build_grid(records: list[dict]) -> dict:
    """
    根据记录列表构建虚拟网格。

    参数：
        records: tile_scanner 返回的记录列表
                 每条记录包含 x, y, laplacian, tenengrad, fft, brenner

    返回：
        一个字典，包含：
          - "df":       pandas DataFrame，每行对应一张图块
          - "x_labels": 按顺序排列的原始 x 坐标值
          - "y_labels": 按顺序排列的原始 y 坐标值
          - "grids":    dict，键为指标名称，值为 (num_y, num_x) 的二维 numpy 数组
    """
    df = pd.DataFrame(records)

    # ── 构建网格坐标映射 ─────────────────────────────────────────────────────
    x_labels = sorted(df["x"].unique())
    y_labels = sorted(df["y"].unique())
    x_to_col = {x: i for i, x in enumerate(x_labels)}
    y_to_row = {y: j for j, y in enumerate(y_labels)}

    num_x = len(x_labels)
    num_y = len(y_labels)

    # 四个指标各创建一个二维数组，初始值为 NaN
    grids = {name: np.full((num_y, num_x), np.nan) for name in METRIC_NAMES}

    # 将每条记录填入对应的网格位置
    for row in df.itertuples():
        col_idx = x_to_col[row.x]
        row_idx = y_to_row[row.y]
        for name in METRIC_NAMES:
            grids[name][row_idx, col_idx] = getattr(row, name)

    return {
        "df": df,
        "x_labels": x_labels,
        "y_labels": y_labels,
        "grids": grids,
    }
