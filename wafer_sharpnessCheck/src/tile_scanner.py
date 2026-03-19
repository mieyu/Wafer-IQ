"""
tile_scanner.py
---------------
扫描指定目录，找出所有命名格式为 `0000_x_y.png` 的分块图像，
并行计算每张图的四种清晰度指标，返回记录列表。
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
from tqdm import tqdm

from src.sharpness_metrics import (
    brenner_gradient,
    compute_sharpness_score,
    fft_high_freq_ratio,
    laplacian_variance,
    tenengrad,
)

logger = logging.getLogger(__name__)

# 匹配文件名 0000_x_y.png，捕获 x 和 y
FILENAME_PATTERN = re.compile(r"^\d+_(\d+)_(\d+)\.png$")


def _process_one_tile(filepath: Path) -> dict | None:
    """
    处理单张分块图像：
    1. 从文件名解析 (x, y) 坐标
    2. 读取图像并转为灰度
    3. 计算四种清晰度指标
    返回一个字典，失败时返回 None。
    """
    match = FILENAME_PATTERN.match(filepath.name)
    if not match:
        return None  # 文件名不符合格式，跳过

    x = int(match.group(1))
    y = int(match.group(2))

    # 读取图像
    img_bgr = cv2.imread(str(filepath))
    if img_bgr is None:
        logger.warning(f"无法读取图像：{filepath}")
        return None

    # 转为灰度图
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 计算各项指标
    laplacian = laplacian_variance(gray)
    tenengrad_val = tenengrad(gray)
    fft_val = fft_high_freq_ratio(gray)
    brenner = brenner_gradient(gray)

    return {
        "filename": filepath.name,
        "x": x,
        "y": y,
        "laplacian": laplacian,
        "tenengrad": tenengrad_val,
        "fft": fft_val,
        "brenner": brenner,
        "sharpness_score": compute_sharpness_score(fft_val),
    }


def scan_tiles(data_dir: str, num_workers: int = 8) -> list[dict]:
    """
    扫描 data_dir 目录，并行处理所有分块图像。

    参数：
        data_dir:    图像所在目录路径
        num_workers: 并行线程数

    返回：
        records: 每个元素为一张图的清晰度记录字典
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"目录不存在：{data_dir}")

    # 收集所有符合格式的 png 文件
    all_files = [f for f in data_path.glob("*.png") if FILENAME_PATTERN.match(f.name)]
    logger.info(
        f"共找到 {len(all_files)} 张分块图像，开始并行处理（线程数={num_workers}）..."
    )

    records = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(_process_one_tile, f): f for f in all_files}

        # 使用进度条等待结果
        for future in tqdm(
            as_completed(future_to_file), total=len(all_files), desc="计算清晰度"
        ):
            result = future.result()
            if result is not None:
                records.append(result)

    logger.info(f"处理完成，有效记录 {len(records)} 条。")
    return records
