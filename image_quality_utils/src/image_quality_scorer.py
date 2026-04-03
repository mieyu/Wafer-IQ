# 图像质量评分打分模块（主观业务逻辑层）
# 专职将底层客观物理指标计算转换为业务上可读的 0~100 得分。
# 不负责真实的像素提取或特征计算。所有评分超参、曲线函数集中于此模块。
# 使用方法：from src.image_quality_scorer import ImageQualityScorer

from __future__ import annotations

import numpy as np
import pandas as pd


class ImageQualityScorer:
    """提供把客观计算结果映射为 0~100 综合分数的规则引擎。"""

    # ==================================================================
    # 1. 亮度评分
    # ==================================================================

    @staticmethod
    def calculate_visual_score(df: pd.DataFrame) -> float:
        """基于高斯分布惩罚偏离中心亮度（128）的视觉得分。

        Args:
            df (pd.DataFrame): 包含 "valid_mean" 列（有效像素亮度均值）的基础数据表。

        Returns:
            float: 计算出的综合视觉得分 (0.0 到 100.0)。
        """
        scores: list[float] = []
        for brightness in df["valid_mean"].tolist():
            sigma = 60 if brightness >= 128 else 40
            score = 100 * np.exp(-((brightness - 128) ** 2) / (2 * sigma**2))
            scores.append(float(score))
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def calculate_consistency_score(df: pd.DataFrame) -> float:
        """基于指数衰减惩罚偏离全局均值的一致性得分。

        Args:
            df (pd.DataFrame): 包含 "valid_mean" 列的基础数据表。

        Returns:
            float: 计算出的一致性得分 (0.0 到 100.0)。
        """
        global_mean = float(df["valid_mean"].mean()) if len(df) > 0 else 0.0
        scores: list[float] = []
        for brightness in df["valid_mean"].tolist():
            deviation = abs(float(brightness) - global_mean)
            scores.append(float(100 * np.exp(-(deviation / 20))))
        return float(np.mean(scores)) if scores else 0.0

    # ==================================================================
    # 2. 清晰度评分
    # ==================================================================

    @staticmethod
    def calculate_base_sharpness_score(fft_ratio: float) -> float:
        """基础清晰度得分。

        目前规则比较简单：主要以 FFT 高频占总能量的比例直接作为总体综合得分。

        Args:
            fft_ratio (float): FFT 高频能量占比（0.0 到 1.0）。

        Returns:
            float: 直接映射为 0.0 到 100.0 之间的分数。
        """
        return float(fft_ratio * 100.0)

    @staticmethod
    def calculate_uniformity_score(
        df: pd.DataFrame, sharpness_metrics_names: list[str]
    ) -> float:
        """根据所有清晰度算法指标的变异系数(CV)计算整体均匀度得分，CV越小均匀度越高。

        Args:
            df (pd.DataFrame): 包含所有清洗度指标的特征数据表。
            sharpness_metrics_names (list[str]): 清洗度指标名称列表。

        Returns:
            float: 批次内整体均匀度得分 (0.0 到 100.0)。
        """
        coefficient_of_variations: list[float] = []
        for name in sharpness_metrics_names:
            if name not in df.columns:
                continue
            col = df[name].dropna()
            if len(col) > 1 and col.mean() > 0:
                coefficient_of_variations.append(float(col.std() / col.mean()))
        if not coefficient_of_variations:
            return 0.0
        return float(100.0 * max(0.0, 1.0 - float(np.mean(coefficient_of_variations))))

    # ==================================================================
    # 3. 位移偏移评分
    # ==================================================================
    @staticmethod
    def calculate_shift_pair_score(
        delta_x: float, delta_y: float, response: float
    ) -> tuple[float, float, float]:
        """单对图像拼缝的偏移评分。

        Args:
            delta_x (float): X方向平移量（像素）。
            delta_y (float): Y方向平移量（像素）。
            response (float): 相位相关的响应强度（置信度）。

        Returns:
            tuple[float, float, float]:
                - total_score (float): 综合偏移总分。
                - score_delta_x (float): X方向得分。
                - score_delta_y (float): Y方向得分。
        """

        def _score(value: float) -> float:
            # 使用线性插值将位移量转化为 0-100 的得分（位移 0.1以内满分，超 5.0 得0分）
            return float(np.clip((abs(value) - 5.0) / (0.10 - 5.0) * 100.0, 0.0, 100.0))

        score_delta_x, score_delta_y = _score(delta_x), _score(delta_y)
        total_score = (score_delta_x + score_delta_y) / 2.0
        # 如果相位相关响应置信度过低，则总得分打85折
        if response < 0.05:
            total_score *= 0.85
        total_score = float(np.clip(total_score, 0.0, 100.0))
        return round(total_score, 2), round(score_delta_x, 2), round(score_delta_y, 2)

    # ==================================================================
    # 4. 畸变评分
    # ==================================================================
    @staticmethod
    def calculate_distortion_pair_score(
        rotation_degree: float, shear: float
    ) -> tuple[float, float, float]:
        """单对图像畸变量打分。

        Args:
            rotation_degree (float): 两图相对旋转角度（度）。
            shear (float): 两图相对切变量。

        Returns:
            tuple[float, float, float]:
                - distortion_score (float): 综合畸变总分。
                - score_rotation (float): 旋转得分。
                - score_shear (float): 切变得分。
        """
        # 旋转评分 (<=0.01满分, >=1.0零分)
        score_rotation = float(
            np.clip((abs(rotation_degree) - 1.0) / (0.01 - 1.0) * 100.0, 0.0, 100.0)
        )
        # 剪切评分 (<=0.001满分, >=0.05零分)
        score_shear = float(
            np.clip((abs(shear) - 0.05) / (0.001 - 0.05) * 100.0, 0.0, 100.0)
        )

        # 综合考虑：旋转占据主要影响(60%)，剪切占辅影响(40%)
        distortion_score = round(score_rotation * 0.6 + score_shear * 0.4, 2)
        return distortion_score, round(score_rotation, 2), round(score_shear, 2)
