# 图像像素级分析工具：ROI 提取、相位相关平移估算、SSIM 结构相似度计算。
# 所有方法仅涉及图像数组操作（cv2 / numpy / skimage），不含数据表或 I/O 逻辑。
# 供 image_quality_calculate 等上层模块复用。
# 使用方法：from src.image_quality_image_utils import ImageQualityImageUtils

from __future__ import annotations

import cv2
import numpy as np


class ImageQualityImageUtils:
    """提供图像像素级分析工具：ROI 提取、相位相关、SSIM 计算。"""

    # ==================================================================
    # ROI 提取
    # ==================================================================

    @staticmethod
    def extract_roi(
        gray_image_a: np.ndarray,
        gray_image_b: np.ndarray,
        overlap_length: int,
        stitch_direction: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """根据拼接方向提取两张图像重叠边缘的感兴趣区域 (ROI)。

        Args:
            gray_a (np.ndarray): 左/上图的灰度数组。
            gray_b (np.ndarray): 右/下图的灰度数组。
            overlap_length (int): 重叠区域的像素宽度。
            stitch_direction (str): 拼接方向，'horizontal' 或 'vertical'。

        Returns:
            tuple[np.ndarray, np.ndarray]: 尺寸对齐的两片重叠区域 (roi_a, roi_b)。
        """
        direction = stitch_direction.lower()
        if direction == "horizontal":
            roi_a, roi_b = (
                gray_image_a[:, -overlap_length:],
                gray_image_b[:, :overlap_length],
            )
        elif direction == "vertical":
            roi_a, roi_b = (
                gray_image_a[-overlap_length:, :],
                gray_image_b[:overlap_length, :],
            )
        else:
            raise ValueError(
                f"stitch_direction 必须是 'horizontal' 或 'vertical'，当前：{direction}"
            )
        # 确保尺寸对齐，截断可能存在的轻微维度不一致
        min_height = min(roi_a.shape[0], roi_b.shape[0])
        min_width = min(roi_a.shape[1], roi_b.shape[1])
        return roi_a[:min_height, :min_width], roi_b[:min_height, :min_width]

    # ==================================================================
    # 相位相关（平移估算）
    # ==================================================================

    @staticmethod
    def phase_correlation(
        roi_a: np.ndarray,
        roi_b: np.ndarray,
    ) -> tuple[float, float, float]:
        """应用相位相关法计算两个图像块之间的亚像素级别平移量。

        Args:
            roi_a (np.ndarray): 前置提取的感兴趣区域 A。
            roi_b (np.ndarray): 前置提取的感兴趣区域 B。

        Returns:
            tuple[float, float, float]:
                - dx (float): 水平平移像素数（亚像素精度）。
                - dy (float): 垂直平移像素数（亚像素精度）。
                - response (float): 相关响应强度，越大置信度越高。
        """
        float_array_a, float_array_b = roi_a.astype(np.float32), roi_b.astype(
            np.float32
        )
        # 添加汉宁窗减轻频域计算中的边缘效应
        window = cv2.createHanningWindow(float_array_a.shape[::-1], cv2.CV_32F)
        (delta_x, delta_y), response = cv2.phaseCorrelate(
            float_array_a, float_array_b, window
        )
        return float(delta_x), float(delta_y), float(response)

    # ==================================================================
    # SSIM 结构相似度
    # ==================================================================

    @staticmethod
    def calculate_ssim(
        img_a: np.ndarray,
        img_b: np.ndarray,
    ) -> float:
        """计算两幅灰度图像的结构相似度 (SSIM)。

        优先使用 skimage 中的优化版本；如果未安装则回退使用基于 OpenCV 的自定义实现。
        注意：当图像极小（< 3x3）时直接返回像素级相似度，避免 win_size 超界报错。

        Args:
            img_a (np.ndarray): 第一张图像的灰度数组。
            img_b (np.ndarray): 第二张图像的灰度数组。

        Returns:
            float: 计算出的结构相似度评分 (0.0 到 1.0)。
        """
        min_side = min(img_a.shape[0], img_a.shape[1])

        # 图像太小，无法计算有意义的结构相似度，退化为像素均值比较
        if min_side < 3:
            difference = np.mean(
                np.abs(img_a.astype(np.float64) - img_b.astype(np.float64))
            )
            return float(max(0.0, 1.0 - float(difference) / 255.0))

        # win_size 必须是奇数，且不超过图像最小边长（最大 7）
        window_size = min(7, min_side)
        if window_size % 2 == 0:
            window_size -= 1

        try:
            from skimage.metrics import structural_similarity as sk_ssim

            data_range = float(img_a.max() - img_a.min())
            data_range = 255.0 if data_range < 1.0 else data_range
            return float(
                sk_ssim(img_a, img_b, data_range=data_range, win_size=window_size)
            )
        except ImportError:
            pass

        # 回退逻辑：基于高斯模糊的 SSIM 计算（核大小适配 win_size）
        kernel_size = (window_size, window_size)
        sigma = 1.5
        constant_1, constant_2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        array_a, array_b = img_a.astype(np.float64), img_b.astype(np.float64)
        mean_a = cv2.GaussianBlur(array_a, kernel_size, sigma)
        mean_b = cv2.GaussianBlur(array_b, kernel_size, sigma)
        ssim_map = (
            (2 * mean_a * mean_b + constant_1)
            * (
                2
                * (
                    cv2.GaussianBlur(array_a * array_b, kernel_size, sigma)
                    - mean_a * mean_b
                )
                + constant_2
            )
        ) / (
            (mean_a**2 + mean_b**2 + constant_1)
            * (
                (cv2.GaussianBlur(array_a**2, kernel_size, sigma) - mean_a**2)
                + (cv2.GaussianBlur(array_b**2, kernel_size, sigma) - mean_b**2)
                + constant_2
            )
        )
        return float(ssim_map.mean())

    # ==================================================================
    # 清晰度底层计算 (无状态的数学特征提取)
    # ==================================================================

    @staticmethod
    def laplacian_variance(gray: np.ndarray) -> float:
        """使用拉普拉斯算子计算图像方差，方差越大边缘信息越丰富。"""
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def tenengrad(gray: np.ndarray) -> float:
        """使用 Sobel 算子求水平和垂直梯度平方和均值。"""
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.mean(gradient_x**2 + gradient_y**2))

    @staticmethod
    def calculate_fft_high_frequency_ratio(
        gray: np.ndarray, cutoff_ratio: float = 0.1
    ) -> float:
        """利用快速傅里叶变换(FFT)计算高频能量占总能量的比例，反映细节丰富度。"""
        height, width = gray.shape
        magnitude = np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(np.float64))))
        center_y, center_x = height // 2, width // 2
        distance = np.sqrt(
            (np.ogrid[:height, :width][0] - center_y) ** 2
            + (np.ogrid[:height, :width][1] - center_x) ** 2
        )
        total_power = magnitude.sum()
        if total_power == 0:
            return 0.0
        return float(
            magnitude[distance > cutoff_ratio * min(height, width) / 2].sum()
            / total_power
        )

    @staticmethod
    def brenner_gradient(gray: np.ndarray) -> float:
        """计算 Brenner 梯度（相差2个像素的差值平方均值）。"""
        img = gray.astype(np.float64)
        return float(np.mean((img[:, 2:] - img[:, :-2]) ** 2))
