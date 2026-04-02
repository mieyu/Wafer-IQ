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

    def extract_roi(
        self,
        gray_a: np.ndarray,
        gray_b: np.ndarray,
        overlap_length: int,
        stitch_direction: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        根据拼接方向提取两张图像重叠边缘的感兴趣区域 (ROI)。

        参数：
            gray_a:           左/上图灰度数组
            gray_b:           右/下图灰度数组
            overlap_length:   重叠区像素宽度
            stitch_direction: 拼接方向，'horizontal' 或 'vertical'

        返回：(roi_a, roi_b) 尺寸对齐的两片重叠区域
        """
        d = stitch_direction.lower()
        if d == "horizontal":
            roi_a, roi_b = gray_a[:, -overlap_length:], gray_b[:, :overlap_length]
        elif d == "vertical":
            roi_a, roi_b = gray_a[-overlap_length:, :], gray_b[:overlap_length, :]
        else:
            raise ValueError(f"stitch_direction 必须是 'horizontal' 或 'vertical'，当前：{d}")
        # 确保尺寸对齐，截断可能存在的轻微维度不一致
        min_h = min(roi_a.shape[0], roi_b.shape[0])
        min_w = min(roi_a.shape[1], roi_b.shape[1])
        return roi_a[:min_h, :min_w], roi_b[:min_h, :min_w]

    # ==================================================================
    # 相位相关（平移估算）
    # ==================================================================

    def phase_correlation(
        self,
        roi_a: np.ndarray,
        roi_b: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        应用相位相关法计算两个图像块之间的亚像素级别平移量。

        返回：(dx, dy, response)
            dx, dy: 平移像素数（亚像素精度）
            response: 相关响应强度，越大置信度越高
        """
        fa, fb = roi_a.astype(np.float32), roi_b.astype(np.float32)
        # 添加汉宁窗减轻频域计算中的边缘效应
        win = cv2.createHanningWindow(fa.shape[::-1], cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(fa, fb, win)
        return float(dx), float(dy), float(response)

    # ==================================================================
    # SSIM 结构相似度
    # ==================================================================

    def calc_ssim(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
    ) -> float:
        """
        计算两幅灰度图像的结构相似度 (SSIM)。
        优先使用 skimage 中的优化版本；如果未安装则回退使用基于 OpenCV 的自定义实现。

        注意：当图像极小（< 3×3）时直接返回像素级相似度，避免 win_size 超界报错。
        """
        min_side = min(img_a.shape[0], img_a.shape[1])

        # 图像太小，无法计算有意义的结构相似度，退化为像素均值比较
        if min_side < 3:
            diff = np.mean(np.abs(img_a.astype(np.float64) - img_b.astype(np.float64)))
            return float(max(0.0, 1.0 - diff / 255.0))

        # win_size 必须是奇数，且不超过图像最小边长（最大 7）
        win_size = min(7, min_side)
        if win_size % 2 == 0:
            win_size -= 1

        try:
            from skimage.metrics import structural_similarity as sk_ssim
            dr = float(img_a.max() - img_a.min())
            dr = 255.0 if dr < 1.0 else dr
            return float(sk_ssim(img_a, img_b, data_range=dr, win_size=win_size))
        except ImportError:
            pass

        # 回退逻辑：基于高斯模糊的 SSIM 计算（核大小适配 win_size）
        ksize = (win_size, win_size)
        sigma = 1.5
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        a, b = img_a.astype(np.float64), img_b.astype(np.float64)
        mu_a = cv2.GaussianBlur(a, ksize, sigma)
        mu_b = cv2.GaussianBlur(b, ksize, sigma)
        ssim_map = (
            (2 * mu_a * mu_b + c1) *
            (2 * (cv2.GaussianBlur(a * b, ksize, sigma) - mu_a * mu_b) + c2)
        ) / (
            (mu_a ** 2 + mu_b ** 2 + c1) *
            ((cv2.GaussianBlur(a ** 2, ksize, sigma) - mu_a ** 2) +
             (cv2.GaussianBlur(b ** 2, ksize, sigma) - mu_b ** 2) + c2)
        )
        return float(ssim_map.mean())

    # ==================================================================
    # 清晰度底层计算 (无状态的数学特征提取)
    # ==================================================================

    def laplacian_variance(self, gray: np.ndarray) -> float:
        """使用拉普拉斯算子计算图像方差，方差越大边缘信息越丰富。"""
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def tenengrad(self, gray: np.ndarray) -> float:
        """使用 Sobel 算子求水平和垂直梯度平方和均值。"""
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.mean(gx ** 2 + gy ** 2))

    def fft_high_freq_ratio(self, gray: np.ndarray, cutoff_ratio: float = 0.1) -> float:
        """利用快速傅里叶变换(FFT)计算高频能量占总能量的比例，反映细节丰富度。"""
        h, w = gray.shape
        magnitude = np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(np.float64))))
        cy, cx = h // 2, w // 2
        dist = np.sqrt((np.ogrid[:h, :w][0] - cy) ** 2 + (np.ogrid[:h, :w][1] - cx) ** 2)
        total_power = magnitude.sum()
        if total_power == 0:
            return 0.0
        return float(magnitude[dist > cutoff_ratio * min(h, w) / 2].sum() / total_power)

    def brenner_gradient(self, gray: np.ndarray) -> float:
        """计算 Brenner 梯度（相差2个像素的差值平方均值）。"""
        img = gray.astype(np.float64)
        return float(np.mean((img[:, 2:] - img[:, :-2]) ** 2))
