"""
sharpness_metrics.py
--------------------
提供四种图像清晰度评估指标，输入均为灰度图（numpy 二维数组）。
值越高表示图像越清晰。
"""

import cv2
import numpy as np


def laplacian_variance(gray: np.ndarray) -> float:
    """
    Laplacian 方差法。
    原理：清晰图像边缘多，Laplacian 响应强，方差大。
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def tenengrad(gray: np.ndarray) -> float:
    """
    Tenengrad 梯度能量法。
    原理：用 Sobel 算子计算水平/垂直梯度，取梯度平方和的均值。
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx**2 + gy**2))


def compute_sharpness_score(fft_val: float) -> float:
    """
    计算基于 FFT 的绝对清晰度评分。
    直接将高频能量占比乘以 100，分数域为 0~100。
    """
    return fft_val * 100.0


def compute_uniformity_score(df) -> float:
    """
    计算批内均匀性评分（百分制）。
    先计算四种指标的变异系数（CV = σ / μ）均值。
    再将 CV 映射为 0~100 分：完全均匀(CV=0) 得 100 分，CV≥1 时得 0 分。
    分数越高，表示该批次图块清晰度越均匀。
    """
    metrics = ["laplacian", "tenengrad", "fft", "brenner"]
    cvs = []

    for metric in metrics:
        if metric in df.columns:
            col = df[metric].dropna()
            if len(col) > 1 and col.mean() > 0:
                cv = col.std() / col.mean()
                cvs.append(cv)

    if not cvs:
        return 0.0

    cv_mean = float(np.mean(cvs))
    # 映射公式： score = 100 * max(0, 1 - cv_mean)
    # 例如：平均 CV 为 0.22 时，得分 = 100 * (1 - 0.22) = 78 分
    score = 100.0 * max(0.0, 1.0 - cv_mean)
    return score


def fft_high_freq_ratio(gray: np.ndarray, cutoff_ratio: float = 0.1) -> float:
    """
    FFT 高频能量比。
    原理：清晰图像含有丰富高频成分，频谱中远离中心的高频区域能量占比更高。

    参数：
        cutoff_ratio: 高频区域半径 = cutoff_ratio × min(H, W) / 2，默认 0.1
    """
    h, w = gray.shape
    # 执行二维 FFT 并将零频移到中心
    f = np.fft.fft2(gray.astype(np.float64))
    f_shift = np.fft.fftshift(f)
    magnitude = np.abs(f_shift)

    # 建立以图像中心为原点的坐标网格
    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.ogrid[:h, :w]
    dist = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)

    # 高频区域 = 距中心大于 cutoff 的部分
    cutoff = cutoff_ratio * min(h, w) / 2
    high_freq_mask = dist > cutoff

    total_power = magnitude.sum()
    if total_power == 0:
        return 0.0
    high_freq_power = magnitude[high_freq_mask].sum()
    return float(high_freq_power / total_power)


def brenner_gradient(gray: np.ndarray) -> float:
    """
    Brenner 梯度法。
    原理：计算水平方向跨越 2 个像素的差值平方和均值。
    公式：mean( (f(x+2, y) - f(x, y))^2 )
    """
    img = gray.astype(np.float64)
    # 取第 3 列到末尾 减去 第 1 列到倒数第 3 列
    diff = img[:, 2:] - img[:, :-2]
    return float(np.mean(diff**2))
