# 图像质量指标计算模块：封装亮度、清晰度、位移偏移、畸变等各类指标的计算逻辑。
# 单图指标（亮度、清晰度）与双图指标（偏移、畸变）均在此模块内，通过滑动窗口机制统一调度。
# 坐标来源：从 placements-BF.yml 的 views[].index 字段解析，与 wafer_shiftCheck/warpCheck 保持一致。
# 使用方法：calc = ImageQualityCalculator(); df = calc.batch_calculate(data_dir); metrics = calc.get_brightness_metrics(df)

from __future__ import annotations

import logging
from collections import defaultdict

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.image_quality_data_utils import ImageQualityDataUtils
from src.image_quality_image_utils import ImageQualityImageUtils
from src.image_quality_scorer import ImageQualityScorer

logger = logging.getLogger(__name__)

# 定义支持的清晰度算法列表
SHARPNESS_METRICS = ["laplacian", "tenengrad", "fft", "brenner"]

# shift / distortion 结果的缺省值（第一列图像无前邻，填 NaN）
_SHIFT_EMPTY: dict[str, Any] = {
    "shift_dx": np.nan, "shift_dy": np.nan,
    "shift_phase_response": np.nan, "shift_ssim_before": np.nan,
    "shift_ssim_after": np.nan, "shift_ssim_delta": np.nan,
    "shift_suspicious": np.nan, "shift_score": np.nan,
    "shift_score_dx": np.nan, "shift_score_dy": np.nan,
    "shift_bias_direction": "",
    "shift_bias_ratio_x": np.nan, "shift_bias_ratio_y": np.nan,
}
_DISTORTION_EMPTY: dict[str, Any] = {
    "dist_rotation_deg": np.nan, "dist_shear": np.nan,
    "dist_scale_x": np.nan, "dist_scale_y": np.nan,
    "dist_inliers": np.nan, "dist_method": "",
    "dist_score": np.nan, "dist_score_rotation": np.nan, "dist_score_shear": np.nan,
}


class ImageQualityCalculator:
    """图像质量计算核心类，负责调度单图与双图各种质量指标的计算。"""

    def __init__(
        self,
        background_threshold: int = 30,
        valid_ratio_threshold: float = 0.1,
        high_brightness_threshold: int = 100,
    ) -> None:
        """初始化计算器参数。
        
        Args:
            background_threshold (int): 区分前景和背景的灰度阈值，低于此值认为是背景。
            valid_ratio_threshold (float): 有效像素（非背景）比例的最低阈值，用于判定图像是否包含足够有效信息。
            high_brightness_threshold (int): 高亮度判定阈值。
        """
        self.background_threshold    = background_threshold
        self.valid_ratio_threshold   = valid_ratio_threshold
        self.high_brightness_threshold = high_brightness_threshold

    # ==================================================================
    # 统一批量入口：滑动窗口机制，读图一次，分发给所有计算模块
    # ==================================================================

    def batch_calculate(
        self,
        data_dir: str | Path,
        yaml_filename: str = "placements-BF.yml",
        stitch_direction: str = "horizontal",
        num_workers: int = 8,
    ) -> pd.DataFrame:
        """滑动窗口批量处理：
          - 坐标与重叠区像素数均从 placements-BF.yml 解析，与 wafer_shiftCheck/warpCheck 保持一致。
          - 单双图计算重新引入 ThreadPoolExecutor 多线程并行加速。
          - 每张图只读取一次，灰度图缓存在内存中复用。

        Args:
            data_dir (str | Path): 图像目录（同时也是 YAML 所在目录）。
            yaml_filename (str): YAML 配置文件名，默认 placements-BF.yml。
            stitch_direction (str): 拼接方向，"horizontal" 或 "vertical"。
            num_workers (int): 多线程并发数，默认 8。

        Returns:
            pd.DataFrame: 整理后的全部检测指标和各类统计特征矩阵。
        """
        data_dir = Path(data_dir)

        # ── 从 YAML 解析图块坐标与 overlap_px ────────────────────────────
        yaml_path = data_dir / yaml_filename
        yaml_meta = ImageQualityDataUtils.load_yaml_meta(yaml_path)
        if not yaml_meta:
            logger.error(f"无法加载 YAML 配置，终止分析：{yaml_path}")
            return pd.DataFrame()

        overlap_length: int = yaml_meta["overlap_px"]
        views: dict[tuple[int, int], Path] = yaml_meta["views"]
        logger.info(f"YAML 解析完成：{len(views)} 个图块，重叠区域 {overlap_length} px")

        # 过滤出实际存在的文件
        image_files = [p for p in views.values() if p.exists()]
        if not image_files:
            logger.warning(f"YAML 中的图块文件均不存在，请检查路径：{data_dir}")
            return pd.DataFrame()

        # ── 步骤 1：顺序单图计算，同时缓存灰度图 ──────────────────────
        logger.info(f"共找到 {len(image_files)} 张图像，开始单图分析...")

        # 构建 path → (col, row) 的反向映射，供后续使用
        path_to_coord: dict[Path, tuple[int, int]] = {p: coord for coord, p in views.items() if p.exists()}

        single_results: dict[Path, dict[str, Any]] = {}
        gray_cache: dict[Path, np.ndarray] = {}

        from concurrent.futures import ThreadPoolExecutor, as_completed

        # 采用 ThreadPoolExecutor 并发执行单张图的读取和指标计算
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_f = {
                executor.submit(self._read_and_calc_single, f, path_to_coord.get(f)): f
                for f in image_files
            }
            for future in tqdm(as_completed(future_to_f), total=len(image_files), desc="Single-image analysis"):
                f = future_to_f[future]
                try:
                    result = future.result()
                    if result is not None:
                        gray_cache[f] = result.pop("_gray")   # 单独存灰度图，不进 DataFrame
                        single_results[f] = result
                except Exception as exc:
                    logger.error(f"处理单图 {f.name} 时出错: {exc}")

        # ── 步骤 2：按列分组，滑动窗口组装并行任务 ──────────────────────
        # 将文件按 col（YAML index 第一维）分组，col 相同的图在同一列
        col_groups: dict[int, list[Path]] = defaultdict(list)
        for path in single_results:
            coord = path_to_coord.get(path)
            if coord is not None:
                col_groups[coord[0]].append(path)

        sorted_cols = sorted(col_groups.keys())
        dual_results: dict[Path, dict[str, Any]] = {}   # key = 图像路径
        dual_tasks = []

        logger.info(f"共 {len(sorted_cols)} 列，开始滑动窗口双图任务构造与多线程分析...")
        prev_col: int | None = None
        for col in sorted_cols:
            curr_paths = col_groups[col]
            if prev_col is None:
                # 第一列：无前邻，双图结果填空
                for path in curr_paths:
                    dual_results[path] = {**_SHIFT_EMPTY, **_DISTORTION_EMPTY}
            else:
                prev_paths = col_groups[prev_col]
                # 按 row（YAML index 第二维）匹配同行相邻图块
                prev_by_y = {path_to_coord[p][1]: p for p in prev_paths if p in path_to_coord}
                curr_by_y = {path_to_coord[p][1]: p for p in curr_paths if p in path_to_coord}

                for y, curr_path in curr_by_y.items():
                    if y not in prev_by_y:
                        # 当前行在前一列无对应图块
                        dual_results[curr_path] = {**_SHIFT_EMPTY, **_DISTORTION_EMPTY}
                        continue

                    # 提取前后相邻图的路径及灰度缓存
                    prev_path = prev_by_y[y]
                    gray_prev = gray_cache.get(prev_path)
                    gray_curr = gray_cache.get(curr_path)

                    if gray_prev is None or gray_curr is None:
                        dual_results[curr_path] = {**_SHIFT_EMPTY, **_DISTORTION_EMPTY}
                        continue

                    # 将有上下文的图片对装入并发任务列表
                    dual_tasks.append((curr_path, gray_prev, gray_curr, overlap_length, stitch_direction))

                # 前一列中没有在当前列找到对应行的图块，也补空
                for y, prev_path in prev_by_y.items():
                    if y not in curr_by_y:
                        dual_results[prev_path] = dual_results.get(prev_path, {**_SHIFT_EMPTY, **_DISTORTION_EMPTY})

            prev_col = col

        # 使用多线程执行双图 SIFT 匹配运算
        def _exec_dual(c_path, g_prev, g_curr, overlap, stitch):
            shift = self.shift_calculate(g_prev, g_curr, overlap, stitch)
            dist = self.distortion_calculate(g_prev, g_curr, overlap, stitch, shift["shift_dx"], shift["shift_dy"])
            return c_path, {**shift, **dist}

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_curr = {
                executor.submit(_exec_dual, *task): task[0]
                for task in dual_tasks
            }
            for future in tqdm(as_completed(future_to_curr), total=len(dual_tasks), desc="Dual-image SIFT & FFT analysis"):
                c_path = future_to_curr[future]
                try:
                    res = future.result()
                    dual_results[c_path] = res[1]
                except Exception as exc:
                    logger.error(f"处理双图对 {c_path.name} 时出错: {exc}")
                    dual_results[c_path] = {**_SHIFT_EMPTY, **_DISTORTION_EMPTY}

        # ── 步骤 3：合并单图 + 双图结果 ──────────────────────────────────
        rows: list[dict[str, Any]] = []
        for path, single in single_results.items():
            dual = dual_results.get(path, {**_SHIFT_EMPTY, **_DISTORTION_EMPTY})
            # 组合成完整的单行记录
            rows.append({**single, **dual})

        if not rows:
            return pd.DataFrame()

        # 转换为 DataFrame 并基于统计值进行区域类型打标
        df = pd.DataFrame(rows)
        return ImageQualityDataUtils.enrich_region_type(df)

    def _read_and_calc_single(
        self, image_path: Path, coord: tuple[int, int] | None = None
    ) -> dict[str, Any] | None:
        """
        读取单张图片，执行单图计算，返回结果（含灰度图缓存）。
        coord: 来自 YAML 的 (col, row) 坐标元组；若为 None 则坐标记为 NaN。
        """
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            logger.warning(f"无法读取图像：{image_path}")
            return None

        # 统一转为灰度图进行计算
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # 坐标直接使用 YAML 中的 (col, row)，不再从文件名解析
        x = coord[0] if coord is not None else None
        y = coord[1] if coord is not None else None

        return {
            "_gray": gray,                              # 临时缓存，batch_calculate 中弹出
            "filename":   image_path.name,
            "image_path": str(image_path),
            "x": x,
            "y": y,
            **self.brightness_calculate(gray),
            **self.sharpness_calculate(gray),
        }

    # ==================================================================
    # 亮度计算
    # ==================================================================

    def brightness_calculate(self, gray: np.ndarray) -> dict[str, Any]:
        """计算单幅灰度图的统计学特征。

        基于给定背景判定阈值和有效像素比例筛选出非背景的主体对象。

        Args:
            gray (np.ndarray): 输入的单图灰度数组。

        Returns:
            dict[str, Any]: 亮度统计特征字典。
        """
        # 利用背景阈值提取有效前景像素
        valid_mask  = gray > self.background_threshold
        valid_ratio = float(np.sum(valid_mask) / gray.size)
        # 如果有效像素比例太低，置为空数组避免计算错误
        valid_pixels = gray[valid_mask] if valid_ratio > 0.01 else np.array([], dtype=gray.dtype)
        return {
            "mean_brightness": float(np.mean(gray)),
            "std_brightness":  float(np.std(gray)),
            "min_brightness":  float(np.min(gray)),
            "max_brightness":  float(np.max(gray)),
            "valid_ratio":     valid_ratio,
            "valid_mean":      float(np.mean(valid_pixels)) if len(valid_pixels) > 0 else 0.0,
            "valid_std":       float(np.std(valid_pixels))  if len(valid_pixels) > 0 else 0.0,
            "is_valid":        valid_ratio > self.valid_ratio_threshold,
        }

    def get_brightness_metrics(
        self,
        stats_source: str | Path | pd.DataFrame,
        data_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """对整个数据集的亮度指标进行汇总、聚合和异常值统计。

        Args:
            stats_source (str | Path | pd.DataFrame): 亮度明细数据表格对象或路径。
            data_dir (str | Path | None): 附加的输入路径，用于报告标记。

        Returns:
            dict[str, Any]: 亮度高阶汇总特征（平均指、异常值）。
        """
        # 支持直接传入 DataFrame 或者 CSV 文件路径
        if isinstance(stats_source, pd.DataFrame):
            df = stats_source.copy()
        else:
            df = pd.read_csv(stats_source)

        if "region_type" not in df.columns:
            df = ImageQualityDataUtils.enrich_region_type(df)

        pure_bg_count = int((df["region_type"] == "pure_bg").sum())
        half_bg_count = int((df["region_type"] == "half_bg").sum())
        normal_count  = int((df["region_type"] == "normal").sum())
        total_images  = int(len(df))

        # 排除纯背景后进行统计
        non_bg_df = df[df["region_type"] != "pure_bg"]
        normal_df = df[df["region_type"] == "normal"]

        avg_brightness_all = float(non_bg_df["valid_mean"].mean()) if len(non_bg_df) > 0 else 0.0
        std_brightness_all = ImageQualityDataUtils.safe_std(non_bg_df["valid_mean"]) if len(non_bg_df) > 0 else 0.0
        # CV(变异系数) 反映数据的离散程度
        cv_all = (std_brightness_all / avg_brightness_all * 100) if avg_brightness_all > 0 else 0.0

        if len(normal_df) > 0:
            high_brightness_count = int((normal_df["valid_mean"] >= self.high_brightness_threshold).sum())
            high_brightness_ratio = high_brightness_count / len(normal_df) * 100
        else:
            high_brightness_count = 0
            high_brightness_ratio = 0.0

        # 计算特定的打分
        score_visual      = ImageQualityScorer.calc_visual_score(non_bg_df)
        score_consistency = ImageQualityScorer.calc_consistency_score(non_bg_df)

        # 统计正常区域的高级指标及偏离中心分布的异常数量
        if len(normal_df) > 0:
            mean_brightness   = float(normal_df["valid_mean"].mean())
            median_brightness = float(normal_df["valid_mean"].median())
            std_brightness    = ImageQualityDataUtils.safe_std(normal_df["valid_mean"])
            min_brightness    = float(normal_df["valid_mean"].min())
            max_brightness    = float(normal_df["valid_mean"].max())
            cv_percent        = (std_brightness / mean_brightness * 100) if mean_brightness > 0 else 0.0
            threshold_low     = mean_brightness - 3 * std_brightness
            threshold_high    = mean_brightness + 2 * std_brightness
            dark_threshold    = mean_brightness - std_brightness
            outliers_low      = normal_df[normal_df["valid_mean"] < threshold_low]
            outliers_high     = normal_df[normal_df["valid_mean"] > threshold_high]
            dark_regions      = normal_df[normal_df["valid_mean"] < dark_threshold]
            dark_ratio_percent = len(dark_regions) / len(normal_df) * 100
        else:
            mean_brightness = median_brightness = std_brightness = 0.0
            min_brightness  = max_brightness = cv_percent = 0.0
            threshold_low   = threshold_high = dark_threshold = 0.0
            outliers_low    = outliers_high = dark_regions = pd.DataFrame()
            dark_ratio_percent = 0.0

        # 解析并记录数据来源目录
        resolved_data_dir = str(data_dir) if data_dir is not None else ""
        if not resolved_data_dir and not isinstance(stats_source, pd.DataFrame):
            resolved_data_dir = str(Path(stats_source).parent)

        return {
            "total_images": total_images,
            "pure_bg_count": pure_bg_count,
            "half_bg_count": half_bg_count,
            "normal_count": normal_count,
            "avg_brightness_all": avg_brightness_all,
            "std_brightness_all": std_brightness_all,
            "cv_all": float(cv_all),
            "score_visual": score_visual,
            "score_consistency": score_consistency,
            "high_brightness_count": high_brightness_count,
            "high_brightness_ratio": float(high_brightness_ratio),
            "high_brightness_threshold": self.high_brightness_threshold,
            "mean_brightness": mean_brightness,
            "median_brightness": median_brightness,
            "std_brightness": std_brightness,
            "min_brightness": min_brightness,
            "max_brightness": max_brightness,
            "brightness_range": max_brightness - min_brightness,
            "cv_percent": cv_percent,
            "outliers_low_count": int(len(outliers_low)),
            "outliers_high_count": int(len(outliers_high)),
            "dark_regions_count": int(len(dark_regions)),
            "dark_ratio_percent": float(dark_ratio_percent),
            "threshold_low": float(threshold_low),
            "threshold_high": float(threshold_high),
            "dark_threshold": float(dark_threshold),
            "data_dir": resolved_data_dir,
        }



    # ==================================================================
    # 清晰度计算
    # ==================================================================

    def sharpness_calculate(self, gray: np.ndarray) -> dict[str, Any]:
        """计算单张灰度图的多种清晰度底层指标特征。

        Args:
            gray (np.ndarray): 已读取的有效灰度图数组。

        Returns:
            dict[str, Any]: 包含 Laplacian方差、十种梯度能量、FFT能量比等的各阶清洗度特征分量字典。
        """
        fft_val = ImageQualityImageUtils.fft_high_freq_ratio(gray)
        return {
            "laplacian":       ImageQualityImageUtils.laplacian_variance(gray),
            "tenengrad":       ImageQualityImageUtils.tenengrad(gray),
            "fft":             fft_val,
            "brenner":         ImageQualityImageUtils.brenner_gradient(gray),
            "sharpness_score": ImageQualityScorer.calc_base_sharpness_score(fft_val),
        }

    def get_sharpness_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        """对所有图块的各种清洗度底层参数做宏观统计分析和汇总。

        Args:
            df (pd.DataFrame): 单图明细数据表。

        Returns:
            dict[str, Any]: 整体均摊汇总性能体系（包括均匀性评估等）。
        """
        if df.empty:
            return {}
        metrics: dict[str, Any] = {
            "total_images":      int(len(df)),
            "sharpness_score":   float(df["sharpness_score"].mean()) if "sharpness_score" in df.columns else 0.0,
            "uniformity_score":  ImageQualityScorer.calc_uniformity_score(df, SHARPNESS_METRICS),
        }
        # 遍历所有支持的清晰度算法，并分别计算统计量
        for name in SHARPNESS_METRICS:
            if name not in df.columns:
                continue
            col = df[name].dropna()
            metrics[f"{name}_mean"]   = float(col.mean())
            metrics[f"{name}_std"]    = ImageQualityDataUtils.safe_std(col)
            metrics[f"{name}_median"] = float(col.median())
            metrics[f"{name}_min"]    = float(col.min())
            metrics[f"{name}_max"]    = float(col.max())
        return metrics


    # ==================================================================
    # 位移偏移计算（双图）
    # ==================================================================

    def shift_calculate(
        self,
        gray_prev: np.ndarray,
        gray_curr: np.ndarray,
        overlap_length: int,
        stitch_direction: str,
    ) -> dict[str, Any]:
        """计算拼缝处相邻两张图在像素域的平移量。

        所有结果键均以 'shift_' 作为前缀。包含防误判处理如结构相似度计算校验（SSIM）。

        Args:
            gray_prev (np.ndarray): 前驱节点图的灰度矩阵。
            gray_curr (np.ndarray): 当前节点图的灰度矩阵。
            overlap_length (int): 定义两图边缘之间的预估重合长度。
            stitch_direction (str): 图块排布与邻里的接壤方向。

        Returns:
            dict[str, Any]: 计算得出的拼合处坐标系平移变量 'dx'，'dy' 以及结构相似度的诊断日志集合。
        """
        # 提取重叠的感兴趣区域(ROI)
        roi_a, roi_b = ImageQualityImageUtils.extract_roi(gray_prev, gray_curr, overlap_length, stitch_direction)
        # 通过相位相关法计算平移量及置信度响应
        dx, dy, response = ImageQualityImageUtils.phase_correlation(roi_a, roi_b)

        # 异常巨大偏移量拦截
        # 物理上，相邻晶圆图像的实际偏差只会在小范围（通常十几像素）。
        # 如果由于纯背景噪声、周期性纹理使得 FFT 相位相关找到了极远的伪峰（比如几百像素），
        # 这种数据不仅无意义，还会严重带偏全局的 mean 和 std 统计。
        max_valid_shift = max(50.0, overlap_length * 0.25)
        if abs(dx) > max_valid_shift or abs(dy) > max_valid_shift:
            logger.debug(f"丢弃异常巨大的平移解算结果: dx={dx:.1f}, dy={dy:.1f}")
            empty = {**_SHIFT_EMPTY}
            empty["shift_suspicious"] = True
            empty["shift_bias_direction"] = "匹配失败(偏移异常过大)"
            return empty

        # SSIM 验证
        h, w = roi_a.shape[:2]
        ssim_before = ImageQualityImageUtils.calc_ssim(roi_a, roi_b)
        # 使用算出的平移量构建仿射矩阵并对齐图像 B
        T = np.float32([[1, 0, -dx], [0, 1, -dy]])
        roi_b_aligned = cv2.warpAffine(roi_b, T, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        # 裁剪掉因平移产生的无效边缘，重新计算 SSIM
        mx, my = int(abs(dx)) + 1, int(abs(dy)) + 1
        x1, x2, y1, y2 = mx, w - mx, my, h - my
        crop_a = roi_a[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else roi_a
        crop_b = roi_b_aligned[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else roi_b_aligned
        ssim_after = ImageQualityImageUtils.calc_ssim(crop_a, crop_b)
        # 如果对齐后 SSIM 反而下降，视为可疑计算结果
        suspicious = bool((ssim_after - ssim_before) < 0)

        # 偏移评分交由评分引擎
        total, s_dx, s_dy = ImageQualityScorer.calc_shift_pair_score(dx, dy, response)

        # 分析偏移的主要方向
        adx, ady = abs(dx), abs(dy)
        t_o = adx + ady
        if t_o < 1e-6:
            bx, by, bias_dir = 0.5, 0.5, "无明显偏移"
        else:
            bx, by = adx / t_o, ady / t_o
            if bx >= 0.7:
                bias_dir = "主要偏向 X 轴（水平漂移）"
            elif by >= 0.7:
                bias_dir = "主要偏向 Y 轴（垂直漂移）"
            else:
                bias_dir = f"X/Y 混合偏移（X占{bx*100:.0f}% Y占{by*100:.0f}%）"

        return {
            "shift_dx":              float(dx),
            "shift_dy":              float(dy),
            "shift_phase_response":  float(response),
            "shift_ssim_before":     round(ssim_before, 4),
            "shift_ssim_after":      round(ssim_after, 4),
            "shift_ssim_delta":      round(ssim_after - ssim_before, 4),
            "shift_suspicious":      suspicious,
            "shift_score":           round(total, 2),
            "shift_score_dx":        round(s_dx, 2),
            "shift_score_dy":        round(s_dy, 2),
            "shift_bias_direction":  bias_dir,
            "shift_bias_ratio_x":    round(bx, 4),
            "shift_bias_ratio_y":    round(by, 4),
        }

    def get_shift_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        """从拼图重合边集合统计批量汇总出平移偏移性能均值和方差报告。

        Args:
            df (pd.DataFrame): 双图配准明细数据表。

        Returns:
            dict[str, Any]: 整体偏移统计数据字典。
        """
        valid = df.dropna(subset=["shift_dx", "shift_dy"])
        if valid.empty:
            return {}
        n_suspicious = int(valid["shift_suspicious"].sum()) if "shift_suspicious" in valid.columns else 0
        return {
            "total_pairs":          int(len(valid)),
            "shift_score_mean":     float(valid["shift_score"].mean()),
            "shift_dx_mean":        float(valid["shift_dx"].mean()),
            "shift_dy_mean":        float(valid["shift_dy"].mean()),
            "shift_dx_std":         ImageQualityDataUtils.safe_std(valid["shift_dx"]),
            "shift_dy_std":         ImageQualityDataUtils.safe_std(valid["shift_dy"]),
            "shift_suspicious_count": n_suspicious,
            "shift_suspicious_ratio": n_suspicious / len(valid) * 100,
        }

    # ==================================================================
    # 畸变计算（双图）
    # ==================================================================

    def distortion_calculate(
        self,
        gray_prev: np.ndarray,
        gray_curr: np.ndarray,
        overlap_length: int,
        stitch_direction: str,
        dx_prior: float = 0.0,
        dy_prior: float = 0.0,
    ) -> dict[str, Any]:
        """使用特征关键点提取（OpenCV SIFT）并进行物理扭曲模型分析，测量相邻两张图拼接时的畸变。

        所有结果键以 'dist_' 作为前缀。

        Args:
            gray_prev (np.ndarray): 头侧图片灰度数组。
            gray_curr (np.ndarray): 尾侧图片灰度数组。
            overlap_length (int): 拼区预估重复带宽。
            stitch_direction (str): 滑动方向匹配，"horizontal" 或 "vertical"。
            dx_prior (float): 平移分析传入的粗筛水平位移参数（辅助剔除无效特征点）。
            dy_prior (float): 平移分析传入的粗筛垂向位移参数（辅助剔除无效特征点）。

        Returns:
            dict[str, Any]: OpenCV 单应性评估提取出的两图间扭曲模型。包含旋转角与错切变量，或在匹配抛锚时反推缺失数据。
        """
        roi_a, roi_b = ImageQualityImageUtils.extract_roi(gray_prev, gray_curr, overlap_length, stitch_direction)
        empty = {**_DISTORTION_EMPTY}

        # 1. 提取 SIFT 特征
        sift = cv2.SIFT_create(nfeatures=2000)
        kp_a, des_a = sift.detectAndCompute(roi_a, None)
        kp_b, des_b = sift.detectAndCompute(roi_b, None)

        if des_a is None or des_b is None or len(kp_a) < 4 or len(kp_b) < 4:
            empty["dist_method"] = "N/A(特征点不足)"
            return empty

        # 2. KNN 匹配特征点，采用 Lowe's ratio test 剔除模糊匹配
        bf = cv2.BFMatcher(cv2.NORM_L2)
        good = [m for m, n in bf.knnMatch(des_a, des_b, k=2) if m.distance < 0.75 * n.distance]
        if len(good) < 4:
            empty["dist_method"] = f"N/A(匹配不足:{len(good)})"
            return empty

        # 3. 利用之前算出的位移平移量 (prior) 作为先验知识，进一步剔除不合理的匹配
        dist_thresh = 20.0
        filtered = [
            m for m in good
            if abs(kp_a[m.queryIdx].pt[0] - kp_b[m.trainIdx].pt[0] - dx_prior) <= dist_thresh
            and abs(kp_a[m.queryIdx].pt[1] - kp_b[m.trainIdx].pt[1] - dy_prior) <= dist_thresh
        ]
        if len(filtered) < 4:
            empty["dist_method"] = f"N/A(先验过滤不足:{len(filtered)})"
            return empty

        # 4. RANSAC 算法估算仿射变换矩阵
        src = np.float32([kp_a[m.queryIdx].pt for m in filtered]).reshape(-1, 1, 2)
        dst = np.float32([kp_b[m.trainIdx].pt for m in filtered]).reshape(-1, 1, 2)
        M, mask = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None:
            empty["dist_method"] = "N/A(RANSAC失败)"
            return empty

        inliers = int(mask.sum())
        # 5. 取仿射矩阵的线性变换部分(2x2)并利用 SVD 分解出旋转和拉伸/剪切成分
        A = M[:2, :2].astype(np.float64)
        U, _, Vt = np.linalg.svd(A)
        R = U @ Vt
        # 处理可能包含反射的情况
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        S = R.T @ A
        rotation_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        shear        = float(S[0, 1])
        scale_x      = float(S[0, 0])
        scale_y      = float(S[1, 1])

        # 畸变评分交由评分引擎
        dist_score, s_r, s_s = ImageQualityScorer.calc_distortion_pair_score(rotation_deg, shear)

        return {
            "dist_rotation_deg":   rotation_deg,
            "dist_shear":          shear,
            "dist_scale_x":        scale_x,
            "dist_scale_y":        scale_y,
            "dist_inliers":        inliers,
            "dist_method":         f"SIFT+RANSAC({inliers})",
            "dist_score":          dist_score,
            "dist_score_rotation": round(s_r, 2),
            "dist_score_shear":    round(s_s, 2),
        }

    def get_distortion_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        """从各块对的配准结果提取全批次异常特征指标。

        Args:
            df (pd.DataFrame): 包含双图 SIFT 参数值的明细 DataFrame。

        Returns:
            dict[str, Any]: 总体形变指标字典。
        """
        valid = df.dropna(subset=["dist_score"])
        n_total  = int(len(df.dropna(subset=["shift_dx"])))   # 参与双图计算的对数
        n_failed = n_total - int(len(valid))
        if valid.empty:
            return {"total_pairs": n_total, "n_failed": n_failed}
        return {
            "total_pairs":           n_total,
            "n_failed":              n_failed,
            "dist_score_mean":       float(valid["dist_score"].mean()),
            "dist_rotation_mean":    float(valid["dist_rotation_deg"].mean()),
            "dist_shear_mean":       float(valid["dist_shear"].mean()),
            "dist_score_std":        ImageQualityDataUtils.safe_std(valid["dist_score"]),
            "dist_failed_ratio":     n_failed / n_total * 100 if n_total > 0 else 0.0,
        }

