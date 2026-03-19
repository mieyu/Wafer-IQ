# main.py
# 晶圆拼接质量评估 - 畸变分析
# 用法: python main.py
# 随机选取一对相邻图块进行畸变解析，输出: warp_eval_dashboard.png, sift_matches_debug.png

import cv2
import numpy as np
from pathlib import Path
import os
import struct
import yaml
import random

def extract_roi(image_a: np.ndarray, image_b: np.ndarray, overlap_length: int, stitch_direction: str) -> tuple[np.ndarray, np.ndarray]:
    direction = stitch_direction.lower()
    if direction == "horizontal": roi_a, roi_b = image_a[:, -overlap_length:], image_b[:, :overlap_length]
    elif direction == "vertical": roi_a, roi_b = image_a[-overlap_length:, :], image_b[:overlap_length, :]
    min_h, min_w = min(roi_a.shape[0], roi_b.shape[0]), min(roi_a.shape[1], roi_b.shape[1])
    return roi_a[:min_h, :min_w], roi_b[:min_h, :min_w]

def _phase_correlation_offset(roi_a: np.ndarray, roi_b: np.ndarray) -> tuple[float, float, float]:
    fa, fb = roi_a.astype(np.float32), roi_b.astype(np.float32)
    win = cv2.createHanningWindow(fa.shape[::-1], cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(fa, fb, win)
    return float(dx), float(dy), float(response)

def evaluate_distortion(roi_a: np.ndarray, roi_b: np.ndarray, dx_prior: float = 0.0, dy_prior: float = 0.0) -> dict:
    sift = cv2.SIFT_create(nfeatures=2000)
    kp_a, des_a = sift.detectAndCompute(roi_a, None)
    kp_b, des_b = sift.detectAndCompute(roi_b, None)

    default_result = {"rotation_deg": 0.0, "shear": 0.0, "scale_x": 1.0, "scale_y": 1.0, "affine_matrix": None, "inliers": 0, "distortion_method": "N/A(特征点不足)", "sift_match_img": cv2.cvtColor(np.hstack([roi_a, roi_b]), cv2.COLOR_GRAY2BGR)}
    if des_a is None or des_b is None or len(kp_a) < 4 or len(kp_b) < 4: return default_result

    bf = cv2.BFMatcher(cv2.NORM_L2)
    good = [m for m, n in bf.knnMatch(des_a, des_b, k=2) if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        default_result["sift_match_img"] = cv2.drawMatches(roi_a, kp_a, roi_b, kp_b, good, None, flags=2, matchColor=(0, 255, 0), singlePointColor=(255, 0, 0))
        default_result["distortion_method"] = f"N/A(匹配不足:{len(good)})"
        return default_result

    dist_thresh = 20.0
    filtered = []
    for m in good:
        match_dx = kp_a[m.queryIdx].pt[0] - kp_b[m.trainIdx].pt[0]
        match_dy = kp_a[m.queryIdx].pt[1] - kp_b[m.trainIdx].pt[1]
        if abs(match_dx - dx_prior) <= dist_thresh and abs(match_dy - dy_prior) <= dist_thresh: filtered.append(m)

    if len(filtered) < 4:
        default_result["sift_match_img"] = cv2.drawMatches(roi_a, kp_a, roi_b, kp_b, filtered, None, flags=2)
        default_result["distortion_method"] = f"N/A(先验过滤不足:{len(filtered)})"
        return default_result

    src = np.float32([kp_a[m.queryIdx].pt for m in filtered]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in filtered]).reshape(-1, 1, 2)
    M, mask = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        default_result["sift_match_img"] = cv2.drawMatches(roi_a, kp_a, roi_b, kp_b, sorted(filtered, key=lambda x: x.distance)[:30], None, flags=2)
        default_result["distortion_method"] = "N/A(RANSAC失败)"
        return default_result

    inliers = [m for i, m in enumerate(filtered) if mask.ravel()[i] == 1]
    match_img = cv2.drawMatches(roi_a, kp_a, roi_b, kp_b, inliers[:50], None, flags=2, matchColor=(0, 255, 0), singlePointColor=(255, 0, 0))
    A = M[:2, :2].astype(np.float64)
    U, sigma, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    S = R.T @ A
    return {"rotation_deg": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), "shear": float(S[0, 1]), "scale_x": float(S[0, 0]), "scale_y": float(S[1, 1]), "affine_matrix": M, "inliers": len(inliers), "distortion_method": f"SIFT+RANSAC({len(inliers)})", "sift_match_img": match_img}

def compute_distortion_score(rot: float, shr: float) -> dict:
    s_r = float(np.clip((abs(rot) - 1.0) / (0.01 - 1.0) * 100.0, 0.0, 100.0))
    s_s = float(np.clip((abs(shr) - 0.05) / (0.001 - 0.05) * 100.0, 0.0, 100.0))
    return {"distortion_score": round(s_r * 0.6 + s_s * 0.4, 2), "score_rotation": round(s_r, 2), "score_shear": round(s_s, 2)}

def evaluate_warp_quality(image_a: np.ndarray, image_b: np.ndarray, overlap_length: int, stitch_direction: str) -> dict:
    roi_a, roi_b = extract_roi(image_a, image_b, overlap_length, stitch_direction)
    dx_prior, dy_prior, _ = _phase_correlation_offset(roi_a, roi_b)
    dist = evaluate_distortion(roi_a, roi_b, dx_prior, dy_prior)
    score = {"distortion_score": None, "score_rotation": None, "score_shear": None} if "N/A" in dist["distortion_method"] else compute_distortion_score(dist["rotation_deg"], dist["shear"])
    return {**dist, **score, "_roi_a": roi_a, "_roi_b": roi_b}

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    WORKSPACE_DIR = BASE_DIR.parent.parent
    data_root_dir = Path(os.getenv("WAFER_DATA_ROOT", WORKSPACE_DIR / "data")).expanduser().resolve()
    output_root_dir = Path(os.getenv("WAFER_OUTPUT_ROOT", WORKSPACE_DIR / "output")).expanduser().resolve()
    if not data_root_dir.exists():
        print(f"data 目录不存在：{data_root_dir}")
        exit()
    dataset_dirs = sorted(path for path in data_root_dir.iterdir() if path.is_dir())
    if not dataset_dirs:
        print(f"未在 data 目录下找到数据集文件夹：{data_root_dir}")
        exit()
    def hex_to_double(hx): return struct.unpack(">d", bytes.fromhex(hx))[0]
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        output_dir = output_root_dir / f"{dataset_name}_输出" / "形变翘曲检测"
        output_dir.mkdir(parents=True, exist_ok=True)
        yaml_file = dataset_dir / "placements-BF.yml"
        if not yaml_file.exists():
            print(f"[{dataset_name}] 找不到配置: {yaml_file}")
            continue
        with open(yaml_file, "r") as f: meta = yaml.safe_load(f)
        px_size = hex_to_double(meta["image_meta"]["pixel_equivalents"].split(",")[0])
        overlap_px = max(1, int(round(hex_to_double(meta["view_meta"]["overlap_width"]) / px_size)))
        views = {tuple(int(i) for i in v["index"].split(",")): dataset_dir / v["filename"] for v in meta["views"]}
        pairs = [ (views[(c,r)], views[(c+1,r)]) for c,r in sorted(views.keys()) if (c+1, r) in views and views[(c,r)].exists() and views[(c+1,r)].exists() ]
        if not pairs:
            print(f"[{dataset_name}] 无相邻图块！")
            continue
        p_a, p_b = random.choice(pairs)
        print(f"[{dataset_name}] 随机抽取测试图块: {p_a.name} ↔ {p_b.name}")
        img_a, img_b = cv2.imread(str(p_a), 0), cv2.imread(str(p_b), 0)
        result = evaluate_warp_quality(img_a, img_b, overlap_px, "horizontal")
    
        print("─" * 55)
        print("  晶圆畸变性评估 (单文件处理体验)")
        print("─" * 55)
        print(f"  特征匹配方法 : {result['distortion_method']}")
        print(f"  内部点数     : {result['inliers']}")
        print(f"  旋转角度     : {result['rotation_deg']:+.4f} °")
        print(f"  切变量       : {result['shear']:+.4f}")
        print(f"  ★ 畸变评分  : {result['distortion_score'] if result['distortion_score'] else 'N/A'}")
        print("─" * 55)
    
        import matplotlib, matplotlib.pyplot as plt, matplotlib.gridspec as gs
        matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']; matplotlib.rcParams['axes.unicode_minus'] = False
        
        fig = plt.figure(figsize=(15, 6), facecolor="#1a1a2e")
        grid = gs.GridSpec(1, 3, figure=fig, hspace=0.4, wspace=0.3, left=0.04, right=0.97, top=0.88, bottom=0.08)
        def style(ax, t): ax.set_title(t, color="white", fontsize=10); ax.set_facecolor("#16213e"); ax.tick_params(colors="#666"); [sp.set_edgecolor("#0f3460") for sp in ax.spines.values()]
        def sc_color(s): return "#2ecc71" if s>=85 else "#3498db" if s>=70 else "#f39c12" if s>=50 else "#e74c3c"
        
        scale = 800 / img_a.shape[0]; thumb_w = max(1, int(img_a.shape[1] * scale))
        ta, tb = cv2.resize(img_a, (thumb_w, 800)), cv2.resize(img_b, (thumb_w, 800))
        stitch_preview = np.hstack([ta, np.full((800, 2), 200, dtype=np.uint8), tb])
        
        ax1 = fig.add_subplot(grid[0, 0]); ax1.imshow(stitch_preview, cmap="gray")
        style(ax1, f"完整全貌 (左:{p_a.name[:10]} 右:{p_b.name[:10]})")
        ax2 = fig.add_subplot(grid[0, 1]); ax2.imshow(cv2.cvtColor(result["sift_match_img"], cv2.COLOR_BGR2RGB))
        style(ax2, f"SIFT 匹配可视化 ({result['inliers']} inliers)"); ax2.axis('off')
        ax3 = fig.add_subplot(grid[0, 2]); ax3.axis("off")
        if result["distortion_score"]:
            ax3.text(0.5, 0.7, f"{result['distortion_score']:.1f}", color=sc_color(result['distortion_score']), fontsize=32, ha="center")
            ax3.text(0.5, 0.3, f"旋转角: {result['rotation_deg']:+.4f}°\n切变: {result['shear']:+.4f}", color="white", ha="center")
        else:
            ax3.text(0.5, 0.5, "N/A (特征提取失败)", color="#e74c3c", fontsize=24, ha="center")
        style(ax3, "★ 畸变评分")
        
        fig.suptitle(f"畸变检测单文件结果 │ {p_a.name} ↔ {p_b.name}", color="white", fontsize=14, y=0.96)
        output_dashboard = output_dir / "随机抽样对比示例.png"
        output_match = output_dir / "随机抽样特征匹配.png"
        plt.savefig(output_dashboard, facecolor=fig.get_facecolor(), dpi=150)
        cv2.imwrite(str(output_match), result["sift_match_img"])
        print(f"[{dataset_name}] 输出已保存至 {output_dashboard} 和 {output_match}")
