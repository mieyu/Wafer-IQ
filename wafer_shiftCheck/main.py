# main.py
# 晶圆拼接质量评估 - 平移偏移过滤
# 用法: python main.py
# 随机选取一对相邻图块进行偏移解析，输出: shift_eval_dashboard.png

import cv2
import numpy as np
from pathlib import Path
import struct
import yaml
import random

try:
    from skimage.metrics import structural_similarity as sk_ssim
    _USE_SK_SSIM = True
except ImportError:
    _USE_SK_SSIM = False

# ═══════════════════════════════════════════════════════════════════════════════
# 模块一：ROI 提取
# ═══════════════════════════════════════════════════════════════════════════════
def extract_roi(image_a: np.ndarray, image_b: np.ndarray, overlap_length: int, stitch_direction: str) -> tuple[np.ndarray, np.ndarray]:
    direction = stitch_direction.lower()
    if direction == "horizontal":
        roi_a = image_a[:, -overlap_length:]
        roi_b = image_b[:, :overlap_length]
    elif direction == "vertical":
        roi_a = image_a[-overlap_length:, :]
        roi_b = image_b[:overlap_length, :]
    else:
        raise ValueError("stitch_direction 必须是 'horizontal' 或 'vertical'")
    min_h = min(roi_a.shape[0], roi_b.shape[0])
    min_w = min(roi_a.shape[1], roi_b.shape[1])
    return roi_a[:min_h, :min_w], roi_b[:min_h, :min_w]

# ═══════════════════════════════════════════════════════════════════════════════
# 模块二：平移偏移量计算
# ═══════════════════════════════════════════════════════════════════════════════
def _phase_correlation_offset(roi_a: np.ndarray, roi_b: np.ndarray) -> tuple[float, float, float]:
    fa = roi_a.astype(np.float32)
    fb = roi_b.astype(np.float32)
    win = cv2.createHanningWindow(fa.shape[::-1], cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(fa, fb, win)
    return float(dx), float(dy), float(response)

def compute_offset(roi_a: np.ndarray, roi_b: np.ndarray) -> dict:
    dx_pc, dy_pc, resp = _phase_correlation_offset(roi_a, roi_b)
    return {"dx": dx_pc, "dy": dy_pc, "method": f"PhaseCorrelation(response={resp:.4f})", "phase_response": resp}

# ═══════════════════════════════════════════════════════════════════════════════
# 模块四：SSIM 验证
# ═══════════════════════════════════════════════════════════════════════════════
def _calc_ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    if _USE_SK_SSIM:
        dr = float(img_a.max() - img_a.min())
        dr = 255.0 if dr < 1.0 else dr
        return float(sk_ssim(img_a, img_b, data_range=dr))
    else:
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        a, b = img_a.astype(np.float64), img_b.astype(np.float64)
        mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
        mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
        ssim_map = ((2*mu_a*mu_b + c1) * (2*(cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a*mu_b) + c2)) / \
                   ((mu_a**2 + mu_b**2 + c1) * ((cv2.GaussianBlur(a**2, (11, 11), 1.5) - mu_a**2) + (cv2.GaussianBlur(b**2, (11, 11), 1.5) - mu_b**2) + c2))
        return float(ssim_map.mean())

def compute_ssim_validation(roi_a: np.ndarray, roi_b: np.ndarray, dx: float, dy: float) -> dict:
    h, w = roi_a.shape[:2]
    ssim_before = _calc_ssim(roi_a, roi_b)
    T = np.float32([[1, 0, -dx], [0, 1, -dy]])
    roi_b_aligned = cv2.warpAffine(roi_b, T, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mx, my = int(abs(dx)) + 1, int(abs(dy)) + 1
    x1, x2 = mx, w - mx
    y1, y2 = my, h - my
    crop_a = roi_a if (x2 <= x1 or y2 <= y1) else roi_a[y1:y2, x1:x2]
    crop_b = roi_b_aligned if (x2 <= x1 or y2 <= y1) else roi_b_aligned[y1:y2, x1:x2]
    ssim_after = _calc_ssim(crop_a, crop_b)
    return {"ssim_before": round(ssim_before, 4), "ssim_after": round(ssim_after, 4),
            "ssim_delta": round(ssim_after - ssim_before, 4), "phase_corr_suspicious": (ssim_after - ssim_before) < 0,
            "roi_b_aligned": roi_b_aligned}

# ═══════════════════════════════════════════════════════════════════════════════
# 模块五：偏移评分
# ═══════════════════════════════════════════════════════════════════════════════
def compute_offset_score(dx: float, dy: float, phase_response: float) -> dict:
    def _score(val): return float(np.clip((abs(val) - 5.0) / (0.10 - 5.0) * 100.0, 0.0, 100.0))
    s_dx, s_dy = _score(dx), _score(dy)
    tot = (s_dx + s_dy) / 2.0
    if phase_response < 0.05: tot *= 0.85
    tot = round(float(np.clip(tot, 0.0, 100.0)), 2)
    s_dx, s_dy = round(s_dx, 2), round(s_dy, 2)
    adx, ady = abs(dx), abs(dy)
    t_o = adx + ady
    if t_o < 1e-6:
        bx, by, bdir = 0.5, 0.5, "无明显偏移"
    else:
        bx, by = adx / t_o, ady / t_o
        bdir = "主要偏向 X 轴（水平漂移）" if bx >= 0.7 else "主要偏向 Y 轴（垂直漂移）" if by >= 0.7 else f"X/Y 混合偏移（X占{bx*100:.0f}% Y占{by*100:.0f}%）"
    return {"offset_score": tot, "score_dx": s_dx, "score_dy": s_dy, "bias_ratio_x": bx, "bias_ratio_y": by, "bias_direction": bdir}

# ═══════════════════════════════════════════════════════════════════════════════
# 顶层集成
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_shift_quality(image_a: np.ndarray, image_b: np.ndarray, overlap_length: int, stitch_direction: str) -> dict:
    roi_a, roi_b = extract_roi(image_a, image_b, overlap_length, stitch_direction)
    off = compute_offset(roi_a, roi_b)
    ssim = compute_ssim_validation(roi_a, roi_b, off["dx"], off["dy"])
    score = compute_offset_score(off["dx"], off["dy"], off["phase_response"])
    return {**off, **ssim, **score, "_roi_a": roi_a, "_roi_b": roi_b}

def print_result(res: dict):
    print("─" * 55)
    print("  晶圆拼接质量评估 - 平移偏移 (单文件处理体验)")
    print("─" * 55)
    print(f"  Δx: {res['dx']:+.3f} px   Δy: {res['dy']:+.3f} px")
    print(f"  SSIM提升量: {res['ssim_delta']:+.4f}   {'⚠ 疑似失效' if res['phase_corr_suspicious'] else '正常'}")
    print(f"  ★ 偏移评分: {res['offset_score']:.2f} / 100  ({res['bias_direction']})")
    print("─" * 55)

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR.parent / "data" / "BF_2_Wafer"
    YAML_FILE = DATA_DIR / "placements-BF.yml"
    if not YAML_FILE.exists():
        print(f"找不到配置: {YAML_FILE}")
        exit()
    def hex_to_double(hx): return struct.unpack(">d", bytes.fromhex(hx))[0]
    with open(YAML_FILE, "r") as f: meta = yaml.safe_load(f)
    px_size = hex_to_double(meta["image_meta"]["pixel_equivalents"].split(",")[0])
    overlap_px = max(1, int(round(hex_to_double(meta["view_meta"]["overlap_width"]) / px_size)))
    
    views = {tuple(int(i) for i in v["index"].split(",")): DATA_DIR / v["filename"] for v in meta["views"]}
    coords = sorted(views.keys())
    pairs = []
    for (c, r) in coords:
        if (c+1, r) in views and views[(c, r)].exists() and views[(c+1, r)].exists():
            pairs.append((views[(c, r)], views[(c+1, r)]))
    
    if not pairs:
        print("无相邻图块！")
        exit()
        
    p_a, p_b = random.choice(pairs)
    print(f"随机抽取测试图块: {p_a.name} ↔ {p_b.name}")
    img_a, img_b = cv2.imread(str(p_a), 0), cv2.imread(str(p_b), 0)
    result = evaluate_shift_quality(img_a, img_b, overlap_px, "horizontal")
    print_result(result)
    
    import matplotlib, matplotlib.pyplot as plt, matplotlib.gridspec as gs, matplotlib.patches as patches
    matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    thumb_h = 800
    scale = thumb_h / img_a.shape[0]
    thumb_w = max(1, int(img_a.shape[1] * scale))
    clahe = cv2.createCLAHE(2.0, (8,8))
    ta, tb = clahe.apply(cv2.resize(img_a, (thumb_w, thumb_h))), clahe.apply(cv2.resize(img_b, (thumb_w, thumb_h)))
    stitch_preview = np.hstack([ta, np.full((thumb_h, 2), 200, dtype=np.uint8), tb])
    
    h_roi = result["_roi_a"].shape[0]
    cy = h_roi // 2
    r1, r2 = max(0, cy - 400), max(0, cy - 400) + min(800, h_roi)
    sa = clahe.apply(result["_roi_a"][r1:r2,:])
    sb = clahe.apply(result["_roi_b"][r1:r2,:])
    sbal = clahe.apply(result["roi_b_aligned"][r1:r2,:])
    
    fig = plt.figure(figsize=(18, 8), facecolor="#1a1a2e")
    grid = gs.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.3, left=0.04, right=0.97, top=0.88, bottom=0.08)
    def style(ax, t): ax.set_title(t, color="white", fontsize=10); ax.set_facecolor("#16213e"); ax.tick_params(colors="#666"); [sp.set_edgecolor("#0f3460") for sp in ax.spines.values()]
    def sc_color(s): return "#2ecc71" if s>=85 else "#3498db" if s>=70 else "#f39c12" if s>=50 else "#e74c3c"
    
    ax1 = fig.add_subplot(grid[0, 0:2]); ax1.imshow(stitch_preview, cmap="gray")
    style(ax1, f"完整全貌 (左:{p_a.name} 右:{p_b.name})")
    
    ax2 = fig.add_subplot(grid[0, 2]); ax2.imshow(np.hstack([sa, sb]), cmap="gray")
    ax2.axvline(sa.shape[1]-0.5, color="#f39c12", ls=":"); style(ax2, "ROI未对齐")
    
    ax3 = fig.add_subplot(grid[0, 3]); ax3.imshow(np.hstack([sa, sbal]), cmap="gray")
    ax3.axvline(sa.shape[1]-0.5, color="#2ecc71", ls=":"); style(ax3, f"ROI对齐后 (Δx={result['dx']:.2f}, Δy={result['dy']:.2f})")
    
    ax4 = fig.add_subplot(grid[1, 0]); diff = cv2.absdiff(sa, sbal); im4 = ax4.imshow(diff, cmap="inferno", vmin=0, vmax=max(diff.max(),1))
    plt.colorbar(im4, ax=ax4, fraction=0.05).ax.tick_params(colors="w"); style(ax4, "差异热图")
    
    ax5 = fig.add_subplot(grid[1, 1]); lim=max(abs(result['dx']), abs(result['dy']), 1)*1.5
    ax5.set_xlim(-lim, lim); ax5.set_ylim(-lim, lim); ax5.axhline(0, color="#0f3460"); ax5.axvline(0, color="#0f3460")
    ax5.annotate("", xy=(result['dx'], -result['dy']), xytext=(0,0), arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2))
    style(ax5, "平移矢量表")
    
    ax6 = fig.add_subplot(grid[1, 2]); ax6.axis("off"); ax6.text(0.5, 0.7, f"{result['offset_score']:.1f}", color=sc_color(result['offset_score']), fontsize=32, ha="center")
    ax6.text(0.5, 0.3, result['bias_direction'], color="white", ha="center"); style(ax6, "★ 评分")
    
    ax7 = fig.add_subplot(grid[1, 3]); ax7.axis("off"); ax7.text(0.5, 0.5, "SSIM 异常" if result['phase_corr_suspicious'] else "SSIM 正常", color="#e74c3c" if result['phase_corr_suspicious'] else "#2ecc71", fontsize=20, ha="center")
    style(ax7, "SSIM验证")
    
    fig.suptitle(f"平移偏移检测单文件结果 │ {p_a.name} ↔ {p_b.name}", color="white", fontsize=14, y=0.96)
    plt.savefig(BASE_DIR / "shift_eval_dashboard.png", facecolor=fig.get_facecolor(), dpi=150)
    print(f"输出已保存至 {BASE_DIR}/shift_eval_dashboard.png")
