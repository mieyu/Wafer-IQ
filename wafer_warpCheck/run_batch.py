# run_batch.py
# 晶圆拼接质量批量评估脚本 - 专用于畸变性分析
# 用法: conda activate wafer_overlap && python run_batch.py
# 输出: batch_results/ 目录下的 CSV、热力图、仪表盘

import struct, time, csv
from pathlib import Path
import cv2, numpy as np, yaml
from main import evaluate_warp_quality

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR.parent / "data" / "BF_1_Wafer"
YAML_FILE  = DATA_DIR / "placements-BF.yml"
OUTPUT_DIR = BASE_DIR / "batch_results"

def hex_to_double(hex_str: str) -> float:
    return struct.unpack(">d", bytes.fromhex(hex_str))[0]

def _save_batch_visualizations(pairs, results, output_dir):
    import matplotlib
    matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec, matplotlib.patches as mpatches

    if not results: return
    n = len(results)
    distortion_scores = [r["distortion_score"] for r in results if r["distortion_score"] is not None]
    n_failed = n - len(distortion_scores)

    BG, PANEL = "#1a1a2e", "#16213e"
    def score_color(s): return "#2ecc71" if s >= 85 else "#3498db" if s >= 70 else "#f39c12" if s >= 50 else "#e74c3c"
    def style(ax, title):
        ax.set_facecolor(PANEL); ax.set_title(title, color="white", fontsize=10, pad=6)
        ax.tick_params(colors="#aaa", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#0f3460")

    col_vals = sorted(set(p[0][0] for p in pairs) | set(p[1][0] for p in pairs))
    row_vals = sorted(set(p[0][1] for p in pairs) | set(p[1][1] for p in pairs))
    col_idx, row_idx = {c: i for i, c in enumerate(col_vals)}, {r: i for i, r in enumerate(row_vals)}
    grid_cols, grid_rows = max(len(col_vals) - 1, 1), len(row_vals)

    grid_distortion = np.full((grid_rows, grid_cols), np.nan)
    for (coord_a, coord_b, _, _), res in zip(pairs, results):
        ci, ri = col_idx[coord_a[0]], row_idx[coord_a[1]]
        if ci < grid_cols and res["distortion_score"] is not None:
            grid_distortion[ri, ci] = res["distortion_score"]

    fig_h, fig_w = max(6, grid_rows * 0.35 + 2), max(10, grid_cols * 0.35 + 2)
    step_c, step_r = max(1, len(col_vals) // 20), max(1, len(row_vals) // 20)

    # 绘制畸变得分热力图
    fig1, ax = plt.subplots(figsize=(fig_w, fig_h)); fig1.patch.set_facecolor(BG); ax.set_facecolor(PANEL)
    im = ax.imshow(grid_distortion, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto", interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors="white")

    if grid_rows * grid_cols <= 2000:
        for ri in range(grid_rows):
            for ci in range(grid_cols):
                if not np.isnan(grid_distortion[ri, ci]):
                    v = grid_distortion[ri, ci]
                    ax.text(ci, ri, f"{v:.0f}", ha="center", va="center", fontsize=6, color="black" if v > 50 else "white", fontweight="bold")

    ax.set_xticks(range(0, grid_cols, step_c)); ax.set_xticklabels([str(col_vals[i]) for i in range(0, grid_cols, step_c)], color="white", fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(0, grid_rows, step_r)); ax.set_yticklabels([str(row_vals[i]) for i in range(0, grid_rows, step_r)], color="white", fontsize=7)
    ax.set_xlabel("Col", color="white", fontsize=9); ax.set_ylabel("Row", color="white", fontsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#0f3460")
    mean_d = np.mean(distortion_scores) if distortion_scores else 0
    ax.set_title(f"畸变评分空间热力图  |  共 {len(distortion_scores)} 对有效  |  均值 {mean_d:.1f}", color="white", fontsize=11, fontweight="bold", pad=8)
    
    plt.tight_layout(); fig1.savefig(output_dir / "heatmap_distortion.png", dpi=150, bbox_inches="tight", facecolor=BG); plt.close(fig1)

    fig3 = plt.figure(figsize=(13, 11)); fig3.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(2, 2, figure=fig3, hspace=0.35, wspace=0.25, left=0.05, right=0.97, top=0.90, bottom=0.07)

    # 分布直方图
    ax1 = fig3.add_subplot(gs[0, 0])
    bins = np.arange(0, 105, 5)
    if distortion_scores:
        counts, edges = np.histogram(distortion_scores, bins=bins)
        ax1.bar(edges[:-1], counts, width=4.5, color=[score_color(e + 2.5) for e in edges[:-1]], align="edge", edgecolor=BG)
        ax1.axvline(np.mean(distortion_scores), color="white", lw=1.5, ls="--", label=f"均值 {np.mean(distortion_scores):.1f}")
        ax1.legend(fontsize=8, facecolor=PANEL, labelcolor="white"); style(ax1, "畸变评分分布")
    else:
        ax1.text(0.5, 0.5, "无可用的畸变评分", transform=ax1.transAxes, color="#aaa", ha="center")
        style(ax1, "畸变评分分布 (空)")

    # 失败饼图
    ax3 = fig3.add_subplot(gs[0, 1])
    n_valid = len(distortion_scores)
    if n > 0:
        ax3.pie([n_valid, n_failed], labels=[f"匹配成功 ({n_valid})", f"特征不足 ({n_failed})"], colors=["#2ecc71", "#e74c3c"], autopct="%1.1f%%", startangle=90, textprops={"color": "white", "fontsize": 10}, wedgeprops={"edgecolor": BG, "linewidth": 1.5})
    style(ax3, "SIFT 匹配成功率")

    # 核心指标
    ax5 = fig3.add_subplot(gs[1, 0])
    ax5.axis("off"); ax5.set_facecolor(PANEL)
    kpis = [("图块对总数", f"{n}"), ("平均畸变评分", f"{np.mean(distortion_scores):.1f}" if distortion_scores else "N/A"), ("失败率", f"{n_failed/n*100:.1f}%")]
    for idx, (label, value) in enumerate(kpis):
        y = 0.8 - idx * 0.25
        ax5.add_patch(mpatches.FancyBboxPatch((0.05, y), 0.9, 0.15, boxstyle="round,pad=0.01", facecolor="#1e2a4a", transform=ax5.transAxes, edgecolor="none"))
        ax5.text(0.1, y + 0.05, label, color="#aaa", fontsize=9, transform=ax5.transAxes)
        ax5.text(0.5, y + 0.05, value, color="white", fontsize=10, transform=ax5.transAxes, fontweight="bold")
    style(ax5, "核心指标")

    # 最差 10 对图块
    ax6 = fig3.add_subplot(gs[1, 1]); ax6.axis("off"); ax6.set_facecolor(PANEL)
    ax6.text(0.02, 0.95, "最差 10 对图块", color="#aaa", fontsize=10, fontweight="bold", transform=ax6.transAxes)
    worst10 = sorted([ (p, r) for p, r in zip(pairs, results) if r["distortion_score"] is not None ], key=lambda x: x[1]["distortion_score"])[:10]
    cols, cx = ["图块A", "图块B", "评分", "旋转角", "切变"], [0.05, 0.25, 0.45, 0.60, 0.85]
    for c, x in zip(cols, cx): ax6.text(x, 0.85, c, color="#888", fontsize=9, transform=ax6.transAxes)
    for wi, ((_, _, pa, pb), res) in enumerate(worst10):
        y = 0.75 - wi * 0.075
        for ci, val in enumerate([pa.name[:10], pb.name[:10], f"{res['distortion_score']:.1f}", f"{res['rotation_deg']:+.3f}°", f"{res['shear']:+.3f}"]):
            ax6.text(cx[ci], y, val, color=score_color(res["distortion_score"]) if ci == 2 else "white", fontsize=8, transform=ax6.transAxes)
    style(ax6, "")

    fig3.suptitle(f"批量畸变评估面板 │ 共 {n} 对 │ 平均分 {mean_d:.1f}", color="white", fontsize=13, fontweight="bold", y=0.97)
    fig3.savefig(output_dir / "score_dashboard.png", dpi=150, bbox_inches="tight", facecolor=BG); plt.close(fig3)

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    if not YAML_FILE.exists():
        print(f"配置文件不存在: {YAML_FILE}"); return
    with open(YAML_FILE, "r") as f: meta = yaml.safe_load(f)
    px_size = hex_to_double(meta["image_meta"]["pixel_equivalents"].split(",")[0])
    overlap_px = max(1, int(round(hex_to_double(meta["view_meta"]["overlap_width"]) / px_size)))

    views = {tuple(int(i) for i in v["index"].split(",")): DATA_DIR / v["filename"] for v in meta["views"]}
    coords = sorted(views.keys())
    pairs = [((c, r), (c + 1, r), views[(c, r)], views[(c + 1, r)]) for c, r in coords if (c + 1, r) in views and views[(c, r)].exists() and views[(c + 1, r)].exists()]

    if not pairs:
        print("无匹配的相邻图块"); return

    csv_file, txt_file = OUTPUT_DIR / "warp_report.csv", OUTPUT_DIR / "warp_summary.txt"
    headers = ["序号", "图块A", "图块B", "Col_A", "Row_A", "Col_B", "Row_B", "评分", "特征法", "匹配数", "旋转角度", "切变量", "X缩放", "Y缩放", "旋转得分", "切变得分"]
    results, t_start = [], time.time()

    with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i, (ca, cb, pa, pb) in enumerate(pairs, 1):
            ia, ib = cv2.imread(str(pa), 0), cv2.imread(str(pb), 0)
            if ia is None or ib is None: continue
            r = evaluate_warp_quality(ia, ib, overlap_px, "horizontal")
            d_str = f"{r['distortion_score']:.1f}" if r["distortion_score"] is not None else "N/A"
            print(f"[{i}/{len(pairs)}] {pa.name} ↔ {pb.name} | 评分: {d_str} | 旋转: {r['rotation_deg']:.3f}°")
            writer.writerow([i, pa.name, pb.name, ca[0], ca[1], cb[0], cb[1], d_str, r["distortion_method"], r["inliers"], r["rotation_deg"], r["shear"], r["scale_x"], r["scale_y"], r["score_rotation"] if r["score_rotation"] is not None else "N/A", r["score_shear"] if r["score_shear"] is not None else "N/A"])
            results.append(r)

    print(f"\n评估完成，耗时 {time.time() - t_start:.1f}s")
    _save_batch_visualizations(pairs, results, OUTPUT_DIR)

if __name__ == "__main__":
    main()
