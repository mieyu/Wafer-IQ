# run_batch.py
# 晶圆拼接质量批量评估脚本 - 专用于平移偏移分析
# 用法: conda activate wafer_overlap && python run_batch.py
# 输出: batch_results/ 目录下的 CSV、热力图、仪表盘

import struct, time, csv
import os
from pathlib import Path
import cv2, numpy as np, yaml
from main import evaluate_shift_quality

BASE_DIR   = Path(__file__).parent
WORKSPACE_DIR = BASE_DIR.parent.parent
DATA_ROOT_DIR = Path(os.getenv("WAFER_DATA_ROOT", WORKSPACE_DIR / "data")).expanduser().resolve()
OUTPUT_ROOT_DIR = Path(os.getenv("WAFER_OUTPUT_ROOT", WORKSPACE_DIR / "output")).expanduser().resolve()

def hex_to_double(hex_str: str) -> float:
    return struct.unpack(">d", bytes.fromhex(hex_str))[0]

def _save_batch_visualizations(pairs, results, output_dir):
    import matplotlib
    matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec, matplotlib.patches as mpatches

    if not results: return
    n = len(results)
    offset_scores = [r["offset_score"] for r in results]
    suspicious_flags = [r["phase_corr_suspicious"] for r in results]

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

    grid_offset = np.full((grid_rows, grid_cols), np.nan)
    grid_suspicious = np.zeros((grid_rows, grid_cols), dtype=bool)

    for (coord_a, coord_b, _, _), res in zip(pairs, results):
        ci, ri = col_idx[coord_a[0]], row_idx[coord_a[1]]
        if ci < grid_cols: grid_offset[ri, ci], grid_suspicious[ri, ci] = res["offset_score"], res["phase_corr_suspicious"]

    fig_h, fig_w = max(6, grid_rows * 0.35 + 2), max(10, grid_cols * 0.35 + 2)
    step_c, step_r = max(1, len(col_vals) // 20), max(1, len(row_vals) // 20)

    grid_dx = np.full((grid_rows, grid_cols), np.nan)
    grid_dy = np.full((grid_rows, grid_cols), np.nan)
    for (coord_a, coord_b, _, _), res in zip(pairs, results):
        ci, ri = col_idx[coord_a[0]], row_idx[coord_a[1]]
        if ci < grid_cols: grid_dx[ri, ci], grid_dy[ri, ci] = res["dx"], res["dy"]

    n_susp = int(grid_suspicious.sum())

    def _draw_offset_heatmap(fig, ax, grid_val, title, unit="px"):
        valid = grid_val[~np.isnan(grid_val)]
        vmax = max(np.abs(valid).max(), 0.1) if len(valid) else 1.0
        im = ax.imshow(grid_val, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(unit, color="white", fontsize=9); cbar.ax.tick_params(colors="white")

        if grid_rows * grid_cols <= 2000:
            for ri in range(grid_rows):
                for ci in range(grid_cols):
                    if not np.isnan(grid_val[ri, ci]):
                        ax.text(ci, ri, f"{grid_val[ri, ci]:+.2f}", ha="center", va="center", fontsize=6, color="black", fontweight="bold")
                    if grid_suspicious[ri, ci]:
                        ax.add_patch(plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1, linewidth=2, edgecolor="#9b59b6", facecolor="none"))

        ax.set_xticks(range(0, grid_cols, step_c)); ax.set_xticklabels([str(col_vals[i]) for i in range(0, grid_cols, step_c)], color="white", fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(0, grid_rows, step_r)); ax.set_yticklabels([str(row_vals[i]) for i in range(0, grid_rows, step_r)], color="white", fontsize=7)
        ax.set_xlabel("Col", color="white", fontsize=9); ax.set_ylabel("Row", color="white", fontsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor("#0f3460")
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)

    dx_vals = [r["dx"] for r in results]
    fig1, ax = plt.subplots(figsize=(fig_w, fig_h)); fig1.patch.set_facecolor(BG); ax.set_facecolor(PANEL)
    _draw_offset_heatmap(fig1, ax, grid_dx, f"X 方向偏移量 (Δx)  |  共 {n} 对  |  均值 {np.mean(dx_vals):+.3f}px")
    plt.tight_layout(); fig1.savefig(output_dir / "晶圆全景图_X方向偏移.png", dpi=150, bbox_inches="tight", facecolor=BG); plt.close(fig1)

    dy_vals = [r["dy"] for r in results]
    fig2, ax = plt.subplots(figsize=(fig_w, fig_h)); fig2.patch.set_facecolor(BG); ax.set_facecolor(PANEL)
    _draw_offset_heatmap(fig2, ax, grid_dy, f"Y 方向偏移量 (Δy)  |  共 {n} 对  |  均值 {np.mean(dy_vals):+.3f}px")
    plt.tight_layout(); fig2.savefig(output_dir / "晶圆全景图_Y方向偏移.png", dpi=150, bbox_inches="tight", facecolor=BG); plt.close(fig2)

    fig3 = plt.figure(figsize=(15, 11)); fig3.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(2, 3, figure=fig3, hspace=0.45, wspace=0.35, left=0.05, right=0.97, top=0.90, bottom=0.07)

    ax1 = fig3.add_subplot(gs[0, 0:2])
    bins = np.arange(0, 105, 5)
    counts, edges = np.histogram(offset_scores, bins=bins)
    ax1.bar(edges[:-1], counts, width=4.5, color=[score_color(e + 2.5) for e in edges[:-1]], align="edge", edgecolor=BG)
    ax1.axvline(np.mean(offset_scores), color="white", lw=1.5, ls="--", label=f"均值 {np.mean(offset_scores):.1f}")
    ax1.legend(fontsize=8, facecolor=PANEL, labelcolor="white"); style(ax1, "偏移评分分布")

    ax3 = fig3.add_subplot(gs[0, 2])
    bias_x_dom, bias_y_dom = sum(1 for r in results if r["bias_ratio_x"] >= 0.70), sum(1 for r in results if r["bias_ratio_y"] >= 0.70)
    bias_mix = n - bias_x_dom - bias_y_dom
    non_zero = [(s, l, c) for s, l, c in zip([bias_x_dom, bias_y_dom, bias_mix], [f"X轴\n({bias_x_dom})", f"Y轴\n({bias_y_dom})", f"混合\n({bias_mix})"], ["#3498db", "#e74c3c", "#f39c12"]) if s > 0]
    if non_zero:
        sz, lb, co = zip(*non_zero)
        ax3.pie(sz, labels=lb, colors=co, autopct="%1.1f%%", startangle=90, textprops={"color": "white", "fontsize": 8}, wedgeprops={"edgecolor": BG, "linewidth": 1.5})
    style(ax3, "主导方向分布")

    ax5 = fig3.add_subplot(gs[1, 0])
    ax5.axis("off"); ax5.set_facecolor(PANEL)
    kpis = [("图块对总数", f"{n}"), ("平均偏移评分", f"{np.mean(offset_scores):.2f}"), ("SSIM异常", f"{n_susp} 对 ({n_susp/n*100:.1f}%)")]
    for idx, (label, value) in enumerate(kpis):
        y = 0.8 - idx * 0.25
        ax5.add_patch(mpatches.FancyBboxPatch((0.05, y), 0.9, 0.15, boxstyle="round,pad=0.01", facecolor="#1e2a4a", transform=ax5.transAxes, edgecolor="none"))
        ax5.text(0.1, y + 0.05, label, color="#aaa", fontsize=9, transform=ax5.transAxes)
        ax5.text(0.5, y + 0.05, value, color="white", fontsize=10, transform=ax5.transAxes, fontweight="bold")
    style(ax5, "核心指标")

    ax6 = fig3.add_subplot(gs[1, 1:3]); ax6.axis("off"); ax6.set_facecolor(PANEL)
    ax6.text(0.02, 0.95, "最差 10 对图块", color="#aaa", fontsize=10, fontweight="bold", transform=ax6.transAxes)
    worst10 = sorted(zip(pairs, results), key=lambda x: x[1]["offset_score"])[:10]
    cols = ["图块A", "图块B", "评分", "Δx", "Δy", "⚠SSIM"]
    cx = [0.05, 0.25, 0.45, 0.60, 0.75, 0.90]
    for c, x in zip(cols, cx): ax6.text(x, 0.85, c, color="#888", fontsize=9, transform=ax6.transAxes)
    for wi, ((_, _, pa, pb), res) in enumerate(worst10):
        y = 0.75 - wi * 0.075
        for ci, val in enumerate([pa.name[:10], pb.name[:10], f"{res['offset_score']:.1f}", f"{res['dx']:.2f}", f"{res['dy']:.2f}", "⚠" if res["phase_corr_suspicious"] else "-"]):
            ax6.text(cx[ci], y, val, color=score_color(res["offset_score"]) if ci == 2 else "#9b59b6" if ci == 5 and val == "⚠" else "white", fontsize=8, transform=ax6.transAxes)
    style(ax6, "")

    fig3.suptitle(f"批量偏移评估面板 │ 共 {n} 对 │ 平均分 {np.mean(offset_scores):.1f}", color="white", fontsize=13, fontweight="bold", y=0.97)
    fig3.savefig(output_dir / "晶圆全景图_评分总览.png", dpi=150, bbox_inches="tight", facecolor=BG); plt.close(fig3)

def _process_dataset(dataset_dir: Path) -> None:
    dataset_name = dataset_dir.name
    yaml_file = dataset_dir / "placements-BF.yml"
    output_dir = OUTPUT_ROOT_DIR / f"{dataset_name}_输出" / "位置偏移检测"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not yaml_file.exists():
        print(f"[{dataset_name}] 配置文件不存在: {yaml_file}")
        return
    with open(yaml_file, "r") as f: meta = yaml.safe_load(f)
    px_size = hex_to_double(meta["image_meta"]["pixel_equivalents"].split(",")[0])
    overlap_px = max(1, int(round(hex_to_double(meta["view_meta"]["overlap_width"]) / px_size)))

    views = {tuple(int(i) for i in v["index"].split(",")): dataset_dir / v["filename"] for v in meta["views"]}
    coords = sorted(views.keys())
    pairs = []
    for c, r in coords:
        if (c + 1, r) in views and views[(c, r)].exists() and views[(c + 1, r)].exists():
            pairs.append(((c, r), (c + 1, r), views[(c, r)], views[(c + 1, r)]))

    if not pairs:
        print(f"[{dataset_name}] 无匹配的相邻图块")
        return

    csv_file, txt_file = output_dir / "明细数据.csv", output_dir / "评分报告.txt"
    headers = ["序号", "图块A", "图块B", "Col_A", "Row_A", "Col_B", "Row_B", "Δx", "Δy", "SSIM提升", "评分", "X子分", "Y子分", "异常"]
    results, t_start = [], time.time()

    with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i, (ca, cb, pa, pb) in enumerate(pairs, 1):
            ia, ib = cv2.imread(str(pa), 0), cv2.imread(str(pb), 0)
            if ia is None or ib is None: continue
            r = evaluate_shift_quality(ia, ib, overlap_px, "horizontal")
            print(f"[{dataset_name}] [{i}/{len(pairs)}] {pa.name} ↔ {pb.name} | 分数: {r['offset_score']:.1f} | Δx: {r['dx']:.2f}")
            writer.writerow([i, pa.name, pb.name, ca[0], ca[1], cb[0], cb[1], r["dx"], r["dy"], r["ssim_delta"], r["offset_score"], r["score_dx"], r["score_dy"], "是" if r["phase_corr_suspicious"] else "否"])
            results.append(r)

    print(f"\n[{dataset_name}] 评估完成，耗时 {time.time() - t_start:.1f}s")
    _save_batch_visualizations(pairs, results, output_dir)


def main():
    if not DATA_ROOT_DIR.exists():
        print(f"data 目录不存在：{DATA_ROOT_DIR}")
        return
    dataset_dirs = sorted(path for path in DATA_ROOT_DIR.iterdir() if path.is_dir())
    if not dataset_dirs:
        print(f"未在 data 目录下找到数据集文件夹：{DATA_ROOT_DIR}")
        return
    for dataset_dir in dataset_dirs:
        _process_dataset(dataset_dir)

if __name__ == "__main__":
    main()
