# Wafer-IQ — 晶圆图像质量分析工具

对晶圆全景拼图执行四项自动化质量检测：**亮度 / 清晰度 / 位置偏移 / 畸变**，并按模块生成 CSV 明细、TXT 评分报告和暗色系可视化热力图。

---

## 快速开始

```bash
cd image_quality_utils
python main.py <输入父目录> <输出父目录>
```

**示例：**

```bash
python main.py ../data ../output
```

程序会自动扫描 `data/` 下的所有子文件夹，每个子文件夹视为一个独立晶圆数据集，结果按名称对应输出。

---

## 输入目录结构

输入父目录下，每个**一级子文件夹**为一个待分析的晶圆数据集，文件夹内需包含：

- 晶圆全景拼图图片（`.tif` / `.png` / `.jpg` 等）
- `placements-BF.yml`（坐标配置文件，记录每张图的拼接坐标与重叠区信息）

```
data/
├── BF_2_Wafer/
│   ├── placements-BF.yml
│   ├── image_000.tif
│   ├── image_001.tif
│   └── ...
├── BF_3_Wafer/
│   ├── placements-BF.yml
│   └── ...
└── ...
```

---

## 输出目录结构

输出结果与输入子文件夹**一一对应**，自动按名称映射：

```
output/
├── BF_2_Wafer/
│   ├── 亮度检测/
│   │   ├── 明细数据.csv
│   │   ├── 评分报告.txt
│   │   ├── 评分指标.json
│   │   └── 热力图.png
│   ├── 清晰度检测/
│   │   ├── 明细数据.csv
│   │   ├── 评分报告.txt
│   │   ├── 热力图_Laplacian 方差.png
│   │   ├── 热力图_FFT 高频能量比.png
│   │   └── ...
│   ├── 位置偏移检测/
│   │   ├── 明细数据.csv
│   │   ├── 评分报告.txt
│   │   ├── 热力图_X方向偏移量(Δx).png
│   │   └── 热力图_Y方向偏移量(Δy).png
│   └── 畸变检测/
│       ├── 明细数据.csv
│       ├── 评分报告.txt
│       └── 热力图.png
└── BF_3_Wafer/
    └── ...
```

---

## 模块架构

```
image_quality_utils/
├── main.py                          # 命令行入口（argparse + 批量循环）
└── src/
    ├── image_quality_calculate.py   # 核心调度器：亮度、清晰度、偏移、畸变计算
    ├── image_quality_scorer.py      # 评分引擎：将物理指标映射为 0~100 分
    ├── image_quality_reporter.py    # 报告输出：CSV / JSON / TXT 生成
    ├── image_quality_plot_utils.py  # 绘图工具：Matplotlib 暗色热力图、直方图
    ├── image_quality_data_utils.py  # 数据工具：YAML 解析、坐标索引、DataFrame 操作
    └── image_quality_image_utils.py # 图像工具：ROI 提取、相位相关、SSIM、清晰度算法
```

---

## 关键参数

在 `main.py` 顶部可调整以下配置：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `YAML_FILENAME` | `placements-BF.yml` | 坐标配置文件名 |
| `STITCH_DIRECTION` | `horizontal` | 拼接方向（`horizontal` / `vertical`） |
| `NUM_WORKERS` | `8` | 多线程并发数，可根据 CPU 核心数调整 |

---

## 环境依赖

```bash
pip install opencv-python numpy pandas matplotlib seaborn scikit-image tqdm pyyaml
```
