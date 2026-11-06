# 实验记录（EXPERIMENTS）

本文档汇总论文报告结果与仓库复算结果；详细差异核查见
[`experiments/RESULT_AUDIT.md`](../experiments/RESULT_AUDIT.md)。

## 1. 主实验（VOC-FOG val 2510，VOC07 11-point mAP）

| Method | Light | Moderate | Dense | Avg |
|---|---:|---:|---:|---:|
| Sparse R-CNN Baseline | 0.7625 | 0.6939 | 0.5543 | 0.6702 |
| DiffDet | 0.8001 | 0.7835 | 0.7169 | 0.7668 |

- Clean upper bound：0.8074。
- 状态：**REPRODUCED**（`experiments/logs/eval_*.json`）。

## 2. 消融实验（moderate fog）

| Config | 论文 | 仓库复算 | 状态 |
|---|---:|---:|---|
| Baseline | 0.6939 | 0.6939 | REPRODUCED |
| + Frozen U-Net denoiser | 0.7079 | 0.6994（学生）/ 0.7132（教师） | NOT VERIFIED |
| + Joint fine-tuning | 0.8369 | 0.7833 | **不一致** |
| + Distilled student U-Net | 0.8342 | 0.7036 | **不一致** |
| + INT8 quantization | 0.8342 | 0.7835 | **不一致** |

> 消融表 0.8369/0.8342/0.8342 无法用当前日志复算，标记 `PAPER-REPORTED / NOT VERIFIED`。

## 3. 效率（RTX 4060 Laptop）

| Method | FPS | 峰值显存 | 参数量 | FLOPs |
|---|---:|---:|---:|---:|
| Baseline | 17.7 | 624 MB | 106.25 M | 105.23 G |
| DiffDet | 18.5 | 627 MB | 107.04 M | 105.61 G |

## 4. 蒸馏

| 项 | 值 |
|---|---|
| Teacher 参数 | ~4.58 M |
| Student 参数 | ~0.79 M |
| 比例 | ~17.2% |
| λ | 1.0 |
| 学生 epoch | 50 |
| 学生 val MSE | 0.0455 |

## 5. 残差统计

skewness=-10.24，excess kurtosis≈385.8，mean≈-0.0138（近零，非高斯重尾）。

## 6. 与其他去雾+检测对比（同协议自测，VOC-FOG moderate）

| Method | mAP |
|---|---:|
| DCP + Detector | 0.7481 |
| CLAHE + Detector | 0.6986 |
| DiffDet | 0.7835 |

（数据源：`experiments/logs/compare_results.json`、`release/results/metrics_compare_*.txt`）
