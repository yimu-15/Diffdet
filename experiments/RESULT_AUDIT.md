# DiffDet 实验结果核查报告（RESULT_AUDIT）

本文档把 **论文报告数字** 与 **本仓库可复现数字（原始日志/checkpoint）** 明确分开，
不修改、不合并、不编造任何数值。凡缺少客观证据的数字一律标注为
`PAPER-REPORTED / NOT VERIFIED`。

> 状态说明：
> - `REPRODUCED`：本仓库日志/checkpoint 可复算并一致。
> - `PAPER-REPORTED / NOT VERIFIED`：论文报告但当前材料无法复算，或数值不一致。
> - `BLOCKED`：缺少数据/权重/日志，无法验证。

---

## 1. 数据与评估协议

| 项 | 值 | 状态 |
|---|---|---|
| 数据集 | VOC-FOG（VOC2007 + 合成雾） | 已确认 |
| 验证集 | VOC2007 val，2510 张 | 已确认（日志 `n_pairs=500` 为残差分析子集；mAP 为 2510 张） |
| 类别数 | 20 | 已确认 |
| 输入分辨率 | 500×375 | 已确认（残差/特征阶段 resize） |
| 雾浓度 | light β=1.0 / moderate β=1.8 / dense β=3.0 | 已确认 |
| 评估指标 | VOC07 11-point mAP（`VOCMetric`, `eval_mode='11points'`） | 已确认 |

---

## 2. 主实验（VOC-FOG，val 2510）

论文主表：

| Method | Light | Moderate | Dense | Avg |
|---|---:|---:|---:|---:|
| Sparse R-CNN Baseline | 0.7625 | 0.6939 | 0.5543 | 0.6702 |
| DiffDet | 0.8001 | 0.7835 | 0.7169 | 0.7668 |

本仓库原始日志复算值（`experiments/logs/eval_*.json`）：

| Method | Light | Moderate | Dense | 状态 |
|---|---:|---:|---:|---|
| Baseline | 0.76248 | 0.69387 | 0.55431 | REPRODUCED |
| DiffDet (final INT8) | 0.80006 | 0.78346 | 0.71687 | REPRODUCED |

**结论：主实验表格与原始日志完全一致（四舍五入到 4 位小数）。**
Clean upper bound = 0.8074（日志 `compare_table_literature.txt`：
"Sparse R-CNN baseline VOC2007 val (clean) mAP 80.74"）。

**独立复跑（2026-11-05，用本仓库 `tools/evaluate.py` + 现有 checkpoint）：**

| 模型 | 独立复跑 mAP | 论文 | 结论 |
|---|---:|---:|---|
| DiffDet final INT8 | **0.7835**（AP50 0.7830，2510 张） | 0.7835 | REPRODUCED（完全一致） |
| Baseline (epoch_7.pth) | **0.6935** | 0.6939 | REPRODUCED（Δ=0.0004，评估噪声） |

---

## 3. 消融实验（moderate fog）与 0.7835 vs 0.8342 核查 ★

论文消融表（moderate）：

| Config | mAP |
|---|---:|
| Baseline | 0.6939 |
| + Frozen U-Net denoiser | 0.7079 |
| + Joint fine-tuning | 0.8369 |
| + Distilled student U-Net | 0.8342 |
| + INT8 quantization | 0.8342 |

本仓库可复算值（原始日志）：

| Config | mAP | 日志来源 | 状态 |
|---|---:|---|---|
| Baseline | 0.6939 | `eval_baseline_moderate.json` | REPRODUCED |
| + Frozen U-Net（学生 c128_b32） | 0.6994 | `eval_ablation_frozen_fp32.json` / `metrics_ablation_frozen_fp32.txt` | 与论文 0.7079 **不一致** |
| + Frozen U-Net（教师 b128） | 0.7132 | `work_dirs/phase1/metrics_det_mAP_b128.txt` | 论文未单列此值 |
| + Joint fine-tuning | 0.7833 | `eval_joint_finetune_ep3.json` / `work_dirs/phase4/metrics_diffdet_finetune_ep3.txt` | 与论文 0.8369 **不一致** |
| + Distilled student（冻结检测器） | 0.7036 | `eval_distilled_student.json` / `work_dirs/phase1/metrics_det_mAP_c128_b32.txt` | 与论文 0.8342 **不一致** |
| + INT8 quantization | 0.7835 | `eval_ablation_ft_int8w.json` / `metrics_ablation_ft_int8w.txt` | 与论文 0.8342 **不一致** |

### 3.1 核心问题：0.7835 与 0.8342

- **主实验** DiffDet moderate = **0.7835**（REPRODUCED）。
- **消融表** "+ INT8 quantization" = **0.8342**（PAPER-REPORTED），与主实验的 0.7835 相差
  **+0.0507（+5.1 点）**，且本仓库任何日志/checkpoint 都复算不出 0.8342。

### 3.2 已核查的差异来源（均不足以解释差异）

| 假设 | 核查结果 |
|---|---|
| 数据子集不同（val vs trainval） | 复算日志均为 VOC-FOG moderate **val 2510**，与主实验一致。消融 0.8342 无对应日志 |
| checkpoint 不同 | 消融日志使用的 checkpoint：frozen→`epoch_7.pth`；ft→`finetune_diffdet/epoch_3.pth`；INT8→`unet_finetuned_c128_b32.pt`。复算均为 0.78 量级 |
| 评估脚本/mAP 定义不同 | 均使用 `VOCMetric` `eval_mode='11points'`，同一评估协议 |
| 训练阶段不同 | 联合微调 `epoch_3` 复算 0.7833；未见更高 epoch 的 checkpoint |

### 3.3 结论

- 论文消融表中的 **0.8369 / 0.8342 / 0.8342** 三项在当前代码、日志与 checkpoint 中
  **均无法复算**，标记为 `PAPER-REPORTED / NOT VERIFIED`。
- 可复现的最终结果（与主实验一致）为 **0.7835**。
- 论文消融表中 "+ Frozen U-Net denoiser 0.7079" 介于教师（0.7132）与学生（0.6994/0.7036）
  之间，无法与唯一 checkpoint 精确对应，同样标记 `NOT VERIFIED`（量级一致但数值无证据）。

**在找到客观证据前，本仓库不修改、不合并、不解释这组数值。**

---

## 4. 效率（RTX 4060 Laptop，`benchmark_results.txt`）

| Method | FPS | 峰值显存 | 参数量 | FLOPs |
|---|---:|---:|---:|---:|
| Baseline | 17.7 | 624 MB | 106.25 M | 105.23 G |
| DiffDet (Ours) | 18.5 | 627 MB | 107.04 M | 105.61 G |

- 论文报告 Baseline 17.7 FPS / 624 MB，DiffDet 18.5 FPS / 627 MB → 与日志一致。
- 注意：INT8 为 **weight-only W8A32 模拟**，不声称 INT8 算子加速；FPS 提升（+0.8）属噪声范围。

---

## 5. 蒸馏

| 项 | 论文 | 实测（代码/checkpoint） | 状态 |
|---|---|---:|---|
| Teacher 参数 | ~4.58 M | `MinimalUNet(base=128)` = **4,611,328（4.611 M）**；`work_dirs/phase1/unet_minimal_b128.pt` 全量张量 4,614,664 | 量级一致，**数值差 +0.7%**（统计口径/版本差异，PAPER-REPORTED） |
| Student 参数 | ~0.791 M | **791,360（0.791 M）**（`StudentUNet(dim=128, base=32)`） | REPRODUCED（精确一致） |
| 比例 | 17.2% | 791,360 / 4,611,328 = **17.16%** | REPRODUCED（≈） |
| λ | 1.0 | `configs/distillation/distill_student_c128_b32.py` lam=1.0 | REPRODUCED |
| 学生 epoch | 50 | `train_log_student_c128_b32.txt` 50 epoch | REPRODUCED |
| 学生 val MSE | ~0.0455 | epoch50 val_mse=0.045574 | REPRODUCED |

---

## 6. 残差统计（`residual_stats.txt`）

| 项 | 论文 | 日志 | 状态 |
|---|---|---:|---|
| skewness | -10.24 | -10.24 | REPRODUCED |
| excess kurtosis | 385.76 | 385.8 | REPRODUCED（≈） |
| mean | ~0（近零） | -0.01381 | 一致（近零，不代表严格加性噪声） |

> 说明：近零均值不等于严格证明加性噪声假设；分析重点应放在**非高斯性与重尾**上
> （skewness=-10.24、excess kurtosis≈385 明确表明重尾非高斯分布）。

---

## 7. 未解决的问题

1. **消融表 0.8369/0.8342/0.8342 无法复算**（核心，见 §3）。可能原因：论文使用了
   未归档的 checkpoint/训练配置/更长微调轮次，或消融表数值有误。**无证据，不臆断。**
2. 论文全文（PDF/LaTeX）不在工作区，无法逐字核对论文措辞；本报告以任务描述与日志为准。
3. 预训练权重（VOC-FOG 训练出的 `epoch_7.pth` 等）体积大且未获授权上传，README 仅描述
   获取方式，不提供下载链接。

---

## 8. 文件清单

- `experiments/logs/`：原始评估 JSON/TXT 日志（mAP、残差、蒸馏、效率）。
- `experiments/configs/`：归档的原始配置。
- `experiments/reported_results/`：论文报告数字（见本目录 README/JSON）。
- 权重：`release/models/sparse_rcnn_voc_rainfog_final_int8.pth`、`unet_finetuned_c128_b32.pt`
  （体积大，默认不纳入 Git，见 `.gitignore`）。
