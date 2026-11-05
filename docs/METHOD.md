# Method：DiffDet — 轻量特征空间去噪的雾天目标检测

## 1. 定位与命名

DiffDet 的核心是 **diffusion-inspired 的 *单次前向* 特征去噪**（single-pass feature
denoising）。它借鉴扩散模型的"从噪声恢复信号"的思想，但**不是**迭代式扩散概率模型：

- **不包含** DDPM 的时间步、噪声调度表（noise schedule）与迭代反向采样。
- 只是用一个小型 U-Net 在骨干网络 C5 特征上预测一个加法修正量，与原始特征做残差融合。

本文档与代码注释严格区分 `diffusion-inspired` 与 `iterative diffusion`。

## 2. 整体框架

DiffDet 以 **Sparse R-CNN**（ResNet-50 + FPN）为检测框架，在 ResNet-50 的最高层
**C5（2048 通道）** 处插入一个轻量残差特征去噪模块。

```
输入雾图
   │
   v
ResNet-50 骨干  ──>  C5 (2048, H/32, W/32)
   │                     │
   │                     v
   │          ┌──────────────────────────┐
   │          │  轻量 U-Net 去噪模块      │
   │          │  x' = x + U(x)           │  (残差连接)
   │          └──────────────────────────┘
   │                     │  x' (修复后的 C5)
   │                     v
   │               FPN (C2~C5)
   │                     │
   │                     v
   └────────────────> Sparse R-CNN 检测头
```

### 关键点：C5 与 FPN 的连接位置

- 去噪模块位于 **backbone 输出的 C5** 与 **FPN 输入** 之间。
- 只修复 C5，C2/C3/C4 原样进入 FPN。
- 修复后特征形状与原始 C5 完全一致（2048 通道），因此与后续 FPN/Sparse R-CNN
  结构完全兼容，不改变检测器任何输出维度。

对应实现见 `diffdet/models/diffdet_detector.py` 的 `extract_feat`：

```python
def extract_feat(self, batch_inputs):
    x = list(self.backbone(batch_inputs))  # [C2, C3, C4, C5]
    x[-1] = denoise_features(self.unet, x[-1])  # 只修复 C5
    if self.with_neck:
        x = self.neck(x)
    return x
```

## 3. 特征级去噪模块

### 3.1 结构

- 输入：C5 特征，2048 通道。
- 1×1 通道降维：2048 → 128（学生）/ 128（教师核心）。
- 两级轻量 U-Net（base 通道：教师 128，学生 32）。
- 1×1 通道还原：128 → 2048。
- 残差连接：`x' = x + U(x)`。

### 3.2 教师 / 学生

| 模型 | 结构 | base | 参数 |
|---|---|---|---:|
| Teacher `MinimalUNet` | 2-level U-Net | 128 | ~4.58 M |
| Student `StudentUNet` | 1×1 head + 2-level U-Net + 1×1 tail | 32 | ~0.79 M |

学生/教师参数比 ≈ 17.2%（0.791 / 4.58）。

实现：`diffdet/denoising/diffusion_unet.py`（`MinimalUNet`, `StudentUNet`）。

## 4. 特征残差分析

对成对 clean/foggy 图像提取 C5 特征，定义残差

```
Δ = f_foggy − f_clean
```

统计量（`experiments/logs/residual_stats.txt`，n=500）：

| 统计量 | 值 |
|---|---:|
| mean | -0.0138（近零） |
| std | 0.2382 |
| skewness | -10.24 |
| excess kurtosis | 385.8 |

- 近零均值 **不代表** 严格证明加性噪声假设。
- 分析重点在于**非高斯性与重尾**：skewness=-10.24、excess kurtosis≈385 表明残差呈
  强非高斯重尾分布，这正是用 U-Net 学习修正量（而非线性/高斯去噪）的动机。

实现：`tools/analyze_residuals.py`（直方图 + Gaussian Q-Q plot）。

## 5. 教师-学生知识蒸馏

- 冻结教师 U-Net，学生 U-Net 以特征级蒸馏学习。
- 损失：`L = L_task + λ·L_distill`，λ = 1.0。
  - `L_task`：学生修复特征 vs clean 特征的 masked MSE。
  - `L_distill`：学生修复特征 vs 教师修复特征的 masked MSE。
- 学生训练 50 epoch，验证 MSE ≈ 0.0455。

实现：`diffdet/distillation/distill.py`、`tools/train_student.py`。

## 6. 联合微调

- 检测器（Sparse R-CNN）从 baseline 权重初始化。
- U-Net（学生）以蒸馏权重初始化。
- `freeze_unet=False`，检测器 + U-Net 联合微调 3 epoch（AMP 混合精度）。
- 训练时保留原始 detection loss 与蒸馏阶段得到的 U-Net 权重，不额外设计新目标函数。

实现：`tools/train_detector.py` + `configs/diffdet/*_finetune.py`。

## 7. INT8 量化

- 对 U-Net 的 Conv2d/ConvTranspose2d/Linear 权重做 **weight-only INT8** 量化
  （per-output-channel 对称量化，scale=amax/127）。
- 推理时**反量化回 fp32** 参与 forward（W8A32 模拟）。
- 模型大小核算：U-Net 权重由 fp32（0.791M×4 ≈ 3.16 MB）降到 int8（0.791 MB）+ scale。
- 说明：weight-only INT8 **不** 等价于 INT8 算子加速，未实测不宣称提速。

实现：`diffdet/quantization/int8.py`。

## 8. 与 Sparse R-CNN 的一致性

DiffDet 的检测头、RPN head（EmbeddingRPNHead）、RoI head（SparseRoIHead）与 loss 均
**直接复用 Sparse R-CNN 官方实现**，仅 `extract_feat` 增加 C5 去噪一步。不使用
RT-DETR、YOLO 等其他架构替代。
