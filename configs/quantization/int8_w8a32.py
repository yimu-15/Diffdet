# weight-only INT8 量化配置（Phase 5）
# 由 tools/benchmark.py / merge_final_model.py 使用。

quantization = dict(
    mode='weight-only',         # 仅权重量化（激活保持 fp32）
    bits=8,                     # int8
    scheme='symmetric',         # per-output-channel 对称量化
    scale='amax/127',           # scale = absmax/127
    clamp=(-128, 127),
    dequantize=True,            # 反量化回 fp32 参与 forward（W8A32 模拟）
    targets=['Conv2d', 'ConvTranspose2d', 'Linear'],
)
