# 学生 U-Net 蒸馏训练配置（Phase 2）
# 由 tools/train_student.py 读取；此处归档便于复现与审阅。

student = dict(
    in_ch=2048,
    dim=128,          # 1x1 降维通道（head 2048->128）
    base=32,          # U-Net 核心通道
)

teacher = dict(
    arch='MinimalUNet',
    base=128,         # 教师核心通道（~4.58M 参数）
)

training = dict(
    epochs=50,
    lr=1e-3,
    batch_size=4,
    seed=42,
    lam=1.0,          # 蒸馏权重 λ = 1.0
    loss='L = L_task + lam * L_distill',  # 特征级蒸馏
)
