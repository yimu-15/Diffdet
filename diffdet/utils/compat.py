"""torch>=2.6 兼容补丁：mmengine 0.10.7 checkpoint 加载适配。

torch 2.6+ 将 torch.load 的 weights_only 默认值改为 True，而 mmengine
0.10.7 的 load_from_local 仍按旧默认调用，导致加载包含 HistoryBuffer /
ConfigDict 等全局类的自训 checkpoint 时抛 UnpicklingError。
本文件在 import 时将 torch.load 恢复为 weights_only=False 的旧行为。
自训 checkpoint 均为本地可信来源，可安全关闭 weights_only。
"""

import torch

if torch.__version__.startswith('2.'):
    _orig_torch_load = torch.load

    def _patched_torch_load(filename, *args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _orig_torch_load(filename, *args, **kwargs)

    torch.load = _patched_torch_load
