import os
import sys

import pytest
import torch

# 确保从仓库根目录导入 diffdet
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(scope='session')
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
