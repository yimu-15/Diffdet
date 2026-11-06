"""共享路径解析：把仓库根目录加入 sys.path 并暴露常用目录。"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA_DIR = os.environ.get('DIFFDET_DATA', os.path.join(ROOT, 'data'))
WORK_DIRS = os.environ.get('DIFFDET_WORKDIRS', os.path.join(ROOT, 'work_dirs'))
CHECKPOINTS = os.environ.get('DIFFDET_CKPT', os.path.join(ROOT, 'checkpoints'))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
