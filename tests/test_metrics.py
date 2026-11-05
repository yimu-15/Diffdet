"""VOC 11-point mAP 评估指标测试。"""

import numpy as np
import pytest

from diffdet.evaluation.metrics import compute_voc_map, voc_ap


def test_voc_ap_perfect():
    rec = np.array([0.0, 0.5, 1.0])
    prec = np.array([1.0, 1.0, 1.0])
    assert voc_ap(rec, prec, use_07_metric=True) == pytest.approx(1.0)


def test_voc_ap_zero():
    rec = np.array([0.0])
    prec = np.array([0.0])
    assert voc_ap(rec, prec, use_07_metric=True) == 0.0


def test_compute_voc_map_perfect():
    # 4 GT, 全部预测为 TP（按置信度降序）
    results = {0: ([1, 1, 1, 1], 4)}
    aps, mean = compute_voc_map(results)
    assert aps[0] == pytest.approx(1.0) and mean == pytest.approx(1.0)


def test_compute_voc_map_randomish():
    # 8 GT, 一半 TP 一半 FP
    tps = [1, 0, 1, 0, 1, 0, 1, 0]
    results = {0: (tps, 8)}
    aps, mean = compute_voc_map(results)
    assert 0.0 <= aps[0] <= 1.0
    assert 0.3 < aps[0] < 0.7
