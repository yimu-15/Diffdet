"""VOC 11-point mAP 计算（与 mmdet VOCMetric 等价的参考实现）。

生产环境评估请使用 mmdet 的 ``VOCMetric``（见 tools/evaluate.py），
本模块提供与官方 11-point AP 定义一致的独立实现，用于单元测试与结果交叉核对。
"""

import numpy as np


def voc_ap(rec, prec, use_07_metric=True):
    """给定 recall/precision 曲线，计算 AP。

    use_07_metric=True -> VOC2007 11-point 插值；False -> 全点积分。
    """
    rec = np.asarray(rec, dtype=np.float64)
    prec = np.asarray(prec, dtype=np.float64)
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            p = prec[rec >= t]
            ap += (np.max(p) if p.size else 0.0) / 11.0
    else:
        # 在 recall 端点补零后按 recall 单调区段积分
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def compute_voc_map(results_by_class, use_07_metric=True):
    """计算各类 AP 与 mAP。

    results_by_class: {class_id: (tp_scores_desc, num_gt)}，其中 tp_scores_desc
    为按置信度降序的 0/1 正确标记（1=TP, 0=FP）。返回值 (ap_dict, mean_ap)。
    """
    aps = {}
    for cls, (tps, npos) in results_by_class.items():
        tps = np.asarray(tps, dtype=np.float64)
        fp = 1.0 - tps
        fp_cum = np.cumsum(fp)
        tp_cum = np.cumsum(tps)
        rec = tp_cum / float(npos) if npos > 0 else np.zeros_like(tp_cum)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
        aps[cls] = voc_ap(rec, prec, use_07_metric)
    mean_ap = float(np.mean(list(aps.values()))) if aps else 0.0
    return aps, mean_ap


__all__ = ['voc_ap', 'compute_voc_map']
