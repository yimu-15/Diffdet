"""Detection inference smoke test (requires a trained checkpoint).

The checkpoint is *not* distributed with this repository, so this test is
skipped (status: BLOCKED) unless one is available. Enable it with::

    DIFFDET_CKPT=/path/to/sparse_rcnn_voc_rainfog_final_int8.pth \
    DIFFDET_IMAGE=/path/to/foggy_image.jpg \
    pytest tests/test_inference_smoke.py -v
"""

import os

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CKPT = os.environ.get(
    'DIFFDET_CKPT',
    os.path.join(ROOT, 'checkpoints', 'sparse_rcnn_voc_rainfog_final_int8.pth'))
CFG = os.path.join(ROOT, 'configs', 'diffdet',
                   'sparse_rcnn_r50_fpn_300_proposals_diffdet_final.py')
IMG = os.environ.get(
    'DIFFDET_IMAGE',
    os.path.join(ROOT, 'data', 'VOC-FOG', 'moderate', 'VOC2007',
                 'JPEGImages', '000125.jpg'))

pytestmark = pytest.mark.skipif(
    not os.path.exists(CKPT),
    reason='BLOCKED: trained checkpoint not found (weights are not '
           'distributed in this repository); set DIFFDET_CKPT to enable')


def _patch_torch_load():
    """torch>=2.6 defaults to weights_only=True, which rejects mmengine ckpts."""
    if getattr(torch.load, '_diffdet_patched', False):
        return
    orig = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return orig(*args, **kwargs)

    _load._diffdet_patched = True
    torch.load = _load


def test_inference_smoke():
    _patch_torch_load()

    from mmdet.apis import init_detector, inference_detector

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = init_detector(CFG, CKPT, device=device)
    model.dataset_meta = {
        'classes': ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                    'train', 'tvmonitor'),
        'palette': 'voc',
    }
    model.eval()

    result = inference_detector(model, IMG)
    pred = result.pred_instances
    assert pred.bboxes.shape[1] == 4
    assert pred.bboxes.shape[0] == pred.scores.shape[0]
    assert pred.scores.numel() == 0 or (pred.scores >= 0).all().item()
    assert pred.labels.max().item() < 20
