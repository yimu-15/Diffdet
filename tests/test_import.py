"""Python import 与 MODELS 注册测试。"""


def test_import_diffdet():
    import diffdet
    assert diffdet.__version__


def test_detector_registered_in_models():
    import diffdet  # noqa: F401  (注册 DiffDetDetector)
    from mmdet.registry import MODELS
    assert 'DiffDetDetector' in MODELS.module_dict


def test_denoiser_symbols_exported():
    from diffdet import (DiffDetDetector, MinimalUNet, StudentUNet,
                         denoise_features)
    assert DiffDetDetector is not None
    assert MinimalUNet is not None
    assert StudentUNet is not None
    assert denoise_features is not None
