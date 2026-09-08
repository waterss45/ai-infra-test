"""batchnorm2d 算子测试：推理态归一化 vs PyTorch oracle。"""
import numpy as np
import pytest
import torch

from sut.operators import batchnorm2d
from tests.core.numeric import assert_allclose, assert_finite

pytestmark = pytest.mark.operators


def _torch_bn(x, g, b, m, v, eps=1e-5):
    layer = torch.nn.BatchNorm2d(x.shape[1], eps=eps).double()
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(g))
        layer.bias.copy_(torch.tensor(b))
        layer.running_mean.copy_(torch.tensor(m))
        layer.running_var.copy_(torch.tensor(v))
    layer.eval()
    with torch.no_grad():
        return layer(torch.tensor(x, dtype=torch.float64)).numpy()


def test_batchnorm_against_torch():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 6, 5, 5)) * 3 + 1
    g, b = rng.standard_normal(6), rng.standard_normal(6)
    m, v = rng.standard_normal(6), rng.random(6) + 0.1
    got = batchnorm2d(x, g, b, m, v)
    want = _torch_bn(x, g, b, m, v)
    assert_allclose(got, want, rtol=1e-8, atol=1e-8)


def test_batchnorm_normalizes_channels():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((8, 3, 4, 4)) * 5
    m = x.mean(axis=(0, 2, 3))
    v = x.var(axis=(0, 2, 3))
    out = batchnorm2d(x, np.ones(3), np.zeros(3), m, v)
    assert_allclose(out.mean(axis=(0, 2, 3)), np.zeros(3), rtol=1e-9, atol=1e-9)
    assert_allclose(out.var(axis=(0, 2, 3)), np.ones(3), rtol=1e-6, atol=1e-7)


def test_batchnorm_zero_variance():
    """边界：方差为 0 时 eps 防止除零。"""
    x = np.ones((2, 2, 3, 3))
    out = batchnorm2d(x, np.ones(2), np.zeros(2), np.zeros(2), np.zeros(2))
    assert_finite(out, note="零方差")


def test_batchnorm_negative_variance_rejected():
    with pytest.raises(ValueError, match="variance"):
        batchnorm2d(np.ones((1, 2, 2, 2)), np.ones(2), np.zeros(2),
                    np.zeros(2), -np.ones(2))


def test_batchnorm_param_shape_mismatch():
    with pytest.raises(ValueError, match="gamma"):
        batchnorm2d(np.ones((1, 2, 2, 2)), np.ones(3), np.zeros(2),
                    np.zeros(2), np.ones(2))
