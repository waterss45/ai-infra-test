"""softmax 算子测试：数值稳定性是重点（大数溢出场景）。"""
import numpy as np
import pytest
import torch

from sut.operators import softmax
from tests.core.numeric import assert_allclose, assert_finite, has_nan

pytestmark = pytest.mark.operators


@pytest.mark.parametrize("shape,axis", [
    ((8,), -1), ((4, 8), -1), ((4, 8), 0), ((2, 3, 4), 2), ((1, 1), 1),
], ids=["1d", "2d_last", "2d_first", "3d", "singleton"])
def test_softmax_against_torch(shape, axis):
    rng = np.random.default_rng(sum(shape))
    x = rng.standard_normal(shape)
    got = softmax(x, axis=axis)
    want = torch.softmax(torch.tensor(x), dim=axis).numpy()
    assert_allclose(got, want, rtol=1e-9, atol=1e-9, note=f"{shape} axis={axis}")
    assert_finite(got)


def test_softmax_large_inputs_stable():
    """数值稳定性核心场景：极大输入不应溢出（torch oracle 同样处理）。"""
    x = np.array([[1e4, 1e4 + 1.0, 1e4 - 1.0]])
    got = softmax(x)
    want = torch.softmax(torch.tensor(x, dtype=torch.float64), dim=-1).numpy()
    assert_finite(got, note="极大值输入")
    assert_allclose(got, want, rtol=1e-9, atol=1e-9)


def test_softmax_rows_sum_to_one():
    rng = np.random.default_rng(7)
    out = softmax(rng.standard_normal((6, 10)))
    assert_allclose(out.sum(axis=-1), np.ones(6), rtol=1e-12, atol=1e-12)


def test_softmax_invalid_axis():
    with pytest.raises(ValueError, match="axis"):
        softmax(np.zeros((2, 2)), axis=5)


def test_softmax_nan_input_propagates():
    x = np.array([1.0, np.nan, 2.0])
    out = softmax(x)
    assert has_nan(out), "NaN 输入应传播"
