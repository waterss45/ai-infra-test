"""conv2d 算子测试：自研 numpy 实现 vs PyTorch oracle。

方法：等价类（stride/padding 组合）+ 边界值（1x1 kernel、最小空间尺寸、kernel>输入）+ 数据驱动。
"""
import numpy as np
import pytest
import torch

from sut.operators import conv2d
from tests.core.numeric import assert_allclose
from tests.core.yaml_data import case_ids, load_yaml_cases

pytestmark = pytest.mark.operators

SHAPES = load_yaml_cases("conv_shapes.yaml")


def _torch_conv(x, w, b, stride, padding):
    return torch.nn.functional.conv2d(
        torch.tensor(x), torch.tensor(w),
        None if b is None else torch.tensor(b),
        stride=stride, padding=padding,
    ).numpy()


def _make_case(case, rng):
    x = rng.standard_normal((case["n"], case["c_in"], case["h"], case["w"]))
    w = rng.standard_normal((case["f"], case["c_in"], case["kh"], case["kw"]))
    b = rng.standard_normal(case["f"]) if case["expect_code"] == 200 else None
    return x, w, b


@pytest.mark.parametrize("case", SHAPES, ids=case_ids(SHAPES))
def test_conv2d_against_torch(case):
    rng = np.random.default_rng(0)
    x, w, b = _make_case(case, rng)
    if case["expect_code"] == 200:
        got = conv2d(x, w, b, stride=case["stride"], padding=case["padding"])
        want = _torch_conv(x, w, b, case["stride"], case["padding"])
        assert_allclose(got, want, rtol=1e-9, atol=1e-8, note=case["id"])
    else:
        with pytest.raises(ValueError, match="输出尺寸为 0"):
            conv2d(x, w, b, stride=case["stride"], padding=case["padding"])


def test_conv2d_channel_mismatch():
    with pytest.raises(ValueError, match="通道数不匹配"):
        conv2d(np.zeros((1, 2, 4, 4)), np.zeros((1, 3, 3, 3)))


def test_conv2d_rejects_bad_stride_padding():
    x, w = np.zeros((1, 1, 4, 4)), np.zeros((1, 1, 2, 2))
    with pytest.raises(ValueError, match="stride"):
        conv2d(x, w, stride=0)
    with pytest.raises(ValueError, match="padding"):
        conv2d(x, w, padding=-1)


def test_conv2d_nan_propagates():
    """错误推测：NaN 输入应显式传播而非静默吞掉。"""
    x = np.ones((1, 1, 3, 3))
    x[0, 0, 1, 1] = np.nan
    out = conv2d(x, np.ones((1, 1, 2, 2)))
    assert np.isnan(out).any(), "NaN 应传播到输出"
