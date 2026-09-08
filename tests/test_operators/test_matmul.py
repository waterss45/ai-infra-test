"""matmul 算子测试：自研实现 vs PyTorch oracle（含边界与异常）。"""
import numpy as np
import pytest
import torch

from sut.operators import matmul
from tests.core.numeric import assert_allclose

pytestmark = pytest.mark.operators


@pytest.mark.parametrize(
    "m,k,n",
    [(1, 1, 1), (1, 8, 1), (8, 1, 8), (16, 16, 16), (32, 64, 16)],
    ids=["min", "row_vec", "col_vec", "square", "wide"],
)
def test_matmul_against_torch(m, k, n):
    rng = np.random.default_rng(m * k + n)
    a = rng.standard_normal((m, k))
    b = rng.standard_normal((k, n))
    got = matmul(a, b)
    want = torch.tensor(a) @ torch.tensor(b)
    assert_allclose(got, want.numpy(), rtol=1e-9, atol=1e-8, note=f"{m}x{k}x{n}")


def test_matmul_zero_matrix():
    a = np.zeros((4, 4))
    b = np.ones((4, 4))
    assert_allclose(matmul(a, b), np.zeros((4, 4)), note="零矩阵")


def test_matmul_shape_mismatch():
    with pytest.raises(ValueError, match="形状不兼容"):
        matmul(np.zeros((2, 3)), np.zeros((2, 3)))


def test_matmul_rejects_1d():
    with pytest.raises(ValueError, match="2D"):
        matmul(np.zeros(4), np.zeros((4, 2)))
