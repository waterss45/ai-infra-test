"""C++ 原生 kernel 测试：ctypes 加载 .so/.dll，与 numpy 对照。

体现「底层软件栈测试」：跨语言 FFI 边界 + 数值对照。
"""
import ctypes

import numpy as np
import pytest

from tests.core.numeric import assert_allclose

pytestmark = pytest.mark.native


def _run_mm(lib, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    out = np.zeros((a32.shape[0], b32.shape[1]), dtype=np.float32)
    lib.mm_f32(
        a32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        a32.shape[0], a32.shape[1], b32.shape[1],
    )
    return out


@pytest.mark.parametrize("m,k,n", [(1, 1, 1), (3, 4, 5), (16, 16, 16), (32, 64, 16)],
                         ids=["min", "rect", "square", "wide"])
def test_native_matmul_matches_numpy(native_lib, m, k, n):
    rng = np.random.default_rng(m + k + n)
    a = rng.standard_normal((m, k))
    b = rng.standard_normal((k, n))
    got = _run_mm(native_lib, a, b)
    assert_allclose(got, a @ b, rtol=1e-5, atol=1e-5, note=f"{m}x{k}x{n}")


def test_native_matmul_zero_matrix(native_lib):
    out = _run_mm(native_lib, np.zeros((2, 2)), np.ones((2, 2)))
    assert_allclose(out, np.zeros((2, 2)), note="零矩阵")
