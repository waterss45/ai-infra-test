"""原生 C++ kernel 加载器：未构建则跳过（Linux CI 会构建）。"""
import ctypes
import sys
from pathlib import Path

import pytest

NATIVE_DIR = Path(__file__).resolve().parent.parent.parent / "sut" / "native"


def _load_lib():
    for name in ("libmm.so", "mm.dll", "libmm.dylib"):
        path = NATIVE_DIR / name
        if path.exists():
            lib = ctypes.CDLL(str(path))
            lib.mm_f32.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ]
            lib.mm_f32.restype = None
            return lib
    return None


@pytest.fixture(scope="session")
def native_lib():
    """已构建的原生库；未构建则 skip（不阻塞平台无关用例）。"""
    lib = _load_lib()
    if lib is None:
        pytest.skip("原生库未构建（运行 scripts/build_native.sh）")
    return lib
