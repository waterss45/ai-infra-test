"""数值断言工具：误差统计 + 容限对照（AI 数值测试的核心手段）。"""
import numpy as np


def assert_allclose(actual, expected, rtol=1e-7, atol=1e-9, note="") -> dict:
    """带误差统计的 allclose 断言，失败时输出 max/mean 误差定位问题。"""
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape, (
        f"形状不一致: actual {actual.shape} vs expected {expected.shape} {note}"
    )
    nan_match = np.array_equal(np.isnan(actual), np.isnan(expected))
    assert nan_match, f"NaN 位置不一致 {note}"
    diff = np.abs(actual - expected)[~np.isnan(actual)]
    stats = {
        "max_abs_err": float(diff.max()) if diff.size else 0.0,
        "mean_abs_err": float(diff.mean()) if diff.size else 0.0,
    }
    tolerance = atol + rtol * np.abs(expected[~np.isnan(expected)])
    ok = bool(np.all(diff <= tolerance)) if diff.size else True
    assert ok, (
        f"数值超差 {note}: max_abs_err={stats['max_abs_err']:.3e} "
        f"(atol={atol:.1e}, rtol={rtol:.1e})"
    )
    return stats


def assert_finite(x, note="") -> None:
    arr = np.asarray(x)
    assert np.all(np.isfinite(arr)), f"存在 NaN/Inf {note}"


def has_nan(x) -> bool:
    return bool(np.isnan(np.asarray(x, dtype=np.float64)).any())
