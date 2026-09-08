"""自研 numpy 算子库（被测对象）。

模拟 AI 芯片软件栈中的算子库：conv2d / matmul / softmax / batchnorm2d。
测试策略：以 PyTorch 对应算子为 oracle 做数值对照（见 tests/test_operators/）。
"""
import numpy as np


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """矩阵乘法（朴素三重循环实现，正确性优先）。"""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"matmul 仅支持 2D 输入, got {a.ndim}D/{b.ndim}D")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"形状不兼容: {a.shape} x {b.shape}")
    out = np.zeros((a.shape[0], b.shape[1]))
    for i in range(a.shape[0]):
        for k in range(a.shape[1]):
            if a[i, k] == 0.0:
                continue  # 稀疏跳零优化
            for j in range(b.shape[1]):
                out[i, j] += a[i, k] * b[k, j]
    return out


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 softmax（减最大值防溢出）。"""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0 or not -x.ndim <= axis < x.ndim:
        raise ValueError(f"非法 axis={axis} for shape {x.shape}")
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def conv2d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """2D 卷积（im2col + 矩阵乘实现）。

    x: (N, C_in, H, W)  weight: (F, C_in, KH, KW)
    """
    x = np.asarray(x, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if x.ndim != 4:
        raise ValueError(f"输入必须 4D (N,C,H,W), got {x.shape}")
    if weight.ndim != 4:
        raise ValueError(f"权重必须 4D (F,C,KH,KW), got {weight.shape}")
    n, c_in, h, w = x.shape
    f, c_w, kh, kw = weight.shape
    if c_in != c_w:
        raise ValueError(f"通道数不匹配: input {c_in} vs weight {c_w}")
    if stride < 1 or padding < 0:
        raise ValueError(f"非法 stride={stride} 或 padding={padding}")

    if padding:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    h_out = (x.shape[2] - kh) // stride + 1
    w_out = (x.shape[3] - kw) // stride + 1
    if h_out <= 0 or w_out <= 0:
        raise ValueError(
            f"输出尺寸为 0: input {x.shape[2:]} kernel {(kh, kw)} stride {stride}"
        )

    # 滑窗 -> (N, C, KH, KW, H_out, W_out) -> 重排为 im2col
    windows = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(2, 3))
    windows = windows[:, :, ::stride, ::stride]          # (N, C, H_out, W_out, KH, KW)
    cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(n * h_out * w_out, c_in * kh * kw)
    w_mat = weight.reshape(f, c_in * kh * kw)

    out = w_mat @ cols.T                                  # (F, N*H_out*W_out)
    if bias is not None:
        bias = np.asarray(bias, dtype=np.float64)
        if bias.shape != (f,):
            raise ValueError(f"bias 形状应为 ({f},), got {bias.shape}")
        out = out + bias[:, None]
    return out.T.reshape(n, h_out, w_out, f).transpose(0, 3, 1, 2)


def batchnorm2d(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """推理态 BatchNorm（使用给定统计量，非训练态）。"""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 4:
        raise ValueError(f"batchnorm2d 仅支持 4D (N,C,H,W), got {x.shape}")
    c = x.shape[1]
    for name, p in (("gamma", gamma), ("beta", beta), ("mean", mean), ("var", var)):
        p = np.asarray(p, dtype=np.float64)
        if p.shape != (c,):
            raise ValueError(f"{name} 形状应为 ({c},), got {p.shape}")
    if np.any(var < 0):
        raise ValueError("variance 不能为负")
    inv_std = 1.0 / np.sqrt(var + eps)
    return (x - mean[None, :, None, None]) * inv_std[None, :, None, None] \
        * gamma[None, :, None, None] + beta[None, :, None, None]


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(x), 0)
