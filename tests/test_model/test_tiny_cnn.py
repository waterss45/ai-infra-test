"""模型级测试：精度 / 确定性 / 精度格式一致性 / 批量一致性。

对位 JD「AI 异构推理框架 + AI 测试经验」：
  - 精度阈值（准出标准写在断言里）
  - 同 seed 可复现（确定性）
  - fp32 vs bfloat16 容限对照（AI 芯片常见 bf16 推理格式）
  - batch=1 vs batch=N 一致性（运行时批量调度正确性）
"""
import numpy as np
import pytest
import torch

from sut.tiny_model import SEED, TinyCNN, load_model, make_synthetic_dataset

pytestmark = pytest.mark.model

ACCURACY_THRESHOLD = 0.95


@pytest.fixture(scope="module")
def model() -> TinyCNN:
    return load_model()


@pytest.fixture(scope="module")
def val_set():
    x, y = make_synthetic_dataset()
    split = int(x.shape[0] * 0.8)
    return x[split:], y[split:]


class TestAccuracy:
    @pytest.mark.smoke
    def test_val_accuracy_meets_threshold(self, model, val_set):
        x, y = val_set
        with torch.no_grad():
            acc = (model(x).argmax(1) == y).float().mean().item()
        assert acc >= ACCURACY_THRESHOLD, f"验证集精度 {acc:.4f} < 准出阈值 {ACCURACY_THRESHOLD}"

    def test_confusion_is_bounded(self, model, val_set):
        """各类别召回均不低于 0.85，避免整体精度掩盖单类坍塌。"""
        x, y = val_set
        with torch.no_grad():
            pred = model(x).argmax(1)
        for cls in range(5):
            mask = y == cls
            recall = (pred[mask] == cls).float().mean().item()
            assert recall >= 0.85, f"类别 {cls} 召回 {recall:.3f} < 0.85"


class TestDeterminism:
    def test_same_seed_reproducible(self):
        """同 seed 数据管线应逐位可复现。"""
        x1, y1 = make_synthetic_dataset(seed=SEED)
        x2, y2 = make_synthetic_dataset(seed=SEED)
        assert torch.equal(x1, x2) and torch.equal(y1, y2)

    def test_model_output_deterministic(self, model, val_set):
        x, _ = val_set
        with torch.no_grad():
            out1, out2 = model(x), model(x)
        assert torch.equal(out1, out2), "eval 态两次前向必须逐位一致"


class TestPrecisionFormat:
    def test_bfloat16_consistency(self, model, val_set):
        """fp32 vs bfloat16：AI 芯片常用 bf16 推理，输出应在容限内一致。"""
        import copy

        x, _ = val_set
        model_bf16 = copy.deepcopy(model).to(torch.bfloat16)
        with torch.no_grad():
            out_fp32 = model(x)
            out_bf16 = model_bf16(x.to(torch.bfloat16)).float()
        diff = (out_fp32 - out_bf16).abs()
        # logits 量级 ~O(10)，bf16 舍入误差按经验容限放宽
        assert diff.mean().item() < 0.5, f"bf16 平均偏差过大: {diff.mean().item():.4f}"
        same_argmax = (out_fp32.argmax(1) == out_bf16.argmax(1)).float().mean().item()
        assert same_argmax >= 0.99, f"bf16 预测类别一致率 {same_argmax:.4f} < 99%"


class TestBatchConsistency:
    @pytest.mark.parametrize("batch", [2, 8, 64], ids=["b2", "b8", "b64"])
    def test_batch_equals_single(self, model, val_set, batch):
        """批量推理与逐样本推理结果必须一致（运行时调度正确性）。"""
        x, _ = val_set
        with torch.no_grad():
            batched = model(x[:batch])
            for i in range(batch):
                single = model(x[i:i + 1])
                assert torch.allclose(batched[i:i + 1], single, atol=1e-6), \
                    f"batch={batch} 第 {i} 条与单条推理不一致"

    def test_init_without_seed_varies(self):
        """错误推测：不同 init 产出的模型输出应不同（防止权重固化）。"""
        torch.manual_seed(1)
        m1 = TinyCNN()
        torch.manual_seed(2)
        m2 = TinyCNN()
        x = torch.randn(4, 1, 8, 8)
        with torch.no_grad():
            diff = (m1(x) - m2(x)).abs().max().item()
        assert diff > 1e-6, "不同初始化的输出不应相同"


class TestOnnxRuntime:
    def test_export_matches_pytorch(self, model, val_set):
        """编译器→运行时链路：ONNX 导出后 onnxruntime 输出与 PyTorch 对照。"""
        import onnxruntime as ort

        x, _ = val_set
        session = ort.InferenceSession("artifacts/model.onnx", providers=["CPUExecutionProvider"])
        got = session.run(None, {"input": x.numpy()})[0]
        with torch.no_grad():
            want = model(x).numpy()
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)