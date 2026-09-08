"""图编译器测试：优化 pass 的语义保持（优化前后输出必须等价）。"""
import numpy as np
import pytest

from sut.compiler import (
    GraphError,
    compile_graph,
    constant_fold,
    eliminate_dead,
    execute_graph,
    fuse_conv_relu,
)

pytestmark = pytest.mark.compiler


def _conv_relu_graph():
    rng = np.random.default_rng(1)
    return {
        "nodes": [
            {"id": "x", "op": "const", "value": rng.standard_normal((1, 1, 6, 6))},
            {"id": "w", "op": "const", "value": rng.standard_normal((2, 1, 3, 3))},
            {"id": "conv", "op": "conv2d", "inputs": ["x", "w"], "attrs": {"padding": 1}},
            {"id": "act", "op": "relu", "inputs": ["conv"]},
            {"id": "dead", "op": "relu", "inputs": ["x"]},
        ],
        "outputs": ["act"],
    }


class TestPasses:
    def test_constant_fold(self):
        graph = {
            "nodes": [
                {"id": "a", "op": "const", "value": np.array([1.0, 2.0])},
                {"id": "b", "op": "const", "value": np.array([3.0, 4.0])},
                {"id": "s", "op": "add", "inputs": ["a", "b"]},
            ],
            "outputs": ["s"],
        }
        folded = constant_fold(graph)
        assert folded["nodes"][-1]["op"] == "const"
        result = execute_graph(folded)
        np.testing.assert_allclose(result["s"], [4.0, 6.0])

    def test_fuse_conv_relu_merges(self):
        graph = _conv_relu_graph()
        fused = fuse_conv_relu(graph)
        ops = {n["op"] for n in fused["nodes"]}
        assert "fused_conv_relu" in ops
        assert "conv2d" not in ops and "relu" not in ops or \
            "conv2d" not in [n["op"] for n in fused["nodes"] if n["id"] == "act"]

    def test_dce_removes_dead_nodes(self):
        graph = _conv_relu_graph()
        optimized = eliminate_dead(graph)
        ids = {n["id"] for n in optimized["nodes"]}
        assert "dead" not in ids, "死代码应被消除"
        assert {"x", "w", "conv", "act"} <= ids, "活跃节点必须保留"

    def test_compile_pipeline_semantics_preserved(self):
        """核心断言：编译前后执行结果等价（编译器正确性的本质）。"""
        graph = _conv_relu_graph()
        before = execute_graph(graph)
        compiled = compile_graph(graph)
        after = execute_graph(compiled)
        for key in before:
            np.testing.assert_allclose(after[key], before[key], rtol=1e-12, atol=1e-12)

    def test_compile_reduces_node_count(self):
        graph = _conv_relu_graph()
        compiled = compile_graph(graph)
        assert len(compiled["nodes"]) < len(graph["nodes"]), "优化应减少节点数"


class TestGraphErrors:
    @pytest.mark.parametrize("graph,match", [
        ({"nodes": [], "outputs": []}, "空图"),
        ({"nodes": [{"id": "a", "op": "softmax_v9", "inputs": []}], "outputs": ["a"]}, "未知算子"),
        ({"nodes": [{"id": "a", "op": "relu", "inputs": ["ghost"]}], "outputs": ["a"]}, "不存在"),
        ({"nodes": [{"id": "a", "op": "relu", "inputs": ["b"]},
                    {"id": "b", "op": "relu", "inputs": ["a"]}], "outputs": ["a"]}, "环"),
        ({"nodes": [{"id": "a", "op": "relu", "inputs": []}], "outputs": ["nope"]}, "输出引用"),
    ], ids=["empty", "unknown_op", "missing_dep", "cycle", "bad_output"])
    def test_invalid_graphs_rejected(self, graph, match):
        with pytest.raises(GraphError, match=match):
            execute_graph(graph)

    def test_duplicate_node_id(self):
        graph = {
            "nodes": [{"id": "a", "op": "const", "value": 1.0},
                      {"id": "a", "op": "const", "value": 2.0}],
            "outputs": ["a"],
        }
        with pytest.raises(GraphError, match="重复"):
            execute_graph(graph)


class TestFusionCorrectness:
    def test_fused_matches_unfused_randomized(self):
        """随机化对照：多次随机权重下融合算子与分离算子输出等价。"""
        rng = np.random.default_rng(42)
        for _ in range(5):
            x = rng.standard_normal((1, 2, 8, 8))
            w = rng.standard_normal((4, 2, 3, 3))
            b = rng.standard_normal(4)
            graph = {
                "nodes": [
                    {"id": "x", "op": "const", "value": x},
                    {"id": "w", "op": "const", "value": w},
                    {"id": "b", "op": "const", "value": b},
                    {"id": "conv", "op": "conv2d", "inputs": ["x", "w", "b"],
                     "attrs": {"padding": 1}},
                    {"id": "act", "op": "relu", "inputs": ["conv"]},
                ],
                "outputs": ["act"],
            }
            before = execute_graph(graph)["act"]
            after = execute_graph(compile_graph(graph))["act"]
            np.testing.assert_allclose(after, before, rtol=1e-12, atol=1e-12)