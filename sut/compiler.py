"""图编译器（被测对象）：简单算子图 IR + 三个优化 pass + 解释执行器。

模拟 AI 编译器前端：给定算子图，依次做
  1) 常量折叠 constant folding
  2) conv+relu 算子融合 fusion
  3) 死代码消除 DCE
再解释执行。测试保证优化前后语义等价（见 tests/test_compiler/）。
"""
import copy
import numpy as np

from sut.operators import conv2d, relu

OPS = {"conv2d", "relu", "add", "mul", "const", "fused_conv_relu"}


class GraphError(ValueError):
    pass


def validate_graph(graph: dict) -> None:
    """结构校验：未知算子 / 引用不存在的节点 / 输出缺失。"""
    nodes = graph["nodes"]
    if not nodes:
        raise GraphError("空图")
    by_id = {}
    for node in nodes:
        if node["op"] not in OPS:
            raise GraphError(f"未知算子: {node['op']}")
        if node["id"] in by_id:
            raise GraphError(f"节点 id 重复: {node['id']}")
        by_id[node["id"]] = node
    for node in nodes:
        for dep in node.get("inputs", []):
            if dep not in by_id:
                raise GraphError(f"节点 {node['id']} 引用不存在的输入 {dep}")
    for out in graph.get("outputs", []):
        if out not in by_id:
            raise GraphError(f"输出引用不存在的节点 {out}")


def constant_fold(graph: dict) -> dict:
    """两个 const 节点的 add/mul 折叠为单个 const。"""
    graph = copy.deepcopy(graph)
    nodes = graph["nodes"]
    changed = True
    while changed:
        changed = False
        by_id = {n["id"]: n for n in nodes}
        for node in nodes:
            if node["op"] not in ("add", "mul"):
                continue
            deps = node.get("inputs", [])
            if len(deps) == 2 and all(by_id[d]["op"] == "const" for d in deps):
                a, b = (np.asarray(by_id[d]["value"]) for d in deps)
                value = a + b if node["op"] == "add" else a * b
                nodes.append({"id": node["id"], "op": "const",
                              "value": value, "inputs": []})
                nodes.remove(node)
                changed = True
                break
    return graph


def fuse_conv_relu(graph: dict) -> dict:
    """conv2d->relu 融合为 fused_conv_relu（经典 AI 编译器 pass）。"""
    graph = copy.deepcopy(graph)
    nodes = graph["nodes"]
    by_id = {n["id"]: n for n in nodes}
    for node in list(nodes):
        if node["op"] != "relu":
            continue
        deps = node.get("inputs", [])
        if len(deps) == 1 and by_id[deps[0]]["op"] == "conv2d":
            conv = by_id[deps[0]]
            if conv["id"] in graph["outputs"]:
                continue  # conv 是输出时不能融合
            nodes.append({
                "id": node["id"], "op": "fused_conv_relu", "inputs": conv["inputs"],
                "attrs": conv.get("attrs", {}),
            })
            nodes.remove(conv)
            nodes.remove(node)
    return graph


def eliminate_dead(graph: dict) -> dict:
    """删除 outputs 不可达的节点。"""
    graph = copy.deepcopy(graph)
    by_id = {n["id"]: n for n in graph["nodes"]}
    reachable, stack = set(), list(graph.get("outputs", []))
    while stack:
        nid = stack.pop()
        if nid in reachable:
            continue
        reachable.add(nid)
        stack.extend(by_id[nid].get("inputs", []))
    graph["nodes"] = [n for n in graph["nodes"] if n["id"] in reachable]
    return graph


def compile_graph(graph: dict) -> dict:
    """完整编译流水线。"""
    validate_graph(graph)
    graph = constant_fold(graph)
    graph = fuse_conv_relu(graph)
    graph = eliminate_dead(graph)
    return graph


def execute_graph(graph: dict, feed: dict | None = None) -> dict:
    """拓扑序解释执行，返回 outputs 字典。"""
    validate_graph(graph)
    feed = feed or {}
    values, done = dict(feed), set()
    pending = {n["id"]: n for n in graph["nodes"]}
    while pending:
        progressed = False
        for nid, node in list(pending.items()):
            deps = node.get("inputs", [])
            if any(d not in values for d in deps):
                continue
            values[nid] = _eval_node(node, [values[d] for d in deps])
            done.add(nid)
            del pending[nid]
            progressed = True
        if not progressed:
            raise GraphError("图中存在环或不可满足的依赖")
    return {out: values[out] for out in graph["outputs"]}


def _eval_node(node: dict, inputs: list) -> np.ndarray:
    op = node["op"]
    if op == "const":
        return np.asarray(node["value"], dtype=np.float64)
    if op == "add":
        return inputs[0] + inputs[1]
    if op == "mul":
        return inputs[0] * inputs[1]
    if op == "relu":
        return relu(inputs[0])
    if op == "conv2d":
        attrs = node.get("attrs", {})
        return conv2d(inputs[0], inputs[1], inputs[2] if len(inputs) > 2 else None,
                      stride=attrs.get("stride", 1), padding=attrs.get("padding", 0))
    if op == "fused_conv_relu":
        attrs = node.get("attrs", {})
        return relu(conv2d(inputs[0], inputs[1], inputs[2] if len(inputs) > 2 else None,
                           stride=attrs.get("stride", 1), padding=attrs.get("padding", 0)))
    raise GraphError(f"未知算子: {op}")
