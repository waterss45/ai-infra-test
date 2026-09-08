# AI 基础软件测试套件 (ai-infra-test)

面向 **AI 基础软件栈** 的分层自动化测试项目，覆盖：**算子库 → 模型 → 图编译器 → 算子调度器** 四层被测对象，并含 **C++ 原生 kernel** 的 FFI 边界测试。全部被测对象（SUT）为项目自研，测试与被测同仓闭环，可在 Linux CI 上一键复现。

## 与岗位要求的映射

| JD 要求 | 项目落点 |
|---|---|
| AI 领域测试策略 / 方案 / 用例输出 | `docs/test-strategy.md` 分层策略 + 每个测试模块头部标注用例设计方法 |
| 算子库知识 | `sut/operators.py` 自研 numpy 算子库，测试以 PyTorch 为 oracle 做数值对照（容限理论、误差统计、NaN 传播） |
| 深度学习框架（PyTorch 等）使用 | `sut/tiny_model.py` TinyCNN：精度阈值、确定性、fp32/bf16 一致性、批量一致性 |
| AI 异构推理框架 / 编译器测试 | `sut/compiler.py` 图编译器（常量折叠 / conv+relu 融合 / DCE）语义保持测试 + ONNX 导出→onnxruntime 运行时一致性 |
| 调度系统测试 | `sut/scheduler.py` DAG 调度器：依赖顺序、资源预算、优先级、死锁/环检测、200 节点随机 DAG 健壮性 |
| C++ / Shell / Linux | C++ matmul kernel（ctypes FFI 对照测试）、`scripts/run_all.sh` 全流程 Shell 脚本、GitHub Actions ubuntu CI |
| 测试报告输出 | `scripts/gen_report.py` 自动生成 Markdown 测报（分模块统计/耗时/结论） |

## 架构

```
被测对象 (sut/)                    测试 (tests/)
├── operators.py   numpy 算子库    ├── test_operators/   vs PyTorch oracle 数值对照
├── tiny_model.py  TinyCNN 模型    ├── test_model/       精度/确定性/bf16/批量/ONNX
├── compiler.py    图编译器+passes ├── test_compiler/    pass 语义保持 + 异构图拒绝
├── scheduler.py   DAG 调度器      ├── test_scheduler/   依赖/资源/优先级/规模
└── native/matmul.cpp  C++ kernel  └── test_native       ctypes FFI 数值对照
```

## 快速开始（Linux）

```bash
# 依赖（CPU 版 torch）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 一键：构建 C++ kernel → ruff → 全量测试 → 生成测报
bash scripts/run_all.sh

# 仅跑某一层
python -m pytest tests/test_operators -v
python -m pytest -m "model and smoke"
```

Windows 本地：C++ kernel 需自行编译；未编译时 native 用例自动 skip，其余用例全量可跑。

## 测试设计亮点

- **Oracle 对照模式**：自研实现 vs 工业标准实现（PyTorch/onnxruntime/numpy），断言容限按数据类型与量级设定，失败输出 max/mean 误差统计快速定位
- **精度格式测试**：fp32 vs bfloat16（AI 芯片主流推理格式），同时校验数值容限与 argmax 类别一致率
- **编译器语义保持**：每个优化 pass 单测 + 全流水线"编译前后执行结果等价"性质断言，随机化多轮对照
- **调度器性质测试**：随机生成 200 节点 DAG，程序化验证依赖序与资源约束全满足；同输入调度结果确定性一致
- **数据驱动**：conv2d 形状矩阵 YAML 化，等价类 + 边界值（1x1 kernel、最小空间尺寸、kernel>输入）
- **可跳过的平台相关用例**：native 测试在未构建环境自动 skip，不阻塞其余平台无关用例

## 测试报告

`python scripts/gen_report.py reports/junit.xml docs/test-report.md` 生成 Markdown 测报：分模块用例数/通过率/耗时与整体结论。CI 中每次推送自动产出。

## 环境变量

无必需环境变量。模型 checkpoint（`artifacts/tiny_cnn.pt`）与 ONNX（`artifacts/model.onnx`）已提交仓库；如需重训：`python scripts/train_model.py`。
