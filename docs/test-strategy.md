# AI 基础软件栈测试策略

## 1. 被测对象与风险分析

被测对象为 AI 基础软件栈的四层，自底向上：

| 层 | 被测对象 | 核心风险 | 测试手段 |
|---|---|---|---|
| L1 算子库 | conv2d / matmul / softmax / batchnorm2d（numpy 实现） | 数值误差、形状处理、异常输入、NaN 传播 | 与 PyTorch oracle 数值对照 + 边界形状矩阵 + 异常注入 |
| L2 模型 | TinyCNN（合成数据 5 分类） | 精度不达标、不可复现、低精度格式失真、批量推理不一致 | 精度阈值准出 + 确定性验证 + fp32/bf16 容限对照 + 批量 vs 单条一致性 |
| L3 编译器 | 图 IR + 常量折叠 / conv+relu 融合 / DCE | 优化改变语义（最致命）、非法图未拦截 | pass 级单测 + 编译前后执行等价性质断言 + 异构图参数化拒绝 |
| L4 调度器 | DAG 贪心调度（资源预算 + 优先级） | 死锁/活锁、约束违反、结果不确定 | 依赖/资源/优先级性质测试 + 环与超预算异常 + 200 节点随机 DAG 程序化验证 |
| L5 原生层 | C++ matmul kernel | FFI 边界错误、跨语言 ABI | ctypes 加载对照 numpy，未构建平台自动 skip |

## 2. 用例设计方法

- **等价类划分**：conv 的 stride/padding 组合空间按 (stride=1/2) x (padding=0/1) 划分；调度任务按依赖深度/优先级/资源占用分域
- **边界值分析**：1x1 kernel、最小空间尺寸 (1,1)、kernel>输入、预算恰好装下/装不下、单任务 cost=预算
- **错误推测**：NaN/Inf 注入、重复 id、自环依赖、未知算子、空图、悬空引用、零方差
- **性质测试（property-based）**：编译前后语义等价、调度结果满足全部约束且确定性、随机 DAG 规模扩展
- **数据驱动**：`tests/test_data/conv_shapes.yaml` 形状矩阵，新增形状不改代码

## 3. 数值容限策略

| 对照组合 | 容限 | 依据 |
|---|---|---|
| numpy vs torch（fp64） | rtol=1e-9, atol=1e-8 | 同为双精度，理论误差仅在求和顺序级 |
| C++ fp32 vs numpy fp64 | rtol=1e-5, atol=1e-5 | fp32 舍入误差量级 |
| fp32 vs bfloat16 | 均值偏差 < 0.5 且 argmax 一致率 ≥ 99% | bf16 尾数 8 位，按 logits 量级放宽容限 |
| torch vs onnxruntime | rtol=1e-4, atol=1e-5 | 导出图算子实现差异 |

所有断言失败时输出 max_abs_err / mean_abs_err 统计，便于定位是系统性偏差还是个别元素。

## 4. 准入 / 准出

- 准入：ruff 通过；artifacts（checkpoint/onnx）存在；全量测试可离线复现
- 准出：0 failed；模型验证集精度 ≥ 0.95 且各类召回 ≥ 0.85；编译 pass 语义保持断言全过

## 5. 执行与报告

- `scripts/run_all.sh` 一键执行（构建 kernel → lint → 测试 → 测报）
- `scripts/gen_report.py` 产出 Markdown 测报（分模块统计 + 结论），CI 每次推送自动生成并归档
