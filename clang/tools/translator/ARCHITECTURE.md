# DACPP Translator AscendC 架构说明

## 1. 文档范围

本文档聚焦 DACPP 到 AscendC 的翻译子系统，覆盖以下目录：

- [translator.cpp](translator.cpp)
- [parser/include](parser/include)
- [parser/lib](parser/lib)
- [rewriter/lib/Rewriter_AscendC.cpp](rewriter/lib/Rewriter_AscendC.cpp)
- [rewriter/lib/ascendc](rewriter/lib/ascendc)
- [rewriter/include/ascendc](rewriter/include/ascendc)
- [rewriter/lib/ascendc_mlir](rewriter/lib/ascendc_mlir)
- [rewriter/include/ascendc_mlir](rewriter/include/ascendc_mlir)

不覆盖 LLVM/Clang 其他通用子系统实现细节。

## 2. 一句话架构

系统采用“三层管线”：

1. 解析层：Clang AST 匹配 -> DACPP 表达式抽取 -> Shell/Calc 解析 -> DacppFile
2. 语义层：Calc AST -> MLIR -> 表达式图 -> 语义分类 -> AscendC 计划
3. 代码层：基于 AscendC 计划生成 kernel 代码；HostFlowBuilder 从 AST 直接构造 `AscendCFlowPlan`，HostFlowEmitter 按结构化 step 顺序发射 host 执行流（不保留基于源码回放的路径）

前端解析入口是 [DacHandler::run](translator.cpp#L142)，重写入口是 [rewriteDac_AscendC](rewriter/lib/Rewriter_AscendC.cpp)；实际 AscendC 后端调度在 [RewriteAscendC](rewriter/lib/ascendc/AscendCRewriteDriver.cpp)。

## 3. 端到端流程

### 3.1 调度总览

1. ASTMatcher 在 [translator.cpp](translator.cpp) 中匹配 `dac_expr`、`main`、`dac_for` 等关键节点。
2. 匹配回调 [DacHandler::run](translator.cpp#L142) 对顶级 DAC 表达式做过滤与去重。
3. 对每个顶级表达式调用 [DacppFile::setExpression](parser/lib/DacppStructure.cpp#L584)，构建 Expression 树。
4. 在 setExpression 内部分别触发 [Shell::parseShell](parser/lib/Shell.cpp#L494) 与 [Calc::parseCalc](parser/lib/Calc.cpp#L4866)。
5. 进入 [rewriteDac_AscendC](rewriter/lib/Rewriter_AscendC.cpp)，并转调 [RewriteAscendC](rewriter/lib/ascendc/AscendCRewriteDriver.cpp)，扫描每个 Expression，判断走 Vector 路径还是 Cube 路径。
6. Vector 路径进入 MLIR 管线，产出计划后生成 custom kernel。
7. Cube 路径验证 matmul 语义后生成 cube kernel 与 tiling。
8. HostFlowBuilder 从 owner function 的 AST 抽取 host 主体语句、tensor alias、prelude 上下文，组装结构化 `AscendCFlowPlan`；每个 DAC 表达式对应一个 `FlowKernelLaunch` step。单卡附带 `CommonDecl` / `DeviceAlloc` / `HostToDeviceCopy` / `KernelLaunch` / `DeviceToHostCopy` / `DeviceFree` role；多卡按 route 生成一次 `MulticardRouteSetup`、每个表达式一个 `MulticardLaunch`、末尾一次 `MulticardRouteCleanup`。
9. HostFlowEmitter 按 `plan.steps` 顺序结构化发射 host main body 字符串；用户源码 Stmt 不参与 host body（替换为按字段重建）。
10. `HostRuntimeEmitter` 按 feature mask 生成一批次共享的 Host runtime，再输出
    host、kernel、tiling 文件。

### 3.2 核心输出文件

- Host: `host0.cpp`, `host1.cpp`, ...（按 route 产物递增）
- Kernel: `kernel0.cpp`, `kernel1.cpp`, ...（Vector/Cube 共享全局产物编号，按表达式产物递增）
- Cube tiling: tiling.cpp
- 默认 compact Host runtime：`dacpp_ascendc_host_runtime.h`（同一翻译批次只写一次，
  所有 `hostN.cpp` 通过带版本断言的 quoted include 共享）；embedded 调试模式不写该文件。

## 4. parser 子系统逐文件说明

### 4.1 parser 在整体中的位置

职责：把 AST 级 DAC 表达式转换为 rewriter 可直接消费的中间结构 DacppFile/Expression。

关键连接点：

- [DacHandler::run](translator.cpp#L142)：匹配回调入口，负责触发 parser。
- [DacppFile::setExpression](parser/lib/DacppStructure.cpp#L584)：构建 Expression，并调用 Shell/Calc 解析。
- [setMainFuncLoc](translator.cpp#L203)：记录主函数位置供后续重写使用。

### 4.2 [ASTParse.cpp](parser/lib/ASTParse.cpp)

职责：提供 parser 通用 AST 工具能力。

- split 表达式递归提取。
- 参数输入/输出属性判定。

关键接口：

- [getSplitExpr](parser/lib/ASTParse.cpp)
- [inputOrOutput](parser/lib/ASTParse.cpp)

### 4.3 [DacppStructure.cpp](parser/lib/DacppStructure.cpp)

职责：维护 parser 的核心中间结构，并完成表达式级装配。

- 管理 Expression/DacppFile 等对象。
- 提取实参 shape，并构建 Expression 的 Shell 与 Calc 子结构。
- 提供 parser 结果调试输出。

关键接口：

- [DacppFile::setExpression](parser/lib/DacppStructure.cpp#L584)
- [printParserResult](parser/lib/DacppStructure.cpp#L665)

### 4.4 [Param.cpp](parser/lib/Param.cpp) 与 [Split.cpp](parser/lib/Split.cpp)

职责：定义参数与划分模型，是 Shell/Calc 解析的数据底座。

- Param 负责读写属性、类型、基础类型、shape/dim。
- Split/IndexSplit/RegularSplit 负责维度划分信息表达。

关键接口：

- [Param::setType](parser/lib/Param.cpp#L67)

### 4.5 [Shell.cpp](parser/lib/Shell.cpp)

职责：解析 shell 侧数据布局与划分绑定关系。

- 遍历 shell 函数体，识别 list/split/index 相关结构。
- 建立绑定图与偏移信息。
- 生成 ShellParam 列表（含 shape、rw、split）。

关键接口：

- [Shell::parseShell](parser/lib/Shell.cpp#L494)

### 4.6 [Calc.cpp](parser/lib/Calc.cpp)

职责：解析 calc 侧计算语义并抽取可重写信息。

- 解析 calc 函数引用、参数、规则注解。
- 提取 body 与 block 切分结果。
- 支持嵌套 Expression 递归。

关键接口：

- [Calc::parseCalc](parser/lib/Calc.cpp#L4866)

### 4.7 [Dacfor.cpp](parser/lib/Dacfor.cpp)

职责：承载 `dac_for` 场景的结构化控制语义。

- 使用 ParamControl 表达参数间结构操作（如 swap/copy）。
- 使用 ControlBlock 持有 loop bound 与参数控制序列。

关键接口：

- [ControlBlock::addParamControl](parser/lib/Dacfor.cpp#L38)
- [ControlBlock::setLoopBound](parser/lib/Dacfor.cpp#L42)

### 4.8 parser 输出到 rewriter 的交接

交接结果是 DacppFile 中的 Expression 序列：

1. 每个 Expression 同时持有 Shell 与 Calc。
2. Shell 提供参数形状、rw、split/window 信息。
3. Calc 提供函数体与表达式结构。
4. rewriter 在 [RewriteAscendC](rewriter/lib/ascendc/AscendCRewriteDriver.cpp) 中按 Expression 消费这些信息。

## 5. ascendc_mlir 子系统逐文件说明

### 5.1 [CalcContext.cpp](rewriter/lib/ascendc_mlir/CalcContext.cpp)

职责：构建 Calc 级别语义上下文。

- 提取数值常量与可纯内联函数。
- 维护未解析符号集合。
- 提供常量表达式求值能力，供 Lowering 与 Host 绑定复用。

关键接口：

- [ExtractCalcContext](rewriter/lib/ascendc_mlir/CalcContext.cpp#L708)
- [TryEvaluateConstExpr](rewriter/lib/ascendc_mlir/CalcContext.cpp#L733)

### 5.2 [CalcShellInfo.cpp](rewriter/lib/ascendc_mlir/CalcShellInfo.cpp)

职责：把 Shell 参数语义显式化。

- 解析每个参数的读写属性、rank、shape。
- 归一化 split/window 信息（含 halo）。
- 识别广播标量参数。

关键接口：

- [BuildCalcShellSummary](rewriter/lib/ascendc_mlir/CalcShellInfo.cpp#L193)
- [DumpCalcShellSummary](rewriter/lib/ascendc_mlir/CalcShellInfo.cpp#L279)

### 5.3 [CalcAstToMlir.cpp](rewriter/lib/ascendc_mlir/CalcAstToMlir.cpp)

职责：把 Calc 的 Clang AST 降级为 MLIR。

- 支持赋值、复合赋值、for、if、数组/下标访问。
- 支持部分 inline call 与 const 折叠。
- 生成可验证的 MLIR module。

关键接口：

- [LowerCalcAstToMlir](rewriter/lib/ascendc_mlir/CalcAstToMlir.cpp#L2091)

### 5.4 [CalcExprGraph.cpp](rewriter/lib/ascendc_mlir/CalcExprGraph.cpp)

职责：从 MLIR 构建表达式图。

- 抽取 Load/Const/Add/Sub/Mul/Div/Exp/Clamp/Store 节点。
- 识别 load 的 offset 与窗口基准信息。
- 生成后续分析可用的图结构。

关键接口：

- [BuildCalcExprGraph](rewriter/lib/ascendc_mlir/CalcExprGraph.cpp#L561)
- [RenderCalcExprGraph](rewriter/lib/ascendc_mlir/CalcExprGraph.cpp#L641)

### 5.5 [CalcAnalysis.cpp](rewriter/lib/ascendc_mlir/CalcAnalysis.cpp)

职责：语义识别与模式归类。

- 将表达式图分类为 Pointwise、SpatialStencil、TemporalStencil、CubeMatmul、Reduction、Fallback。
- 提取结构化模式：
  - StructuredShiftedMapUpdatePattern
  - StructuredMultiViewExprPattern
- 产出 CalcSemanticFinding 列表。

关键接口：

- [AnalyzeCalcMlirModule](rewriter/lib/ascendc_mlir/CalcAnalysis.cpp#L2308)
- [RenderCalcAnalysisPlan](rewriter/lib/ascendc_mlir/CalcAnalysis.cpp#L2596)

### 5.6 [AscendCPlan.cpp](rewriter/lib/ascendc_mlir/AscendCPlan.cpp)

职责：把语义结果转换成可发射计划。

- 把 finding 映射为 AscendCPlanStep 序列。
- 为每步确定 transferPolicy、windowOffsets、halo、payload。
- 处理 CubeMatmul 维度与类型映射策略。

关键接口：

- [BuildAscendCPlan](rewriter/lib/ascendc_mlir/AscendCPlan.cpp#L1574)
- [DumpAscendCPlan](rewriter/lib/ascendc_mlir/AscendCPlan.cpp#L1715)

### 5.7 [AscendCEmitter.cpp](rewriter/lib/ascendc_mlir/AscendCEmitter.cpp)

职责：按计划发射 AscendC 代码。

- 发射 pointwise、stencil、clamp、shifted elementwise 等阶段。
- 支持 structured expression fast path。
- 对不可安全发射场景回退到 scalar fallback。

关键接口：

- [EmitAscendCFromPlan](rewriter/lib/ascendc_mlir/AscendCEmitter.cpp#L1919)

### 5.8 [CalcMlirPipeline.cpp](rewriter/lib/ascendc_mlir/CalcMlirPipeline.cpp)

职责：统一编排完整 MLIR 管线。

- 解析命令行与环境变量开关。
- 串联 Context -> Lower -> Analysis -> Plan -> Emit。
- 在失败场景按策略进行 fallback 或 hard fail。

关键接口：

- [ResolveCalcMlirPipelineOptions](rewriter/lib/ascendc_mlir/CalcMlirPipeline.cpp#L133)
- [RunCalcMlirPipeline](rewriter/lib/ascendc_mlir/CalcMlirPipeline.cpp#L153)

## 6. AscendC 后端工作明细

[rewriter/lib/Rewriter_AscendC.cpp](rewriter/lib/Rewriter_AscendC.cpp) 已退化为薄入口，
仅转调 `dacppTranslator::ascendc::RewriteAscendC`。所有 AscendC 后端实现已经按职责拆分到
[rewriter/lib/ascendc/](rewriter/lib/ascendc) 与 [rewriter/include/ascendc/](rewriter/include/ascendc)
下的若干真实 `.h/.cpp` 模块，不再使用 `.inc` 片段。driver 主循环通过 artifact-returning façade
处理 Vector/Cube lowering，并把 tensor alias 解析/收集委托给 tensor alias 模块。

### 6.1 主入口层

入口：

- [rewriteDac_AscendC](rewriter/lib/Rewriter_AscendC.cpp)：外部公开调用入口，仅做一次转调。
- [RewriteAscendC](rewriter/lib/ascendc/AscendCRewriteDriver.cpp)：实际后端总驱动。

主要工作：

1. 构建 AscendCRewriteContext，并用 RuntimeBindingContextGuard 管理运行时绑定上下文。
2. 识别 temporal schedule（需要时启用结构化桥接）。
3. 遍历全部 Expression，执行 Vector/Cube 路由；每个表达式产出一个
   `AscendCExpressionArtifact`，含用户 prelude、声明式 runtime/name metadata、
   `FlowKernelLaunch` descriptor 与 role-split 的 `FlowGeneratedCodeBlock` 列表。
4. 调用 [`BuildHostFlowPlanFromAst`](rewriter/lib/ascendc/HostFlowBuilder.cpp) 把全部 expression artifact + 用户 main 体 AST 装配成 `AscendCFlowPlan`，写出 `ascendc_flow_plan.txt`。
5. 调用 [`RenderHostFlowBody`](rewriter/lib/ascendc/HostFlowEmitter.cpp) 按 plan 顺序结构化发射 host main body；任何 builder/emitter Error 都让 host 文件进入 `BuildHostGenerationErrorBody`，不允许半成品 host 输出。
6. 把 main body 拼进 vector/cube host 模板并连同 kernel/tiling 一起写盘。

### 6.2 模块划分

公开接口位于 [rewriter/include/ascendc/](rewriter/include/ascendc)，实现位于
[rewriter/lib/ascendc/](rewriter/lib/ascendc)。模块清单与职责如下：

- [AscendCRewriteDriver.h](rewriter/include/ascendc/AscendCRewriteDriver.h) / [.cpp](rewriter/lib/ascendc/AscendCRewriteDriver.cpp)：
  后端总驱动。承载 Vector/Cube 路由、Expression 遍历、`AscendCFlowPlan` 构造与 host body
  渲染编排，以及最终的 host/kernel/tiling 产物写出。仍是体量最大的实现文件，主要原因是
  OperatorGenerator/CubeOperatorGenerator 类体目前位于此处，后续会继续向下分拆。
- [AscendCArtifacts.h](rewriter/include/ascendc/AscendCArtifacts.h)：
  per-expression `AscendCExpressionArtifact`（kernel/tiling 文本、用户 prelude、
  `HostRuntimeRequirements`、名称审计记录以及 `FlowKernelLaunch` /
  `FlowGeneratedCodeBlock` descriptor）；公共 runtime 类型没有文件级文本通道。
- [HostTensorAliasInfo.h](rewriter/include/ascendc/HostTensorAliasInfo.h)：
  host 侧 tensor alias 的 shape/stride/offset 元数据，供 driver / HostFlowBuilder /
  HostFlowEmitter 共享。
- [AscendCFlowPlan.h](rewriter/include/ascendc/AscendCFlowPlan.h) / [.cpp](rewriter/lib/ascendc/AscendCFlowPlan.cpp)：
  `AscendCFlowPlan` 数据模型与 dump（`ascendc_flow_plan.txt`）。
- [HostFlowBuilder.h](rewriter/include/ascendc/HostFlowBuilder.h) / [.cpp](rewriter/lib/ascendc/HostFlowBuilder.cpp)：
  从 owner function AST 抽取 host inputs / tensor buffers / loops / branches / helper
  invocations / outputs / cleanups，组装结构化 `FlowStep` 序列。
- [HostFlowEmitter.h](rewriter/include/ascendc/HostFlowEmitter.h) / [.cpp](rewriter/lib/ascendc/HostFlowEmitter.cpp)：
  按 `plan.steps` 与 `FlowKernelLaunch.{commonDecl,alloc,copyIn,copyOut,free}StepIds`
  顺序发射 host main body 字符串；`FlowDiagnostic(Error)` 与 `FlowStepKind::Raw` 一律
  视为硬失败，`result.ok=false`。
- [HostCodeNames.h](rewriter/include/ascendc/HostCodeNames.h) / [.cpp](rewriter/lib/ascendc/HostCodeNames.cpp)：
  收集 owner/TU 可见标识符，按 route/expression scope 确定性分配短名，并把来源类别、
  冲突结果和 20/24 字符硬上限写入名称审计。
- [HostRuntimeFeatures.h](rewriter/include/ascendc/HostRuntimeFeatures.h)：
  定义 single/multicard、tiling、FP16、print 等 feature bit，以及 compact/embedded
  packaging 和日志级别；artifact 与 flow plan 只传声明式需求。
- [HostRuntimeEmitter.h](rewriter/include/ascendc/HostRuntimeEmitter.h) / [.cpp](rewriter/lib/ascendc/HostRuntimeEmitter.cpp)：
  唯一生成版本 19 的 C++11 Host runtime；模板只接收 insertion，不得复制 executor、
  ACL 生命周期或名称策略。
- [AscendCPolicy.h](rewriter/include/ascendc/AscendCPolicy.h) / [.cpp](rewriter/lib/ascendc/AscendCPolicy.cpp)：
  翻译策略与 AscendCRewriteContext。
- [AscendCSourceUtils.h](rewriter/include/ascendc/AscendCSourceUtils.h) / [.cpp](rewriter/lib/ascendc/AscendCSourceUtils.cpp)：
  源文本读取、片段提取与简单常量解析工具（用于 kernel/tiling 表达式抽取，不参与 host body 发射）。
- [AscendCTypeMapping.h](rewriter/include/ascendc/AscendCTypeMapping.h) / [.cpp](rewriter/lib/ascendc/AscendCTypeMapping.cpp)：
  host/kernel/tiling 各侧的 dtype 映射。
- [AscendCRuntimeBinding.h](rewriter/include/ascendc/AscendCRuntimeBinding.h) / [.cpp](rewriter/lib/ascendc/AscendCRuntimeBinding.cpp)：
  runtime binding 共享类型、record/AoS layout、tensor alias key 与类型识别 helper。
- [AscendCTensorAlias.h](rewriter/include/ascendc/AscendCTensorAlias.h) / [.cpp](rewriter/lib/ascendc/AscendCTensorAlias.cpp)：
  tensor alias 的 rank/type/stride/offset 解析、AST alias resolution、`TensorAliasAccessPlan`
  与 live alias collection。
- [AscendCKernelSupport.h](rewriter/include/ascendc/AscendCKernelSupport.h) / [.cpp](rewriter/lib/ascendc/AscendCKernelSupport.cpp)：
  MLIR pipeline facade 与 kernel 发射所需的支撑工具。

子目录的最新拆分情况另见 [rewriter/lib/ascendc/README.md](rewriter/lib/ascendc/README.md)。

后续拆分目标：把仍在 `AscendCRewriteDriver.cpp` 内的 OperatorGenerator /
CubeOperatorGenerator 类体进一步拆成独立的 Vector lowering / Cube lowering 编译单元。

### 6.3 HostFlow 主路径（采集 → plan → 发射）

host body 完全由 HostFlow 三段式管线驱动；用户源码 Stmt 不参与 host body 输出：

1. **采集**：[`BuildHostFlowPlanFromAst`](rewriter/lib/ascendc/HostFlowBuilder.cpp)（其中
   `CollectHostPreludeFromAst` 负责 helper / const / global / record / using decls 与
   tensor alias 的汇总）在 owner function 的 AST 上抽取下列 step kind：

   - `FlowHostInput` / `FlowTensorBuffer`：非 DAC 容器声明（标量常量、std::vector、
     dacpp 容器壳等）。
   - `FlowHostLoop` / `FlowHostBranch`：用户控制流（按 `bodyStepIds` / `thenStepIds` /
     `elseStepIds` 递归装配）。
   - `FlowHelperInvocation`：host 侧 helper / vector 成员调用。
   - `FlowHostOutput`：`std::cout` / `printf` / `.print()` / tensor 赋值 / scalar
     compound assignment；alias materialization 走 `generatedCode` 字段。
   - `FlowCleanupStep`：`delete` / `aclrtFree` / `aclrtDestroyStream` / `aclFinalize`。
   - `FlowKernelLaunch`：每个 DAC `<->` 表达式一个，含 route / kernel 名 / blockDim /
     paramOrder / scalarParams；其 `FlowRuntimeBuffer` 用稳定整数 `BufferId` 描述 access、
     distribution、halo、stride 和 producer residency，并通过 `commonDeclStepIds` 等索引
     引用只含用例薄代码的 role-split `FlowGeneratedCodeBlock`。
   - `FlowDiagnostic`：builder 内任意分类失败都通过 Error diagnostic 报告。

2. **plan dump**：`DumpAscendCFlowPlan` 把整个 plan 序列化到 `ascendc_flow_plan.txt`，
   同时保存 route runtime names、`runtimeFeatureMask`、BufferId/reuse 审计，供
   `atest_diff.sh` 与 `atest_hostflow_arch_gate.sh` 校验。

3. **发射**：[`RenderHostFlowBody`](rewriter/lib/ascendc/HostFlowEmitter.cpp) 按
   `plan.steps` 顺序结构化发射 host main body 字符串。`FlowStepKind::Raw` / Error
   diagnostic / `plan.partial == true` 都立即把 `result.ok = false`，driver 据此切到
   `BuildHostGenerationErrorBody`，**不允许半成品 host 输出**。

### 6.4 Host runtime blocks

单卡 `FlowKernelLaunch` 仍按
`commonDecl → alloc → copyIn → launch → copyOut → postLaunch → free` 消费薄 block，
但 ACL 生命周期和错误检查统一属于 `dacpp_host::Device`。参数 staging 使用
`HostBuffer<T>`；`Device::allocate/copyIn/copyOut/sync/release/close` 把每个 ACL 返回值
写入同一个 `Session`。CPU Debug 分支保留工具链的 GM API，不改变 kernel ABI。

公共 runtime 的契约是：

- `Session` / `Status` 实现 first-error，保存首个 `stage/code/rank/detail`；后续失败不得覆盖首因。
  `--ascendc-host-log=none|error|debug` 控制最终日志，默认至多输出一行 fatal。
- `BufferSpec` 与 `FlowRuntimeBuffer` 对应；runtime 只以整数 `BufferId` 索引
  `std::vector`，不使用字符串 key、`std::map` 或 `std::set`。
- `Route::prepare` 一次接收整个 expression 的 descriptor 数组，批量建立连续 rank
  Host slab；`host/device/copyIn/copyOut/copyDevice/sync/run/close` 复用已建立的资源。
- `RankSpan/RankArgs` 表达 rank 分片；Host 不再为每个 expression 生成 worker 类型或把
  所有参数机械展开进长期对象。

多卡采用 route ownership 三段式：`MulticardRouteSetup` 每个 route 恰好一次构造
`dacpp_host::Route`，每个 `MulticardLaunch` 提交一个 rank-local command，
`MulticardRouteCleanup` 恰好一次 stop/join、释放缓存并 finalize。每个 rank 的
context/stream 只创建一次，Device buffer 按容量复用；active rank 为 1 时在主线程走同一
`_n` artifact 快路径而不创建线程。普通 command 保持 async H2D → launch → async D2H
顺序，pointwise/temporal 的边界同步与 residency 拓扑不变。emitter 继续 hard-check
setup/cleanup/command 基数并拒绝 legacy `MulticardRuntime`。

### 6.5 Vector 路径

核心类 `OperatorGenerator` 目前位于
[AscendCRewriteDriver.cpp](rewriter/lib/ascendc/AscendCRewriteDriver.cpp)，关键阶段：

1. `ParseParameters`：运行时绑定（依赖 [AscendCRuntimeBinding.cpp](rewriter/lib/ascendc/AscendCRuntimeBinding.cpp)）、shape 解析、broadcast/AoS 处理。
2. `RunCalcMlirPipelineDetailed`：通过 [AscendCKernelSupport.cpp](rewriter/lib/ascendc/AscendCKernelSupport.cpp) 调用 ascendc_mlir 总管线。
3. `GenerateKernelFile`：生成 custom kernel 与 skeleton 组装。
4. `BuildKernelLaunchDescriptor`：填 kernelName / kernelArtifactFileName / blockDimExpr /
   paramOrder / scalarParams。
5. `BuildRuntimeBlocks`：单卡按 §6.4 拆 role；参数使用具体 staging struct，输入填充、
   record/AoS 字段拆分与 D2H 回写均在生成时展开为具体循环。temporal / pointwise
   scalar loop 使用 `CommonDecl` + `TemporalLoopBody` + `DeviceFree`。
6. `BuildMulticardRuntimeBlocks`：按 pointwise/halo/AoS、linear output shard、scalar
   persistent loop、loop-carried temporal 四类策略生成 `MulticardLaunch` command；route
   setup/cleanup 由 HostFlowBuilder 统一建立。

#### 6.5.1 Record 部分写入

record/AoS 参数拆成字段级物理参数后，`GenerateKernelFile` 根据 AscendC plan 的 `dst`
集合识别 calc 未显式写入的输出字段。未写字段不允许从 output GM 回读；生成器只在
`HostParamPlan` 中查找同时满足以下条件的输入字段：具备读权限、record 类型相同、字段名
相同且 kernel dtype 相同。恰有一个候选时，通过零增量 `AscendC::Adds` 从 input local
tensor 复制到 output local tensor，因此单卡和多卡共用同一语义。没有候选或存在多个
候选时，translator 直接报告翻译期错误并要求 calc 显式赋值，不会猜测来源。该规则仅
覆盖 record 字段的部分写入，不定义普通 tensor 完全未写输出的行为。

#### 6.5.2 Vector UB 资源模型与运行时 tiling

[VectorTiling.h](rewriter/include/ascendc/VectorTiling.h) 与
[VectorTiling.cpp](rewriter/lib/ascendc/VectorTiling.cpp) 是独立的纯计算模块。生成 TQue/TBuf
的阶段同时登记 `VectorUbAllocation`，覆盖输入/输出队列、broadcast、halo 和 scratch；
模块不解析生成后的 C++ 字符串。资源估算把 queue depth、元素宽度、halo、multiplicity、
fixed bytes 和安全 LCM 对齐纳入同一个 `VectorKernelResourceModel`。

`BuildVectorKernelTilingPlan` 在 `UB budget - reserve` 内先尝试双缓冲，必要时自动降为
单缓冲；显式指定 1/2 时不降级。结果固定 `BUFFER_NUM`、`TILE_ALIGN_ELEMENTS` 和
`MAX_TILE_ELEMENTS`，并把预算与 allocation breakdown 同时写入 translator 日志和 kernel
顶部注释。配置优先级为 CLI、环境变量、默认值，入口定义在
[AscendCPolicy.cpp](rewriter/lib/ascendc/AscendCPolicy.cpp) 与 [translator.cpp](translator.cpp)：

- `--ascendc-ub-budget-bytes` / `DACPP_ASCENDC_UB_BUDGET_BYTES`
- `--ascendc-ub-reserve-bytes` / `DACPP_ASCENDC_UB_RESERVE_BYTES`
- `--ascendc-buffer-num=auto|1|2` / `DACPP_ASCENDC_BUFFER_NUM`
- `--ascendc-max-tile-elements` / `DACPP_ASCENDC_MAX_TILE_ELEMENTS`

Host 在最终逻辑长度和 blockDim 已知后，按 `coreNum * TILE_ALIGN_ELEMENTS` 扩展并清零
padding，再计算每核 `blockLength`、`tileLength` 和 `tileCount`；所有 `size_t` 到
`uint32_t` 的转换前都有溢出检查。单卡、CPU Debug 和多卡 rank 共用相同公式，多卡按
各 rank 的实际长度重新计算。kernel ABI 尾部统一为
`totalLength, coreNum, tileLength, tileCount`，Process 以 `currentTileLength` 处理尾 tile，
输出只写当前有效 tile。LinearReduction 的权重按 `inFeatures * currentTileLength` 分块，
不再按 totalLength 一次性占用 UB。Cube 路径及 Cube Tiling API 不参与这套流程。

### 6.6 Cube 路径

核心类 `CubeOperatorGenerator` 目前位于
[AscendCRewriteDriver.cpp](rewriter/lib/ascendc/AscendCRewriteDriver.cpp)，关键阶段：

1. 从 plan 提取 CubeMatmulStepPayload。
2. 绑定 A/B/C 与 M/N/K 维度来源（dtype 映射走 [AscendCTypeMapping.cpp](rewriter/lib/ascendc/AscendCTypeMapping.cpp)）。
3. 生成 cube kernel、tiling 函数、cube host 执行流。
4. `BuildCubeKernelLaunchDescriptor` + `BuildCubeRuntimeBlocks` 把 cube host 切成
   `CommonDecl` / `DeviceAlloc` / `HostToDeviceCopy` / `KernelLaunch` /
   `DeviceToHostCopy` / `DeviceFree` 6 块，与 Vector 常规路径保持一致。
5. `BuildDirectCubeMulticardRuntimeBlocks` 按 M 维均衡分片；A/C 使用 rank 局部字节数，
   B replicated，rank 局部 shape 以 `blockDim`（默认 8）生成 tiling。Cube command 复用
   `dacpp_host::Route` 的 rank context/stream、连续 pinned slab 与 Device capacity cache；不注入
   Vector buffer runtime。

### 6.7 模板组装与产物写出

模板来源：

- [rewriter/lib/ascendc_template.cpp](rewriter/lib/ascendc_template.cpp)
- [rewriter/lib/universal_template.cpp](rewriter/lib/universal_template.cpp)

主入口最终把 Vector 与 Cube 的块级代码分别装配并写盘。默认 compact 模式由
`HostRuntimeEmitter` 写一份 `dacpp_ascendc_host_runtime.h`，每个 Host 恰好 include 一次并
`static_assert` ABI version 19；`--ascendc-embed-host-runtime` 仅改变 packaging，仍调用同一
emitter。feature mask 决定单卡 `Device`、多卡 `Route`、tiling、FP16 和 print 能力；
单卡 runtime 不含 thread/cv，多卡 runtime 不含字符串 buffer map。kernel 继续作为独立产物，
compact Host 的完整包必须同时携带 companion header。
这里的 compact 只表示“共享 companion header”的打包方式，不表示压缩 C++ 源码；
`HostRuntimeEmitter` 必须保留 runtime 模板的换行和层级缩进。

### 6.8 生成代码的 C++ 语言基线

AscendC 后端生成的 `hostN.cpp`、companion/embedded Host runtime 与 `kernelN.cpp`
**必须严格兼容 ISO C++11，禁止发射
任何 C++14、C++17、C++20 或更新标准才提供的语法和标准库 API**。允许使用的上限
包括 C++11 的 `auto`、`decltype`、`constexpr` 变量、`using` 类型别名、普通 lambda
捕获、列表初始化、`nullptr` 以及 `<thread>`/`<mutex>` 等 C++11 标准库能力。

禁止项包括但不限于：泛型 lambda、lambda 初始化捕获、函数返回类型推导、
`decltype(auto)`、变量模板、结构化绑定、`if constexpr`、折叠表达式、概念/
`requires`、指定成员初始化，以及 `std::make_unique`、`std::optional`、
`std::string_view`、`std::filesystem`、`std::span` 等 C++14 以上标准库 API。
`__aicore__`、`__global__`、`GM_ADDR` 和 kernel launch 语法属于 AscendC 工具链扩展，
不放宽其余 C++ 代码的 C++11 上限。

translator 自身的构建标准、解析 DAC++ 输入时使用的 `-extra-arg` 与生成产物的语言
基线是三个独立概念，不得通过修改输入标准来代替产物检查。host 回归必须在 fake ACL
环境中使用 `g++ -std=c++11 -pedantic-errors -fsyntax-only`；kernel 回归应在可用的
AscendC 工具链中显式选择 C++11，并辅以对 C++14 以上语法和标准库符号的产物扫描。
`tools/atest_hostflow_arch_gate.sh` 通过 `ASCENDC_CPP11_KERNEL_CHECK` 调用版本相关的
CI wrapper；wrapper 接收一个生成 kernel 源文件并负责其 C++11 编译。未配置 wrapper
时闸门明确报告 skip，不得把源码扫描表述为 kernel 编译通过。

## 7. 关键数据流

### 7.1 Parser 数据流

Clang AST
-> [DacHandler::run](translator.cpp#L142)
-> [DacppFile::setExpression](parser/lib/DacppStructure.cpp#L584)
-> [Shell::parseShell](parser/lib/Shell.cpp#L494) + [Calc::parseCalc](parser/lib/Calc.cpp#L4866)
-> DacppFile/Expression
-> [rewriteDac_AscendC](rewriter/lib/Rewriter_AscendC.cpp)
-> [RewriteAscendC](rewriter/lib/ascendc/AscendCRewriteDriver.cpp)

### 7.2 语义数据流

Calc + Shell
-> CalcContextSummary + CalcShellSummary
-> Calc MLIR Module
-> CalcExprGraph
-> CalcSemanticFinding
-> AscendCPlan
-> AscendC API 代码片段

### 7.3 Host 数据流

Owner Function AST
-> `HostFlowBuilder::ClassifyTopLevelStmt`
-> `FlowStep` 序列（HostInput/TensorBuffer/HostLoop/HostBranch/HelperInvocation/HostOutput/Cleanup/KernelLaunch/Diagnostic）
-> `AscendCFlowPlan` + `ascendc_flow_plan.txt` dump
-> `RenderHostFlowBody`（按 step ids 顺序）
-> 可执行 Host Main Body（或 `BuildHostGenerationErrorBody` 错误体）

## 8. 主要开关与调试入口

MLIR 管线开关定义在 [CalcMlirPipeline.cpp](rewriter/lib/ascendc_mlir/CalcMlirPipeline.cpp)。

常用开关：

- `--ascendc-use-mlir`
- `--ascendc-dump-mlir`
- `--ascendc-dump-ascendc-plan`
- `--ascendc-require-mlir-recognized-cases`
- `--ascendc-dump-flow-plan`：把 HostFlow 的 `AscendCFlowPlan` dump 到 stdout（写
  `ascendc_flow_plan.txt` 是默认行为，与此 flag 无关）。
- `--ascendc-core-num=N` / `DACPP_ASCENDC_CORE_NUM`：每个 Device 的 AI Core launch
  数，默认 8，命令行优先于环境变量。
- `--ascendc-multicard`：启用多 Device Host runtime；没有 rank 覆盖时 compiled fallback
  为 8 个 Device。
- `--ascendc-rank-dim=N` / `DACPP_ASCENDC_RANK_DIM`：覆盖多卡 requested Device/rank 数。
  单卡模式始终为 1 Device；显式多卡 rank=1 仅用于调试快路径。生成 host 运行时的合法
  rank env 优先于 compiled fallback。blockDim 与 rankDim 是两级正交并行度，默认拓扑为
  多卡 `8 ranks × 8 AI Cores`、单卡 `1 × 8`。
- `--ascendc-embed-host-runtime`：把同一版本化 runtime 嵌入 Host，供单文件调试；默认
  compact 模式输出 companion header。
- `--ascendc-host-log=none|error|debug`：运行时日志级别；非法值是翻译期配置错误。

### 8.1 Host 生成回归

- `tools/atest_diff.sh` 对八个 tests 同时生成单卡和只传
  `--ascendc-multicard` 的默认八卡产物；单卡基准位于用例目录，多卡基准
  位于 `tests/<case>/multicard/`。Host/kernel/flow plan/companion header 均参与 diff；
  t1/t4/t7 另做 compact/embed ABI 等价检查。
- `tools/atest_hostflow_arch_gate.sh` 使用 fake ACL/launch header 对两种模式做 host
  与 companion header 的独立 C++11 语法检查；`atest_host_compact_gate.py` 机器可读地检查
  runtime include/version、短名/保留标识符、静态分支、ACL 封装、逐用例与总行数（runtime
  header 上限 450 行，八个多卡 Host 总上限 2890 行）。闸门还覆盖 invalid log 与
  compact/embed packaging。
## 9. 扩展建议

1. 新增 parser 语法/结构提取能力：优先改 [parser/lib/DacppStructure.cpp](parser/lib/DacppStructure.cpp)、[parser/lib/Shell.cpp](parser/lib/Shell.cpp)、[parser/lib/Calc.cpp](parser/lib/Calc.cpp)。
2. 新增语义模式：优先改 [CalcAnalysis.cpp](rewriter/lib/ascendc_mlir/CalcAnalysis.cpp) 与 [AscendCPlan.cpp](rewriter/lib/ascendc_mlir/AscendCPlan.cpp)。
3. 新增 kernel 发射策略：主要改 [AscendCEmitter.cpp](rewriter/lib/ascendc_mlir/AscendCEmitter.cpp)。
4. 新增 host 主体语句的结构化识别（新 `FlowStepKind` / 新 AST 模式）：改 [HostFlowBuilder.cpp](rewriter/lib/ascendc/HostFlowBuilder.cpp)；如需新增 `FlowStep` 字段或新 role，同步改 [AscendCFlowPlan.h](rewriter/include/ascendc/AscendCFlowPlan.h) 与 [HostFlowEmitter.cpp](rewriter/lib/ascendc/HostFlowEmitter.cpp) 的 dispatch。
5. 调整 role-split 边界（如把 `KernelLaunch` 内的 stream sync 进一步拆出）：改 driver 中 `OperatorGenerator::BuildRuntimeBlocks` / `CubeOperatorGenerator::BuildCubeRuntimeBlocks` 与 `EmitKernelLaunch` 中的消费顺序。
6. 新增 tensor alias 行为：改 [AscendCTensorAlias.cpp](rewriter/lib/ascendc/AscendCTensorAlias.cpp)；alias key/layout 共享类型与 driver / HostFlowBuilder 共用，相应同步改 [AscendCRuntimeBinding.cpp](rewriter/lib/ascendc/AscendCRuntimeBinding.cpp) 与 [HostTensorAliasInfo.h](rewriter/include/ascendc/HostTensorAliasInfo.h)。
7. 新增 runtime binding 类型或 dtype 映射：分别改 [AscendCRuntimeBinding.cpp](rewriter/lib/ascendc/AscendCRuntimeBinding.cpp) 与 [AscendCTypeMapping.cpp](rewriter/lib/ascendc/AscendCTypeMapping.cpp)。
8. 新增 buffer 行为时先扩展 `FlowRuntimeBuffer`/`BufferSpec` 和稳定 `BufferId`，再由
   builder/emitter 消费；新增 runtime 能力必须登记 `HostRuntimeFeature`，禁止向模板塞另一份
   raw executor 或 ACL 生命周期字符串。

## 10. 维护注意事项

1. 语义识别与发射策略必须一一对应，避免“识别成功但计划不可发射”。
2. `AscendCFlowPlan.steps` 的顺序是 host 语义正确性的关键：HostFlowBuilder 必须按 owner
   function AST 源码顺序生成 step；driver 在写完 `ascendc_flow_plan.txt` 后会做主流程
   错误检查，`plan.partial==true` 或含 Error diagnostic 时直接切到
   `BuildHostGenerationErrorBody`，**禁止**输出半成品 host 文件。
3. 用户源码 Stmt 不允许作为 host body 输出：所有 `FlowHostInput` / `FlowTensorBuffer` /
   `FlowHostLoop` / `FlowHostBranch` / `FlowHelperInvocation` / `FlowCleanupStep` 都按
   结构化字段重建。仅 `FlowHostOutput::generatedCode` 与 `FlowGeneratedCodeBlock::code`
   能携带成段 translator-generated 文本（alias materialization、ACL/runtime/kernel-launch）。
4. `FlowStepKind::Raw` 是 diagnostic-only sentinel，emitter 见到即 hard fail；任何
   "RawStmt 兜底"路径都会被 `tools/atest_hostflow_arch_gate.sh` 闸门捕获，禁止重新引入。
5. role-split 的 `FlowGeneratedCodeBlock` 顺序由 `FlowKernelLaunch.{commonDecl,alloc,
   copyIn,copyOut,free}StepIds` + `role == KernelLaunch|TemporalLoopBody && ownerExpr 匹配`
   共同决定；修改 builder/driver 拼装时必须保证 emit 时 ACL 顺序
   `alloc → H2D → launch → D2H → free` 不变。
6. `FlowHostLoop` 的 for-init 使用结构化字段（`initIsDecl`/`initIsAssignment` + type/name/value），
   不保存完整 init stmt 源文本。`HostFlowBuilder.cpp` 中不允许 `GetStmtSourceText`。
7. 不可结构化的 host stmt（如 `goto`、复杂 control flow）由 builder 通过
   `FlowDiagnostic(Error)` 报告，host 文件进入 generation error body，不允许源码回放。
8. `BufferId` 在同一 route 内按表达式/参数语义稳定分配；resident reuse 只传整数 ID，
   不得恢复字符串 cache key。
9. 每个 route 恰好一次 setup/cleanup，每个 expression 恰好一次 command；`Session`
    只保留第一个失败，生产 Host 不加入 benchmark repeat。
10. 任何 fallback 都应在 plan 或日志中保留 reason，便于定位。
