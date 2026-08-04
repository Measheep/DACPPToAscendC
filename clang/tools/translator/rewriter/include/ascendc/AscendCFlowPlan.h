#ifndef TRANSLATOR_REWRITER_ASCENDC_FLOW_PLAN_H
#define TRANSLATOR_REWRITER_ASCENDC_FLOW_PLAN_H

// AscendCFlowPlan：HostFlow 结构化 host body 数据模型。
//
// 设计约束：
//   1. 这个头文件 **不得** 包含 MLIR / Clang / DACPP 任何头，
//      保证 `DACPP_TRANSLATOR_HAS_MLIR=0` 的构建也能使用 HostFlow。
//   2. 所有 host 发射所需的语义都用字符串/整型/枚举/索引承载；用户源码 Stmt 原文 **不得**
//      作为 host body 的发射内容存放。FlowRawStmt 仅供 dump/diagnostic 显示，禁止 emitter 消费。
//   3. translator 自己生成的 ACL/runtime/kernel launch 字符串则放进 FlowGeneratedCodeBlock，
//      与用户源码完全隔离。

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace dacppTranslator {
class DacppFile;
namespace ascendc {

// ============================================================
// 通用辅助类型
// ============================================================

struct FlowSourceLoc {
    std::string file;
    int line = -1;
    int column = -1;

    bool isValid() const { return line > 0; }
};

using FlowStepId = std::size_t;
static constexpr FlowStepId kInvalidFlowStepId = static_cast<FlowStepId>(-1);

using FlowExprIndex = int;
static constexpr FlowExprIndex kInvalidFlowExprIndex = -1;

using FlowBufferId = uint32_t;
static constexpr FlowBufferId kInvalidFlowBufferId = UINT32_MAX;

enum class FlowRouteKind {
    Unknown,
    Vector,
    Cube,
};

enum class FlowMulticardCommunicationKind {
    None,
    HostBoundary,
    PeerBoundary,
    AllGather,
    ReduceScatter,
    AllReduce,
};

enum class FlowMulticardParamDistribution {
    Sharded,
    Replicated,
};

struct FlowMulticardParamPartition {
    std::string name;
    FlowBufferId bufferId = kInvalidFlowBufferId;
    FlowMulticardParamDistribution distribution =
        FlowMulticardParamDistribution::Sharded;
    std::string shardAxis;
    int readHaloLeft = 0;
    int readHaloRight = 0;
    int readWriteMode = 0;
    bool deviceResident = false;
};

enum class FlowBufferAccess {
    Read,
    Write,
    ReadWrite,
};

struct FlowRuntimeBuffer {
    FlowBufferId id = kInvalidFlowBufferId;
    std::string logicalName;
    std::string elementTypeText;
    std::string logicalCountExpr;
    std::string byteCountExpr;
    std::string hostStrideExpr;
    FlowBufferAccess access = FlowBufferAccess::Read;
    FlowMulticardParamDistribution distribution =
        FlowMulticardParamDistribution::Sharded;
    int leftHalo = 0;
    int rightHalo = 0;
    bool producerResident = false;
};

enum class FlowGeneratedNameKind {
    FrameworkLocal,
    ParameterDerived,
    UserPreserved,
    PublicType,
};

struct FlowGeneratedName {
    std::string text;
    FlowGeneratedNameKind kind = FlowGeneratedNameKind::FrameworkLocal;
    FlowExprIndex ownerExpr = kInvalidFlowExprIndex;
};

struct FlowRouteRuntimeNames {
    FlowRouteKind route = FlowRouteKind::Unknown;
    std::string session;
    std::string rankCount;
    std::string routeObject;
    std::string device;
    std::string stream;
};

struct FlowMulticardConfig {
    bool enabled = false;
    // rankDimExpr remains as a compatibility alias for existing consumers.
    std::string rankDimExpr;
    std::string requestedRankDimExpr;
    std::string activeRankDimExpr;
    std::string coreNumExpr;
    std::string deviceMapExpr;
    std::string rankDimEnvName;
    std::string shardDim;
    std::string shardAxis;
    std::string logicalLengthExpr;
    std::string alignedLengthExpr;
    std::string strategy;
    std::string replicatedParams;
    std::string shardedParams;
    std::string resourceScope;
    std::string workerLifetime;
    std::string copyMode;
    FlowMulticardCommunicationKind communication =
        FlowMulticardCommunicationKind::None;
    std::vector<FlowMulticardParamPartition> paramPartitions;
    int readHaloLeft = 0;
    int readHaloRight = 0;
    bool inputReplicated = false;
    bool outputSharded = true;
    bool persistentWorkers = false;
};

// ============================================================
// 1. host 输入：标量/常量/普通指针等不属于 vector / dacpp tensor 的声明
// ============================================================
struct FlowHostInput {
    std::string name;             // 变量名，例如 "numIsotopes" / "T" / "WIDTH"
    std::string declTypeText;     // 原文类型，例如 "const float" / "size_t"
    std::string initializerText;  // 初始化表达式片段（可为空：tail-declared 变量）
    std::string storageClass;     // "static" / "extern" / "" — VarDecl 的存储类别
    std::string arrayExtentExpr;  // 数组维度表达式（如 "MAX_INPUTS"），非数组为空
    // 禁止整条 VarDecl 原文字段——emitter 必须按上面的结构化字段重建声明。
    bool isConst = false;
    bool isConstexpr = false;
    bool isHostGlobal = false;    // 函数体外的全局常量
    FlowSourceLoc location;
};

// ============================================================
// 2. tensor / 容器 buffer
// ============================================================
enum class FlowBufferOrigin {
    Unknown,
    StdVectorCtor,          // std::vector<T>(N, init)
    DacppSlice,             // 例如 u_tensor[{}][k] / new_rho_tensor[{1, WIDTH-1}]
    HelperReturnAlias,      // 由 helper 调用返回得到，rare
};

struct FlowTensorBuffer {
    std::string name;                     // host 端名字，例如 "vecC_tensor"
    std::string elementTypeText;          // 例如 "float" / "std::complex<float>" / "Pixel"
    std::string dacppContainerText;       // "Vector" / "Matrix" / "Tensor"（空表示是 std::vector）
    std::vector<std::string> shapeExprs;  // 形状表达式（字符串），可为空
    std::string sourceVectorName;         // 如果由 std::vector 构造，则记录源向量名
    FlowBufferOrigin origin = FlowBufferOrigin::Unknown;

    // 切片来源：例如 `new_rho_tensor[{1, WIDTH-1}]` 中的根 buffer
    std::string sliceRootBufferName;
    std::string sliceOffsetExpr;
    std::vector<std::string> sliceStrideExprs;

    // std::vector 初始化的结构化捕获（Step 2 会在 builder 中填充）。
    //   - 形如 `std::vector<T> v(n, init)`：ctorArgs = ["n", "init"]
    //   - 形如 `std::vector<T> v{a, b, c}`：initListElems = ["a", "b", "c"]
    //   - 形如 `std::vector<T> v = other`：copyFromVectorName = "other"
    // 同时只会有一个分支非空；都为空则 emitter 仅按 name/elementType 发声明（无初始化）。
    std::vector<std::string> ctorArgs;
    std::vector<std::string> initListElems;
    std::string copyFromVectorName;

    // 禁止整条 VarDecl 原文字段。
    FlowSourceLoc location;
};

// ============================================================
// 3. 标量绑定：kernel launch 需要的标量参数（broadcast tensor[{}] 之类）
// ============================================================
struct FlowScalarBinding {
    std::string name;            // tensor 端的名字（如 "t_tensor"）
    std::string dtypeText;       // 标量 dtype（host 侧）
    std::string valueExprText;   // 取值表达式（如 "t_tensor[0]"）
    FlowExprIndex ownerExpr = kInvalidFlowExprIndex;
    FlowSourceLoc location;
};

// ============================================================
// 4. kernel launch（Vector / Cube）
// ============================================================
// 注意：launch step 通过 *StepIds 引用一组与本次 launch 配套的 translator-generated
// runtime 代码（FlowGeneratedCodeBlock）。emitter 按下面字段的顺序消费：
//   commonDeclStepIds → allocStepIds → copyInStepIds → KernelLaunch 自身
//   → copyOutStepIds → freeStepIds
// 任何用户源码 Stmt 原文都不允许出现在这些 step 中。
struct FlowKernelLaunch {
    FlowRouteKind route = FlowRouteKind::Unknown;
    FlowExprIndex expressionIndex = kInvalidFlowExprIndex;
    std::string kernelFunctionName;       // e.g. "vecAdd_calc_do" / "matMul_cube_do"
    std::string kernelArtifactFileName;   // e.g. "kernel0.cpp"
    std::vector<std::string> paramOrder;  // tensor 实参顺序
    std::vector<std::string> scalarParams;
    std::string blockDimExpr;             // "1" / "AICORE_NUM" / 具体值
    std::string tilingDataSymbol;         // cube tiling 用，可空
    std::string launchDescriptorText;     // 简短摘要，用于 dump（builder 可灵活填）
    bool multicardEnabled = false;
    std::string rankLaunchFunctionName;
    std::vector<std::string> perRankParamOrder;
    std::vector<FlowBufferId> internalReuseBufferIds;
    std::vector<FlowRuntimeBuffer> runtimeBuffers;
    FlowMulticardConfig multicard;

    std::vector<FlowStepId> commonDeclStepIds;
    std::vector<FlowStepId> allocStepIds;
    std::vector<FlowStepId> copyInStepIds;
    std::vector<FlowStepId> copyOutStepIds;
    std::vector<FlowStepId> freeStepIds;

    // 紧跟 kernel launch、需要 runtime buffer 仍存活的 post-launch 步骤。
    // （如 tensor print）。emitter 在 D2H copy 后、free 前发射。
    std::vector<FlowStepId> postLaunchStepIds;

    FlowSourceLoc location;
};

// ============================================================
// 5. host 输出：打印、std::cout、helper print、tensor 之间赋值
// ============================================================
enum class FlowHostOutputKind {
    Unknown,
    TensorPrint,        // xxx_tensor.print();
    StdOutStream,       // std::cout << ...
    PrintfCall,         // printf(...) / cout-equivalent
    TensorAssign,       // 例如 A_tensor[10*t_tensor[0]] = local_A_tensor;
    ScalarSelfUpdate,   // 例如 t_tensor[0] += dt;
    VectorElemWrite,    // 例如 rho[i] = 40;
};

struct FlowHostOutput {
    FlowHostOutputKind kind = FlowHostOutputKind::Unknown;
    std::string targetExprText;
    std::string valueExprText;

    // StdOutStream：std::cout / std::cerr / 其它流名 + operator<< 链拆分得到的操作数序列。
    //   streamName 例如 "std::cout"；streamOperands 例如 ["\"x=\"", "x", "std::endl"]。
    std::string streamName;
    std::vector<std::string> streamOperands;

    // PrintfCall：printf / fprintf / 兼容形式的 callee 与按源序传递的参数文本。
    std::string printfCallee;
    std::vector<std::string> printfArgs;

    // ScalarSelfUpdate：复合运算符文本，如 "+=" / "-=" / "*=" / "/="。
    std::string compoundOp;

    // TensorPrint / TensorAssign / VectorElemWrite 的 alias materialization 输出 —
    // 严格要求：仅可保存 translator 生成（EmitTensorAliasAccessPlan 等）的代码片段，
    // 不得保存用户语句原文。emitter 命中时按 generatedCode 直出。
    std::string generatedCode;

    FlowSourceLoc location;
};

// ============================================================
// 6. 清理（free / delete / aclrtFree …）
// ============================================================
enum class FlowCleanupKind {
    Unknown,
    HostFree,           // free()
    HostDelete,         // delete / delete[]
    DeviceFree,         // aclrtFree(...)
    HostStreamDestroy,  // aclrtDestroyStream
    AclFinalize,
};

struct FlowCleanupStep {
    FlowCleanupKind kind = FlowCleanupKind::Unknown;
    std::string targetExprText;
    // 禁止整条 stmt 原文字段。emitter 仅按 kind + targetExprText 重建调用；
    // 无法识别的 cleanup 由 builder 通过 FlowDiagnostic(Error) 报告。
    FlowSourceLoc location;
};

// ============================================================
// 7. 诊断信息（fallback=strict 时由 driver 转 clang::DiagnosticsEngine）
// ============================================================
enum class FlowDiagnosticSeverity {
    Note,
    Warning,
    Error,
};

struct FlowDiagnostic {
    FlowDiagnosticSeverity severity = FlowDiagnosticSeverity::Note;
    std::string code;        // 短代码，例如 "hostflow.unsupported_stmt"
    std::string message;
    FlowExprIndex relatedExpr = kInvalidFlowExprIndex;
    FlowSourceLoc location;

    static FlowDiagnostic MakeError(const char* diagCode, std::string diagMessage) {
        FlowDiagnostic d;
        d.severity = FlowDiagnosticSeverity::Error;
        d.code = diagCode;
        d.message = std::move(diagMessage);
        return d;
    }
};

// ============================================================
// 8a. host for / while / do-while 循环
// ============================================================
enum class FlowHostLoopKind {
    For,
    While,
    DoWhile,
};

struct FlowHostLoop {
    FlowHostLoopKind kind = FlowHostLoopKind::For;
    std::string inductionVar;

    // for-init 结构化字段（initIsDecl / initIsAssignment 互斥）
    bool initIsDecl = false;
    bool initIsAssignment = false;
    std::string initStorageClass;     // "static" / "" 等
    std::string initTypeText;         // "int" / "size_t" 等
    std::string initVarName;          // 声明变量名
    std::string initValueExpr;        // 初始化表达式
    std::string initAssignTargetExpr; // assignment LHS
    std::string initAssignValueExpr;  // assignment RHS

    std::string upperBoundExpr;       // for 的条件表达式片段
    std::string stepExpr;             // for 的增量表达式片段
    std::string conditionText;        // while/do-while 的判断条件片段
    std::vector<FlowStepId> bodyStepIds;
    bool containsKernelLaunch = false;
    FlowSourceLoc location;
};

// ============================================================
// 8b. host if 分支
// ============================================================
struct FlowHostBranch {
    std::string conditionText;
    std::vector<FlowStepId> thenStepIds;
    std::vector<FlowStepId> elseStepIds;  // 空 = 无 else
    bool containsKernelLaunch = false;
    // 禁止整段 if 源码字段。emitter 按 conditionText + then/else step 重建。
    FlowSourceLoc location;
};

// ============================================================
// 9. helper 函数调用
// ============================================================
enum class FlowHelperRole {
    Unknown,
    InitOnly,          // initializeXxx / load / fill / prepare
    OutputMutating,    // calculate* / run* / dft / compute
    TileSizeHint,      // 计算 tile / block 尺寸
    PrintLike,         // PrintStats / dump
};

struct FlowHelperInvocation {
    std::string functionName;
    FlowHelperRole role = FlowHelperRole::Unknown;
    std::vector<std::string> argTexts;
    // 禁止整条 call stmt 原文字段。emitter 仅按
    //   [resultTargetName " = "] functionName "(" argTexts... ");"
    // 重建。
    std::string resultTargetName;     // 返回值赋给的变量名；空表示丢弃返回值
    bool returnsValue = false;        // 是否声明返回非 void
    FlowSourceLoc location;
};

// ============================================================
// 10. 跨 expression 的 buffer 依赖
// ============================================================
enum class FlowExprDataflowKind {
    Unknown,
    HostToDevice,      // host 准备数据 → 该 buffer 被 kernel 用作输入
    DeviceToHost,      // kernel 写出 → host 读取（包括 print 之前）
    InternalReuse,     // expression N 的 output 直接做 expression N+1 的 input
};

struct FlowExprDataflowEdge {
    FlowExprIndex producerExpr = kInvalidFlowExprIndex;
    FlowExprIndex consumerExpr = kInvalidFlowExprIndex;
    std::string bufferName;
    FlowExprDataflowKind kind = FlowExprDataflowKind::Unknown;
    FlowSourceLoc location;
};

// ============================================================
// 11. tensor alias 生命周期
// ============================================================
struct FlowAliasLifetime {
    std::string aliasName;
    std::string rootBufferName;
    std::string elementTypeText;
    int rank = 0;
    std::vector<std::string> shapeExprs;
    std::vector<std::string> strideExprs;
    std::string offsetExpr;
    FlowSourceLoc declLocation;
    FlowSourceLoc lastUseLocation;
};

// ============================================================
// 12. dac_for 嵌套（kernel launch 位于 host for/while 内部）
// ============================================================
enum class FlowDacForNestKind {
    Unknown,
    TimeStep,            // 例如 t2decay / decay_broadcast：while(t<T) kernel; t += dt;
    TemporalStencil,     // 例如 test13-temp / test14-FOuLa：for(k<n) k-th step
    SpatialIteration,    // 例如 t4liuliang：固定空间网格的多步演化
};

struct FlowDacForNesting {
    FlowStepId enclosingLoopStep = kInvalidFlowStepId;
    std::vector<FlowExprIndex> enclosedExpressions;
    FlowDacForNestKind kind = FlowDacForNestKind::Unknown;

    std::string stateTensorName;
    std::string inputSliceVar;
    int inputTimeOffset = 0;
    std::string outputSliceVar;
    int outputTimeOffset = 0;

    FlowSourceLoc location;
};

// ============================================================
// 13. translator 自身生成的 ACL/runtime/kernel-launch 代码块
// ============================================================
// 严格规则：FlowGeneratedCodeBlock.code 必须由 translator 直接生成，
// 不允许保存任何用户源码 Stmt 原文。任何"原文 replay"语义已被禁用。
enum class FlowGeneratedCodeRole {
    Unknown,
    AclRuntimeInit,        // aclInit / aclrtSetDevice / aclrtCreateContext
    AclRuntimeFinalize,    // aclrtResetDevice / aclFinalize
    StreamCreate,          // aclrtCreateStream
    StreamDestroy,         // aclrtDestroyStream
    CommonDecl,            // tilingData/host buffer 容器声明等纯生成声明
    DeviceAlloc,           // aclrtMalloc(... DEVICE ...)
    DeviceFree,            // aclrtFree
    HostToDeviceCopy,      // aclrtMemcpy(H2D)
    DeviceToHostCopy,      // aclrtMemcpy(D2H)
    KernelLaunch,          // ACLRT_LAUNCH_KERNEL / 等价宏 — 作为 FlowKernelLaunch 的尾随代码
    TemporalLoopBody,      // temporal / pointwise scalar loop 的 fused schedule body
    AuxiliaryPrelude,      // 其它 translator-generated 前导代码
    MulticardRouteSetup,   // route 级 ACL init、device preflight 与 executor 构造
    MulticardLaunch,       // 单个 expression 的 rank-local command
    MulticardRouteCleanup, // route 级 worker stop/join、buffer cleanup 与 ACL finalize
    MulticardRuntime,      // legacy per-launch 生命周期；emitter 明确拒绝
};

struct FlowGeneratedCodeBlock {
    FlowGeneratedCodeRole role = FlowGeneratedCodeRole::Unknown;
    std::string label;     // 简短标签（dump 用），如 "alloc.A" / "copy.h2d.A"
    std::string code;      // translator 生成的源码；**严禁** 保存用户源码 Stmt 原文
    FlowExprIndex ownerExpr = kInvalidFlowExprIndex;
    FlowRouteKind route = FlowRouteKind::Unknown;
    FlowSourceLoc location;
};

// ============================================================
// 顶层 step
// ============================================================
enum class FlowStepKind {
    None,
    HostInput,
    TensorBuffer,
    ScalarBinding,
    KernelLaunch,
    HostOutput,
    Cleanup,
    Diagnostic,
    HostLoop,
    HostBranch,
    HelperInvocation,
    ExprDataflowEdge,
    AliasLifetime,
    DacForNesting,
    GeneratedCodeBlock,   // translator-generated ACL/runtime 代码
    Raw,                  // FlowRawStmt：仅供 dump/diagnostic，禁止 emitter 消费
};

// FlowRawStmt 仅用于诊断，不得用于代码发射。
//   - builder 只能在生成 unsupported-stmt diagnostic 的同时填充（便于 dump 显示原文上下文）。
//   - emitter 绝不允许读取 FlowRawStmt::text 生成 host body；
//     driver 在 plan.steps 中看到 FlowStepKind::Raw 视为 hard error。
struct FlowRawStmt {
    std::string text;
    std::string ownerFunctionName;
    FlowSourceLoc location;
};

struct FlowStep {
    FlowStepKind kind = FlowStepKind::None;
    std::size_t payloadIndex = static_cast<std::size_t>(-1);
    std::string note;

    bool hasValidPayload(std::size_t poolSize) const {
        return payloadIndex != static_cast<std::size_t>(-1) && payloadIndex < poolSize;
    }
};

// ============================================================
// AscendCFlowPlan：聚合容器
// ============================================================
struct AscendCFlowPlan {
    // 元数据
    std::string sourceFileHint;
    bool builderRan = false;
    bool partial = false;
    int totalExpressions = 0;
    int vectorExpressionCount = 0;
    int cubeExpressionCount = 0;
    uint32_t runtimeFeatureMask = 0;
    std::vector<FlowGeneratedName> generatedNames;
    std::vector<FlowRouteRuntimeNames> routeRuntimeNames;

    // 节点池
    std::vector<FlowHostInput> hostInputs;
    std::vector<FlowTensorBuffer> tensorBuffers;
    std::vector<FlowScalarBinding> scalarBindings;
    std::vector<FlowKernelLaunch> kernelLaunches;
    std::vector<FlowHostOutput> hostOutputs;
    std::vector<FlowCleanupStep> cleanups;
    std::vector<FlowDiagnostic> diagnostics;
    std::vector<FlowHostLoop> hostLoops;
    std::vector<FlowHostBranch> hostBranches;
    std::vector<FlowHelperInvocation> helperInvocations;
    std::vector<FlowExprDataflowEdge> dataflowEdges;
    std::vector<FlowAliasLifetime> aliasLifetimes;
    std::vector<FlowDacForNesting> dacForNestings;
    std::vector<FlowGeneratedCodeBlock> generatedBlocks;
    std::vector<FlowRawStmt> rawStmts;

    // 主体：按源码顺序排列的 step 列表
    std::vector<FlowStep> steps;

    // ---- 便捷 append ----
    FlowStepId addHostInput(FlowHostInput n);
    FlowStepId addTensorBuffer(FlowTensorBuffer n);
    FlowStepId addScalarBinding(FlowScalarBinding n);
    FlowStepId addKernelLaunch(FlowKernelLaunch n);
    FlowStepId addHostOutput(FlowHostOutput n);
    FlowStepId addCleanup(FlowCleanupStep n);
    FlowStepId addDiagnostic(FlowDiagnostic n);
    FlowStepId addHostLoop(FlowHostLoop n);
    FlowStepId addHostBranch(FlowHostBranch n);
    FlowStepId addHelperInvocation(FlowHelperInvocation n);
    FlowStepId addDataflowEdge(FlowExprDataflowEdge n);
    FlowStepId addAliasLifetime(FlowAliasLifetime n);
    FlowStepId addDacForNesting(FlowDacForNesting n);
    FlowStepId addGeneratedCodeBlock(FlowGeneratedCodeBlock n);
    FlowStepId addRawStmt(FlowRawStmt n);
    void addGeneratedName(FlowGeneratedName n);
    const FlowRouteRuntimeNames* findRouteRuntimeNames(
        FlowRouteKind route) const;
};

// ============================================================
// Builder 入口（详细 AST/descriptor 实现在 HostFlowBuilder.cpp 等模块）
// ============================================================
struct AscendCFlowPlanBuildInput {
    DacppFile* dacppFile = nullptr;
    const std::string* sourceFileHint = nullptr;
    bool mixedRouteConflict = false;
    std::string mixedRouteReason;
};

AscendCFlowPlan BuildAscendCFlowPlan(const AscendCFlowPlanBuildInput& input);

// ============================================================
// 文本化：dump 给 --ascendc-dump-flow-plan 用，也供 atest_diff.sh 比对。
// 输出保证稳定（不含进程 id / 时间），方便 diff。
// ============================================================
std::string DumpAscendCFlowPlan(const AscendCFlowPlan& plan);

// 字符串化枚举（dump / 单测 / 错误信息共享）
const char* ToString(FlowRouteKind v);
const char* ToString(FlowMulticardCommunicationKind v);
const char* ToString(FlowMulticardParamDistribution v);
const char* ToString(FlowBufferAccess v);
const char* ToString(FlowGeneratedNameKind v);
const char* ToString(FlowBufferOrigin v);
const char* ToString(FlowHostOutputKind v);
const char* ToString(FlowCleanupKind v);
const char* ToString(FlowDiagnosticSeverity v);
const char* ToString(FlowHostLoopKind v);
const char* ToString(FlowHelperRole v);
const char* ToString(FlowExprDataflowKind v);
const char* ToString(FlowDacForNestKind v);
const char* ToString(FlowGeneratedCodeRole v);
const char* ToString(FlowStepKind v);

} // namespace ascendc
} // namespace dacppTranslator

#endif // TRANSLATOR_REWRITER_ASCENDC_FLOW_PLAN_H
