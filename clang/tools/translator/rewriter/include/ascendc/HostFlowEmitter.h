#ifndef TRANSLATOR_REWRITER_ASCENDC_HOST_FLOW_EMITTER_H
#define TRANSLATOR_REWRITER_ASCENDC_HOST_FLOW_EMITTER_H

// HostFlowEmitter：从 AscendCFlowPlan.steps 直接生成 hostN.cpp 的 main body
// 字符串。
//
// emitter 完全按结构化 FlowStepKind 分派：
//   HostInput / TensorBuffer / HelperInvocation / HostLoop / HostBranch /
//   KernelLaunch / HostOutput / Cleanup / GeneratedCodeBlock / Diagnostic / Raw
//
// 严格约束：
//   - 用户源码 Stmt 原文绝不参与 host body 输出；FlowHostInput / FlowTensorBuffer /
//     FlowHostLoop / FlowHostBranch / FlowHelperInvocation / FlowCleanupStep 等节点
//     都只携带结构化字段，emitter 按这些字段重建语句。
//   - FlowHostOutput::generatedCode、FlowGeneratedCodeBlock::code 才允许携带成段
//     translator-generated 文本（alias materialization、ACL/runtime/kernel-launch）。
//   - FlowStepKind::Raw 视为 hard error：result.ok = false 并停止发射；FlowDiagnostic(Error)
//     同理（errorOnHardDiagnostic=true 时）。driver 据 result.ok=false 走 host
//     generation error 路径。
//   - 不再依赖任何旧的文本回放路径。

#include <string>
#include <vector>

#include "ascendc/AscendCFlowPlan.h"

namespace dacppTranslator {
namespace ascendc {

struct HostFlowEmitOptions {
    std::string indent = "    ";       // 每层缩进；当前实现内层缩进沿用同一字符串
    bool errorOnHardDiagnostic = true; // FlowDiagnostic(Error) 是否置 result.ok = false
    // 路由过滤：只发射 route 匹配的 FlowKernelLaunch；Unknown 表示不过滤。
    // 同一个 plan 可用 Vector/Cube 两次发射出 host0.cpp / host1.cpp。
    FlowRouteKind routeFilter = FlowRouteKind::Unknown;
};

struct HostFlowEmitResult {
    std::string body;
    bool ok = true;
    std::vector<std::string> diagnostics;
};

HostFlowEmitResult RenderHostFlowBody(const AscendCFlowPlan& plan,
                                      const HostFlowEmitOptions& options);

} // namespace ascendc
} // namespace dacppTranslator

#endif // TRANSLATOR_REWRITER_ASCENDC_HOST_FLOW_EMITTER_H
