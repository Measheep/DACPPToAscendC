#ifndef TRANSLATOR_REWRITER_ASCENDC_HOST_FLOW_BUILDER_H
#define TRANSLATOR_REWRITER_ASCENDC_HOST_FLOW_BUILDER_H

// HostFlowBuilder：从 Clang AST + AscendCExpressionArtifact 列表构造 AscendCFlowPlan。
//
// Host main body 不做用户源码 Stmt 原文回放：
//   - 由 BuildHostFlowPlanFromAst 直接读 owner function 的 AST，把每条语句分类为
//     结构化 FlowStep（FlowKernelLaunch / FlowHostInput / FlowTensorBuffer /
//     FlowHelperInvocation / FlowHostLoop / FlowHostBranch / FlowHostOutput / ...）。
//   - 不能结构化的语句 → FlowDiagnostic(Error)，由 driver 转 host generation error。
//
// 头文件 forward declare Clang/DACPP 类型，不 include Clang 头，以兼容
// DACPP_TRANSLATOR_HAS_MLIR=0 构建，并保持输入结构与 MLIR 解耦。

#include <string>
#include <vector>

#include "ascendc/AscendCFlowPlan.h"

namespace clang {
class TranslationUnitDecl;
class SourceManager;
class LangOptions;
}

namespace dacppTranslator {
class DacppFile;
namespace ascendc {

struct AscendCExpressionArtifact;

struct HostFlowBuildInput {
    DacppFile* dacppFile = nullptr;
    const clang::TranslationUnitDecl* tuDecl = nullptr;
    const clang::SourceManager* sourceManager = nullptr;
    const clang::LangOptions* langOptions = nullptr;
    // Vector/Cube lowering 完成后产出的 artifact 列表（按 expressionIndex 排列）。
    const std::vector<AscendCExpressionArtifact>* expressionArtifacts = nullptr;
    const std::string* sourceFileHint = nullptr;
    bool mixedRouteConflict = false;
    std::string mixedRouteReason;
};

// 主入口：直接从 AST + expressionArtifacts 构造结构化 AscendCFlowPlan。
// 调用后 plan.builderRan = true；如发现无法结构化的语义会在 plan.diagnostics 中
// 记录 Error 级条目，driver 据此决定是否生成 host generation error body。
AscendCFlowPlan BuildHostFlowPlanFromAst(const HostFlowBuildInput& input);

} // namespace ascendc
} // namespace dacppTranslator

#endif // TRANSLATOR_REWRITER_ASCENDC_HOST_FLOW_BUILDER_H
