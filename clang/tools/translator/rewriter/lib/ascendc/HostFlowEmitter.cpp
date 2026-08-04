// HostFlowEmitter：按 AscendCFlowPlan.steps 顺序结构化发射 host main body。
//
// 关键约束：
//   1. FlowStepKind::Raw 视为 hard error — emitter 把 result.ok 置 false 并停止发射；
//      driver 据此切换 host generation error body。
//   2. FlowDiagnostic(Error) 阻止后续发射，行为同上。
//   3. 不存在任何"raw stmt 文本回放"路径——用户源码 Stmt 不参与 host body。
//      FlowHostOutput::generatedCode / FlowGeneratedCodeBlock::code 才能承载成段
//      translator-generated 文本，且只覆盖 alias materialization / ACL/runtime / kernel-launch。
//   4. KernelLaunch 通过 FlowGeneratedCodeBlock 子步装配
//      (commonDecl/alloc/copyIn → launch body → copyOut/free)，launch body 来自
//      plan.generatedBlocks 中 role==KernelLaunch 或 TemporalLoopBody 且 ownerExpr 匹配的块。
//   5. 控制流（HostLoop / HostBranch）按 bodyStepIds/thenStepIds/elseStepIds 递归发射；
//      top-level 线性遍历会跳过这些被父级 step 消费的 step id。

#include "ascendc/HostFlowEmitter.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dacppTranslator {
namespace ascendc {

namespace {

// 文本辅助
std::string EnsureTrailingNewline(const std::string& s) {
    if (s.empty()) return s;
    if (s.back() == '\n') return s;
    std::string out = s;
    out.push_back('\n');
    return out;
}

void AppendCompactLine(std::string& output, const std::string& indent,
                       const std::string& line) {
    if (line.empty()) return;
    output += indent;
    output += line;
    if (output.back() != '\n') output.push_back('\n');
}

std::string IndentBlock(const std::string& code, const std::string& indent) {
    if (code.empty()) return code;
    std::string out;
    out.reserve(code.size() + indent.size());
    std::size_t i = 0;
    bool atLineStart = true;
    while (i < code.size()) {
        if (atLineStart && code[i] != '\n') {
            out += indent;
        }
        out.push_back(code[i]);
        atLineStart = (code[i] == '\n');
        ++i;
    }
    return out;
}

// ============================================================
// EmitContext + 顶层"是否被父级消费"集合的预计算
// ============================================================

struct EmitContext {
    const AscendCFlowPlan& plan;
    const HostFlowEmitOptions& options;
    HostFlowEmitResult& result;
    std::string body;
    const std::unordered_set<FlowStepId>& consumed;
    bool fatal = false;
};

std::unordered_set<FlowStepId> ComputeConsumedSteps(const AscendCFlowPlan& plan) {
    std::unordered_set<FlowStepId> consumed;
    for (const FlowHostLoop& loop : plan.hostLoops) {
        for (FlowStepId id : loop.bodyStepIds) consumed.insert(id);
    }
    for (const FlowHostBranch& br : plan.hostBranches) {
        for (FlowStepId id : br.thenStepIds) consumed.insert(id);
        for (FlowStepId id : br.elseStepIds) consumed.insert(id);
    }
    for (const FlowKernelLaunch& launch : plan.kernelLaunches) {
        for (FlowStepId id : launch.commonDeclStepIds)  consumed.insert(id);
        for (FlowStepId id : launch.allocStepIds)       consumed.insert(id);
        for (FlowStepId id : launch.copyInStepIds)      consumed.insert(id);
        for (FlowStepId id : launch.copyOutStepIds)     consumed.insert(id);
        for (FlowStepId id : launch.freeStepIds)        consumed.insert(id);
        for (FlowStepId id : launch.postLaunchStepIds)  consumed.insert(id);
    }
    // KernelLaunch role 的 FlowGeneratedCodeBlock step 由 EmitKernelLaunch 通过
    // ownerExpr + role 主动拾取，在 top-level walk 跳过。
    for (std::size_t i = 0; i < plan.steps.size(); ++i) {
        const FlowStep& step = plan.steps[i];
        if (step.kind != FlowStepKind::GeneratedCodeBlock) continue;
        if (step.payloadIndex >= plan.generatedBlocks.size()) continue;
        const FlowGeneratedCodeBlock& blk = plan.generatedBlocks[step.payloadIndex];
        if (blk.role == FlowGeneratedCodeRole::KernelLaunch ||
            blk.role == FlowGeneratedCodeRole::TemporalLoopBody ||
            blk.role == FlowGeneratedCodeRole::MulticardLaunch ||
            blk.role == FlowGeneratedCodeRole::MulticardRuntime) {
            consumed.insert(static_cast<FlowStepId>(i));
        }
    }
    return consumed;
}

// ============================================================
// 各 FlowStepKind 的发射器
// ============================================================

// 前向声明：HostLoop / HostBranch 递归发射 body step。
void EmitStep(EmitContext& cx, FlowStepId stepId);
void EmitStepList(EmitContext& cx, const std::vector<FlowStepId>& ids);

std::string EmitNestedBody(EmitContext& cx, const std::vector<FlowStepId>& stepIds) {
    EmitContext inner = cx;
    inner.body.clear();
    EmitStepList(inner, stepIds);
    cx.result.diagnostics.insert(cx.result.diagnostics.end(),
                                  inner.result.diagnostics.begin(),
                                  inner.result.diagnostics.end());
    if (inner.fatal) cx.fatal = true;
    return IndentBlock(inner.body, cx.options.indent);
}

// 把 translator-generated 文本片段按指定缩进追加到 body：
// 多行按行加缩进，单行单次缩进，并保证结尾换行。
// 注意：这只用于 generatedCode / FlowGeneratedCodeBlock::code 这种 translator
// 自己生成的片段；用户源码 Stmt 不允许走这条路径。
void AppendGeneratedSnippet(EmitContext& cx, const std::string& snippet,
                            const std::string& indent) {
    if (snippet.empty()) return;
    if (snippet.find('\n') == std::string::npos) {
        AppendCompactLine(cx.body, indent, snippet);
        return;
    }
    cx.body += IndentBlock(EnsureTrailingNewline(snippet), indent);
}

void EmitHostInput(EmitContext& cx, const FlowHostInput& hi) {
    if (hi.name.empty()) return;
    // 仅按结构化字段重建声明，不读取任何原文回放路径。
    std::string line = cx.options.indent;
    if (!hi.storageClass.empty()) {
        line += hi.storageClass;
        line.push_back(' ');
    }
    if (hi.isConstexpr) line += "constexpr ";
    if (!hi.declTypeText.empty()) {
        line += hi.declTypeText;
        line.push_back(' ');
    }
    line += hi.name;
    if (!hi.arrayExtentExpr.empty()) {
        line += "[" + hi.arrayExtentExpr + "]";
    }
    if (!hi.initializerText.empty()) {
        line += " = " + hi.initializerText;
    }
    line += ";\n";
    cx.body += line;
}

void EmitTensorBuffer(EmitContext& cx, const FlowTensorBuffer& tb) {
    if (tb.name.empty()) return;
    // 按 ctorArgs / initListElems / copyFromVectorName 三路初始化字段
    // 完整重建 std::vector 声明（或 dacpp::Vector/Matrix/Tensor 声明的容器壳）。
    // 用户源码 Stmt 不参与发射。
    std::string line = cx.options.indent;
    if (tb.dacppContainerText.empty()) {
        line += "std::vector<";
        line += tb.elementTypeText.empty() ? "auto" : tb.elementTypeText;
        line += "> ";
    } else {
        line += "dacpp::" + tb.dacppContainerText + "<";
        line += tb.elementTypeText.empty() ? "auto" : tb.elementTypeText;
        line += "> ";
    }
    line += tb.name;

    // 互斥三路初始化（builder 保证最多命中一种；同时为空表示 default-ctor）。
    if (!tb.initListElems.empty()) {
        // std::vector<T> v{a, b, c}
        line += "{";
        for (std::size_t i = 0; i < tb.initListElems.size(); ++i) {
            if (i) line += ", ";
            line += tb.initListElems[i];
        }
        line += "}";
    } else if (!tb.copyFromVectorName.empty()) {
        // std::vector<T> v(other)
        line += "(" + tb.copyFromVectorName + ")";
    } else if (!tb.ctorArgs.empty()) {
        // std::vector<T> v(n) / std::vector<T> v(n, init)
        line += "(";
        for (std::size_t i = 0; i < tb.ctorArgs.size(); ++i) {
            if (i) line += ", ";
            line += tb.ctorArgs[i];
        }
        line += ")";
    } else if (!tb.shapeExprs.empty()) {
        // dacpp 容器（非 std::vector）保留原有 shape-ctor 重建路径。
        line += "(";
        for (std::size_t i = 0; i < tb.shapeExprs.size(); ++i) {
            if (i) line += ", ";
            line += tb.shapeExprs[i];
        }
        line += ")";
    }
    line += ";\n";
    cx.body += line;
}

void EmitHelperInvocation(EmitContext& cx, const FlowHelperInvocation& h) {
    // 仅按 functionName/argTexts 重建调用，不读取任何原文回放路径。
    if (h.functionName.empty()) return;
    std::string line = cx.options.indent;
    if (!h.resultTargetName.empty()) {
        line += h.resultTargetName + " = ";
    }
    line += h.functionName + "(";
    for (std::size_t i = 0; i < h.argTexts.size(); ++i) {
        if (i) line += ", ";
        line += h.argTexts[i];
    }
    line += ");\n";
    cx.body += line;
}

void EmitHostOutput(EmitContext& cx, const FlowHostOutput& o) {
    const std::string& ind = cx.options.indent;
    switch (o.kind) {
        case FlowHostOutputKind::TensorPrint: {
            // builder 把 alias materialization 灌入 generatedCode；只允许发这一路。
            // 缺 generatedCode 视作 builder 未能完成 lowering — hard fail。
            if (o.generatedCode.empty()) {
                cx.result.diagnostics.push_back(
                    "[HostFlow][Error] tensor print without generatedCode "
                    "(builder should have emitted hostflow.tensor_print_unresolved)");
                cx.fatal = true;
                break;
            }
            cx.body += EnsureTrailingNewline(o.generatedCode);
            break;
        }
        case FlowHostOutputKind::ScalarSelfUpdate: {
            // 按 target + compoundOp + value 重建；缺字段视作破坏性 hard fail。
            if (o.targetExprText.empty() || o.valueExprText.empty() ||
                o.compoundOp.empty()) {
                cx.result.diagnostics.push_back(
                    "[HostFlow][Error] ScalarSelfUpdate missing "
                    "target/value/compoundOp");
                cx.fatal = true;
                break;
            }
            cx.body += ind + o.targetExprText + " " + o.compoundOp + " " +
                       o.valueExprText + ";\n";
            break;
        }
        case FlowHostOutputKind::TensorAssign:
        case FlowHostOutputKind::VectorElemWrite: {
            // alias materialization 走 generatedCode 直出；否则按 target = value 重建。
            if (!o.generatedCode.empty()) {
                AppendGeneratedSnippet(cx, o.generatedCode, ind);
            } else if (!o.targetExprText.empty() && !o.valueExprText.empty()) {
                cx.body += ind + o.targetExprText + " = " + o.valueExprText + ";\n";
            } else {
                cx.result.diagnostics.push_back(
                    "[HostFlow][Error] tensor/vector assign without "
                    "target/value or generatedCode");
                cx.fatal = true;
            }
            break;
        }
        case FlowHostOutputKind::StdOutStream: {
            if (o.streamName.empty() || o.streamOperands.empty()) {
                cx.result.diagnostics.push_back(
                    "[HostFlow][Error] StdOutStream missing streamName/operands");
                cx.fatal = true;
                break;
            }
            std::string line = ind + o.streamName;
            for (const std::string& op : o.streamOperands) {
                line += " << " + op;
            }
            line += ";\n";
            cx.body += line;
            break;
        }
        case FlowHostOutputKind::PrintfCall: {
            if (o.printfCallee.empty()) {
                cx.result.diagnostics.push_back(
                    "[HostFlow][Error] PrintfCall missing callee");
                cx.fatal = true;
                break;
            }
            std::string line = ind + o.printfCallee + "(";
            for (std::size_t i = 0; i < o.printfArgs.size(); ++i) {
                if (i) line += ", ";
                line += o.printfArgs[i];
            }
            line += ");\n";
            cx.body += line;
            break;
        }
        case FlowHostOutputKind::Unknown:
            // HostOutput 不允许出现 Unknown — builder 应在分类失败时
            // 通过 FlowDiagnostic(Error) 报告，并跳过 addHostOutput。
            cx.result.diagnostics.push_back(
                "[HostFlow][Error] HostOutput with kind=Unknown reached emitter; "
                "builder contract violation");
            cx.fatal = true;
            break;
    }
}

void EmitCleanup(EmitContext& cx, const FlowCleanupStep& c) {
    const std::string& ind = cx.options.indent;
    auto emitCall = [&](const std::string& fn) {
        if (c.targetExprText.empty()) return;
        cx.body += ind + fn + "(" + c.targetExprText + ");\n";
    };
    switch (c.kind) {
        case FlowCleanupKind::HostFree:           emitCall("free"); break;
        case FlowCleanupKind::DeviceFree:         emitCall("aclrtFree"); break;
        case FlowCleanupKind::HostStreamDestroy:  emitCall("aclrtDestroyStream"); break;
        case FlowCleanupKind::AclFinalize:
            cx.body += ind + "aclFinalize();\n";
            break;
        case FlowCleanupKind::HostDelete:
            if (!c.targetExprText.empty()) {
                cx.body += ind + "delete " + c.targetExprText + ";\n";
            }
            break;
        case FlowCleanupKind::Unknown:
            // Cleanup 不允许出现 Unknown — builder 应在分类失败时
            // 通过 FlowDiagnostic(Error) 报告，并跳过 addCleanup。
            cx.result.diagnostics.push_back(
                "[HostFlow][Error] Cleanup with kind=Unknown reached emitter; "
                "builder contract violation");
            cx.fatal = true;
            break;
    }
}

std::string RenderForInit(const FlowHostLoop& loop) {
    if (loop.initIsDecl) {
        std::string s;
        if (!loop.initStorageClass.empty())
            s += loop.initStorageClass + " ";
        s += loop.initTypeText + " " + loop.initVarName;
        if (!loop.initValueExpr.empty())
            s += " = " + loop.initValueExpr;
        return s;
    }
    if (loop.initIsAssignment) {
        return loop.initAssignTargetExpr + " = " + loop.initAssignValueExpr;
    }
    return {};
}

void EmitHostLoop(EmitContext& cx, const FlowHostLoop& loop) {
    const std::string& ind = cx.options.indent;
    std::string header;
    switch (loop.kind) {
        case FlowHostLoopKind::For: {
            std::string initPart = RenderForInit(loop);
            header = ind + "for (" + initPart + "; "
                     + loop.upperBoundExpr + "; " + loop.stepExpr + ") {\n";
            break;
        }
        case FlowHostLoopKind::While:
            header = ind + "while (" + loop.conditionText + ") {\n";
            break;
        case FlowHostLoopKind::DoWhile:
            header = ind + "do {\n";
            break;
    }
    cx.body += header;

    cx.body += EmitNestedBody(cx, loop.bodyStepIds);

    if (loop.kind == FlowHostLoopKind::DoWhile) {
        cx.body += ind + "} while (" + loop.conditionText + ");\n";
    } else {
        cx.body += ind + "}\n";
    }
}

void EmitHostBranch(EmitContext& cx, const FlowHostBranch& br) {
    const std::string& ind = cx.options.indent;
    // 所有分支按 conditionText + then/else step 重建；不保留 raw text fast path。
    cx.body += ind + "if (" + br.conditionText + ") {\n";

    cx.body += EmitNestedBody(cx, br.thenStepIds);
    cx.body += ind + "}";

    if (!br.elseStepIds.empty()) {
        cx.body += " else {\n";
        cx.body += EmitNestedBody(cx, br.elseStepIds);
        cx.body += ind + "}";
    }
    cx.body += "\n";
}

void EmitGeneratedCodeBlockById(EmitContext& cx, FlowStepId id) {
    if (id >= cx.plan.steps.size()) return;
    const FlowStep& step = cx.plan.steps[id];
    if (step.kind != FlowStepKind::GeneratedCodeBlock) return;
    if (step.payloadIndex >= cx.plan.generatedBlocks.size()) return;
    const FlowGeneratedCodeBlock& blk = cx.plan.generatedBlocks[step.payloadIndex];
    if (cx.options.routeFilter != FlowRouteKind::Unknown &&
        blk.route != FlowRouteKind::Unknown &&
        blk.route != cx.options.routeFilter) {
        return;
    }
    if (blk.code.empty()) return;
    cx.body += EnsureTrailingNewline(blk.code);
}

const FlowGeneratedCodeBlock* FindOwnedBlock(
    const AscendCFlowPlan& plan,
    const FlowKernelLaunch& launch,
    FlowGeneratedCodeRole role) {
    for (const FlowGeneratedCodeBlock& blk : plan.generatedBlocks) {
        if (blk.role == role && blk.ownerExpr == launch.expressionIndex) {
            return &blk;
        }
    }
    return nullptr;
}

void EmitKernelLaunch(EmitContext& cx, const FlowKernelLaunch& launch) {
    // route filter：跨 Vector/Cube 共用同一 plan，发射时按路由过滤掉不属于本路径的 launch。
    if (cx.options.routeFilter != FlowRouteKind::Unknown &&
        launch.route != cx.options.routeFilter) {
        return;
    }
    if (launch.multicardEnabled) {
        const FlowGeneratedCodeBlock* command =
            FindOwnedBlock(cx.plan, launch, FlowGeneratedCodeRole::MulticardLaunch);
        if (!command) {
            cx.result.diagnostics.push_back(
                "[HostFlow][Error] multicard kernel launch missing role MulticardLaunch for expr#" +
                std::to_string(launch.expressionIndex));
            cx.fatal = true;
            return;
        }
        cx.body += cx.options.indent + "{\n";
        cx.body += EnsureTrailingNewline(command->code);
        if (!launch.postLaunchStepIds.empty()) {
            HostFlowEmitOptions nestedOptions = cx.options;
            nestedOptions.indent += cx.options.indent;
            EmitContext nested{cx.plan, nestedOptions, cx.result, std::string(),
                               cx.consumed, false};
            EmitStepList(nested, launch.postLaunchStepIds);
            cx.body += nested.body;
            if (nested.fatal) {
                cx.fatal = true;
                return;
            }
        }
        cx.body += cx.options.indent + "}\n";
        return;
    }

    // 顺序：commonDecls → alloc → copyIn → launch body → copyOut → free
    for (FlowStepId id : launch.commonDeclStepIds) EmitGeneratedCodeBlockById(cx, id);
    for (FlowStepId id : launch.allocStepIds)      EmitGeneratedCodeBlockById(cx, id);
    for (FlowStepId id : launch.copyInStepIds)     EmitGeneratedCodeBlockById(cx, id);

    // launch body：plan.generatedBlocks 中 role==KernelLaunch 或 TemporalLoopBody，ownerExpr 匹配
    for (const FlowGeneratedCodeBlock& blk : cx.plan.generatedBlocks) {
        if (blk.role != FlowGeneratedCodeRole::KernelLaunch &&
            blk.role != FlowGeneratedCodeRole::TemporalLoopBody) continue;
        if (blk.ownerExpr != launch.expressionIndex) continue;
        if (blk.code.empty()) continue;
        cx.body += EnsureTrailingNewline(blk.code);
    }

    for (FlowStepId id : launch.copyOutStepIds)    EmitGeneratedCodeBlockById(cx, id);
    // Post-launch steps（如 tensor print）在 D2H 后、free 前发射。
    EmitStepList(cx, launch.postLaunchStepIds);
    for (FlowStepId id : launch.freeStepIds)       EmitGeneratedCodeBlockById(cx, id);
}

void EmitDiagnostic(EmitContext& cx, const FlowDiagnostic& d) {
    std::ostringstream os;
    os << "[HostFlow][";
    switch (d.severity) {
        case FlowDiagnosticSeverity::Note:    os << "Note"; break;
        case FlowDiagnosticSeverity::Warning: os << "Warn"; break;
        case FlowDiagnosticSeverity::Error:   os << "Error"; break;
    }
    os << "] ";
    if (!d.code.empty()) os << d.code << ": ";
    os << d.message;
    if (d.location.isValid()) {
        os << "  (" << d.location.file << ":" << d.location.line << ")";
    }
    cx.result.diagnostics.push_back(os.str());
    if (d.severity == FlowDiagnosticSeverity::Error && cx.options.errorOnHardDiagnostic) {
        cx.fatal = true;
    }
}

// ============================================================
// Dispatch
// ============================================================

void EmitStep(EmitContext& cx, FlowStepId stepId) {
    if (cx.fatal) return;
    if (stepId >= cx.plan.steps.size()) return;
    const FlowStep& step = cx.plan.steps[stepId];

    switch (step.kind) {
        case FlowStepKind::HostInput:
            if (step.hasValidPayload(cx.plan.hostInputs.size()))
                EmitHostInput(cx, cx.plan.hostInputs[step.payloadIndex]);
            break;
        case FlowStepKind::TensorBuffer:
            if (step.hasValidPayload(cx.plan.tensorBuffers.size()))
                EmitTensorBuffer(cx, cx.plan.tensorBuffers[step.payloadIndex]);
            break;
        case FlowStepKind::HelperInvocation:
            if (step.hasValidPayload(cx.plan.helperInvocations.size()))
                EmitHelperInvocation(cx, cx.plan.helperInvocations[step.payloadIndex]);
            break;
        case FlowStepKind::HostLoop:
            if (step.hasValidPayload(cx.plan.hostLoops.size()))
                EmitHostLoop(cx, cx.plan.hostLoops[step.payloadIndex]);
            break;
        case FlowStepKind::HostBranch:
            if (step.hasValidPayload(cx.plan.hostBranches.size()))
                EmitHostBranch(cx, cx.plan.hostBranches[step.payloadIndex]);
            break;
        case FlowStepKind::KernelLaunch:
            if (step.hasValidPayload(cx.plan.kernelLaunches.size()))
                EmitKernelLaunch(cx, cx.plan.kernelLaunches[step.payloadIndex]);
            break;
        case FlowStepKind::HostOutput:
            if (step.hasValidPayload(cx.plan.hostOutputs.size()))
                EmitHostOutput(cx, cx.plan.hostOutputs[step.payloadIndex]);
            break;
        case FlowStepKind::Cleanup:
            if (step.hasValidPayload(cx.plan.cleanups.size()))
                EmitCleanup(cx, cx.plan.cleanups[step.payloadIndex]);
            break;
        case FlowStepKind::GeneratedCodeBlock:
            EmitGeneratedCodeBlockById(cx, stepId);
            break;
        case FlowStepKind::Diagnostic:
            if (step.hasValidPayload(cx.plan.diagnostics.size()))
                EmitDiagnostic(cx, cx.plan.diagnostics[step.payloadIndex]);
            break;
        case FlowStepKind::Raw:
            // FlowStepKind::Raw 视为 hard error —
            // builder 应仅在 diagnostic 路径下创建 Raw step，发射器看到即停止。
            cx.result.diagnostics.push_back(
                "[HostFlow][Error] FlowStepKind::Raw is not emittable "
                "(diagnostic-only); driver must switch to host generation "
                "error body. step#" + std::to_string(stepId));
            cx.fatal = true;
            break;
        case FlowStepKind::ScalarBinding:
        case FlowStepKind::ExprDataflowEdge:
        case FlowStepKind::AliasLifetime:
        case FlowStepKind::DacForNesting:
            // 这些步在 plan 中是辅助信息，不直接发射 host 代码。
            break;
        case FlowStepKind::None:
            break;
    }
}

void EmitStepList(EmitContext& cx, const std::vector<FlowStepId>& ids) {
    for (FlowStepId id : ids) {
        if (cx.fatal) return;
        EmitStep(cx, id);
    }
}

void ValidateMulticardRoute(const AscendCFlowPlan& plan,
                            FlowRouteKind route,
                            HostFlowEmitResult& result) {
    std::size_t setupCount = 0;
    std::size_t cleanupCount = 0;
    std::size_t legacyCount = 0;
    FlowStepId setupId = kInvalidFlowStepId;
    FlowStepId cleanupId = kInvalidFlowStepId;
    std::unordered_map<FlowExprIndex, std::size_t> commandCount;
    std::unordered_map<FlowExprIndex, FlowStepId> launchStepIds;

    for (std::size_t id = 0; id < plan.steps.size(); ++id) {
        const FlowStep& step = plan.steps[id];
        if (step.kind == FlowStepKind::KernelLaunch &&
            step.payloadIndex < plan.kernelLaunches.size()) {
            const FlowKernelLaunch& launch =
                plan.kernelLaunches[step.payloadIndex];
            if (launch.route == route && launch.multicardEnabled) {
                launchStepIds[launch.expressionIndex] = id;
            }
            continue;
        }
        if (step.kind != FlowStepKind::GeneratedCodeBlock ||
            step.payloadIndex >= plan.generatedBlocks.size()) {
            continue;
        }
        const FlowGeneratedCodeBlock& block =
            plan.generatedBlocks[step.payloadIndex];
        if (block.route != route) continue;
        switch (block.role) {
            case FlowGeneratedCodeRole::MulticardRouteSetup:
                ++setupCount;
                setupId = id;
                break;
            case FlowGeneratedCodeRole::MulticardLaunch:
                ++commandCount[block.ownerExpr];
                break;
            case FlowGeneratedCodeRole::MulticardRouteCleanup:
                ++cleanupCount;
                cleanupId = id;
                break;
            case FlowGeneratedCodeRole::MulticardRuntime:
                ++legacyCount;
                break;
            default:
                break;
        }
    }

    auto Fail = [&](const std::string& message) {
        result.ok = false;
        result.diagnostics.push_back(
            "[HostFlow][Error] multicard route " +
            std::string(ToString(route)) + ": " + message);
    };

    if (launchStepIds.empty()) {
        if (setupCount != 0 || cleanupCount != 0 || !commandCount.empty() ||
            legacyCount != 0) {
            Fail("lifecycle blocks exist without a multicard launch");
        }
        return;
    }
    if (legacyCount != 0) {
        Fail("legacy MulticardRuntime blocks are forbidden");
    }
    if (setupCount != 1 || cleanupCount != 1) {
        Fail("expected exactly one MulticardRouteSetup and one "
             "MulticardRouteCleanup (setup=" + std::to_string(setupCount) +
             ", cleanup=" + std::to_string(cleanupCount) + ")");
        return;
    }
    if (setupId >= cleanupId) {
        Fail("route cleanup must be reachable after route setup");
    }
    if (launchStepIds.size() > 1) {
        for (const FlowKernelLaunch& launch : plan.kernelLaunches) {
            if (launch.route == route && launch.multicardEnabled &&
                launch.multicard.persistentWorkers) {
                Fail("a scalar/temporal persistent schedule cannot share a route "
                     "with another launch until it uses the common executor barrier");
                break;
            }
        }
    }
    for (const auto& entry : launchStepIds) {
        const FlowExprIndex expr = entry.first;
        const FlowStepId launchId = entry.second;
        if (commandCount[expr] != 1) {
            Fail("expr#" + std::to_string(expr) +
                 " must own exactly one MulticardLaunch block");
        }
        if (!(setupId < launchId && launchId < cleanupId)) {
            Fail("expr#" + std::to_string(expr) +
                 " is outside the route setup/cleanup interval");
        }
        for (const FlowKernelLaunch& launch : plan.kernelLaunches) {
            if (launch.expressionIndex != expr || launch.route != route) continue;
            if (launch.multicard.resourceScope != "route" ||
                launch.multicard.workerLifetime != "persistent-route" ||
                launch.multicard.copyMode != "async-stream") {
                Fail("expr#" + std::to_string(expr) +
                     " has an invalid route resource/copy contract");
            }
            break;
        }
    }
    for (const auto& entry : commandCount) {
        if (launchStepIds.find(entry.first) == launchStepIds.end()) {
            Fail("orphan MulticardLaunch block for expr#" +
                 std::to_string(entry.first));
        }
    }
}

bool IsIdentifier(const std::string& text) {
    if (text.empty()) return false;
    const unsigned char first = static_cast<unsigned char>(text.front());
    if (!(std::isalpha(first) || text.front() == '_')) return false;
    for (char c : text) {
        const unsigned char value = static_cast<unsigned char>(c);
        if (!(std::isalnum(value) || c == '_')) return false;
    }
    return true;
}

bool IsReservedIdentifier(const std::string& text) {
    return text.size() >= 2 &&
           ((text[0] == '_' && text[1] == '_') ||
            (text[0] == '_' &&
             std::isupper(static_cast<unsigned char>(text[1]))));
}

void ValidateCompletedPlan(const AscendCFlowPlan& plan,
                           HostFlowEmitResult& result) {
    auto Fail = [&](const std::string& message) {
        result.ok = false;
        result.diagnostics.push_back("[HostFlow][Error] " + message);
    };

    for (const FlowGeneratedName& name : plan.generatedNames) {
        if (name.kind == FlowGeneratedNameKind::UserPreserved) continue;
        if (!IsIdentifier(name.text) || IsReservedIdentifier(name.text)) {
            Fail("invalid framework identifier '" + name.text + "'");
            continue;
        }
        const std::size_t limit =
            name.kind == FlowGeneratedNameKind::PublicType ? 24u : 20u;
        if (name.text.size() > limit) {
            Fail("framework identifier exceeds " + std::to_string(limit) +
                 " characters: '" + name.text + "'");
        }
    }

    for (FlowRouteKind route :
         {FlowRouteKind::Vector, FlowRouteKind::Cube}) {
        std::unordered_set<FlowBufferId> ids;
        FlowBufferId maxId = 0;
        bool any = false;
        for (const FlowKernelLaunch& launch : plan.kernelLaunches) {
            if (launch.route != route) continue;
            for (const FlowRuntimeBuffer& buffer : launch.runtimeBuffers) {
                if (buffer.id == kInvalidFlowBufferId) {
                    Fail("runtime buffer has invalid id");
                    continue;
                }
                any = true;
                maxId = std::max(maxId, buffer.id);
                ids.insert(buffer.id);
            }
        }
        if (any) {
            for (FlowBufferId id = 0; id <= maxId; ++id) {
                if (ids.count(id) == 0) {
                    Fail(std::string(ToString(route)) +
                         " runtime BufferId sequence is not contiguous");
                    break;
                }
            }
        }
    }

    static const char* forbidden[] = {
        "class DacppMulticardExecutor",
        "PrepareHostSlab",
        "std::map<std::string"
    };
    for (const FlowGeneratedCodeBlock& block : plan.generatedBlocks) {
        for (const char* token : forbidden) {
            if (block.code.find(token) != std::string::npos) {
                Fail("generated block contains forbidden runtime text: " +
                     std::string(token));
            }
        }
    }
}

} // namespace

HostFlowEmitResult RenderHostFlowBody(const AscendCFlowPlan& plan,
                                      const HostFlowEmitOptions& options) {
    HostFlowEmitResult result;

    // builderRan=false 视作严重契约破坏。
    if (!plan.builderRan) {
        result.ok = false;
        result.diagnostics.push_back(
            "[HostFlow][Error] plan.builderRan=false; HostFlowBuilder must run before "
            "RenderHostFlowBody");
        return result;
    }

    ValidateCompletedPlan(plan, result);
    if (!result.ok) return result;

    if (options.routeFilter == FlowRouteKind::Unknown) {
        ValidateMulticardRoute(plan, FlowRouteKind::Vector, result);
        ValidateMulticardRoute(plan, FlowRouteKind::Cube, result);
    } else {
        ValidateMulticardRoute(plan, options.routeFilter, result);
    }
    if (!result.ok) return result;

    const std::unordered_set<FlowStepId> consumed = ComputeConsumedSteps(plan);
    EmitContext cx{plan, options, result, std::string(), consumed, false};

    for (std::size_t id = 0; id < plan.steps.size(); ++id) {
        if (cx.fatal) break;
        const auto stepId = static_cast<FlowStepId>(id);
        if (consumed.count(stepId)) continue;
        EmitStep(cx, stepId);
    }

    while (!cx.body.empty() && cx.body.back() == '\n') cx.body.pop_back();
    result.body = std::move(cx.body);
    if (cx.fatal) result.ok = false;
    // plan.partial=true 意味着 builder 已经记录了 Error 级 diagnostic
    // 或不可发射的破坏性场景。即便发射过程没遇到 fatal step，也必须把 result.ok 置 false，
    // 让 driver 切换到 host generation error body（Step 4 的契约）。
    if (plan.partial) result.ok = false;
    return result;
}

} // namespace ascendc
} // namespace dacppTranslator
