/*
编译器驱动（Driver）。
将 AST Lowering、MLIR Analysis、AscendC Planning 和 Code Emitting 四个独立模块串联起来，
同时负责处理命令行参数、环境变量、异常情况下的平滑回退，
以及在各个阶段输出调试信息，
是理解整个 MLIR 到 AscendC 转换流程的最佳入口。
*/
#include "ascendc_mlir/CalcMlirPipeline.h"
#include "ascendc_mlir/CalcAstToMlir.h"
#include "ascendc_mlir/CalcAnalysis.h"
#include "ascendc_mlir/AscendCPlan.h"
#include "ascendc_mlir/AscendCEmitter.h"
#include "ascendc_mlir/CalcContext.h"
#include "ascendc_mlir/CalcShellInfo.h"

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "llvm/Support/raw_ostream.h"

namespace dacppTranslator {
namespace ascendc_mlir {
namespace {

// 控制编译器行为，通过环境变量或进程参数指定。
bool IsTruthyEnv(const char* key) {
    const char* value = std::getenv(key);
    if (!value) {
        return false;
    }

    std::string raw(value);
    for (char& c : raw) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }

    return raw == "1" || raw == "true" || raw == "on" || raw == "yes";
}

// 读取 /proc/self/cmdline 文件，解析出当前进程启动时的所有命令行参数，并存入字符串数组中。
std::vector<std::string> ReadProcessCmdlineArgs() {
    std::vector<std::string> args;
    std::ifstream cmdline("/proc/self/cmdline", std::ios::binary);
    if (!cmdline) {
        return args;
    }

    std::string raw((std::istreambuf_iterator<char>(cmdline)), std::istreambuf_iterator<char>());
    std::string token;
    for (char ch : raw) {
        if (ch == '\0') {
            if (!token.empty()) {
                args.push_back(token);
                token.clear();
            }
        } else {
            token.push_back(ch);
        }
    }
    if (!token.empty()) {
        args.push_back(token);
    }

    return args;
}

// 遍历上一个函数获取的命令行参数列表，检查是否包含特定的flag，用来看编译器行为。
bool HasProcessFlag(const std::vector<std::string>& args, const char* flag) {
    for (const std::string& arg : args) {
        if (arg == flag) {
            return true;
        }
    }
    return false;
}

// 判断给定的计算语义家族（Semantic Family）是否是被 MLIR 良好支持和识别。
// 目前识别为 true 的有：逐点运算 (Pointwise)、空间模板 (SpatialStencil)、时间模板 (TemporalStencil)；
// 归约 (Reduction) 和标量回退 (ScalarFallback) 则返回 false。
bool IsRecognizedFamily(CalcSemanticFamily family) {
    switch (family) {
    case CalcSemanticFamily::PointwiseFamily:
    case CalcSemanticFamily::SpatialStencilFamily:
    case CalcSemanticFamily::TemporalStencilFamily:
    case CalcSemanticFamily::CubeMatmulFamily:
    case CalcSemanticFamily::LinearReductionFamily:
        return true;
    case CalcSemanticFamily::ReductionFamily:
    case CalcSemanticFamily::ScalarFallbackFamily:
        return false;
    }
    return false;
}

// 遍历 MLIR 分析结果中的所有Findings，检查是否包含上述被识别的计算家族。
bool HasRecognizedFindings(const CalcAnalysisResult& analysisResult) {
    for (const CalcSemanticFinding& finding : analysisResult.findings) {
        if (IsRecognizedFamily(finding.family)) {
            return true;
        }
    }
    return false;
}

// 检查分析结果中是否存在不得不回退到Scalar计算的情况。
bool HasScalarFallbackFinding(const CalcAnalysisResult& analysisResult) {
    for (const CalcSemanticFinding& finding : analysisResult.findings) {
        if (finding.family == CalcSemanticFamily::ScalarFallbackFamily ||
            finding.category == CalcSemanticCategory::ScalarFallback) {
            return true;
        }
    }
    return false;
}

// 检查最终生成的AscendC执行Plan的步骤中，是否包含了标量回退（ScalarFallback）的步骤。
bool HasScalarFallbackStep(const AscendCPlan& plan) {
    for (const AscendCPlanStep& step : plan.steps) {
        if (step.kind == AscendCPlanStepKind::ScalarFallback) {
            return true;
        }
    }
    return false;
}

} // namespace

// 结合前面解析的环境变量和命令行参数，生成并返回流水线的配置项。
CalcMlirPipelineOptions ResolveCalcMlirPipelineOptions() {
    CalcMlirPipelineOptions options;
    const std::vector<std::string> args = ReadProcessCmdlineArgs();

    options.useMlir = HasProcessFlag(args, "--ascendc-use-mlir") || IsTruthyEnv("DACPP_ASCENDC_USE_MLIR");
    options.dumpMlir = HasProcessFlag(args, "--ascendc-dump-mlir") || IsTruthyEnv("DACPP_ASCENDC_DUMP_MLIR");
    options.dumpAscendCPlan = HasProcessFlag(args, "--ascendc-dump-ascendc-plan") ||
                              IsTruthyEnv("DACPP_ASCENDC_DUMP_ASCENDC_PLAN");
    options.requireMlirForRecognizedCases =
        HasProcessFlag(args, "--ascendc-require-mlir-recognized-cases") ||
        IsTruthyEnv("DACPP_ASCENDC_REQUIRE_MLIR_RECOGNIZED_CASES");
    options.dumpFlowPlan = HasProcessFlag(args, "--ascendc-dump-flow-plan") ||
                           IsTruthyEnv("DACPP_ASCENDC_DUMP_FLOW_PLAN");

    // useHostFlow / hostFlowFallback 已从 options 移除。HostFlow 是
    // 唯一路径，无需 runtime 选择。

    if (options.dumpMlir || options.dumpAscendCPlan || options.requireMlirForRecognizedCases) {
        options.useMlir = true;
    }

    return options;
}

// 整个文件的核心枢纽函数，它执行了一个完整的端到端翻译流程。
CalcMlirPipelineResult RunCalcMlirPipeline(const CalcMlirPipelineInput& input,
                                           const CalcMlirPipelineOptions& options,
                                           const LegacyCalcLoweringFn& legacyLowering) {
    CalcMlirPipelineResult pipelineResult;

    auto BuildFailureStub = [](std::string reason) {
        if (reason.empty()) {
            reason = "unknown reason";
        }
        for (char& c : reason) {
            if (c == '"') {
                c = '\'';
            } else if (c == '\n' || c == '\r' || c == '\t') {
                c = ' ';
            }
        }
        return std::string("        static_assert(false, \"ascendc-mlir pipeline failed: ") + reason + "\");\n";
    };

    if (!legacyLowering) {
        pipelineResult.selectedPath = CalcPipelinePath::LegacyRegex;
        pipelineResult.emittedCode = std::string();
        return pipelineResult;
    }

    if (!options.useMlir) {
        pipelineResult.selectedPath = CalcPipelinePath::LegacyRegex;
        pipelineResult.emittedCode = legacyLowering(input);
        return pipelineResult;
    }

    pipelineResult.selectedPath = CalcPipelinePath::MlirScaffold;

    CalcContextSummary localContext;
    const CalcContextSummary* contextSummary = input.contextSummary;
    if (!contextSummary) {
        localContext = ExtractCalcContext(input.calc, input.sourceText);
        contextSummary = &localContext;
    }

    CalcAstToMlirOptions lowerOptions;
    lowerOptions.verifyModule = true;
    lowerOptions.contextSummary = contextSummary;
    lowerOptions.paramBindings = input.paramBindings;
    lowerOptions.maxInlineDepth = 8;
    CalcAstToMlirResult lowerResult = LowerCalcAstToMlir(input.calc, lowerOptions);

    const bool loweringFailed =
        !lowerResult.lowered ||
        !lowerResult.verified ||
        !lowerResult.loweringDiagnostics.empty();

    if (options.dumpMlir) {
        llvm::outs() << "[ascendc-mlir] dump mlir module (pipeline facade):\n";
        if (lowerResult.lowered && !lowerResult.moduleText.empty()) {
            llvm::outs() << lowerResult.moduleText;
            if (lowerResult.moduleText.back() != '\n') {
                llvm::outs() << "\n";
            }
        } else {
            // 功能：处理outs相关逻辑。
            llvm::outs() << "module {\n}\n";
        }
        if (!lowerResult.diagnostics.empty()) {
            llvm::outs() << "[ascendc-mlir] diagnostics: " << lowerResult.diagnostics << "\n";
        }
        if (contextSummary) {
            llvm::outs() << "[ascendc-mlir] calc-context const-count="
                         << contextSummary->numericConsts.size()
                         << " pure-inline-fn-count="
                         << contextSummary->pureInlineFns.size()
                         << " unresolved-count="
                         << contextSummary->unresolvedSymbols.size() << "\n";
        }
        if (input.shellSummary) {
            llvm::outs() << DumpCalcShellSummary(*input.shellSummary);
        }
        if (!lowerResult.loweringDiagnostics.empty()) {
            for (const auto& d : lowerResult.loweringDiagnostics) {
                llvm::outs() << "[ascendc-mlir] lowering-diagnostic: [" << d.nodeKind << "] "
                             << d.message << "\n";
            }
        }
        if (lowerResult.lowered) {
            llvm::outs() << "[ascendc-mlir] verify: " << (lowerResult.verified ? "pass" : "fail") << "\n";
        }
    }

    if (loweringFailed) {
        pipelineResult.selectedPath = CalcPipelinePath::MlirFailed;
        llvm::errs() << "[ascendc-mlir] path: MlirFailed\n";
        llvm::errs() << "[ascendc-mlir] pipeline failed in AST->MLIR lowering"
                     << (lowerResult.diagnostics.empty() ? "" : ": ")
                     << lowerResult.diagnostics << "\n";

        if (options.requireMlirForRecognizedCases) {
            pipelineResult.emittedCode = BuildFailureStub(
                "mlir lowering failed while requireMlirForRecognizedCases is enabled");
            return pipelineResult;
        }

        std::string fallbackCode = legacyLowering(input);
        if (!fallbackCode.empty()) {
            pipelineResult.emittedCode = fallbackCode;
            return pipelineResult;
        }
        pipelineResult.emittedCode = BuildFailureStub(lowerResult.diagnostics);
        return pipelineResult;
    }

    CalcAnalysisOptions analysisOptions;
    analysisOptions.shellSummary = input.shellSummary;
    analysisOptions.paramOrder = &input.shellParamOrder;
    CalcAnalysisResult analysisResult = AnalyzeCalcMlirModule(lowerResult.artifact, analysisOptions);
    if (!lowerResult.diagnostics.empty()) {
        analysisResult.notes.push_back("lowering diagnostics: " + lowerResult.diagnostics);
    }
    AscendCPlanBuildOptions planOptions;
    planOptions.shellSummary = input.shellSummary;
    AscendCPlan plan = BuildAscendCPlan(
        analysisResult,
        lowerResult.artifact,
        input.shellParamOrder,
        input.inLocalVars,
        input.outLocalVars,
        planOptions);
    pipelineResult.plan = plan;
    pipelineResult.analysisFindings = analysisResult.findings;

    for (const CalcSemanticFinding& finding : analysisResult.findings) {
        if (!pipelineResult.dominantFamily.has_value() &&
            finding.family != CalcSemanticFamily::ScalarFallbackFamily) {
            pipelineResult.dominantFamily = finding.family;
        }
        if (!pipelineResult.dominantFamilyPayload.has_value() &&
            finding.family != CalcSemanticFamily::ScalarFallbackFamily &&
            finding.hasFamilyPayload) {
            pipelineResult.dominantFamilyPayload = finding.familyPayload;
        }
    }

    for (const AscendCPlanStep& step : plan.steps) {
        if (step.kind != AscendCPlanStepKind::StructuredShiftedMapUpdate) {
            continue;
        }
        pipelineResult.structuredShiftedMapUpdate = step.structuredShiftedMapUpdate;
        break;
    }

    for (const AscendCPlanStep& step : plan.steps) {
        if (step.kind != AscendCPlanStepKind::CoeffStencil) {
            continue;
        }
        if (step.structuredMultiViewExpr.mainExpr.empty()) {
            continue;
        }
        pipelineResult.structuredMultiViewExpr = step.structuredMultiViewExpr;
        break;
    }

    if (pipelineResult.structuredShiftedMapUpdate.has_value()) {
        for (const AscendCPlanStep& step : plan.steps) {
            if (step.kind != AscendCPlanStepKind::Clamp) {
                continue;
            }
            if (step.dst != pipelineResult.structuredShiftedMapUpdate->dstTensor) {
                continue;
            }
            if (!pipelineResult.structuredShiftedMapUpdate->hasPost) {
                pipelineResult.structuredShiftedMapUpdate->hasPost = true;
                pipelineResult.structuredShiftedMapUpdate->postExpr = "max(UpdateResult,0)";
            }
            break;
        }
    }

    for (const AscendCPlanStep& step : plan.steps) {
        if (!pipelineResult.dominantFamily.has_value() &&
            step.family != CalcSemanticFamily::ScalarFallbackFamily) {
            pipelineResult.dominantFamily = step.family;
        }
        if (!pipelineResult.dominantFamilyPayload.has_value() &&
            step.family != CalcSemanticFamily::ScalarFallbackFamily &&
            step.hasFamilyPayload) {
            pipelineResult.dominantFamilyPayload = step.familyPayload;
        }
    }

    const bool recognizedCase = HasRecognizedFindings(analysisResult);
    const bool hasScalarFallbackFinding = HasScalarFallbackFinding(analysisResult);
    const bool hasScalarFallbackPlan = HasScalarFallbackStep(plan);

    if (options.requireMlirForRecognizedCases && recognizedCase &&
        (hasScalarFallbackFinding || hasScalarFallbackPlan)) {
        pipelineResult.selectedPath = CalcPipelinePath::MlirFailed;
        llvm::errs() << "[ascendc-mlir] path: MlirFailed\n";
        llvm::errs() << "[ascendc-mlir] recognized case produced scalar fallback, aborting due to requireMlirForRecognizedCases\n";
        pipelineResult.emittedCode = BuildFailureStub("recognized case fell back to scalar path");
        return pipelineResult;
    }

    AscendCEmitterOptions emitterOptions;
    emitterOptions.emitReasonComments = true;
    emitterOptions.allowScalarBoundaryFallback = true;
    emitterOptions.exprGraph = &analysisResult.exprGraph;
    std::string planGeneratedCode = EmitAscendCFromPlan(plan, input.tileLengthVar, emitterOptions);

    if (options.dumpAscendCPlan) {
        llvm::outs() << "[ascendc-mlir] dump ascendc semantic plan (pipeline facade):\n";
        llvm::outs() << DumpAscendCPlan(plan);
        llvm::outs() << "[ascendc-mlir] dump calc expression graph (pipeline facade):\n";
        llvm::outs() << RenderCalcExprGraph(analysisResult.exprGraph);
        llvm::outs() << "[ascendc-mlir] emit ascendc from plan (pipeline facade):\n";
        if (planGeneratedCode.empty()) {
            llvm::outs() << "// empty plan-generated AscendC code\n";
        } else {
            llvm::outs() << planGeneratedCode;
            if (planGeneratedCode.back() != '\n') {
                llvm::outs() << "\n";
            }
        }
    }

    if (planGeneratedCode.empty()) {
        pipelineResult.selectedPath = CalcPipelinePath::MlirFailed;
        llvm::errs() << "[ascendc-mlir] path: MlirFailed\n";
        llvm::errs() << "[ascendc-mlir] pipeline failed: plan emission produced empty code\n";
        pipelineResult.emittedCode = BuildFailureStub("plan emission produced empty code");
        return pipelineResult;
    }

    pipelineResult.emittedCode = planGeneratedCode;
    return pipelineResult;
}

// 一层封装
std::string RunCalcMlirPipelineFacade(const CalcMlirPipelineInput& input,
                                      const CalcMlirPipelineOptions& options,
                                      const LegacyCalcLoweringFn& legacyLowering,
                                      CalcPipelinePath* selectedPath) {
    CalcMlirPipelineResult result = RunCalcMlirPipeline(input, options, legacyLowering);
    if (selectedPath) {
        *selectedPath = result.selectedPath;
    }
    return result.emittedCode;
}

} // namespace ascendc_mlir
} // namespace dacppTranslator
