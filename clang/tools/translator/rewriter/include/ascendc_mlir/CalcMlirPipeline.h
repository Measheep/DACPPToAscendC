#ifndef TRANSLATOR_REWRITER_ASCENDC_MLIR_CALC_MLIR_PIPELINE_H
#define TRANSLATOR_REWRITER_ASCENDC_MLIR_CALC_MLIR_PIPELINE_H

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "ascendc_mlir/AscendCPlan.h"
#include "ascendc_mlir/CalcAnalysis.h"

namespace dacppTranslator {
class Calc;

namespace ascendc_mlir {

struct CalcContextSummary;
struct CalcShellSummary;

enum class CalcPipelinePath {
    LegacyRegex,
    MlirScaffold,
    MlirFailed
};

struct CalcMlirPipelineOptions {
    bool useMlir = false;
    bool dumpMlir = false;
    bool dumpAscendCPlan = false;
    bool requireMlirForRecognizedCases = false;

    // 开启时把 AscendCFlowPlan dump 出来。
    // 开关：--ascendc-dump-flow-plan 或环境变量 DACPP_ASCENDC_DUMP_FLOW_PLAN。
    bool dumpFlowPlan = false;
};

struct CalcMlirPipelineInput {
    Calc* calc = nullptr;
    std::string sourceText;
    const CalcContextSummary* contextSummary = nullptr;
    const CalcShellSummary* shellSummary = nullptr;
    const std::vector<CalcParamBinding>* paramBindings = nullptr;
    std::string tileLengthVar;
    std::vector<std::string> inLocalVars;
    std::vector<std::string> outLocalVars;
    std::vector<std::string> shellParamOrder;
};

using LegacyCalcLoweringFn = std::function<std::string(const CalcMlirPipelineInput&)>;

struct CalcMlirPipelineResult {
    CalcPipelinePath selectedPath = CalcPipelinePath::LegacyRegex;
    std::string emittedCode;
    AscendCPlan plan;
    std::vector<CalcSemanticFinding> analysisFindings;
    std::optional<CalcSemanticFamily> dominantFamily;
    std::optional<CalcFamilyPayload> dominantFamilyPayload;
    std::optional<StructuredShiftedMapUpdatePattern> structuredShiftedMapUpdate;
    std::optional<StructuredMultiViewExprPattern> structuredMultiViewExpr;
};

CalcMlirPipelineOptions ResolveCalcMlirPipelineOptions();

CalcMlirPipelineResult RunCalcMlirPipeline(const CalcMlirPipelineInput& input,
                                           const CalcMlirPipelineOptions& options,
                                           const LegacyCalcLoweringFn& legacyLowering);

std::string RunCalcMlirPipelineFacade(const CalcMlirPipelineInput& input,
                                      const CalcMlirPipelineOptions& options,
                                      const LegacyCalcLoweringFn& legacyLowering,
                                      CalcPipelinePath* selectedPath = nullptr);

} // namespace ascendc_mlir
} // namespace dacppTranslator

#endif // TRANSLATOR_REWRITER_ASCENDC_MLIR_CALC_MLIR_PIPELINE_H
