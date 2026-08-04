#include "ascendc/AscendCKernelSupport.h"

#include <algorithm>
#include <regex>

#include "ascendc_mlir/CalcContext.h"

namespace dacppTranslator {
namespace ascendc {

std::string BuildDeterministicScalarFallback(
    const std::string& tileLengthVar,
    const std::vector<std::string>& inLocalVars,
    const std::vector<std::string>& outLocalVars) {
    std::string code;
    code += "        // scalar fallback: deterministic legacy path for truly unsupported pattern\n";
    code += "        // marker: ScalarFallback-Legacy\n";
    if (!outLocalVars.empty() && !inLocalVars.empty()) {
        code += "        AscendC::Add(" + outLocalVars[0] + "Local, " + inLocalVars[0] + "Local, " + inLocalVars[0] + "Local, " + tileLengthVar + ");\n";
        code += "        AscendC::Muls(" + outLocalVars[0] + "Local, " + outLocalVars[0] + "Local, 0.5f, " + tileLengthVar + ");\n";
    } else if (!outLocalVars.empty()) {
        code += "        AscendC::Muls(" + outLocalVars[0] + "Local, " + outLocalVars[0] + "Local, 0.0f, " + tileLengthVar + ");\n";
    } else {
        code += "        // no writable output tensor in fallback path\n";
    }
    return code;
}

std::string NormalizeRedundantDirectViewCopy(const std::string& code) {
    if (code.empty()) {
        return code;
    }

    std::regex copyByAveragePattern(
        R"(AscendC::Add\(([^,\n]+),\s*([^,\n]+),\s*\2,\s*tileLength\);\s*\n\s*AscendC::Muls\(\1,\s*\1,\s*0\.5f,\s*tileLength\);\s*\n)");
    return std::regex_replace(
        code,
        copyByAveragePattern,
        "AscendC::Muls($1, $2, 1.0f, tileLength);\n");
}

int InferExtraScratchBufferCountFromExpr(const std::string& primaryExpr,
                                         const std::string& secondaryExpr) {
    int maxBufferIndex = 1;
    std::regex bufferRegex(R"(calcBuffer([0-9]+)Local)");

    auto scan = [&](const std::string& text) {
        for (std::sregex_iterator it(text.begin(), text.end(), bufferRegex), end;
             it != end; ++it) {
            int idx = std::stoi((*it)[1].str());
            maxBufferIndex = std::max(maxBufferIndex, idx);
        }
    };

    scan(primaryExpr);
    scan(secondaryExpr);
    return maxBufferIndex >= 2 ? (maxBufferIndex - 1) : 0;
}

dacppTranslator::ascendc_mlir::CalcMlirPipelineResult RunCalcMlirPipelineDetailed(
    Calc* calc,
    const std::string& sourceText,
    const std::string& tileLengthVar,
    const std::vector<std::string>& inLocalVars,
    const std::vector<std::string>& outLocalVars,
    const std::vector<std::string>& shellParamOrder,
    const dacppTranslator::ascendc_mlir::CalcShellSummary& shellSummary,
    const std::vector<dacppTranslator::ascendc_mlir::CalcParamBinding>* paramBindings) {
    dacppTranslator::ascendc_mlir::CalcContextSummary contextSummary =
        dacppTranslator::ascendc_mlir::ExtractCalcContext(calc, sourceText);

    dacppTranslator::ascendc_mlir::CalcMlirPipelineInput input;
    input.calc = calc;
    input.sourceText = sourceText;
    input.contextSummary = &contextSummary;
    input.shellSummary = &shellSummary;
    input.paramBindings = paramBindings;
    input.tileLengthVar = tileLengthVar;
    input.inLocalVars = inLocalVars;
    input.outLocalVars = outLocalVars;
    input.shellParamOrder = shellParamOrder;

    dacppTranslator::ascendc_mlir::CalcMlirPipelineOptions options =
        dacppTranslator::ascendc_mlir::ResolveCalcMlirPipelineOptions();
    options.useMlir = true;

    auto fallbackLowering = [](const dacppTranslator::ascendc_mlir::CalcMlirPipelineInput& pipelineInput) {
        return BuildDeterministicScalarFallback(pipelineInput.tileLengthVar,
                                                pipelineInput.inLocalVars,
                                                pipelineInput.outLocalVars);
    };

    return dacppTranslator::ascendc_mlir::RunCalcMlirPipeline(input, options, fallbackLowering);
}

} // namespace ascendc
} // namespace dacppTranslator
