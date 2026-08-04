#ifndef TRANSLATOR_REWRITER_ASCENDC_KERNEL_SUPPORT_H
#define TRANSLATOR_REWRITER_ASCENDC_KERNEL_SUPPORT_H

#include <string>
#include <vector>

#include "ascendc_mlir/CalcMlirPipeline.h"
#include "ascendc_mlir/CalcShellInfo.h"

namespace dacppTranslator {
class Calc;

namespace ascendc {

std::string BuildDeterministicScalarFallback(const std::string& tileLengthVar,
                                             const std::vector<std::string>& inLocalVars,
                                             const std::vector<std::string>& outLocalVars);

std::string NormalizeRedundantDirectViewCopy(const std::string& code);

int InferExtraScratchBufferCountFromExpr(const std::string& primaryExpr,
                                         const std::string& secondaryExpr);

dacppTranslator::ascendc_mlir::CalcMlirPipelineResult RunCalcMlirPipelineDetailed(
    Calc* calc,
    const std::string& sourceText,
    const std::string& tileLengthVar,
    const std::vector<std::string>& inLocalVars,
    const std::vector<std::string>& outLocalVars,
    const std::vector<std::string>& shellParamOrder,
    const dacppTranslator::ascendc_mlir::CalcShellSummary& shellSummary,
    const std::vector<dacppTranslator::ascendc_mlir::CalcParamBinding>* paramBindings);

} // namespace ascendc
} // namespace dacppTranslator

#endif // TRANSLATOR_REWRITER_ASCENDC_KERNEL_SUPPORT_H
