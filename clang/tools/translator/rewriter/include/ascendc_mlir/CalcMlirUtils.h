#ifndef TRANSLATOR_REWRITER_ASCENDC_MLIR_CALC_MLIR_UTILS_H
#define TRANSLATOR_REWRITER_ASCENDC_MLIR_CALC_MLIR_UTILS_H

#include <functional>
#include <optional>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "ascendc_mlir/CalcAnalysis.h"
#include "ascendc_mlir/CalcExprGraph.h"
#include "ascendc_mlir/CalcShellInfo.h"

#if defined(DACPP_TRANSLATOR_HAS_MLIR)
#include "mlir/IR/Value.h"
#endif

namespace dacppTranslator {
namespace ascendc_mlir {

#if defined(DACPP_TRANSLATOR_HAS_MLIR)
bool IsArgMemref(mlir::Value memref);
std::string ValueToken(mlir::Value value);
#endif

bool TryParseArgToken(const std::string& token, size_t& outIndex);

const CalcShellParamInfo* FindShellParam(const CalcShellSummary* shellSummary,
                                         const std::string& tensor);

int ExprPrecedence(CalcExprNodeKind kind);

std::string FormatFloatConst(double value);

std::string ResolveArgIndexToTensorName(size_t argIdx,
                                        const std::vector<std::string>& paramOrder,
                                        const CalcShellSummary* shellSummary);

std::string ResolveTensorNameFromToken(const std::string& token,
                                       const std::vector<std::string>& paramOrder,
                                       const CalcShellSummary* shellSummary);

#if defined(DACPP_TRANSLATOR_HAS_MLIR)
std::string ResolveTensorNameFromValue(mlir::Value value,
                                       const std::vector<std::string>& paramOrder,
                                       const CalcShellSummary* shellSummary);
#endif

void InferHaloAndLengths(const CalcShellSummary* shellSummary,
                         const std::string& tensor,
                         const std::set<int>& offsets,
                         int& outLeftHalo,
                         int& outRightHalo,
                         int& outValidLength,
                         int& outCopyLength);

CalcBoundaryPolicy BuildCalcBoundaryPolicy(const std::string& tensor,
                                           const CalcShellSummary* shellSummary);

bool ContainsTensorLoad(const CalcExprGraph& graph,
                        int nodeId,
                        const std::string& tensor);

bool TryGetShape2DFromShell(const CalcShellSummary* shellSummary,
                            const std::string& tensor,
                            int& outDim0,
                            int& outDim1,
                            std::string* outDim0Source = nullptr,
                            std::string* outDim1Source = nullptr);

using ExprNodeRenderer =
    std::function<std::optional<std::string>(int nodeId,
                                             const CalcExprNode& node,
                                             bool& ok)>;

std::string RenderExprCommon(const CalcExprGraph& graph,
                             int nodeId,
                             ExprNodeRenderer renderer,
                             bool& ok,
                             std::unordered_set<int>& visiting,
                             int parentPrec = 0);

} // namespace ascendc_mlir
} // namespace dacppTranslator

#endif // TRANSLATOR_REWRITER_ASCENDC_MLIR_CALC_MLIR_UTILS_H
