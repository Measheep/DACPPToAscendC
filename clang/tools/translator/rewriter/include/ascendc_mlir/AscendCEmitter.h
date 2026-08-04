#ifndef TRANSLATOR_REWRITER_ASCENDC_MLIR_ASCENDC_EMITTER_H
#define TRANSLATOR_REWRITER_ASCENDC_MLIR_ASCENDC_EMITTER_H

#include <string>
#include <unordered_map>
#include <vector>

#include "ascendc_mlir/AscendCPlan.h"
#include "ascendc_mlir/CalcExprGraph.h"

namespace dacppTranslator {
namespace ascendc_mlir {

struct AscendCEmitterOptions {
    bool emitReasonComments = true;
    bool allowScalarBoundaryFallback = true;
    const CalcExprGraph* exprGraph = nullptr;
};

std::string EmitStructuredMapExpr(const StructuredShiftedMapUpdatePattern& pattern,
                                  bool shiftedInput,
                                  const std::string& tileLengthVar,
                                  int* outScratchBufferCount = nullptr);

std::string EmitStructuredUpdateExpr(const StructuredShiftedMapUpdatePattern& pattern,
                                     bool emitPostExpr,
                                     const std::string& tileLengthVar,
                                     int* outScratchBufferCount = nullptr);

std::string EmitStructuredMultiViewMainExpr(const StructuredMultiViewExprPattern& pattern,
                                            const std::string& tileLengthVar,
                                            int* outScratchBufferCount = nullptr);

std::string EmitStructuredMultiViewPostExpr(const StructuredMultiViewExprPattern& pattern,
                                            const std::string& tileLengthVar,
                                            int* outScratchBufferCount = nullptr);

std::string EmitExprTreeToAscendC(const CalcExprGraph& graph,
                                  int rootNodeId,
                                  const std::string& dstLocalTensor,
                                  const std::string& auxLocalTensor,
                                  const std::string& tileLengthVar,
                                  const std::unordered_map<int, std::string>& loadNodeLocalTensors,
                                  bool* outSuccess = nullptr);

std::string EmitPointwiseStage(const AscendCPlanStep& step,
                               const std::string& tileLengthVar,
                               const AscendCEmitterOptions& options = {});

std::string EmitWeightedWindowStage(const AscendCPlanStep& step,
                                    const std::string& tileLengthVar,
                                    const AscendCEmitterOptions& options = {});

std::string EmitCoeffStencil(const AscendCPlanStep& step,
                     const std::string& tileLengthVar,
                     const AscendCEmitterOptions& options = {});

std::string EmitDecayExpMulScalarFastPath(const AscendCPlanStep& step,
                                          const std::string& tileLengthVar,
                                          const AscendCEmitterOptions& options = {});

std::string EmitShiftedElementwise(const AscendCPlanStep& step,
                         const std::string& tileLengthVar,
                         const AscendCEmitterOptions& options = {});

std::string EmitClamp(const AscendCPlanStep& step,
                const std::string& tileLengthVar,
                const AscendCEmitterOptions& options = {});

std::string EmitScalarFallback(const AscendCPlanStep& step,
                               const std::string& tileLengthVar);

std::string EmitAscendCFromPlan(const AscendCPlan& plan,
                                const std::string& tileLengthVar,
                                const AscendCEmitterOptions& options = {});

} // namespace ascendc_mlir
} // namespace dacppTranslator

#endif // TRANSLATOR_REWRITER_ASCENDC_MLIR_ASCENDC_EMITTER_H
