#ifndef TRANSLATOR_REWRITER_ASCENDC_MLIR_CALC_ANALYSIS_H
#define TRANSLATOR_REWRITER_ASCENDC_MLIR_CALC_ANALYSIS_H

#include <string>
#include <vector>

#include "ascendc_mlir/CalcAstToMlir.h"
#include "ascendc_mlir/CalcExprGraph.h"

namespace dacppTranslator {
namespace ascendc_mlir {

struct CalcShellSummary;

enum class CalcSemanticCategory {
    StructuredShiftedMapUpdate,
    ElementwiseBinary,
    DecayExpMul,
    CoeffStencil,
    WindowedPointwise,
    ShiftedElementwise,
    PointwiseExpr,
    Clamp,
    MatmulReduction,
    Reduction,
    LinearReduction,
    ScalarFallback,

    // Backward-compatible aliases for legacy code paths.
    Stencil = CoeffStencil,
    ComplexCall = ScalarFallback,
    ControlFlowFallback = ScalarFallback,
};

enum class CalcSemanticFamily {
    PointwiseFamily,
    SpatialStencilFamily,
    TemporalStencilFamily,
    CubeMatmulFamily,
    ReductionFamily,
    LinearReductionFamily,
    ScalarFallbackFamily,
};

enum class CalcBoundaryPolicy {
    Replicate,
    Clamp,
    RequireHalo,
};

struct StructuredShiftedMapUpdatePattern {
    std::string srcTensor;
    std::string dstTensor;
    int shiftOffset = 0;
    bool hasMap = false;
    bool hasPost = false;
    CalcBoundaryPolicy boundaryPolicy = CalcBoundaryPolicy::Clamp;

    // Normalized expression templates.
    // mapExpr placeholder: RawInput
    // updateExpr placeholders: CenterRaw / ShiftedRaw / CenterMapped / ShiftedMapped
    // postExpr placeholder: UpdateResult
    std::string mapExpr;
    std::string updateExpr;
    std::string postExpr;
};

struct StructuredMultiViewExprPattern {
    std::string srcTensor;
    std::string dstTensor;
    std::vector<int> viewOffsets;
    std::vector<std::string> scalarInputs;
    std::string mainExpr;
    bool hasPost = false;
    std::string postExpr;
    CalcBoundaryPolicy boundaryPolicy = CalcBoundaryPolicy::Clamp;
    int leftHalo = 0;
    int rightHalo = 0;
    int validLength = 0;
    int copyLength = 0;
};

struct PointwiseFamilyPayload {
    CalcSemanticCategory operation = CalcSemanticCategory::ScalarFallback;
};

struct CalcFamilyPayload {
    std::string srcTensor;
    std::string dstTensor;
    std::vector<int> offsets;
    std::vector<std::string> scalarInputs;
    std::string mainExpr;
    bool hasPostExpr = false;
    std::string postExpr;
    bool loopCarried = false;
    CalcBoundaryPolicy boundaryPolicy = CalcBoundaryPolicy::Clamp;
    int leftHalo = 0;
    int rightHalo = 0;
    int validLength = 0;
    int copyLength = 0;
};

struct LinearReductionPayload {
    std::string inputTensor;
    std::string weightTensor;
    std::string biasTensor;
    std::string outputTensor;
    int K = 0;
    bool hasBias = true;
};

struct CubeMatmulPayload {
    std::string srcA;
    std::string srcB;
    std::string dst;
    bool accumulateToDst = true;
    bool hasInitialDst = true;
    std::string logicalInputType;
    std::string kernelInputType;
    std::string kernelOutputType;
};

struct AffineAccess {
    std::string tensor;
    int offset = 0;
    double coeff = 1.0;
};

struct CalcSemanticFinding {
    CalcSemanticCategory category = CalcSemanticCategory::ScalarFallback;
    CalcSemanticFamily family = CalcSemanticFamily::ScalarFallbackFamily;
    std::string target;
    std::vector<AffineAccess> inputs;
    std::string reason;
    bool hasPointwisePayload = false;
    PointwiseFamilyPayload pointwisePayload;
    bool hasFamilyPayload = false;
    CalcFamilyPayload familyPayload;
    bool hasCubeMatmulPayload = false;
    CubeMatmulPayload cubeMatmulPayload;
    bool hasLinearReductionPayload = false;
    LinearReductionPayload linearReductionPayload;
    bool hasStructuredShiftedMapUpdate = false;
    StructuredShiftedMapUpdatePattern structuredShiftedMapUpdate;
    bool hasStructuredMultiViewExpr = false;
    StructuredMultiViewExprPattern structuredMultiViewExpr;
};

struct CalcAnalysisResult {
    bool analyzed = false;
    std::vector<CalcSemanticFinding> findings;
    std::vector<std::string> notes;
    CalcExprGraph exprGraph;
};

struct CalcAnalysisOptions {
    const CalcShellSummary* shellSummary = nullptr;
    const std::vector<std::string>* paramOrder = nullptr;
};

CalcAnalysisResult AnalyzeCalcMlirModule(const CalcMlirArtifact& artifact,
                                         const CalcAnalysisOptions& options);

std::string RenderCalcAnalysisPlan(const CalcAnalysisResult& result);

const char* ToString(CalcSemanticCategory category);
const char* ToString(CalcSemanticFamily family);
const char* ToString(CalcBoundaryPolicy policy);

} // namespace ascendc_mlir
} // namespace dacppTranslator

#endif // TRANSLATOR_REWRITER_ASCENDC_MLIR_CALC_ANALYSIS_H
