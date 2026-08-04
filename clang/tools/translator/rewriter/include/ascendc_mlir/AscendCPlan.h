#ifndef TRANSLATOR_REWRITER_ASCENDC_MLIR_ASCENDC_PLAN_H
#define TRANSLATOR_REWRITER_ASCENDC_MLIR_ASCENDC_PLAN_H

#include <string>
#include <vector>

#include "ascendc_mlir/CalcAnalysis.h"

namespace dacppTranslator {
namespace ascendc_mlir {

struct CalcShellSummary;

enum class AscendCPlanStepKind {
    StructuredShiftedMapUpdate,
    ElementwiseBinary,
    DecayExpMul,
    CoeffStencil,
    ShiftedElementwise,
    WindowLoad,
    PointwiseExpr,
    ClampZero,
    Shift,
    Q,
    ShiftedQ,
    Diff,
    Scale,
    Sub,
    Clamp,
    CubeMatmul,
    LinearReduction,
    ScalarFallback,

    // Backward-compatible alias for legacy emitter code paths.
    Stencil = CoeffStencil,
};

enum class AscendCBinaryOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Unknown,
};

enum class AscendCBoundaryPolicy {
    Replicate,
    Clamp,
    RequireHalo,
};

enum class AscendCTransferPolicy {
    PointwiseDirect,
    SpatialWindow,
    TemporalLoopCarried,
    BroadcastScalar,
    CubeMatmul,
    ScalarFallback,
};

struct LinearReductionStepPayload {
    std::string inputTensor;
    std::string weightTensor;
    std::string biasTensor;
    std::string outputTensor;
    int K = 0;
    bool hasBias = true;
};

struct CubeMatmulStepPayload {
    std::string srcA;
    std::string srcB;
    std::string dst;
    bool accumulateToDst = true;
    bool hasInitialDst = true;

    std::string dimM;
    std::string dimN;
    std::string dimK;
    std::string dimMSource;
    std::string dimNSource;
    std::string dimKSource;

    bool transA = false;
    bool transB = false;

    std::string userTypeA;
    std::string userTypeB;
    std::string userTypeC;
    std::string kernelTypeA;
    std::string kernelTypeB;
    std::string kernelTypeC;
};

struct AscendCPlanStep {
    AscendCPlanStepKind kind = AscendCPlanStepKind::ScalarFallback;
    CalcSemanticFamily family = CalcSemanticFamily::ScalarFallbackFamily;
    AscendCTransferPolicy transferPolicy = AscendCTransferPolicy::ScalarFallback;
    AscendCBinaryOpKind binaryOp = AscendCBinaryOpKind::Unknown;
    int stencilRadius = 0;
    int leftHalo = 0;
    int rightHalo = 0;
    int validLength = 0;
    int copyLength = 0;
    std::vector<int> inputNodeIds;
    int exprRootNodeId = -1;
    std::string exprRole;

    std::string dst;
    std::string srcA;
    std::string srcB;
    std::string srcC;

    std::vector<int> windowOffsets;
    std::vector<std::string> windowCoeffs;
    std::string scalarTensor;
    double scalarImm = 0.0;
    std::string tempTensor;
    AscendCBoundaryPolicy boundaryPolicy = AscendCBoundaryPolicy::Clamp;

    std::string reason;
    std::string loopBody;

    bool hasPointwisePayload = false;
    PointwiseFamilyPayload pointwisePayload;
    bool hasFamilyPayload = false;
    CalcFamilyPayload familyPayload;
    bool hasCubeMatmulPayload = false;
    CubeMatmulPayload cubeMatmulPayload;
    bool hasCubeMatmulStepPayload = false;
    CubeMatmulStepPayload cubeMatmulStepPayload;
    bool hasLinearReductionPayload = false;
    LinearReductionStepPayload linearReductionPayload;

    StructuredShiftedMapUpdatePattern structuredShiftedMapUpdate;
    StructuredMultiViewExprPattern structuredMultiViewExpr;
};

struct AscendCPlanBuildOptions {
    const CalcShellSummary* shellSummary = nullptr;
};

struct AscendCPlan {
    bool reusable = false;
    std::vector<AscendCPlanStep> steps;
    std::vector<std::string> notes;
};

AscendCPlan BuildAscendCPlan(const CalcAnalysisResult& analysis,
                             const CalcMlirArtifact& artifact,
                             const std::vector<std::string>& shellParamOrder,
                             const std::vector<std::string>& inLocalVars,
                             const std::vector<std::string>& outLocalVars,
                             const AscendCPlanBuildOptions& options);

std::string DumpAscendCPlan(const AscendCPlan& plan);

const char* ToString(AscendCPlanStepKind kind);
const char* ToString(AscendCBinaryOpKind kind);
const char* ToString(AscendCBoundaryPolicy policy);
const char* ToString(AscendCTransferPolicy policy);

} // namespace ascendc_mlir
} // namespace dacppTranslator

#endif // TRANSLATOR_REWRITER_ASCENDC_MLIR_ASCENDC_PLAN_H
