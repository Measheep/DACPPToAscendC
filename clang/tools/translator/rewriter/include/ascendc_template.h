#ifndef ASCENDC_TEMPLATE_H
#define ASCENDC_TEMPLATE_H

#include <string>
#include <cstdint>
#include <vector>

namespace dacppTranslator {
namespace ascendc {
struct HostRuntimeRequirements;
}
}

namespace ASCENDC_TEMPLATE {

struct TemporalLoopSchedule {
    bool recognized = false;
    bool requireNoFallback = false;
    std::string failureReason;

    int expressionIndex = -1;

    // Loop schedule metadata.
    std::string ownerFunctionName;
    std::string loopIndexVar;
    std::string loopBeginExpr;
    std::string loopUpperBoundExpr;
    std::string loopStepExpr;

    // State/input-output binding.
    std::string stateTensorName;
    std::string inputSliceVar;
    std::string outputSliceVar;
    int inputTimeOffset = 0;
    int outputTimeOffset = 1;

    // Loop-carried writeback relation.
    bool hasLoopCarriedWriteback = false;
    std::string writebackLoopIndexVar;
    std::string writebackRangeBeginExpr;
    std::string writebackRangeEndExpr;
    int writebackDstSpatialOffset = 0;
    int writebackDstTimeOffset = 1;
    int writebackSrcSpatialOffset = -1;

    // Boundary update metadata (optional).
    bool hasBoundaryUpdate = false;
    std::string boundaryUpdateSummary;

    // Scalar binding metadata.
    std::vector<std::string> scalarParamNames;
    std::vector<std::string> scalarArgNames;
};

using TemporalStencilHostSpec = TemporalLoopSchedule;

// Host 侧模板接口
// Vector Kernel 侧模板接口
std::string CodeGen_KernelFile(
    std::string kernelName,
    std::string gmAddrParams,
    std::string paramsNames,
    std::string hostWrapperParams,
    std::string initCode,
    std::string copyInCode,
    std::string computeCode,
    std::string copyOutCode,
    std::string tQueDecl,
    int vectorComputeAlignBytes = 256,
    uint32_t bufferNum = 1,
    uint32_t tileAlignElements = 1,
    uint32_t maxTileElements = 1,
    std::string tilingDiagnostics = "");

struct StructuredShiftedMapUpdateFastPathCode {
    std::string initCode;
    std::string copyInCode;
    std::string computeCode;
    std::string copyOutCode;
    std::string tQueDecl;
};

struct StructuredMultiViewExprFastPathCode {
    std::string initCode;
    std::string copyInCode;
    std::string computeCode;
    std::string copyOutCode;
    std::string tQueDecl;
};

StructuredShiftedMapUpdateFastPathCode CodeGen_StructuredShiftedMapUpdateFastPath(
    std::string srcKernelType,
    std::string dstKernelType,
    std::string srcName,
    std::string dstName,
    int shiftOffset,
    int scratchBufferCount,
    std::string mapCenterCode,
    std::string mapShiftCode,
    std::string updateCode,
    std::string postCode,
    int vectorComputeAlignBytes = 256);

StructuredMultiViewExprFastPathCode CodeGen_StructuredMultiViewExprFastPath(
    std::string srcKernelType,
    std::string dstKernelType,
    std::string scalarKernelType,
    std::string srcName,
    std::string dstName,
    std::string scalarName,
    const std::vector<int>& viewOffsets,
    bool hasScalarInput,
    int scratchBufferCount,
    std::string mainExprCode,
    std::string postExprCode,
    int vectorComputeAlignBytes = 256);

// Cube 模板接口
std::string CodeGen_CubeKernelFile(
    std::string kernelName,
    std::string typeA, std::string typeB, std::string typeC,
    std::string nameA, std::string nameB, std::string nameC,
    std::string transA, std::string transB,
    std::string isBias = "false");

std::string CodeGen_CubeTilingFunc(
    std::string tilingName,
    std::string shapeTilingName,
    std::string dimM, std::string dimN, std::string dimK,
    std::string typeA, std::string typeB, std::string typeC,
    std::string transA, std::string transB,
    std::string isBias = "false");

std::string CodeGen_CubeTilingFile(std::string tilingFuncs);

// Vector/Cube Host 文件模板接口
std::string CodeGen_VectorMainBody(
    std::string hostFlowBody,
    std::string sessionName);
std::string CodeGen_VectorHostFile(
    std::string vecExternDoDecls,
    std::string vecPreludeCode,
    std::string vectorMainBody,
    const dacppTranslator::ascendc::HostRuntimeRequirements& runtimeRequirements);

std::string CodeGen_VectorTemporalStencilSchedule(
    const TemporalLoopSchedule& spec,
    std::string launchEntry,
    std::string blockDimVar,
    std::string launchArgs,
    std::string inputTensorName,
    std::string outputTensorName,
    std::string deviceName,
    std::string streamName);

std::string CodeGen_CubeMainBody(
    std::string hostFlowBody,
    std::string sessionName);
std::string CodeGen_CubeHostFile(
    std::string cubeExternDoDecls,
    std::string cubeTilingDecls,
    std::string cubeMainBody,
    std::string cubePreludeCode,
    const dacppTranslator::ascendc::HostRuntimeRequirements& runtimeRequirements);

} // namespace ASCENDC_TEMPLATE

#endif
