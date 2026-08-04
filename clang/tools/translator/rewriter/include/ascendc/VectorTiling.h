#ifndef TRANSLATOR_REWRITER_ASCENDC_VECTOR_TILING_H
#define TRANSLATOR_REWRITER_ASCENDC_VECTOR_TILING_H

#include "ascendc/AscendCPolicy.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dacppTranslator {
namespace ascendc {

enum class UbAllocationKind {
    InputQueue,
    OutputQueue,
    CalcBuffer,
    BroadcastQueue,
    HaloQueue
};

struct VectorUbAllocation {
    std::string name;
    UbAllocationKind kind = UbAllocationKind::CalcBuffer;
    uint32_t elementBytes = 0;
    uint32_t multiplicity = 1;
    uint32_t leftHaloElements = 0;
    uint32_t rightHaloElements = 0;
    bool usesQueueDepth = false;
};

struct VectorKernelResourceModel {
    std::vector<VectorUbAllocation> allocations;
    uint32_t tileAlignElements = 1;
    size_t fixedBytes = 0;
};

struct VectorUbAllocationEstimate {
    std::string name;
    UbAllocationKind kind = UbAllocationKind::CalcBuffer;
    size_t bytes = 0;
};

struct VectorKernelTilingPlan {
    uint32_t bufferNum = 1;
    uint32_t tileAlignElements = 1;
    uint32_t maxTileElements = 1;
    size_t ubBudgetBytes = 0;
    size_t usableUbBytes = 0;
    size_t estimatedBytesAtMaxTile = 0;
    std::string selectionSource;
    std::vector<VectorUbAllocationEstimate> breakdown;
};

struct VectorRuntimeTiling {
    uint32_t totalLength = 0;
    uint32_t coreNum = 1;
    uint32_t alignedKernelLength = 0;
    uint32_t blockLength = 0;
    uint32_t tileLength = 0;
    uint32_t tileCount = 0;
};

VectorKernelTilingPlan BuildVectorKernelTilingPlan(
    const VectorKernelResourceModel& model,
    const TranslationPolicy& policy);

VectorRuntimeTiling BuildVectorRuntimeTiling(
    size_t kernelLength,
    size_t blockDim,
    const VectorKernelTilingPlan& plan);

std::string FormatVectorTilingDiagnostics(
    const VectorKernelResourceModel& model,
    const VectorKernelTilingPlan& plan);

const char* UbAllocationKindName(UbAllocationKind kind);

} // namespace ascendc
} // namespace dacppTranslator

#endif
