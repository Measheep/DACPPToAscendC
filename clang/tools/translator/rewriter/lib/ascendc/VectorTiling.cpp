#include "ascendc/VectorTiling.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace dacppTranslator {
namespace ascendc {
namespace {

size_t CheckedAdd(size_t a, size_t b, const char* what) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        throw std::overflow_error(std::string("Vector tiling overflow while adding ") + what);
    }
    return a + b;
}

size_t CheckedMul(size_t a, size_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::overflow_error(std::string("Vector tiling overflow while multiplying ") + what);
    }
    return a * b;
}

size_t AlignUp(size_t value, size_t align) {
    if (align == 0) {
        throw std::invalid_argument("Vector tiling alignment must be non-zero");
    }
    const size_t remainder = value % align;
    return remainder == 0 ? value : CheckedAdd(value, align - remainder, "alignment");
}

uint32_t CheckedLcm(uint32_t a, uint32_t b) {
    if (a == 0 || b == 0) {
        throw std::invalid_argument("Vector tiling LCM operands must be non-zero");
    }
    const uint32_t divisor = std::gcd(a, b);
    const uint64_t result = static_cast<uint64_t>(a / divisor) * b;
    if (result > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("Vector tiling element alignment LCM overflow");
    }
    return static_cast<uint32_t>(result);
}

uint32_t ComputeAlignment(const VectorKernelResourceModel& model) {
    uint32_t alignment = model.tileAlignElements == 0 ? 1 : model.tileAlignElements;
    for (const VectorUbAllocation& allocation : model.allocations) {
        if (allocation.elementBytes == 0) {
            throw std::invalid_argument("Vector UB allocation '" + allocation.name +
                                        "' has zero-sized elements");
        }
        // AscendC DataCopy requires a 32-byte transfer granularity. LCM keeps
        // mixed element types safe without relying on the 2/4/8-byte shortcut.
        const uint32_t transferAlign =
            32u / std::gcd(32u, allocation.elementBytes);
        alignment = CheckedLcm(alignment, transferAlign);
    }
    return alignment;
}

size_t EstimateAllocation(const VectorUbAllocation& allocation,
                          uint32_t tileElements,
                          uint32_t bufferNum) {
    size_t elements = tileElements;
    elements = CheckedAdd(elements, allocation.leftHaloElements, "left halo");
    elements = CheckedAdd(elements, allocation.rightHaloElements, "right halo");
    const size_t elementAlign = 32u / std::gcd(32u, allocation.elementBytes);
    elements = AlignUp(elements, elementAlign);
    size_t bytes = CheckedMul(elements, allocation.elementBytes, "allocation bytes");
    bytes = CheckedMul(bytes, allocation.multiplicity, "allocation multiplicity");
    if (allocation.usesQueueDepth) {
        bytes = CheckedMul(bytes, bufferNum, "queue depth");
    }
    return bytes;
}

size_t EstimateTotal(const VectorKernelResourceModel& model,
                     uint32_t tileElements,
                     uint32_t bufferNum,
                     std::vector<VectorUbAllocationEstimate>* breakdown) {
    size_t total = model.fixedBytes;
    if (breakdown) {
        breakdown->clear();
    }
    for (const VectorUbAllocation& allocation : model.allocations) {
        const size_t bytes = EstimateAllocation(allocation, tileElements, bufferNum);
        total = CheckedAdd(total, bytes, "UB allocations");
        if (breakdown) {
            breakdown->push_back({allocation.name, allocation.kind, bytes});
        }
    }
    return total;
}

uint32_t FindMaxTile(const VectorKernelResourceModel& model,
                     uint32_t alignment,
                     uint32_t bufferNum,
                     size_t usableBytes) {
    if (EstimateTotal(model, alignment, bufferNum, nullptr) > usableBytes) {
        return 0;
    }
    uint64_t low = 1;
    uint64_t high = std::numeric_limits<uint32_t>::max() / alignment;
    while (low < high) {
        const uint64_t mid = low + (high - low + 1) / 2;
        const uint32_t tile = static_cast<uint32_t>(mid * alignment);
        bool fits = false;
        try {
            fits = EstimateTotal(model, tile, bufferNum, nullptr) <= usableBytes;
        } catch (const std::overflow_error&) {
            fits = false;
        }
        if (fits) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    return static_cast<uint32_t>(low * alignment);
}

} // namespace

const char* UbAllocationKindName(UbAllocationKind kind) {
    switch (kind) {
    case UbAllocationKind::InputQueue: return "input-queue";
    case UbAllocationKind::OutputQueue: return "output-queue";
    case UbAllocationKind::CalcBuffer: return "calc-buffer";
    case UbAllocationKind::BroadcastQueue: return "broadcast-queue";
    case UbAllocationKind::HaloQueue: return "halo-queue";
    }
    return "unknown";
}

VectorKernelTilingPlan BuildVectorKernelTilingPlan(
    const VectorKernelResourceModel& model,
    const TranslationPolicy& policy) {
    if (policy.vectorUbBudgetBytes == 0 || policy.vectorUbReserveBytes == 0 ||
        policy.vectorUbReserveBytes >= policy.vectorUbBudgetBytes) {
        throw std::invalid_argument(
            "Vector UB budget and reserve must be positive, with reserve smaller than budget");
    }
    const uint32_t alignment = ComputeAlignment(model);
    const size_t usableBytes = policy.vectorUbBudgetBytes - policy.vectorUbReserveBytes;

    uint32_t bufferNum = 0;
    std::string source;
    auto CanFitMinimum = [&](uint32_t depth) {
        return EstimateTotal(model, alignment, depth, nullptr) <= usableBytes;
    };
    switch (policy.bufferDepthMode) {
    case BufferDepthMode::Auto:
        if (CanFitMinimum(2)) {
            bufferNum = 2;
        } else if (CanFitMinimum(1)) {
            bufferNum = 1;
        }
        source = "auto (" + policy.bufferDepthSource + ")";
        break;
    case BufferDepthMode::Single:
        bufferNum = CanFitMinimum(1) ? 1 : 0;
        source = "explicit-1 (" + policy.bufferDepthSource + ")";
        break;
    case BufferDepthMode::Double:
        bufferNum = CanFitMinimum(2) ? 2 : 0;
        source = "explicit-2 (" + policy.bufferDepthSource + ")";
        break;
    }
    source += ", ub-budget=" + policy.vectorUbBudgetSource +
              ", ub-reserve=" + policy.vectorUbReserveSource;
    if (bufferNum == 0) {
        std::ostringstream message;
        message << "Vector kernel cannot fit one aligned tile in usable UB: budget="
                << policy.vectorUbBudgetBytes << ", reserve="
                << policy.vectorUbReserveBytes << ", usable=" << usableBytes
                << ", tileAlignElements=" << alignment;
        if (model.fixedBytes != 0) {
            message << "\n  fixed: " << model.fixedBytes << " bytes";
        }
        for (const VectorUbAllocation& allocation : model.allocations) {
            message << "\n  " << allocation.name << " ("
                    << UbAllocationKindName(allocation.kind) << "): "
                    << EstimateAllocation(allocation, alignment,
                                          policy.bufferDepthMode == BufferDepthMode::Double ? 2 : 1)
                    << " bytes";
        }
        throw std::runtime_error(message.str());
    }

    uint32_t maxTile = FindMaxTile(model, alignment, bufferNum, usableBytes);
    if (policy.maxTileElementsOverride != 0) {
        const uint32_t alignedOverride =
            policy.maxTileElementsOverride - policy.maxTileElementsOverride % alignment;
        if (alignedOverride == 0) {
            throw std::invalid_argument(
                "Explicit max tile elements is smaller than the required alignment");
        }
        maxTile = std::min(maxTile, alignedOverride);
        source += ", tile-limit (" + policy.maxTileElementsSource + ")";
        if (alignedOverride != policy.maxTileElementsOverride) {
            source += " aligned-down " +
                std::to_string(policy.maxTileElementsOverride) + "->" +
                std::to_string(alignedOverride);
        }
    }

    VectorKernelTilingPlan plan;
    plan.bufferNum = bufferNum;
    plan.tileAlignElements = alignment;
    plan.maxTileElements = maxTile;
    plan.ubBudgetBytes = policy.vectorUbBudgetBytes;
    plan.usableUbBytes = usableBytes;
    plan.selectionSource = source;
    plan.estimatedBytesAtMaxTile =
        EstimateTotal(model, maxTile, bufferNum, &plan.breakdown);
    return plan;
}

VectorRuntimeTiling BuildVectorRuntimeTiling(
    size_t kernelLength,
    size_t blockDim,
    const VectorKernelTilingPlan& plan) {
    if (plan.tileAlignElements == 0 || plan.maxTileElements == 0) {
        throw std::invalid_argument("Vector runtime tiling requires a valid compile-time plan");
    }
    const size_t coreNum = blockDim == 0 ? 1 : blockDim;
    const size_t combinedAlign = CheckedMul(coreNum, plan.tileAlignElements,
                                            "core/tile alignment");
    const size_t alignedLength = kernelLength == 0 ? 0 : AlignUp(kernelLength, combinedAlign);
    const size_t blockLength = coreNum == 0 ? 0 : alignedLength / coreNum;
    size_t tileLength = std::min(blockLength, static_cast<size_t>(plan.maxTileElements));
    if (tileLength != 0) {
        tileLength -= tileLength % plan.tileAlignElements;
        tileLength = std::max(tileLength, static_cast<size_t>(plan.tileAlignElements));
    }
    const size_t tileCount = tileLength == 0 ? 0 :
        (blockLength - 1) / tileLength + 1;
    auto ToU32 = [](size_t value, const char* name) {
        if (value > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error(std::string("Vector runtime ") + name +
                                      " exceeds uint32_t");
        }
        return static_cast<uint32_t>(value);
    };
    return {ToU32(alignedLength, "totalLength"), ToU32(coreNum, "coreNum"),
            ToU32(alignedLength, "alignedKernelLength"),
            ToU32(blockLength, "blockLength"), ToU32(tileLength, "tileLength"),
            ToU32(tileCount, "tileCount")};
}

std::string FormatVectorTilingDiagnostics(
    const VectorKernelResourceModel& model,
    const VectorKernelTilingPlan& plan) {
    std::ostringstream out;
    out << "UB budget: " << plan.ubBudgetBytes << " bytes\n"
        << "UB reserve: " << (plan.ubBudgetBytes - plan.usableUbBytes) << " bytes\n"
        << "selected BUFFER_NUM: " << plan.bufferNum << "\n"
        << "tile alignment: " << plan.tileAlignElements << " elements\n"
        << "max tile elements: " << plan.maxTileElements << "\n"
        << "bytes per tile estimate: " << plan.estimatedBytesAtMaxTile << "\n"
        << "selection source: " << plan.selectionSource << "\n"
        << "allocation breakdown:\n";
    if (model.fixedBytes != 0) {
        out << "  fixed: " << model.fixedBytes << " bytes\n";
    }
    for (const VectorUbAllocationEstimate& allocation : plan.breakdown) {
        out << "  " << allocation.name << " ("
            << UbAllocationKindName(allocation.kind) << "): "
            << allocation.bytes << " bytes\n";
    }
    return out.str();
}

} // namespace ascendc
} // namespace dacppTranslator
