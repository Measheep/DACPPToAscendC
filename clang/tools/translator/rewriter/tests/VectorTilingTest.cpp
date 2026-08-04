#include "ascendc/VectorTiling.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace dacppTranslator::ascendc;

namespace {
void Check(bool condition, const char* expression, int line) {
    if (!condition) {
        throw std::runtime_error(std::string("check failed at line ") +
                                 std::to_string(line) + ": " + expression);
    }
}
} // namespace

#undef assert
#define assert(condition) Check(static_cast<bool>(condition), #condition, __LINE__)

namespace dacppTranslator {
namespace ascendc {
// Keep this pure algorithm test independent of Clang/AST linkage. Policy
// parsing itself is exercised by the translator integration checks.
TranslationPolicy::TranslationPolicy() = default;
} // namespace ascendc
} // namespace dacppTranslator

namespace {

TranslationPolicy Policy(size_t budget = 128 * 1024,
                         size_t reserve = 16 * 1024) {
    TranslationPolicy policy;
    policy.vectorUbBudgetBytes = budget;
    policy.vectorUbReserveBytes = reserve;
    policy.bufferDepthMode = BufferDepthMode::Auto;
    policy.maxTileElementsOverride = 0;
    return policy;
}

VectorKernelResourceModel PointwiseModel() {
    VectorKernelResourceModel model;
    model.allocations = {
        {"input", UbAllocationKind::InputQueue, 4, 1, 0, 0, true},
        {"output", UbAllocationKind::OutputQueue, 4, 1, 0, 0, true},
    };
    return model;
}

template <typename Exception, typename Fn>
void ExpectThrows(Fn&& fn) {
    bool threw = false;
    try {
        fn();
    } catch (const Exception&) {
        threw = true;
    }
    assert(threw);
}

} // namespace

int main() {
    {
        const auto plan = BuildVectorKernelTilingPlan(PointwiseModel(), Policy());
        assert(plan.bufferNum == 2);
        assert(plan.tileAlignElements == 8);
        assert(plan.estimatedBytesAtMaxTile <= plan.usableUbBytes);
    }
    {
        VectorKernelResourceModel model = PointwiseModel();
        model.allocations.push_back(
            {"inputs-and-scratch", UbAllocationKind::CalcBuffer, 4, 20, 0, 0, false});
        const auto large = BuildVectorKernelTilingPlan(PointwiseModel(), Policy());
        const auto constrained = BuildVectorKernelTilingPlan(model, Policy());
        assert(constrained.maxTileElements < large.maxTileElements);
    }
    {
        const auto plan = BuildVectorKernelTilingPlan(PointwiseModel(), Policy(116, 16));
        assert(plan.bufferNum == 1);
    }
    {
        ExpectThrows<std::runtime_error>([] {
            (void)BuildVectorKernelTilingPlan(PointwiseModel(), Policy(70, 16));
        });
    }
    {
        VectorKernelResourceModel mixed;
        mixed.allocations = {
            {"half", UbAllocationKind::InputQueue, 2, 1, 0, 0, true},
            {"float", UbAllocationKind::InputQueue, 4, 1, 0, 0, true},
            {"int32", UbAllocationKind::CalcBuffer, 4, 1, 0, 0, false},
            {"int64", UbAllocationKind::OutputQueue, 8, 1, 0, 0, true},
        };
        assert(BuildVectorKernelTilingPlan(mixed, Policy()).tileAlignElements == 16);
    }
    {
        VectorKernelResourceModel halo;
        halo.allocations = {
            {"window", UbAllocationKind::HaloQueue, 4, 1, 1, 2, true},
        };
        auto policy = Policy(1024, 128);
        policy.bufferDepthMode = BufferDepthMode::Single;
        const auto plan = BuildVectorKernelTilingPlan(halo, policy);
        assert(plan.tileAlignElements == 8);
        assert(!plan.breakdown.empty());
        assert(plan.breakdown.front().bytes % 32 == 0);
    }
    {
        auto policy = Policy();
        policy.bufferDepthMode = BufferDepthMode::Single;
        policy.maxTileElementsOverride = 101;
        const auto plan = BuildVectorKernelTilingPlan(PointwiseModel(), policy);
        assert(plan.bufferNum == 1);
        assert(plan.maxTileElements <= 96);
        assert(plan.maxTileElements % plan.tileAlignElements == 0);
    }
    {
        auto policy = Policy(116, 16);
        policy.bufferDepthMode = BufferDepthMode::Double;
        ExpectThrows<std::runtime_error>([&] {
            (void)BuildVectorKernelTilingPlan(PointwiseModel(), policy);
        });
    }
    {
        auto policy = Policy();
        policy.bufferDepthMode = BufferDepthMode::Double;
        assert(BuildVectorKernelTilingPlan(PointwiseModel(), policy).bufferNum == 2);
    }
    {
        VectorKernelResourceModel overflow;
        overflow.allocations = {
            {"huge", UbAllocationKind::CalcBuffer,
             std::numeric_limits<uint32_t>::max(),
             std::numeric_limits<uint32_t>::max(), 0, 0, false},
        };
        ExpectThrows<std::overflow_error>([&] {
            (void)BuildVectorKernelTilingPlan(overflow, Policy());
        });
    }
    {
        VectorKernelResourceModel lcmOverflow;
        lcmOverflow.tileAlignElements = std::numeric_limits<uint32_t>::max();
        lcmOverflow.allocations = {
            {"float", UbAllocationKind::InputQueue, 4, 1, 0, 0, true},
        };
        ExpectThrows<std::overflow_error>([&] {
            (void)BuildVectorKernelTilingPlan(lcmOverflow, Policy());
        });
    }
    {
        auto plan = BuildVectorKernelTilingPlan(PointwiseModel(), Policy());
        plan.maxTileElements = 32;
        const auto runtime = BuildVectorRuntimeTiling(101, 3, plan);
        assert(runtime.alignedKernelLength % (3 * plan.tileAlignElements) == 0);
        assert(runtime.blockLength == runtime.alignedKernelLength / 3);
        assert(runtime.tileLength == 32);
        assert(runtime.tileCount ==
               (runtime.blockLength + runtime.tileLength - 1) / runtime.tileLength);
        const auto empty = BuildVectorRuntimeTiling(0, 0, plan);
        assert(empty.coreNum == 1 && empty.tileLength == 0 && empty.tileCount == 0);
        const auto tiny = BuildVectorRuntimeTiling(1, 1, plan);
        assert(tiny.alignedKernelLength == plan.tileAlignElements);
        assert(tiny.tileLength == plan.tileAlignElements && tiny.tileCount == 1);
        const auto exact = BuildVectorRuntimeTiling(32, 1, plan);
        assert(exact.tileLength == 32 && exact.tileCount == 1);
        const auto tail = BuildVectorRuntimeTiling(65, 1, plan);
        assert(tail.tileCount == 3 && tail.blockLength % plan.tileAlignElements == 0);
        const auto many = BuildVectorRuntimeTiling(64000, 1, plan);
        assert(many.tileCount == 2000);
        ExpectThrows<std::overflow_error>([&] {
            (void)BuildVectorRuntimeTiling(
                static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1,
                1, plan);
        });
    }

    std::cout << "VectorTilingTest passed\n";
    return 0;
}
