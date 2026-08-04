#ifndef TRANSLATOR_REWRITER_ASCENDC_ARTIFACTS_H
#define TRANSLATOR_REWRITER_ASCENDC_ARTIFACTS_H

#include <cstddef>
#include <string>
#include <vector>

#include "ascendc/AscendCFlowPlan.h"
#include "ascendc/HostRuntimeFeatures.h"
#include "ascendc/HostTensorAliasInfo.h"

namespace dacppTranslator {
namespace ascendc {

enum class AscendCRouteKind {
    Vector,
    Cube,
};

// ============================================================
// AscendCExpressionArtifact：DAC 表达式 lowering 的统一产物。
//
// 原 "replay 文本回放" 类型已彻底删除，host body 完全由
// FlowKernelLaunch + FlowGeneratedCodeBlock descriptor 驱动；本结构保留的
// string 字段都是 kernel/tiling 头部及装配模板的必要源。
// ============================================================
struct AscendCExpressionArtifact {
    AscendCRouteKind routeKind = AscendCRouteKind::Vector;
    int expressionIndex = -1;
    size_t sourceOrder = 0;

    // kernel 侧产物（写盘）
    std::string kernelFileName;
    std::string kernelCode;

    // host 侧需要拼装的非-body 文本
    std::string externDoDecl;
    std::string tilingDecl;
    std::string tilingFunc;
    std::string hostPreludeCode;
    // Host body descriptor used by the structured emission path.
    FlowKernelLaunch kernelLaunch;
    std::vector<FlowGeneratedCodeBlock> generatedBlocks;
    // setupBlockIndices/cleanupBlockIndices 引用 generatedBlocks 内的位置；
    // setup 表示与 launch 前置的 alloc/H2D copy，cleanup 表示 launch 后置的 D2H/free。
    // 空时表示 emitter 应按 generatedBlocks 自身顺序依次发射。
    std::vector<std::size_t> setupBlockIndices;
    std::vector<std::size_t> cleanupBlockIndices;

    // 标记：lowering 已把用户 host 侧 while-loop 替换为 translator-generated 的
    // temporal wrapper（嵌在 launchBody 内）。HostFlowBuilder 据此跳过外层 host loop
    // 的 FlowHostLoop step，把 DAC launch 直接挂到 main body 上。
    bool temporalScheduleApplied = false;

    // lowering 生成的 runtime staging buffer 与源码 tensor alias 的映射关系。
    std::vector<HostTensorAliasInfo> runtimeTensorAliases;
    HostRuntimeRequirements runtimeRequirements;
    std::vector<FlowGeneratedName> generatedNames;
    FlowRouteRuntimeNames runtimeNames;
};

} // namespace ascendc
} // namespace dacppTranslator

#endif // TRANSLATOR_REWRITER_ASCENDC_ARTIFACTS_H
