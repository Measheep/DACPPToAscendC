#ifndef TRANSLATOR_REWRITER_ASCENDC_POLICY_H
#define TRANSLATOR_REWRITER_ASCENDC_POLICY_H

#include <string>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

#include "ascendc_mlir/CalcMlirPipeline.h"
#include "ascendc/HostRuntimeFeatures.h"

namespace clang {
class Rewriter;
}

namespace dacppTranslator {
class DacppFile;

namespace ascendc {

enum class BufferDepthMode {
    Auto,
    Single,
    Double
};

struct TranslationPolicy {
    int defaultCoreNum = 8;
    int npuCopyAlignBytes = 32;
    int vectorComputeAlignBytes = 256;
    bool emitResultHexLog = true;
    std::string hostFloatStorageType = "uint16_t";
    std::string kernelFloatType = "half";
    std::string cubeInputKernelType = "half";
    std::string cubeOutputKernelType = "float";
    bool enableMulticard = false;
    int multicardDefaultRankDim = 8;
    int requestedRankDim = 1;
    bool rankDimExplicit = false;
    std::string rankDimSource = "single-device";
    std::string rankDimEnvName = "DACPP_ASCENDC_RANK_DIM";
    size_t vectorUbBudgetBytes = 128 * 1024;
    size_t vectorUbReserveBytes = 16 * 1024;
    BufferDepthMode bufferDepthMode = BufferDepthMode::Auto;
    uint32_t maxTileElementsOverride = 0;
    bool debugOutput = false;
    bool emitTilingDiagnostics = false;
    std::string vectorUbBudgetSource = "default";
    std::string vectorUbReserveSource = "default";
    std::string bufferDepthSource = "default";
    std::string maxTileElementsSource = "default";
    std::string coreNumSource = "default";
    HostRuntimePackaging hostRuntimePackaging =
        HostRuntimePackaging::Compact;
    HostRuntimeLogLevel hostRuntimeLogLevel =
        HostRuntimeLogLevel::Error;
    bool valid = true;
    std::string firstConfigError;

    std::unordered_map<std::string, std::string> stdTypeAliases;
    std::unordered_map<std::string, std::string> kernelTypeAliases;
    std::unordered_set<std::string> floatLikeTypes;

    TranslationPolicy();
};

const TranslationPolicy& GetTranslationPolicy();

struct AscendCRewriteContext {
    DacppFile* dacppFile = nullptr;
    clang::Rewriter* rewriter = nullptr;
    std::string sourceText;
    const TranslationPolicy& policy;
    dacppTranslator::ascendc_mlir::CalcMlirPipelineOptions mlirOptions;

    AscendCRewriteContext(DacppFile* file, clang::Rewriter* rw);
};

} // namespace ascendc
} // namespace dacppTranslator

#endif // TRANSLATOR_REWRITER_ASCENDC_POLICY_H
