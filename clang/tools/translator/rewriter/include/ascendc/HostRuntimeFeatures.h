#ifndef TRANSLATOR_REWRITER_ASCENDC_HOST_RUNTIME_FEATURES_H
#define TRANSLATOR_REWRITER_ASCENDC_HOST_RUNTIME_FEATURES_H

#include <cstdint>

namespace dacppTranslator {
namespace ascendc {

enum class HostRuntimeFeature : uint32_t {
    CheckedMath = 1u << 0,
    VectorTiling = 1u << 1,
    SingleCard = 1u << 2,
    MulticardRoute = 1u << 3,
    PersistentLoop = 1u << 4,
    Fp16Bridge = 1u << 5,
    TensorPrint = 1u << 6,
    CubeTiling = 1u << 7
};

enum class HostRuntimePackaging {
    Compact,
    Embedded
};

enum class HostRuntimeLogLevel {
    None,
    Error,
    Debug
};

struct HostRuntimeRequirements {
    uint32_t features = 0;
    HostRuntimePackaging packaging = HostRuntimePackaging::Compact;
    HostRuntimeLogLevel logLevel = HostRuntimeLogLevel::Error;

    void require(HostRuntimeFeature feature) {
        features |= static_cast<uint32_t>(feature);
    }

    bool has(HostRuntimeFeature feature) const {
        return (features & static_cast<uint32_t>(feature)) != 0;
    }

    void merge(const HostRuntimeRequirements& rhs) {
        features |= rhs.features;
        if (rhs.packaging == HostRuntimePackaging::Embedded) {
            packaging = HostRuntimePackaging::Embedded;
        }
        if (static_cast<uint32_t>(rhs.logLevel) >
            static_cast<uint32_t>(logLevel)) {
            logLevel = rhs.logLevel;
        }
    }
};

} // namespace ascendc
} // namespace dacppTranslator

#endif // TRANSLATOR_REWRITER_ASCENDC_HOST_RUNTIME_FEATURES_H
