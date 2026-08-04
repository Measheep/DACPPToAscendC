#ifndef TRANSLATOR_REWRITER_ASCENDC_HOST_RUNTIME_EMITTER_H
#define TRANSLATOR_REWRITER_ASCENDC_HOST_RUNTIME_EMITTER_H

#include <string>
#include <vector>

#include "ascendc/HostRuntimeFeatures.h"

namespace dacppTranslator {
namespace ascendc {

struct AscendCExpressionArtifact;
struct AscendCFlowPlan;
struct TranslationPolicy;

HostRuntimeRequirements CollectHostRuntimeRequirements(
    const std::vector<AscendCExpressionArtifact>& artifacts,
    const AscendCFlowPlan& plan,
    const TranslationPolicy& policy);

std::string EmitHostRuntimeBody(const HostRuntimeRequirements& req);
std::string EmitHostRuntimeHeader(const HostRuntimeRequirements& req);
std::string EmitHostRuntimeInsertion(const HostRuntimeRequirements& req);

} // namespace ascendc
} // namespace dacppTranslator

#endif // TRANSLATOR_REWRITER_ASCENDC_HOST_RUNTIME_EMITTER_H
