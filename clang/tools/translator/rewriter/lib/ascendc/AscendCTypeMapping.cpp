#include "ascendc/AscendCTypeMapping.h"

#include "ascendc/AscendCPolicy.h"

namespace dacppTranslator {
namespace ascendc {

std::string MapToStdType(const std::string& basicType) {
    const auto& policy = GetTranslationPolicy();
    if (basicType == "float") {
        return "float";
    }
    if (policy.floatLikeTypes.count(basicType) > 0) {
        return policy.hostFloatStorageType;
    }

    auto it = policy.stdTypeAliases.find(basicType);
    if (it != policy.stdTypeAliases.end()) {
        return it->second;
    }

    return basicType;
}

std::string MapToKernelType(const std::string& basicType) {
    const auto& policy = GetTranslationPolicy();
    if (basicType == "float") {
        return "float";
    }
    if (policy.floatLikeTypes.count(basicType) > 0) {
        return policy.kernelFloatType;
    }

    auto it = policy.kernelTypeAliases.find(basicType);
    if (it != policy.kernelTypeAliases.end()) {
        return it->second;
    }

    return basicType;
}

std::string MapToTilingDtype(const std::string& kernelType) {
    if (kernelType == "half" || kernelType == "float16") return "DataType::DT_FLOAT16";
    if (kernelType == "float") return "DataType::DT_FLOAT";
    if (kernelType == "int8_t") return "DataType::DT_INT8";
    if (kernelType == "int32_t") return "DataType::DT_INT32";
    return "DataType::DT_FLOAT16";
}

std::string MapToCubeKernelType(const std::string& basicType, bool isOutput) {
    const auto& policy = GetTranslationPolicy();
    if (isOutput) {
        return policy.cubeOutputKernelType;
    }
    (void)basicType;
    return policy.cubeInputKernelType;
}

std::string MapToCubeDeviceStagingType(const std::string& kernelType) {
    if (kernelType == "half" || kernelType == "float16") return "uint16_t";
    if (kernelType == "float") return "float";
    if (kernelType == "int8_t") return "int8_t";
    if (kernelType == "int32_t") return "int32_t";
    return "uint16_t";
}

std::string MapToCubeUserHostType(const std::string& userType) {
    return MapToStdType(userType);
}

} // namespace ascendc
} // namespace dacppTranslator
