#ifndef TRANSLATOR_REWRITER_ASCENDC_TYPE_MAPPING_H
#define TRANSLATOR_REWRITER_ASCENDC_TYPE_MAPPING_H

#include <string>

namespace dacppTranslator {
namespace ascendc {

std::string MapToStdType(const std::string& basicType);
std::string MapToKernelType(const std::string& basicType);
std::string MapToTilingDtype(const std::string& kernelType);
std::string MapToCubeKernelType(const std::string& basicType, bool isOutput);
std::string MapToCubeDeviceStagingType(const std::string& kernelType);
std::string MapToCubeUserHostType(const std::string& userType);

} // namespace ascendc
} // namespace dacppTranslator

#endif // TRANSLATOR_REWRITER_ASCENDC_TYPE_MAPPING_H
