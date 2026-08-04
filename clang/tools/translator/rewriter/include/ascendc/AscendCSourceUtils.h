#ifndef TRANSLATOR_REWRITER_ASCENDC_SOURCE_UTILS_H
#define TRANSLATOR_REWRITER_ASCENDC_SOURCE_UTILS_H

#include <string>

namespace dacppTranslator {
namespace ascendc {

std::string TrimCopy(const std::string& text);
std::string ReplaceAllRegex(const std::string& input,
                            const std::string& pattern,
                            const std::string& replacement);
std::string EscapeRegexText(const std::string& text);
bool IsSimpleIdentifierText(const std::string& text);
bool IsIdentCharText(char c);
std::string StripTrailingSemicolon(std::string text);
bool TryParseDoubleValue(const std::string& text, double& outValue);
bool TryResolveSimpleSymbolValue(const std::string& sourceText,
                                 const std::string& symbol,
                                 double& outValue);

} // namespace ascendc
} // namespace dacppTranslator

#endif // TRANSLATOR_REWRITER_ASCENDC_SOURCE_UTILS_H
