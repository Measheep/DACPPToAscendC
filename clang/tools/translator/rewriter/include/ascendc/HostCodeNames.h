#ifndef TRANSLATOR_REWRITER_ASCENDC_HOST_CODE_NAMES_H
#define TRANSLATOR_REWRITER_ASCENDC_HOST_CODE_NAMES_H

#include <string>
#include <unordered_set>
#include <vector>

namespace clang {
class FunctionDecl;
class TranslationUnitDecl;
}

namespace dacppTranslator {
namespace ascendc {

enum class HostNameRole {
    Session,
    Route,
    Device,
    Stream,
    RankCount,
    RankList,
    RankArg,
    Tile,
    Length,
    Count,
    Bytes,
    Begin,
    End,
    Stride,
    HostPtr,
    DevicePtr,
    Buffer,
    Control,
    LoopState,
    HelperFunction,
    UserPreserved,
    PublicType
};

enum class HostNameSource {
    FrameworkLocal,
    ParameterDerived,
    UserPreserved,
    PublicType
};

struct HostAllocatedName {
    std::string text;
    HostNameSource source = HostNameSource::FrameworkLocal;
};

class HostNameAllocator {
public:
    HostNameAllocator();

    void reserve(const std::string& identifier);
    void pushScope(const std::string& stableScopeKey);
    void popScope();
    HostAllocatedName get(HostNameRole role,
                          const std::string& semanticStem = "");

    bool ok() const { return errorText.empty(); }
    const std::string& error() const { return errorText; }
    const std::vector<HostAllocatedName>& allocatedNames() const {
        return audit;
    }

private:
    struct Scope {
        std::string key;
        std::unordered_set<std::string> names;
    };

    bool conflicts(const std::string& identifier) const;
    void fail(const std::string& message);

    std::unordered_set<std::string> reserved;
    std::vector<Scope> scopes;
    std::vector<HostAllocatedName> audit;
    std::string errorText;
};

std::vector<std::string> CollectHostVisibleIdentifiers(
    const clang::FunctionDecl& owner,
    const clang::TranslationUnitDecl& translationUnit);

} // namespace ascendc
} // namespace dacppTranslator

#endif // TRANSLATOR_REWRITER_ASCENDC_HOST_CODE_NAMES_H
