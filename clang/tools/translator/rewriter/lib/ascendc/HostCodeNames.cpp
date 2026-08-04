#include "ascendc/HostCodeNames.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"

#include <algorithm>
#include <cctype>
#include <unordered_set>

namespace dacppTranslator {
namespace ascendc {
namespace {

bool IsIdentifier(const std::string& text) {
    if (text.empty()) return false;
    const unsigned char first = static_cast<unsigned char>(text.front());
    if (!(std::isalpha(first) || text.front() == '_')) return false;
    for (char c : text) {
        const unsigned char value = static_cast<unsigned char>(c);
        if (!(std::isalnum(value) || c == '_')) return false;
    }
    return true;
}

bool IsReservedIdentifier(const std::string& text) {
    if (text.find("__") != std::string::npos) return true;
    return text.size() > 1 && text[0] == '_' &&
           (text[1] == '_' ||
            std::isupper(static_cast<unsigned char>(text[1])));
}

std::string UpperFirst(std::string text) {
    if (!text.empty()) {
        text[0] = static_cast<char>(
            std::toupper(static_cast<unsigned char>(text[0])));
    }
    return text;
}

std::string LowerFirst(std::string text) {
    if (!text.empty()) {
        text[0] = static_cast<char>(
            std::tolower(static_cast<unsigned char>(text[0])));
    }
    return text;
}

std::string SanitizeStem(const std::string& raw) {
    std::string out;
    bool upperNext = false;
    for (char c : raw) {
        const unsigned char value = static_cast<unsigned char>(c);
        if (!std::isalnum(value)) {
            upperNext = !out.empty();
            continue;
        }
        char next = c;
        if (upperNext) {
            next = static_cast<char>(std::toupper(value));
            upperNext = false;
        }
        out.push_back(next);
    }
    if (out.empty()) return out;
    if (std::isdigit(static_cast<unsigned char>(out.front()))) {
        out.insert(out.begin(), 'v');
    }
    return LowerFirst(out);
}

HostNameSource SourceForRole(HostNameRole role,
                             const std::string& semanticStem) {
    if (role == HostNameRole::UserPreserved) {
        return HostNameSource::UserPreserved;
    }
    if (role == HostNameRole::PublicType) {
        return HostNameSource::PublicType;
    }
    if (!semanticStem.empty() &&
        (role == HostNameRole::Buffer || role == HostNameRole::Count ||
         role == HostNameRole::Bytes || role == HostNameRole::Stride ||
         role == HostNameRole::HostPtr || role == HostNameRole::DevicePtr)) {
        return HostNameSource::ParameterDerived;
    }
    return HostNameSource::FrameworkLocal;
}

std::vector<std::string> Candidates(HostNameRole role,
                                    const std::string& semanticStem) {
    const std::string stem = SanitizeStem(semanticStem);
    switch (role) {
        case HostNameRole::Session: return {"app", "session"};
        case HostNameRole::Route: return {"route", "rankRoute"};
        case HostNameRole::Device: return {"dev", "device"};
        case HostNameRole::Stream: return {"stream", "runtimeStream"};
        case HostNameRole::RankCount: return {"rankCount", "deviceCount"};
        case HostNameRole::RankList: return {"ranks", "rankArgs"};
        case HostNameRole::RankArg: return {"r", "rankArg"};
        case HostNameRole::Tile: return {"tile", "rankTile"};
        case HostNameRole::Length:
            return stem.empty() ? std::vector<std::string>{"len", "workLen"}
                                : std::vector<std::string>{stem + "Len", "workLen"};
        case HostNameRole::Count:
            return stem.empty() ? std::vector<std::string>{"count", "itemCount"}
                                : std::vector<std::string>{stem + "Count", "count"};
        case HostNameRole::Bytes:
            return stem.empty() ? std::vector<std::string>{"bytes", "byteCount"}
                                : std::vector<std::string>{stem + "Bytes", "bytes"};
        case HostNameRole::Begin: return {"begin", "srcBegin"};
        case HostNameRole::End: return {"end", "srcEnd"};
        case HostNameRole::Stride:
            return stem.empty() ? std::vector<std::string>{"stride", "rankStride"}
                                : std::vector<std::string>{stem + "Stride", "stride"};
        case HostNameRole::HostPtr:
            return stem.empty() ? std::vector<std::string>{"host", "hostPtr"}
                                : std::vector<std::string>{stem + "Host", "host"};
        case HostNameRole::DevicePtr:
            return stem.empty() ? std::vector<std::string>{"dev", "devicePtr"}
                                : std::vector<std::string>{stem + "Dev", "dev"};
        case HostNameRole::Buffer:
            return stem.empty() ? std::vector<std::string>{"buf", "buffer"}
                                : std::vector<std::string>{stem + "Buf", "buf"};
        case HostNameRole::Control: return {"control", "phaseControl"};
        case HostNameRole::LoopState: return {"state", "loopState"};
        case HostNameRole::HelperFunction:
            return stem.empty() ? std::vector<std::string>{"hostHelper"}
                                : std::vector<std::string>{stem};
        case HostNameRole::UserPreserved:
            return {semanticStem};
        case HostNameRole::PublicType:
            return {semanticStem.empty() ? "HostType" : UpperFirst(SanitizeStem(semanticStem))};
    }
    return {"value"};
}

class IdentifierCollector
    : public clang::RecursiveASTVisitor<IdentifierCollector> {
public:
    IdentifierCollector(const clang::SourceManager& sm,
                        clang::FileID ownerFile,
                        std::unordered_set<std::string>& names)
        : sourceManager(sm), file(ownerFile), output(names) {}

    bool VisitNamedDecl(clang::NamedDecl* decl) {
        if (!decl || decl->getName().empty()) return true;
        clang::SourceLocation loc =
            sourceManager.getExpansionLoc(decl->getLocation());
        if (loc.isInvalid() || sourceManager.getFileID(loc) != file) return true;
        output.insert(decl->getNameAsString());
        return true;
    }

private:
    const clang::SourceManager& sourceManager;
    clang::FileID file;
    std::unordered_set<std::string>& output;
};

} // namespace

HostNameAllocator::HostNameAllocator() {
    pushScope("route");
}

void HostNameAllocator::reserve(const std::string& identifier) {
    if (IsIdentifier(identifier)) reserved.insert(identifier);
}

void HostNameAllocator::pushScope(const std::string& stableScopeKey) {
    Scope scope;
    scope.key = stableScopeKey;
    scopes.push_back(std::move(scope));
}

void HostNameAllocator::popScope() {
    if (scopes.size() <= 1) {
        fail("attempted to pop the root host-name scope");
        return;
    }
    scopes.pop_back();
}

bool HostNameAllocator::conflicts(const std::string& identifier) const {
    if (reserved.count(identifier)) return true;
    for (const Scope& scope : scopes) {
        if (scope.names.count(identifier)) return true;
    }
    return false;
}

void HostNameAllocator::fail(const std::string& message) {
    if (errorText.empty()) errorText = message;
}

HostAllocatedName HostNameAllocator::get(
    HostNameRole role, const std::string& semanticStem) {
    HostAllocatedName result;
    result.source = SourceForRole(role, semanticStem);
    if (!ok()) return result;
    if (scopes.empty()) pushScope("route");

    std::vector<std::string> candidates = Candidates(role, semanticStem);
    std::string selected;
    for (const std::string& candidate : candidates) {
        if (!candidate.empty() && !conflicts(candidate)) {
            selected = candidate;
            break;
        }
    }
    if (selected.empty() && !candidates.empty() && !candidates.front().empty()) {
        for (unsigned suffix = 2; suffix < 10000; ++suffix) {
            const std::string candidate =
                candidates.front() + std::to_string(suffix);
            if (!conflicts(candidate)) {
                selected = candidate;
                break;
            }
        }
    }
    if (!IsIdentifier(selected) || IsReservedIdentifier(selected)) {
        fail("cannot allocate a non-reserved C++ identifier for host role");
        return result;
    }
    const std::size_t limit =
        result.source == HostNameSource::PublicType ? 24u : 20u;
    if (result.source != HostNameSource::UserPreserved &&
        selected.size() > limit) {
        fail("generated host identifier '" + selected +
             "' exceeds the generated-name length limit");
        return result;
    }
    scopes.back().names.insert(selected);
    result.text = selected;
    audit.push_back(result);
    return result;
}

std::vector<std::string> CollectHostVisibleIdentifiers(
    const clang::FunctionDecl& owner,
    const clang::TranslationUnitDecl& translationUnit) {
    std::unordered_set<std::string> names;
    const clang::ASTContext& context = owner.getASTContext();
    const clang::SourceManager& sm = context.getSourceManager();
    clang::SourceLocation ownerLoc = sm.getExpansionLoc(owner.getLocation());
    if (ownerLoc.isInvalid()) return {};
    const clang::FileID ownerFile = sm.getFileID(ownerLoc);

    IdentifierCollector ownerCollector(sm, ownerFile, names);
    ownerCollector.TraverseDecl(
        const_cast<clang::FunctionDecl*>(&owner));

    for (clang::Decl* decl : translationUnit.decls()) {
        clang::SourceLocation loc = sm.getExpansionLoc(decl->getLocation());
        if (loc.isInvalid() || sm.getFileID(loc) != ownerFile) continue;
        if (llvm::isa<clang::FunctionDecl>(decl) ||
            llvm::isa<clang::RecordDecl>(decl) ||
            llvm::isa<clang::TypedefNameDecl>(decl) ||
            llvm::isa<clang::VarDecl>(decl) ||
            llvm::isa<clang::UsingDecl>(decl) ||
            llvm::isa<clang::NamespaceAliasDecl>(decl)) {
            ownerCollector.TraverseDecl(decl);
        }
    }

    std::vector<std::string> result(names.begin(), names.end());
    std::sort(result.begin(), result.end());
    return result;
}

} // namespace ascendc
} // namespace dacppTranslator
