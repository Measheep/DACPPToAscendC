#ifndef TRANSLATOR_REWRITER_ASCENDC_RUNTIME_BINDING_H
#define TRANSLATOR_REWRITER_ASCENDC_RUNTIME_BINDING_H

#include <string>
#include <vector>

#include "clang/Basic/SourceLocation.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
class Expr;
class FunctionDecl;
class LangOptions;
class SourceManager;
class Stmt;
}

namespace dacppTranslator {
class Param;

namespace ascendc {

enum class RuntimeBindingNodeKind {
    StdVector,
    Scalar,
    TensorWrapper,
    TensorView,
    Unknown
};

struct RuntimeBindingResolveResult {
    bool found = false;
    bool isStdVector = false;
    bool sourceElemIsComplex = false;
    bool sourceElemIsPair = false;
    bool hasScalarExpr = false;
    bool isBroadcastScalar = false;
    RuntimeBindingNodeKind rootKind = RuntimeBindingNodeKind::Unknown;
    std::string sourceVar;
    std::string sourceElemType;
    std::string rootSourceVar;
    std::string rootElemType;
    std::string scalarExpr;
    std::string broadcastExpr;
    std::string bindingOwnerFunction;
    std::vector<std::string> aliasChain;
    const clang::Stmt* resolvedDecl = nullptr;
};

struct RuntimeParamBindingInfo {
    bool hasSourceVar = false;
    bool sourceIsStdVector = false;
    bool sourceElemIsComplex = false;
    bool sourceElemIsPair = false;
    bool hasScalarSourceExpr = false;
    bool isBroadcastScalar = false;
    bool hasShapeLength = false;
    bool bindingResolved = false;
    RuntimeBindingNodeKind bindingKind = RuntimeBindingNodeKind::Unknown;
    std::string sourceVar;
    std::string rootSourceVar;
    std::string sourceElemType;
    std::string scalarSourceExpr;
    std::string broadcastExpr;
    std::string bindingOwnerFunction;
    std::string shapeLength;
};

struct RuntimeBindingAstContext {
    const clang::SourceManager* sourceManager = nullptr;
    const clang::LangOptions* langOptions = nullptr;
    const clang::FunctionDecl* currentOwnerFunction = nullptr;
    clang::SourceLocation currentDacExprLoc;
};

struct RecordFieldLayoutEntry {
    std::string name;
    std::string typeName;
    clang::QualType qualType;
};

struct RecordFieldLayout {
    std::string recordTypeName;
    std::vector<RecordFieldLayoutEntry> fields;
};

bool LooksLikeStdVectorTypeName(const std::string& typeName);
bool LooksLikeTensorWrapperTypeName(const std::string& typeName);
bool LooksLikeTensorLikeTypeName(const std::string& typeName);
bool IsRootRuntimeBindingKind(RuntimeBindingNodeKind kind);
bool TryExtractIdentifierFromExpr(const clang::Expr* expr,
                                  std::string& outName);
bool TryExtractAliasedSourceFromExpr(const clang::Expr* expr,
                                     std::string& outBaseVar,
                                     bool& isView);
std::string ExtractVectorElemTypeFromTypeName(const std::string& typeName);
RuntimeBindingResolveResult ResolveRuntimeBindingFromScope(
    const RuntimeBindingAstContext& context,
    const clang::Stmt* scopeRoot,
    llvm::StringRef varName);

std::string BuildHostTensorAliasKey(const std::string& ownerFunctionName,
                                    const std::string& aliasName);

bool TryExtractRecordFieldLayout(clang::QualType qt,
                                 RecordFieldLayout& outLayout);
std::string BuildRecordFieldPhysicalName(const std::string& logicalName,
                                         const std::string& fieldName);
bool ParamUsesRecordElementType(const Param* param,
                                RecordFieldLayout* outLayout = nullptr);

} // namespace ascendc
} // namespace dacppTranslator

#endif // TRANSLATOR_REWRITER_ASCENDC_RUNTIME_BINDING_H
