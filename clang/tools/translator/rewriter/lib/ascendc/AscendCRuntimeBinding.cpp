#include "ascendc/AscendCRuntimeBinding.h"

#include <algorithm>
#include <cctype>
#include <utility>
#include <unordered_map>
#include <unordered_set>

#include "ascendc/AscendCSourceUtils.h"
#include "Param.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"

namespace dacppTranslator {
namespace ascendc {

bool LooksLikeStdVectorTypeName(const std::string& typeName) {
    return typeName.find("std::vector<") != std::string::npos ||
           typeName.find("vector<") != std::string::npos;
}

bool LooksLikeTensorWrapperTypeName(const std::string& typeName) {
    return typeName.find("dacpp::Vector") != std::string::npos ||
           typeName.find("dacpp::Tensor") != std::string::npos ||
           typeName.find("dacpp::Matrix") != std::string::npos ||
           typeName.find("Vector<") != std::string::npos ||
           typeName.find("Tensor<") != std::string::npos ||
           typeName.find("Matrix<") != std::string::npos;
}

bool LooksLikeTensorLikeTypeName(const std::string& typeName) {
    return LooksLikeTensorWrapperTypeName(typeName) ||
           typeName.find("dacpp::TensorProxy") != std::string::npos ||
           typeName.find("TensorProxy<") != std::string::npos;
}

bool IsRootRuntimeBindingKind(RuntimeBindingNodeKind kind) {
    return kind == RuntimeBindingNodeKind::StdVector ||
           kind == RuntimeBindingNodeKind::Scalar;
}

bool TryExtractIdentifierFromExpr(const clang::Expr* expr, std::string& outName) {
    outName.clear();
    if (!expr) {
        return false;
    }
    expr = expr->IgnoreParenImpCasts();
    if (const auto* declRef = clang::dyn_cast<clang::DeclRefExpr>(expr)) {
        if (declRef->getDecl()) {
            outName = declRef->getDecl()->getNameAsString();
            return !outName.empty();
        }
    }

    if (const auto* member = clang::dyn_cast<clang::MemberExpr>(expr)) {
        return TryExtractIdentifierFromExpr(member->getBase(), outName);
    }

    if (const auto* subscript = clang::dyn_cast<clang::ArraySubscriptExpr>(expr)) {
        return TryExtractIdentifierFromExpr(subscript->getBase(), outName);
    }

    if (const auto* opCall = clang::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
        if (opCall->getOperator() == clang::OverloadedOperatorKind::OO_Subscript &&
            opCall->getNumArgs() > 0) {
            return TryExtractIdentifierFromExpr(opCall->getArg(0), outName);
        }
    }

    return false;
}

bool TryExtractAliasedSourceFromExpr(const clang::Expr* expr,
                                     std::string& outBaseVar,
                                     bool& isView) {
    outBaseVar.clear();
    isView = false;
    if (!expr) {
        return false;
    }

    expr = expr->IgnoreParenImpCasts();
    if (const auto* cleanups = clang::dyn_cast<clang::ExprWithCleanups>(expr)) {
        return TryExtractAliasedSourceFromExpr(cleanups->getSubExpr(), outBaseVar, isView);
    }
    if (const auto* matTmp = clang::dyn_cast<clang::MaterializeTemporaryExpr>(expr)) {
        return TryExtractAliasedSourceFromExpr(matTmp->getSubExpr(), outBaseVar, isView);
    }
    if (const auto* bindTmp = clang::dyn_cast<clang::CXXBindTemporaryExpr>(expr)) {
        return TryExtractAliasedSourceFromExpr(bindTmp->getSubExpr(), outBaseVar, isView);
    }
    if (const auto* ctor = clang::dyn_cast<clang::CXXConstructExpr>(expr)) {
        for (unsigned i = 0; i < ctor->getNumArgs(); ++i) {
            bool argIsView = false;
            std::string argBase;
            if (TryExtractAliasedSourceFromExpr(ctor->getArg(i), argBase, argIsView)) {
                outBaseVar = argBase;
                isView = argIsView;
                return true;
            }
        }
        return false;
    }

    if (const auto* opCall = clang::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
        if (opCall->getOperator() == clang::OverloadedOperatorKind::OO_Subscript &&
            opCall->getNumArgs() > 0) {
            if (TryExtractIdentifierFromExpr(opCall->getArg(0), outBaseVar)) {
                isView = true;
                return true;
            }
        }
    }

    if (const auto* subscript = clang::dyn_cast<clang::ArraySubscriptExpr>(expr)) {
        if (TryExtractIdentifierFromExpr(subscript->getBase(), outBaseVar)) {
            isView = true;
            return true;
        }
    }

    if (TryExtractIdentifierFromExpr(expr, outBaseVar)) {
        isView = false;
        return true;
    }

    return false;
}

std::string ExtractVectorElemTypeFromTypeName(const std::string& typeName) {
    size_t vectorPos = typeName.find("vector");
    if (vectorPos == std::string::npos) {
        return "";
    }
    size_t openPos = typeName.find('<', vectorPos);
    if (openPos == std::string::npos) {
        return "";
    }

    int depth = 0;
    size_t closePos = std::string::npos;
    for (size_t i = openPos + 1; i < typeName.size(); ++i) {
        char c = typeName[i];
        if (c == '<') {
            ++depth;
        } else if (c == '>') {
            if (depth == 0) {
                closePos = i;
                break;
            }
            --depth;
        }
    }
    if (closePos == std::string::npos || closePos <= openPos + 1) {
        return "";
    }

    std::string inside = typeName.substr(openPos + 1, closePos - openPos - 1);
    depth = 0;
    for (size_t i = 0; i < inside.size(); ++i) {
        char c = inside[i];
        if (c == '<') {
            ++depth;
        } else if (c == '>') {
            --depth;
        } else if (c == ',' && depth == 0) {
            inside = inside.substr(0, i);
            break;
        }
    }
    return TrimCopy(inside);
}

static std::string GetRuntimeExprSourceText(const clang::Expr* expr,
                                            const RuntimeBindingAstContext& context) {
    if (!expr || !context.sourceManager || !context.langOptions) {
        return "";
    }
    clang::CharSourceRange range = clang::CharSourceRange::getTokenRange(expr->getSourceRange());
    return clang::Lexer::getSourceText(range, *context.sourceManager, *context.langOptions).str();
}

struct RuntimeBindingVarState {
    bool seenDecl = false;
    bool isStdVector = false;
    bool isTensorWrapper = false;
    bool hasScalarInit = false;
    bool hasBroadcastExpr = false;
    RuntimeBindingNodeKind kind = RuntimeBindingNodeKind::Unknown;
    std::string aliasTargetVar;
    const clang::Stmt* declStmt = nullptr;
    const clang::Expr* initExpr = nullptr;
    std::vector<const clang::Stmt*> mutatingStmts;
    std::string canonicalTypeName;
    std::string vectorElemType;
    std::string scalarInitExpr;
    std::string broadcastExpr;
};

class RuntimeBindingScopeCollector
    : public clang::RecursiveASTVisitor<RuntimeBindingScopeCollector> {
public:
    RuntimeBindingScopeCollector(const RuntimeBindingAstContext& runtimeContext,
                                 const clang::SourceManager& sm,
                                 const clang::SourceLocation cutoff)
        : context(runtimeContext), sourceManager(sm), cutoffLoc(sm.getExpansionLoc(cutoff)) {}

    std::unordered_map<std::string, RuntimeBindingVarState> varStates;

    bool VisitDeclStmt(clang::DeclStmt* declStmt) {
        if (!declStmt || !IsBeforeCutoff(declStmt->getBeginLoc())) {
            return true;
        }
        for (const clang::Decl* decl : declStmt->decls()) {
            const auto* varDecl = clang::dyn_cast<clang::VarDecl>(decl);
            if (!varDecl || !varDecl->getIdentifier()) {
                continue;
            }
            RuntimeBindingVarState& state = varStates[varDecl->getNameAsString()];
            if (!state.declStmt) {
                state.declStmt = declStmt;
            }
        }
        return true;
    }

    bool VisitVarDecl(clang::VarDecl* varDecl) {
        if (!varDecl || !varDecl->getIdentifier() || !IsBeforeCutoff(varDecl->getBeginLoc())) {
            return true;
        }

        std::string name = varDecl->getNameAsString();
        if (name.empty()) {
            return true;
        }

        RuntimeBindingVarState& state = varStates[name];
        state.seenDecl = true;
        state.initExpr = varDecl->hasInit() ? varDecl->getInit() : nullptr;

        std::string typeName = varDecl->getType().getAsString();
        std::string canonicalTypeName = varDecl->getType().getCanonicalType().getAsString();
        state.canonicalTypeName = canonicalTypeName;
        if (LooksLikeStdVectorTypeName(typeName) ||
            LooksLikeStdVectorTypeName(canonicalTypeName)) {
            state.isStdVector = true;
            state.kind = RuntimeBindingNodeKind::StdVector;
            std::string elemType = ExtractVectorElemTypeFromTypeName(typeName);
            if (elemType.empty()) {
                elemType = ExtractVectorElemTypeFromTypeName(canonicalTypeName);
            }
            if (!elemType.empty()) {
                state.vectorElemType = elemType;
                std::string lower = elemType;
                std::transform(lower.begin(), lower.end(), lower.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                (void)lower;
            }
        }

        if (LooksLikeTensorWrapperTypeName(typeName) ||
            LooksLikeTensorWrapperTypeName(canonicalTypeName)) {
            state.isTensorWrapper = true;
            if (state.kind == RuntimeBindingNodeKind::Unknown) {
                state.kind = RuntimeBindingNodeKind::TensorWrapper;
            }
        }

        if ((varDecl->getType()->isArithmeticType() || varDecl->getType()->isEnumeralType()) &&
            state.kind == RuntimeBindingNodeKind::Unknown) {
            state.kind = RuntimeBindingNodeKind::Scalar;
        }

        if (varDecl->hasInit()) {
            const clang::Expr* initExpr = varDecl->getInit();
            const clang::Expr* peeled = initExpr ? initExpr->IgnoreParenImpCasts() : nullptr;
            if (!state.isStdVector) {
                std::string aliasedSource;
                bool isViewAlias = false;
                if (TryExtractAliasedSourceFromExpr(initExpr, aliasedSource, isViewAlias) &&
                    !aliasedSource.empty() && aliasedSource != name) {
                    state.aliasTargetVar = aliasedSource;
                    if (state.kind == RuntimeBindingNodeKind::Unknown) {
                        state.kind = isViewAlias
                                     ? RuntimeBindingNodeKind::TensorView
                                     : RuntimeBindingNodeKind::TensorWrapper;
                    } else if (state.kind == RuntimeBindingNodeKind::TensorWrapper && isViewAlias) {
                        state.kind = RuntimeBindingNodeKind::TensorView;
                    }
                }
            }

            if (state.isStdVector) {
                if (const auto* ctor = clang::dyn_cast<clang::CXXConstructExpr>(peeled)) {
                    if (ctor->getNumArgs() >= 2 && !state.hasBroadcastExpr) {
                        state.hasBroadcastExpr = true;
                        state.broadcastExpr = TrimCopy(GetRuntimeExprSourceText(ctor->getArg(1), context));
                    }
                }
                if (const auto* initList = clang::dyn_cast<clang::InitListExpr>(peeled)) {
                    if (initList->getNumInits() > 0 && !state.hasBroadcastExpr) {
                        state.hasBroadcastExpr = true;
                        state.broadcastExpr = TrimCopy(GetRuntimeExprSourceText(initList->getInit(0), context));
                    }
                }
            }

            if ((varDecl->getType()->isArithmeticType() || varDecl->getType()->isEnumeralType()) &&
                !state.hasScalarInit) {
                std::string initText = TrimCopy(GetRuntimeExprSourceText(initExpr, context));
                if (!initText.empty()) {
                    state.hasScalarInit = true;
                    state.scalarInitExpr = initText;
                    if (state.kind == RuntimeBindingNodeKind::Unknown) {
                        state.kind = RuntimeBindingNodeKind::Scalar;
                    }
                }
            }
        }

        return true;
    }

    bool VisitCXXMemberCallExpr(clang::CXXMemberCallExpr* callExpr) {
        if (!callExpr || !IsBeforeCutoff(callExpr->getExprLoc())) {
            return true;
        }
        const clang::CXXMethodDecl* methodDecl = callExpr->getMethodDecl();
        if (!methodDecl) {
            return true;
        }
        std::string methodName = methodDecl->getNameAsString();
        if (methodName != "push_back" && methodName != "emplace_back") {
            return true;
        }

        std::string baseName;
        if (!TryExtractIdentifierFromExpr(callExpr->getImplicitObjectArgument(), baseName)) {
            return true;
        }
        if (callExpr->getNumArgs() == 0) {
            return true;
        }

        RuntimeBindingVarState& state = varStates[baseName];
        state.mutatingStmts.push_back(callExpr);
        if (!state.hasBroadcastExpr) {
            state.hasBroadcastExpr = true;
            state.broadcastExpr = TrimCopy(GetRuntimeExprSourceText(callExpr->getArg(0), context));
        }
        return true;
    }

    bool VisitBinaryOperator(clang::BinaryOperator* opExpr) {
        if (!opExpr || !IsBeforeCutoff(opExpr->getExprLoc())) {
            return true;
        }

        if (!opExpr->isAssignmentOp()) {
            return true;
        }

        const clang::Expr* lhs = opExpr->getLHS()->IgnoreParenImpCasts();
        std::string lhsName;
        TryExtractIdentifierFromExpr(lhs, lhsName);
        if (!lhsName.empty()) {
            std::string rhsAliasSource;
            bool rhsIsViewAlias = false;
            if (TryExtractAliasedSourceFromExpr(opExpr->getRHS(), rhsAliasSource, rhsIsViewAlias) &&
                !rhsAliasSource.empty() && rhsAliasSource != lhsName) {
                RuntimeBindingVarState& lhsState = varStates[lhsName];
                lhsState.aliasTargetVar = rhsAliasSource;
                lhsState.mutatingStmts.push_back(opExpr);

                if (lhsState.kind == RuntimeBindingNodeKind::Unknown) {
                    if (!lhsState.canonicalTypeName.empty() &&
                        LooksLikeTensorWrapperTypeName(lhsState.canonicalTypeName)) {
                        lhsState.kind = rhsIsViewAlias
                                        ? RuntimeBindingNodeKind::TensorView
                                        : RuntimeBindingNodeKind::TensorWrapper;
                    }
                } else if (lhsState.kind == RuntimeBindingNodeKind::TensorWrapper && rhsIsViewAlias) {
                    lhsState.kind = RuntimeBindingNodeKind::TensorView;
                }
            }
        }

        std::string baseName;
        if (const auto* subscript = clang::dyn_cast<clang::ArraySubscriptExpr>(lhs)) {
            TryExtractIdentifierFromExpr(subscript->getBase(), baseName);
        } else if (const auto* opCall = clang::dyn_cast<clang::CXXOperatorCallExpr>(lhs)) {
            if (opCall->getOperator() == clang::OverloadedOperatorKind::OO_Subscript &&
                opCall->getNumArgs() > 0) {
                TryExtractIdentifierFromExpr(opCall->getArg(0), baseName);
            }
        }

        if (baseName.empty()) {
            return true;
        }

        RuntimeBindingVarState& state = varStates[baseName];
        state.mutatingStmts.push_back(opExpr);
        if (!state.hasBroadcastExpr) {
            state.hasBroadcastExpr = true;
            state.broadcastExpr = TrimCopy(GetRuntimeExprSourceText(opExpr->getRHS(), context));
        }

        return true;
    }

    bool VisitWhileStmt(clang::WhileStmt* whileStmt) {
        if (!whileStmt || !IsBeforeCutoff(whileStmt->getWhileLoc())) {
            return true;
        }
        const clang::Expr* condExpr = whileStmt->getCond();
        if (!condExpr) {
            return true;
        }
        condExpr = condExpr->IgnoreParenImpCasts();
        const auto* compareOp = clang::dyn_cast<clang::BinaryOperator>(condExpr);
        if (!compareOp || !compareOp->isComparisonOp()) {
            return true;
        }

        std::string lhsName;
        if (!TryExtractIdentifierFromExpr(compareOp->getLHS(), lhsName) || lhsName.empty()) {
            return true;
        }

        RuntimeBindingVarState& state = varStates[lhsName];
        state.mutatingStmts.push_back(whileStmt);
        if (!state.hasBroadcastExpr &&
            !state.hasScalarInit &&
            !state.isStdVector &&
            !state.isTensorWrapper) {
            state.hasBroadcastExpr = true;
            state.broadcastExpr = TrimCopy(GetRuntimeExprSourceText(compareOp->getRHS(), context));
        }

        return true;
    }

private:
    bool IsBeforeCutoff(clang::SourceLocation loc) const {
        if (cutoffLoc.isInvalid()) {
            return true;
        }
        clang::SourceLocation probe = sourceManager.getExpansionLoc(loc);
        if (probe.isInvalid()) {
            return false;
        }
        return sourceManager.isBeforeInTranslationUnit(probe, cutoffLoc);
    }

    const RuntimeBindingAstContext& context;
    const clang::SourceManager& sourceManager;
    clang::SourceLocation cutoffLoc;
};

RuntimeBindingResolveResult ResolveRuntimeBindingFromScope(
    const RuntimeBindingAstContext& context,
    const clang::Stmt* scopeRoot,
    llvm::StringRef varName) {
    RuntimeBindingResolveResult result;
    result.sourceVar = TrimCopy(varName.str());
    if (result.sourceVar.empty() || !scopeRoot || !context.sourceManager) {
        return result;
    }

    RuntimeBindingScopeCollector collector(
        context,
        *context.sourceManager,
        context.currentDacExprLoc);

    if (context.currentOwnerFunction) {
        for (const clang::ParmVarDecl* paramDecl : context.currentOwnerFunction->parameters()) {
            if (!paramDecl || !paramDecl->getIdentifier()) {
                continue;
            }

            RuntimeBindingVarState& state =
                collector.varStates[paramDecl->getNameAsString()];
            state.seenDecl = true;
            state.canonicalTypeName = paramDecl->getType().getCanonicalType().getAsString();

            std::string typeName = paramDecl->getType().getAsString();
            if (typeName.find("vector<") != std::string::npos ||
                typeName.find("std::vector<") != std::string::npos) {
                state.isStdVector = true;
                state.kind = RuntimeBindingNodeKind::StdVector;
                std::string elemType = ExtractVectorElemTypeFromTypeName(typeName);
                if (!elemType.empty()) {
                    state.vectorElemType = elemType;
                }
            }

            if ((paramDecl->getType()->isArithmeticType() || paramDecl->getType()->isEnumeralType()) &&
                !state.hasScalarInit) {
                state.hasScalarInit = true;
                state.scalarInitExpr = paramDecl->getNameAsString();
                if (state.kind == RuntimeBindingNodeKind::Unknown) {
                    state.kind = RuntimeBindingNodeKind::Scalar;
                }
            }

            if (state.kind == RuntimeBindingNodeKind::Unknown &&
                LooksLikeTensorWrapperTypeName(typeName)) {
                state.isTensorWrapper = true;
                state.kind = RuntimeBindingNodeKind::TensorWrapper;
            }
        }
    }

    collector.TraverseStmt(const_cast<clang::Stmt*>(scopeRoot));

    std::unordered_set<std::string> seen;
    std::string current = result.sourceVar;
    std::string propagatedBroadcastExpr;
    for (int depth = 0; depth < 16; ++depth) {
        if (current.empty() || seen.count(current) > 0) {
            break;
        }
        seen.insert(current);
        result.aliasChain.push_back(current);

        auto it = collector.varStates.find(current);
        if (it == collector.varStates.end()) {
            break;
        }

        const RuntimeBindingVarState& state = it->second;
        result.sourceVar = current;
        result.resolvedDecl = state.declStmt;
        if (state.hasBroadcastExpr && propagatedBroadcastExpr.empty()) {
            propagatedBroadcastExpr = state.broadcastExpr;
        }

        if (!state.aliasTargetVar.empty() && state.aliasTargetVar != current) {
            current = state.aliasTargetVar;
            continue;
        }

        result.rootSourceVar = current;
        result.rootKind = state.kind;
        result.rootElemType.clear();

        if (state.kind == RuntimeBindingNodeKind::StdVector) {
            result.found = true;
            result.isStdVector = true;
            result.rootElemType = state.vectorElemType.empty() ? "float" : state.vectorElemType;
            result.sourceElemType = result.rootElemType;

            std::string elemTypeLower = result.rootElemType;
            std::transform(elemTypeLower.begin(), elemTypeLower.end(), elemTypeLower.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            result.sourceElemIsComplex = elemTypeLower.find("complex") != std::string::npos;
            result.sourceElemIsPair = elemTypeLower.find("pair") != std::string::npos;

            if (!propagatedBroadcastExpr.empty()) {
                result.isBroadcastScalar = true;
                result.broadcastExpr = propagatedBroadcastExpr;
            }
            break;
        }

        if (state.kind == RuntimeBindingNodeKind::Scalar) {
            result.found = true;
            if (state.hasScalarInit) {
                result.hasScalarExpr = true;
                result.scalarExpr = state.scalarInitExpr;
            }
            if (!propagatedBroadcastExpr.empty()) {
                result.isBroadcastScalar = true;
                result.broadcastExpr = propagatedBroadcastExpr;
            } else if (result.hasScalarExpr) {
                result.isBroadcastScalar = true;
                result.broadcastExpr = result.scalarExpr;
            }
            break;
        }

        if (!propagatedBroadcastExpr.empty()) {
            result.isBroadcastScalar = true;
            result.broadcastExpr = propagatedBroadcastExpr;
        }

        break;
    }

    if (!result.rootSourceVar.empty()) {
        result.sourceVar = result.rootSourceVar;
        if (result.sourceElemType.empty()) {
            result.sourceElemType = result.rootElemType;
        }
    }

    if (context.currentOwnerFunction) {
        result.bindingOwnerFunction = context.currentOwnerFunction->getNameAsString();
    }

    return result;
}

std::string BuildHostTensorAliasKey(const std::string& ownerFunctionName,
                                    const std::string& aliasName) {
    return ownerFunctionName + "::" + aliasName;
}

static bool IsSupportedRecordFieldType(clang::QualType qt) {
    qt = qt.getCanonicalType();
    return qt->isArithmeticType() || qt->isEnumeralType();
}

bool TryExtractRecordFieldLayout(clang::QualType qt,
                                 RecordFieldLayout& outLayout) {
    outLayout = RecordFieldLayout{};
    qt = qt.getCanonicalType();

    const clang::RecordType* recordTy = qt->getAs<clang::RecordType>();
    if (!recordTy) {
        return false;
    }

    const clang::RecordDecl* recordDecl = recordTy->getDecl();
    if (!recordDecl) {
        return false;
    }
    recordDecl = recordDecl->getDefinition();
    if (!recordDecl || !recordDecl->isCompleteDefinition()) {
        return false;
    }

    outLayout.recordTypeName = recordDecl->getNameAsString();
    for (const clang::FieldDecl* fieldDecl : recordDecl->fields()) {
        if (!fieldDecl || fieldDecl->getName().empty()) {
            return false;
        }

        clang::QualType fieldType = fieldDecl->getType().getCanonicalType();
        if (!IsSupportedRecordFieldType(fieldType)) {
            return false;
        }

        RecordFieldLayoutEntry entry;
        entry.name = fieldDecl->getNameAsString();
        entry.typeName = fieldType.getAsString();
        entry.qualType = fieldType;
        outLayout.fields.push_back(std::move(entry));
    }

    return !outLayout.fields.empty();
}

std::string BuildRecordFieldPhysicalName(const std::string& logicalName,
                                         const std::string& fieldName) {
    std::string stem = logicalName;
    const std::string tensorMarker = "_tensor";
    std::size_t marker = stem.find(tensorMarker);
    if (marker != std::string::npos) {
        stem.erase(marker, tensorMarker.size());
    }
    std::string fieldStem = fieldName;
    if (!fieldStem.empty()) {
        fieldStem.front() = static_cast<char>(
            std::toupper(static_cast<unsigned char>(fieldStem.front())));
    }
    return stem + fieldStem;
}

bool ParamUsesRecordElementType(const Param* param,
                                RecordFieldLayout* outLayout) {
    if (!param) {
        return false;
    }
    RecordFieldLayout layout;
    if (!TryExtractRecordFieldLayout(param->BasicType, layout)) {
        return false;
    }
    if (outLayout) {
        *outLayout = layout;
    }
    return true;
}

} // namespace ascendc
} // namespace dacppTranslator
