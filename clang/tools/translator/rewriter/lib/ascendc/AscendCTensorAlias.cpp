#include "ascendc/AscendCTensorAlias.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <functional>
#include <unordered_map>

#include "ascendc/AscendCRuntimeBinding.h"
#include "ascendc/AscendCSourceUtils.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"

namespace dacppTranslator {
namespace ascendc {

namespace {

std::string GetExprSourceText(const clang::Expr* expr,
                              const TensorAliasResolverContext& context) {
    if (!expr || !context.sourceManager || !context.langOptions) {
        return "";
    }
    clang::CharSourceRange range =
        clang::CharSourceRange::getTokenRange(expr->getSourceRange());
    return clang::Lexer::getSourceText(
        range,
        *context.sourceManager,
        *context.langOptions).str();
}

bool TryLookupHostTensorAliasByName(
    llvm::StringRef ownerFunctionName,
    llvm::StringRef aliasName,
    const std::unordered_map<std::string, HostTensorAliasInfo>& aliasMap,
    HostTensorAliasInfo& outInfo) {
    outInfo = HostTensorAliasInfo();
    const std::string trimmedAlias = TrimCopy(aliasName.str());
    if (trimmedAlias.empty()) {
        return false;
    }

    auto it = aliasMap.find(BuildHostTensorAliasKey(ownerFunctionName.str(), trimmedAlias));
    if (it == aliasMap.end()) {
        return false;
    }

    outInfo = it->second;
    return outInfo.valid;
}

bool TryLookupHostTensorAliasAcrossOwners(
    llvm::StringRef aliasName,
    const std::unordered_map<std::string, HostTensorAliasInfo>& aliasMap,
    HostTensorAliasInfo& outInfo) {
    outInfo = HostTensorAliasInfo();
    const std::string trimmedAlias = TrimCopy(aliasName.str());
    if (trimmedAlias.empty()) {
        return false;
    }

    bool found = false;
    HostTensorAliasInfo candidate;
    for (const auto& entry : aliasMap) {
        const HostTensorAliasInfo& info = entry.second;
        if (!info.valid || TrimCopy(info.aliasName) != trimmedAlias) {
            continue;
        }
        if (found) {
            return false;
        }
        candidate = info;
        found = true;
    }

    if (!found) {
        return false;
    }
    outInfo = candidate;
    return outInfo.valid;
}

} // namespace

std::vector<std::string> SplitTopLevelByCommaPreserveNesting(const std::string& text) {
    std::vector<std::string> parts;
    std::string current;
    int parenDepth = 0;
    int bracketDepth = 0;
    int braceDepth = 0;
    int angleDepth = 0;
    bool inSingleQuote = false;
    bool inDoubleQuote = false;
    bool escaping = false;

    auto FlushCurrent = [&]() {
        std::string trimmed = TrimCopy(current);
        if (!trimmed.empty()) {
            parts.push_back(trimmed);
        }
        current.clear();
    };

    for (char c : text) {
        if (escaping) {
            current += c;
            escaping = false;
            continue;
        }
        if (c == '\\') {
            current += c;
            escaping = true;
            continue;
        }
        if (inSingleQuote) {
            current += c;
            if (c == '\'') {
                inSingleQuote = false;
            }
            continue;
        }
        if (inDoubleQuote) {
            current += c;
            if (c == '"') {
                inDoubleQuote = false;
            }
            continue;
        }
        if (c == '\'') {
            current += c;
            inSingleQuote = true;
            continue;
        }
        if (c == '"') {
            current += c;
            inDoubleQuote = true;
            continue;
        }

        switch (c) {
        case '(':
            ++parenDepth;
            break;
        case ')':
            --parenDepth;
            break;
        case '[':
            ++bracketDepth;
            break;
        case ']':
            --bracketDepth;
            break;
        case '{':
            ++braceDepth;
            break;
        case '}':
            --braceDepth;
            break;
        case '<':
            ++angleDepth;
            break;
        case '>':
            if (angleDepth > 0) {
                --angleDepth;
            }
            break;
        case ',':
            if (parenDepth == 0 && bracketDepth == 0 &&
                braceDepth == 0 && angleDepth == 0) {
                FlushCurrent();
                continue;
            }
            break;
        default:
            break;
        }
        current += c;
    }
    FlushCurrent();
    return parts;
}

int ExtractTensorRankFromTypeName(const std::string& typeName) {
    if (typeName.find("Vector<") != std::string::npos ||
        typeName.find("dacpp::Vector") != std::string::npos) {
        return 1;
    }
    if (typeName.find("Matrix<") != std::string::npos ||
        typeName.find("dacpp::Matrix") != std::string::npos) {
        return 2;
    }

    size_t tensorPos = typeName.find("Tensor<");
    if (tensorPos == std::string::npos) {
        tensorPos = typeName.find("TensorProxy<");
    }
    if (tensorPos == std::string::npos) {
        tensorPos = typeName.find("dacpp::Tensor<");
    }
    if (tensorPos == std::string::npos) {
        tensorPos = typeName.find("dacpp::TensorProxy<");
    }
    if (tensorPos == std::string::npos) {
        return 0;
    }

    size_t openPos = typeName.find('<', tensorPos);
    size_t closePos = typeName.rfind('>');
    if (openPos == std::string::npos || closePos == std::string::npos || closePos <= openPos) {
        return 0;
    }
    std::string inside = typeName.substr(openPos + 1, closePos - openPos - 1);
    std::vector<std::string> parts = SplitTopLevelByCommaPreserveNesting(inside);
    if (parts.size() < 2) {
        return 0;
    }
    try {
        return std::stoi(TrimCopy(parts.back()));
    } catch (...) {
        return 0;
    }
}

std::string ExtractTensorElementTypeFromTypeName(const std::string& typeName) {
    size_t vectorPos = typeName.find("Vector<");
    if (vectorPos == std::string::npos) {
        vectorPos = typeName.find("Matrix<");
    }
    if (vectorPos == std::string::npos) {
        vectorPos = typeName.find("dacpp::Vector<");
    }
    if (vectorPos == std::string::npos) {
        vectorPos = typeName.find("dacpp::Matrix<");
    }
    if (vectorPos != std::string::npos) {
        size_t openPos = typeName.find('<', vectorPos);
        size_t closePos = typeName.rfind('>');
        if (openPos != std::string::npos && closePos != std::string::npos && closePos > openPos) {
            return TrimCopy(typeName.substr(openPos + 1, closePos - openPos - 1));
        }
    }

    size_t tensorPos = typeName.find("Tensor<");
    if (tensorPos == std::string::npos) {
        tensorPos = typeName.find("TensorProxy<");
    }
    if (tensorPos == std::string::npos) {
        tensorPos = typeName.find("dacpp::Tensor<");
    }
    if (tensorPos == std::string::npos) {
        tensorPos = typeName.find("dacpp::TensorProxy<");
    }
    if (tensorPos == std::string::npos) {
        return "";
    }

    size_t openPos = typeName.find('<', tensorPos);
    size_t closePos = typeName.rfind('>');
    if (openPos == std::string::npos || closePos == std::string::npos || closePos <= openPos) {
        return "";
    }
    std::string inside = typeName.substr(openPos + 1, closePos - openPos - 1);
    std::vector<std::string> parts = SplitTopLevelByCommaPreserveNesting(inside);
    if (parts.empty()) {
        return "";
    }
    return TrimCopy(parts.front());
}

std::string MakeParenthesizedExpr(const std::string& expr) {
    const std::string trimmed = TrimCopy(expr);
    if (trimmed.empty()) {
        return "0";
    }
    return "(" + trimmed + ")";
}

std::string BuildSumExpr(const std::string& lhs, const std::string& rhs) {
    const std::string lhsTrimmed = TrimCopy(lhs);
    const std::string rhsTrimmed = TrimCopy(rhs);
    if (lhsTrimmed.empty() || lhsTrimmed == "0") {
        return rhsTrimmed.empty() ? std::string("0") : rhsTrimmed;
    }
    if (rhsTrimmed.empty() || rhsTrimmed == "0") {
        return lhsTrimmed;
    }
    return MakeParenthesizedExpr(lhsTrimmed) + " + " + MakeParenthesizedExpr(rhsTrimmed);
}

std::string BuildProductExpr(const std::string& lhs, const std::string& rhs) {
    const std::string lhsTrimmed = TrimCopy(lhs);
    const std::string rhsTrimmed = TrimCopy(rhs);
    if (lhsTrimmed.empty() || rhsTrimmed.empty()) {
        return "0";
    }
    if (lhsTrimmed == "0" || rhsTrimmed == "0") {
        return "0";
    }
    if (lhsTrimmed == "1") {
        return rhsTrimmed;
    }
    if (rhsTrimmed == "1") {
        return lhsTrimmed;
    }
    return MakeParenthesizedExpr(lhsTrimmed) + " * " + MakeParenthesizedExpr(rhsTrimmed);
}

std::vector<std::string> BuildRowMajorStrideExprs(const std::vector<std::string>& shapeExprs) {
    std::vector<std::string> strideExprs(shapeExprs.size(), "1");
    for (ptrdiff_t i = static_cast<ptrdiff_t>(shapeExprs.size()) - 2; i >= 0; --i) {
        strideExprs[static_cast<size_t>(i)] =
            BuildProductExpr(strideExprs[static_cast<size_t>(i + 1)],
                             shapeExprs[static_cast<size_t>(i + 1)]);
    }
    return strideExprs;
}

bool TryApplyTensorIndexText(const HostTensorAliasInfo& baseInfo,
                             llvm::StringRef rawIndexText,
                             HostTensorAliasInfo& outInfo) {
    outInfo = HostTensorAliasInfo();
    if (!baseInfo.valid || baseInfo.rank <= 0 || baseInfo.currentDim < 0 ||
        baseInfo.currentDim >= static_cast<int>(baseInfo.shapeExprs.size()) ||
        baseInfo.currentDim >= static_cast<int>(baseInfo.strideExprs.size())) {
        return false;
    }

    const std::string indexText = TrimCopy(rawIndexText.str());
    if (indexText.empty()) {
        return false;
    }

    outInfo = baseInfo;
    outInfo.valid = true;
    outInfo.isView = true;

    if (indexText.find('{') != std::string::npos &&
        indexText.find('}') != std::string::npos) {
        size_t openPos = indexText.find('{');
        size_t closePos = indexText.rfind('}');
        std::string inside = indexText.substr(openPos + 1, closePos - openPos - 1);
        std::vector<std::string> parts = SplitTopLevelByCommaPreserveNesting(inside);
        const std::string startExpr = parts.empty() ? "0" : parts[0];
        const std::string endExpr =
            parts.empty() ? baseInfo.shapeExprs[static_cast<size_t>(baseInfo.currentDim)]
                          : (parts.size() == 1
                                 ? BuildSumExpr(startExpr, "1")
                                 : parts[1]);
        const std::string sliceStrideExpr = parts.size() >= 3 ? parts[2] : "1";

        outInfo.offsetExpr = BuildSumExpr(
            baseInfo.offsetExpr,
            BuildProductExpr(startExpr, baseInfo.strideExprs[static_cast<size_t>(baseInfo.currentDim)]));
        outInfo.shapeExprs[static_cast<size_t>(baseInfo.currentDim)] =
            "((" + TrimCopy(endExpr) + ") - (" + TrimCopy(startExpr) + ") - 1) / (" +
            TrimCopy(sliceStrideExpr) + ") + 1";
        outInfo.strideExprs[static_cast<size_t>(baseInfo.currentDim)] =
            BuildProductExpr(baseInfo.strideExprs[static_cast<size_t>(baseInfo.currentDim)],
                             sliceStrideExpr);
        outInfo.currentDim = outInfo.rank > 0
            ? std::min(baseInfo.currentDim + 1, outInfo.rank - 1)
            : 0;
        return true;
    }

    outInfo.offsetExpr = BuildSumExpr(
        baseInfo.offsetExpr,
        BuildProductExpr(indexText, baseInfo.strideExprs[static_cast<size_t>(baseInfo.currentDim)]));
    outInfo.rank = std::max(0, baseInfo.rank - 1);
    outInfo.shapeExprs.erase(outInfo.shapeExprs.begin() + baseInfo.currentDim);
    outInfo.strideExprs.erase(outInfo.strideExprs.begin() + baseInfo.currentDim);
    outInfo.currentDim = outInfo.rank > 0
        ? std::min(baseInfo.currentDim, outInfo.rank - 1)
        : 0;
    return true;
}

std::string StripBalancedOuterParens(const std::string& rawText) {
    std::string text = TrimCopy(rawText);
    while (text.size() >= 2 && text.front() == '(' && text.back() == ')') {
        int depth = 0;
        bool wrapsWholeExpr = true;
        for (size_t i = 0; i < text.size(); ++i) {
            const char ch = text[i];
            if (ch == '(') {
                ++depth;
            } else if (ch == ')') {
                --depth;
                if (depth == 0 && i + 1 < text.size()) {
                    wrapsWholeExpr = false;
                    break;
                }
            }
            if (depth < 0) {
                wrapsWholeExpr = false;
                break;
            }
        }
        if (!wrapsWholeExpr || depth != 0) {
            break;
        }
        text = TrimCopy(text.substr(1, text.size() - 2));
    }
    return text;
}

static bool TryBuildTensorSuffixFallbackAlias(const std::string& name,
                                               llvm::StringRef ownerFn,
                                               HostTensorAliasInfo& outInfo) {
    if (!IsSimpleIdentifierText(name) || name.size() <= 7 ||
        name.rfind("_tensor") != name.size() - 7) {
        return false;
    }
    const std::string rootName = name.substr(0, name.size() - 7);
    if (rootName.empty()) return false;
    outInfo.valid = true;
    outInfo.ownerFunctionName = ownerFn.str();
    outInfo.aliasName = name;
    outInfo.rootContainerName = rootName;
    outInfo.rootDataExpr = rootName + ".data()";
    outInfo.offsetExpr = "0";
    outInfo.rank = 1;
    outInfo.currentDim = 0;
    outInfo.shapeExprs = {rootName + ".size()"};
    outInfo.strideExprs = {"1"};
    outInfo.isView = false;
    return true;
}

bool TryResolveHostTensorAliasFromText(
    llvm::StringRef rawExprText,
    llvm::StringRef ownerFunctionName,
    const std::unordered_map<std::string, HostTensorAliasInfo>& aliasMap,
    HostTensorAliasInfo& outInfo) {
    outInfo = HostTensorAliasInfo();
    const std::string exprText = StripBalancedOuterParens(rawExprText.str());
    if (exprText.empty()) {
        return false;
    }

    size_t firstOpenBracket = exprText.find('[');
    if (firstOpenBracket == std::string::npos) {
        if (TryLookupHostTensorAliasByName(
                ownerFunctionName,
                exprText,
                aliasMap,
                outInfo)) {
            return true;
        }
        if (TryLookupHostTensorAliasAcrossOwners(exprText, aliasMap, outInfo)) {
            return true;
        }

        if (TryBuildTensorSuffixFallbackAlias(exprText, ownerFunctionName, outInfo)) {
            return true;
        }
        return false;
    }

    const std::string baseText = TrimCopy(exprText.substr(0, firstOpenBracket));
    if (!TryLookupHostTensorAliasByName(ownerFunctionName, baseText, aliasMap, outInfo) &&
        !TryLookupHostTensorAliasAcrossOwners(baseText, aliasMap, outInfo)) {
        outInfo = HostTensorAliasInfo();
    }
    if (!outInfo.valid) {
        TryBuildTensorSuffixFallbackAlias(baseText, ownerFunctionName, outInfo);
    }
    if (!outInfo.valid) {
        return false;
    }

    size_t cursor = firstOpenBracket;
    while (cursor < exprText.size()) {
        while (cursor < exprText.size() &&
               std::isspace(static_cast<unsigned char>(exprText[cursor]))) {
            ++cursor;
        }
        if (cursor >= exprText.size()) {
            break;
        }
        if (exprText[cursor] != '[') {
            return false;
        }

        int bracketDepth = 0;
        size_t closeBracket = cursor;
        for (; closeBracket < exprText.size(); ++closeBracket) {
            if (exprText[closeBracket] == '[') {
                ++bracketDepth;
            } else if (exprText[closeBracket] == ']') {
                --bracketDepth;
                if (bracketDepth == 0) {
                    break;
                }
            }
        }
        if (closeBracket >= exprText.size() || bracketDepth != 0) {
            return false;
        }

        HostTensorAliasInfo indexedInfo;
        if (!TryApplyTensorIndexText(
                outInfo,
                exprText.substr(cursor + 1, closeBracket - cursor - 1),
                indexedInfo) ||
            !indexedInfo.valid) {
            return false;
        }
        outInfo = indexedInfo;
        cursor = closeBracket + 1;
    }

    return outInfo.valid;
}

namespace {

bool TryExtractTensorShapeExprs(const clang::Expr* expr,
                                int expectedRank,
                                const TensorAliasResolverContext& context,
                                std::vector<std::string>& outShapeExprs) {
    outShapeExprs.clear();
    if (!expr) {
        return false;
    }

    std::string text = TrimCopy(GetExprSourceText(expr, context));
    if (text.empty()) {
        return false;
    }

    size_t openPos = text.find('{');
    size_t closePos = text.rfind('}');
    if (openPos == std::string::npos || closePos == std::string::npos || closePos <= openPos) {
        return false;
    }

    std::string inside = text.substr(openPos + 1, closePos - openPos - 1);
    outShapeExprs = SplitTopLevelByCommaPreserveNesting(inside);
    if (expectedRank > 0 && static_cast<int>(outShapeExprs.size()) != expectedRank) {
        return false;
    }
    return !outShapeExprs.empty();
}

bool BuildHostDataPointerExprFromExpr(const clang::Expr* expr,
                                      const TensorAliasResolverContext& context,
                                      std::string& outRootName,
                                      std::string& outDataExpr) {
    outRootName.clear();
    outDataExpr.clear();
    if (!expr) {
        return false;
    }

    expr = expr->IgnoreParenImpCasts();
    if (const auto* cleanups = clang::dyn_cast<clang::ExprWithCleanups>(expr)) {
        return BuildHostDataPointerExprFromExpr(
            cleanups->getSubExpr(), context, outRootName, outDataExpr);
    }
    if (const auto* matTmp = clang::dyn_cast<clang::MaterializeTemporaryExpr>(expr)) {
        return BuildHostDataPointerExprFromExpr(
            matTmp->getSubExpr(), context, outRootName, outDataExpr);
    }
    if (const auto* bindTmp = clang::dyn_cast<clang::CXXBindTemporaryExpr>(expr)) {
        return BuildHostDataPointerExprFromExpr(
            bindTmp->getSubExpr(), context, outRootName, outDataExpr);
    }
    if (const auto* ctor = clang::dyn_cast<clang::CXXConstructExpr>(expr)) {
        if (ctor->getNumArgs() == 0) {
            return false;
        }
        return BuildHostDataPointerExprFromExpr(
            ctor->getArg(ctor->getNumArgs() - 1),
            context,
            outRootName,
            outDataExpr);
    }
    if (const auto* declRef = clang::dyn_cast<clang::DeclRefExpr>(expr)) {
        const clang::ValueDecl* valueDecl = declRef->getDecl();
        if (!valueDecl) {
            return false;
        }
        outRootName = valueDecl->getNameAsString();
        if (outRootName.empty()) {
            return false;
        }
        const clang::QualType type = valueDecl->getType();
        const std::string typeName = type.getAsString();
        const std::string canonicalTypeName = type.getCanonicalType().getAsString();
        if (LooksLikeStdVectorTypeName(typeName) ||
            LooksLikeStdVectorTypeName(canonicalTypeName)) {
            outDataExpr = outRootName + ".data()";
        } else {
            outDataExpr = outRootName;
        }
        return true;
    }

    if (clang::isa<clang::MemberExpr>(expr)) {
        outRootName = TrimCopy(GetExprSourceText(expr, context));
        if (outRootName.empty()) {
            return false;
        }
        outDataExpr = outRootName;
        return true;
    }

    return false;
}

bool TryResolveHostTensorAliasImpl(
    const clang::Expr* expr,
    llvm::StringRef ownerFunctionName,
    const std::unordered_map<std::string, HostTensorAliasInfo>& aliasMap,
    const TensorAliasResolverContext& context,
    HostTensorAliasInfo& outInfo);

bool TryResolveTensorSliceAlias(
    const clang::Expr* expr,
    llvm::StringRef ownerFunctionName,
    const std::unordered_map<std::string, HostTensorAliasInfo>& aliasMap,
    const TensorAliasResolverContext& context,
    HostTensorAliasInfo& outInfo) {
    outInfo = HostTensorAliasInfo();
    if (!expr) {
        return false;
    }

    expr = expr->IgnoreParenImpCasts();
    const clang::Expr* baseExpr = nullptr;
    const clang::Expr* indexExpr = nullptr;
    bool isSubscript = false;

    if (const auto* opCall = clang::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
        if (opCall->getOperator() == clang::OverloadedOperatorKind::OO_Subscript &&
            opCall->getNumArgs() >= 2) {
            baseExpr = opCall->getArg(0);
            indexExpr = opCall->getArg(1);
            isSubscript = true;
        }
    } else if (const auto* subscript = clang::dyn_cast<clang::ArraySubscriptExpr>(expr)) {
        baseExpr = subscript->getBase();
        indexExpr = subscript->getIdx();
        isSubscript = true;
    }

    if (!isSubscript || !baseExpr || !indexExpr) {
        return false;
    }

    HostTensorAliasInfo baseInfo;
    if (!TryResolveHostTensorAliasImpl(
            baseExpr, ownerFunctionName, aliasMap, context, baseInfo) ||
        !baseInfo.valid || baseInfo.rank <= 0 ||
        baseInfo.currentDim < 0 ||
        baseInfo.currentDim >= static_cast<int>(baseInfo.shapeExprs.size()) ||
        baseInfo.currentDim >= static_cast<int>(baseInfo.strideExprs.size())) {
        return false;
    }

    return TryApplyTensorIndexText(
        baseInfo,
        GetExprSourceText(indexExpr, context),
        outInfo);
}

bool IsScalarFillTensorCtorArg(const clang::Expr* expr) {
    if (!expr) return false;
    expr = expr->IgnoreParenImpCasts();
    if (clang::isa<clang::IntegerLiteral>(expr) ||
        clang::isa<clang::FloatingLiteral>(expr) ||
        clang::isa<clang::CXXBoolLiteralExpr>(expr)) {
        return true;
    }
    if (const auto* ctor = clang::dyn_cast<clang::CXXConstructExpr>(expr)) {
        if (ctor->getNumArgs() == 1) {
            return IsScalarFillTensorCtorArg(ctor->getArg(0));
        }
    }
    if (const auto* cleanups = clang::dyn_cast<clang::ExprWithCleanups>(expr)) {
        return IsScalarFillTensorCtorArg(cleanups->getSubExpr());
    }
    if (const auto* matTmp = clang::dyn_cast<clang::MaterializeTemporaryExpr>(expr)) {
        return IsScalarFillTensorCtorArg(matTmp->getSubExpr());
    }
    return false;
}

bool TryResolveDirectTensorCtorAlias(
    const clang::Expr* expr,
    llvm::StringRef ownerFunctionName,
    const std::unordered_map<std::string, HostTensorAliasInfo>& aliasMap,
    const TensorAliasResolverContext& context,
    const std::string& typeNameHint,
    HostTensorAliasInfo& outInfo) {
    outInfo = HostTensorAliasInfo();
    if (!expr) {
        return false;
    }

    expr = expr->IgnoreParenImpCasts();
    if (const auto* cleanups = clang::dyn_cast<clang::ExprWithCleanups>(expr)) {
        return TryResolveDirectTensorCtorAlias(
            cleanups->getSubExpr(),
            ownerFunctionName,
            aliasMap,
            context,
            typeNameHint,
            outInfo);
    }
    if (const auto* matTmp = clang::dyn_cast<clang::MaterializeTemporaryExpr>(expr)) {
        return TryResolveDirectTensorCtorAlias(
            matTmp->getSubExpr(),
            ownerFunctionName,
            aliasMap,
            context,
            typeNameHint,
            outInfo);
    }
    if (const auto* bindTmp = clang::dyn_cast<clang::CXXBindTemporaryExpr>(expr)) {
        return TryResolveDirectTensorCtorAlias(
            bindTmp->getSubExpr(),
            ownerFunctionName,
            aliasMap,
            context,
            typeNameHint,
            outInfo);
    }

    const auto* ctor = clang::dyn_cast<clang::CXXConstructExpr>(expr);
    if (!ctor) {
        return false;
    }

    const std::string typeName =
        typeNameHint.empty() ? expr->getType().getAsString() : typeNameHint;
    const int rank = ExtractTensorRankFromTypeName(typeName);
    if (rank <= 0) {
        return false;
    }

    outInfo.valid = true;
    outInfo.rank = rank;
    outInfo.currentDim = 0;
    outInfo.elementType = ExtractTensorElementTypeFromTypeName(typeName);

    if (rank == 1) {
        if (ctor->getNumArgs() == 0) {
            return false;
        }

        const clang::Expr* dataExpr = ctor->getArg(ctor->getNumArgs() - 1);
        HostTensorAliasInfo sourceAlias;
        if (TryResolveHostTensorAliasImpl(
                dataExpr, ownerFunctionName, aliasMap, context, sourceAlias) &&
            sourceAlias.valid) {
            outInfo.rootContainerName = sourceAlias.rootContainerName;
            outInfo.rootDataExpr = sourceAlias.rootDataExpr;
            outInfo.offsetExpr = sourceAlias.offsetExpr;
            outInfo.shapeExprs = sourceAlias.shapeExprs;
            outInfo.strideExprs = sourceAlias.strideExprs;
            outInfo.rank = sourceAlias.rank;
            outInfo.currentDim = sourceAlias.currentDim;
            outInfo.isView = sourceAlias.isView;
            if (outInfo.rank > 0) {
                return true;
            }
        }

        std::string rootName;
        std::string dataPointerExpr;
        if (!BuildHostDataPointerExprFromExpr(dataExpr, context, rootName, dataPointerExpr)) {
            return false;
        }
        outInfo.rootContainerName = rootName;
        outInfo.rootDataExpr = dataPointerExpr;
        outInfo.shapeExprs = {rootName + ".size()"};
        outInfo.strideExprs = {"1"};
        return true;
    }

    if (ctor->getNumArgs() < 2) {
        return false;
    }

    if (!TryExtractTensorShapeExprs(
            ctor->getArg(0),
            rank,
            context,
            outInfo.shapeExprs)) {
        return false;
    }
    outInfo.strideExprs = BuildRowMajorStrideExprs(outInfo.shapeExprs);

    // rank>1 ctor 的第二参数若是标量填充值（整数/浮点字面量），
    // 则该 ctor 没有 host 数据指针可解析；返回 false，让 runtime alias 覆盖。
    if (IsScalarFillTensorCtorArg(ctor->getArg(1))) {
        return false;
    }

    HostTensorAliasInfo sourceAlias;
    if (TryResolveHostTensorAliasImpl(
            ctor->getArg(1), ownerFunctionName, aliasMap, context, sourceAlias) &&
        sourceAlias.valid) {
        outInfo.rootContainerName = sourceAlias.rootContainerName;
        outInfo.rootDataExpr = sourceAlias.rootDataExpr;
        outInfo.offsetExpr = sourceAlias.offsetExpr;
        outInfo.isView = sourceAlias.isView;
        return true;
    }

    std::string rootName;
    std::string dataPointerExpr;
    if (!BuildHostDataPointerExprFromExpr(
            ctor->getArg(1), context, rootName, dataPointerExpr)) {
        return false;
    }
    outInfo.rootContainerName = rootName;
    outInfo.rootDataExpr = dataPointerExpr;
    return true;
}

bool TryResolveHostTensorAliasImpl(
    const clang::Expr* expr,
    llvm::StringRef ownerFunctionName,
    const std::unordered_map<std::string, HostTensorAliasInfo>& aliasMap,
    const TensorAliasResolverContext& context,
    HostTensorAliasInfo& outInfo) {
    outInfo = HostTensorAliasInfo();
    if (!expr) {
        return false;
    }

    expr = expr->IgnoreParenImpCasts();
    if (const auto* cleanups = clang::dyn_cast<clang::ExprWithCleanups>(expr)) {
        return TryResolveHostTensorAliasImpl(
            cleanups->getSubExpr(), ownerFunctionName, aliasMap, context, outInfo);
    }
    if (const auto* matTmp = clang::dyn_cast<clang::MaterializeTemporaryExpr>(expr)) {
        return TryResolveHostTensorAliasImpl(
            matTmp->getSubExpr(), ownerFunctionName, aliasMap, context, outInfo);
    }
    if (const auto* bindTmp = clang::dyn_cast<clang::CXXBindTemporaryExpr>(expr)) {
        return TryResolveHostTensorAliasImpl(
            bindTmp->getSubExpr(), ownerFunctionName, aliasMap, context, outInfo);
    }

    if (TryResolveTensorSliceAlias(expr, ownerFunctionName, aliasMap, context, outInfo)) {
        return outInfo.valid;
    }

    std::string aliasName;
    if (TryExtractIdentifierFromExpr(expr, aliasName) && !aliasName.empty()) {
        if (TryLookupHostTensorAliasByName(ownerFunctionName, aliasName, aliasMap, outInfo)) {
            return outInfo.valid;
        }
    }

    const std::string typeName = expr->getType().getAsString();
    const std::string canonicalTypeName = expr->getType().getCanonicalType().getAsString();
    if (LooksLikeTensorLikeTypeName(typeName) || LooksLikeTensorLikeTypeName(canonicalTypeName)) {
        if (TryResolveDirectTensorCtorAlias(
                expr,
                ownerFunctionName,
                aliasMap,
                context,
                typeName.empty() ? canonicalTypeName : typeName,
                outInfo)) {
            return outInfo.valid;
        }
    }

    return false;
}

} // namespace

void MergeHostTensorAliasMaps(
    std::unordered_map<std::string, HostTensorAliasInfo>& ioDst,
    const std::unordered_map<std::string, HostTensorAliasInfo>& src) {
    for (const auto& entry : src) {
        ioDst.insert_or_assign(entry.first, entry.second);
    }
}

void CollectLiveTensorAliasesInFunction(
    const clang::FunctionDecl* ownerFn,
    const TensorAliasResolverContext& context,
    std::unordered_map<std::string, HostTensorAliasInfo>& ioAliases) {
    if (!ownerFn || !ownerFn->hasBody() || !context.sourceManager) {
        return;
    }

    const clang::SourceManager& sm = *context.sourceManager;
    const std::string ownerName = ownerFn->getNameAsString();

    struct TensorAliasEvent {
        unsigned offset = 0;
        const clang::VarDecl* varDecl = nullptr;
        const clang::BinaryOperator* assignExpr = nullptr;
    };

    class TensorAliasEventCollector
        : public clang::RecursiveASTVisitor<TensorAliasEventCollector> {
    public:
        explicit TensorAliasEventCollector(const clang::SourceManager& sourceManager)
            : sm(sourceManager) {}

        bool VisitVarDecl(clang::VarDecl* varDecl) {
            if (!varDecl || !varDecl->getIdentifier() || !varDecl->hasInit() ||
                llvm::isa<clang::ParmVarDecl>(varDecl)) {
                return true;
            }
            clang::SourceLocation loc = sm.getExpansionLoc(varDecl->getBeginLoc());
            if (loc.isInvalid()) {
                return true;
            }
            TensorAliasEvent event;
            event.offset = sm.getFileOffset(loc);
            event.varDecl = varDecl;
            events.push_back(event);
            return true;
        }

        bool VisitBinaryOperator(clang::BinaryOperator* opExpr) {
            if (!opExpr || !opExpr->isAssignmentOp()) {
                return true;
            }
            clang::SourceLocation loc = sm.getExpansionLoc(opExpr->getBeginLoc());
            if (loc.isInvalid()) {
                return true;
            }
            TensorAliasEvent event;
            event.offset = sm.getFileOffset(loc);
            event.assignExpr = opExpr;
            events.push_back(event);
            return true;
        }

        std::vector<TensorAliasEvent> events;

    private:
        const clang::SourceManager& sm;
    } collector(sm);

    collector.TraverseStmt(const_cast<clang::Stmt*>(ownerFn->getBody()));
    std::sort(collector.events.begin(),
              collector.events.end(),
              [](const TensorAliasEvent& lhs, const TensorAliasEvent& rhs) {
                  if (lhs.offset != rhs.offset) {
                      return lhs.offset < rhs.offset;
                  }
                  return lhs.varDecl != nullptr && rhs.varDecl == nullptr;
              });

    std::unordered_map<std::string, HostTensorAliasInfo> localAliases;
    for (const auto& entry : ioAliases) {
        if (entry.second.ownerFunctionName == ownerName) {
            localAliases.insert_or_assign(entry.first, entry.second);
        }
    }

    for (const TensorAliasEvent& event : collector.events) {
        if (event.varDecl) {
            HostTensorAliasInfo info;
            const clang::VarDecl* varDecl = event.varDecl;
            const std::string typeName = varDecl->getType().getAsString();
            const std::string canonicalTypeName = varDecl->getType().getCanonicalType().getAsString();
            if (!LooksLikeTensorLikeTypeName(typeName) &&
                !LooksLikeTensorLikeTypeName(canonicalTypeName)) {
                continue;
            }
            if (TryResolveDirectTensorCtorAlias(
                    varDecl->getInit(),
                    ownerName,
                    localAliases,
                    context,
                    typeName.empty() ? canonicalTypeName : typeName,
                    info) ||
                TryResolveHostTensorAliasImpl(
                    varDecl->getInit(), ownerName, localAliases, context, info)) {
                info.valid = true;
                info.ownerFunctionName = ownerName;
                info.aliasName = varDecl->getNameAsString();
                if (info.elementType.empty()) {
                    info.elementType = ExtractTensorElementTypeFromTypeName(
                        typeName.empty() ? canonicalTypeName : typeName);
                }
                localAliases.insert_or_assign(
                    BuildHostTensorAliasKey(ownerName, info.aliasName),
                    info);
            }
            continue;
        }

        if (!event.assignExpr) {
            continue;
        }

        const clang::Expr* lhsExpr = event.assignExpr->getLHS()->IgnoreParenImpCasts();
        const clang::ValueDecl* lhsDecl = nullptr;
        if (const auto* lhsRef = clang::dyn_cast<clang::DeclRefExpr>(lhsExpr)) {
            lhsDecl = lhsRef->getDecl();
        }
        if (!lhsDecl) {
            continue;
        }

        std::string lhsName;
        if (const auto* lhsRef = clang::dyn_cast<clang::DeclRefExpr>(lhsExpr)) {
            lhsName = lhsRef->getDecl() ? lhsRef->getDecl()->getNameAsString() : "";
        }
        if (lhsName.empty()) {
            continue;
        }

        if (lhsDecl) {
            const std::string lhsTypeName = lhsDecl->getType().getAsString();
            const std::string lhsCanonicalTypeName = lhsDecl->getType().getCanonicalType().getAsString();
            if (!LooksLikeTensorLikeTypeName(lhsTypeName) &&
                !LooksLikeTensorLikeTypeName(lhsCanonicalTypeName) &&
                localAliases.count(BuildHostTensorAliasKey(ownerName, lhsName)) == 0) {
                continue;
            }
        }

        HostTensorAliasInfo rhsInfo;
        if (!TryResolveHostTensorAliasImpl(
                event.assignExpr->getRHS(), ownerName, localAliases, context, rhsInfo) ||
            !rhsInfo.valid) {
            continue;
        }

        rhsInfo.ownerFunctionName = ownerName;
        rhsInfo.aliasName = lhsName;
        localAliases.insert_or_assign(BuildHostTensorAliasKey(ownerName, lhsName), rhsInfo);
    }

    MergeHostTensorAliasMaps(ioAliases, localAliases);
}

// ============================================================
// Tensor alias 发射（plan + emitter）
// ============================================================
//
// 这些函数原本是 AscendCRewriteDriver.cpp 内的 static 实现，
// 现按 plan2detail Step 4 要求迁出并改成 plan-driven 形式。
// 任何代码片段都由 translator 自身生成（仅依赖 HostTensorAliasInfo 字段），
// 严禁保存或拼接用户源码 Stmt 原文。

std::string BuildHostTensorIndexedAccessExpr(
    const HostTensorAliasInfo& aliasInfo,
    const std::vector<std::string>& indexExprs) {
    if (!aliasInfo.valid || aliasInfo.rootDataExpr.empty()) {
        return "";
    }

    std::string offsetExpr = aliasInfo.offsetExpr;
    const size_t dimCount = std::min(indexExprs.size(), aliasInfo.strideExprs.size());
    for (size_t dim = 0; dim < dimCount; ++dim) {
        offsetExpr = BuildSumExpr(
            offsetExpr,
            BuildProductExpr(indexExprs[dim], aliasInfo.strideExprs[dim]));
    }
    return "(" + aliasInfo.rootDataExpr + ")[static_cast<long long>(" + offsetExpr + ")]";
}

std::string MaybeCastTensorAssignmentRhs(
    const std::string& rhsExpr,
    const HostTensorAliasInfo& lhsInfo,
    const HostTensorAliasInfo* rhsInfo) {
    const std::string trimmed = TrimCopy(rhsExpr);
    if (trimmed.empty()) return "";
    if (lhsInfo.elementType.empty()) return trimmed;
    if (rhsInfo && !rhsInfo->elementType.empty() &&
        lhsInfo.elementType == rhsInfo->elementType) {
        return trimmed;
    }
    return "static_cast<" + lhsInfo.elementType + ">(" + trimmed + ")";
}

namespace {

std::string EmitScalarReadInternal(const HostTensorAliasInfo& aliasInfo) {
    if (!aliasInfo.valid || aliasInfo.rank != 0 || aliasInfo.rootDataExpr.empty()) {
        return "";
    }
    return "(" + aliasInfo.rootDataExpr + ")[static_cast<long long>(" +
           aliasInfo.offsetExpr + ")]";
}

std::string EmitViewPrintInternal(const HostTensorAliasInfo& aliasInfo,
                                  const std::string& uniqueSuffix) {
    if (!aliasInfo.valid || aliasInfo.rootDataExpr.empty()) return "";
    std::string code;
    code += "    {\n";
    const size_t rank = std::min(aliasInfo.shapeExprs.size(), aliasInfo.strideExprs.size());
    if (rank == 0) {
        code += "        std::cout << (" + aliasInfo.rootDataExpr +
                ")[static_cast<long long>(" + aliasInfo.offsetExpr +
                ")] << std::endl;\n";
        code += "    }\n";
        return code;
    }

    std::string indent = "        ";
    std::string offsetExpr = "static_cast<long long>(" + aliasInfo.offsetExpr + ")";
    for (size_t dim = 0; dim < rank; ++dim) {
        const std::string indexName = "dacppPrintI" + uniqueSuffix + "_" +
                                      std::to_string(dim);
        code += indent + "std::cout << \"{\";\n";
        code += indent + "for (long long " + indexName + " = 0; " + indexName +
                " < static_cast<long long>(" + aliasInfo.shapeExprs[dim] +
                "); ++" + indexName + ") {\n";
        indent += "    ";
        code += indent + "if (" + indexName + " != 0) { std::cout << \", \"; }\n";
        offsetExpr += " + " + indexName + " * static_cast<long long>(" +
                      aliasInfo.strideExprs[dim] + ")";
    }
    code += indent + "std::cout << (" + aliasInfo.rootDataExpr + ")[" +
            offsetExpr + "];\n";
    for (size_t dim = rank; dim-- > 0;) {
        indent.resize(indent.size() - 4);
        code += indent + "}\n";
        code += indent + "std::cout << \"}\";\n";
    }
    code += "        std::cout << std::endl;\n";
    code += "    }\n";
    return code;
}

std::string EmitAssignmentInternal(
    const HostTensorAliasInfo& lhsInfo,
    const HostTensorAliasInfo* rhsInfo,
    const std::string& rhsScalarExpr,
    std::size_t uniqueIndex) {
    if (!lhsInfo.valid || lhsInfo.rootDataExpr.empty()) return "";

    const bool rhsIsAlias = rhsInfo && rhsInfo->valid;
    if (lhsInfo.rank == 0) {
        const std::string lhsScalarExpr = EmitScalarReadInternal(lhsInfo);
        std::string rhsExpr;
        if (rhsIsAlias) {
            if (rhsInfo->rank != 0) return "";
            rhsExpr = EmitScalarReadInternal(*rhsInfo);
        } else {
            rhsExpr = TrimCopy(rhsScalarExpr);
        }
        rhsExpr = MaybeCastTensorAssignmentRhs(
            rhsExpr, lhsInfo, rhsIsAlias ? rhsInfo : nullptr);
        if (lhsScalarExpr.empty() || rhsExpr.empty()) return "";
        return lhsScalarExpr + " = " + rhsExpr + ";\n";
    }

    if (lhsInfo.rank < 0 ||
        lhsInfo.shapeExprs.size() < static_cast<size_t>(lhsInfo.rank) ||
        lhsInfo.strideExprs.size() < static_cast<size_t>(lhsInfo.rank)) {
        return "";
    }
    if (rhsIsAlias && rhsInfo->rank > 0 &&
        (rhsInfo->rank != lhsInfo.rank ||
         rhsInfo->shapeExprs.size() < static_cast<size_t>(rhsInfo->rank) ||
         rhsInfo->strideExprs.size() < static_cast<size_t>(rhsInfo->rank))) {
        return "";
    }

    std::string code;
    code += "{\n";

    std::vector<std::string> loopIdxVars;
    std::vector<std::string> extentVars;
    loopIdxVars.reserve(static_cast<size_t>(lhsInfo.rank));
    extentVars.reserve(static_cast<size_t>(lhsInfo.rank));

    for (int dim = 0; dim < lhsInfo.rank; ++dim) {
        const std::string dimSuffix =
            std::to_string(uniqueIndex) + "_" + std::to_string(dim);
        const std::string extentVar = "copyLen" + dimSuffix;
        const std::string idxVar = "copyIdx" + dimSuffix;

        std::string extentExpr =
            "static_cast<long long>(" + lhsInfo.shapeExprs[static_cast<size_t>(dim)] + ")";
        if (rhsIsAlias && rhsInfo->rank > 0) {
            extentExpr =
                "std::min(static_cast<long long>(" +
                lhsInfo.shapeExprs[static_cast<size_t>(dim)] +
                "), static_cast<long long>(" +
                rhsInfo->shapeExprs[static_cast<size_t>(dim)] + "))";
        }
        code += "    const long long " + extentVar + " = " + extentExpr + ";\n";
        extentVars.push_back(extentVar);
        loopIdxVars.push_back(idxVar);
    }

    std::function<void(size_t, const std::string&)> appendLoopNest =
        [&](size_t dim, const std::string& indent) {
            if (dim >= loopIdxVars.size()) {
                const std::string lhsAccessExpr =
                    BuildHostTensorIndexedAccessExpr(lhsInfo, loopIdxVars);
                std::string rhsExpr;
                if (rhsIsAlias) {
                    rhsExpr = rhsInfo->rank == 0
                        ? EmitScalarReadInternal(*rhsInfo)
                        : BuildHostTensorIndexedAccessExpr(*rhsInfo, loopIdxVars);
                } else {
                    rhsExpr = TrimCopy(rhsScalarExpr);
                }
                rhsExpr = MaybeCastTensorAssignmentRhs(
                    rhsExpr, lhsInfo, rhsIsAlias ? rhsInfo : nullptr);
                if (!lhsAccessExpr.empty() && !rhsExpr.empty()) {
                    code += indent + lhsAccessExpr + " = " + rhsExpr + ";\n";
                }
                return;
            }
            code += indent + "for (long long " + loopIdxVars[dim] + " = 0; " +
                    loopIdxVars[dim] + " < " + extentVars[dim] + "; ++" +
                    loopIdxVars[dim] + ") {\n";
            appendLoopNest(dim + 1, indent + "    ");
            code += indent + "}\n";
        };

    appendLoopNest(0, "    ");
    code += "}\n";
    return code;
}

} // namespace

// ---- Plan builders ----

TensorAliasAccessPlan BuildTensorScalarReadPlan(const HostTensorAliasInfo& aliasInfo) {
    TensorAliasAccessPlan plan;
    plan.kind = TensorAliasAccessKind::ScalarRead;
    plan.lhsInfo = aliasInfo;
    plan.valid = aliasInfo.valid && aliasInfo.rank == 0 && !aliasInfo.rootDataExpr.empty();
    return plan;
}

TensorAliasAccessPlan BuildTensorViewPrintPlan(
    const HostTensorAliasInfo& aliasInfo,
    const std::string& uniqueSuffix) {
    TensorAliasAccessPlan plan;
    plan.kind = TensorAliasAccessKind::ViewPrint;
    plan.lhsInfo = aliasInfo;
    plan.uniqueSuffix = uniqueSuffix;
    plan.valid = aliasInfo.valid && !aliasInfo.rootDataExpr.empty();
    return plan;
}

TensorAliasAccessPlan BuildTensorAssignmentPlan(
    const HostTensorAliasInfo& lhsInfo,
    const HostTensorAliasInfo* rhsInfo,
    const std::string& rhsScalarExpr,
    std::size_t uniqueIndex) {
    TensorAliasAccessPlan plan;
    plan.kind = TensorAliasAccessKind::Assignment;
    plan.lhsInfo = lhsInfo;
    plan.rhsIsAlias = rhsInfo && rhsInfo->valid;
    if (plan.rhsIsAlias) plan.rhsInfo = *rhsInfo;
    plan.rhsScalarExpr = rhsScalarExpr;
    plan.uniqueIndex = uniqueIndex;

    bool ok = lhsInfo.valid && !lhsInfo.rootDataExpr.empty();
    if (ok && plan.rhsIsAlias) {
        // rank 一致性（rank-0 → rank-0；rank>0 时 rhs 要么 rank-0 标量、要么形状可对齐）。
        if (lhsInfo.rank == 0) {
            ok = plan.rhsInfo.rank == 0;
        } else {
            ok = plan.rhsInfo.rank == 0 ||
                 plan.rhsInfo.rank == lhsInfo.rank;
        }
    }
    if (ok && !plan.rhsIsAlias && lhsInfo.rank == 0 && TrimCopy(rhsScalarExpr).empty()) {
        ok = false;  // rank-0 必须有 RHS
    }
    plan.valid = ok;
    return plan;
}

// ---- Plan dispatcher ----

std::string EmitTensorAliasAccessPlan(const TensorAliasAccessPlan& plan) {
    if (!plan.valid) return "";
    switch (plan.kind) {
        case TensorAliasAccessKind::ScalarRead:
            return EmitScalarReadInternal(plan.lhsInfo);
        case TensorAliasAccessKind::ViewPrint:
            return EmitViewPrintInternal(plan.lhsInfo, plan.uniqueSuffix);
        case TensorAliasAccessKind::Assignment:
            return EmitAssignmentInternal(
                plan.lhsInfo,
                plan.rhsIsAlias ? &plan.rhsInfo : nullptr,
                plan.rhsScalarExpr,
                plan.uniqueIndex);
        case TensorAliasAccessKind::Unknown:
            break;
    }
    return "";
}

// ---- 旧字符串接口的 plan-wrapper（driver 在 Step 8 拆除前的过渡 API）----

std::string EmitTensorScalarRead(const HostTensorAliasInfo& aliasInfo) {
    return EmitTensorAliasAccessPlan(BuildTensorScalarReadPlan(aliasInfo));
}

std::string EmitTensorViewPrint(
    const HostTensorAliasInfo& aliasInfo,
    const std::string& uniqueSuffix) {
    return EmitTensorAliasAccessPlan(
        BuildTensorViewPrintPlan(aliasInfo, uniqueSuffix));
}

std::string EmitTensorAssignment(
    const HostTensorAliasInfo& lhsInfo,
    const HostTensorAliasInfo* rhsInfo,
    const std::string& rhsScalarExpr,
    std::size_t uniqueIndex) {
    return EmitTensorAliasAccessPlan(
        BuildTensorAssignmentPlan(lhsInfo, rhsInfo, rhsScalarExpr, uniqueIndex));
}

} // namespace ascendc
} // namespace dacppTranslator
