/*
构建calc语义的上下文。
- 提取数值常量与可纯内联函数。
- 维护未解析符号集合。
- 提供常量表达式求值能力，供 Lowering 与 Host 绑定复用。
*/
#include "ascendc_mlir/CalcContext.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Calc.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/Lex/Lexer.h"

namespace dacppTranslator {
namespace ascendc_mlir {
namespace {

constexpr int kMaxEvalDepth = 8;

// 处理Trim Copy相关逻辑。
std::string TrimCopy(llvm::StringRef text) {
    size_t begin = 0;
    size_t end = text.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return text.substr(begin, end - begin).str();
}

// 检查Is Ident Start相关条件。
bool IsIdentStart(char c) {
    return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
}

// 检查Is Ident Char相关条件。
bool IsIdentChar(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

// 处理To Literal相关逻辑。
std::string ToLiteral(double value) {
    std::ostringstream oss;
    oss << std::setprecision(17) << value;
    return oss.str();
}

// 查找Find Matching相关数据。
size_t FindMatching(const std::string& text, size_t openPos, char openCh, char closeCh) {
    if (openPos >= text.size() || text[openPos] != openCh) {
        return std::string::npos;
    }

    int depth = 0;
    bool inSingleQuote = false;
    bool inDoubleQuote = false;
    bool escaping = false;

    for (size_t i = openPos; i < text.size(); ++i) {
        char c = text[i];
        if (escaping) {
            escaping = false;
            continue;
        }
        if (c == '\\') {
            escaping = true;
            continue;
        }
        if (!inDoubleQuote && c == '\'') {
            inSingleQuote = !inSingleQuote;
            continue;
        }
        if (!inSingleQuote && c == '"') {
            inDoubleQuote = !inDoubleQuote;
            continue;
        }
        if (inSingleQuote || inDoubleQuote) {
            continue;
        }

        if (c == openCh) {
            ++depth;
        } else if (c == closeCh) {
            --depth;
            if (depth == 0) {
                return i;
            }
        }
    }

    return std::string::npos;
}

// 检查Replace Identifier相关逻辑。
std::string ReplaceIdentifier(const std::string& text,
                              llvm::StringRef ident,
                              llvm::StringRef replacement) {
    if (ident.empty()) {
        return text;
    }

    std::string out;
    out.reserve(text.size() + replacement.size());

    for (size_t i = 0; i < text.size();) {
        if (i + ident.size() <= text.size() &&
            text.compare(i, ident.size(), ident.data(), ident.size()) == 0) {
            bool leftOk = (i == 0) || !IsIdentChar(text[i - 1]);
            bool rightOk = (i + ident.size() >= text.size()) ||
                           !IsIdentChar(text[i + ident.size()]);
            if (leftOk && rightOk) {
                out += replacement.str();
                i += ident.size();
                continue;
            }
        }

        out.push_back(text[i]);
        ++i;
    }

    return out;
}

bool TryEvaluateConstExprInternal(llvm::StringRef expr,
                                  const CalcContextSummary& summary,
                                  const std::unordered_map<std::string, double>& locals,
                                  int depth,
                                  double& outValue);

struct EvalParser {
    llvm::StringRef expr;
    const CalcContextSummary& summary;
    const std::unordered_map<std::string, double>& locals;
    int depth = 0;
    size_t pos = 0;
    bool ok = true;

    // 处理Skip Spaces相关逻辑。
    void SkipSpaces() {
        while (pos < expr.size() && std::isspace(static_cast<unsigned char>(expr[pos]))) {
            ++pos;
        }
    }

    // 提取或解析Parse Identifier相关信息。
    std::string ParseIdentifier() {
        if (pos >= expr.size() || !IsIdentStart(expr[pos])) {
            ok = false;
            return "";
        }

        size_t begin = pos++;
        while (pos < expr.size() && IsIdentChar(expr[pos])) {
            ++pos;
        }
        return expr.substr(begin, pos - begin).str();
    }

    // 提取或解析Parse Number相关信息。
    double ParseNumber() {
        const char* beginPtr = expr.data() + pos;
        char* endPtr = nullptr;
        double value = std::strtod(beginPtr, &endPtr);
        if (endPtr == beginPtr) {
            ok = false;
            return 0.0;
        }

        pos += static_cast<size_t>(endPtr - beginPtr);
        while (pos < expr.size() && std::isalpha(static_cast<unsigned char>(expr[pos]))) {
            ++pos;
        }
        return value;
    }

    // 提取或解析Parse Call Argument相关信息。
    bool ParseCallArgument(llvm::StringRef ident, double& outArg) {
        if (pos >= expr.size() || expr[pos] != '(') {
            ok = false;
            return false;
        }

        size_t argBegin = pos + 1;
        size_t closePos = FindMatching(expr.str(), pos, '(', ')');
        if (closePos == std::string::npos) {
            ok = false;
            return false;
        }

        llvm::StringRef argText = expr.substr(argBegin, closePos - argBegin);
        std::string trimmedArg = TrimCopy(argText);
        pos = closePos + 1;
        if (trimmedArg.empty()) {
            ok = false;
            return false;
        }

        if (!TryEvaluateConstExprInternal(trimmedArg, summary, locals, depth + 1, outArg)) {
            ok = false;
            return false;
        }

        (void)ident;
        return true;
    }

    // 尝试处理Try Eval Pure Inline Fn相关逻辑。
    bool TryEvalPureInlineFn(llvm::StringRef fnName, double argValue, double& outResult) {
        auto fnIt = summary.pureInlineFns.find(fnName.str());
        if (fnIt == summary.pureInlineFns.end()) {
            return false;
        }

        const CalcInlineFnInfo& fnInfo = fnIt->second;
        if (fnInfo.paramNames.size() != 1 || fnInfo.returnExpr.empty()) {
            return false;
        }

        std::string substituted = ReplaceIdentifier(fnInfo.returnExpr, fnInfo.paramNames[0], ToLiteral(argValue));
        return TryEvaluateConstExprInternal(substituted, summary, locals, depth + 1, outResult);
    }

    // 提取或解析Parse Factor相关信息。
    double ParseFactor() {
        SkipSpaces();
        if (!ok || pos >= expr.size()) {
            ok = false;
            return 0.0;
        }

        if (expr[pos] == '+') {
            ++pos;
            return ParseFactor();
        }
        if (expr[pos] == '-') {
            ++pos;
            return -ParseFactor();
        }

        if (expr[pos] == '(') {
            ++pos;
            double v = ParseExpression();
            SkipSpaces();
            if (pos >= expr.size() || expr[pos] != ')') {
                ok = false;
                return 0.0;
            }
            ++pos;
            return v;
        }

        if (std::isdigit(static_cast<unsigned char>(expr[pos])) || expr[pos] == '.') {
            return ParseNumber();
        }

        if (IsIdentStart(expr[pos])) {
            std::string ident = ParseIdentifier();
            if (!ok) {
                return 0.0;
            }

            SkipSpaces();
            if (pos < expr.size() && expr[pos] == '(') {
                double argValue = 0.0;
                if (!ParseCallArgument(ident, argValue)) {
                    ok = false;
                    return 0.0;
                }

                double callResult = 0.0;
                if (TryEvalPureInlineFn(ident, argValue, callResult)) {
                    return callResult;
                }

                ok = false;
                return 0.0;
            }

            auto localIt = locals.find(ident);
            if (localIt != locals.end()) {
                return localIt->second;
            }

            auto constIt = summary.numericConsts.find(ident);
            if (constIt != summary.numericConsts.end()) {
                return constIt->second;
            }

            if (ident == "M_PI") {
                return 3.14159265358979323846;
            }
            if (ident == "M_E") {
                return 2.71828182845904523536;
            }

            ok = false;
            return 0.0;
        }

        ok = false;
        return 0.0;
    }

    // 提取或解析Parse Term相关信息。
    double ParseTerm() {
        double lhs = ParseFactor();
        while (ok) {
            SkipSpaces();
            if (pos >= expr.size() || (expr[pos] != '*' && expr[pos] != '/')) {
                break;
            }

            char op = expr[pos++];
            double rhs = ParseFactor();
            if (!ok) {
                return 0.0;
            }

            if (op == '*') {
                lhs *= rhs;
            } else {
                if (std::fabs(rhs) < 1e-15) {
                    ok = false;
                    return 0.0;
                }
                lhs /= rhs;
            }
        }
        return lhs;
    }

    // 提取或解析Parse Expression相关信息。
    double ParseExpression() {
        double lhs = ParseTerm();
        while (ok) {
            SkipSpaces();
            if (pos >= expr.size() || (expr[pos] != '+' && expr[pos] != '-')) {
                break;
            }

            char op = expr[pos++];
            double rhs = ParseTerm();
            if (!ok) {
                return 0.0;
            }

            if (op == '+') {
                lhs += rhs;
            } else {
                lhs -= rhs;
            }
        }
        return lhs;
    }
};

// 尝试处理Try Evaluate Const Expr Internal相关逻辑。
bool TryEvaluateConstExprInternal(llvm::StringRef expr,
                                  const CalcContextSummary& summary,
                                  const std::unordered_map<std::string, double>& locals,
                                  int depth,
                                  double& outValue) {
    std::string trimmed = TrimCopy(expr);
    if (trimmed.empty() || depth > kMaxEvalDepth) {
        return false;
    }

    EvalParser parser{trimmed, summary, locals, depth};
    double value = parser.ParseExpression();
    parser.SkipSpaces();

    if (!parser.ok || parser.pos != parser.expr.size() || !std::isfinite(value)) {
        return false;
    }

    outValue = value;
    return true;
}

// 尝试处理Try APValue To Double相关逻辑。
bool TryAPValueToDouble(const clang::APValue& value, double& outValue) {
    if (value.isInt()) {
        outValue = static_cast<double>(value.getInt().getSExtValue());
        return true;
    }
    if (value.isFloat()) {
        outValue = value.getFloat().convertToDouble();
        return true;
    }
    return false;
}

// 功能：尝试处理Try Eval Expr As Double相关逻辑。
bool TryEvalExprAsDouble(const clang::Expr* expr,
                         const clang::ASTContext& astContext,
                         double& outValue) {
    if (!expr) {
        return false;
    }

    clang::Expr::EvalResult eval;
    if (!expr->EvaluateAsRValue(eval, astContext)) {
        return false;
    }

    return TryAPValueToDouble(eval.Val, outValue);
}

// 功能：尝试处理Try Get Expr Source Text相关逻辑。
bool TryGetExprSourceText(const clang::Expr* expr,
                          const clang::ASTContext& astContext,
                          std::string& outText) {
    outText.clear();
    if (!expr) {
        return false;
    }

    const clang::SourceManager& sourceManager = astContext.getSourceManager();
    clang::CharSourceRange range = clang::CharSourceRange::getTokenRange(expr->getSourceRange());
    bool invalid = false;
    llvm::StringRef text = clang::Lexer::getSourceText(range, sourceManager, astContext.getLangOpts(), &invalid);
    if (invalid || text.empty()) {
        return false;
    }

    outText = TrimCopy(text);
    return !outText.empty();
}

// 功能：收集Collect Decls Recursively相关数据。
void CollectDeclsRecursively(const clang::DeclContext* declContext,
                             std::vector<const clang::Decl*>& outDecls) {
    if (!declContext) {
        return;
    }

    for (const clang::Decl* decl : declContext->decls()) {
        outDecls.push_back(decl);
        if (const auto* subContext = llvm::dyn_cast<clang::DeclContext>(decl)) {
            CollectDeclsRecursively(subContext, outDecls);
        }
    }
}

// 功能：检查Is Numeric Type相关条件。
bool IsNumericType(clang::QualType type) {
    return type->isArithmeticType() || type->isEnumeralType();
}

// 功能：检查Is Const Like Var相关条件。
bool IsConstLikeVar(const clang::VarDecl* varDecl) {
    if (!varDecl) {
        return false;
    }
    return varDecl->isConstexpr() || varDecl->getType().isConstQualified();
}

// 功能：提取或解析Extract Numeric Consts From Ast相关信息。
void ExtractNumericConstsFromAst(const clang::ASTContext& astContext,
                                 CalcContextSummary& summary,
                                 std::unordered_set<std::string>& unresolvedSet) {
    std::vector<const clang::Decl*> decls;
    CollectDeclsRecursively(astContext.getTranslationUnitDecl(), decls);

    for (const clang::Decl* decl : decls) {
        if (const auto* enumConst = llvm::dyn_cast<clang::EnumConstantDecl>(decl)) {
            summary.numericConsts[enumConst->getNameAsString()] =
                static_cast<double>(enumConst->getInitVal().getSExtValue());
            continue;
        }

        const auto* varDecl = llvm::dyn_cast<clang::VarDecl>(decl);
        if (!varDecl || !varDecl->hasInit()) {
            continue;
        }
        if (!IsConstLikeVar(varDecl) || !IsNumericType(varDecl->getType())) {
            continue;
        }

        const std::string name = varDecl->getNameAsString();
        if (name.empty()) {
            continue;
        }

        double value = 0.0;
        if (TryEvalExprAsDouble(varDecl->getInit(), astContext, value)) {
            summary.numericConsts[name] = value;
        } else {
            unresolvedSet.insert(name);
        }
    }
}

struct PendingInlineFn {
    std::string name;
    std::vector<std::string> paramNames;
    std::vector<std::pair<std::string, std::string>> locals;
    std::string returnExpr;
    const clang::Expr* returnExprAst = nullptr;
    const clang::FunctionDecl* functionDecl = nullptr;
};

// 功能：检查Is Simple Numeric Helper相关条件。
bool IsSimpleNumericHelper(const clang::FunctionDecl* functionDecl,
                           std::vector<std::string>& outParamNames) {
    outParamNames.clear();
    if (!functionDecl || !functionDecl->isThisDeclarationADefinition()) {
        return false;
    }
    if (functionDecl->isDependentContext() || functionDecl->isTemplated()) {
        return false;
    }
    if (functionDecl->getNumParams() == 0) {
        return false;
    }
    if (!IsNumericType(functionDecl->getReturnType())) {
        return false;
    }

    for (unsigned i = 0; i < functionDecl->getNumParams(); ++i) {
        const clang::ParmVarDecl* param = functionDecl->getParamDecl(i);
        if (!param || !IsNumericType(param->getType())) {
            return false;
        }
        std::string paramName = param->getNameAsString();
        if (paramName.empty()) {
            return false;
        }
        outParamNames.push_back(paramName);
    }

    return !outParamNames.empty();
}

// 功能：尝试处理Try Collect Pending Inline Fn相关逻辑。
bool TryCollectPendingInlineFn(const clang::FunctionDecl* functionDecl,
                               const clang::ASTContext& astContext,
                               PendingInlineFn& outPending) {
    outPending = PendingInlineFn();
    outPending.name = functionDecl->getNameAsString();
    outPending.functionDecl = functionDecl;

    if (outPending.name.empty()) {
        return false;
    }

    if (!IsSimpleNumericHelper(functionDecl, outPending.paramNames)) {
        return false;
    }

    const auto* body = llvm::dyn_cast<clang::CompoundStmt>(functionDecl->getBody());
    if (!body) {
        return false;
    }

    bool sawReturn = false;
    for (const clang::Stmt* stmt : body->body()) {
        if (!stmt || llvm::isa<clang::NullStmt>(stmt)) {
            continue;
        }

        if (const auto* declStmt = llvm::dyn_cast<clang::DeclStmt>(stmt)) {
            for (const clang::Decl* decl : declStmt->decls()) {
                const auto* varDecl = llvm::dyn_cast<clang::VarDecl>(decl);
                if (!varDecl || !varDecl->hasInit()) {
                    return false;
                }
                if (!IsNumericType(varDecl->getType())) {
                    return false;
                }

                std::string initExpr;
                if (!TryGetExprSourceText(varDecl->getInit(), astContext, initExpr)) {
                    return false;
                }
                outPending.locals.emplace_back(varDecl->getNameAsString(), initExpr);
            }
            continue;
        }

        if (const auto* returnStmt = llvm::dyn_cast<clang::ReturnStmt>(stmt)) {
            if (sawReturn || !returnStmt->getRetValue()) {
                return false;
            }
            if (!TryGetExprSourceText(returnStmt->getRetValue(), astContext, outPending.returnExpr)) {
                return false;
            }
            outPending.returnExprAst = returnStmt->getRetValue()->IgnoreParenImpCasts();
            sawReturn = true;
            continue;
        }

        return false;
    }

    return sawReturn && !outPending.returnExpr.empty();
}

// 功能：提取或解析Extract Pure Inline Fns From Ast相关信息。
void ExtractPureInlineFnsFromAst(const clang::ASTContext& astContext,
                                 CalcContextSummary& summary,
                                 std::unordered_set<std::string>& unresolvedSet) {
    std::vector<const clang::Decl*> decls;
    CollectDeclsRecursively(astContext.getTranslationUnitDecl(), decls);

    std::vector<PendingInlineFn> pending;
    for (const clang::Decl* decl : decls) {
        const auto* fn = llvm::dyn_cast<clang::FunctionDecl>(decl);
        if (!fn || !fn->hasBody()) {
            continue;
        }

        PendingInlineFn entry;
        if (TryCollectPendingInlineFn(fn, astContext, entry)) {
            pending.push_back(std::move(entry));
        }
    }

    for (size_t pass = 0; pass < pending.size() + 8; ++pass) {
        bool progressed = false;

        for (const PendingInlineFn& item : pending) {
            if (summary.pureInlineFns.count(item.name) > 0) {
                continue;
            }

            std::unordered_map<std::string, double> locals;
            bool localsReady = true;
            for (const auto& local : item.locals) {
                if (local.first.empty()) {
                    localsReady = false;
                    break;
                }

                double localValue = 0.0;
                if (!TryEvaluateConstExprInternal(local.second, summary, locals, 0, localValue)) {
                    localsReady = false;
                    break;
                }
                locals[local.first] = localValue;
            }
            if (!localsReady) {
                continue;
            }

            std::string resolvedReturn = item.returnExpr;
            for (const auto& local : locals) {
                resolvedReturn = ReplaceIdentifier(resolvedReturn, local.first, ToLiteral(local.second));
            }

            double probe = 0.0;
            std::unordered_map<std::string, double> probeLocals;
            for (const std::string& paramName : item.paramNames) {
                probeLocals[paramName] = 1.0;
            }
            if (!TryEvaluateConstExprInternal(resolvedReturn, summary, probeLocals, 0, probe)) {
                continue;
            }

            CalcInlineFnInfo info;
            info.paramNames = item.paramNames;
            info.returnExpr = resolvedReturn;
            info.returnExprAst = item.returnExprAst;
            info.functionDecl = item.functionDecl;

            summary.pureInlineFns[item.name] = info;
            if (item.functionDecl) {
                std::string qualifiedName = item.functionDecl->getQualifiedNameAsString();
                if (!qualifiedName.empty()) {
                    summary.pureInlineFns[qualifiedName] = info;
                }
            }
            progressed = true;
        }

        if (!progressed) {
            break;
        }
    }

    for (const PendingInlineFn& item : pending) {
        if (summary.pureInlineFns.count(item.name) == 0) {
            unresolvedSet.insert(item.name);
        }
    }
}

} // namespace

// 提取calc相关上下文信息。
CalcContextSummary ExtractCalcContext(const Calc* calc, llvm::StringRef sourceText) {
    (void)sourceText;

    CalcContextSummary summary;
    if (!calc) {
        return summary;
    }

    const clang::FunctionDecl* functionDecl = calc->getCalcFunctionDecl();
    if (!functionDecl) {
        return summary;
    }

    const clang::ASTContext& astContext = functionDecl->getASTContext();
    std::unordered_set<std::string> unresolvedSet;

    ExtractNumericConstsFromAst(astContext, summary, unresolvedSet);
    ExtractPureInlineFnsFromAst(astContext, summary, unresolvedSet);

    summary.unresolvedSymbols.assign(unresolvedSet.begin(), unresolvedSet.end());
    std::sort(summary.unresolvedSymbols.begin(), summary.unresolvedSymbols.end());
    return summary;
}

// 尝试处理常量表达式。
bool TryEvaluateConstExpr(llvm::StringRef expr,
                          const CalcContextSummary& summary,
                          double& out) {
    return TryEvaluateConstExprInternal(expr, summary, {}, 0, out);
}

} // namespace ascendc_mlir
} // namespace dacppTranslator
