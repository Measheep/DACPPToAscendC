#ifndef TRANSLATOR_REWRITER_ASCENDC_MLIR_CALC_CONTEXT_H
#define TRANSLATOR_REWRITER_ASCENDC_MLIR_CALC_CONTEXT_H

#include <string>
#include <unordered_map>
#include <vector>

#include "llvm/ADT/StringRef.h"

namespace clang {
class Expr;
class FunctionDecl;
}

namespace dacppTranslator {
class Calc;

namespace ascendc_mlir {

struct CalcInlineFnInfo {
    std::vector<std::string> paramNames;
    std::string returnExpr;
    const clang::Expr* returnExprAst = nullptr;
    const clang::FunctionDecl* functionDecl = nullptr;
};

struct CalcContextSummary {
    std::unordered_map<std::string, double> numericConsts;
    std::unordered_map<std::string, CalcInlineFnInfo> pureInlineFns;
    std::vector<std::string> unresolvedSymbols;
};

CalcContextSummary ExtractCalcContext(const Calc* calc, llvm::StringRef sourceText);
bool TryEvaluateConstExpr(llvm::StringRef expr, const CalcContextSummary& summary, double& out);

} // namespace ascendc_mlir
} // namespace dacppTranslator

#endif // TRANSLATOR_REWRITER_ASCENDC_MLIR_CALC_CONTEXT_H
