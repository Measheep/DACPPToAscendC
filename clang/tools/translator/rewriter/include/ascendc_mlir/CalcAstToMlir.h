#ifndef TRANSLATOR_REWRITER_ASCENDC_MLIR_CALC_AST_TO_MLIR_H
#define TRANSLATOR_REWRITER_ASCENDC_MLIR_CALC_AST_TO_MLIR_H

#include <memory>
#include <string>
#include <vector>

#if defined(DACPP_TRANSLATOR_HAS_MLIR)
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#endif

namespace dacppTranslator {
class Calc;

namespace ascendc_mlir {

struct CalcContextSummary;

struct CalcRecordFieldBinding {
    std::string fieldName;
    std::string physicalName;
    std::string fieldType;
};

struct CalcParamBinding {
    std::string logicalName;
    std::vector<CalcRecordFieldBinding> fields;
};

struct CalcMlirArtifact {
    bool valid = false;
#if defined(DACPP_TRANSLATOR_HAS_MLIR)
    std::shared_ptr<mlir::MLIRContext> context;
    mlir::ModuleOp module;
#endif
};

struct CalcLoweringDiagnostic {
    std::string message;
    std::string nodeKind;
};

struct CalcAstToMlirOptions {
    bool verifyModule = true;
    const CalcContextSummary* contextSummary = nullptr;
    const std::vector<CalcParamBinding>* paramBindings = nullptr;
    unsigned maxInlineDepth = 8;
};

struct CalcAstToMlirResult {
    bool mlirAvailable = false;
    bool lowered = false;
    bool verified = false;
    std::string moduleText;
    std::string diagnostics;
    std::vector<CalcLoweringDiagnostic> loweringDiagnostics;
    CalcMlirArtifact artifact;
};

CalcAstToMlirResult LowerCalcAstToMlir(const Calc* calc,
                                       const CalcAstToMlirOptions& options = {});

} // namespace ascendc_mlir
} // namespace dacppTranslator

#endif // TRANSLATOR_REWRITER_ASCENDC_MLIR_CALC_AST_TO_MLIR_H
