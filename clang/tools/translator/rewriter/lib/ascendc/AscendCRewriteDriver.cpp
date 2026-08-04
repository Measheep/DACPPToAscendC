#include <iostream>
#include <fstream>
#include <regex>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "Rewriter.h"
#include "ascendc/AscendCArtifacts.h"
#include "ascendc/AscendCFlowPlan.h"
#include "ascendc/HostFlowBuilder.h"
#include "ascendc/HostFlowEmitter.h"
#include "ascendc/HostCodeNames.h"
#include "ascendc/HostRuntimeEmitter.h"
#include "ascendc/HostTensorAliasInfo.h"
#include "ascendc/AscendCKernelSupport.h"
#include "ascendc/AscendCPolicy.h"
#include "ascendc/AscendCRewriteDriver.h"
#include "ascendc/AscendCRuntimeBinding.h"
#include "ascendc/AscendCSourceUtils.h"
#include "ascendc/AscendCTensorAlias.h"
#include "ascendc/AscendCTypeMapping.h"
#include "ascendc/VectorTiling.h"
#include "ascendc_template.h"
#include "Split.h"
#include "Param.h"
#include "ascendc_mlir/CalcAstToMlir.h"
#include "ascendc_mlir/CalcAnalysis.h"
#include "ascendc_mlir/AscendCPlan.h"
#include "ascendc_mlir/AscendCEmitter.h"
#include "ascendc_mlir/CalcMlirPipeline.h"
#include "ascendc_mlir/CalcContext.h"
#include "ascendc_mlir/CalcShellInfo.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Lex/Lexer.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "llvm/ADT/StringRef.h"

using namespace dacppTranslator;

using dacppTranslator::ascendc::AscendCRewriteContext;
using dacppTranslator::ascendc::AscendCExpressionArtifact;
using dacppTranslator::ascendc::AscendCRouteKind;
using dacppTranslator::ascendc::EscapeRegexText;
using dacppTranslator::ascendc::HostTensorAliasInfo;
using dacppTranslator::ascendc::InferExtraScratchBufferCountFromExpr;
using dacppTranslator::ascendc::GetTranslationPolicy;
using dacppTranslator::ascendc::BuildHostTensorAliasKey;
using dacppTranslator::ascendc::BuildProductExpr;
using dacppTranslator::ascendc::BuildRecordFieldPhysicalName;
using dacppTranslator::ascendc::BuildSumExpr;
using dacppTranslator::ascendc::IsIdentCharText;
using dacppTranslator::ascendc::IsRootRuntimeBindingKind;
using dacppTranslator::ascendc::IsSimpleIdentifierText;
using dacppTranslator::ascendc::LooksLikeTensorWrapperTypeName;
using dacppTranslator::ascendc::MapToCubeDeviceStagingType;
using dacppTranslator::ascendc::MapToCubeKernelType;
using dacppTranslator::ascendc::MapToCubeUserHostType;
using dacppTranslator::ascendc::MapToKernelType;
using dacppTranslator::ascendc::MapToStdType;
using dacppTranslator::ascendc::MapToTilingDtype;
using dacppTranslator::ascendc::NormalizeRedundantDirectViewCopy;
using dacppTranslator::ascendc::ReplaceAllRegex;
using dacppTranslator::ascendc::ParamUsesRecordElementType;
using dacppTranslator::ascendc::RecordFieldLayout;
using dacppTranslator::ascendc::RecordFieldLayoutEntry;
using dacppTranslator::ascendc::RuntimeBindingNodeKind;
using dacppTranslator::ascendc::RuntimeBindingAstContext;
using dacppTranslator::ascendc::RuntimeBindingResolveResult;
using dacppTranslator::ascendc::RuntimeParamBindingInfo;
using dacppTranslator::ascendc::ResolveRuntimeBindingFromScope;
using dacppTranslator::ascendc::RunCalcMlirPipelineDetailed;
using dacppTranslator::ascendc::StripBalancedOuterParens;
using dacppTranslator::ascendc::StripTrailingSemicolon;
using dacppTranslator::ascendc::TensorAliasResolverContext;
using dacppTranslator::ascendc::TranslationPolicy;
using dacppTranslator::ascendc::TrimCopy;
using dacppTranslator::ascendc::TryExtractAliasedSourceFromExpr;
using dacppTranslator::ascendc::TryExtractIdentifierFromExpr;
using dacppTranslator::ascendc::TryApplyTensorIndexText;
using dacppTranslator::ascendc::TryResolveHostTensorAliasFromText;
using dacppTranslator::ascendc::TryResolveSimpleSymbolValue;


namespace {

extern const clang::SourceManager* gRuntimeBindingSourceManager;
static std::string GetStmtSourceText(const clang::Stmt* stmt);
static std::string GetExprSourceText(const clang::Expr* expr);
static const clang::FunctionDecl* FindEnclosingFunction(const clang::BinaryOperator* dacExpr);
static std::vector<std::string> ResolveShellCallArgumentsFromAst(const clang::BinaryOperator* dacExpr,
                                                                  Shell* shell);

// ============================================================
// HostPreludeCollection：BuildHostPreludeCodeFromCollection 与
// tensor alias 查找的输入容器。
// ============================================================
struct HostPreludeCollection {
    std::vector<std::string> usingDecls;
    std::vector<std::string> constDecls;
    std::vector<std::string> globalVarDecls;
    std::vector<std::string> recordDecls;
    std::vector<std::string> helperFunctionDefs;
    std::unordered_map<std::string, HostTensorAliasInfo> tensorAliases;
    // resolvedRoots 是 collector 用于 caller-chain 递归时的去重容器；emitter 不消费。
    std::unordered_set<std::string> resolvedRoots;
};

// ===== Temporal schedule recognition =====
static bool IsPositiveLength(const std::string& lengthStr) {
    try {
        return std::stoll(lengthStr) > 0;
    } catch (...) {
        return false;
    }
}

static bool TryGetShapeTotalLength(ShellParam* param, std::string& outLength) {
    if (!param) {
        return false;
    }

    int dim = param->getDim();
    if (dim <= 0) {
        return false;
    }

    long long total = 1;
    for (int i = 0; i < dim; ++i) {
        int shapeDim = param->getShape(i);
        if (shapeDim <= 0) {
            return false;
        }
        total *= shapeDim;
    }

    outLength = std::to_string(total);
    return true;
}

static const std::unordered_set<std::string> kBlockedIdentifierTokens = {
    "if", "for", "while", "switch", "return", "sizeof",
    "static_cast", "reinterpret_cast", "const_cast", "dynamic_cast",
    "std", "size_t", "int", "long", "short", "float", "double",
    "void", "char", "bool", "unsigned", "signed",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "int8_t", "int16_t", "int32_t", "int64_t",
    "true", "false"
};

static bool IsStmtContainingLoc(const clang::Stmt* stmt,
                                const clang::SourceManager& sm,
                                clang::SourceLocation loc) {
    if (!stmt) {
        return false;
    }
    clang::SourceLocation begin = sm.getExpansionLoc(stmt->getBeginLoc());
    clang::SourceLocation end = sm.getExpansionLoc(stmt->getEndLoc());
    loc = sm.getExpansionLoc(loc);
    if (begin.isInvalid() || end.isInvalid() || loc.isInvalid()) {
        return false;
    }
    return !sm.isBeforeInTranslationUnit(loc, begin) &&
           !sm.isBeforeInTranslationUnit(end, loc);
}

static unsigned GetStmtSpan(const clang::Stmt* stmt, const clang::SourceManager& sm) {
    if (!stmt) {
        return std::numeric_limits<unsigned>::max();
    }
    clang::SourceLocation begin = sm.getExpansionLoc(stmt->getBeginLoc());
    clang::SourceLocation end = sm.getExpansionLoc(stmt->getEndLoc());
    if (begin.isInvalid() || end.isInvalid()) {
        return std::numeric_limits<unsigned>::max();
    }
    if (sm.getFileID(begin) != sm.getFileID(end)) {
        return std::numeric_limits<unsigned>::max();
    }
    unsigned beginOff = sm.getFileOffset(begin);
    unsigned endOff = sm.getFileOffset(end);
    return endOff >= beginOff ? (endOff - beginOff) : std::numeric_limits<unsigned>::max();
}

template <typename StmtT>
static const StmtT* FindInnermostStmtOfType(const clang::Stmt* root,
                                             const clang::SourceManager& sm,
                                             clang::SourceLocation loc) {
    const StmtT* best = nullptr;
    unsigned bestSpan = std::numeric_limits<unsigned>::max();

    std::function<void(const clang::Stmt*)> visit = [&](const clang::Stmt* stmt) {
        if (!stmt || !IsStmtContainingLoc(stmt, sm, loc)) {
            return;
        }
        if (const auto* target = llvm::dyn_cast<StmtT>(stmt)) {
            unsigned span = GetStmtSpan(target, sm);
            if (!best || span < bestSpan) {
                best = target;
                bestSpan = span;
            }
        }
        for (const clang::Stmt* child : stmt->children()) {
            visit(child);
        }
    };

    visit(root);
    return best;
}

static bool TryExtractIdentifierFromExprSimple(const clang::Expr* expr, std::string& outName) {
    outName.clear();
    if (!expr) {
        return false;
    }
    expr = expr->IgnoreParenImpCasts();
    if (const auto* declRef = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
        outName = declRef->getNameInfo().getAsString();
        return !outName.empty();
    }
    if (const auto* memberExpr = llvm::dyn_cast<clang::MemberExpr>(expr)) {
        outName = memberExpr->getMemberNameInfo().getAsString();
        return !outName.empty();
    }
    return false;
}

static bool TryGetForLoopControl(const clang::ForStmt* forStmt,
                                 std::string& outIndexVar,
                                 std::string& outBeginExpr,
                                 std::string& outUpperBoundExpr,
                                 std::string& outStepExpr) {
    outIndexVar.clear();
    outBeginExpr.clear();
    outUpperBoundExpr.clear();
    outStepExpr.clear();
    if (!forStmt) {
        return false;
    }

    if (const auto* declStmt = llvm::dyn_cast_or_null<clang::DeclStmt>(forStmt->getInit())) {
        if (!declStmt->isSingleDecl()) {
            return false;
        }
        if (const auto* varDecl = llvm::dyn_cast<clang::VarDecl>(declStmt->getSingleDecl())) {
            outIndexVar = varDecl->getNameAsString();
            if (const clang::Expr* initExpr = varDecl->getInit()) {
                outBeginExpr = TrimCopy(GetExprSourceText(initExpr));
            }
        }
    }
    if (outIndexVar.empty()) {
        return false;
    }
    if (outBeginExpr.empty()) {
        outBeginExpr = "0";
    }

    if (const auto* cond = llvm::dyn_cast_or_null<clang::BinaryOperator>(forStmt->getCond())) {
        std::string lhsName;
        if (TryExtractIdentifierFromExprSimple(cond->getLHS(), lhsName) && lhsName == outIndexVar) {
            outUpperBoundExpr = TrimCopy(GetExprSourceText(cond->getRHS()));
        }
    }
    if (outUpperBoundExpr.empty()) {
        return false;
    }

    if (const clang::Stmt* incStmt = forStmt->getInc()) {
        outStepExpr = TrimCopy(GetStmtSourceText(incStmt));
    }
    if (outStepExpr.empty()) {
        outStepExpr = "++" + outIndexVar;
    }
    return true;
}

static bool TryFindContainingStmtIndex(const clang::CompoundStmt* body,
                                       const clang::SourceManager& sm,
                                       clang::SourceLocation loc,
                                       unsigned& outIndex) {
    outIndex = 0;
    if (!body) {
        return false;
    }
    unsigned idx = 0;
    for (const clang::Stmt* stmt : body->body()) {
        if (IsStmtContainingLoc(stmt, sm, loc)) {
            outIndex = idx;
            return true;
        }
        ++idx;
    }
    return false;
}

static bool TryGetAssignedExprToVar(const clang::Stmt* stmt,
                                    llvm::StringRef varName,
                                    const clang::Expr*& outExpr) {
    outExpr = nullptr;
    if (!stmt || varName.empty()) {
        return false;
    }

    if (const auto* declStmt = llvm::dyn_cast<clang::DeclStmt>(stmt)) {
        for (const clang::Decl* decl : declStmt->decls()) {
            const auto* varDecl = llvm::dyn_cast<clang::VarDecl>(decl);
            if (!varDecl) {
                continue;
            }
            if (varDecl->getName() == varName && varDecl->getInit()) {
                outExpr = varDecl->getInit();
                return true;
            }
        }
    }

    const auto* binOp = llvm::dyn_cast<clang::BinaryOperator>(stmt);
    if (!binOp && llvm::isa<clang::Expr>(stmt)) {
        binOp = llvm::dyn_cast<clang::BinaryOperator>(llvm::cast<clang::Expr>(stmt)->IgnoreParenImpCasts());
    }
    if (binOp && binOp->isAssignmentOp()) {
        std::string lhsName;
        if (TryExtractIdentifierFromExprSimple(binOp->getLHS(), lhsName) && lhsName == varName) {
            outExpr = binOp->getRHS();
            return true;
        }
    }

    return false;
}

static bool TryParseAffineExprAgainstVar(const std::string& exprText,
                                         llvm::StringRef varName,
                                         int& outOffset) {
    outOffset = 0;
    std::string expr = TrimCopy(exprText);
    while (expr.size() >= 2 && expr.front() == '(' && expr.back() == ')') {
        expr = TrimCopy(expr.substr(1, expr.size() - 2));
    }
    if (expr == varName.str()) {
        outOffset = 0;
        return true;
    }

    const std::string escapedVar = std::regex_replace(varName.str(), std::regex(R"([.^$|()\[\]{}*+?\\])"), R"(\\$&)" );
    std::smatch match;
    std::regex plusMinusRegex("^" + escapedVar + R"(\s*([+-])\s*(\d+)$)");
    if (std::regex_match(expr, match, plusMinusRegex)) {
        int value = std::stoi(match[2].str());
        outOffset = (match[1].str() == "-") ? -value : value;
        return true;
    }

    std::regex reverseRegex(R"(^(\d+)\s*([+-])\s*)" + escapedVar + R"($)");
    if (std::regex_match(expr, match, reverseRegex)) {
        int value = std::stoi(match[1].str());
        outOffset = (match[2].str() == "-") ? -value : value;
        return true;
    }

    return false;
}

static bool TryParseTemporalSliceInitExpr(const std::string& initExprText,
                                          llvm::StringRef loopIndexVar,
                                          std::string& outStateTensor,
                                          int& outTimeOffset) {
    outStateTensor.clear();
    outTimeOffset = 0;
    std::string text = TrimCopy(initExprText);
    std::smatch match;
    std::regex sliceRegex(R"(^\s*([A-Za-z_]\w*)\s*\[[^\]]+\]\s*\[\s*([^\]]+)\s*\]\s*$)");
    if (!std::regex_match(text, match, sliceRegex)) {
        return false;
    }

    std::string stateTensor = TrimCopy(match[1].str());
    std::string timeExpr = TrimCopy(match[2].str());
    int offset = 0;
    if (stateTensor.empty() || !TryParseAffineExprAgainstVar(timeExpr, loopIndexVar, offset)) {
        return false;
    }

    outStateTensor = stateTensor;
    outTimeOffset = offset;
    return true;
}

static bool TryResolveTemporalSliceBinding(const clang::CompoundStmt* loopBody,
                                            unsigned dacStmtIndex,
                                            llvm::StringRef loopIndexVar,
                                            llvm::StringRef argVar,
                                            bool isOutput,
                                            ASCENDC_TEMPLATE::TemporalLoopSchedule& outSchedule) {
    if (!loopBody || argVar.empty()) {
        return false;
    }

    std::vector<const clang::Stmt*> stmtList;
    stmtList.reserve(std::distance(loopBody->body_begin(), loopBody->body_end()));
    for (const clang::Stmt* stmt : loopBody->body()) {
        stmtList.push_back(stmt);
    }

    const int lastIdx = std::min(static_cast<int>(dacStmtIndex), static_cast<int>(stmtList.size()) - 1);
    for (int idx = lastIdx; idx >= 0; --idx) {
        const clang::Stmt* stmt = stmtList[static_cast<size_t>(idx)];
        const clang::Expr* assignedExpr = nullptr;
        if (!TryGetAssignedExprToVar(stmt, argVar, assignedExpr)) {
            continue;
        }

        std::string stateTensor;
        int timeOffset = 0;
        if (!TryParseTemporalSliceInitExpr(GetExprSourceText(assignedExpr), loopIndexVar, stateTensor, timeOffset)) {
            continue;
        }

        if (isOutput && !outSchedule.stateTensorName.empty() && outSchedule.stateTensorName != stateTensor) {
            return false;
        }

        if (isOutput) {
            outSchedule.outputSliceVar = argVar.str();
            outSchedule.outputTimeOffset = timeOffset;
        } else {
            outSchedule.inputSliceVar = argVar.str();
            outSchedule.inputTimeOffset = timeOffset;
        }
        outSchedule.stateTensorName = stateTensor;
        return true;
    }

    return false;
}

static bool TryResolveLoopCarriedWriteback(const clang::CompoundStmt* loopBody,
                                           unsigned dacStmtIndex,
                                           llvm::StringRef outerLoopIndexVar,
                                           ASCENDC_TEMPLATE::TemporalLoopSchedule& outSchedule) {
    if (!loopBody || outSchedule.stateTensorName.empty() || outSchedule.outputSliceVar.empty()) {
        return false;
    }

    unsigned idx = 0;
    for (const clang::Stmt* stmt : loopBody->body()) {
        if (idx <= dacStmtIndex) {
            ++idx;
            continue;
        }
        ++idx;

        const auto* forStmt = llvm::dyn_cast<clang::ForStmt>(stmt);
        if (!forStmt) {
            continue;
        }

        std::string innerIdxVar;
        std::string innerBeginExpr;
        std::string innerEndExpr;
        std::string innerStepExpr;
        if (!TryGetForLoopControl(forStmt, innerIdxVar, innerBeginExpr, innerEndExpr, innerStepExpr)) {
            continue;
        }

        std::string bodyText = GetStmtSourceText(forStmt->getBody());
        std::smatch match;
        std::regex writebackRegex(
            R"(([A-Za-z_]\w*)\s*\[\s*([^\]]+)\s*\]\s*\[\s*([^\]]+)\s*\]\s*=\s*([A-Za-z_]\w*)\s*\[\s*([^\]]+)\s*\]\s*;)");
        if (!std::regex_search(bodyText, match, writebackRegex)) {
            continue;
        }

        const std::string lhsTensor = TrimCopy(match[1].str());
        const std::string lhsSpatialExpr = TrimCopy(match[2].str());
        const std::string lhsTimeExpr = TrimCopy(match[3].str());
        const std::string rhsTensor = TrimCopy(match[4].str());
        const std::string rhsSpatialExpr = TrimCopy(match[5].str());

        if (lhsTensor != outSchedule.stateTensorName || rhsTensor != outSchedule.outputSliceVar) {
            continue;
        }

        int dstSpatialOffset = 0;
        int dstTimeOffset = 0;
        int srcSpatialOffset = 0;
        if (!TryParseAffineExprAgainstVar(lhsSpatialExpr, innerIdxVar, dstSpatialOffset)) {
            continue;
        }
        if (!TryParseAffineExprAgainstVar(lhsTimeExpr, outerLoopIndexVar, dstTimeOffset)) {
            continue;
        }
        if (!TryParseAffineExprAgainstVar(rhsSpatialExpr, innerIdxVar, srcSpatialOffset)) {
            continue;
        }

        outSchedule.hasLoopCarriedWriteback = true;
        outSchedule.writebackLoopIndexVar = innerIdxVar;
        outSchedule.writebackRangeBeginExpr = innerBeginExpr;
        outSchedule.writebackRangeEndExpr = innerEndExpr;
        outSchedule.writebackDstSpatialOffset = dstSpatialOffset;
        outSchedule.writebackDstTimeOffset = dstTimeOffset;
        outSchedule.writebackSrcSpatialOffset = srcSpatialOffset;
        return true;
    }

    return false;
}

static bool TryResolveTemporalBoundaryUpdate(const clang::CompoundStmt* loopBody,
                                             unsigned dacStmtIndex,
                                             const ASCENDC_TEMPLATE::TemporalLoopSchedule& schedule,
                                             std::string& outBoundarySummary) {
    outBoundarySummary = "none";
    if (!loopBody || schedule.stateTensorName.empty()) {
        return false;
    }

    int boundaryStmtCount = 0;
    unsigned idx = 0;
    for (const clang::Stmt* stmt : loopBody->body()) {
        if (idx == dacStmtIndex) {
            ++idx;
            continue;
        }
        ++idx;

        std::string stmtText = TrimCopy(GetStmtSourceText(stmt));
        if (stmtText.empty()) {
            continue;
        }
        if (stmtText.find(schedule.stateTensorName + "[") == std::string::npos) {
            continue;
        }
        if (stmtText.find(schedule.outputSliceVar) != std::string::npos) {
            continue;
        }
        if (stmtText.find(schedule.loopIndexVar) == std::string::npos) {
            continue;
        }
        ++boundaryStmtCount;
    }

    if (boundaryStmtCount <= 0) {
        return false;
    }
    outBoundarySummary = "count=" + std::to_string(boundaryStmtCount);
    return true;
}

static bool TryExtractTemporalLoopScheduleFromAst(
    dacppTranslator::DacppFile* dacppFile,
    ASCENDC_TEMPLATE::TemporalLoopSchedule& outSchedule) {
    outSchedule = ASCENDC_TEMPLATE::TemporalLoopSchedule();
    if (!dacppFile || !gRuntimeBindingSourceManager) {
        return false;
    }

    // Keep this path deterministic for current in-place rewrite: one DAC expression -> one schedule.
    if (dacppFile->getNumExpression() != 1) {
        return false;
    }

    const clang::SourceManager& sm = *gRuntimeBindingSourceManager;
    bool hasTemporalCue = false;
    std::string lastFailure;

    for (int exprIdx = 0; exprIdx < dacppFile->getNumExpression(); ++exprIdx) {
        Expression* expr = dacppFile->getExpression(exprIdx);
        if (!expr || !expr->getShell()) {
            continue;
        }

        const clang::BinaryOperator* dacExpr = expr->getDacExpr();
        if (!dacExpr) {
            continue;
        }

        const clang::FunctionDecl* ownerFunction = FindEnclosingFunction(dacExpr);
        if (!ownerFunction || !ownerFunction->getBody()) {
            continue;
        }

        clang::SourceLocation dacLoc = sm.getExpansionLoc(dacExpr->getExprLoc());
        const clang::ForStmt* outerFor = FindInnermostStmtOfType<clang::ForStmt>(ownerFunction->getBody(), sm, dacLoc);
        if (!outerFor) {
            continue;
        }

        const auto* loopBody = llvm::dyn_cast<clang::CompoundStmt>(outerFor->getBody());
        if (!loopBody) {
            continue;
        }

        std::string loopIndexVar;
        std::string loopBeginExpr;
        std::string loopUpperBoundExpr;
        std::string loopStepExpr;
        if (!TryGetForLoopControl(outerFor,
                                  loopIndexVar,
                                  loopBeginExpr,
                                  loopUpperBoundExpr,
                                  loopStepExpr)) {
            continue;
        }

        unsigned dacStmtIndex = 0;
        if (!TryFindContainingStmtIndex(loopBody, sm, dacLoc, dacStmtIndex)) {
            continue;
        }

        Shell* shell = expr->getShell();
        std::vector<std::string> shellCallArgs = ResolveShellCallArgumentsFromAst(dacExpr, shell);
        if (shellCallArgs.empty()) {
            lastFailure = "shell call args cannot be resolved from AST";
            continue;
        }

        dacppTranslator::ascendc_mlir::CalcShellSummary shellSummary =
            dacppTranslator::ascendc_mlir::BuildCalcShellSummary(shell);

        int inputParamIdx = -1;
        int outputParamIdx = -1;
        std::vector<int> scalarParamIdxs;
        for (int j = 0; j < shell->getNumParams(); ++j) {
            ShellParam* shellParam = shell->getShellParam(j);
            if (!shellParam || static_cast<size_t>(j) >= shellSummary.params.size()) {
                continue;
            }
            int rw = shellParam->getRw();
            const auto& summary = shellSummary.params[static_cast<size_t>(j)];
            const bool isScalarLike = summary.isScalar || summary.isBroadcastScalar;
            if ((rw == 0 || rw == 2) && !isScalarLike && inputParamIdx < 0) {
                inputParamIdx = j;
            }
            if ((rw == 1 || rw == 2) && outputParamIdx < 0) {
                outputParamIdx = j;
            }
            if ((rw == 0 || rw == 2) && isScalarLike) {
                scalarParamIdxs.push_back(j);
            }
        }

        if (inputParamIdx < 0 || outputParamIdx < 0 ||
            static_cast<size_t>(inputParamIdx) >= shellCallArgs.size() ||
            static_cast<size_t>(outputParamIdx) >= shellCallArgs.size()) {
            lastFailure = "cannot determine temporal input/output binding from shell args";
            continue;
        }

        std::string inputArg = TrimCopy(shellCallArgs[static_cast<size_t>(inputParamIdx)]);
        std::string outputArg = TrimCopy(shellCallArgs[static_cast<size_t>(outputParamIdx)]);
        if (!IsSimpleIdentifierText(inputArg) || !IsSimpleIdentifierText(outputArg)) {
            lastFailure = "temporal input/output shell args must be identifier references";
            continue;
        }

        ASCENDC_TEMPLATE::TemporalLoopSchedule candidate;
        candidate.expressionIndex = exprIdx;
        candidate.ownerFunctionName = ownerFunction->getNameInfo().getAsString();
        candidate.loopIndexVar = loopIndexVar;
        candidate.loopBeginExpr = loopBeginExpr;
        candidate.loopUpperBoundExpr = loopUpperBoundExpr;
        candidate.loopStepExpr = loopStepExpr;

        bool hasInputBinding = TryResolveTemporalSliceBinding(
            loopBody, dacStmtIndex, loopIndexVar, inputArg,
            false, candidate);
        bool hasOutputBinding = TryResolveTemporalSliceBinding(
            loopBody, dacStmtIndex, loopIndexVar, outputArg,
            true, candidate);

        if (hasInputBinding || hasOutputBinding) {
            hasTemporalCue = true;
        }

        if (!hasInputBinding || !hasOutputBinding) {
            lastFailure = "failed to resolve temporal input/output slice binding from AST";
            continue;
        }

        for (int scalarParamIdx : scalarParamIdxs) {
            if (scalarParamIdx < 0 || static_cast<size_t>(scalarParamIdx) >= shellSummary.params.size() ||
                static_cast<size_t>(scalarParamIdx) >= shellCallArgs.size()) {
                continue;
            }
            candidate.scalarParamNames.push_back(shellSummary.params[static_cast<size_t>(scalarParamIdx)].name);
            candidate.scalarArgNames.push_back(TrimCopy(shellCallArgs[static_cast<size_t>(scalarParamIdx)]));
        }

        if (!TryResolveLoopCarriedWriteback(loopBody, dacStmtIndex, loopIndexVar, candidate)) {
            hasTemporalCue = true;
            lastFailure = "failed to resolve loop-carried writeback relation from AST";
            continue;
        }

        std::string boundarySummary;
        candidate.hasBoundaryUpdate = TryResolveTemporalBoundaryUpdate(
            loopBody,
            dacStmtIndex,
            candidate,
            boundarySummary);
        candidate.boundaryUpdateSummary = boundarySummary;

        candidate.recognized = true;
        candidate.requireNoFallback = true;
        outSchedule = candidate;
        return true;
    }

    if (hasTemporalCue) {
        outSchedule.requireNoFallback = true;
        outSchedule.failureReason = lastFailure.empty()
            ? "temporal schedule cues found in AST but required bindings are incomplete"
            : lastFailure;
    }
    return false;
}


// ===== Runtime binding and source utilities =====
const clang::TranslationUnitDecl* gRuntimeBindingTUDecl = nullptr;
const clang::SourceManager* gRuntimeBindingSourceManager = nullptr;
const clang::LangOptions* gRuntimeBindingLangOptions = nullptr;
clang::LangOptions gRuntimeBindingLangOptionsStorage;
const clang::FunctionDecl* gRuntimeBindingCurrentOwnerFunction = nullptr;
clang::SourceLocation gRuntimeBindingCurrentDacExprLoc;

class RuntimeBufferIdRegistry {
public:
    void reset() {
        ids.clear();
        next.clear();
        produced.clear();
    }

    dacppTranslator::ascendc::FlowBufferId get(
        dacppTranslator::ascendc::FlowRouteKind route,
        const std::string& logicalKey) {
        const int routeKey = static_cast<int>(route);
        const std::string key = std::to_string(routeKey) + ":" + logicalKey;
        const auto found = ids.find(key);
        if (found != ids.end()) return found->second;
        dacppTranslator::ascendc::FlowBufferId& value = next[routeKey];
        const dacppTranslator::ascendc::FlowBufferId result = value++;
        ids.emplace(key, result);
        return result;
    }

    bool wasProduced(dacppTranslator::ascendc::FlowRouteKind route,
                     const std::string& logicalKey) const {
        return produced.count(std::to_string(static_cast<int>(route)) + ":" +
                              logicalKey) != 0;
    }

    void markProduced(dacppTranslator::ascendc::FlowRouteKind route,
                      const std::string& logicalKey) {
        produced.insert(std::to_string(static_cast<int>(route)) + ":" +
                        logicalKey);
    }

private:
    std::unordered_map<std::string,
        dacppTranslator::ascendc::FlowBufferId> ids;
    std::unordered_map<int,
        dacppTranslator::ascendc::FlowBufferId> next;
    std::unordered_set<std::string> produced;
};

RuntimeBufferIdRegistry gRuntimeBufferIds;
dacppTranslator::ascendc::FlowRouteRuntimeNames gVectorRuntimeNames;
dacppTranslator::ascendc::FlowRouteRuntimeNames gCubeRuntimeNames;
std::string gHostNameError;

static dacppTranslator::ascendc::FlowRouteRuntimeNames AllocateRuntimeNames(
    dacppTranslator::ascendc::FlowRouteKind route,
    const clang::FunctionDecl* owner,
    const clang::TranslationUnitDecl* tu) {
    using namespace dacppTranslator::ascendc;
    FlowRouteRuntimeNames result;
    result.route = route;
    HostNameAllocator names;
    if (owner && tu) {
        for (const std::string& identifier :
             CollectHostVisibleIdentifiers(*owner, *tu)) {
            names.reserve(identifier);
        }
    }
    result.session = names.get(HostNameRole::Session).text;
    result.routeObject = names.get(HostNameRole::Route).text;
    result.rankCount = names.get(HostNameRole::RankCount).text;
    result.device = names.get(HostNameRole::Device).text;
    result.stream = names.get(HostNameRole::Stream).text;
    if (!names.ok() && gHostNameError.empty()) gHostNameError = names.error();
    return result;
}

static void PopulateArtifactRuntimeContract(
    dacppTranslator::ascendc::AscendCExpressionArtifact& artifact,
    const TranslationPolicy& policy) {
    using namespace dacppTranslator::ascendc;
    artifact.runtimeRequirements.packaging = policy.hostRuntimePackaging;
    artifact.runtimeRequirements.logLevel = policy.hostRuntimeLogLevel;
    if (artifact.routeKind == AscendCRouteKind::Vector) {
        artifact.runtimeRequirements.require(HostRuntimeFeature::CheckedMath);
        artifact.runtimeRequirements.require(HostRuntimeFeature::VectorTiling);
    } else {
        artifact.runtimeRequirements.require(HostRuntimeFeature::CubeTiling);
        artifact.runtimeRequirements.require(HostRuntimeFeature::Fp16Bridge);
    }
    artifact.runtimeRequirements.require(
        policy.enableMulticard ? HostRuntimeFeature::MulticardRoute
                               : HostRuntimeFeature::SingleCard);
    for (const std::string& text :
         {artifact.runtimeNames.session, artifact.runtimeNames.rankCount,
          artifact.runtimeNames.routeObject, artifact.runtimeNames.device,
          artifact.runtimeNames.stream}) {
        FlowGeneratedName name;
        name.text = text;
        name.kind = FlowGeneratedNameKind::FrameworkLocal;
        artifact.generatedNames.push_back(std::move(name));
    }
}

static RuntimeBindingAstContext MakeRuntimeBindingAstContext() {
    RuntimeBindingAstContext context;
    context.sourceManager = gRuntimeBindingSourceManager;
    context.langOptions = gRuntimeBindingLangOptions;
    context.currentOwnerFunction = gRuntimeBindingCurrentOwnerFunction;
    context.currentDacExprLoc = gRuntimeBindingCurrentDacExprLoc;
    return context;
}

static TensorAliasResolverContext MakeTensorAliasResolverContext() {
    TensorAliasResolverContext context;
    context.sourceManager = gRuntimeBindingSourceManager;
    context.langOptions = gRuntimeBindingLangOptions;
    return context;
}

class RuntimeBindingContextGuard {
public:
    explicit RuntimeBindingContextGuard(const AscendCRewriteContext& context) {
        gRuntimeBindingTUDecl =
            context.dacppFile ? context.dacppFile->getTranslationUnitDecl() : nullptr;
        if (context.rewriter) {
            gRuntimeBindingSourceManager = &context.rewriter->getSourceMgr();
            gRuntimeBindingLangOptionsStorage = context.rewriter->getLangOpts();
            gRuntimeBindingLangOptions = &gRuntimeBindingLangOptionsStorage;
        }
        gRuntimeBindingCurrentOwnerFunction = nullptr;
        gRuntimeBindingCurrentDacExprLoc = clang::SourceLocation();
    }

    ~RuntimeBindingContextGuard() {
        gRuntimeBindingCurrentOwnerFunction = nullptr;
        gRuntimeBindingCurrentDacExprLoc = clang::SourceLocation();
        gRuntimeBindingLangOptions = nullptr;
        gRuntimeBindingSourceManager = nullptr;
        gRuntimeBindingTUDecl = nullptr;
    }

    RuntimeBindingContextGuard(const RuntimeBindingContextGuard&) = delete;
    RuntimeBindingContextGuard& operator=(const RuntimeBindingContextGuard&) = delete;
};

struct PointwiseScalarLoopSchedule {
    bool recognized = false;
    bool absorbsOwningWhile = false;
    std::string ownerFunctionName;
    std::string scalarParamName;
    std::string scalarArgName;
    std::string scalarSourceVar;
    std::string scalarInitExpr;
    std::string loopCompareOp;
    std::string loopUpperBoundExpr;
    std::string loopStepExpr;
    std::string outputParamName;
    std::string outputArgName;
    std::string snapshotTensorName;
    std::string snapshotIndexExpr;
    std::string snapshotPrintRowExpr;
};

struct LoopCarriedVectorSchedule {
    bool recognized = false;
    bool absorbsOwningFor = false;
    std::string ownerFunctionName;
    std::string loopIndexVar;
    std::string loopBeginExpr;
    std::string loopUpperBoundExpr;
    std::string loopStepExpr;
    std::string inputParamName;
    std::string inputArgName;
    std::string outputParamName;
    std::string outputArgName;
    std::string writebackIndexVar;
    std::string writebackBeginExpr;
    std::string writebackEndExpr;
    int writebackDstOffset = 0;
    int writebackSrcOffset = -1;
    bool hasLeftBoundaryUpdate = false;
};

static void AppendReferencedScalarConstDecls(const std::string& sourceText,
                                             const std::vector<std::string>& exprs,
                                             std::string& ioPrelude) {
    std::unordered_set<std::string> referencedTokens;
    std::regex identRegex(R"(\b([A-Za-z_]\w*)\b)");
    for (const std::string& expr : exprs) {
        for (std::sregex_iterator it(expr.begin(), expr.end(), identRegex), end;
             it != end; ++it) {
            const std::string token = (*it)[1].str();
            if (kBlockedIdentifierTokens.count(token) > 0) {
                continue;
            }
            referencedTokens.insert(token);
        }
    }

    if (referencedTokens.empty()) {
        return;
    }

    std::vector<std::string> declsToPrepend;
    std::unordered_set<std::string> seenDecls;
    std::regex constDeclRegexTemplate(
        R"((?:^|\n)\s*((?:constexpr\s+)?(?:const\s+)?(?:size_t|int|long|float|double|uint32_t|uint64_t|std::size_t)\s+([A-Za-z_]\w*)\s*=\s*[^;]+;))");

    for (std::sregex_iterator it(sourceText.begin(), sourceText.end(), constDeclRegexTemplate), end;
         it != end; ++it) {
        const std::string decl = TrimCopy((*it)[1].str());
        const std::string token = TrimCopy((*it)[2].str());
        if (decl.empty() || token.empty()) {
            continue;
        }
        if (referencedTokens.count(token) == 0) {
            continue;
        }
        if (ioPrelude.find(decl) != std::string::npos || seenDecls.count(decl) > 0) {
            continue;
        }
        seenDecls.insert(decl);
        declsToPrepend.push_back(decl);
    }

    if (declsToPrepend.empty()) {
        return;
    }

    std::string prepended;
    for (const auto& decl : declsToPrepend) {
        prepended += decl + "\n";
    }
    if (!ioPrelude.empty()) {
        prepended += "\n" + ioPrelude;
    }
    ioPrelude = TrimCopy(prepended);
}

template <typename NodeT>
static std::string GetNodeSourceText(const NodeT* node) {
    if (!node || !gRuntimeBindingSourceManager || !gRuntimeBindingLangOptions) {
        return "";
    }
    clang::CharSourceRange range = clang::CharSourceRange::getTokenRange(node->getSourceRange());
    return clang::Lexer::getSourceText(range, *gRuntimeBindingSourceManager, *gRuntimeBindingLangOptions).str();
}

static std::string GetStmtSourceText(const clang::Stmt* stmt) {
    return GetNodeSourceText(stmt);
}

static std::string GetExprSourceText(const clang::Expr* expr) {
    return GetNodeSourceText(expr);
}

static const clang::FunctionDecl* FindEnclosingFunction(const clang::BinaryOperator* dacExpr) {
    if (!dacExpr || !gRuntimeBindingTUDecl || !gRuntimeBindingSourceManager) {
        return nullptr;
    }

    const clang::SourceManager& sourceManager = *gRuntimeBindingSourceManager;
    clang::SourceLocation exprLoc = sourceManager.getExpansionLoc(dacExpr->getExprLoc());
    if (exprLoc.isInvalid()) {
        return nullptr;
    }

    const clang::FunctionDecl* best = nullptr;
    unsigned bestSpan = std::numeric_limits<unsigned>::max();

    std::function<void(const clang::DeclContext*)> visitDeclContext =
        [&](const clang::DeclContext* declContext) {
            if (!declContext) {
                return;
            }

            for (const clang::Decl* decl : declContext->decls()) {
                if (!decl) {
                    continue;
                }

                if (const auto* functionDecl = clang::dyn_cast<clang::FunctionDecl>(decl)) {
                    if (functionDecl->hasBody()) {
                        const clang::Stmt* body = functionDecl->getBody();
                        if (body) {
                            clang::SourceLocation begin = sourceManager.getExpansionLoc(body->getBeginLoc());
                            clang::SourceLocation end = sourceManager.getExpansionLoc(body->getEndLoc());
                            if (!begin.isInvalid() && !end.isInvalid() &&
                                sourceManager.isWrittenInSameFile(begin, exprLoc) &&
                                sourceManager.isWrittenInSameFile(end, exprLoc) &&
                                !sourceManager.isBeforeInTranslationUnit(exprLoc, begin) &&
                                !sourceManager.isBeforeInTranslationUnit(end, exprLoc)) {
                                unsigned beginOff = sourceManager.getFileOffset(begin);
                                unsigned endOff = sourceManager.getFileOffset(end);
                                if (endOff >= beginOff) {
                                    unsigned span = endOff - beginOff;
                                    if (!best || span < bestSpan) {
                                        best = functionDecl;
                                        bestSpan = span;
                                    }
                                }
                            }
                        }
                    }
                }

                if (const auto* childCtx = clang::dyn_cast<clang::DeclContext>(decl)) {
                    visitDeclContext(childCtx);
                }
            }
        };

    visitDeclContext(gRuntimeBindingTUDecl);
    return best;
}

static std::vector<std::string> ResolveShellCallArgumentsFromAst(const clang::BinaryOperator* dacExpr,
                                                                  Shell* shell) {
    std::vector<std::string> args;
    if (!dacExpr || !shell) {
        return args;
    }

    const clang::Expr* shellSideExpr =
        dacppTranslator::Expression::shellLHS_p(dacExpr) ? dacExpr->getLHS() : dacExpr->getRHS();
    const clang::CallExpr* shellCall =
        dacppTranslator::getNode<clang::CallExpr>(const_cast<clang::Expr*>(shellSideExpr));
    if (!shellCall) {
        return args;
    }

    if (const clang::FunctionDecl* callee = shellCall->getDirectCallee()) {
        if (!shell->getName().empty() && callee->getNameAsString() != shell->getName()) {
            return args;
        }
    }

    for (unsigned i = 0; i < shellCall->getNumArgs(); ++i) {
        const clang::Expr* argExpr = shellCall->getArg(i);
        std::string ident;
        if (TryExtractIdentifierFromExpr(argExpr, ident)) {
            args.push_back(ident);
            continue;
        }

        std::string aliasBase;
        bool aliasIsView = false;
        if (TryExtractAliasedSourceFromExpr(argExpr, aliasBase, aliasIsView)) {
            (void)aliasIsView;
            args.push_back(aliasBase);
            continue;
        }

        std::string rawText = TrimCopy(GetExprSourceText(argExpr));
        if (!rawText.empty()) {
            args.push_back(rawText);
        }
    }

    return args;
}

static bool TryBindSourceCandidateFromAst(
    const clang::Stmt* bindingScopeRoot,
    const std::string& candidate,
    const std::function<std::string(const std::string&)>& materializeScalarExpr,
    RuntimeParamBindingInfo& ioInfo,
    std::unordered_set<std::string>& ioCapturedSourceVars) {
    if (!IsSimpleIdentifierText(candidate)) {
        return false;
    }

    RuntimeBindingResolveResult binding =
        dacppTranslator::ascendc::ResolveRuntimeBindingFromScope(
            MakeRuntimeBindingAstContext(), bindingScopeRoot, candidate);
    const bool hasRoot = binding.found && IsRootRuntimeBindingKind(binding.rootKind);
    const std::string resolvedSource = !binding.rootSourceVar.empty()
        ? binding.rootSourceVar
        : (!binding.sourceVar.empty() ? binding.sourceVar : TrimCopy(candidate));
    ioInfo.bindingResolved = hasRoot;
    ioInfo.bindingKind = hasRoot ? binding.rootKind : RuntimeBindingNodeKind::Unknown;
    if (hasRoot && !resolvedSource.empty()) {
        ioInfo.rootSourceVar = resolvedSource;
    }
    if (!resolvedSource.empty()) {
        ioInfo.hasSourceVar = true;
        ioInfo.sourceVar = resolvedSource;
    }

    if (!binding.bindingOwnerFunction.empty()) {
        ioInfo.bindingOwnerFunction = binding.bindingOwnerFunction;
    }

    if (hasRoot && binding.rootKind == RuntimeBindingNodeKind::StdVector) {
        ioInfo.sourceIsStdVector = true;
        ioInfo.sourceElemType = binding.rootElemType.empty() ? binding.sourceElemType : binding.rootElemType;
        ioInfo.sourceElemIsComplex = binding.sourceElemIsComplex;
        ioInfo.sourceElemIsPair = binding.sourceElemIsPair;
        if (!resolvedSource.empty()) {
            ioCapturedSourceVars.insert(resolvedSource);
        }
    }

    if (binding.hasScalarExpr && !binding.scalarExpr.empty()) {
        ioInfo.hasScalarSourceExpr = true;
        ioInfo.scalarSourceExpr = materializeScalarExpr(binding.scalarExpr);
    }

    const bool allowRuntimeBroadcastFromBinding =
        !ioInfo.sourceIsStdVector || ioInfo.isBroadcastScalar;
    if (allowRuntimeBroadcastFromBinding &&
        binding.isBroadcastScalar && !binding.broadcastExpr.empty()) {
        ioInfo.isBroadcastScalar = true;
        ioInfo.broadcastExpr = materializeScalarExpr(binding.broadcastExpr);
        if (!ioInfo.hasScalarSourceExpr) {
            ioInfo.hasScalarSourceExpr = true;
            ioInfo.scalarSourceExpr = ioInfo.broadcastExpr;
        }
    }

    return ioInfo.sourceIsStdVector || ioInfo.hasScalarSourceExpr;
}

static std::string GetDeclSourceText(const clang::Decl* decl) {
    return GetNodeSourceText(decl);
}

static std::string BuildToKernelCastExpr(const std::string& valueExpr,
                                          const std::string& mappedType,
    bool useFloatBitBridge) {
    if (useFloatBitBridge) {
        return "static_cast<" + mappedType +
               ">(dacpp_host::fp32ToFp16(static_cast<float>(" + valueExpr +
               ")))";
    }
    return "static_cast<" + mappedType + ">(" + valueExpr + ")";
}

static std::string BuildFromKernelCastExpr(const std::string& valueExpr,
                                            const std::string& dstTypeExpr,
    bool useFloatBitBridge) {
    if (useFloatBitBridge) {
        return "static_cast<" + dstTypeExpr +
               ">(dacpp_host::fp16ToFp32(" + valueExpr + "))";
    }
    return "static_cast<" + dstTypeExpr + ">(" + valueExpr + ")";
}

static void WriteOutputFile(const std::string& filename,
                            const std::string& content,
                            const char* description = nullptr) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << content;
        file.close();
    }
    if (description) {
        std::cout << description << filename << std::endl;
    }
}

static std::string BuildGeneratedHostFileName(int index) {
    return "host" + std::to_string(index) + ".cpp";
}

static std::string BuildGeneratedKernelFileName(int index) {
    return "kernel" + std::to_string(index) + ".cpp";
}

static dacppTranslator::ascendc::FlowGeneratedCodeBlock
MakeFlowCodeBlock(dacppTranslator::ascendc::FlowGeneratedCodeRole role,
                  std::string label,
                  std::string code,
                  dacppTranslator::ascendc::FlowExprIndex ownerExpr) {
    dacppTranslator::ascendc::FlowGeneratedCodeBlock blk;
    blk.role = role;
    blk.label = std::move(label);
    blk.ownerExpr = ownerExpr;
    blk.code = std::move(code);
    return blk;
}

static std::string MaterializeScalarExpr(const std::string& sourceText,
                                          const std::string& rawExpr) {
    std::string trimmed = TrimCopy(rawExpr);
    if (trimmed.empty()) {
        return trimmed;
    }
    double numeric = 0.0;
    if (TryResolveSimpleSymbolValue(sourceText, trimmed, numeric) && std::isfinite(numeric)) {
        std::ostringstream oss;
        oss << std::setprecision(12) << numeric;
        return oss.str();
    }
    return trimmed;
}

static void CollectRecordDeclsFromQualType(clang::QualType qt,
                                           std::unordered_set<const clang::RecordDecl*>& outDecls) {
    qt = qt.getCanonicalType();

    if (const auto* refType = qt->getAs<clang::ReferenceType>()) {
        CollectRecordDeclsFromQualType(refType->getPointeeType(), outDecls);
        return;
    }
    if (const auto* ptrType = qt->getAs<clang::PointerType>()) {
        CollectRecordDeclsFromQualType(ptrType->getPointeeType(), outDecls);
        return;
    }
    if (const auto* recordType = qt->getAs<clang::RecordType>()) {
        const clang::RecordDecl* recordDecl = recordType->getDecl();
        if (recordDecl) {
            if (const clang::RecordDecl* def = recordDecl->getDefinition()) {
                if (const auto* canonical = llvm::dyn_cast<clang::RecordDecl>(def->getCanonicalDecl())) {
                    outDecls.insert(canonical);
                }
            } else {
                if (const auto* canonical = llvm::dyn_cast<clang::RecordDecl>(recordDecl->getCanonicalDecl())) {
                    outDecls.insert(canonical);
                }
            }
            if (const auto* specializationDecl =
                    llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(recordDecl)) {
                for (const clang::TemplateArgument& arg : specializationDecl->getTemplateArgs().asArray()) {
                    if (arg.getKind() == clang::TemplateArgument::Type) {
                        CollectRecordDeclsFromQualType(arg.getAsType(), outDecls);
                    }
                }
            }
        }
        return;
    }
    if (const auto* specType = qt->getAs<clang::TemplateSpecializationType>()) {
        for (const clang::TemplateArgument& arg : specType->template_arguments()) {
            if (arg.getKind() == clang::TemplateArgument::Type) {
                CollectRecordDeclsFromQualType(arg.getAsType(), outDecls);
            }
        }
    }
}

static bool IsBeforeCutoffInTranslationUnit(const clang::SourceManager& sm,
                                            clang::SourceLocation probe,
                                            clang::SourceLocation cutoff) {
    clang::SourceLocation probeLoc = sm.getExpansionLoc(probe);
    clang::SourceLocation cutoffLoc = sm.getExpansionLoc(cutoff);
    if (probeLoc.isInvalid() || cutoffLoc.isInvalid()) {
        return false;
    }
    return sm.isBeforeInTranslationUnit(probeLoc, cutoffLoc);
}

class TrackedVarMentionVisitor
    : public clang::RecursiveASTVisitor<TrackedVarMentionVisitor> {
public:
    explicit TrackedVarMentionVisitor(const std::unordered_set<std::string>& tracked)
        : trackedVars(tracked) {}

    bool VisitDeclRefExpr(clang::DeclRefExpr* refExpr) {
        if (!refExpr || !refExpr->getDecl()) {
            return true;
        }
        if (trackedVars.count(refExpr->getDecl()->getNameAsString()) > 0) {
            mentioned = true;
            return false;
        }
        return true;
    }

    bool Mentioned() const {
        return mentioned;
    }

private:
    const std::unordered_set<std::string>& trackedVars;
    bool mentioned = false;
};

class TrackedVarWriteVisitor
    : public clang::RecursiveASTVisitor<TrackedVarWriteVisitor> {
public:
    explicit TrackedVarWriteVisitor(const std::unordered_set<std::string>& tracked)
        : trackedVars(tracked) {}

    bool VisitBinaryOperator(clang::BinaryOperator* binOp) {
        if (!binOp || !binOp->isAssignmentOp()) {
            return true;
        }
        std::string lhsBase;
        if (TryExtractIdentifierFromExpr(binOp->getLHS(), lhsBase) &&
            trackedVars.count(lhsBase) > 0) {
            wroteTrackedVar = true;
            return false;
        }
        return true;
    }

    bool WroteTrackedVar() const {
        return wroteTrackedVar;
    }

private:
    const std::unordered_set<std::string>& trackedVars;
    bool wroteTrackedVar = false;
};

static bool ExprMentionsTrackedVars(const clang::Expr* expr,
                                    const std::unordered_set<std::string>& trackedVars) {
    if (!expr || trackedVars.empty()) {
        return false;
    }
    TrackedVarMentionVisitor visitor(trackedVars);
    visitor.TraverseStmt(const_cast<clang::Expr*>(expr));
    return visitor.Mentioned();
}

static bool StmtMentionsTrackedVars(const clang::Stmt* stmt,
                                    const std::unordered_set<std::string>& trackedVars) {
    if (!stmt || trackedVars.empty()) {
        return false;
    }
    TrackedVarMentionVisitor visitor(trackedVars);
    visitor.TraverseStmt(const_cast<clang::Stmt*>(stmt));
    return visitor.Mentioned();
}

static bool StmtWritesTrackedVars(const clang::Stmt* stmt,
                                  const std::unordered_set<std::string>& trackedVars) {
    if (!stmt || trackedVars.empty()) {
        return false;
    }
    TrackedVarWriteVisitor visitor(trackedVars);
    visitor.TraverseStmt(const_cast<clang::Stmt*>(stmt));
    return visitor.WroteTrackedVar();
}

static std::unordered_set<std::string> ExtractIdentifierTokensFromTexts(
    const std::vector<std::string>& texts) {
    std::unordered_set<std::string> tokens;
    std::regex identRegex(R"(\b([A-Za-z_]\w*)\b)");
    for (const std::string& text : texts) {
        for (std::sregex_iterator it(text.begin(), text.end(), identRegex), end;
             it != end; ++it) {
            const std::string token = (*it)[1].str();
            if (kBlockedIdentifierTokens.count(token) > 0) {
                continue;
            }
            tokens.insert(token);
        }
    }
    return tokens;
}

static void PrependTextuallyReferencedLocalConstDecls(
    const clang::CompoundStmt* body,
    const clang::SourceManager& sm,
    clang::SourceLocation cutoffLoc,
    std::vector<std::string>& ioMainInitStatements) {
    if (!body) {
        return;
    }

    std::unordered_set<std::string> seenStatements(ioMainInitStatements.begin(),
                                                   ioMainInitStatements.end());

    bool changed = false;
    do {
        changed = false;
        std::unordered_set<std::string> referencedTokens =
            ExtractIdentifierTokensFromTexts(ioMainInitStatements);
        if (referencedTokens.empty()) {
            break;
        }

        std::vector<std::string> prependedStatements;
        for (const clang::Stmt* stmt : body->body()) {
            if (!stmt || !IsBeforeCutoffInTranslationUnit(sm, stmt->getBeginLoc(), cutoffLoc)) {
                continue;
            }

            const auto* declStmt = llvm::dyn_cast<clang::DeclStmt>(stmt);
            if (!declStmt) {
                continue;
            }

            bool includeStmt = false;
            for (const clang::Decl* decl : declStmt->decls()) {
                const auto* varDecl = llvm::dyn_cast<clang::VarDecl>(decl);
                if (!varDecl || !varDecl->getIdentifier() || !varDecl->isLocalVarDecl()) {
                    continue;
                }
                const bool isConstLike =
                    varDecl->isConstexpr() || varDecl->getType().isConstQualified();
                if (!isConstLike) {
                    continue;
                }
                if (referencedTokens.count(varDecl->getNameAsString()) == 0) {
                    continue;
                }
                includeStmt = true;
                break;
            }
            if (!includeStmt) {
                continue;
            }

            std::string stmtText = TrimCopy(GetStmtSourceText(stmt));
            if (stmtText.empty()) {
                continue;
            }
            if (stmtText.back() != ';') {
                stmtText += ";";
            }
            if (seenStatements.insert(stmtText).second) {
                prependedStatements.push_back(stmtText);
            }
        }

        if (!prependedStatements.empty()) {
            std::vector<std::string> mergedStatements;
            mergedStatements.reserve(prependedStatements.size() + ioMainInitStatements.size());
            mergedStatements.insert(mergedStatements.end(),
                                    prependedStatements.begin(),
                                    prependedStatements.end());
            mergedStatements.insert(mergedStatements.end(),
                                    ioMainInitStatements.begin(),
                                    ioMainInitStatements.end());
            ioMainInitStatements.swap(mergedStatements);
            changed = true;
        }
    } while (changed);
}

static void CollectTextuallyReferencedPreludeDecls(
    const clang::TranslationUnitDecl* tuDecl,
    const clang::SourceManager& sm,
    clang::SourceLocation cutoffLoc,
    clang::SourceLocation ownerLoc,
    const std::vector<std::string>& referencedTexts,
    std::vector<std::string>& ioConstDecls,
    std::vector<std::string>& ioUsingDecls,
    std::vector<std::string>& ioRecordDecls,
    std::vector<std::string>* ioGlobalVarDecls = nullptr) {
    if (!tuDecl) {
        return;
    }

    std::unordered_set<std::string> referencedTokens =
        ExtractIdentifierTokensFromTexts(referencedTexts);
    if (referencedTokens.empty()) {
        return;
    }

    std::unordered_set<std::string> seenConstDecls(ioConstDecls.begin(), ioConstDecls.end());
    std::unordered_set<std::string> seenGlobalDecls;
    if (ioGlobalVarDecls) {
        seenGlobalDecls.insert(ioGlobalVarDecls->begin(), ioGlobalVarDecls->end());
    }
    std::unordered_set<std::string> seenPreludeVarNames;
    auto SeedSeenPreludeVarNames = [&](const std::vector<std::string>& decls) {
        for (const std::string& declText : decls) {
            std::vector<std::string> texts{declText};
            std::unordered_set<std::string> tokens = ExtractIdentifierTokensFromTexts(texts);
            for (const std::string& token : tokens) {
                if (referencedTokens.count(token) > 0) {
                    seenPreludeVarNames.insert(token);
                }
            }
        }
    };
    SeedSeenPreludeVarNames(ioConstDecls);
    if (ioGlobalVarDecls) {
        SeedSeenPreludeVarNames(*ioGlobalVarDecls);
    }
    std::unordered_set<std::string> seenUsingDecls(ioUsingDecls.begin(), ioUsingDecls.end());
    std::unordered_set<std::string> seenRecordDecls(ioRecordDecls.begin(), ioRecordDecls.end());

    auto TryCollectDecl = [](const clang::Decl* d, const char* rejectSubstr,
                             std::unordered_set<std::string>& seen,
                             std::vector<std::string>& out) -> bool {
        std::string text = TrimCopy(GetDeclSourceText(d));
        if (text.empty() || text.find(rejectSubstr) != std::string::npos) return false;
        if (text.back() != ';') text += ";";
        if (!seen.insert(text).second) return false;
        out.push_back(text);
        return true;
    };

    for (const clang::Decl* decl : tuDecl->decls()) {
        if (!decl) continue;
        if (!IsBeforeCutoffInTranslationUnit(sm, decl->getBeginLoc(), cutoffLoc)) continue;
        if (sm.isInSystemHeader(decl->getBeginLoc()) ||
            !sm.isWrittenInSameFile(sm.getExpansionLoc(decl->getBeginLoc()), ownerLoc)) {
            continue;
        }

        if (const auto* varDecl = llvm::dyn_cast<clang::VarDecl>(decl)) {
            if (!varDecl->getIdentifier() || varDecl->isLocalVarDecl()) continue;
            const std::string varName = varDecl->getNameAsString();
            if (referencedTokens.count(varName) == 0 ||
                seenPreludeVarNames.count(varName) > 0) {
                continue;
            }

            auto MarkCoveredVarNames = [&](const std::string& text) {
                std::vector<std::string> texts{text};
                std::unordered_set<std::string> tokens = ExtractIdentifierTokensFromTexts(texts);
                for (const std::string& token : tokens) {
                    if (referencedTokens.count(token) > 0) {
                        seenPreludeVarNames.insert(token);
                    }
                }
            };
            const bool isConstLike =
                varDecl->isConstexpr() || varDecl->getType().isConstQualified();
            if (isConstLike) {
                if (TryCollectDecl(varDecl, "dacpp::", seenConstDecls, ioConstDecls))
                    MarkCoveredVarNames(ioConstDecls.back());
            } else if (ioGlobalVarDecls &&
                       TryCollectDecl(varDecl, "dacpp::", seenGlobalDecls, *ioGlobalVarDecls)) {
                MarkCoveredVarNames(ioGlobalVarDecls->back());
            }
            continue;
        }

        if (const auto* typeAlias = llvm::dyn_cast<clang::TypeAliasDecl>(decl)) {
            if (typeAlias->getIdentifier() &&
                referencedTokens.count(typeAlias->getNameAsString()) > 0)
                TryCollectDecl(typeAlias, "dacpp", seenUsingDecls, ioUsingDecls);
            continue;
        }

        if (const auto* usingDecl = llvm::dyn_cast<clang::UsingDecl>(decl)) {
            if (usingDecl->getIdentifier() &&
                referencedTokens.count(usingDecl->getNameAsString()) > 0)
                TryCollectDecl(usingDecl, "dacpp", seenUsingDecls, ioUsingDecls);
            continue;
        }

        if (llvm::isa<clang::UsingDirectiveDecl>(decl)) {
            TryCollectDecl(decl, "dacpp", seenUsingDecls, ioUsingDecls);
            continue;
        }

        const auto* recordDecl = llvm::dyn_cast<clang::RecordDecl>(decl);
        if (!recordDecl || !recordDecl->getIdentifier() ||
            referencedTokens.count(recordDecl->getNameAsString()) == 0) {
            continue;
        }
        const clang::RecordDecl* def = recordDecl->getDefinition();
        TryCollectDecl(def ? def : recordDecl, "dacpp", seenRecordDecls, ioRecordDecls);
    }
}

static bool IsHostSideHelperFunction(const clang::FunctionDecl* fnDecl) {
    if (!fnDecl || !fnDecl->hasBody() || fnDecl->isImplicit()) {
        return false;
    }
    if (llvm::isa<clang::CXXMethodDecl>(fnDecl)) {
        return false;
    }
    const std::string fnName = fnDecl->getNameAsString();
    if (fnName.empty() || fnName == "main") {
        return false;
    }
    if (fnName.find("shell") != std::string::npos ||
        fnName.find("calc") != std::string::npos) {
        return false;
    }
    return true;
}

static void CollectHelperCallsFromStmt(
    const clang::Stmt* stmt,
    std::vector<const clang::FunctionDecl*>& ioHelperFns,
    std::unordered_set<const clang::FunctionDecl*>& ioSeenFns,
    const std::unordered_set<std::string>* trackedVars = nullptr) {
    if (!stmt) return;
    if (trackedVars && trackedVars->empty()) return;

    class HelperCallCollector : public clang::RecursiveASTVisitor<HelperCallCollector> {
    public:
        HelperCallCollector(std::vector<const clang::FunctionDecl*>& helperFns,
                            std::unordered_set<const clang::FunctionDecl*>& seenFns,
                            const std::unordered_set<std::string>* tracked)
            : collectedFns(helperFns), seenFunctionDecls(seenFns), trackedVars(tracked) {}

        bool VisitCallExpr(clang::CallExpr* callExpr) {
            if (!callExpr) return true;
            const clang::FunctionDecl* callee = callExpr->getDirectCallee();
            if (!IsHostSideHelperFunction(callee)) return true;

            if (trackedVars) {
                bool related = false;
                for (unsigned i = 0; i < callExpr->getNumArgs(); ++i) {
                    if (ExprMentionsTrackedVars(callExpr->getArg(i), *trackedVars)) {
                        related = true;
                        break;
                    }
                }
                if (!related) {
                    if (const auto* mc = clang::dyn_cast<clang::CXXMemberCallExpr>(callExpr)) {
                        related = ExprMentionsTrackedVars(
                            mc->getImplicitObjectArgument(), *trackedVars);
                    }
                }
                if (!related) return true;
            }

            const clang::FunctionDecl* canonical = callee->getCanonicalDecl();
            if (canonical && seenFunctionDecls.insert(canonical).second) {
                collectedFns.push_back(canonical);
            }
            return true;
        }

    private:
        std::vector<const clang::FunctionDecl*>& collectedFns;
        std::unordered_set<const clang::FunctionDecl*>& seenFunctionDecls;
        const std::unordered_set<std::string>* trackedVars;
    };

    HelperCallCollector collector(ioHelperFns, ioSeenFns, trackedVars);
    collector.TraverseStmt(const_cast<clang::Stmt*>(stmt));
}

static const clang::FunctionDecl* ResolveHostPreludeFunctionDef(
    const clang::FunctionDecl* fnDecl) {
    if (!fnDecl) {
        return nullptr;
    }
    if (fnDecl->hasBody()) {
        return fnDecl;
    }
    if (const clang::FunctionDecl* def = fnDecl->getDefinition()) {
        return def;
    }
    return fnDecl->getCanonicalDecl();
}

static bool IsCollectibleHostPreludeFunction(
    const clang::FunctionDecl* fnDecl,
    const clang::SourceManager& sm,
    clang::SourceLocation ownerLoc) {
    const clang::FunctionDecl* def = ResolveHostPreludeFunctionDef(fnDecl);
    if (!IsHostSideHelperFunction(def) || !def) {
        return false;
    }
    if (sm.isInSystemHeader(def->getBeginLoc())) {
        return false;
    }
    return sm.isWrittenInSameFile(sm.getExpansionLoc(def->getBeginLoc()), ownerLoc);
}

static bool FunctionMentionsAnyReferencedToken(
    const clang::FunctionDecl* fnDecl,
    const std::unordered_set<std::string>& referencedTokens) {
    if (!fnDecl || referencedTokens.empty()) {
        return false;
    }

    const std::string fnName = fnDecl->getNameAsString();
    if (!fnName.empty() && referencedTokens.count(fnName) > 0) {
        return true;
    }

    if (!(fnDecl->isOverloadedOperator() &&
          fnDecl->getOverloadedOperator() == clang::OverloadedOperatorKind::OO_LessLess)) {
        return false;
    }

    for (unsigned i = 0; i < fnDecl->getNumParams(); ++i) {
        const clang::ParmVarDecl* paramDecl = fnDecl->getParamDecl(i);
        if (!paramDecl) {
            continue;
        }
        std::vector<std::string> paramTypeTexts{
            paramDecl->getType().getAsString(),
            paramDecl->getType().getCanonicalType().getAsString()};
        std::unordered_set<std::string> paramTokens =
            ExtractIdentifierTokensFromTexts(paramTypeTexts);
        for (const std::string& token : paramTokens) {
            if (referencedTokens.count(token) > 0) {
                return true;
            }
        }
    }

    return false;
}

static void CollectTextuallyReferencedPreludeFunctionRoots(
    const clang::TranslationUnitDecl* tuDecl,
    const clang::SourceManager& sm,
    clang::SourceLocation ownerLoc,
    const std::vector<std::string>& referencedTexts,
    std::vector<const clang::FunctionDecl*>& ioRootFns,
    std::unordered_set<const clang::FunctionDecl*>& ioSeenFns) {
    if (!tuDecl) {
        return;
    }

    const std::unordered_set<std::string> referencedTokens =
        ExtractIdentifierTokensFromTexts(referencedTexts);
    if (referencedTokens.empty()) {
        return;
    }

    for (const clang::Decl* decl : tuDecl->decls()) {
        const auto* fnDecl = llvm::dyn_cast_or_null<clang::FunctionDecl>(decl);
        if (!fnDecl) {
            continue;
        }

        const clang::FunctionDecl* def = ResolveHostPreludeFunctionDef(fnDecl);
        if (!IsCollectibleHostPreludeFunction(def, sm, ownerLoc) ||
            !FunctionMentionsAnyReferencedToken(def, referencedTokens)) {
            continue;
        }

        const clang::FunctionDecl* canonical = def->getCanonicalDecl();
        if (canonical && ioSeenFns.insert(canonical).second) {
            ioRootFns.push_back(def);
        }
    }
}

static void CollectTransitiveHostPreludeFunctions(
    const std::vector<const clang::FunctionDecl*>& rootFns,
    const clang::SourceManager& sm,
    clang::SourceLocation ownerLoc,
    std::vector<const clang::FunctionDecl*>& outOrderedFns) {
    outOrderedFns.clear();
    std::unordered_set<const clang::FunctionDecl*> emittedFns;
    std::unordered_set<const clang::FunctionDecl*> visitingFns;

    std::function<void(const clang::FunctionDecl*)> VisitFn =
        [&](const clang::FunctionDecl* fnDecl) {
            const clang::FunctionDecl* def = ResolveHostPreludeFunctionDef(fnDecl);
            if (!IsCollectibleHostPreludeFunction(def, sm, ownerLoc) || !def) {
                return;
            }

            const clang::FunctionDecl* canonical = def->getCanonicalDecl();
            if (!canonical || emittedFns.count(canonical) > 0) {
                return;
            }
            if (!visitingFns.insert(canonical).second) {
                return;
            }

            if (def->hasBody()) {
                std::vector<const clang::FunctionDecl*> nestedFns;
                std::unordered_set<const clang::FunctionDecl*> seenNestedFns;
                CollectHelperCallsFromStmt(def->getBody(), nestedFns, seenNestedFns);
                for (const clang::FunctionDecl* nestedFn : nestedFns) {
                    const clang::FunctionDecl* nestedDef =
                        ResolveHostPreludeFunctionDef(nestedFn);
                    if (nestedDef && nestedDef->getCanonicalDecl() != canonical) {
                        VisitFn(nestedDef);
                    }
                }
            }

            visitingFns.erase(canonical);
            if (emittedFns.insert(canonical).second) {
                outOrderedFns.push_back(def);
            }
        };

    for (const clang::FunctionDecl* rootFn : rootFns) {
        VisitFn(rootFn);
    }
}

static void AppendPreludeFunctionDefsFromRoots(
    const std::vector<const clang::FunctionDecl*>& rootFns,
    const clang::SourceManager& sm,
    clang::SourceLocation ownerLoc,
    std::vector<std::string>& ioFunctionDefs) {
    std::vector<const clang::FunctionDecl*> orderedFns;
    CollectTransitiveHostPreludeFunctions(rootFns, sm, ownerLoc, orderedFns);

    std::unordered_set<std::string> seenFunctionDefs(
        ioFunctionDefs.begin(), ioFunctionDefs.end());
    for (const clang::FunctionDecl* fnDecl : orderedFns) {
        std::string defText = TrimCopy(GetDeclSourceText(fnDecl));
        if (!defText.empty() && seenFunctionDefs.insert(defText).second) {
            ioFunctionDefs.push_back(defText);
        }
    }
}

static bool NeedsStmtTerminatingSemicolon(const clang::Stmt* stmt) {
    return stmt &&
           (llvm::isa<clang::DeclStmt>(stmt) ||
            llvm::isa<clang::Expr>(stmt) ||
            llvm::isa<clang::NullStmt>(stmt) ||
            llvm::isa<clang::ReturnStmt>(stmt) ||
            llvm::isa<clang::BreakStmt>(stmt) ||
            llvm::isa<clang::ContinueStmt>(stmt) ||
            llvm::isa<clang::GotoStmt>(stmt));
}

static std::string NormalizeStmtSourceText(const clang::Stmt* stmt) {
    std::string stmtText = TrimCopy(GetStmtSourceText(stmt));
    if (stmtText.empty()) {
        return stmtText;
    }
    if (NeedsStmtTerminatingSemicolon(stmt) && stmtText.back() != ';') {
        stmtText += ";";
    }
    return stmtText;
}

static bool StmtContainsDacOperatorToken(const clang::Stmt* stmt) {
    const std::string text = NormalizeStmtSourceText(stmt);
    return !text.empty() && text.find("<->") != std::string::npos;
}


// ===== Host prelude collection =====

static bool LooksLikeHostSimpleOutputStmtText(const std::string& stmtText) {
    if (stmtText.empty()) {
        return false;
    }
    return stmtText.find("std::cout") != std::string::npos ||
           stmtText.find("cout <<") != std::string::npos ||
           stmtText.find("printf(") != std::string::npos;
}

static bool LooksLikeHostTensorPrintStmtText(const std::string& stmtText) {
    if (stmtText.empty()) {
        return false;
    }
    return stmtText.find(".print(") != std::string::npos ||
           stmtText.find(" print(") != std::string::npos;
}

static bool LooksLikeHostOutputStmtText(const std::string& stmtText) {
    return LooksLikeHostSimpleOutputStmtText(stmtText) ||
           LooksLikeHostTensorPrintStmtText(stmtText);
}

static bool StmtCallsAnyHostHelper(const clang::Stmt* stmt) {
    std::vector<const clang::FunctionDecl*> helperFns;
    std::unordered_set<const clang::FunctionDecl*> seenHelperFns;
    CollectHelperCallsFromStmt(stmt, helperFns, seenHelperFns);
    return !helperFns.empty();
}

// 取默认 resolver context 的便捷适配器：上层 alias resolution 走
// ascendc::CollectLiveTensorAliasesInFunction / TryResolveHostTensorAliasFromText 等 AST 入口。
static void CollectLiveTensorAliasesInFunction(
    const clang::FunctionDecl* ownerFn,
    std::unordered_map<std::string, HostTensorAliasInfo>& ioAliases) {
    dacppTranslator::ascendc::CollectLiveTensorAliasesInFunction(
        ownerFn,
        MakeTensorAliasResolverContext(),
        ioAliases);
}

// 把 rank-0 的 host alias 还原成 `(root.data())[offset]` 形式的标量读。
static std::string BuildHostTensorScalarReadExpr(const HostTensorAliasInfo& aliasInfo) {
    if (!aliasInfo.valid || aliasInfo.rank != 0 || aliasInfo.rootDataExpr.empty()) {
        return "";
    }
    return "(" + aliasInfo.rootDataExpr + ")[static_cast<long long>(" +
           aliasInfo.offsetExpr + ")]";
}

static std::string RewriteTensorScalarReadsInFreeformText(
    const std::string& rawText,
    llvm::StringRef ownerFunctionName,
    const std::unordered_map<std::string, HostTensorAliasInfo>& aliasMap) {
    auto IsIdentStart = [](char c) {
        return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
    };
    auto IsIdentChar = [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
    };

    std::string rewritten;
    rewritten.reserve(rawText.size() + 32);
    size_t i = 0;
    while (i < rawText.size()) {
        if (!IsIdentStart(rawText[i])) {
            rewritten.push_back(rawText[i]);
            ++i;
            continue;
        }

        const size_t identBegin = i;
        size_t identEnd = i + 1;
        while (identEnd < rawText.size() && IsIdentChar(rawText[identEnd])) {
            ++identEnd;
        }

        size_t candidateEnd = identEnd;
        size_t cursor = identEnd;
        while (cursor < rawText.size()) {
            size_t next = cursor;
            while (next < rawText.size() &&
                   std::isspace(static_cast<unsigned char>(rawText[next]))) {
                ++next;
            }
            if (next >= rawText.size() || rawText[next] != '[') {
                break;
            }

            int depth = 0;
            size_t bracketEnd = next;
            for (; bracketEnd < rawText.size(); ++bracketEnd) {
                if (rawText[bracketEnd] == '[') {
                    ++depth;
                } else if (rawText[bracketEnd] == ']') {
                    --depth;
                    if (depth == 0) {
                        ++bracketEnd;
                        break;
                    }
                }
            }
            if (depth != 0) {
                break;
            }

            candidateEnd = bracketEnd;
            cursor = bracketEnd;
        }

        HostTensorAliasInfo aliasInfo;
        const std::string candidateText =
            TrimCopy(rawText.substr(identBegin, candidateEnd - identBegin));
        if (!candidateText.empty() &&
            TryResolveHostTensorAliasFromText(
                candidateText,
                ownerFunctionName,
                aliasMap,
                aliasInfo) &&
            aliasInfo.valid &&
            aliasInfo.rank == 0) {
            rewritten += BuildHostTensorScalarReadExpr(aliasInfo);
            i = candidateEnd;
            continue;
        }

        rewritten.append(rawText, identBegin, identEnd - identBegin);
        i = identEnd;
    }
    return rewritten;
}

static void RewriteTensorScalarReadsInAliasInfo(
    HostTensorAliasInfo& ioInfo,
    llvm::StringRef ownerFunctionName,
    const std::unordered_map<std::string, HostTensorAliasInfo>& aliasMap) {
    if (!ioInfo.valid) {
        return;
    }

    ioInfo.offsetExpr = RewriteTensorScalarReadsInFreeformText(
        ioInfo.offsetExpr,
        ownerFunctionName,
        aliasMap);
    for (std::string& shapeExpr : ioInfo.shapeExprs) {
        shapeExpr = RewriteTensorScalarReadsInFreeformText(
            shapeExpr,
            ownerFunctionName,
            aliasMap);
    }
    for (std::string& strideExpr : ioInfo.strideExprs) {
        strideExpr = RewriteTensorScalarReadsInFreeformText(
            strideExpr,
            ownerFunctionName,
            aliasMap);
    }
}

static std::string DriverBuildHostTensorIndexedAccessExpr(
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

static std::string DriverMaybeCastTensorAssignmentRhs(
    const std::string& rhsExpr,
    const HostTensorAliasInfo& lhsInfo,
    const HostTensorAliasInfo* rhsInfo) {
    const std::string trimmed = TrimCopy(rhsExpr);
    if (trimmed.empty()) {
        return "";
    }

    if (lhsInfo.elementType.empty()) {
        return trimmed;
    }
    if (rhsInfo && !rhsInfo->elementType.empty() &&
        lhsInfo.elementType == rhsInfo->elementType) {
        return trimmed;
    }
    return "static_cast<" + lhsInfo.elementType + ">(" + trimmed + ")";
}

static std::string EmitTensorAliasAssignmentFromInfos(
    const HostTensorAliasInfo& lhsInfo,
    const HostTensorAliasInfo* rhsInfo,
    const std::string& rhsScalarExpr,
    size_t uniqueIndex) {
    if (!lhsInfo.valid || lhsInfo.rootDataExpr.empty()) {
        return "";
    }

    const bool rhsIsAlias = rhsInfo && rhsInfo->valid;
    if (lhsInfo.rank == 0) {
        const std::string lhsScalarExpr = BuildHostTensorScalarReadExpr(lhsInfo);
        std::string rhsExpr;
        if (rhsIsAlias) {
            if (rhsInfo->rank != 0) {
                return "";
            }
            rhsExpr = BuildHostTensorScalarReadExpr(*rhsInfo);
        } else {
            rhsExpr = TrimCopy(rhsScalarExpr);
        }
        rhsExpr = DriverMaybeCastTensorAssignmentRhs(
            rhsExpr,
            lhsInfo,
            rhsIsAlias ? rhsInfo : nullptr);
        if (lhsScalarExpr.empty() || rhsExpr.empty()) {
            return "";
        }
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
                rhsInfo->shapeExprs[static_cast<size_t>(dim)] +
                "))";
        }

        code += "    const long long " + extentVar + " = " + extentExpr + ";\n";
        extentVars.push_back(extentVar);
        loopIdxVars.push_back(idxVar);
    }

    std::function<void(size_t, const std::string&)> appendLoopNest =
        [&](size_t dim, const std::string& indent) {
            if (dim >= loopIdxVars.size()) {
                const std::string lhsAccessExpr =
                    DriverBuildHostTensorIndexedAccessExpr(lhsInfo, loopIdxVars);
                std::string rhsExpr;
                if (rhsIsAlias) {
                    rhsExpr = rhsInfo->rank == 0
                        ? BuildHostTensorScalarReadExpr(*rhsInfo)
                        : DriverBuildHostTensorIndexedAccessExpr(*rhsInfo, loopIdxVars);
                } else {
                    rhsExpr = TrimCopy(rhsScalarExpr);
                }
                rhsExpr = DriverMaybeCastTensorAssignmentRhs(
                    rhsExpr,
                    lhsInfo,
                    rhsIsAlias ? rhsInfo : nullptr);
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

static const clang::Stmt* FindOwningStmtForDacExpr(const clang::FunctionDecl* ownerFn,
                                                   clang::SourceLocation anchorLoc) {
    if (!ownerFn || !ownerFn->hasBody() || !gRuntimeBindingSourceManager) {
        return nullptr;
    }
    const auto* body = llvm::dyn_cast<clang::CompoundStmt>(ownerFn->getBody());
    if (!body) {
        return nullptr;
    }

    const clang::SourceManager& sm = *gRuntimeBindingSourceManager;
    for (const clang::Stmt* stmt : body->body()) {
        if (stmt && IsStmtContainingLoc(stmt, sm, anchorLoc)) {
            return stmt;
        }
    }
    return nullptr;
}

static void TrackPreludeDerivedVarsFromStmt(
    const clang::Stmt* stmt,
    std::unordered_set<std::string>& ioTrackedVars,
    std::unordered_set<std::string>& ioResolvedRoots,
    std::unordered_set<const clang::RecordDecl*>& ioReferencedRecordDecls,
    bool forceTrackDeclaredVars) {
    if (!stmt) {
        return;
    }

    if (const auto* declStmt = llvm::dyn_cast<clang::DeclStmt>(stmt)) {
        for (const clang::Decl* decl : declStmt->decls()) {
            const auto* varDecl = llvm::dyn_cast<clang::VarDecl>(decl);
            if (!varDecl || !varDecl->getIdentifier()) {
                continue;
            }

            const std::string varName = varDecl->getNameAsString();
            const bool nameTracked = ioTrackedVars.count(varName) > 0;
            if (nameTracked) {
                ioResolvedRoots.insert(varName);
            }

            CollectRecordDeclsFromQualType(varDecl->getType(), ioReferencedRecordDecls);

            std::string typeName = varDecl->getType().getAsString();
            std::string canonicalTypeName = varDecl->getType().getCanonicalType().getAsString();
            const bool isWrapper =
                LooksLikeTensorWrapperTypeName(typeName) ||
                LooksLikeTensorWrapperTypeName(canonicalTypeName);
            const bool initMentionsTracked =
                varDecl->hasInit() && ExprMentionsTrackedVars(varDecl->getInit(), ioTrackedVars);
            if (!isWrapper && (forceTrackDeclaredVars || nameTracked || initMentionsTracked)) {
                ioTrackedVars.insert(varName);
            }
        }
    }

    const clang::Expr* exprStmt = llvm::dyn_cast<clang::Expr>(stmt);
    if (!exprStmt) {
        return;
    }
    exprStmt = exprStmt->IgnoreParenImpCasts();
    const auto* assignOp = llvm::dyn_cast<clang::BinaryOperator>(exprStmt);
    if (!assignOp || !assignOp->isAssignmentOp()) {
        return;
    }

    std::string lhsName;
    if (!TryExtractIdentifierFromExpr(assignOp->getLHS(), lhsName) || lhsName.empty()) {
        return;
    }
    if (forceTrackDeclaredVars || ExprMentionsTrackedVars(assignOp->getRHS(), ioTrackedVars)) {
        ioTrackedVars.insert(lhsName);
    }
}

static bool ShouldIncludeStmtInPrelude(const clang::Stmt* stmt,
                                      std::unordered_set<std::string>& ioTrackedVars,
                                      std::unordered_set<std::string>& ioResolvedRoots,
                                      std::unordered_set<const clang::RecordDecl*>& ioReferencedRecordDecls) {
    if (!stmt) {
        return false;
    }

    const std::string stmtText = NormalizeStmtSourceText(stmt);
    if (LooksLikeHostOutputStmtText(stmtText) ||
        StmtCallsAnyHostHelper(stmt)) {
        return true;
    }

    bool includeStmt = false;
    if (const auto* declStmt = llvm::dyn_cast<clang::DeclStmt>(stmt)) {
        for (const clang::Decl* decl : declStmt->decls()) {
            const auto* varDecl = llvm::dyn_cast<clang::VarDecl>(decl);
            if (!varDecl || !varDecl->getIdentifier()) {
                continue;
            }
            const std::string varName = varDecl->getNameAsString();
            std::string typeName = varDecl->getType().getAsString();
            std::string canonicalTypeName = varDecl->getType().getCanonicalType().getAsString();
            const bool isWrapper =
                LooksLikeTensorWrapperTypeName(typeName) ||
                LooksLikeTensorWrapperTypeName(canonicalTypeName);
            const bool nameTracked = ioTrackedVars.count(varName) > 0;
            const bool initMentionsTracked =
                varDecl->hasInit() && ExprMentionsTrackedVars(varDecl->getInit(), ioTrackedVars);

            if (nameTracked) {
                ioResolvedRoots.insert(varName);
            }
            if ((nameTracked || initMentionsTracked) && !isWrapper) {
                includeStmt = true;
            }
            CollectRecordDeclsFromQualType(varDecl->getType(), ioReferencedRecordDecls);
        }
    } else if (StmtWritesTrackedVars(stmt, ioTrackedVars) ||
               StmtMentionsTrackedVars(stmt, ioTrackedVars)) {
        includeStmt = true;
    }

    if (includeStmt) {
        TrackPreludeDerivedVarsFromStmt(
            stmt,
            ioTrackedVars,
            ioResolvedRoots,
            ioReferencedRecordDecls,
            false);
    }
    return includeStmt;
}

// 从 owner function 的 body 抽取 host prelude 上下文
// （helper function defs / const decls / global var decls / using decls /
//  record decls）。这些文本不参与 host main body 发射。
static void CollectHostPreludeFromAst(
    const clang::FunctionDecl* ownerFn,
    clang::SourceLocation cutoffLoc,
    const std::unordered_set<std::string>& rootSourceVars,
    HostPreludeCollection& out) {
    out = HostPreludeCollection();
    if (!ownerFn || !ownerFn->hasBody() || !gRuntimeBindingSourceManager) {
        return;
    }

    const auto* body = llvm::dyn_cast<clang::CompoundStmt>(ownerFn->getBody());
    if (!body) {
        return;
    }

    const clang::SourceManager& sm = *gRuntimeBindingSourceManager;
    const clang::SourceLocation ownerLoc = sm.getExpansionLoc(ownerFn->getBeginLoc());
    const clang::Stmt* dacOwnerStmt = FindOwningStmtForDacExpr(ownerFn, cutoffLoc);
    std::vector<const clang::Stmt*> bodyStmts(body->body_begin(), body->body_end());
    ptrdiff_t ownerStmtIndex = -1;
    ptrdiff_t prevDacStmtIndex = -1;
    ptrdiff_t nextDacStmtIndex = static_cast<ptrdiff_t>(bodyStmts.size());
    const bool ownerStmtIsDacExpr = dacOwnerStmt && StmtContainsDacOperatorToken(dacOwnerStmt);
    if (ownerStmtIsDacExpr) {
        for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(bodyStmts.size()); ++idx) {
            if (bodyStmts[static_cast<size_t>(idx)] == dacOwnerStmt) {
                ownerStmtIndex = idx;
                break;
            }
        }
        if (ownerStmtIndex >= 0) {
            for (ptrdiff_t idx = 0; idx < ownerStmtIndex; ++idx) {
                if (StmtContainsDacOperatorToken(bodyStmts[static_cast<size_t>(idx)])) {
                    prevDacStmtIndex = idx;
                }
            }
            for (ptrdiff_t idx = ownerStmtIndex + 1; idx < static_cast<ptrdiff_t>(bodyStmts.size()); ++idx) {
                if (StmtContainsDacOperatorToken(bodyStmts[static_cast<size_t>(idx)])) {
                    nextDacStmtIndex = idx;
                    break;
                }
            }
        }
    }
    std::unordered_set<std::string> trackedVars = rootSourceVars;
    std::vector<const clang::FunctionDecl*> helperFns;
    std::unordered_set<const clang::FunctionDecl*> seenHelperFns;
    std::unordered_set<const clang::RecordDecl*> referencedRecordDecls;

    // 用于喂 PrependTextuallyReferencedLocalConstDecls /
    // CollectTextuallyReferencedPreludeFunctionRoots / CollectTextuallyReferencedPreludeDecls
    // 的 stmt 文本集。和 host body 输出无关。
    std::vector<std::string> preludeStmtTexts;
    // DacExprSlot 对应 stmt 的文本：只能进 decl-only 扫描，避免把 shell/calc 函数名误归为 helper。
    std::vector<std::string> dacSlotStmtTexts;

    for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(bodyStmts.size()); ++idx) {
        const clang::Stmt* stmt = bodyStmts[static_cast<size_t>(idx)];
        if (!stmt) {
            continue;
        }
        if (ownerStmtIsDacExpr && ownerStmtIndex >= 0 &&
            (idx <= prevDacStmtIndex || idx >= nextDacStmtIndex)) {
            continue;
        }

        if (stmt == dacOwnerStmt) {
            const std::string slotText = NormalizeStmtSourceText(stmt);
            if (!slotText.empty()) {
                dacSlotStmtTexts.push_back(slotText);
            }
            if (StmtContainsDacOperatorToken(stmt)) {
                CollectHelperCallsFromStmt(stmt, helperFns, seenHelperFns, &trackedVars);
            }
            TrackPreludeDerivedVarsFromStmt(
                stmt,
                trackedVars,
                out.resolvedRoots,
                referencedRecordDecls,
                true);
            continue;
        }
        if (StmtContainsDacOperatorToken(stmt)) {
            continue;
        }

        if (!ShouldIncludeStmtInPrelude(stmt, trackedVars, out.resolvedRoots, referencedRecordDecls)) {
            continue;
        }

        const std::string stmtText = NormalizeStmtSourceText(stmt);
        if (stmtText.empty()) {
            continue;
        }

        CollectHelperCallsFromStmt(stmt, helperFns, seenHelperFns);
        preludeStmtTexts.push_back(stmtText);
    }

    // PrependTextuallyReferencedLocalConstDecls 把 body 内部声明且被 stmt 文本引用的
    // 常量 stmt 文本 prepend 进来，让后续 helper/const decl 扫描覆盖到。
    std::vector<std::string> preludeStmtTextsWithConsts = preludeStmtTexts;
    PrependTextuallyReferencedLocalConstDecls(
        body,
        sm,
        cutoffLoc,
        preludeStmtTextsWithConsts);

    // helper function root 扫描：不能包含 DacExprSlot 文本（含 shell/calc 名）。
    std::vector<std::string> functionRootScanTexts = preludeStmtTextsWithConsts;

    // decl 扫描：包含 DacExprSlot 文本，以便 for/while 条件里引用的常量（如 TIME_STEPS）能被识别。
    std::vector<std::string> declScanTexts = preludeStmtTextsWithConsts;
    declScanTexts.insert(declScanTexts.end(),
                         dacSlotStmtTexts.begin(),
                         dacSlotStmtTexts.end());

    std::unordered_set<std::string> seenRecordDecls;
    for (const clang::RecordDecl* recordDecl : referencedRecordDecls) {
        if (!recordDecl) {
            continue;
        }
        const clang::RecordDecl* def = recordDecl->getDefinition();
        if (!def) {
            def = recordDecl;
        }
        if (sm.isInSystemHeader(def->getBeginLoc()) ||
            !sm.isWrittenInSameFile(sm.getExpansionLoc(def->getBeginLoc()), ownerLoc)) {
            continue;
        }

        std::string declText = TrimCopy(GetDeclSourceText(def));
        if (declText.empty() || declText.find("dacpp") != std::string::npos) {
            continue;
        }
        if (declText.back() != ';') {
            declText += ";";
        }
        if (seenRecordDecls.insert(declText).second) {
            out.recordDecls.push_back(declText);
        }
    }

    const clang::TranslationUnitDecl* tuDecl =
        gRuntimeBindingTUDecl ? gRuntimeBindingTUDecl : ownerFn->getTranslationUnitDecl();
    if (tuDecl) {
        CollectTextuallyReferencedPreludeFunctionRoots(
            tuDecl,
            sm,
            ownerLoc,
            functionRootScanTexts,
            helperFns,
            seenHelperFns);
    }

    AppendPreludeFunctionDefsFromRoots(helperFns, sm, ownerLoc, out.helperFunctionDefs);
    functionRootScanTexts.insert(
        functionRootScanTexts.end(),
        out.helperFunctionDefs.begin(),
        out.helperFunctionDefs.end());
    declScanTexts.insert(
        declScanTexts.end(),
        out.helperFunctionDefs.begin(),
        out.helperFunctionDefs.end());

    if (tuDecl) {
        CollectTextuallyReferencedPreludeDecls(
            tuDecl,
            sm,
            cutoffLoc,
            ownerLoc,
            declScanTexts,
            out.constDecls,
            out.usingDecls,
            out.recordDecls,
            &out.globalVarDecls);
    }
}

// 同名工具：把 src 中尚未出现在 ioDst 的字符串按出现顺序追加。
// （原 MergeUniqueStringList 同语义；HostPreludeCollection 内联使用。）
static void AppendUniqueStrings(std::vector<std::string>& ioDst,
                                const std::vector<std::string>& src) {
    std::unordered_set<std::string> seen(ioDst.begin(), ioDst.end());
    for (const std::string& text : src) {
        if (!text.empty() && seen.insert(text).second) {
            ioDst.push_back(text);
        }
    }
}

static void MergeHostPreludeCollection(const HostPreludeCollection& srcCollection,
                                       HostPreludeCollection& ioDstCollection) {
    AppendUniqueStrings(ioDstCollection.helperFunctionDefs, srcCollection.helperFunctionDefs);
    AppendUniqueStrings(ioDstCollection.constDecls,         srcCollection.constDecls);
    AppendUniqueStrings(ioDstCollection.globalVarDecls,     srcCollection.globalVarDecls);
    AppendUniqueStrings(ioDstCollection.usingDecls,         srcCollection.usingDecls);
    AppendUniqueStrings(ioDstCollection.recordDecls,        srcCollection.recordDecls);
    for (const auto& entry : srcCollection.tensorAliases) {
        ioDstCollection.tensorAliases.insert_or_assign(entry.first, entry.second);
    }
    for (const auto& rootName : srcCollection.resolvedRoots) {
        ioDstCollection.resolvedRoots.insert(rootName);
    }
}

static void BuildHostPreludeCollectionImpl(
    const clang::FunctionDecl* ownerFn,
    clang::SourceLocation cutoffLoc,
    const std::unordered_set<std::string>& rootSourceVars,
    HostPreludeCollection& out,
    std::unordered_set<const clang::FunctionDecl*>& visitingFns,
    unsigned depth) {
    out = HostPreludeCollection();
    if (!ownerFn || depth > 8) {
        return;
    }

    const clang::FunctionDecl* ownerCanonical = ownerFn->getCanonicalDecl();
    if (!visitingFns.insert(ownerCanonical).second) {
        return;
    }

    CollectHostPreludeFromAst(ownerFn, cutoffLoc, rootSourceVars, out);
    CollectLiveTensorAliasesInFunction(ownerFn, out.tensorAliases);

    const clang::TranslationUnitDecl* tuDecl =
        gRuntimeBindingTUDecl ? gRuntimeBindingTUDecl : ownerFn->getTranslationUnitDecl();
    if (tuDecl && gRuntimeBindingSourceManager) {
        const clang::SourceManager& sm = *gRuntimeBindingSourceManager;
        const clang::SourceLocation ownerLoc = sm.getExpansionLoc(ownerFn->getBeginLoc());
        std::unordered_map<unsigned, std::string> unresolvedParamRoots;
        for (unsigned i = 0; i < ownerFn->getNumParams(); ++i) {
            const clang::ParmVarDecl* paramDecl = ownerFn->getParamDecl(i);
            if (!paramDecl || !paramDecl->getIdentifier()) {
                continue;
            }
            const std::string paramName = paramDecl->getNameAsString();
            if (rootSourceVars.count(paramName) == 0) {
                continue;
            }
            if (out.resolvedRoots.count(paramName) > 0) {
                continue;
            }
            unresolvedParamRoots[i] = paramName;
        }

        const clang::FunctionDecl* targetCanonical = ownerFn->getCanonicalDecl();
        for (const clang::Decl* decl : tuDecl->decls()) {
            const auto* callerFn = llvm::dyn_cast_or_null<clang::FunctionDecl>(decl);
            if (!callerFn || !callerFn->hasBody() || callerFn == ownerFn) {
                continue;
            }
            if (sm.isInSystemHeader(callerFn->getBeginLoc()) ||
                !sm.isWrittenInSameFile(sm.getExpansionLoc(callerFn->getBeginLoc()), ownerLoc)) {
                continue;
            }

            class OwnerCallSiteCollector
                : public clang::RecursiveASTVisitor<OwnerCallSiteCollector> {
            public:
                explicit OwnerCallSiteCollector(const clang::FunctionDecl* target)
                    : targetFn(target) {}

                bool VisitCallExpr(clang::CallExpr* callExpr) {
                    if (!callExpr) {
                        return true;
                    }
                    const clang::FunctionDecl* callee = callExpr->getDirectCallee();
                    if (callee && callee->getCanonicalDecl() == targetFn) {
                        callSites.push_back(callExpr);
                    }
                    return true;
                }

                std::vector<const clang::CallExpr*> callSites;

            private:
                const clang::FunctionDecl* targetFn;
            } callCollector(targetCanonical);

            callCollector.TraverseStmt(const_cast<clang::Stmt*>(callerFn->getBody()));
            for (const clang::CallExpr* callExpr : callCollector.callSites) {
                std::unordered_set<std::string> callerRoots;
                for (const auto& entry : unresolvedParamRoots) {
                    if (entry.first >= callExpr->getNumArgs()) {
                        continue;
                    }
                    std::string argName;
                    if (TryExtractIdentifierFromExpr(callExpr->getArg(entry.first), argName) &&
                        !argName.empty()) {
                        callerRoots.insert(argName);
                    }
                }

                HostPreludeCollection callerCollection;
                BuildHostPreludeCollectionImpl(
                    callerFn,
                    callExpr->getExprLoc(),
                    callerRoots,
                    callerCollection,
                    visitingFns,
                    depth + 1);
                MergeHostPreludeCollection(callerCollection, out);
            }
        }
    }

    visitingFns.erase(ownerCanonical);
}

static void BuildHostPreludeCollection(
    const clang::FunctionDecl* ownerFn,
    clang::SourceLocation cutoffLoc,
    const std::unordered_set<std::string>& rootSourceVars,
    HostPreludeCollection& out) {
    std::unordered_set<const clang::FunctionDecl*> visitingFns;
    BuildHostPreludeCollectionImpl(
        ownerFn,
        cutoffLoc,
        rootSourceVars,
        out,
        visitingFns,
        0);
}

static std::vector<std::string> PruneRedundantPreludeVarDecls(
    const std::vector<std::string>& decls) {
    std::vector<std::unordered_set<std::string>> tokenSets;
    tokenSets.reserve(decls.size());
    for (const std::string& declText : decls) {
        std::vector<std::string> texts{declText};
        tokenSets.push_back(ExtractIdentifierTokensFromTexts(texts));
    }

    std::vector<bool> keep(decls.size(), true);
    for (size_t i = 0; i < decls.size(); ++i) {
        if (!keep[i]) {
            continue;
        }
        for (size_t j = 0; j < decls.size(); ++j) {
            if (i == j || !keep[j] || decls[j].size() <= decls[i].size()) {
                continue;
            }
            bool covered = !tokenSets[i].empty();
            for (const std::string& token : tokenSets[i]) {
                if (tokenSets[j].count(token) == 0) {
                    covered = false;
                    break;
                }
            }
            if (covered) {
                keep[i] = false;
                break;
            }
        }
    }

    std::vector<std::string> filtered;
    filtered.reserve(decls.size());
    for (size_t i = 0; i < decls.size(); ++i) {
        if (keep[i]) {
            filtered.push_back(decls[i]);
        }
    }
    return filtered;
}

static std::string BuildHostPreludeCodeFromCollection(const HostPreludeCollection& program) {
    std::string prelude;
    std::unordered_set<std::string> seenPreludeLines;
    const std::vector<std::string> prunedConstDecls =
        PruneRedundantPreludeVarDecls(program.constDecls);
    const std::vector<std::string> prunedGlobalDecls =
        PruneRedundantPreludeVarDecls(program.globalVarDecls);

    auto AppendSnippet = [&](const std::string& snippet, bool separateWithBlankLine) {
        const std::string trimmed = TrimCopy(snippet);
        if (trimmed.empty() || !seenPreludeLines.insert(trimmed).second) {
            return;
        }
        if (separateWithBlankLine && !prelude.empty()) {
            prelude += "\n";
        }
        prelude += trimmed;
        if (prelude.empty() || prelude.back() != '\n') {
            prelude += "\n";
        }
    };

    for (const auto& decl : program.usingDecls) {
        AppendSnippet(decl, false);
    }
    for (const auto& decl : prunedConstDecls) {
        AppendSnippet(decl, false);
    }
    for (const auto& decl : prunedGlobalDecls) {
        AppendSnippet(decl, false);
    }
    for (const auto& decl : program.recordDecls) {
        AppendSnippet(decl, true);
    }
    for (const auto& fnDef : program.helperFunctionDefs) {
        AppendSnippet(fnDef, true);
    }

    return TrimCopy(prelude);
}

static std::string IndentCodeBlock(const std::string& code,
                                   const std::string& indent) {
    if (code.empty()) {
        return "";
    }

    std::stringstream ss(code);
    std::string line;
    std::string indented;
    while (std::getline(ss, line)) {
        indented += indent + line + "\n";
    }
    return indented;
}

// Build a host exec block from common decls + exec code. When `scoped` is true
// the block is wrapped in an enclosing `{ ... }`; when false the body is emitted
// without braces so declared variables stay in scope for a surrounding for-loop
// (pre-loop alloc) / post-loop free block.
static std::string BuildHostExecBlock(const std::string& commonDecls,
                                      const std::string& execCode,
                                      bool scoped) {
    std::string block = scoped ? "    {\n" : "";
    const std::string trimmedCommonDecls = TrimCopy(commonDecls);
    const std::string trimmedExecCode = TrimCopy(execCode);
    if (!trimmedCommonDecls.empty()) {
        block += IndentCodeBlock(trimmedCommonDecls, "    ");
    }
    if (!trimmedExecCode.empty()) {
        block += IndentCodeBlock(trimmedExecCode, "    ");
    }
    if (scoped) {
        block += "    }\n";
    }
    return block;
}

static std::string BuildHostGenerationErrorBody(const std::string& message,
                                                const std::string& sessionName) {
    std::string escaped;
    const std::size_t limit = std::min<std::size_t>(message.size(), 160);
    escaped.reserve(limit + 8);
    for (std::size_t i = 0; i < limit; ++i) {
        char c = message[i];
        if (c == '\n' || c == '\r') c = ' ';
        if (c == '\\' || c == '"') {
            escaped.push_back('\\');
        }
        escaped.push_back(c);
    }
    return "    " + sessionName +
           ".fail(dacpp_host::Stage::HostFlow, 1, \"" + escaped + "\");\n";
}

// host body 由 HostFlowEmitter::RenderHostFlowBody 驱动。

static std::string RemoveAllSpaces(const std::string& text) {
    std::string compact;
    compact.reserve(text.size());
    for (char c : text) {
        if (!std::isspace(static_cast<unsigned char>(c))) {
            compact.push_back(c);
        }
    }
    return compact;
}

static size_t FindMatchingTextDelimiter(const std::string& text,
                                        size_t openPos,
                                        char openCh,
                                        char closeCh) {
    if (openPos >= text.size() || text[openPos] != openCh) {
        return std::string::npos;
    }

    int depth = 0;
    for (size_t i = openPos; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '"' || c == '\'') {
            const char quote = c;
            ++i;
            while (i < text.size()) {
                if (text[i] == '\\') {
                    i += 2;
                    continue;
                }
                if (text[i] == quote) {
                    break;
                }
                ++i;
            }
            continue;
        }
        if (c == '/' && i + 1 < text.size()) {
            if (text[i + 1] == '/') {
                i += 2;
                while (i < text.size() && text[i] != '\n') {
                    ++i;
                }
                continue;
            }
            if (text[i + 1] == '*') {
                i += 2;
                while (i + 1 < text.size() && !(text[i] == '*' && text[i + 1] == '/')) {
                    ++i;
                }
                if (i + 1 < text.size()) {
                    ++i;
                }
                continue;
            }
        }
        if (c == openCh) {
            ++depth;
            continue;
        }
        if (c == closeCh) {
            --depth;
            if (depth == 0) {
                return i;
            }
        }
    }
    return std::string::npos;
}

static size_t FindTopLevelAssignmentPos(const std::string& text) {
    int parenDepth = 0;
    int bracketDepth = 0;
    int braceDepth = 0;
    for (size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        switch (c) {
        case '(':
            ++parenDepth;
            break;
        case ')':
            if (parenDepth > 0) {
                --parenDepth;
            }
            break;
        case '[':
            ++bracketDepth;
            break;
        case ']':
            if (bracketDepth > 0) {
                --bracketDepth;
            }
            break;
        case '{':
            ++braceDepth;
            break;
        case '}':
            if (braceDepth > 0) {
                --braceDepth;
            }
            break;
        case '=':
            if (parenDepth == 0 && bracketDepth == 0 && braceDepth == 0) {
                const char prev = (i > 0) ? text[i - 1] : '\0';
                const char next = (i + 1 < text.size()) ? text[i + 1] : '\0';
                if (prev != '=' && prev != '!' && prev != '<' && prev != '>' && next != '=') {
                    return i;
                }
            }
            break;
        default:
            break;
        }
    }
    return std::string::npos;
}

static bool MatchesScalarStateLhs(const std::string& lhsText,
                                  const std::vector<std::string>& candidateNames) {
    const std::string compact = RemoveAllSpaces(StripTrailingSemicolon(lhsText));
    for (const std::string& candidate : candidateNames) {
        if (candidate.empty()) {
            continue;
        }
        if (compact == candidate || compact == candidate + "[0]") {
            return true;
        }
    }
    return false;
}

static bool TryExtractPlusAssignRhs(const std::string& stmtText,
                                    const std::vector<std::string>& candidateNames,
                                    std::string& outRhs) {
    outRhs.clear();
    std::string text = StripTrailingSemicolon(stmtText);
    size_t pos = text.find("+=");
    if (pos == std::string::npos) {
        return false;
    }

    const std::string lhsText = TrimCopy(text.substr(0, pos));
    if (!MatchesScalarStateLhs(lhsText, candidateNames)) {
        return false;
    }

    outRhs = TrimCopy(text.substr(pos + 2));
    return !outRhs.empty();
}

static bool TryExtractTensorSnapshotAssign(const std::string& stmtText,
                                          const std::string& rhsName,
                                          std::string& outTensorName,
                                          std::string& outIndexExpr) {
    outTensorName.clear();
    outIndexExpr.clear();

    const std::string text = StripTrailingSemicolon(stmtText);
    const size_t assignPos = FindTopLevelAssignmentPos(text);
    if (assignPos == std::string::npos) {
        return false;
    }

    const std::string lhsText = TrimCopy(text.substr(0, assignPos));
    const std::string rhsText = TrimCopy(text.substr(assignPos + 1));
    if (RemoveAllSpaces(rhsText) != RemoveAllSpaces(rhsName)) {
        return false;
    }

    const std::string lhsCompact = RemoveAllSpaces(lhsText);
    const size_t openBracket = lhsCompact.find('[');
    if (openBracket == std::string::npos || lhsCompact.back() != ']') {
        return false;
    }

    outTensorName = lhsCompact.substr(0, openBracket);
    if (!IsSimpleIdentifierText(outTensorName)) {
        outTensorName.clear();
        return false;
    }

    outIndexExpr = lhsCompact.substr(openBracket + 1, lhsCompact.size() - openBracket - 2);
    return !outIndexExpr.empty();
}

static bool TryExtractPrintRowExpr(const std::string& bodyText,
                                   const std::string& tensorName,
                                   std::string& outRowExpr) {
    outRowExpr.clear();
    if (tensorName.empty()) {
        return false;
    }

    size_t searchPos = 0;
    while (true) {
        size_t tensorPos = bodyText.find(tensorName, searchPos);
        if (tensorPos == std::string::npos) {
            return false;
        }

        if (tensorPos > 0 && IsIdentCharText(bodyText[tensorPos - 1])) {
            searchPos = tensorPos + tensorName.size();
            continue;
        }

        size_t openBracket = tensorPos + tensorName.size();
        while (openBracket < bodyText.size() &&
               std::isspace(static_cast<unsigned char>(bodyText[openBracket]))) {
            ++openBracket;
        }
        if (openBracket >= bodyText.size() || bodyText[openBracket] != '[') {
            searchPos = tensorPos + tensorName.size();
            continue;
        }

        const size_t closeBracket =
            FindMatchingTextDelimiter(bodyText, openBracket, '[', ']');
        if (closeBracket == std::string::npos) {
            return false;
        }

        size_t suffixPos = closeBracket + 1;
        while (suffixPos < bodyText.size() &&
               std::isspace(static_cast<unsigned char>(bodyText[suffixPos]))) {
            ++suffixPos;
        }
        if (suffixPos >= bodyText.size() || bodyText[suffixPos] != '.') {
            searchPos = closeBracket + 1;
            continue;
        }
        ++suffixPos;
        while (suffixPos < bodyText.size() &&
               std::isspace(static_cast<unsigned char>(bodyText[suffixPos]))) {
            ++suffixPos;
        }
        if (bodyText.compare(suffixPos, 5, "print") != 0) {
            searchPos = closeBracket + 1;
            continue;
        }
        suffixPos += 5;
        while (suffixPos < bodyText.size() &&
               std::isspace(static_cast<unsigned char>(bodyText[suffixPos]))) {
            ++suffixPos;
        }
        if (suffixPos >= bodyText.size() || bodyText[suffixPos] != '(') {
            searchPos = closeBracket + 1;
            continue;
        }
        const size_t closeParen =
            FindMatchingTextDelimiter(bodyText, suffixPos, '(', ')');
        if (closeParen == std::string::npos) {
            return false;
        }
        if (TrimCopy(bodyText.substr(suffixPos + 1, closeParen - suffixPos - 1)).empty()) {
            outRowExpr = TrimCopy(bodyText.substr(openBracket + 1, closeBracket - openBracket - 1));
            return !outRowExpr.empty();
        }
        searchPos = closeParen + 1;
    }
}

static std::string ReverseComparisonOperator(const std::string& op) {
    if (op == "<") return ">";
    if (op == "<=") return ">=";
    if (op == ">") return "<";
    if (op == ">=") return "<=";
    return op;
}

static bool ExprTextMentionsScalarState(const std::string& exprText,
                                        const std::string& scalarArgName,
                                        const std::string& scalarSourceVar) {
    const std::string text = TrimCopy(exprText);
    if (text.empty()) {
        return false;
    }

    auto MatchesBase = [&](const std::string& baseName) {
        if (baseName.empty()) {
            return false;
        }
        const std::string escaped = EscapeRegexText(baseName);
        std::regex accessRegex("\\b" + escaped + R"(\s*\[\s*0\s*\])");
        std::regex directRegex("\\b" + escaped + "\\b");
        return std::regex_search(text, accessRegex) || std::regex_search(text, directRegex);
    };

    return MatchesBase(scalarArgName) || MatchesBase(scalarSourceVar);
}

static bool TryExtractPointwiseScalarLoopScheduleFromAst(
    Expression* expr,
    const dacppTranslator::ascendc_mlir::CalcShellSummary& shellSummary,
    PointwiseScalarLoopSchedule& outSchedule) {
    outSchedule = PointwiseScalarLoopSchedule();
    if (!expr || !expr->getShell() || !gRuntimeBindingSourceManager) {
        return false;
    }

    const clang::BinaryOperator* dacExpr = expr->getDacExpr();
    if (!dacExpr) {
        return false;
    }

    const clang::FunctionDecl* ownerFunction = gRuntimeBindingCurrentOwnerFunction;
    if (!ownerFunction) {
        ownerFunction = FindEnclosingFunction(dacExpr);
    }
    if (!ownerFunction || !ownerFunction->getBody()) {
        return false;
    }

    const clang::SourceManager& sm = *gRuntimeBindingSourceManager;
    const clang::SourceLocation dacLoc = sm.getExpansionLoc(dacExpr->getExprLoc());
    const clang::WhileStmt* outerWhile =
        FindInnermostStmtOfType<clang::WhileStmt>(ownerFunction->getBody(), sm, dacLoc);
    if (!outerWhile) {
        return false;
    }

    const auto* loopBody = llvm::dyn_cast<clang::CompoundStmt>(outerWhile->getBody());
    if (!loopBody) {
        return false;
    }

    unsigned dacStmtIndex = 0;
    if (!TryFindContainingStmtIndex(loopBody, sm, dacLoc, dacStmtIndex)) {
        return false;
    }

    Shell* shell = expr->getShell();
    const std::vector<std::string> shellCallArgs = ResolveShellCallArgumentsFromAst(dacExpr, shell);
    if (shellCallArgs.empty()) {
        return false;
    }

    int scalarParamIdx = -1;
    int outputParamIdx = -1;
    for (int j = 0; j < shell->getNumParams(); ++j) {
        ShellParam* shellParam = shell->getShellParam(j);
        if (!shellParam || static_cast<size_t>(j) >= shellSummary.params.size()) {
            continue;
        }
        const auto& summary = shellSummary.params[static_cast<size_t>(j)];
        const int rw = shellParam->getRw();
        if ((rw == 0 || rw == 2) && summary.isBroadcastScalar && scalarParamIdx < 0) {
            scalarParamIdx = j;
        }
        if ((rw == 1 || rw == 2) && outputParamIdx < 0) {
            outputParamIdx = j;
        }
    }

    if (scalarParamIdx < 0 || outputParamIdx < 0 ||
        static_cast<size_t>(scalarParamIdx) >= shellCallArgs.size() ||
        static_cast<size_t>(outputParamIdx) >= shellCallArgs.size()) {
        return false;
    }

    const std::string scalarArgName = TrimCopy(shellCallArgs[static_cast<size_t>(scalarParamIdx)]);
    const std::string outputArgName = TrimCopy(shellCallArgs[static_cast<size_t>(outputParamIdx)]);
    if (!IsSimpleIdentifierText(scalarArgName) || !IsSimpleIdentifierText(outputArgName)) {
        return false;
    }

    RuntimeBindingResolveResult scalarBinding =
        dacppTranslator::ascendc::ResolveRuntimeBindingFromScope(
            MakeRuntimeBindingAstContext(), ownerFunction->getBody(), scalarArgName);

    std::string scalarInitExpr;
    if (!scalarBinding.broadcastExpr.empty()) {
        scalarInitExpr = TrimCopy(scalarBinding.broadcastExpr);
    } else if (!scalarBinding.scalarExpr.empty()) {
        scalarInitExpr = TrimCopy(scalarBinding.scalarExpr);
    } else if (scalarBinding.isStdVector && !scalarBinding.sourceVar.empty()) {
        scalarInitExpr = scalarBinding.sourceVar + "[0]";
    }
    if (scalarInitExpr.empty()) {
        scalarInitExpr = "0";
    }

    std::string loopCompareOp;
    std::string loopUpperBoundExpr;
    if (const auto* cond = llvm::dyn_cast_or_null<clang::BinaryOperator>(outerWhile->getCond())) {
        const std::string lhsText = TrimCopy(GetExprSourceText(cond->getLHS()));
        const std::string rhsText = TrimCopy(GetExprSourceText(cond->getRHS()));
        const std::string opText =
            clang::BinaryOperator::getOpcodeStr(cond->getOpcode()).str();
        if (ExprTextMentionsScalarState(lhsText, scalarArgName, scalarBinding.sourceVar)) {
            loopCompareOp = opText;
            loopUpperBoundExpr = rhsText;
        } else if (ExprTextMentionsScalarState(rhsText, scalarArgName, scalarBinding.sourceVar)) {
            loopCompareOp = ReverseComparisonOperator(opText);
            loopUpperBoundExpr = lhsText;
        }
    }
    if (loopCompareOp.empty() || loopUpperBoundExpr.empty()) {
        return false;
    }

    std::string loopStepExpr;
    std::string snapshotTensorName;
    std::string snapshotIndexExpr;
    std::vector<std::string> scalarStateNames;
    scalarStateNames.push_back(scalarArgName);
    if (!scalarBinding.sourceVar.empty() && scalarBinding.sourceVar != scalarArgName) {
        scalarStateNames.push_back(scalarBinding.sourceVar);
    }

    for (unsigned stmtIdx = dacStmtIndex + 1; stmtIdx < loopBody->size(); ++stmtIdx) {
        const clang::Stmt* stmt = *(loopBody->body_begin() + stmtIdx);
        if (!stmt) {
            continue;
        }

        const std::string stmtText = TrimCopy(GetStmtSourceText(stmt));
        if (stmtText.empty()) {
            continue;
        }

        if (loopStepExpr.empty()) {
            TryExtractPlusAssignRhs(stmtText, scalarStateNames, loopStepExpr);
        }

        if (snapshotTensorName.empty()) {
            TryExtractTensorSnapshotAssign(
                stmtText, outputArgName, snapshotTensorName, snapshotIndexExpr);
        }
    }

    if (loopStepExpr.empty()) {
        return false;
    }

    std::string snapshotPrintRowExpr;
    if (!snapshotTensorName.empty()) {
        const std::string ownerBodyText = GetStmtSourceText(ownerFunction->getBody());
        TryExtractPrintRowExpr(ownerBodyText, snapshotTensorName, snapshotPrintRowExpr);
    }

    outSchedule.recognized = true;
    outSchedule.ownerFunctionName = ownerFunction->getNameInfo().getAsString();
    outSchedule.scalarParamName = shellSummary.params[static_cast<size_t>(scalarParamIdx)].name;
    outSchedule.scalarArgName = scalarArgName;
    outSchedule.scalarSourceVar = scalarBinding.sourceVar;
    outSchedule.scalarInitExpr = scalarInitExpr;
    outSchedule.loopCompareOp = loopCompareOp;
    outSchedule.loopUpperBoundExpr = loopUpperBoundExpr;
    outSchedule.loopStepExpr = loopStepExpr;
    outSchedule.outputParamName = shellSummary.params[static_cast<size_t>(outputParamIdx)].name;
    outSchedule.outputArgName = outputArgName;
    outSchedule.snapshotTensorName = snapshotTensorName;
    outSchedule.snapshotIndexExpr = snapshotIndexExpr;
    outSchedule.snapshotPrintRowExpr = snapshotPrintRowExpr;
    outSchedule.absorbsOwningWhile = true;
    return true;
}

static bool TryExtractLoopCarriedVectorScheduleFromAst(
    Expression* expr,
    const dacppTranslator::ascendc_mlir::CalcShellSummary& shellSummary,
    LoopCarriedVectorSchedule& outSchedule) {
    outSchedule = LoopCarriedVectorSchedule();
    if (!expr || !expr->getShell() || !gRuntimeBindingSourceManager) {
        return false;
    }

    const clang::BinaryOperator* dacExpr = expr->getDacExpr();
    if (!dacExpr) {
        return false;
    }

    const clang::FunctionDecl* ownerFunction = gRuntimeBindingCurrentOwnerFunction;
    if (!ownerFunction) {
        ownerFunction = FindEnclosingFunction(dacExpr);
    }
    if (!ownerFunction || !ownerFunction->getBody()) {
        return false;
    }

    const clang::SourceManager& sm = *gRuntimeBindingSourceManager;
    const clang::SourceLocation dacLoc = sm.getExpansionLoc(dacExpr->getExprLoc());
    const clang::ForStmt* outerFor =
        FindInnermostStmtOfType<clang::ForStmt>(ownerFunction->getBody(), sm, dacLoc);
    if (!outerFor) {
        return false;
    }

    const auto* loopBody = llvm::dyn_cast<clang::CompoundStmt>(outerFor->getBody());
    if (!loopBody) {
        return false;
    }

    std::string loopIndexVar;
    std::string loopBeginExpr;
    std::string loopUpperBoundExpr;
    std::string loopStepExpr;
    if (!TryGetForLoopControl(
            outerFor,
            loopIndexVar,
            loopBeginExpr,
            loopUpperBoundExpr,
            loopStepExpr)) {
        return false;
    }

    unsigned dacStmtIndex = 0;
    if (!TryFindContainingStmtIndex(loopBody, sm, dacLoc, dacStmtIndex)) {
        return false;
    }

    Shell* shell = expr->getShell();
    const std::vector<std::string> shellCallArgs =
        ResolveShellCallArgumentsFromAst(dacExpr, shell);
    if (shellCallArgs.empty()) {
        return false;
    }

    int inputParamIdx = -1;
    int outputParamIdx = -1;
    for (int j = 0; j < shell->getNumParams(); ++j) {
        ShellParam* shellParam = shell->getShellParam(j);
        if (!shellParam || static_cast<size_t>(j) >= shellSummary.params.size()) {
            continue;
        }
        const int rw = shellParam->getRw();
        if ((rw == 0 || rw == 2) && inputParamIdx < 0) {
            inputParamIdx = j;
        }
        if ((rw == 1 || rw == 2) && outputParamIdx < 0) {
            outputParamIdx = j;
        }
    }

    if (inputParamIdx < 0 || outputParamIdx < 0 ||
        static_cast<size_t>(inputParamIdx) >= shellCallArgs.size() ||
        static_cast<size_t>(outputParamIdx) >= shellCallArgs.size()) {
        return false;
    }

    const std::string inputArgName =
        TrimCopy(shellCallArgs[static_cast<size_t>(inputParamIdx)]);
    const std::string outputArgName =
        TrimCopy(shellCallArgs[static_cast<size_t>(outputParamIdx)]);
    if (!IsSimpleIdentifierText(inputArgName) ||
        !IsSimpleIdentifierText(outputArgName)) {
        return false;
    }

    bool foundWriteback = false;
    bool foundBoundary = false;
    unsigned idx = 0;
    for (const clang::Stmt* stmt : loopBody->body()) {
        if (idx <= dacStmtIndex) {
            ++idx;
            continue;
        }
        ++idx;

        if (!foundWriteback) {
            if (const auto* forStmt = llvm::dyn_cast<clang::ForStmt>(stmt)) {
                std::string innerIdxVar;
                std::string innerBeginExpr;
                std::string innerEndExpr;
                std::string innerStepExpr;
                if (TryGetForLoopControl(
                        forStmt,
                        innerIdxVar,
                        innerBeginExpr,
                        innerEndExpr,
                        innerStepExpr)) {
                    const std::string bodyText = GetStmtSourceText(forStmt->getBody());
                    const std::string escapedInput = EscapeRegexText(inputArgName);
                    const std::string escapedOutput = EscapeRegexText(outputArgName);
                    std::smatch match;
                    const std::regex writebackRegex(
                        escapedInput + R"(\s*\[\s*([^\]]+)\s*\]\s*=\s*)" +
                        escapedOutput + R"(\s*\[\s*([^\]]+)\s*\]\s*;)");
                    if (std::regex_search(bodyText, match, writebackRegex)) {
                        int dstOffset = 0;
                        int srcOffset = 0;
                        if (TryParseAffineExprAgainstVar(
                                TrimCopy(match[1].str()),
                                innerIdxVar,
                                dstOffset) &&
                            TryParseAffineExprAgainstVar(
                                TrimCopy(match[2].str()),
                                innerIdxVar,
                                srcOffset)) {
                            outSchedule.writebackIndexVar = innerIdxVar;
                            outSchedule.writebackBeginExpr = innerBeginExpr;
                            outSchedule.writebackEndExpr = innerEndExpr;
                            outSchedule.writebackDstOffset = dstOffset;
                            outSchedule.writebackSrcOffset = srcOffset;
                            foundWriteback = true;
                        }
                    }
                }
            }
        }

        if (!foundBoundary) {
            const std::string stmtText = TrimCopy(GetStmtSourceText(stmt));
            const std::string escapedInput = EscapeRegexText(inputArgName);
            const std::string escapedOutput = EscapeRegexText(outputArgName);
            const std::regex boundaryRegex(
                escapedInput + R"(\s*\[\s*0\s*\]\s*=\s*)" +
                escapedOutput + R"(\s*\[\s*0\s*\]\s*;)");
            foundBoundary = std::regex_search(stmtText, boundaryRegex);
        }
    }

    if (!foundWriteback || outSchedule.writebackSrcOffset != -1) {
        return false;
    }

    outSchedule.recognized = true;
    outSchedule.absorbsOwningFor = true;
    outSchedule.ownerFunctionName = ownerFunction->getNameInfo().getAsString();
    outSchedule.loopIndexVar = loopIndexVar;
    outSchedule.loopBeginExpr = loopBeginExpr;
    outSchedule.loopUpperBoundExpr = loopUpperBoundExpr;
    outSchedule.loopStepExpr = loopStepExpr;
    outSchedule.inputParamName = shellSummary.params[static_cast<size_t>(inputParamIdx)].name;
    outSchedule.inputArgName = inputArgName;
    outSchedule.outputParamName = shellSummary.params[static_cast<size_t>(outputParamIdx)].name;
    outSchedule.outputArgName = outputArgName;
    outSchedule.hasLeftBoundaryUpdate = foundBoundary;
    return true;
}


// ===== Kernel route helpers =====
static std::string GetFunctionDeclSourceText(const clang::FunctionDecl* fnDecl,
                                             clang::Rewriter* rewriter) {
    if (!fnDecl || !rewriter) {
        return "";
    }
    return clang::Lexer::getSourceText(
               clang::CharSourceRange::getTokenRange(fnDecl->getSourceRange()),
               rewriter->getSourceMgr(),
               rewriter->getLangOpts())
        .str();
}

static bool ShouldUseCubeRoute(Expression* expr,
                               clang::Rewriter* rewriter) {
    if (!expr) {
        return false;
    }

    Shell* shell = expr->getShell();
    if (!shell || shell->getNumParams() <= 0) {
        return false;
    }

    std::vector<ShellParam*> inputMatrices;
    std::vector<ShellParam*> outputMatrices;
    for (int j = 0; j < shell->getNumParams(); ++j) {
        ShellParam* param = shell->getShellParam(j);
        if (!param) {
            return false;
        }
        const std::string paramType = param->getType();
        if (paramType.find("Matrix") == std::string::npos) {
            return false;
        }
        if (ParamUsesRecordElementType(param)) {
            return false;
        }

        const int rw = param->getRw();
        if (rw == 0) {
            inputMatrices.push_back(param);
        } else if (rw == 1 || rw == 2) {
            outputMatrices.push_back(param);
        }
    }

    if (inputMatrices.size() < 2 || outputMatrices.empty()) {
        return false;
    }

    const std::string shellText = GetFunctionDeclSourceText(shell->getShellLoc(), rewriter);
    if (shellText.empty()) {
        return false;
    }

    auto MatchesMatrixSlicePattern = [&](const std::string& name, bool rowSlice) {
        const std::string escapedName = EscapeRegexText(name);
        const std::string pattern = rowSlice
            ? (escapedName + R"(\s*\[[^\]]+\]\s*\[\s*\{\s*\}\s*\])")
            : (escapedName + R"(\s*\[\s*\{\s*\}\s*\]\s*\[[^\]]+\])");
        return std::regex_search(shellText, std::regex(pattern));
    };

    auto MatchesScalarOutputPattern = [&](const std::string& name) {
        const std::string escapedName = EscapeRegexText(name);
        const std::string pattern =
            escapedName + R"(\s*\[\s*[^\{\}\]]+\s*\]\s*\[\s*[^\{\}\]]+\s*\])";
        return std::regex_search(shellText, std::regex(pattern));
    };

    bool hasRowSliceInput = false;
    bool hasColSliceInput = false;
    for (ShellParam* param : inputMatrices) {
        hasRowSliceInput = hasRowSliceInput || MatchesMatrixSlicePattern(param->getName(), true);
        hasColSliceInput = hasColSliceInput || MatchesMatrixSlicePattern(param->getName(), false);
    }

    bool hasScalarOutput = false;
    for (ShellParam* param : outputMatrices) {
        hasScalarOutput = hasScalarOutput || MatchesScalarOutputPattern(param->getName());
    }

    return hasRowSliceInput && hasColSliceInput && hasScalarOutput;
}

static bool TryExtractCubeMatmulStepFromPlan(
    const dacppTranslator::ascendc_mlir::AscendCPlan& plan,
    dacppTranslator::ascendc_mlir::AscendCPlanStep& outStep) {
    for (const auto& step : plan.steps) {
        if (step.kind != dacppTranslator::ascendc_mlir::AscendCPlanStepKind::CubeMatmul) {
            continue;
        }
        outStep = step;
        return true;
    }
    return false;
}

// 提取矩阵维度的辅助函数 
// 用于匹配类似于 matA({x, y}, dataA) 或 matA(M, K) 的声明
static void ExtractMatrixDimsFromSource(const std::string& sourceText, const std::string& paramName, std::string& dim1, std::string& dim2) {
    // 匹配 DACPP 原生的带大括号初始化格式，例如 matA({4, 5}, dataA)
    std::string pattern1 = paramName + "\\s*\\(\\s*\\{\\s*([0-9a-zA-Z_]+)\\s*,\\s*([0-9a-zA-Z_]+)\\s*\\}";
    std::regex reg1(pattern1);
    std::smatch match1;

    // 匹配普通构造函数格式，例如 matA(4, 5)
    std::string pattern2 = paramName + "\\s*\\(\\s*([0-9a-zA-Z_]+)\\s*,\\s*([0-9a-zA-Z_]+)\\s*\\)";
    std::regex reg2(pattern2);
    std::smatch match2;

    if (std::regex_search(sourceText, match1, reg1)) {
        dim1 = match1[1].str();
        dim2 = match1[2].str();
    } else if (std::regex_search(sourceText, match2, reg2)) {
        dim1 = match2[1].str();
        dim2 = match2[2].str();
    }
}
struct VectorKernelSkeletonFragments {
    std::string initCode;
    std::string copyInCode;
    std::string computeCode;
    std::string copyOutCode;
    std::string tQueDecl;
};

static dacppTranslator::ascendc_mlir::AscendCTransferPolicy ResolveTransferPolicyForParam(
    bool isInputParam,
    bool isBroadcastScalar,
    int leftHalo,
    int rightHalo) {
    using dacppTranslator::ascendc_mlir::AscendCTransferPolicy;
    if (isInputParam && isBroadcastScalar) {
        return AscendCTransferPolicy::BroadcastScalar;
    }
    if (isInputParam && (leftHalo > 0 || rightHalo > 0)) {
        return AscendCTransferPolicy::SpatialWindow;
    }
    return AscendCTransferPolicy::PointwiseDirect;
}

static void EmitBroadcastScalarInputTransfer(
    const std::string& kernelMappedType,
    const std::string& paramName,
    const std::string& gmName,
    const std::string& queueName,
    const std::string&,
    const std::string&,
    const std::string&,
    std::string& tQueDecl,
    std::string& initCode,
    std::string& copyInCode,
    std::string& computeDeQueCode,
    std::string& computeFreeCode) {
    const std::string scalarValueName = paramName + "ScalarValue";
    (void)tQueDecl;
    initCode += "        " + gmName + ".SetGlobalBuffer((__gm__ " + kernelMappedType + " *)" + paramName + ", 1);\n";
    initCode += "        pipe.InitBuffer(" + queueName + ", BUFFER_NUM, tileLength * sizeof(" + kernelMappedType + "));\n";

    copyInCode += "        AscendC::LocalTensor<" + kernelMappedType + "> " + paramName + "Local = " + queueName + ".AllocTensor<" + kernelMappedType + ">();\n";
    copyInCode += "        " + kernelMappedType + " " + scalarValueName + " = " + gmName + ".GetValue(0);\n";
    copyInCode += "        " + paramName + "Local.SetValue(0, " + scalarValueName + ");\n";
    copyInCode += "        for (int32_t fill = 1; fill < static_cast<int32_t>(currentTileLength); ++fill) {\n";
    copyInCode += "            " + paramName + "Local.SetValue(fill, " + scalarValueName + ");\n";
    copyInCode += "        }\n";
    copyInCode += "        " + queueName + ".EnQue(" + paramName + "Local);\n";

    computeDeQueCode += "        AscendC::LocalTensor<" + kernelMappedType + "> " + paramName + "Local = " + queueName + ".DeQue<" + kernelMappedType + ">();\n";
    computeDeQueCode += "        const auto " + scalarValueName + " = " + gmName + ".GetValue(0);\n";
    computeFreeCode += "        " + queueName + ".FreeTensor(" + paramName + "Local);\n";
}

static void EmitAlignedInputTransfer(
    const std::string& kernelMappedType,
    const std::string& paramName,
    const std::string& gmName,
    const std::string& queueName,
    const std::string&,
    const std::string&,
    const std::string&,
    std::string& tQueDecl,
    std::string& initCode,
    std::string& copyInCode,
    std::string& computeDeQueCode,
    std::string& computeFreeCode) {
    (void)tQueDecl;
    initCode += "        " + gmName + ".SetGlobalBuffer((__gm__ " + kernelMappedType + " *)" + paramName + " + blockStart, blockLength);\n";
    initCode += "        pipe.InitBuffer(" + queueName + ", BUFFER_NUM, tileLength * sizeof(" + kernelMappedType + "));\n";

    copyInCode += "        AscendC::LocalTensor<" + kernelMappedType + "> " + paramName + "Local = " + queueName + ".AllocTensor<" + kernelMappedType + ">();\n";
    copyInCode += "        AscendC::DataCopy(" + paramName + "Local, " + gmName + "[tileOffset], currentTileLength);\n";
    copyInCode += "        " + queueName + ".EnQue(" + paramName + "Local);\n";

    computeDeQueCode += "        AscendC::LocalTensor<" + kernelMappedType + "> " + paramName + "Local = " + queueName + ".DeQue<" + kernelMappedType + ">();\n";
    computeFreeCode += "        " + queueName + ".FreeTensor(" + paramName + "Local);\n";
}

static void EmitHaloInputTransfer(
    const std::string& kernelMappedType,
    const std::string& paramName,
    const std::string& gmName,
    const std::string& queueName,
    const std::string& leftHaloName,
    const std::string& rightHaloName,
    const std::string& validLengthName,
    const std::string& copyLengthName,
    const std::string& gmStartName,
    const std::string& gmLengthName,
    const std::string&,
    int leftHalo,
    int rightHalo,
    int alignBytes,
    std::string& tQueDecl,
    std::string& initCode,
    std::string& copyInCode,
    std::string& computeDeQueCode,
    std::string& computeFreeCode) {
    tQueDecl += "    int32_t " + leftHaloName + " = 0;\n";
    tQueDecl += "    int32_t " + rightHaloName + " = 0;\n";
    tQueDecl += "    uint32_t " + validLengthName + " = 0;\n";
    tQueDecl += "    uint32_t " + copyLengthName + " = 0;\n";
    tQueDecl += "    int32_t " + gmStartName + " = 0;\n";
    tQueDecl += "    uint32_t " + gmLengthName + " = 0;\n";

    initCode += "        " + leftHaloName + " = " + std::to_string(leftHalo) + ";\n";
    initCode += "        " + rightHaloName + " = " + std::to_string(rightHalo) + ";\n";
    initCode += "        " + validLengthName + " = tileLength;\n";
    initCode += "        " + copyLengthName + " = (" + validLengthName + " + static_cast<uint32_t>(" + leftHaloName + " + " + rightHaloName + ") + static_cast<uint32_t>((" + std::to_string(alignBytes) + " + sizeof(" + kernelMappedType + ") - 1) / sizeof(" + kernelMappedType + ")) - 1) / static_cast<uint32_t>((" + std::to_string(alignBytes) + " + sizeof(" + kernelMappedType + ") - 1) / sizeof(" + kernelMappedType + ")) * static_cast<uint32_t>((" + std::to_string(alignBytes) + " + sizeof(" + kernelMappedType + ") - 1) / sizeof(" + kernelMappedType + "));\n";
    initCode += "        " + gmStartName + " = static_cast<int32_t>(blockStart) - " + leftHaloName + ";\n";
    initCode += "        if (" + gmStartName + " < 0) { " + gmStartName + " = 0; }\n";
    initCode += "        " + gmLengthName + " = totalLength - static_cast<uint32_t>(" + gmStartName + ") + static_cast<uint32_t>(" + rightHaloName + ");\n";
    initCode += "        " + gmName + ".SetGlobalBuffer((__gm__ " + kernelMappedType + " *)" + paramName + " + " + gmStartName + ", " + gmLengthName + ");\n";
    initCode += "        pipe.InitBuffer(" + queueName + ", BUFFER_NUM, " + copyLengthName + " * sizeof(" + kernelMappedType + "));\n";

    copyInCode += "        AscendC::LocalTensor<" + kernelMappedType + "> " + paramName + "Local = " + queueName + ".AllocTensor<" + kernelMappedType + ">();\n";
    copyInCode += "        const uint32_t " + paramName + "CurrentCopyLength = (currentTileLength + static_cast<uint32_t>(" + leftHaloName + " + " + rightHaloName + ") + static_cast<uint32_t>((" + std::to_string(alignBytes) + " + sizeof(" + kernelMappedType + ") - 1) / sizeof(" + kernelMappedType + ")) - 1) / static_cast<uint32_t>((" + std::to_string(alignBytes) + " + sizeof(" + kernelMappedType + ") - 1) / sizeof(" + kernelMappedType + ")) * static_cast<uint32_t>((" + std::to_string(alignBytes) + " + sizeof(" + kernelMappedType + ") - 1) / sizeof(" + kernelMappedType + "));\n";
    copyInCode += "        AscendC::DataCopy(" + paramName + "Local, " + gmName + "[tileOffset], " + paramName + "CurrentCopyLength);\n";
    copyInCode += "        " + queueName + ".EnQue(" + paramName + "Local);\n";

    computeDeQueCode += "        AscendC::LocalTensor<" + kernelMappedType + "> " + paramName + "Local = " + queueName + ".DeQue<" + kernelMappedType + ">();\n";
    computeFreeCode += "        " + queueName + ".FreeTensor(" + paramName + "Local);\n";
}

static void EmitAlignedOutputTransfer(
    const std::string& kernelMappedType,
    const std::string& paramName,
    const std::string& gmName,
    const std::string& queueName,
    const std::string&,
    const std::string&,
    std::string& tQueDecl,
    std::string& initCode,
    std::string& computeAllocCode,
    std::string& computeEnQueCode,
    std::string& copyOutCode) {
    (void)tQueDecl;
    initCode += "        " + gmName + ".SetGlobalBuffer((__gm__ " + kernelMappedType + " *)" + paramName + " + blockStart, blockLength);\n";
    initCode += "        pipe.InitBuffer(" + queueName + ", BUFFER_NUM, tileLength * sizeof(" + kernelMappedType + "));\n";

    computeAllocCode += "        AscendC::LocalTensor<" + kernelMappedType + "> " + paramName + "Local = " + queueName + ".AllocTensor<" + kernelMappedType + ">();\n";
    computeEnQueCode += "        " + queueName + ".EnQue<" + kernelMappedType + ">(" + paramName + "Local);\n";

    copyOutCode += "        AscendC::LocalTensor<" + kernelMappedType + "> " + paramName + "Local = " + queueName + ".DeQue<" + kernelMappedType + ">();\n";
    copyOutCode += "        AscendC::DataCopy(" + gmName + "[tileOffset], " + paramName + "Local, currentTileLength);\n";
    copyOutCode += "        " + queueName + ".FreeTensor(" + paramName + "Local);\n";
}

static VectorKernelSkeletonFragments EmitKernelSkeletonForFamily(
    const std::string& initCode,
    const std::string& copyInCode,
    const std::string& computeCode,
    const std::string& copyOutCode,
    const std::string& tQueDecl) {
    VectorKernelSkeletonFragments fragments{initCode, copyInCode, computeCode,
                                            copyOutCode, tQueDecl};
    return fragments;
}

struct MulticardAnalysis {
    bool enabled = false;
    std::string rankDimExpr;
    std::string rankDimEnvName;
    std::string shardDim;
    std::string shardAxis;
    std::string logicalLengthExpr;
    std::string alignedLengthExpr;
    std::string strategy;
    std::string replicatedParams;
    std::string shardedParams;
    int readHaloLeft = 0;
    int readHaloRight = 0;
    bool temporalLoopCarried = false;
    bool inputReplicated = false;
    bool outputSharded = true;
    bool persistentWorkers = false;
};

// 单算子代码生成器 (Vector 向量计算) 


// ===== Vector lowering =====
class OperatorGenerator {
private:
    struct RecordFieldSpec {
        RecordFieldLayoutEntry layout;
    };

    struct RecordGroupPlan {
        std::string logicalName;
        std::string paramType;
        int rw = 0;
        RuntimeParamBindingInfo runtimeInfo;
        int leftHalo = 0;
        int rightHalo = 0;
        std::string rootVector;
        std::vector<RecordFieldSpec> fields;
    };

    struct LogicalCountPlan {
        std::string expr;
        bool sourceKnownNonEmpty = false;
        std::string requiredOutputSize;
    };

    struct HostParamPlan {
        std::string name;
        std::string logicalName;
        std::string paramType;
        std::string basicType;
        std::string mappedType;
        std::string kernelType;
        int rw = 0;
        bool useFloatBitBridge = false;
        bool isBroadcastScalar = false;
        int leftHalo = 0;
        int rightHalo = 0;
        bool isRecordField = false;
        bool producerResident = false;
        std::string recordFieldName;
        std::string rootSourceVar;
        dacppTranslator::ascendc::FlowBufferId bufferId =
            dacppTranslator::ascendc::kInvalidFlowBufferId;
    };

    struct LinearMulticardPlan {
        bool recognized = false;
        std::string inputName;
        std::string weightName;
        std::string biasName;
        std::string outputName;
        int inFeatures = 0;
        int outFeatures = 0;
        bool hasBias = false;
    };

    Expression* expr;
    int index;
    clang::Rewriter* rewriter;
    const TranslationPolicy& policy;
    std::string sourceText;

    std::string kernelName;
    std::string generatedKernelFileName;
    std::string coreNumStr;
    std::string hostPreludeCode;
    dacppTranslator::ascendc::HostNameAllocator hostNames;
    std::string temporalWrapperName;
    HostPreludeCollection hostPreludeCollection;

    std::string commonCode;
    std::string cpuAllocs, cpuMemcpysIn, cpuArgs, cpuMemcpysOut, cpuFrees;
    std::string npuPtrs, npuAllocs, npuMemcpysIn, npuArgs, npuMemcpysOut, npuFrees;
    std::string hostWriteBacks;

    std::string initCode, tQueDecl, copyInCode, computeDeQueCode, computeAllocCode, computeEnQueCode, computeFreeCode, copyOutCode, kernelParamsDecl, kernelCallArgs;
    std::string doParams, cpuParams;

    std::vector<std::string> inLocalVars;
    std::vector<std::string> outLocalVars;
    std::vector<std::string> temporalTensorInputs;
    std::vector<std::string> temporalTensorOutputs;
    std::vector<std::string> temporalScalarInputs;
    std::vector<std::string> resolvedShellCallArgs;
    std::vector<HostParamPlan> hostParamPlans;
    std::vector<std::string> hostKernelLengthBuffers;
    std::unordered_set<std::string> emittedOutputResizeKeys;
    std::vector<dacppTranslator::ascendc_mlir::CalcParamBinding> calcParamBindings;
    std::vector<std::string> shellParamOrder;
    dacppTranslator::ascendc_mlir::CalcShellSummary shellSummary;
    PointwiseScalarLoopSchedule pointwiseScalarLoopSchedule;
    LoopCarriedVectorSchedule loopCarriedVectorSchedule;
    LinearMulticardPlan linearMulticardPlan;
    dacppTranslator::ascendc_mlir::CalcSemanticFamily dominantFamily =
        dacppTranslator::ascendc_mlir::CalcSemanticFamily::PointwiseFamily;
    dacppTranslator::ascendc_mlir::AscendCTransferPolicy transferPolicy =
        dacppTranslator::ascendc_mlir::AscendCTransferPolicy::PointwiseDirect;
    dacppTranslator::ascendc::VectorKernelTilingPlan vectorTilingPlan;
    bool hasVectorTilingPlan = false;

public:
    OperatorGenerator(Expression* e,
                      int idx,
                      clang::Rewriter* rw,
                      const std::string& srcText,
                      const TranslationPolicy& policy,
                      std::string kernelFileName)
        : expr(e),
          index(idx),
          rewriter(rw),
          policy(policy),
          sourceText(srcText),
          generatedKernelFileName(std::move(kernelFileName)) {
        kernelName = "vector_kernel_" + std::to_string(index);
        coreNumStr = std::to_string(policy.defaultCoreNum);
        const clang::BinaryOperator* dacExpr = expr ? expr->getDacExpr() : nullptr;
        const clang::FunctionDecl* owner = FindEnclosingFunction(dacExpr);
        if (owner && gRuntimeBindingTUDecl) {
            for (const std::string& name :
                 dacppTranslator::ascendc::CollectHostVisibleIdentifiers(
                     *owner, *gRuntimeBindingTUDecl)) {
                hostNames.reserve(name);
            }
        }
        for (const std::string& name :
             {gVectorRuntimeNames.session, gVectorRuntimeNames.rankCount,
              gVectorRuntimeNames.routeObject, gVectorRuntimeNames.device,
              gVectorRuntimeNames.stream}) {
            hostNames.reserve(name);
        }
        hostNames.pushScope("expr." + std::to_string(index));
        const std::string semanticName = index == 0
            ? "temporalStep"
            : "temporalStep" + std::to_string(index + 1);
        temporalWrapperName = hostNames.get(
            dacppTranslator::ascendc::HostNameRole::HelperFunction,
            semanticName).text;
        if (!hostNames.ok() && gHostNameError.empty()) {
            gHostNameError = hostNames.error();
        }
    }

    const std::string& GetTemporalWrapperName() const {
        return temporalWrapperName;
    }

private:
    const std::string& AppName() const {
        return gVectorRuntimeNames.session;
    }

    const std::string& RouteName() const {
        return gVectorRuntimeNames.routeObject;
    }

    const std::string& DeviceName() const {
        return gVectorRuntimeNames.device;
    }

    const std::string& StreamName() const {
        return gVectorRuntimeNames.stream;
    }

    static std::string HostBufferName(const std::string& name) {
        return name + "Buf";
    }

    static std::string LogicalBufferKey(const HostParamPlan& plan) {
        if (plan.isRecordField && !plan.rootSourceVar.empty() &&
            !plan.recordFieldName.empty()) {
            return "record:" + plan.rootSourceVar + ":" + plan.recordFieldName;
        }
        return "vector:" + plan.name;
    }

public:
    void AssignRuntimeBufferIds() {
        for (HostParamPlan& plan : hostParamPlans) {
            const std::string key = LogicalBufferKey(plan);
            plan.producerResident =
                (plan.rw == 0 || plan.rw == 2) &&
                gRuntimeBufferIds.wasProduced(
                    dacppTranslator::ascendc::FlowRouteKind::Vector, key);
            plan.bufferId = gRuntimeBufferIds.get(
                dacppTranslator::ascendc::FlowRouteKind::Vector,
                key);
        }
        for (const HostParamPlan& plan : hostParamPlans) {
            if (plan.rw == 1 || plan.rw == 2) {
                gRuntimeBufferIds.markProduced(
                    dacppTranslator::ascendc::FlowRouteKind::Vector,
                    LogicalBufferKey(plan));
            }
        }
    }

private:
    static uint32_t KernelElementBytes(const std::string& typeName) {
        const std::string type = TrimCopy(typeName);
        if (type.find("_Complex") != std::string::npos || type == "double" ||
            type == "int64_t" || type == "uint64_t" || type == "long" ||
            type == "unsigned long") {
            return 8;
        }
        if (type == "half" || type == "float16" || type == "int16_t" ||
            type == "uint16_t" || type == "short" || type == "unsigned short") {
            return 2;
        }
        if (type == "int8_t" || type == "uint8_t" || type == "char" ||
            type == "unsigned char" || type == "bool") {
            return 1;
        }
        return 4;
    }

    std::string BuildPerRankTilingCode(const std::string& exprId,
                                       const std::string& indent) const {
        const uint32_t align = hasVectorTilingPlan
            ? vectorTilingPlan.tileAlignElements : 1;
        const uint32_t maxTile = hasVectorTilingPlan
            ? vectorTilingPlan.maxTileElements : 1;
        std::string code;
        code += indent + "const dacpp_host::Tile tile = "
                "dacpp_host::makeTile(" + AppName() +
                ", r.len, blockDim" + exprId + ", " +
                std::to_string(align) + ", " + std::to_string(maxTile) +
                ");\n";
        return code;
    }

    std::string BufferCopyInCode(const std::string& bufferName,
                                 const std::string& indent = "    ") const {
        return indent + DeviceName() + ".copyIn(" + bufferName + ");\n";
    }

    std::string BufferCopyOutCode(const std::string& bufferName,
                                  const std::string& indent = "    ") const {
        return indent + DeviceName() + ".copyOut(" + bufferName + ");\n";
    }

    void AppendConcreteBufferDeclaration(const std::string& bufferName,
                                         const std::string& mappedType,
                                         const std::string& logicalCountExpr,
                                         int leftHalo,
                                         int rightHalo) {
        commonCode += "    dacpp_host::HostBuffer<" + mappedType + "> " +
                      bufferName + "(" + AppName() + ", " + logicalCountExpr +
                      ", static_cast<size_t>(" + std::to_string(leftHalo) +
                      "), static_cast<size_t>(" + std::to_string(rightHalo) +
                      "), static_cast<size_t>((" +
                      std::to_string(policy.vectorComputeAlignBytes) +
                      " + sizeof(" + mappedType + ") - 1) / sizeof(" +
                      mappedType + ")));\n";
    }

    static std::string BuildDirectVectorWriteBack(
        const std::string& rootSourceVar,
        const std::string& bufferName,
        const std::string& writeBackOffsetExpr,
        const std::string& mappedType,
        const std::string& logicalName,
        bool useFloatBitBridge,
        bool sourceElemIsComplex,
        bool sourceElemIsPair,
        const std::string& bridgeSoAImagName,
        const std::string& bridgeSoASecondName,
        bool sourceMatchesMappedType,
        const std::string& indent = "    ") {
        const std::string offsetName = logicalName + "Offset";
        const std::string elemTypeAlias = logicalName + "ElemType";
        std::string code;
        code += indent + "const size_t " + offsetName + " = " +
                writeBackOffsetExpr + ";\n";
        if (sourceElemIsComplex) {
            code += indent + "using " + elemTypeAlias +
                    " = typename std::remove_reference<decltype(" +
                    rootSourceVar + "[0])>::type;\n";
            code += indent + "for (size_t i = 0; i < " + bufferName +
                    ".logicalElems; ++i) {\n";
            code += indent + "    using ElemValueType = typename " + elemTypeAlias +
                    "::value_type;\n";
            code += indent + "    " + rootSourceVar + "[" + offsetName +
                    " + i] = " + elemTypeAlias + "(" +
                    BuildFromKernelCastExpr(bufferName + ".data[i]", "ElemValueType",
                                            useFloatBitBridge) + ", " +
                    BuildFromKernelCastExpr(bridgeSoAImagName + "[i]", "ElemValueType",
                                            useFloatBitBridge) + ");\n";
            code += indent + "}\n";
        } else if (sourceElemIsPair) {
            code += indent + "using " + elemTypeAlias +
                    " = typename std::remove_reference<decltype(" +
                    rootSourceVar + "[0])>::type;\n";
            code += indent + "for (size_t i = 0; i < " + bufferName +
                    ".logicalElems; ++i) {\n";
            code += indent + "    " + rootSourceVar + "[" + offsetName +
                    " + i] = " + elemTypeAlias + "(" +
                    BuildFromKernelCastExpr(bufferName + ".data[i]",
                                            "typename " + elemTypeAlias + "::first_type",
                                            useFloatBitBridge) + ", " +
                    BuildFromKernelCastExpr(bridgeSoASecondName + "[i]",
                                            "typename " + elemTypeAlias + "::second_type",
                                            useFloatBitBridge) + ");\n";
            code += indent + "}\n";
        } else if (sourceMatchesMappedType && !useFloatBitBridge) {
            code += indent + "std::memcpy(" + rootSourceVar + ".data() + " +
                    offsetName + ", " + bufferName + ".data.data(), " +
                    bufferName + ".logicalElems * sizeof(" + mappedType + "));\n";
        } else {
            code += indent + "using " + elemTypeAlias +
                    " = typename std::remove_reference<decltype(" +
                    rootSourceVar + "[0])>::type;\n";
            code += indent + "for (size_t i = 0; i < " + bufferName +
                    ".logicalElems; ++i) {\n";
            code += indent + "    " + rootSourceVar + "[" + offsetName +
                    " + i] = " +
                    BuildFromKernelCastExpr(bufferName + ".data[i]", elemTypeAlias,
                                            useFloatBitBridge) + ";\n";
            code += indent + "}\n";
        }
        return code;
    }

    std::string BuildPriorLogicalFallbackExpr() const {
        if (hostKernelLengthBuffers.empty()) {
            return "static_cast<size_t>(0)";
        }
        std::string expr = "std::max({static_cast<size_t>(0)";
        for (size_t i = 0; i < hostKernelLengthBuffers.size(); ++i) {
            expr += ", " + hostKernelLengthBuffers[i] + ".logicalElems";
        }
        expr += "})";
        return expr;
    }

    std::string BuildHostWriteBackOnlyCode() const {
        std::stringstream ss(npuMemcpysOut);
        std::string line;
        std::string code;
        while (std::getline(ss, line)) {
            if (line.find("aclrtMemcpy(") != std::string::npos ||
                line.find(DeviceName() + ".copyOut(") != std::string::npos ||
                (line.find("std::memcpy(") != std::string::npos &&
                 line.find(".host") != std::string::npos)) {
                continue;
            }
            code += line + "\n";
        }
        return code;
    }

    bool UsesPointwiseScalarLoopSchedule() const {
        return pointwiseScalarLoopSchedule.recognized &&
               dominantFamily == dacppTranslator::ascendc_mlir::CalcSemanticFamily::PointwiseFamily &&
               transferPolicy == dacppTranslator::ascendc_mlir::AscendCTransferPolicy::PointwiseDirect;
    }

    bool UsesLoopCarriedVectorSchedule() const {
        return loopCarriedVectorSchedule.recognized &&
               loopCarriedVectorSchedule.absorbsOwningFor &&
               dominantFamily == dacppTranslator::ascendc_mlir::CalcSemanticFamily::SpatialStencilFamily &&
               transferPolicy == dacppTranslator::ascendc_mlir::AscendCTransferPolicy::SpatialWindow;
    }

    bool UsesLinearOutputShardSchedule() const {
        return linearMulticardPlan.recognized &&
               dominantFamily == dacppTranslator::ascendc_mlir::CalcSemanticFamily::LinearReductionFamily;
    }

    std::string GetEmittedKernelName() const {
        return policy.enableMulticard ? (kernelName + "_n") : kernelName;
    }

    std::string GetLaunchWrapperName() const {
        return GetEmittedKernelName() + "_do";
    }

    const HostParamPlan* FindHostParamPlanByName(const std::string& paramName) const {
        for (const auto& plan : hostParamPlans) {
            if (plan.name == paramName) {
                return &plan;
            }
        }
        return nullptr;
    }

    std::string BuildUntouchedRecordOutputsCode(
        const std::unordered_set<std::string>& writtenOutputTensors) const {
        std::string code;
        for (const HostParamPlan& output : hostParamPlans) {
            if (!output.isRecordField ||
                !(output.rw == 1 || output.rw == 2) ||
                output.isBroadcastScalar ||
                writtenOutputTensors.count(output.name) > 0) {
                continue;
            }

            std::vector<const HostParamPlan*> matchingInputs;
            for (const HostParamPlan& input : hostParamPlans) {
                if (!input.isRecordField ||
                    !(input.rw == 0 || input.rw == 2) ||
                    input.name == output.name ||
                    input.recordFieldName != output.recordFieldName ||
                    input.paramType != output.paramType ||
                    input.kernelType != output.kernelType) {
                    continue;
                }
                matchingInputs.push_back(&input);
            }

            if (matchingInputs.size() != 1) {
                std::string message =
                    "record output field '" + output.logicalName + "." +
                    output.recordFieldName + "' (physical parameter '" +
                    output.name + "') is not assigned by calc and has " +
                    std::to_string(matchingInputs.size()) +
                    " compatible input source(s); assign the field explicitly";
                if (!matchingInputs.empty()) {
                    message += " (candidates:";
                    for (const HostParamPlan* input : matchingInputs) {
                        message += " " + input->logicalName + "." +
                                   input->recordFieldName;
                    }
                    message += ")";
                }
                throw std::runtime_error(message);
            }

            const HostParamPlan& input = *matchingInputs.front();
            code += "        AscendC::Adds(" + output.name + "Local, " +
                    input.name + "Local, static_cast<decltype(+" + output.name +
                    "Local.GetValue(0))>(0), currentTileLength);\n";
        }
        return code;
    }

    bool IsMulticardBroadcastScalarParam(const HostParamPlan& plan) const {
        return plan.isBroadcastScalar ||
               (UsesPointwiseScalarLoopSchedule() &&
                plan.name == pointwiseScalarLoopSchedule.scalarParamName);
    }

    const HostTensorAliasInfo* FindHostTensorAliasByName(const std::string& aliasName) const {
        auto it = hostPreludeCollection.tensorAliases.find(
            BuildHostTensorAliasKey(
                pointwiseScalarLoopSchedule.ownerFunctionName,
                TrimCopy(aliasName)));
        if (it == hostPreludeCollection.tensorAliases.end()) {
            return nullptr;
        }
        return &it->second;
    }

    std::string MaterializePointwiseLoopExpr(const std::string& rawExpr,
                                             const std::string& scalarCurrentVar,
                                             const std::string& fallbackExpr) const {
        std::string exprText = TrimCopy(rawExpr);
        if (exprText.empty()) {
            return fallbackExpr;
        }

        auto ReplaceScalarAccess = [&](const std::string& baseName) {
            if (baseName.empty()) {
                return;
            }
            const std::string escapedBase = EscapeRegexText(baseName);
            const std::regex accessRegex("\\b" + escapedBase + R"(\s*\[\s*0\s*\])");
            exprText = std::regex_replace(exprText, accessRegex, scalarCurrentVar);
        };

        ReplaceScalarAccess(pointwiseScalarLoopSchedule.scalarArgName);
        ReplaceScalarAccess(pointwiseScalarLoopSchedule.scalarSourceVar);
        exprText = TrimCopy(exprText);
        return exprText.empty() ? fallbackExpr : exprText;
    }

    std::string BuildPointwiseScalarLoopSnapshotCopyCode(
        const std::string& outputBufferName,
        const std::string& scalarCurrentVar) const {
        const HostTensorAliasInfo* snapshotTensorAlias =
            FindHostTensorAliasByName(pointwiseScalarLoopSchedule.snapshotTensorName);
        const HostTensorAliasInfo* outputAlias =
            FindHostTensorAliasByName(pointwiseScalarLoopSchedule.outputArgName);
        if (!snapshotTensorAlias ||
            !outputAlias ||
            !snapshotTensorAlias->valid ||
            !outputAlias->valid ||
            pointwiseScalarLoopSchedule.snapshotIndexExpr.empty()) {
            return "";
        }

        HostTensorAliasInfo snapshotAliasInfo = *snapshotTensorAlias;
        RewriteTensorScalarReadsInAliasInfo(
            snapshotAliasInfo,
            pointwiseScalarLoopSchedule.ownerFunctionName,
            hostPreludeCollection.tensorAliases);

        const std::string snapshotIndexExpr = MaterializePointwiseLoopExpr(
            pointwiseScalarLoopSchedule.snapshotIndexExpr,
            scalarCurrentVar,
            "0");
        if (snapshotAliasInfo.rank > 0) {
            HostTensorAliasInfo indexedSnapshotInfo;
            if (TryApplyTensorIndexText(
                    snapshotAliasInfo,
                    snapshotIndexExpr,
                    indexedSnapshotInfo) &&
                indexedSnapshotInfo.valid) {
                snapshotAliasInfo = indexedSnapshotInfo;
            }
        }

        HostTensorAliasInfo outputBufferAliasInfo = *outputAlias;
        outputBufferAliasInfo.rootContainerName = outputBufferName + ".data";
        outputBufferAliasInfo.rootDataExpr = outputBufferName + ".data.data()";
        RewriteTensorScalarReadsInAliasInfo(
            outputBufferAliasInfo,
            pointwiseScalarLoopSchedule.ownerFunctionName,
            hostPreludeCollection.tensorAliases);

        return EmitTensorAliasAssignmentFromInfos(
            snapshotAliasInfo,
            &outputBufferAliasInfo,
            outputBufferName + ".data",
            static_cast<size_t>(index) * 1000 + 100);
    }

    MulticardAnalysis AnalyzeMulticardPlan() const {
        MulticardAnalysis analysis;
        if (!policy.enableMulticard) {
            return analysis;
        }
        analysis.enabled = true;
        analysis.rankDimExpr = std::to_string(policy.requestedRankDim);
        analysis.rankDimEnvName = policy.rankDimEnvName.empty()
                                      ? "DACPP_ASCENDC_RANK_DIM"
                                      : policy.rankDimEnvName;
        analysis.logicalLengthExpr = "kernelLength" + std::to_string(index);
        analysis.shardDim = "1";
        analysis.shardAxis = "1";
        analysis.temporalLoopCarried =
            transferPolicy == dacppTranslator::ascendc_mlir::AscendCTransferPolicy::TemporalLoopCarried ||
            dominantFamily == dacppTranslator::ascendc_mlir::CalcSemanticFamily::TemporalStencilFamily;
        for (const HostParamPlan& plan : hostParamPlans) {
            if (IsMulticardBroadcastScalarParam(plan)) {
                continue;
            }
            if (plan.rw == 0 || plan.rw == 2) {
                analysis.readHaloLeft = std::max(analysis.readHaloLeft, plan.leftHalo);
                analysis.readHaloRight = std::max(analysis.readHaloRight, plan.rightHalo);
            }
        }
        const bool hasRecordFieldParam =
            std::any_of(hostParamPlans.begin(), hostParamPlans.end(),
                        [](const HostParamPlan& plan) {
                            return plan.isRecordField;
                        });
        analysis.alignedLengthExpr = "rank.logicalLength";
        if (UsesLoopCarriedVectorSchedule()) {
            analysis.logicalLengthExpr = "workLen" + std::to_string(index);
        }
        if (dominantFamily ==
                   dacppTranslator::ascendc_mlir::CalcSemanticFamily::LinearReductionFamily) {
            analysis.strategy = "linear-output-shard";
            analysis.inputReplicated = true;
        } else if (UsesPointwiseScalarLoopSchedule()) {
            analysis.strategy = "pointwise-scalar-loop";
            analysis.persistentWorkers = true;
        } else if (hasRecordFieldParam &&
                   analysis.readHaloLeft == 0 &&
                   analysis.readHaloRight == 0 &&
                   !analysis.temporalLoopCarried &&
                   !UsesLoopCarriedVectorSchedule()) {
            analysis.strategy = "record-field-pointwise";
        } else if (analysis.readHaloRight == 1 && analysis.readHaloLeft == 0) {
            if (analysis.temporalLoopCarried || UsesLoopCarriedVectorSchedule()) {
                analysis.strategy = "temporal-stencil-persistent";
                analysis.persistentWorkers = true;
            } else {
                analysis.strategy = "stencil-right-halo1";
            }
        } else if (analysis.readHaloRight == 2 && analysis.readHaloLeft == 0) {
            analysis.strategy = "stencil-right-halo2";
        } else if (analysis.temporalLoopCarried || UsesLoopCarriedVectorSchedule()) {
            analysis.strategy = "temporal-stencil-persistent";
            analysis.persistentWorkers = true;
        } else if (analysis.readHaloLeft != 0 || analysis.readHaloRight != 0) {
            analysis.strategy = "stencil-halo";
        } else {
            analysis.strategy = "pointwise-1d";
        }
        std::vector<std::string> replicated;
        std::vector<std::string> sharded;
        for (const HostParamPlan& plan : hostParamPlans) {
            if (IsMulticardBroadcastScalarParam(plan)) {
                replicated.push_back(plan.name);
            } else if (analysis.strategy == "linear-output-shard" &&
                       plan.name == linearMulticardPlan.inputName) {
                replicated.push_back(plan.name);
            } else {
                sharded.push_back(plan.name);
            }
        }
        analysis.replicatedParams = BuildHostBufferList(replicated);
        analysis.shardedParams = BuildHostBufferList(sharded);
        return analysis;
    }

    static std::string BuildHostBufferList(const std::vector<std::string>& names) {
        std::string out;
        for (const std::string& name : names) {
            if (!out.empty()) {
                out += ", ";
            }
            out += name;
        }
        return out;
    }

    static std::string BuildVectorElemTypeExpr(const std::string& vectorExpr) {
        return "typename std::remove_reference<decltype(" + vectorExpr +
               "[0])>::type";
    }

    static std::string BuildConstRefParam(const std::string& typeExpr,
                                          const std::string& name) {
        return typeExpr + " const& " + name;
    }

    static std::string BuildRefParam(const std::string& typeExpr,
                                     const std::string& name) {
        return typeExpr + "& " + name;
    }

    LogicalCountPlan BuildLogicalCountExpr(
        const RuntimeParamBindingInfo& runtimeInfo,
        bool useBroadcastScalar,
        bool hasRootStdVector,
        const std::string& rootSourceVar,
        int rw) const {
        LogicalCountPlan result;
        if (useBroadcastScalar && (rw == 0 || rw == 2)) {
            result.expr = "static_cast<size_t>(1)";
            result.sourceKnownNonEmpty = true;
        } else if (hasRootStdVector) {
            result.expr = rootSourceVar + ".size()";
            result.sourceKnownNonEmpty = runtimeInfo.hasShapeLength;
        } else if (runtimeInfo.hasShapeLength) {
            result.expr =
                "static_cast<size_t>(" + runtimeInfo.shapeLength + ")";
            result.sourceKnownNonEmpty = true;
        } else if (runtimeInfo.hasScalarSourceExpr &&
                   (rw == 0 || rw == 2)) {
            result.expr = "static_cast<size_t>(1)";
            result.sourceKnownNonEmpty = true;
        } else {
            result.expr = BuildPriorLogicalFallbackExpr();
        }
        result.requiredOutputSize = runtimeInfo.hasShapeLength
            ? "static_cast<size_t>(" + runtimeInfo.shapeLength + ")"
            : result.expr;
        return result;
    }

    void EmitOutputResizeIfNeeded(const std::string& rootSourceVar,
                                  int rw,
                                  const LogicalCountPlan& count) {
        if (rootSourceVar.empty() || !(rw == 1 || rw == 2)) {
            return;
        }
        const std::string& required = count.requiredOutputSize;
        const std::string key = rootSourceVar + "\n" + required;
        if (!emittedOutputResizeKeys.insert(key).second) {
            return;
        }
        commonCode += "    dacpp_host::ensureSize(" + rootSourceVar + ", " +
                      required + ");\n";
    }

    void EmitRecordFieldRuntime(const std::string& logicalName,
                                const std::string& paramType,
                                int rw,
                                const RuntimeParamBindingInfo& runtimeInfo,
                                int paramLeftHalo,
                                int paramRightHalo,
                                const std::string& rootSourceVar,
                                const RecordFieldLayoutEntry& fieldInfo) {
        const std::string physicalName = BuildRecordFieldPhysicalName(logicalName, fieldInfo.name);
        const std::string originalType = fieldInfo.typeName;
        const std::string mappedType = MapToStdType(originalType);
        const std::string kernelMappedType = MapToKernelType(originalType);
        const bool useFloatBitBridge =
            (policy.floatLikeTypes.count(originalType) > 0 &&
             mappedType == policy.hostFloatStorageType);

        auto BuildToKernelExpr = [&](const std::string& valueExpr) {
            return BuildToKernelCastExpr(valueExpr, mappedType, useFloatBitBridge);
        };
        auto BuildFromKernelExpr = [&](const std::string& valueExpr, const std::string& dstTypeExpr) {
            return BuildFromKernelCastExpr(valueExpr, dstTypeExpr, useFloatBitBridge);
        };

        shellParamOrder.push_back(physicalName);
        if (rw == 0 || rw == 2) {
            inLocalVars.push_back(physicalName);
            temporalTensorInputs.push_back(physicalName);
        }
        if (rw == 1 || rw == 2) {
            outLocalVars.push_back(physicalName);
            temporalTensorOutputs.push_back(physicalName);
        }

        HostParamPlan paramPlan;
        paramPlan.name = physicalName;
        paramPlan.logicalName = logicalName;
        paramPlan.paramType = paramType;
        paramPlan.basicType = originalType;
        paramPlan.mappedType = mappedType;
        paramPlan.kernelType = kernelMappedType;
        paramPlan.rw = rw;
        paramPlan.useFloatBitBridge = useFloatBitBridge;
        paramPlan.leftHalo = paramLeftHalo;
        paramPlan.rightHalo = paramRightHalo;
        paramPlan.isRecordField = true;
        paramPlan.recordFieldName = fieldInfo.name;
        paramPlan.rootSourceVar = rootSourceVar;
        hostParamPlans.push_back(paramPlan);

        const std::string bufferName = HostBufferName(physicalName);
        const bool hasRootStdVector =
            runtimeInfo.bindingResolved &&
            runtimeInfo.bindingKind == RuntimeBindingNodeKind::StdVector &&
            !rootSourceVar.empty();
        const bool needsInitialBridge = hasRootStdVector && (rw == 0 || rw == 2);
        const LogicalCountPlan logicalCount =
            BuildLogicalCountExpr(runtimeInfo, /*useBroadcastScalar=*/false,
                                  hasRootStdVector, rootSourceVar, rw);

        EmitOutputResizeIfNeeded(rootSourceVar, rw, logicalCount);
        AppendConcreteBufferDeclaration(
            bufferName, mappedType, logicalCount.expr,
            paramLeftHalo, paramRightHalo);
        if (needsInitialBridge) {
            commonCode += "    for (size_t i = 0; i < std::min(" + bufferName +
                          ".logicalElems, " + rootSourceVar + ".size()); ++i) {\n";
            commonCode += "        " + bufferName + ".data[i] = " +
                          BuildToKernelExpr(rootSourceVar + "[i]." + fieldInfo.name) +
                          ";\n";
            commonCode += "    }\n";
        }
        hostKernelLengthBuffers.push_back(bufferName);

        const std::string cpuGmName = "cpuGm_" + physicalName;
        cpuAllocs += "    uint8_t *" + cpuGmName + " = (uint8_t *)AscendC::GmAlloc(" + bufferName + ".bytes());\n";
        if (needsInitialBridge) {
            cpuMemcpysIn += "    memcpy(" + cpuGmName + ", " + bufferName + ".data.data(), " + bufferName + ".bytes());\n";
        }
        if (rw == 1 || rw == 2) {
            cpuMemcpysOut += "    memcpy(" + bufferName + ".data.data(), " + cpuGmName + ", " + bufferName + ".bytes());\n";
            {
                const std::string elemTypeAlias = physicalName + "FieldType";
                const std::string elemParam =
                    BuildRefParam(BuildVectorElemTypeExpr(rootSourceVar), "elem");
                std::string writeBack;
                writeBack += "    for (size_t i = 0; i < " + bufferName +
                             ".logicalElems; ++i) {\n";
                writeBack += "        auto &elem = " + rootSourceVar + "[i];\n";
                writeBack += "        using " + elemTypeAlias + " = typename std::remove_reference<decltype(elem." + fieldInfo.name + ")>::type;\n";
                writeBack += "        elem." + fieldInfo.name + " = " + BuildFromKernelExpr(bufferName + ".data[i]", elemTypeAlias) + ";\n";
                writeBack += "    }\n";
                cpuMemcpysOut += writeBack;
                hostWriteBacks += writeBack;
            }
        }
        cpuFrees += "    AscendC::GmFree((void *)" + cpuGmName + ");\n";

        if (!cpuArgs.empty()) cpuArgs += ", ";
        cpuArgs += cpuGmName;

        npuAllocs += "    " + DeviceName() + ".allocate(" + bufferName + ");\n";
        if (needsInitialBridge) {
            npuMemcpysIn += BufferCopyInCode(bufferName);
        }
        if (rw == 1 || rw == 2) {
            npuMemcpysOut += BufferCopyOutCode(bufferName);
            {
                const std::string elemTypeAlias = physicalName + "FieldType";
                std::string writeBack;
                writeBack += "    for (size_t i = 0; i < " + bufferName +
                             ".logicalElems; ++i) {\n";
                writeBack += "        auto &elem = " + rootSourceVar + "[i];\n";
                writeBack += "        using " + elemTypeAlias + " = typename std::remove_reference<decltype(elem." + fieldInfo.name + ")>::type;\n";
                writeBack += "        elem." + fieldInfo.name + " = " + BuildFromKernelExpr(bufferName + ".data[i]", elemTypeAlias) + ";\n";
                writeBack += "    }\n";
                npuMemcpysOut += writeBack;
            }
        }
        npuFrees += "    " + DeviceName() + ".release(" + bufferName + ");\n";

        if (!npuArgs.empty()) npuArgs += ", ";
        npuArgs += bufferName + ".device";

        if (!doParams.empty()) {
            doParams += ", ";
            cpuParams += ", ";
            kernelParamsDecl += ", ";
            kernelCallArgs += ", ";
        }
        doParams += "uint8_t *" + physicalName;
        cpuParams += "GM_ADDR " + physicalName;
        kernelParamsDecl += "GM_ADDR " + physicalName;
        kernelCallArgs += physicalName;

        const std::string NameGm = physicalName + "Gm";
        const std::string QueueName = (rw == 0 || rw == 2) ? "inQueue" + physicalName : "outQueue" + physicalName;
        const std::string leftHaloName = physicalName + "LeftHalo";
        const std::string rightHaloName = physicalName + "RightHalo";
        const std::string validLengthName = physicalName + "ValidLength";
        const std::string copyLengthName = physicalName + "CopyLength";
        const std::string gmStartName = physicalName + "GmStart";
        const std::string gmLengthName = physicalName + "GmLength";
        const std::string writeLengthName = physicalName + "WriteLength";
        const std::string inputMinCopyElemsName = physicalName + "InputMinCopyElems";
        const std::string outputSpanName = physicalName + "OutputSpan";

        tQueDecl += "    AscendC::GlobalTensor<" + kernelMappedType + "> " + NameGm + ";\n";
        tQueDecl += "    AscendC::TQue<AscendC::QuePosition::" + std::string((rw == 0 || rw == 2) ? "VECIN" : "VECOUT") + ", BUFFER_NUM> " + QueueName + ";\n";

        if (rw == 0 || rw == 2) {
            EmitAlignedInputTransfer(
                kernelMappedType,
                physicalName,
                NameGm,
                QueueName,
                validLengthName,
                copyLengthName,
                gmLengthName,
                tQueDecl,
                initCode,
                copyInCode,
                computeDeQueCode,
                computeFreeCode);
        }
        if (rw == 1 || rw == 2) {
            EmitAlignedOutputTransfer(
                kernelMappedType,
                physicalName,
                NameGm,
                QueueName,
                writeLengthName,
                outputSpanName,
                tQueDecl,
                initCode,
                computeAllocCode,
                computeEnQueCode,
                copyOutCode);
        }
    }

    static void AppendAoSFieldParam(RecordGroupPlan& group,
                                    const RecordFieldLayoutEntry& fieldInfo) {
        RecordFieldSpec field;
        field.layout = fieldInfo;
        group.fields.push_back(std::move(field));
    }

    void EmitRecordGroupRuntime(const RecordGroupPlan& group) {
        for (const RecordFieldSpec& field : group.fields) {
            EmitRecordFieldRuntime(
                group.logicalName, group.paramType, group.rw,
                group.runtimeInfo, group.leftHalo, group.rightHalo,
                group.rootVector, field.layout);
        }
    }

public:

    void ParseParameters() {
        Shell* shell = expr->getShell();
        Calc* calc = expr->getCalc();
        if (!shell) {
            return;
        }

        resolvedShellCallArgs.clear();
        hostParamPlans.clear();
        calcParamBindings.clear();
        hostPreludeCollection = HostPreludeCollection();
        hostPreludeCode.clear();

        shellSummary = dacppTranslator::ascendc_mlir::BuildCalcShellSummary(shell);

        std::vector<std::string> shellCallArgs;
        const clang::BinaryOperator* dacExpr = expr->getDacExpr();
        gRuntimeBindingCurrentDacExprLoc = dacExpr ? dacExpr->getExprLoc() : clang::SourceLocation();
        gRuntimeBindingCurrentOwnerFunction = FindEnclosingFunction(dacExpr);
        const clang::Stmt* bindingScopeRoot =
            gRuntimeBindingCurrentOwnerFunction ? gRuntimeBindingCurrentOwnerFunction->getBody() : nullptr;

        shellCallArgs = ResolveShellCallArgumentsFromAst(dacExpr, shell);
        resolvedShellCallArgs = shellCallArgs;

        auto BuildDataCandidates = [&](int paramIndex, const std::string& paramName) {
            std::vector<std::string> candidates;

            if (paramIndex < static_cast<int>(shellCallArgs.size())) {
                std::string argExpr = TrimCopy(shellCallArgs[paramIndex]);
                if (IsSimpleIdentifierText(argExpr)) {
                    candidates.push_back(argExpr);
                }
            }

            candidates.push_back(paramName);

            std::unordered_set<std::string> seen;
            std::vector<std::string> uniq;
            for (const std::string& name : candidates) {
                std::string trimmed = TrimCopy(name);
                if (trimmed.empty() || seen.count(trimmed) > 0) {
                    continue;
                }
                seen.insert(trimmed);
                uniq.push_back(trimmed);
            }
            return uniq;
        };

        std::vector<RuntimeParamBindingInfo> runtimeInfos(shell->getNumParams());
        std::unordered_set<std::string> capturedSourceVars;

        auto MaterializeScalar = [&](const std::string& rawExpr) {
            return MaterializeScalarExpr(sourceText, rawExpr);
        };

        auto FindShellParamSummary = [&](const std::string& paramName)
            -> const dacppTranslator::ascendc_mlir::CalcShellParamInfo* {
            for (const auto& info : shellSummary.params) {
                if (info.name == paramName) {
                    return &info;
                }
            }
            return nullptr;
        };

        auto ResolveCalcParamLogicalName = [&](int paramIndex, const std::string& shellParamName) {
            if (!calc || paramIndex < 0 || paramIndex >= calc->getNumParams()) {
                return shellParamName;
            }
            Param* calcParam = calc->getParam(paramIndex);
            if (!calcParam) {
                return shellParamName;
            }
            const std::string calcParamName = TrimCopy(calcParam->getName());
            return calcParamName.empty() ? shellParamName : calcParamName;
        };

        int dacCallCount = 0;
        {
            std::regex dacCallRegex(R"(<->)");
            for (std::sregex_iterator it(sourceText.begin(), sourceText.end(), dacCallRegex), end;
                 it != end; ++it) {
                ++dacCallCount;
            }
        }
        bool isSingleKernelCallScenario = dacCallCount <= 1;

        auto TryResolvePhysicalTimeFallbackExpr = [&](std::string& outExpr) {
            outExpr.clear();
            std::regex timeConstRegex(
                R"((?:^|[\n;])\s*(?:constexpr\s+)?(?:const\s+)?(?:float|double)\s+T\s*=\s*([^;]+);)");
            std::smatch match;
            if (!std::regex_search(sourceText, match, timeConstRegex)) {
                return false;
            }

            std::string candidate = MaterializeScalar(TrimCopy(match[1].str()));
            if (candidate.empty()) {
                return false;
            }
            outExpr = candidate;
            return true;
        };

        for (int j = 0; j < shell->getNumParams(); ++j) {
            ShellParam* param = shell->getShellParam(j);
            if (!param) {
                continue;
            }

            RuntimeParamBindingInfo info;
            if (const auto* shellParamSummary = FindShellParamSummary(param->getName())) {
                info.isBroadcastScalar = shellParamSummary->isBroadcastScalar;
            }
            std::vector<std::string> candidates = BuildDataCandidates(j, param->getName());
            for (const std::string& candidate : candidates) {
                TryBindSourceCandidateFromAst(
                    bindingScopeRoot,
                    candidate,
                    MaterializeScalar,
                    info,
                    capturedSourceVars);
                if (info.sourceIsStdVector || info.hasScalarSourceExpr) {
                    break;
                }
            }

            std::string shapeLength;
            if (TryGetShapeTotalLength(param, shapeLength)) {
                info.hasShapeLength = true;
                info.shapeLength = shapeLength;
            }

            runtimeInfos[j] = info;
        }

        std::unordered_set<std::string> rootSourceVars = capturedSourceVars;
        for (const auto& info : runtimeInfos) {
            if (info.bindingResolved &&
                info.bindingKind == RuntimeBindingNodeKind::StdVector &&
                !info.rootSourceVar.empty()) {
                rootSourceVars.insert(info.rootSourceVar);
            }
        }

        BuildHostPreludeCollection(
            gRuntimeBindingCurrentOwnerFunction,
            gRuntimeBindingCurrentDacExprLoc,
            rootSourceVars,
            hostPreludeCollection);
        hostPreludeCode = BuildHostPreludeCodeFromCollection(hostPreludeCollection);

        for (int j = 0; j < shell->getNumParams(); ++j) {
            ShellParam* param = shell->getShellParam(j);
            if (!param) {
                continue;
            }

            RuntimeParamBindingInfo& info = runtimeInfos[j];
            int rw = param->getRw();
            if (!info.isBroadcastScalar || !(rw == 0 || rw == 2)) {
                continue;
            }

            std::string broadcastExpr;
            // 优先级 1: 调用点容器首元素
            if (info.sourceIsStdVector && info.hasSourceVar) {
                broadcastExpr = info.sourceVar + "[0]";
            }
            // 优先级 2: 可解析单标量表达式
            if (broadcastExpr.empty() && info.hasScalarSourceExpr) {
                broadcastExpr = info.scalarSourceExpr;
            }
            // 优先级 3: 单次 kernel 调用场景下尝试 T 常量兜底
            std::string paramNameLower = param->getName();
            std::transform(paramNameLower.begin(), paramNameLower.end(), paramNameLower.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            const bool looksLikeTimeScalar =
                paramNameLower == "t" ||
                paramNameLower.find("time") != std::string::npos;

            if (broadcastExpr.empty() && (isSingleKernelCallScenario || looksLikeTimeScalar)) {
                std::string fallbackExpr;
                if (TryResolvePhysicalTimeFallbackExpr(fallbackExpr)) {
                    broadcastExpr = fallbackExpr;
                }
            }
            // 对输入广播标量，禁止回落到 0；使用非零安全值避免静默错误。
            if (broadcastExpr.empty()) {
                broadcastExpr = "1.0";
            }

            info.broadcastExpr = MaterializeScalar(broadcastExpr);
            info.hasScalarSourceExpr = true;
            info.scalarSourceExpr = info.broadcastExpr;
        }

        commonCode += "    //  算子 " + kernelName + " 基础参数与数据定义 \n";
        commonCode += "    uint32_t blockDim" + std::to_string(index) + " = " + coreNumStr + ";\n";

        for(int j = 0; j < shell->getNumParams(); j++) {
            ShellParam* param = shell->getShellParam(j);
            std::string name = param->getName();
            const std::string calcLogicalName = ResolveCalcParamLogicalName(j, name);
            std::string originalType = param->getBasicType();
            
            std::string mappedType = MapToStdType(originalType);           
            std::string kernelMappedType = MapToKernelType(originalType);  
            bool useFloatBitBridge =
                (policy.floatLikeTypes.count(originalType) > 0 &&
                 mappedType == policy.hostFloatStorageType);

            auto BuildToKernelExpr = [&](const std::string& valueExpr) {
                return BuildToKernelCastExpr(valueExpr, mappedType, useFloatBitBridge);
            };
            int rw = param->getRw();
            const RuntimeParamBindingInfo& runtimeInfo = runtimeInfos[j];
            const bool hasRootStdVector =
                runtimeInfo.bindingResolved &&
                runtimeInfo.bindingKind == RuntimeBindingNodeKind::StdVector &&
                !runtimeInfo.rootSourceVar.empty();
            // A scalar DAC view (for example tensor[{}]) remains a broadcast
            // even when its backing value lives in a one-element std::vector.
            // Treating that backing vector as a normal tensor makes later
            // tiles read zero padding instead of the scalar value.
            const bool useBroadcastScalar = runtimeInfo.isBroadcastScalar;
            const std::string& rootSourceVar = runtimeInfo.rootSourceVar;
            int paramLeftHalo = 0;
            int paramRightHalo = 0;
            if (const auto* shellParamSummary = FindShellParamSummary(name)) {
                for (const auto& window : shellParamSummary->windows) {
                    if (!window.isWindow) {
                        continue;
                    }
                    paramLeftHalo = std::max(paramLeftHalo, window.leftHalo);
                    paramRightHalo = std::max(paramRightHalo, window.rightHalo);
                }
            }

            RecordFieldLayout recordLayout;
            const bool isRecordParam =
                !useBroadcastScalar &&
                hasRootStdVector &&
                ParamUsesRecordElementType(param, &recordLayout);

            if (isRecordParam) {
                dacppTranslator::ascendc_mlir::CalcParamBinding paramBinding;
                paramBinding.logicalName = calcLogicalName;
                RecordGroupPlan recordGroup;
                recordGroup.logicalName = name;
                recordGroup.paramType = recordLayout.recordTypeName;
                recordGroup.rw = rw;
                recordGroup.runtimeInfo = runtimeInfo;
                recordGroup.leftHalo = paramLeftHalo;
                recordGroup.rightHalo = paramRightHalo;
                recordGroup.rootVector = rootSourceVar;
                for (const auto& fieldInfo : recordLayout.fields) {
                    dacppTranslator::ascendc_mlir::CalcRecordFieldBinding fieldBinding;
                    fieldBinding.fieldName = fieldInfo.name;
                    fieldBinding.physicalName = BuildRecordFieldPhysicalName(name, fieldInfo.name);
                    fieldBinding.fieldType = fieldInfo.typeName;
                    paramBinding.fields.push_back(fieldBinding);
                    AppendAoSFieldParam(recordGroup, fieldInfo);
                }
                EmitRecordGroupRuntime(recordGroup);
                if (!paramBinding.fields.empty()) {
                    calcParamBindings.push_back(std::move(paramBinding));
                }
                continue;
            }
            
            shellParamOrder.push_back(name);
            if (rw == 0 || rw == 2) inLocalVars.push_back(name);
            if (rw == 1 || rw == 2) outLocalVars.push_back(name);
            if (rw == 0 || rw == 2) {
                if (useBroadcastScalar) {
                    temporalScalarInputs.push_back(name);
                } else {
                    temporalTensorInputs.push_back(name);
                }
            }
            if (rw == 1 || rw == 2) {
                temporalTensorOutputs.push_back(name);
            }

            HostParamPlan paramPlan;
            paramPlan.name = name;
            paramPlan.logicalName = name;
            paramPlan.paramType = param->getType();
            paramPlan.basicType = originalType;
            paramPlan.mappedType = mappedType;
            paramPlan.kernelType = kernelMappedType;
            paramPlan.rw = rw;
            paramPlan.useFloatBitBridge = useFloatBitBridge;
            paramPlan.isBroadcastScalar = useBroadcastScalar;
            paramPlan.leftHalo = paramLeftHalo;
            paramPlan.rightHalo = paramRightHalo;
            paramPlan.rootSourceVar = rootSourceVar;
            hostParamPlans.push_back(paramPlan);

            std::string bufferName = HostBufferName(name);
            std::string bridgeSoAImagName = name + "BridgeSoAImag";
            std::string bridgeSoASecondName = name + "BridgeSoASecond";
            const LogicalCountPlan logicalCount =
                BuildLogicalCountExpr(runtimeInfo, useBroadcastScalar,
                                      hasRootStdVector, rootSourceVar, rw);
            const bool sourceMatchesMappedType =
                !runtimeInfo.sourceElemType.empty() &&
                MapToStdType(runtimeInfo.sourceElemType) == mappedType;

            EmitOutputResizeIfNeeded(rootSourceVar, rw, logicalCount);
            AppendConcreteBufferDeclaration(
                bufferName, mappedType, logicalCount.expr,
                paramLeftHalo, paramRightHalo);
            if (useBroadcastScalar && (rw == 0 || rw == 2)) {
                std::string scalarExpr = runtimeInfo.scalarSourceExpr.empty() ? "1.0" : runtimeInfo.scalarSourceExpr;
                commonCode += "    std::fill_n(" + bufferName + ".data.begin(), " +
                              bufferName + ".logicalElems, " +
                              BuildToKernelExpr(scalarExpr) + ");\n";
            } else if (hasRootStdVector) {
                if ((rw == 0 || rw == 2) && runtimeInfo.sourceElemIsComplex) {
                    commonCode += "    std::vector<" + mappedType + "> " + bridgeSoAImagName + "(" + bufferName + ".logicalElems, static_cast<" + mappedType + ">(0));\n";
                    commonCode += "    for (size_t bridgeIdx = 0; bridgeIdx < std::min(" + bufferName + ".logicalElems, " + rootSourceVar + ".size()); ++bridgeIdx) {\n";
                    commonCode += "        " + bufferName + ".data[bridgeIdx] = " + BuildToKernelExpr(rootSourceVar + "[bridgeIdx].real()") + ";\n";
                    commonCode += "        " + bridgeSoAImagName + "[bridgeIdx] = " + BuildToKernelExpr(rootSourceVar + "[bridgeIdx].imag()") + ";\n";
                    commonCode += "    }\n";
                } else if ((rw == 0 || rw == 2) && runtimeInfo.sourceElemIsPair) {
                    commonCode += "    std::vector<" + mappedType + "> " + bridgeSoASecondName + "(" + bufferName + ".logicalElems, static_cast<" + mappedType + ">(0));\n";
                    commonCode += "    for (size_t bridgeIdx = 0; bridgeIdx < std::min(" + bufferName + ".logicalElems, " + rootSourceVar + ".size()); ++bridgeIdx) {\n";
                    commonCode += "        " + bufferName + ".data[bridgeIdx] = " + BuildToKernelExpr(rootSourceVar + "[bridgeIdx].first") + ";\n";
                    commonCode += "        " + bridgeSoASecondName + "[bridgeIdx] = " + BuildToKernelExpr(rootSourceVar + "[bridgeIdx].second") + ";\n";
                    commonCode += "    }\n";
                } else if (rw == 0 || rw == 2) {
                    const std::string copyCount = "std::min(" + bufferName +
                                                  ".logicalElems, " + rootSourceVar +
                                                  ".size())";
                    if (sourceMatchesMappedType && !useFloatBitBridge) {
                        commonCode += "        std::memcpy(" + bufferName +
                                      ".data.data(), " + rootSourceVar +
                                      ".data(), " + copyCount + " * sizeof(" +
                                      mappedType + "));\n";
                    } else {
                        commonCode += "    for (size_t item = 0; item < " +
                                      copyCount + "; ++item) {\n";
                        commonCode += "        " + bufferName +
                                      ".data[item] = " +
                                      BuildToKernelExpr(
                                          rootSourceVar + "[item]") + ";\n";
                        commonCode += "    }\n";
                    }
                }
            } else if (runtimeInfo.hasScalarSourceExpr && (rw == 0 || rw == 2)) {
                commonCode += "    std::fill_n(" + bufferName + ".data.begin(), " +
                              bufferName + ".logicalElems, " +
                              BuildToKernelExpr(runtimeInfo.scalarSourceExpr) + ");\n";
            }
            if (!(useBroadcastScalar && (rw == 0 || rw == 2))) {
                hostKernelLengthBuffers.push_back(bufferName);
            }

            const std::string cpuGmName = "cpuGm_" + name;
            cpuAllocs += "    uint8_t *" + cpuGmName + " = (uint8_t *)AscendC::GmAlloc(" + bufferName + ".bytes());\n";
            if (rw == 0 || rw == 2) {
                cpuMemcpysIn += "    memcpy(" + cpuGmName + ", " + bufferName + ".data.data(), " + bufferName + ".bytes());\n";
            }
            if (rw == 1 || rw == 2) {
                cpuMemcpysOut += "    memcpy(" + bufferName + ".data.data(), " + cpuGmName + ", " + bufferName + ".bytes());\n";
                if (hasRootStdVector) {
                    if (runtimeInfo.sourceElemIsComplex) {
                        std::string elemTypeAlias = name + "ElemType";
                        std::string writeBack = BuildDirectVectorWriteBack(
                            rootSourceVar, bufferName, "static_cast<size_t>(0)",
                            mappedType, name, useFloatBitBridge, true, false,
                            bridgeSoAImagName, bridgeSoASecondName,
                            sourceMatchesMappedType);
                        cpuMemcpysOut += writeBack;
                        hostWriteBacks += writeBack;
                    } else if (runtimeInfo.sourceElemIsPair) {
                        std::string elemTypeAlias = name + "ElemType";
                        std::string writeBack = BuildDirectVectorWriteBack(
                            rootSourceVar, bufferName, "static_cast<size_t>(0)",
                            mappedType, name, useFloatBitBridge, false, true,
                            bridgeSoAImagName, bridgeSoASecondName,
                            sourceMatchesMappedType);
                        cpuMemcpysOut += writeBack;
                        hostWriteBacks += writeBack;
                    } else {
                        std::string elemTypeAlias = name + "ElemType";
                        std::string writeBack = BuildDirectVectorWriteBack(
                            rootSourceVar, bufferName, "static_cast<size_t>(0)",
                            mappedType, name, useFloatBitBridge, false, false,
                            bridgeSoAImagName, bridgeSoASecondName,
                            sourceMatchesMappedType);
                        cpuMemcpysOut += writeBack;
                        hostWriteBacks += writeBack;
                    }
                }
            }
            cpuFrees += "    AscendC::GmFree((void *)" + cpuGmName + ");\n";
            
            if (j > 0) cpuArgs += ", ";
            cpuArgs += cpuGmName;

            npuAllocs += "    " + DeviceName() + ".allocate(" + bufferName + ");\n";
            if (rw == 0 || rw == 2) { 
                npuMemcpysIn += BufferCopyInCode(bufferName);
            }
            if (rw == 1 || rw == 2) {
                npuMemcpysOut += BufferCopyOutCode(bufferName);
                if (hasRootStdVector) {
                    // Find the alias offset for this output parameter's root container.
                    // When the shell call argument is a tensor view (e.g. new_rho_tensor[{1,WIDTH-1}]),
                    // the kernel writes to bufferData[0..N-1] but the host vector slice starts at
                    // offsetExpr. We must write back to rootSourceVar[offset + bridgeIdx] so that
                    // the subsequent host update loop (which uses the alias offset) reads correctly.
                    std::string writeBackOffset = "0";
                    for (const auto& aliasEntry : hostPreludeCollection.tensorAliases) {
                        const HostTensorAliasInfo& aliasInfo = aliasEntry.second;
                        if (aliasInfo.valid &&
                            aliasInfo.rootContainerName == rootSourceVar &&
                            !aliasInfo.offsetExpr.empty() &&
                            aliasInfo.offsetExpr != "0") {
                            writeBackOffset = aliasInfo.offsetExpr;
                            break;
                        }
                    }
                    const std::string writeBackOffsetExpr =
                        writeBackOffset == "0" ? "static_cast<size_t>(0)" : "static_cast<size_t>(" + writeBackOffset + ")";
                    if (runtimeInfo.sourceElemIsComplex) {
                        std::string writeBack = BuildDirectVectorWriteBack(
                            rootSourceVar, bufferName, writeBackOffsetExpr,
                            mappedType, name, useFloatBitBridge, true, false,
                            bridgeSoAImagName, bridgeSoASecondName,
                            sourceMatchesMappedType);
                        npuMemcpysOut += writeBack;
                    } else if (runtimeInfo.sourceElemIsPair) {
                        std::string writeBack = BuildDirectVectorWriteBack(
                            rootSourceVar, bufferName, writeBackOffsetExpr,
                            mappedType, name, useFloatBitBridge, false, true,
                            bridgeSoAImagName, bridgeSoASecondName,
                            sourceMatchesMappedType);
                        npuMemcpysOut += writeBack;
                    } else {
                        std::string writeBack = BuildDirectVectorWriteBack(
                            rootSourceVar, bufferName, writeBackOffsetExpr,
                            mappedType, name, useFloatBitBridge, false, false,
                            bridgeSoAImagName, bridgeSoASecondName,
                            sourceMatchesMappedType);
                        npuMemcpysOut += writeBack;
                        hostWriteBacks += writeBack;
                    }
                }
            }
            npuFrees += "    " + DeviceName() + ".release(" + bufferName + ");\n";

            if (j > 0) npuArgs += ", ";
            npuArgs += bufferName + ".device";

            if (j > 0) {
                doParams += ", "; cpuParams += ", "; 
                kernelParamsDecl += ", "; kernelCallArgs += ", ";
            }
            doParams += "uint8_t *" + name;
            cpuParams += "GM_ADDR " + name;
            kernelParamsDecl += "GM_ADDR " + name;
            kernelCallArgs += name;

            std::string NameGm = name + "Gm";
            std::string QueueName = (rw == 0 || rw == 2) ? "inQueue" + name : "outQueue" + name;
            std::string leftHaloName = name + "LeftHalo";
            std::string rightHaloName = name + "RightHalo";
            std::string validLengthName = name + "ValidLength";
            std::string copyLengthName = name + "CopyLength";
            std::string gmStartName = name + "GmStart";
            std::string gmLengthName = name + "GmLength";
            std::string writeLengthName = name + "WriteLength";
            std::string inputMinCopyElemsName = name + "InputMinCopyElems";
            std::string outputSpanName = name + "OutputSpan";

            tQueDecl += "    AscendC::GlobalTensor<" + kernelMappedType + "> " + NameGm + ";\n";
            tQueDecl += "    AscendC::TQue<AscendC::QuePosition::" + std::string((rw == 0 || rw == 2) ? "VECIN" : "VECOUT") + ", BUFFER_NUM> " + QueueName + ";\n";

            if (rw == 0 || rw == 2) {
                const auto inputTransferPolicy = ResolveTransferPolicyForParam(
                    true,
                    useBroadcastScalar,
                    paramLeftHalo,
                    paramRightHalo);
                if (inputTransferPolicy == dacppTranslator::ascendc_mlir::AscendCTransferPolicy::BroadcastScalar) {
                    EmitBroadcastScalarInputTransfer(
                        kernelMappedType,
                        name,
                        NameGm,
                        QueueName,
                        validLengthName,
                        copyLengthName,
                        gmLengthName,
                        tQueDecl,
                        initCode,
                        copyInCode,
                        computeDeQueCode,
                        computeFreeCode);
                } else if (inputTransferPolicy == dacppTranslator::ascendc_mlir::AscendCTransferPolicy::SpatialWindow) {
                    EmitHaloInputTransfer(
                        kernelMappedType,
                        name,
                        NameGm,
                        QueueName,
                        leftHaloName,
                        rightHaloName,
                        validLengthName,
                        copyLengthName,
                        gmStartName,
                        gmLengthName,
                        inputMinCopyElemsName,
                        paramLeftHalo,
                        paramRightHalo,
                        policy.npuCopyAlignBytes,
                        tQueDecl,
                        initCode,
                        copyInCode,
                        computeDeQueCode,
                        computeFreeCode);
                } else {
                    EmitAlignedInputTransfer(
                        kernelMappedType,
                        name,
                        NameGm,
                        QueueName,
                        validLengthName,
                        copyLengthName,
                        gmLengthName,
                        tQueDecl,
                        initCode,
                        copyInCode,
                        computeDeQueCode,
                        computeFreeCode);
                }
            }
            if (rw == 1 || rw == 2) {
                EmitAlignedOutputTransfer(
                    kernelMappedType,
                    name,
                    NameGm,
                    QueueName,
                    writeLengthName,
                    outputSpanName,
                    tQueDecl,
                    initCode,
                    computeAllocCode,
                    computeEnQueCode,
                    copyOutCode);
            }
        }

        const std::string exprId = std::to_string(index);
        const std::string kernelLengthSize = "kernelSize" + exprId;
        const std::string coreCount = "coreCount" + exprId;
        commonCode += "    size_t " + kernelLengthSize + " = static_cast<size_t>(0);\n";
        commonCode += "    const size_t " + coreCount + " = blockDim" + exprId +
                      " == 0 ? static_cast<size_t>(1) : static_cast<size_t>(blockDim" +
                      exprId + ");\n";
        for (const std::string& bufferName : hostKernelLengthBuffers) {
            commonCode += "    " + kernelLengthSize + " = std::max(" + kernelLengthSize +
                          ", " + bufferName + ".kernelElems);\n";
        }
        commonCode += "    uint32_t kernelLength" + exprId + " = 0;\n";

        std::string prefix = shell->getNumParams() > 0 ? ", " : "";
        cpuArgs += prefix + "kernelLength" + std::to_string(index) + ", blockDim" + std::to_string(index) +
                   ", tileLength" + std::to_string(index) + ", tileCount" + std::to_string(index);
        npuArgs += prefix + "kernelLength" + std::to_string(index) + ", blockDim" + std::to_string(index) +
                   ", tileLength" + std::to_string(index) + ", tileCount" + std::to_string(index);
        
        doParams += prefix + "uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount";
        cpuParams += prefix + "uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount";
        kernelParamsDecl += prefix + "uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount";
        kernelCallArgs += prefix + "totalLength, coreNum, tileLength, tileCount";
    }

    void GenerateKernelFile() {
        Calc* calc = expr->getCalc();
        dacppTranslator::ascendc_mlir::CalcMlirPipelineResult pipelineResult = RunCalcMlirPipelineDetailed(
            calc,
            sourceText,
            "currentTileLength",
            inLocalVars,
            outLocalVars,
            shellParamOrder,
            shellSummary,
            &calcParamBindings);
        dominantFamily = pipelineResult.dominantFamily.value_or(
            dacppTranslator::ascendc_mlir::CalcSemanticFamily::PointwiseFamily);
        if (!pipelineResult.plan.steps.empty()) {
            transferPolicy = pipelineResult.plan.steps.front().transferPolicy;
        } else {
            switch (dominantFamily) {
            case dacppTranslator::ascendc_mlir::CalcSemanticFamily::TemporalStencilFamily:
                transferPolicy = dacppTranslator::ascendc_mlir::AscendCTransferPolicy::TemporalLoopCarried;
                break;
            case dacppTranslator::ascendc_mlir::CalcSemanticFamily::SpatialStencilFamily:
                transferPolicy = dacppTranslator::ascendc_mlir::AscendCTransferPolicy::SpatialWindow;
                break;
            case dacppTranslator::ascendc_mlir::CalcSemanticFamily::PointwiseFamily:
            default:
                transferPolicy = dacppTranslator::ascendc_mlir::AscendCTransferPolicy::PointwiseDirect;
                break;
            }
        }

        pointwiseScalarLoopSchedule = PointwiseScalarLoopSchedule();
        loopCarriedVectorSchedule = LoopCarriedVectorSchedule();
        linearMulticardPlan = LinearMulticardPlan();
        if (dominantFamily == dacppTranslator::ascendc_mlir::CalcSemanticFamily::PointwiseFamily &&
            transferPolicy == dacppTranslator::ascendc_mlir::AscendCTransferPolicy::PointwiseDirect) {
            TryExtractPointwiseScalarLoopScheduleFromAst(expr, shellSummary, pointwiseScalarLoopSchedule);
            if (pointwiseScalarLoopSchedule.recognized) {
                AppendReferencedScalarConstDecls(
                    sourceText,
                    {
                        pointwiseScalarLoopSchedule.scalarInitExpr,
                        pointwiseScalarLoopSchedule.loopUpperBoundExpr,
                        pointwiseScalarLoopSchedule.loopStepExpr,
                        pointwiseScalarLoopSchedule.snapshotIndexExpr,
                        pointwiseScalarLoopSchedule.snapshotPrintRowExpr
                    },
                    hostPreludeCode);
            }
        }
        if (dominantFamily == dacppTranslator::ascendc_mlir::CalcSemanticFamily::SpatialStencilFamily &&
            transferPolicy == dacppTranslator::ascendc_mlir::AscendCTransferPolicy::SpatialWindow) {
            TryExtractLoopCarriedVectorScheduleFromAst(
                expr,
                shellSummary,
                loopCarriedVectorSchedule);
        }

        std::string vectorApiCode = pipelineResult.emittedCode;
        for (const auto& paramPlan : hostParamPlans) {
            if (!paramPlan.isBroadcastScalar || !(paramPlan.rw == 0 || paramPlan.rw == 2)) {
                continue;
            }
            const std::string localGetValuePattern =
                "\\b" + EscapeRegexText(paramPlan.name) + R"(Local\s*\.\s*GetValue\s*\(\s*0\s*\))";
            vectorApiCode = ReplaceAllRegex(
                vectorApiCode,
                localGetValuePattern,
                paramPlan.name + "ScalarValue");
        }

        std::unordered_set<std::string> writtenOutputTensors;
        for (const auto& step : pipelineResult.plan.steps) {
            const std::string dstName = TrimCopy(step.dst);
            if (!dstName.empty()) {
                writtenOutputTensors.insert(dstName);
            }
        }

        const std::string preserveUntouchedOutputsCode =
            BuildUntouchedRecordOutputsCode(writtenOutputTensors);

        std::string computeCode =
            computeDeQueCode + computeAllocCode + vectorApiCode +
            preserveUntouchedOutputsCode + computeEnQueCode + computeFreeCode;
        VectorKernelSkeletonFragments skeleton = EmitKernelSkeletonForFamily(
            initCode,
            copyInCode,
            computeCode,
            copyOutCode,
            tQueDecl);
        std::string fastInitCode = skeleton.initCode;
        std::string fastCopyInCode = skeleton.copyInCode;
        std::string fastComputeCode = skeleton.computeCode;
        std::string fastCopyOutCode = skeleton.copyOutCode;
        std::string fastTQueDecl = skeleton.tQueDecl;
        uint32_t extraInputQueueCount = 0;
        uint32_t scratchBufferCountForPlan = 0;
        uint32_t scratchElementBytesForPlan = 4;
        bool linearTiledFastPath = false;
        uint32_t linearInFeaturesForPlan = 0;
        bool linearHasBiasForPlan = false;
        std::string structuredFixedScalarName;
        uint32_t structuredFixedScalarBytes = 0;

        auto FindShellParamByName = [&](const std::string& paramName) -> ShellParam* {
            Shell* shell = expr->getShell();
            if (!shell) {
                return nullptr;
            }
            for (int j = 0; j < shell->getNumParams(); ++j) {
                ShellParam* param = shell->getShellParam(j);
                if (param && param->getName() == paramName) {
                    return param;
                }
            }
            return nullptr;
        };

        if (pipelineResult.structuredShiftedMapUpdate.has_value()) {
            const auto& pattern = pipelineResult.structuredShiftedMapUpdate.value();
            ShellParam* srcParam = FindShellParamByName(pattern.srcTensor);
            ShellParam* dstParam = FindShellParamByName(pattern.dstTensor);
            if (srcParam && dstParam) {
                const std::string srcKernelType = MapToKernelType(srcParam->getBasicType());
                const std::string dstKernelType = MapToKernelType(dstParam->getBasicType());
                if (srcKernelType == dstKernelType) {
                    int scratchCenter = 0;
                    int scratchShift = 0;
                    int scratchUpdate = 0;
                    int scratchPost = 0;

                    std::string mapCenterCode = dacppTranslator::ascendc_mlir::EmitStructuredMapExpr(
                        pattern, false, "currentTileLength", &scratchCenter);
                    std::string mapShiftCode = dacppTranslator::ascendc_mlir::EmitStructuredMapExpr(
                        pattern, true, "currentTileLength", &scratchShift);
                    std::string updateCode = dacppTranslator::ascendc_mlir::EmitStructuredUpdateExpr(
                        pattern, false, "currentTileLength", &scratchUpdate);
                    std::string postCode = dacppTranslator::ascendc_mlir::EmitStructuredUpdateExpr(
                        pattern, true, "currentTileLength", &scratchPost);

                    int scratchBufferCount = std::max(std::max(scratchCenter, scratchShift),
                                                      std::max(scratchUpdate, scratchPost));
                    extraInputQueueCount = 1;
                    scratchBufferCountForPlan = static_cast<uint32_t>(2 + scratchBufferCount);
                    scratchElementBytesForPlan = KernelElementBytes(dstKernelType);
                    fastInitCode.clear();
                    fastCopyInCode.clear();
                    fastComputeCode.clear();
                    fastCopyOutCode.clear();
                    fastTQueDecl.clear();

                    ASCENDC_TEMPLATE::StructuredShiftedMapUpdateFastPathCode fastPathCode =
                        ASCENDC_TEMPLATE::CodeGen_StructuredShiftedMapUpdateFastPath(
                            srcKernelType,
                            dstKernelType,
                            pattern.srcTensor,
                            pattern.dstTensor,
                            pattern.shiftOffset,
                            scratchBufferCount,
                            mapCenterCode,
                            mapShiftCode,
                            updateCode,
                            postCode,
                            policy.vectorComputeAlignBytes);

                    fastInitCode = fastPathCode.initCode;
                    fastCopyInCode = fastPathCode.copyInCode;
                    fastComputeCode = fastPathCode.computeCode;
                    fastCopyOutCode = fastPathCode.copyOutCode;
                    fastTQueDecl = fastPathCode.tQueDecl;
                }
            }
        } else if (pipelineResult.structuredMultiViewExpr.has_value()) {
            const auto& pattern = pipelineResult.structuredMultiViewExpr.value();
            ShellParam* srcParam = FindShellParamByName(pattern.srcTensor);
            ShellParam* dstParam = FindShellParamByName(pattern.dstTensor);
            ShellParam* scalarParam = nullptr;
            if (!pattern.scalarInputs.empty()) {
                scalarParam = FindShellParamByName(pattern.scalarInputs.front());
            }
            if (srcParam && dstParam) {
                const std::string srcKernelType = MapToKernelType(srcParam->getBasicType());
                const std::string dstKernelType = MapToKernelType(dstParam->getBasicType());
                const bool hasScalarInput = scalarParam != nullptr;
                const std::string scalarKernelType = hasScalarInput
                    ? MapToKernelType(scalarParam->getBasicType())
                    : srcKernelType;
                const std::string scalarName = hasScalarInput
                    ? scalarParam->getName()
                    : std::string();

                if (srcKernelType == dstKernelType) {
                    int scratchMain = 0;
                    int scratchPost = 0;
                    std::string mainExprCode = dacppTranslator::ascendc_mlir::EmitStructuredMultiViewMainExpr(
                        pattern, "currentTileLength", &scratchMain);
                    std::string postExprCode = dacppTranslator::ascendc_mlir::EmitStructuredMultiViewPostExpr(
                        pattern, "currentTileLength", &scratchPost);
                    mainExprCode = NormalizeRedundantDirectViewCopy(mainExprCode);
                    postExprCode = NormalizeRedundantDirectViewCopy(postExprCode);
                    int inferredScratchCount = InferExtraScratchBufferCountFromExpr(mainExprCode, postExprCode);
                    int scratchBufferCount = std::max(std::max(scratchMain, scratchPost), inferredScratchCount);

                    if (!mainExprCode.empty() &&
                        (!hasScalarInput || pattern.scalarInputs.size() == 1)) {
                        fastInitCode.clear();
                        fastCopyInCode.clear();
                        fastComputeCode.clear();
                        fastCopyOutCode.clear();
                        fastTQueDecl.clear();
                        extraInputQueueCount = pattern.viewOffsets.empty()
                            ? 0
                            : static_cast<uint32_t>(pattern.viewOffsets.size() - 1);
                        scratchBufferCountForPlan = static_cast<uint32_t>(2 + scratchBufferCount);
                        scratchElementBytesForPlan = KernelElementBytes(dstKernelType);
                        if (hasScalarInput) {
                            structuredFixedScalarName = scalarName;
                            structuredFixedScalarBytes =
                                KernelElementBytes(scalarKernelType);
                        }

                        ASCENDC_TEMPLATE::StructuredMultiViewExprFastPathCode fastPathCode =
                            ASCENDC_TEMPLATE::CodeGen_StructuredMultiViewExprFastPath(
                                srcKernelType,
                                dstKernelType,
                                scalarKernelType,
                                pattern.srcTensor,
                                pattern.dstTensor,
                                scalarName,
                                pattern.viewOffsets,
                                hasScalarInput,
                                scratchBufferCount,
                                mainExprCode,
                                postExprCode,
                                policy.vectorComputeAlignBytes);

                        fastInitCode = fastPathCode.initCode;
                        fastCopyInCode = fastPathCode.copyInCode;
                        fastComputeCode = fastPathCode.computeCode;
                        fastCopyOutCode = fastPathCode.copyOutCode;
                        fastTQueDecl = fastPathCode.tQueDecl;
                    }
                }
            }
        }

        // Linear reduction uses output-feature tiling. The input vector and
        // reduction scratch are fixed-size, while weight/bias/output storage
        // scales with the runtime output tile length.
        if (dominantFamily == dacppTranslator::ascendc_mlir::CalcSemanticFamily::LinearReductionFamily &&
            !pipelineResult.plan.steps.empty() &&
            pipelineResult.plan.steps.front().hasLinearReductionPayload) {

            const dacppTranslator::ascendc_mlir::LinearReductionStepPayload& lrp =
                pipelineResult.plan.steps.front().linearReductionPayload;

            // Determine IN_FEATURES (K) and OUT_FEATURES from shell shape info.
            int inFeatures = lrp.K > 0 ? lrp.K : 8;
            int outFeatures = 1;

            // Try to get OUT_FEATURES from the output tensor shape.
            Shell* shell = expr->getShell();
            if (shell) {
                for (int j = 0; j < shell->getNumParams(); ++j) {
                    ShellParam* sp = shell->getShellParam(j);
                    if (!sp) continue;
                    if (sp->getName() == lrp.outputTensor && sp->getDim() >= 1) {
                        int s = sp->getShape(0);
                        if (s > 0) outFeatures = s;
                        break;
                    }
                }
            }

            const std::string inF  = std::to_string(inFeatures);
            const std::string outF = std::to_string(outFeatures);
            const std::string& inp = lrp.inputTensor;
            const std::string& wgt = lrp.weightTensor;
            const std::string& bia = lrp.biasTensor;
            const std::string& out = lrp.outputTensor;
            const bool hasBias = lrp.hasBias && !bia.empty();
            linearMulticardPlan.recognized = true;
            linearMulticardPlan.inputName = inp;
            linearMulticardPlan.weightName = wgt;
            linearMulticardPlan.biasName = bia;
            linearMulticardPlan.outputName = out;
            linearMulticardPlan.inFeatures = inFeatures;
            linearMulticardPlan.outFeatures = outFeatures;
            linearMulticardPlan.hasBias = hasBias;
            linearTiledFastPath = true;
            linearInFeaturesForPlan = static_cast<uint32_t>(std::max(inFeatures, 1));
            linearHasBiasForPlan = hasBias;

            // Build a self-contained kernel that matches the reference linear-kernel.cpp.
            std::string lrInitCode;
            lrInitCode += "        " + inp + "Gm.SetGlobalBuffer((__gm__ float*)" + inp + ", " + inF + ");\n";
            lrInitCode += "        " + wgt + "Gm.SetGlobalBuffer((__gm__ float*)" + wgt + " + static_cast<size_t>(blockStart) * " + inF + ", " + inF + " * blockLength);\n";
            if (hasBias) {
                lrInitCode += "        " + bia + "Gm.SetGlobalBuffer((__gm__ float*)" + bia + " + blockStart, blockLength);\n";
            }
            lrInitCode += "        " + out + "Gm.SetGlobalBuffer((__gm__ float*)" + out + " + blockStart, blockLength);\n";
            lrInitCode += "        pipe.InitBuffer(inQueue" + inp + ", BUFFER_NUM, " + inF + " * sizeof(float));\n";
            lrInitCode += "        pipe.InitBuffer(inQueue" + wgt + ", BUFFER_NUM, " + inF + " * tileLength * sizeof(float));\n";
            if (hasBias) {
                lrInitCode += "        pipe.InitBuffer(inQueue" + bia + ", BUFFER_NUM, tileLength * sizeof(float));\n";
            }
            lrInitCode += "        pipe.InitBuffer(outQueue" + out + ", BUFFER_NUM, tileLength * sizeof(float));\n";
            lrInitCode += "        pipe.InitBuffer(calcBuf, " + inF + " * sizeof(float));\n";
            lrInitCode += "        pipe.InitBuffer(calcBufReduce, " + inF + " * sizeof(float));\n";

            std::string lrCopyInCode;
            lrCopyInCode += "        AscendC::LocalTensor<float> " + inp + "Local = inQueue" + inp + ".AllocTensor<float>();\n";
            lrCopyInCode += "        AscendC::LocalTensor<float> " + wgt + "Local = inQueue" + wgt + ".AllocTensor<float>();\n";
            if (hasBias) {
                lrCopyInCode += "        AscendC::LocalTensor<float> " + bia + "Local = inQueue" + bia + ".AllocTensor<float>();\n";
            }
            lrCopyInCode += "        AscendC::DataCopy(" + inp + "Local, " + inp + "Gm, " + inF + ");\n";
            lrCopyInCode += "        AscendC::DataCopy(" + wgt + "Local, " + wgt + "Gm[static_cast<size_t>(tileOffset) * " + inF + "], " + inF + " * currentTileLength);\n";
            lrCopyInCode += "        inQueue" + inp + ".EnQue(" + inp + "Local);\n";
            lrCopyInCode += "        inQueue" + wgt + ".EnQue(" + wgt + "Local);\n";
            if (hasBias) {
                lrCopyInCode += "        AscendC::DataCopy(" + bia + "Local, " + bia + "Gm[tileOffset], currentTileLength);\n";
                lrCopyInCode += "        inQueue" + bia + ".EnQue(" + bia + "Local);\n";
            }

            std::string lrComputeCode;
            lrComputeCode += "        AscendC::LocalTensor<float> " + inp + "Local = inQueue" + inp + ".DeQue<float>();\n";
            lrComputeCode += "        AscendC::LocalTensor<float> " + wgt + "Local = inQueue" + wgt + ".DeQue<float>();\n";
            if (hasBias) {
                lrComputeCode += "        AscendC::LocalTensor<float> " + bia + "Local = inQueue" + bia + ".DeQue<float>();\n";
            }
            lrComputeCode += "        AscendC::LocalTensor<float> " + out + "Local = outQueue" + out + ".AllocTensor<float>();\n";
            lrComputeCode += "        AscendC::LocalTensor<float> tmpLocal = calcBuf.Get<float>();\n";
            lrComputeCode += "        AscendC::LocalTensor<float> reduceLocal = calcBufReduce.Get<float>();\n";
            lrComputeCode += "        for (uint32_t i = 0; i < currentTileLength; ++i) {\n";
            lrComputeCode += "            AscendC::LocalTensor<float> weightRow = " + wgt + "Local[i * " + inF + "];\n";
            lrComputeCode += "            AscendC::Mul(tmpLocal, " + inp + "Local, weightRow, " + inF + ");\n";
            lrComputeCode += "            AscendC::WholeReduceSum(reduceLocal, tmpLocal, " + inF + ", 1, 1, 1, 1);\n";
            if (hasBias) {
                lrComputeCode += "            float sum = " + bia + "Local.GetValue(i) + reduceLocal.GetValue(0);\n";
            } else {
                lrComputeCode += "            float sum = reduceLocal.GetValue(0);\n";
            }
            lrComputeCode += "            " + out + "Local.SetValue(i, sum);\n";
            lrComputeCode += "        }\n";
            lrComputeCode += "        outQueue" + out + ".EnQue<float>(" + out + "Local);\n";
            lrComputeCode += "        inQueue" + inp + ".FreeTensor(" + inp + "Local);\n";
            lrComputeCode += "        inQueue" + wgt + ".FreeTensor(" + wgt + "Local);\n";
            if (hasBias) {
                lrComputeCode += "        inQueue" + bia + ".FreeTensor(" + bia + "Local);\n";
            }

            std::string lrCopyOutCode;
            lrCopyOutCode += "        AscendC::LocalTensor<float> outCopy = outQueue" + out + ".DeQue<float>();\n";
            lrCopyOutCode += "        AscendC::DataCopy(" + out + "Gm[tileOffset], outCopy, currentTileLength);\n";
            lrCopyOutCode += "        outQueue" + out + ".FreeTensor(outCopy);\n";

            // TQue/TBuf declarations.
            std::string lrTQueDecl;
            lrTQueDecl += "    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;\n";
            lrTQueDecl += "    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBufReduce;\n";
            lrTQueDecl += "    AscendC::GlobalTensor<float> " + inp + "Gm;\n";
            lrTQueDecl += "    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue" + inp + ";\n";
            lrTQueDecl += "    AscendC::GlobalTensor<float> " + wgt + "Gm;\n";
            lrTQueDecl += "    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue" + wgt + ";\n";
            if (hasBias) {
                lrTQueDecl += "    AscendC::GlobalTensor<float> " + bia + "Gm;\n";
                lrTQueDecl += "    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue" + bia + ";\n";
            }
            lrTQueDecl += "    AscendC::GlobalTensor<float> " + out + "Gm;\n";
            lrTQueDecl += "    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue" + out + ";\n";

            fastInitCode    = lrInitCode;
            fastCopyInCode  = lrCopyInCode;
            fastComputeCode = lrComputeCode;
            fastCopyOutCode = lrCopyOutCode;
            fastTQueDecl    = lrTQueDecl;
        }

        dacppTranslator::ascendc::VectorKernelResourceModel resourceModel;
        resourceModel.tileAlignElements = 1;
        for (const HostParamPlan& param : hostParamPlans) {
            const uint32_t bytes = KernelElementBytes(param.kernelType);
            const uint32_t computeAlign = static_cast<uint32_t>(
                (std::max(policy.vectorComputeAlignBytes, 1) + bytes - 1) / bytes);
            resourceModel.tileAlignElements =
                std::max(resourceModel.tileAlignElements, computeAlign);
            if (param.rw == 0 || param.rw == 2) {
                if (!structuredFixedScalarName.empty() &&
                    param.name == structuredFixedScalarName) {
                    continue;
                }
                dacppTranslator::ascendc::UbAllocationKind kind =
                    param.isBroadcastScalar
                        ? dacppTranslator::ascendc::UbAllocationKind::BroadcastQueue
                        : ((param.leftHalo != 0 || param.rightHalo != 0)
                               ? dacppTranslator::ascendc::UbAllocationKind::HaloQueue
                               : dacppTranslator::ascendc::UbAllocationKind::InputQueue);
                resourceModel.allocations.push_back({
                    "inQueue" + param.name, kind, bytes, 1,
                    static_cast<uint32_t>(std::max(param.leftHalo, 0)),
                    static_cast<uint32_t>(std::max(param.rightHalo, 0)), true});
            }
            if (param.rw == 1 || param.rw == 2) {
                resourceModel.allocations.push_back({
                    "outQueue" + param.name,
                    dacppTranslator::ascendc::UbAllocationKind::OutputQueue,
                    bytes, 1, 0, 0, true});
            }
        }
        if (structuredFixedScalarBytes != 0) {
            resourceModel.fixedBytes +=
                ((static_cast<size_t>(structuredFixedScalarBytes) + 31) / 32) * 32;
        }
        if (linearTiledFastPath) {
            resourceModel.allocations.clear();
            resourceModel.tileAlignElements = static_cast<uint32_t>(
                (std::max(policy.vectorComputeAlignBytes, 1) + sizeof(float) - 1) /
                sizeof(float));
            // The input vector and two reduction temporaries are independent of
            // output tile length. Reserve the worst-case (double-buffered) input.
            resourceModel.fixedBytes =
                static_cast<size_t>(linearInFeaturesForPlan) * sizeof(float) * 4;
            resourceModel.allocations.push_back({
                "linear-weight-tile",
                dacppTranslator::ascendc::UbAllocationKind::InputQueue,
                sizeof(float), linearInFeaturesForPlan, 0, 0, true});
            if (linearHasBiasForPlan) {
                resourceModel.allocations.push_back({
                    "linear-bias-tile",
                    dacppTranslator::ascendc::UbAllocationKind::InputQueue,
                    sizeof(float), 1, 0, 0, true});
            }
            resourceModel.allocations.push_back({
                "linear-output-tile",
                dacppTranslator::ascendc::UbAllocationKind::OutputQueue,
                sizeof(float), 1, 0, 0, true});
        }
        if (extraInputQueueCount != 0) {
            uint32_t inputBytes = 4;
            for (const HostParamPlan& param : hostParamPlans) {
                if (param.rw == 0 || param.rw == 2) {
                    inputBytes = KernelElementBytes(param.kernelType);
                    break;
                }
            }
            resourceModel.allocations.push_back({
                "structured-extra-view-queues",
                dacppTranslator::ascendc::UbAllocationKind::HaloQueue,
                inputBytes, extraInputQueueCount, 0, 0, true});
        }
        if (scratchBufferCountForPlan != 0) {
            resourceModel.allocations.push_back({
                "structured-scratch",
                dacppTranslator::ascendc::UbAllocationKind::CalcBuffer,
                scratchElementBytesForPlan, scratchBufferCountForPlan,
                0, 0, false});
        }

        vectorTilingPlan =
            dacppTranslator::ascendc::BuildVectorKernelTilingPlan(resourceModel, policy);
        hasVectorTilingPlan = true;
        const dacppTranslator::ascendc::VectorKernelTilingPlan& tilingPlan =
            vectorTilingPlan;
        const std::string tilingDiagnostics =
            dacppTranslator::ascendc::FormatVectorTilingDiagnostics(resourceModel, tilingPlan);
        if (policy.emitTilingDiagnostics) {
            std::cerr << "[AscendC tiling] " << GetEmittedKernelName() << "\n"
                      << tilingDiagnostics;
        }

        const std::string exprId = std::to_string(index);
        const std::string kernelLengthSize = "kernelSize" + exprId;
        const std::string coreCount = "coreCount" + exprId;
        const std::string runtimeTiling = "vectorRuntimeTiling" + exprId;
        if (linearTiledFastPath) {
            commonCode += "    " + kernelLengthSize + " = " +
                          HostBufferName(linearMulticardPlan.outputName) +
                          ".logicalElems;\n";
        }
        commonCode += "    const dacpp_host::Tile " + runtimeTiling +
                      " = dacpp_host::makeTile(" + AppName() + ", " +
                      kernelLengthSize + ", " +
                      coreCount + ", " + std::to_string(tilingPlan.tileAlignElements) +
                      ", " + std::to_string(tilingPlan.maxTileElements) + ");\n";
        commonCode += "    " + kernelLengthSize + " = static_cast<size_t>(" +
                      runtimeTiling + ".len);\n";
        commonCode += "    kernelLength" + exprId + " = " + runtimeTiling +
                      ".len;\n";
        commonCode += "    uint32_t tileLength" + exprId + " = " + runtimeTiling +
                      ".size;\n";
        commonCode += "    uint32_t tileCount" + exprId + " = " + runtimeTiling +
                      ".count;\n";
        if (linearTiledFastPath) {
            const std::string inputBuffer =
                HostBufferName(linearMulticardPlan.inputName);
            const std::string weightBuffer =
                HostBufferName(linearMulticardPlan.weightName);
            const std::string outputBuffer =
                HostBufferName(linearMulticardPlan.outputName);
            commonCode += "    " + inputBuffer + ".data.resize(std::max(" +
                          inputBuffer + ".data.size(), static_cast<size_t>(" +
                          std::to_string(linearInFeaturesForPlan) + ")));\n";
            commonCode += "    if (" + kernelLengthSize +
                          " > std::numeric_limits<size_t>::max() / static_cast<size_t>(" +
                          std::to_string(linearInFeaturesForPlan) +
                          ")) { " + AppName() +
                          ".fail(dacpp_host::Stage::Config, 2, "
                          "\"linear weight padding overflow\"); return " +
                          AppName() + ".finish(); }\n";
            commonCode += "    " + weightBuffer + ".data.resize(std::max(" +
                          weightBuffer + ".data.size(), " + kernelLengthSize +
                          " * static_cast<size_t>(" +
                          std::to_string(linearInFeaturesForPlan) + ")));\n";
            commonCode += "    " + outputBuffer + ".data.resize(std::max(" +
                          outputBuffer + ".data.size(), " + kernelLengthSize + "));\n";
            if (linearHasBiasForPlan) {
                const std::string biasBuffer =
                    HostBufferName(linearMulticardPlan.biasName);
                commonCode += "    " + biasBuffer + ".data.resize(std::max(" +
                              biasBuffer + ".data.size(), " + kernelLengthSize + "));\n";
            }
        }
        for (const std::string& bufferName : hostKernelLengthBuffers) {
            commonCode += "    " + bufferName + ".kernelElems = std::max(" + bufferName +
                          ".kernelElems, " + kernelLengthSize + ");\n";
            commonCode += "    { size_t withLeftHalo = 0, storage = 0, physical = 0; "
                          "dacpp_host::checkedAdd(" + AppName() + ", " +
                          bufferName + ".kernelElems, " + bufferName +
                          ".leftHaloElems, withLeftHalo); "
                          "dacpp_host::checkedAdd(" + AppName() +
                          ", withLeftHalo, " + bufferName +
                          ".rightHaloElems, storage); "
                          "dacpp_host::checkedAlign(" + AppName() +
                          ", storage, " + bufferName +
                          ".alignElems, physical); if (" + AppName() +
                          ".ok() && " + bufferName +
                          ".data.size() < physical) " + bufferName +
                          ".data.resize(physical); }\n";
        }

        const std::string emittedKernelName = GetEmittedKernelName();
        std::string kernelCode = ASCENDC_TEMPLATE::CodeGen_KernelFile(
            emittedKernelName, kernelParamsDecl, kernelCallArgs, doParams, 
            fastInitCode, fastCopyInCode, fastComputeCode, fastCopyOutCode, fastTQueDecl,
            policy.vectorComputeAlignBytes,
            tilingPlan.bufferNum,
            tilingPlan.tileAlignElements,
            tilingPlan.maxTileElements,
            tilingDiagnostics
        );
        
        WriteOutputFile(generatedKernelFileName, kernelCode, "Vector Kernel侧成功生成: ");
    }

    std::string GetHostPreludeCode() { return hostPreludeCode; }
    const HostPreludeCollection& GetHostPreludeCollection() const { return hostPreludeCollection; }
    std::string GetCommonDecls() { return commonCode; }
    std::string GetNpuAllocCode() {
        std::string code = "\n    //  算子 " + kernelName + " NPU 执行流 \n";
        code += npuPtrs + npuAllocs;
        return code;
    }
    std::string GetNpuFreeCode() { return npuFrees; }
    std::string GetCpuCode() {
        std::string code = "\n    //  算子 " + kernelName + " CPU 执行流 \n";
        code += cpuAllocs + cpuMemcpysIn;
        code += "    ICPU_RUN_KF(" + kernelName + ", blockDim" + std::to_string(index) + ", " + cpuArgs + ");\n";
        code += cpuMemcpysOut + cpuFrees;
        return code;
    }
    std::string GetNpuCode(const ASCENDC_TEMPLATE::TemporalLoopSchedule* temporalSpec = nullptr,
                           const std::string& temporalLaunchEntry = "") {
        std::string code = "\n    //  算子 " + kernelName + " NPU 执行流 \n";
        code += npuPtrs + npuAllocs + npuMemcpysIn;

        const bool usePointwiseScalarLoopSchedule =
            pointwiseScalarLoopSchedule.recognized &&
            dominantFamily == dacppTranslator::ascendc_mlir::CalcSemanticFamily::PointwiseFamily &&
            transferPolicy == dacppTranslator::ascendc_mlir::AscendCTransferPolicy::PointwiseDirect;
        const bool useTemporalSchedule =
            temporalSpec &&
            temporalSpec->recognized &&
            transferPolicy == dacppTranslator::ascendc_mlir::AscendCTransferPolicy::TemporalLoopCarried;
        if (usePointwiseScalarLoopSchedule) {
            auto FindParamPlanByName = [&](const std::string& paramName) -> const HostParamPlan* {
                for (const auto& plan : hostParamPlans) {
                    if (plan.name == paramName) {
                        return &plan;
                    }
                }
                return nullptr;
            };

            const HostParamPlan* scalarPlan =
                FindParamPlanByName(pointwiseScalarLoopSchedule.scalarParamName);
            const HostParamPlan* outputPlan =
                FindParamPlanByName(pointwiseScalarLoopSchedule.outputParamName);

            auto FindAliasByName = [&](const std::string& aliasName) -> const HostTensorAliasInfo* {
                auto it = hostPreludeCollection.tensorAliases.find(
                    BuildHostTensorAliasKey(
                        pointwiseScalarLoopSchedule.ownerFunctionName,
                        TrimCopy(aliasName)));
                if (it == hostPreludeCollection.tensorAliases.end()) {
                    return nullptr;
                }
                return &it->second;
            };

            auto MaterializeLoopExpr = [&](const std::string& rawExpr,
                                           const std::string& scalarCurrentVar,
                                           const std::string& fallbackExpr) {
                std::string exprText = TrimCopy(rawExpr);
                if (exprText.empty()) {
                    return fallbackExpr;
                }

                auto ReplaceScalarAccess = [&](const std::string& baseName) {
                    if (baseName.empty()) {
                        return;
                    }
                    const std::string escapedBase = EscapeRegexText(baseName);
                    const std::regex accessRegex("\\b" + escapedBase + R"(\s*\[\s*0\s*\])");
                    exprText = std::regex_replace(exprText, accessRegex, scalarCurrentVar);
                };

                ReplaceScalarAccess(pointwiseScalarLoopSchedule.scalarArgName);
                ReplaceScalarAccess(pointwiseScalarLoopSchedule.scalarSourceVar);
                exprText = TrimCopy(exprText);
                return exprText.empty() ? fallbackExpr : exprText;
            };

            if (!scalarPlan || !outputPlan) {
                code += "    " + AppName() +
                        ".fail(dacpp_host::Stage::HostFlow, 1, "
                        "\"scalar loop binding\");\n";
            } else {
                const std::string scalarName = scalarPlan->name;
                const std::string scalarMappedType = scalarPlan->mappedType;
                const std::string scalarCurrentVar = scalarName + "LoopScalarCurrent";
                const std::string scalarBufferName = HostBufferName(scalarName);

                const std::string outputName = outputPlan->name;
                const std::string outputBufferName = HostBufferName(outputName);
                const HostTensorAliasInfo* snapshotTensorAlias =
                    FindAliasByName(pointwiseScalarLoopSchedule.snapshotTensorName);
                const HostTensorAliasInfo* outputAlias =
                    FindAliasByName(pointwiseScalarLoopSchedule.outputArgName);

                const std::string loopInitExpr = MaterializeLoopExpr(
                    pointwiseScalarLoopSchedule.scalarInitExpr,
                    scalarCurrentVar,
                    "0");
                const std::string loopUpperExpr = MaterializeLoopExpr(
                    pointwiseScalarLoopSchedule.loopUpperBoundExpr,
                    scalarCurrentVar,
                    loopInitExpr);
                const std::string loopStepExpr = MaterializeLoopExpr(
                    pointwiseScalarLoopSchedule.loopStepExpr,
                    scalarCurrentVar,
                    "1");
                const std::string loopCompareOp =
                    pointwiseScalarLoopSchedule.loopCompareOp.empty()
                        ? "<="
                        : pointwiseScalarLoopSchedule.loopCompareOp;
                const std::string snapshotIndexExpr = MaterializeLoopExpr(
                    pointwiseScalarLoopSchedule.snapshotIndexExpr,
                    scalarCurrentVar,
                    "0");

                code += "    auto " + scalarCurrentVar + " = static_cast<" + scalarMappedType + ">(" + loopInitExpr + ");\n";

                code += "    while (" + scalarCurrentVar + " " + loopCompareOp + " static_cast<" + scalarMappedType + ">(" + loopUpperExpr + ")) {\n";
                code += "        if (!" + scalarBufferName + ".data.empty()) {\n";
                code += "            " + scalarBufferName + ".data[0] = static_cast<" + scalarMappedType + ">(" + scalarCurrentVar + ");\n";
                code += "        }\n";
                code += BufferCopyInCode(scalarBufferName, "        ");
                code += "        " + DeviceName() + ".launch(" + kernelName +
                        "_do(blockDim" + std::to_string(index) +
                        ", " + StreamName() + ", " + npuArgs + "));\n";
                code += "        " + DeviceName() + ".sync();\n";
                code += BufferCopyOutCode(outputBufferName, "        ");
                if (outputAlias &&
                    snapshotTensorAlias &&
                    snapshotTensorAlias->valid &&
                    outputAlias->valid &&
                    !pointwiseScalarLoopSchedule.snapshotIndexExpr.empty()) {
                    HostTensorAliasInfo snapshotAliasInfo = *snapshotTensorAlias;
                    RewriteTensorScalarReadsInAliasInfo(
                        snapshotAliasInfo,
                        pointwiseScalarLoopSchedule.ownerFunctionName,
                        hostPreludeCollection.tensorAliases);
                    if (snapshotAliasInfo.rank > 0) {
                        HostTensorAliasInfo indexedSnapshotInfo;
                        if (TryApplyTensorIndexText(
                                snapshotAliasInfo,
                                snapshotIndexExpr,
                                indexedSnapshotInfo) &&
                            indexedSnapshotInfo.valid) {
                            snapshotAliasInfo = indexedSnapshotInfo;
                        }
                    }

                    HostTensorAliasInfo outputBufferAliasInfo = *outputAlias;
                    outputBufferAliasInfo.rootContainerName = outputBufferName + ".data";
                    outputBufferAliasInfo.rootDataExpr = outputBufferName + ".data.data()";
                    RewriteTensorScalarReadsInAliasInfo(
                        outputBufferAliasInfo,
                        pointwiseScalarLoopSchedule.ownerFunctionName,
                        hostPreludeCollection.tensorAliases);

                    const std::string snapshotCopyCode = EmitTensorAliasAssignmentFromInfos(
                        snapshotAliasInfo,
                        &outputBufferAliasInfo,
                        outputBufferName + ".data",
                        static_cast<size_t>(index) * 1000 + 100);
                    if (!snapshotCopyCode.empty()) {
                        code += IndentCodeBlock(TrimCopy(snapshotCopyCode), "        ");
                    }
                }
                code += "        " + scalarCurrentVar + " += static_cast<" + scalarMappedType + ">(" + loopStepExpr + ");\n";
                code += "    }\n";
            }
        } else if (useTemporalSchedule) {
            auto IsTensorInput = [](int rw) {
                return rw == 0 || rw == 2;
            };
            auto IsTensorOutput = [](int rw) {
                return rw == 1 || rw == 2;
            };

            auto HasTensorParam = [&](const std::string& name, bool requireInput, bool requireOutput) {
                for (const auto& plan : hostParamPlans) {
                    if (plan.name != name || plan.isBroadcastScalar) {
                        continue;
                    }
                    if (requireInput && !IsTensorInput(plan.rw)) {
                        continue;
                    }
                    if (requireOutput && !IsTensorOutput(plan.rw)) {
                        continue;
                    }
                    return true;
                }
                return false;
            };

            auto ResolveArgToShellParamName = [&](const std::string& logicalName) {
                const std::string trimmed = TrimCopy(logicalName);
                for (size_t argIdx = 0; argIdx < resolvedShellCallArgs.size(); ++argIdx) {
                    if (TrimCopy(resolvedShellCallArgs[argIdx]) != trimmed) {
                        continue;
                    }
                    if (argIdx < shellParamOrder.size()) {
                        return shellParamOrder[argIdx];
                    }
                }
                return trimmed;
            };

            auto PickFallbackTensorName = [&](bool inputTensor) {
                const std::vector<std::string>& candidates = inputTensor ? temporalTensorInputs : temporalTensorOutputs;
                for (const auto& candidate : candidates) {
                    if (HasTensorParam(candidate, inputTensor, !inputTensor)) {
                        return candidate;
                    }
                }
                for (const auto& plan : hostParamPlans) {
                    if (plan.isBroadcastScalar) {
                        continue;
                    }
                    if (inputTensor && IsTensorInput(plan.rw)) {
                        return plan.name;
                    }
                    if (!inputTensor && IsTensorOutput(plan.rw)) {
                        return plan.name;
                    }
                }
                return std::string();
            };

            std::string inputTensorName = ResolveArgToShellParamName(temporalSpec->inputSliceVar);
            std::string outputTensorName = ResolveArgToShellParamName(temporalSpec->outputSliceVar);

            if (!HasTensorParam(inputTensorName, true, false)) {
                inputTensorName = PickFallbackTensorName(true);
            }
            if (!HasTensorParam(outputTensorName, false, true)) {
                outputTensorName = PickFallbackTensorName(false);
            }

            auto FormatNumericLiteral = [](double value) {
                const double rounded = std::round(value);
                if (std::fabs(value - rounded) < 1e-9 &&
                    rounded >= static_cast<double>(std::numeric_limits<long long>::min()) &&
                    rounded <= static_cast<double>(std::numeric_limits<long long>::max())) {
                    return std::to_string(static_cast<long long>(rounded));
                }
                std::ostringstream oss;
                oss << std::setprecision(17) << value;
                return oss.str();
            };

            const std::string blockDimName = "blockDim" + std::to_string(index);
            const std::string kernelLengthName = "kernelLength" + std::to_string(index);

            auto IsKnownHostIdentifier = [&](const std::string& ident,
                                             const std::string& loopIndexIdent,
                                             const std::string& writebackIndexIdent) {
                if (ident.empty()) {
                    return false;
                }
                if (ident == blockDimName || ident == kernelLengthName ||
                    ident == "stream" || ident == "temporalWriteCount") {
                    return true;
                }
                if (ident == "data" || ident == "logicalElems" ||
                    ident == "kernelElems" || ident == "bytes") {
                    return true;
                }
                if (!loopIndexIdent.empty() && ident == loopIndexIdent) {
                    return true;
                }
                if (!writebackIndexIdent.empty() && ident == writebackIndexIdent) {
                    return true;
                }
                for (const auto& plan : hostParamPlans) {
                    if (ident == plan.name ||
                        ident == HostBufferName(plan.name)) {
                        return true;
                    }
                }
                return false;
            };

            auto SanitizeScheduleExpr = [&](const std::string& rawExpr,
                                            const std::string& fallbackExpr,
                                            const std::string& loopIndexIdent,
                                            const std::string& writebackIndexIdent) {
                std::string exprText = TrimCopy(rawExpr);
                if (exprText.empty()) {
                    return fallbackExpr;
                }

                std::regex identRegex(R"([A-Za-z_]\w*)");
                for (std::sregex_iterator it(exprText.begin(), exprText.end(), identRegex), end; it != end; ++it) {
                    const std::string ident = it->str();
                    if (ident == "true" || ident == "false") {
                        continue;
                    }
                    if (IsKnownHostIdentifier(ident, loopIndexIdent, writebackIndexIdent)) {
                        continue;
                    }

                    double resolved = 0.0;
                    if (!TryResolveSimpleSymbolValue(sourceText, ident, resolved) || !std::isfinite(resolved)) {
                        return fallbackExpr;
                    }

                    const std::string replacement = FormatNumericLiteral(resolved);
                    const std::regex replaceRegex("\\b" + ident + "\\b");
                    exprText = std::regex_replace(exprText, replaceRegex, replacement);
                }

                return TrimCopy(exprText);
            };

            ASCENDC_TEMPLATE::TemporalLoopSchedule scheduleForCode = *temporalSpec;
            if (scheduleForCode.loopIndexVar.empty()) {
                scheduleForCode.loopIndexVar = "k";
            }
            if (scheduleForCode.writebackLoopIndexVar.empty()) {
                scheduleForCode.writebackLoopIndexVar = "i";
            }
            scheduleForCode.loopBeginExpr = SanitizeScheduleExpr(
                scheduleForCode.loopBeginExpr,
                "0",
                scheduleForCode.loopIndexVar,
                scheduleForCode.writebackLoopIndexVar);
            scheduleForCode.loopUpperBoundExpr = SanitizeScheduleExpr(
                scheduleForCode.loopUpperBoundExpr,
                "static_cast<int>(" + kernelLengthName + ")",
                scheduleForCode.loopIndexVar,
                scheduleForCode.writebackLoopIndexVar);
            scheduleForCode.loopStepExpr = SanitizeScheduleExpr(
                scheduleForCode.loopStepExpr,
                "++" + scheduleForCode.loopIndexVar,
                scheduleForCode.loopIndexVar,
                scheduleForCode.writebackLoopIndexVar);
            scheduleForCode.writebackRangeBeginExpr = SanitizeScheduleExpr(
                scheduleForCode.writebackRangeBeginExpr,
                "1",
                scheduleForCode.loopIndexVar,
                scheduleForCode.writebackLoopIndexVar);
            scheduleForCode.writebackRangeEndExpr = SanitizeScheduleExpr(
                scheduleForCode.writebackRangeEndExpr,
                "temporalWriteCount",
                scheduleForCode.loopIndexVar,
                scheduleForCode.writebackLoopIndexVar);

            if (inputTensorName.empty() || outputTensorName.empty()) {
                code += "    " + AppName() +
                        ".fail(dacpp_host::Stage::HostFlow, 1, "
                        "\"temporal tensor binding\");\n";
            } else {
                const std::string launchEntry = temporalLaunchEntry.empty()
                    ? GetLaunchWrapperName()
                    : temporalLaunchEntry;
                code += ASCENDC_TEMPLATE::CodeGen_VectorTemporalStencilSchedule(
                    scheduleForCode,
                    launchEntry,
                    "blockDim" + std::to_string(index),
                    npuArgs,
                    inputTensorName,
                    outputTensorName,
                    DeviceName(),
                    StreamName());
            }
        } else {
            code += "    " + DeviceName() + ".launch(" +
                    GetLaunchWrapperName() + "(blockDim" +
                    std::to_string(index) + ", " + StreamName() +
                    ", " + npuArgs + "));\n";
            code += "    " + DeviceName() + ".sync();\n";
        }

        code += npuMemcpysOut + npuFrees;
        return code;
    }
    bool AbsorbsOwningWhile() const {
        return pointwiseScalarLoopSchedule.recognized &&
               pointwiseScalarLoopSchedule.absorbsOwningWhile;
    }
    bool AbsorbsOwningFor() const {
        return UsesLoopCarriedVectorSchedule();
    }
    std::string GetExternDoDecl() {
        std::string code;
        code += "#ifndef ASCENDC_CPU_DEBUG\n";
        code += "extern \"C\" uint32_t " + GetLaunchWrapperName() +
                "(uint32_t blockDim, void *stream, " + doParams + ");\n";
        code += "#else\n";
        code += "extern \"C\" __global__ __aicore__ void " + kernelName +
                "(" + cpuParams + ");\n";
        code += "#endif\n";
        return code;
    }
    std::string GetExternCpuDecl() { return "extern \"C\" __global__ __aicore__ void " + kernelName + "(" + cpuParams + ");\n"; }

    std::vector<std::string> GetResolvedShellCallArgs() const { return resolvedShellCallArgs; }

    std::string BuildTemporalWrapperFunction(
        const std::string& wrapperName,
        const ASCENDC_TEMPLATE::TemporalLoopSchedule& schedule) const {
        if (!schedule.recognized || wrapperName.empty()) {
            return "";
        }

        std::string code;
        code += "static inline int " + wrapperName + "(uint32_t blockDim, void *stream";
        if (!doParams.empty()) {
            code += ", " + doParams;
        }
        code += ") {\n";
        code += "#ifndef ASCENDC_CPU_DEBUG\n";
        code += "    return static_cast<int>(" + GetLaunchWrapperName() +
                "(blockDim, stream";
        if (!kernelCallArgs.empty()) {
            code += ", " + kernelCallArgs;
        }
        code += "));\n";
        code += "#else\n";
        code += "    (void)stream;\n";
        code += "    ICPU_RUN_KF(" + kernelName + ", blockDim";
        if (!kernelCallArgs.empty()) {
            code += ", " + kernelCallArgs;
        }
        code += ");\n";
        code += "    return 0;\n";
        code += "#endif\n";
        code += "}\n";

        return code;
    }

    // ========================================================
    // FlowKernelLaunch / FlowGeneratedCodeBlock descriptor 输出。
    // 旧 Get*Code 字符串接口仍存在，但只作为 BuildRuntimeBlocks() 的底层数据源；
    // 任何用户源码 Stmt 原文都不会被写入返回的 descriptor。
    // ========================================================

    dacppTranslator::ascendc::FlowKernelLaunch BuildKernelLaunchDescriptor() const {
        dacppTranslator::ascendc::FlowKernelLaunch launch;
        launch.route = dacppTranslator::ascendc::FlowRouteKind::Vector;
        launch.expressionIndex = index;
        launch.kernelFunctionName = GetLaunchWrapperName();
        launch.kernelArtifactFileName = generatedKernelFileName;
        launch.blockDimExpr = "blockDim" + std::to_string(index);

        // paramOrder = npuArgs（kernel_do 的实参顺序，含 *Device 名 +
        // kernelLength/blockDim）。emitter 仅作 descriptor dump 使用；真实 launch
        // 调用文本由 KernelLaunch 角色的 generatedBlocks 直接生成。
        if (!npuArgs.empty()) {
            std::string token;
            for (char c : npuArgs) {
                if (c == ',') {
                    std::string trimmed = TrimCopy(token);
                    if (!trimmed.empty()) launch.paramOrder.push_back(trimmed);
                    token.clear();
                } else {
                    token.push_back(c);
                }
            }
            std::string trimmed = TrimCopy(token);
            if (!trimmed.empty()) launch.paramOrder.push_back(trimmed);
        }

        // scalarParams = 来自 hostParamPlans 中的 broadcast scalar 形参，
        // 用 host 端 logical 名（plan.name）登记，供 descriptor 消费者诊断/对比。
        for (const HostParamPlan& plan : hostParamPlans) {
            if (plan.isBroadcastScalar) {
                launch.scalarParams.push_back(plan.name);
            }
            dacppTranslator::ascendc::FlowRuntimeBuffer buffer;
            buffer.id = plan.bufferId;
            buffer.logicalName = plan.name;
            buffer.elementTypeText = plan.mappedType;
            buffer.logicalCountExpr =
                HostBufferName(plan.name) + ".logicalElems";
            buffer.byteCountExpr = HostBufferName(plan.name) + ".bytes()";
            buffer.hostStrideExpr = buffer.byteCountExpr;
            buffer.access = plan.rw == 0
                ? dacppTranslator::ascendc::FlowBufferAccess::Read
                : (plan.rw == 1
                    ? dacppTranslator::ascendc::FlowBufferAccess::Write
                    : dacppTranslator::ascendc::FlowBufferAccess::ReadWrite);
            buffer.leftHalo = plan.leftHalo;
            buffer.rightHalo = plan.rightHalo;
            buffer.producerResident = plan.producerResident;
            buffer.distribution =
                dacppTranslator::ascendc::FlowMulticardParamDistribution::Sharded;
            launch.runtimeBuffers.push_back(std::move(buffer));
        }
        const MulticardAnalysis mc = AnalyzeMulticardPlan();
        if (mc.enabled) {
            launch.multicardEnabled = true;
            launch.rankLaunchFunctionName = GetLaunchWrapperName();
            launch.perRankParamOrder = launch.paramOrder;
            const std::string kernelLengthParam =
                "kernelLength" + std::to_string(index);
            for (std::string& param : launch.perRankParamOrder) {
                if (TrimCopy(param) == kernelLengthParam) {
                    param = "rank.logicalLength";
                }
            }
            launch.multicard.enabled = true;
            launch.multicard.rankDimExpr = mc.rankDimExpr;
            launch.multicard.requestedRankDimExpr = mc.rankDimExpr;
            launch.multicard.activeRankDimExpr =
                "min(requestedRanks,shardExtent)";
            launch.multicard.coreNumExpr = coreNumStr;
            launch.multicard.deviceMapExpr = "rankId->deviceId=rankId";
            launch.multicard.rankDimEnvName = mc.rankDimEnvName;
            launch.multicard.shardDim = mc.shardDim;
            launch.multicard.shardAxis = mc.shardAxis;
            launch.multicard.logicalLengthExpr = mc.logicalLengthExpr;
            launch.multicard.alignedLengthExpr = mc.alignedLengthExpr;
            launch.multicard.strategy = mc.strategy;
            launch.multicard.replicatedParams = mc.replicatedParams;
            launch.multicard.shardedParams = mc.shardedParams;
            launch.multicard.readHaloLeft = mc.readHaloLeft;
            launch.multicard.readHaloRight = mc.readHaloRight;
            launch.multicard.inputReplicated = mc.inputReplicated;
            launch.multicard.outputSharded = mc.outputSharded;
            launch.multicard.persistentWorkers = mc.persistentWorkers;
            launch.multicard.resourceScope = "route";
            launch.multicard.workerLifetime = "persistent-route";
            launch.multicard.copyMode = "async-stream";
            launch.multicard.communication =
                mc.strategy == "temporal-stencil-persistent"
                    ? dacppTranslator::ascendc::FlowMulticardCommunicationKind::HostBoundary
                    : dacppTranslator::ascendc::FlowMulticardCommunicationKind::None;
            for (const HostParamPlan& plan : hostParamPlans) {
                dacppTranslator::ascendc::FlowMulticardParamPartition partition;
                partition.name = plan.name;
                partition.bufferId = plan.bufferId;
                partition.distribution = IsMulticardBroadcastScalarParam(plan) ||
                        (mc.strategy == "linear-output-shard" &&
                         plan.name == linearMulticardPlan.inputName)
                    ? dacppTranslator::ascendc::FlowMulticardParamDistribution::Replicated
                    : dacppTranslator::ascendc::FlowMulticardParamDistribution::Sharded;
                partition.shardAxis = partition.distribution ==
                        dacppTranslator::ascendc::FlowMulticardParamDistribution::Replicated
                    ? "-" : mc.shardAxis;
                partition.readHaloLeft = plan.leftHalo;
                partition.readHaloRight = plan.rightHalo;
                partition.readWriteMode = plan.rw;
                partition.deviceResident = mc.persistentWorkers ||
                    mc.strategy == "record-field-pointwise" ||
                    partition.distribution ==
                        dacppTranslator::ascendc::FlowMulticardParamDistribution::Replicated;
                launch.multicard.paramPartitions.push_back(std::move(partition));
                for (dacppTranslator::ascendc::FlowRuntimeBuffer& buffer :
                     launch.runtimeBuffers) {
                    if (buffer.logicalName == plan.name) {
                        buffer.distribution =
                            launch.multicard.paramPartitions.back().distribution;
                        break;
                    }
                }
            }
            launch.launchDescriptorText =
                "multicard rankDim=" + mc.rankDimExpr +
                " shard=1 strategy=" + mc.strategy +
                " halo=[" + std::to_string(mc.readHaloLeft) + "," +
                std::to_string(mc.readHaloRight) + "] rankLaunch=" +
                launch.rankLaunchFunctionName;
        }
        return launch;
    }

    dacppTranslator::ascendc::FlowGeneratedCodeBlock
    MakeRuntimeBlock(dacppTranslator::ascendc::FlowGeneratedCodeRole role,
                     std::string label,
                     std::string code) const {
        return MakeFlowCodeBlock(role, std::move(label), std::move(code), index);
    }

    // Role-typed block helper。
    dacppTranslator::ascendc::FlowGeneratedCodeBlock BuildVectorDeviceAllocBlock() {
        const std::string allocCode = TrimCopy(GetNpuAllocCode());
        return MakeRuntimeBlock(
            dacppTranslator::ascendc::FlowGeneratedCodeRole::DeviceAlloc,
            "alloc",
            BuildHostExecBlock("", allocCode, /*scoped=*/false));
    }

    dacppTranslator::ascendc::FlowGeneratedCodeBlock BuildVectorHostToDeviceCopyBlock() {
        return MakeRuntimeBlock(
            dacppTranslator::ascendc::FlowGeneratedCodeRole::HostToDeviceCopy,
            "h2d.inloop",
            IndentCodeBlock(npuMemcpysIn, "    "));
    }

    dacppTranslator::ascendc::FlowGeneratedCodeBlock BuildVectorKernelLaunchBlock() {
        std::string body =
            "    " + DeviceName() + ".launch(" + GetLaunchWrapperName() +
            "(blockDim" + std::to_string(index) + ", " + StreamName() +
            ", " + npuArgs + "));\n";
        body += "    " + DeviceName() + ".sync();\n";
        return MakeRuntimeBlock(
            dacppTranslator::ascendc::FlowGeneratedCodeRole::KernelLaunch,
            "launch.inloop",
            IndentCodeBlock(body, "    "));
    }

    dacppTranslator::ascendc::FlowGeneratedCodeBlock BuildVectorDeviceToHostCopyBlock() {
        return MakeRuntimeBlock(
            dacppTranslator::ascendc::FlowGeneratedCodeRole::DeviceToHostCopy,
            "d2h.inloop",
            IndentCodeBlock(npuMemcpysOut, "    "));
    }

    dacppTranslator::ascendc::FlowGeneratedCodeBlock BuildVectorDeviceFreeBlock() {
        const std::string npuFreePart = TrimCopy(GetNpuFreeCode());
        return MakeRuntimeBlock(
            dacppTranslator::ascendc::FlowGeneratedCodeRole::DeviceFree,
            "free.postloop",
            BuildHostExecBlock("", npuFreePart, /*scoped=*/true));
    }

    static std::string RankAlignedElemsExpr(const std::string& bufferName,
                                            const std::string& tileExpr) {
        (void)bufferName;
        return "static_cast<size_t>(" + tileExpr + ".len)";
    }

    static std::string RankStorageCountExpr(const std::string& bufferName,
                                            const std::string& tileExpr,
                                            int leftHalo,
                                            int rightHalo,
                                            bool broadcast) {
        const std::string aligned = RankAlignedElemsExpr(bufferName, tileExpr);
        if (broadcast) {
            return aligned;
        }
        return "((" + aligned + " + static_cast<size_t>(" +
               std::to_string(leftHalo + rightHalo) + ") + " + bufferName +
               ".alignElems - 1) / " + bufferName + ".alignElems) * " +
               bufferName + ".alignElems";
    }

    std::vector<dacppTranslator::ascendc::FlowGeneratedCodeBlock>
    BuildDirectMulticardRuntimeBlocks(const MulticardAnalysis& mc) {
        using dacppTranslator::ascendc::FlowGeneratedCodeBlock;
        using dacppTranslator::ascendc::FlowGeneratedCodeRole;
        const std::string exprId = std::to_string(index);
        const std::string& appName = AppName();
        const std::string& routeName = RouteName();
        const std::string workersName = "ranks";
        const std::string logicalLengthName = "workLen";
        std::string code;
        const std::string commonDecls = TrimCopy(GetCommonDecls());
        if (!commonDecls.empty()) {
            code += IndentCodeBlock(commonDecls, "    ");
        }
        code += "        const size_t " + logicalLengthName +
                " = static_cast<size_t>(kernelLength" + exprId + ");\n";
        code += "        const dacpp_host::RankSpan " + workersName +
                " = dacpp_host::splitRanks(" + logicalLengthName + ", " +
                routeName + ".size());\n";
        for (const HostParamPlan& plan : hostParamPlans) {
            if (plan.producerResident) continue;
            code += "        size_t " + plan.name +
                    "HostStride = static_cast<size_t>(1);\n";
        }
        code += "        for (size_t id = 0; id < " + workersName +
                ".size(); ++id) {\n";
        code += "            const dacpp_host::RankArgs &r = " + workersName +
                "[id];\n";
        code += "            const dacpp_host::Tile slabTile = "
                "dacpp_host::makeTile(" + appName +
                ", r.len, blockDim" +
                exprId + ", " +
                std::to_string(hasVectorTilingPlan
                                   ? vectorTilingPlan.tileAlignElements : 1) +
                ", " +
                std::to_string(hasVectorTilingPlan
                                   ? vectorTilingPlan.maxTileElements : 1) +
                ");\n";
        for (const HostParamPlan& plan : hostParamPlans) {
            if (plan.producerResident) continue;
            const std::string bufferName = HostBufferName(plan.name);
            const bool broadcast = IsMulticardBroadcastScalarParam(plan);
            const int leftHalo = (plan.rw == 0 || plan.rw == 2) && !broadcast
                                     ? plan.leftHalo : 0;
            const int rightHalo = (plan.rw == 0 || plan.rw == 2) && !broadcast
                                      ? plan.rightHalo : 0;
            const std::string elems = RankStorageCountExpr(
                bufferName, "slabTile", leftHalo, rightHalo, broadcast);
            code += "            " + plan.name + "HostStride = std::max(" 
                    "" + plan.name + "HostStride, (" + elems +
                    ") * sizeof(" + plan.mappedType + "));\n";
        }
        code += "        }\n";
        code += "        const dacpp_host::BufferSpec specs[] = {\n";
        for (const HostParamPlan& plan : hostParamPlans) {
            if (plan.producerResident) continue;
            const char* access = plan.rw == 0 ? "Read"
                : (plan.rw == 1 ? "Write" : "ReadWrite");
            const bool broadcast = IsMulticardBroadcastScalarParam(plan);
            code += "            {" + std::to_string(plan.bufferId) +
                    ", dacpp_host::Access::" + access + ", " +
                    std::to_string(KernelElementBytes(plan.mappedType)) + ", " +
                    std::to_string(plan.leftHalo) + ", " +
                    std::to_string(plan.rightHalo) + ", " +
                    (broadcast ? "true" : "false") +
                    ", dacpp_host::Distribution::" +
                    (broadcast ? "Replicated" : "Sharded") + ", " +
                    plan.name + "HostStride, \"" + plan.name + "\"},\n";
        }
        code += "        };\n";
        code += "        if (!" + routeName +
                ".prepare(specs, sizeof(specs) / sizeof(specs[0]))) return " +
                appName + ".finish();\n";
        const std::string runRankName = "runRank";
        code += "        auto " + runRankName +
                " = [&](uint32_t id, aclrtContext ctx, aclrtStream stream) {\n";
        code += "                const dacpp_host::RankArgs &r = " +
                workersName + "[id];\n";
        code += "                (void)ctx;\n";
        code += BuildPerRankTilingCode(exprId, "                ");

        for (const HostParamPlan& plan : hostParamPlans) {
            const std::string bufferName = HostBufferName(plan.name);
            const bool broadcast = IsMulticardBroadcastScalarParam(plan);
            const int leftHalo = (plan.rw == 0 || plan.rw == 2) && !broadcast
                                     ? plan.leftHalo : 0;
            const int rightHalo = (plan.rw == 0 || plan.rw == 2) && !broadcast
                                      ? plan.rightHalo : 0;
            const std::string elems = RankStorageCountExpr(
                bufferName, "tile", leftHalo, rightHalo, broadcast);
            const std::string bytes = "(" + elems + ") * sizeof(" +
                                      plan.mappedType + ")";
            code += "                const size_t " + plan.name +
                    "Count = " + elems + ";\n";
            code += "                const size_t " + plan.name +
                    "Bytes = " + bytes + ";\n";
            code += "                uint8_t *" + plan.name +
                    "Device = " + routeName + ".device(id, " +
                    std::to_string(plan.bufferId) + ", " + plan.name +
                    "Bytes);\n";
            if (plan.producerResident) {
                code += "                if (" + plan.name +
                        "Device == nullptr) { " + routeName +
                        ".fail(id, dacpp_host::Stage::Alloc, 1); return; }\n";
            } else {
                code += "                uint8_t *" + plan.name +
                        "Host = " + routeName + ".host(id, " +
                        std::to_string(plan.bufferId) + ", " +
                        plan.name + "Bytes);\n";
                code += "                if (" + plan.name +
                        "Host == nullptr || " + plan.name +
                        "Device == nullptr) { " + routeName +
                        ".fail(id, dacpp_host::Stage::Alloc, 1); return; }\n";
            }
        }

        for (const HostParamPlan& plan : hostParamPlans) {
            const bool isInput = plan.rw == 0 || plan.rw == 2;
            const bool isOutput = plan.rw == 1 || plan.rw == 2;
            if (isInput && plan.producerResident) continue;
            const bool broadcast = IsMulticardBroadcastScalarParam(plan);
            const std::string bufferName = HostBufferName(plan.name);
            const std::string hostPtr = "reinterpret_cast<" + plan.mappedType +
                                        "*>(" + plan.name + "Host)";
            if (isInput && broadcast) {
                code += "                std::fill_n(" + hostPtr + ", " + plan.name +
                        "Count, " + bufferName +
                        ".data.empty() ? static_cast<" + plan.mappedType +
                        ">(0) : " + bufferName + ".data[0]);\n";
            } else if (isInput) {
                const std::string left = std::to_string(plan.leftHalo);
                const std::string right = std::to_string(plan.rightHalo);
                code += "                std::fill_n(" + hostPtr + ", " + plan.name +
                        "Count, static_cast<" + plan.mappedType + ">(0));\n";
                code += "                const size_t " + plan.name +
                        "Begin = r.begin > static_cast<size_t>(" + left +
                        ") ? r.begin - static_cast<size_t>(" + left +
                        ") : static_cast<size_t>(0);\n";
                code += "                const size_t " + plan.name +
                        "End = std::min(r.begin + r.len + static_cast<size_t>(" +
                        right + "), " + bufferName + ".data.size());\n";
                code += "                const size_t " + plan.name +
                        "LeftGap = r.begin - " + plan.name +
                        "Begin;\n";
                code += "                const size_t " + plan.name +
                        "Offset = static_cast<size_t>(" + left + ") > " +
                        plan.name + "LeftGap ? static_cast<size_t>(" + left +
                        ") - " + plan.name +
                        "LeftGap : static_cast<size_t>(0);\n";
                code += "                if (" + plan.name + "End > " +
                        plan.name + "Begin) { std::memcpy(" + hostPtr +
                        " + " + plan.name + "Offset, " + bufferName +
                        ".data.data() + " + plan.name + "Begin, (" +
                        plan.name + "End - " + plan.name +
                        "Begin) * sizeof(" + plan.mappedType + ")); }\n";
            } else if (isOutput) {
                code += "                std::memset(" + plan.name +
                        "Host, 0, " + plan.name + "Bytes);\n";
            }
            if (isInput) {
                if (!plan.producerResident) {
                    code += "                " + routeName +
                            ".copyIn(id, " + plan.name + "Device, " +
                            plan.name + "Bytes, " + plan.name + "Host, " +
                            plan.name + "Bytes, stream);\n";
                }
            }
        }

        std::string launchArgs;
        for (const HostParamPlan& plan : hostParamPlans) {
            if (!launchArgs.empty()) launchArgs += ", ";
            launchArgs += plan.name + "Device";
            if (!IsMulticardBroadcastScalarParam(plan) &&
                (plan.rw == 0 || plan.rw == 2) && plan.leftHalo > 0) {
                launchArgs += " + static_cast<size_t>(" +
                              std::to_string(plan.leftHalo) + ") * sizeof(" +
                              plan.mappedType + ")";
            }
        }
        if (!launchArgs.empty()) launchArgs += ", ";
        launchArgs += "tile.len, blockDim" + exprId +
                      ", tile.size, tile.count";
        code += "                " + routeName + ".launch(id, " +
                GetLaunchWrapperName() + "(blockDim" + exprId +
                ", stream, " + launchArgs + "));\n";

        for (const HostParamPlan& plan : hostParamPlans) {
            if (!(plan.rw == 1 || plan.rw == 2)) continue;
            code += "                " + routeName +
                    ".copyOut(id, " + plan.name + "Host, " +
                    plan.name + "Bytes, " + plan.name + "Device, " +
                    plan.name + "Bytes, stream);\n";
        }
        code += "                " + routeName +
                ".sync(id, stream);\n";
        for (const HostParamPlan& plan : hostParamPlans) {
            if (!(plan.rw == 1 || plan.rw == 2) || !plan.isRecordField ||
                plan.rootSourceVar.empty()) {
                continue;
            }
            code += "                " + routeName + ".markResident(id, " +
                    std::to_string(plan.bufferId) + ");\n";
        }
        for (const HostParamPlan& plan : hostParamPlans) {
            if (!(plan.rw == 1 || plan.rw == 2)) continue;
            const std::string bufferName = HostBufferName(plan.name);
            const std::string typedHost = "reinterpret_cast<const " + plan.mappedType +
                                          "*>(" + plan.name + "Host)";
            code += "                if (r.len > 0 && r.begin < " +
                    bufferName + ".data.size()) { std::memcpy(" + bufferName +
                    ".data.data() + r.begin, " + typedHost +
                    ", std::min(r.len, " + bufferName +
                    ".data.size() - r.begin) * sizeof(" +
                    plan.mappedType + ")); }\n";
        }
        code += "        };\n";
        code += "        if (!" + routeName +
                ".run(static_cast<uint32_t>(" + workersName + ".size())" +
                ", " + runRankName + ")) return " + appName + ".finish();\n";
        code += IndentCodeBlock(BuildHostWriteBackOnlyCode(), "    ");
        std::vector<FlowGeneratedCodeBlock> blocks;
        blocks.push_back(MakeRuntimeBlock(
            FlowGeneratedCodeRole::MulticardLaunch,
            "multicard.runtime",
            std::move(code)));
        return blocks;
    }

    std::vector<dacppTranslator::ascendc::FlowGeneratedCodeBlock>
    BuildDirectLinearMulticardRuntimeBlocks() {
        using dacppTranslator::ascendc::FlowGeneratedCodeBlock;
        using dacppTranslator::ascendc::FlowGeneratedCodeRole;
        const std::string& appName = AppName();
        const std::string& routeName = RouteName();
        const HostParamPlan* inputPlan =
            FindHostParamPlanByName(linearMulticardPlan.inputName);
        const HostParamPlan* weightPlan =
            FindHostParamPlanByName(linearMulticardPlan.weightName);
        const HostParamPlan* outputPlan =
            FindHostParamPlanByName(linearMulticardPlan.outputName);
        const HostParamPlan* biasPlan = linearMulticardPlan.hasBias
            ? FindHostParamPlanByName(linearMulticardPlan.biasName)
            : nullptr;
        if (!inputPlan || !weightPlan || !outputPlan ||
            (linearMulticardPlan.hasBias && !biasPlan)) {
            return {MakeRuntimeBlock(
                FlowGeneratedCodeRole::MulticardLaunch,
                "linear.multicard.error",
                "    " + appName +
                ".fail(dacpp_host::Stage::HostFlow, 1, \"linear binding\");\n"
                "    return " + appName + ".finish();\n")};
        }

        const std::string exprId = std::to_string(index);
        const std::string workersName = "ranks";
        const std::string inputBuf = HostBufferName(inputPlan->name);
        const std::string weightBuf = HostBufferName(weightPlan->name);
        const std::string outputBuf = HostBufferName(outputPlan->name);
        const std::string biasBuf = biasPlan ? HostBufferName(biasPlan->name) : "";
        std::string code;
        const std::string commonDecls = TrimCopy(GetCommonDecls());
        if (!commonDecls.empty()) code += IndentCodeBlock(commonDecls, "    ");
        code += "        const size_t inputFeatures = static_cast<size_t>(" +
                std::to_string(linearMulticardPlan.inFeatures) + ");\n";
        code += "        const size_t outputFeatures = " + outputBuf +
                ".logicalElems;\n";
        code += "        const dacpp_host::RankSpan " + workersName +
                " = dacpp_host::splitRanks(outputFeatures, " + routeName +
                ".size());\n";
        code += "        size_t inputHostStride = std::max<size_t>("
                "static_cast<size_t>(1), inputFeatures * sizeof(" +
                inputPlan->mappedType + "));\n";
        code += "        size_t weightHostStride = static_cast<size_t>(1);\n";
        if (biasPlan) {
            code += "        size_t biasHostStride = static_cast<size_t>(1);\n";
        }
        code += "        size_t outputHostStride = static_cast<size_t>(1);\n";
        code += "        for (size_t id = 0; id < " + workersName +
                ".size(); ++id) {\n";
        code += "            const dacpp_host::RankArgs &r = " + workersName +
                "[id];\n";
        code += "            const dacpp_host::Tile slabTile = "
                "dacpp_host::makeTile(" + appName +
                ", r.len, blockDim" +
                exprId + ", " +
                std::to_string(hasVectorTilingPlan
                                   ? vectorTilingPlan.tileAlignElements : 1) +
                ", " +
                std::to_string(hasVectorTilingPlan
                                   ? vectorTilingPlan.maxTileElements : 1) +
                ");\n";
        code += "            const size_t aligned = "
                "static_cast<size_t>(slabTile.len);\n";
        code += "            weightHostStride = std::max("
                "weightHostStride, inputFeatures * aligned * sizeof(" +
                weightPlan->mappedType + "));\n";
        if (biasPlan) {
            code += "            biasHostStride = std::max("
                    "biasHostStride, aligned * sizeof(" +
                    biasPlan->mappedType + "));\n";
        }
        code += "            outputHostStride = std::max("
                "outputHostStride, aligned * sizeof(" +
                outputPlan->mappedType + "));\n";
        code += "        }\n";
        code += "        const dacpp_host::BufferSpec specs[] = {\n";
        code += "            {" + std::to_string(inputPlan->bufferId) +
                ", dacpp_host::Access::Read, " +
                std::to_string(KernelElementBytes(inputPlan->mappedType)) +
                ", 0, 0, true, dacpp_host::Distribution::Replicated, inputHostStride, \"input\"},\n";
        code += "            {" + std::to_string(weightPlan->bufferId) +
                ", dacpp_host::Access::Read, " +
                std::to_string(KernelElementBytes(weightPlan->mappedType)) +
                ", 0, 0, false, dacpp_host::Distribution::Sharded, weightHostStride, \"weight\"},\n";
        if (biasPlan) {
            code += "            {" + std::to_string(biasPlan->bufferId) +
                    ", dacpp_host::Access::Read, " +
                    std::to_string(KernelElementBytes(biasPlan->mappedType)) +
                    ", 0, 0, false, dacpp_host::Distribution::Sharded, biasHostStride, \"bias\"},\n";
        }
        code += "            {" + std::to_string(outputPlan->bufferId) +
                ", dacpp_host::Access::Write, " +
                std::to_string(KernelElementBytes(outputPlan->mappedType)) +
                ", 0, 0, false, dacpp_host::Distribution::Sharded, outputHostStride, \"output\"},\n";
        code += "        };\n";
        code += "        if (!" + routeName +
                ".prepare(specs, sizeof(specs) / sizeof(specs[0]))) return " +
                appName + ".finish();\n";
        const std::string runRankName = "runRank";
        code += "        auto " + runRankName +
                " = [&](uint32_t id, aclrtContext ctx, aclrtStream stream) {\n";
        code += "                const dacpp_host::RankArgs &r = " +
                workersName + "[id];\n";
        code += "                (void)ctx;\n";
        code += BuildPerRankTilingCode(exprId, "                ");
        code += "                const size_t inputBytes = inputFeatures * sizeof(" +
                inputPlan->mappedType + ");\n";
        code += "                const size_t weightBytes = inputFeatures * static_cast<size_t>(tile.len) * sizeof(" +
                weightPlan->mappedType + ");\n";
        if (biasPlan) {
            code += "                const size_t biasBytes = static_cast<size_t>(tile.len) * sizeof(" +
                    biasPlan->mappedType + ");\n";
        }
        code += "                const size_t outputBytes = static_cast<size_t>(tile.len) * sizeof(" +
                outputPlan->mappedType + ");\n";
        auto AppendAcquire = [&](const std::string& name,
                                 const HostParamPlan& plan) {
            code += "                uint8_t *" + name +
                    "Host = " + routeName + ".host(id, " +
                    std::to_string(plan.bufferId) + ", " + name + "Bytes);\n";
            code += "                uint8_t *" + name +
                    "Device = " + routeName + ".device(id, " +
                    std::to_string(plan.bufferId) + ", " + name + "Bytes);\n";
            code += "                if (!" + name + "Host || !" +
                    name + "Device) { " + routeName +
                    ".fail(id, dacpp_host::Stage::Alloc, 1); return; }\n";
        };
        AppendAcquire("input", *inputPlan);
        AppendAcquire("weight", *weightPlan);
        if (biasPlan) AppendAcquire("bias", *biasPlan);
        AppendAcquire("output", *outputPlan);
        code += "                auto *inputValues = reinterpret_cast<" +
                inputPlan->mappedType + "*>(inputHost);\n";
        code += "                auto *weightValues = reinterpret_cast<" +
                weightPlan->mappedType + "*>(weightHost);\n";
        if (biasPlan) {
            code += "                auto *biasValues = reinterpret_cast<" +
                    biasPlan->mappedType + "*>(biasHost);\n";
        }
        code += "                auto *outputValues = reinterpret_cast<" +
                outputPlan->mappedType + "*>(outputHost);\n";
        code += "                std::memset(weightHost, 0, weightBytes);\n";
        if (biasPlan) {
            code += "                std::memset(biasHost, 0, biasBytes);\n";
        }
        code += "                std::memset(outputHost, 0, outputBytes);\n";
        code += "                std::memset(inputHost, 0, inputBytes);\n";
        code += "                const size_t inputCopyCount = std::min(inputFeatures, " +
                inputBuf + ".data.size());\n";
        code += "                if (inputCopyCount > 0) { std::memcpy(inputValues, " +
                inputBuf + ".data.data(), inputCopyCount * sizeof(" +
                inputPlan->mappedType + ")); }\n";
        code += "                for (size_t out = 0; out < r.len; ++out) {\n";
        code += "                    const size_t globalOut = r.begin + out;\n";
        code += "                    for (size_t in = 0; in < inputFeatures; ++in) {\n";
        code += "                        const size_t src = globalOut * inputFeatures + in;\n";
        code += "                        weightValues[out * inputFeatures + in] = src < " +
                weightBuf + ".data.size() ? " + weightBuf +
                ".data[src] : static_cast<" + weightPlan->mappedType + ">(0);\n";
        code += "                    }\n";
        if (biasPlan) {
            code += "                    biasValues[out] = globalOut < " + biasBuf +
                    ".data.size() ? " + biasBuf +
                    ".data[globalOut] : static_cast<" + biasPlan->mappedType +
                    ">(0);\n";
        }
        code += "                    outputValues[out] = static_cast<" +
                outputPlan->mappedType + ">(0);\n";
        code += "                }\n";
        code += "                " + routeName + ".copyIn(id, inputDevice, inputBytes, inputHost, inputBytes, stream);\n";
        code += "                " + routeName + ".copyIn(id, weightDevice, weightBytes, weightHost, weightBytes, stream);\n";
        if (biasPlan) {
            code += "                " + routeName + ".copyIn(id, biasDevice, biasBytes, biasHost, biasBytes, stream);\n";
        }
        code += "                " + routeName + ".launch(id, " +
                GetLaunchWrapperName() + "(blockDim" + exprId +
                ", stream, inputDevice, weightDevice, " +
                (biasPlan ? "biasDevice" : "nullptr") +
                ", outputDevice, tile.len, blockDim" +
                exprId + ", tile.size, tile.count));\n";
        code += "                " + routeName + ".copyOut(id, outputHost, outputBytes, outputDevice, outputBytes, stream);\n";
        code += "                " + routeName + ".sync(id, stream);\n";
        code += "                if (r.len > 0) { std::memcpy(" +
                outputBuf + ".data.data() + r.begin, outputHost, " +
                "r.len * sizeof(" + outputPlan->mappedType + ")); }\n";
        code += "        };\n";
        code += "        if (!" + routeName +
                ".run(static_cast<uint32_t>(" + workersName + ".size())" +
                ", " + runRankName + ")) return " + appName + ".finish();\n";
        code += IndentCodeBlock(BuildHostWriteBackOnlyCode(), "    ");
        return {MakeRuntimeBlock(
            FlowGeneratedCodeRole::MulticardLaunch,
            "linear.multicard.runtime",
            std::move(code))};
    }

    std::vector<dacppTranslator::ascendc::FlowGeneratedCodeBlock>
    BuildCompactPointwiseScalarLoopMulticardRuntimeBlocks(
        const MulticardAnalysis& mc) {
        using dacppTranslator::ascendc::FlowGeneratedCodeBlock;
        using dacppTranslator::ascendc::FlowGeneratedCodeRole;
        const std::string& appName = AppName();
        const std::string& routeName = RouteName();
        const HostParamPlan* scalarPlan =
            FindHostParamPlanByName(pointwiseScalarLoopSchedule.scalarParamName);
        const HostParamPlan* outputPlan =
            FindHostParamPlanByName(pointwiseScalarLoopSchedule.outputParamName);
        if (!scalarPlan || !outputPlan) {
            return {MakeRuntimeBlock(
                FlowGeneratedCodeRole::MulticardLaunch,
                "pointwise.scalar.error",
                "    " + appName +
                ".fail(dacpp_host::Stage::HostFlow, 1, \"scalar binding\");\n"
                "    return " + appName + ".finish();\n")};
        }

        const std::string exprId = std::to_string(index);
        const std::string scalarCurrent = "loopState.value";
        const std::string outputBuffer = HostBufferName(outputPlan->name);
        const std::string loopInit = MaterializePointwiseLoopExpr(
            pointwiseScalarLoopSchedule.scalarInitExpr, scalarCurrent, "0");
        const std::string loopUpper = MaterializePointwiseLoopExpr(
            pointwiseScalarLoopSchedule.loopUpperBoundExpr,
            scalarCurrent, loopInit);
        const std::string loopStep = MaterializePointwiseLoopExpr(
            pointwiseScalarLoopSchedule.loopStepExpr, scalarCurrent, "1");
        const std::string compare =
            pointwiseScalarLoopSchedule.loopCompareOp.empty()
                ? "<=" : pointwiseScalarLoopSchedule.loopCompareOp;
        const std::string snapshot = BuildPointwiseScalarLoopSnapshotCopyCode(
            outputBuffer, scalarCurrent);
        const uint32_t align = hasVectorTilingPlan
            ? vectorTilingPlan.tileAlignElements : 1;
        const uint32_t maxTile = hasVectorTilingPlan
            ? vectorTilingPlan.maxTileElements : 1;

        std::string code;
        const std::string commonDecls = TrimCopy(GetCommonDecls());
        if (!commonDecls.empty()) code += IndentCodeBlock(commonDecls, "    ");
        code += "        const size_t len = static_cast<size_t>(kernelLength" +
                exprId + ");\n";
        code += "        const dacpp_host::RankSpan ranks = "
                "dacpp_host::splitRanks(len, " + routeName + ".size());\n";
        code += "        const size_t maxRankLen = "
                "(len + ranks.size() - 1) / ranks.size();\n";
        code += "        const dacpp_host::Tile slabTile = dacpp_host::makeTile("
                + appName + ", maxRankLen, blockDim" + exprId + ", " +
                std::to_string(align) + ", " + std::to_string(maxTile) + ");\n";
        for (const HostParamPlan& plan : hostParamPlans) {
            const std::string buffer = HostBufferName(plan.name);
            const bool broadcast = IsMulticardBroadcastScalarParam(plan);
            const std::string storage = broadcast
                ? "static_cast<size_t>(slabTile.len)"
                : "((static_cast<size_t>(slabTile.len) + " +
                    std::to_string(plan.leftHalo + plan.rightHalo) + " + " +
                    buffer + ".alignElems - 1) / " + buffer +
                    ".alignElems) * " + buffer + ".alignElems";
            code += "        const size_t " + plan.name + "Stride = (" +
                    storage + ") * sizeof(" + plan.mappedType + ");\n";
        }
        code += "        const dacpp_host::BufferSpec specs[] = {\n";
        for (const HostParamPlan& plan : hostParamPlans) {
            const char* access = plan.rw == 0 ? "Read"
                : (plan.rw == 1 ? "Write" : "ReadWrite");
            const bool broadcast = IsMulticardBroadcastScalarParam(plan);
            code += "            {" + std::to_string(plan.bufferId) +
                    ", dacpp_host::Access::" + access + ", " +
                    std::to_string(KernelElementBytes(plan.mappedType)) + ", " +
                    std::to_string(plan.leftHalo) + ", " +
                    std::to_string(plan.rightHalo) + ", " +
                    (broadcast ? "true" : "false") +
                    ", dacpp_host::Distribution::" +
                    (broadcast ? "Replicated" : "Sharded") + ", " +
                    plan.name + "Stride, \"" + plan.name + "\"},\n";
        }
        code += "        };\n";
        code += "        if (!" + routeName +
                ".prepare(specs, sizeof(specs) / sizeof(specs[0]))) return " +
                appName + ".finish();\n";
        code += "        dacpp_host::PersistentControl<" +
                scalarPlan->mappedType + "> loopState(static_cast<" +
                scalarPlan->mappedType + ">(" + loopInit + "));\n";
        code += "        while (" + scalarCurrent + " " + compare +
                " static_cast<" + scalarPlan->mappedType + ">(" + loopUpper +
                ")) {\n";
        code += "            if (!" + routeName +
                ".run(static_cast<uint32_t>(ranks.size()), "
                "[&](uint32_t id, aclrtContext ctx, aclrtStream stream) {\n";
        code += "                (void)ctx;\n";
        code += "                const dacpp_host::RankArgs &r = ranks[id];\n";
        code += "                const dacpp_host::Tile tile = dacpp_host::makeTile("
                + appName + ", r.len, blockDim" + exprId + ", " +
                std::to_string(align) + ", " + std::to_string(maxTile) + ");\n";
        for (const HostParamPlan& plan : hostParamPlans) {
            const std::string buffer = HostBufferName(plan.name);
            const bool broadcast = IsMulticardBroadcastScalarParam(plan);
            const std::string storage = broadcast
                ? "static_cast<size_t>(tile.len)"
                : "((static_cast<size_t>(tile.len) + " +
                    std::to_string(plan.leftHalo + plan.rightHalo) + " + " +
                    buffer + ".alignElems - 1) / " + buffer +
                    ".alignElems) * " + buffer + ".alignElems";
            code += "                const size_t " + plan.name +
                    "Count = " + storage + ";\n";
            code += "                const size_t " + plan.name + "Bytes = " +
                    plan.name + "Count * sizeof(" + plan.mappedType + ");\n";
            code += "                uint8_t *" + plan.name + "Host = " +
                    routeName + ".host("
                    "id, " + std::to_string(plan.bufferId) + ", " +
                    plan.name + "Bytes);\n";
            code += "                uint8_t *" + plan.name +
                    "Device = " + routeName + ".device(id, " +
                    std::to_string(plan.bufferId) + ", " + plan.name +
                    "Bytes);\n";
            code += "                if (!" + plan.name + "Host || !" +
                    plan.name + "Device) { " + routeName + ".fail(id, "
                    "dacpp_host::Stage::Alloc, 1); return; }\n";
            const std::string typed = "reinterpret_cast<" + plan.mappedType +
                "*>(" + plan.name + "Host)";
            if (broadcast) {
                const std::string value = plan.name == scalarPlan->name
                    ? scalarCurrent
                    : (buffer + ".data.empty() ? static_cast<" +
                       plan.mappedType + ">(0) : " + buffer + ".data[0]");
                code += "                std::fill_n(" + typed + ", " +
                        plan.name + "Count, " + value + ");\n";
            } else if (plan.rw == 0 || plan.rw == 2) {
                code += "                std::fill_n(" + typed + ", " +
                        plan.name + "Count, static_cast<" + plan.mappedType +
                        ">(0));\n";
                code += "                const size_t " + plan.name +
                        "Begin = r.begin > " + std::to_string(plan.leftHalo) +
                        " ? r.begin - " + std::to_string(plan.leftHalo) +
                        " : 0;\n";
                code += "                const size_t " + plan.name +
                        "End = std::min(r.begin + r.len + " +
                        std::to_string(plan.rightHalo) + ", " + buffer +
                        ".data.size());\n";
                code += "                if (" + plan.name + "End > " +
                        plan.name + "Begin) std::memcpy(" + typed + ", " +
                        buffer + ".data.data() + " + plan.name + "Begin, (" +
                        plan.name + "End - " + plan.name +
                        "Begin) * sizeof(" + plan.mappedType + "));\n";
            } else {
                code += "                std::memset(" + plan.name +
                        "Host, 0, " + plan.name + "Bytes);\n";
            }
            if (plan.rw == 0 || plan.rw == 2) {
                code += "                " + routeName +
                        ".copyIn(id, " + plan.name + "Device, " +
                        plan.name + "Bytes, " + plan.name + "Host, " +
                        plan.name + "Bytes, stream);\n";
            }
        }
        std::string launchArgs;
        for (const HostParamPlan& plan : hostParamPlans) {
            if (!launchArgs.empty()) launchArgs += ", ";
            launchArgs += plan.name + "Device";
            if (!IsMulticardBroadcastScalarParam(plan) &&
                (plan.rw == 0 || plan.rw == 2) && plan.leftHalo > 0) {
                launchArgs += " + " + std::to_string(plan.leftHalo) +
                              " * sizeof(" + plan.mappedType + ")";
            }
        }
        if (!launchArgs.empty()) launchArgs += ", ";
        launchArgs += "tile.len, blockDim" + exprId + ", tile.size, tile.count";
        code += "                " + routeName + ".launch(id, " +
                GetLaunchWrapperName() + "(blockDim" +
                exprId + ", stream, " + launchArgs + "));\n";
        for (const HostParamPlan& plan : hostParamPlans) {
            if (!(plan.rw == 1 || plan.rw == 2)) continue;
            code += "                " + routeName +
                    ".copyOut(id, " + plan.name + "Host, " +
                    plan.name + "Bytes, " + plan.name + "Device, " +
                    plan.name + "Bytes, stream);\n";
        }
        code += "                if (!" + routeName +
                ".sync(id, stream)) return;\n";
        for (const HostParamPlan& plan : hostParamPlans) {
            if (!(plan.rw == 1 || plan.rw == 2)) continue;
            const std::string buffer = HostBufferName(plan.name);
            code += "                const size_t " + plan.name +
                    "Write = std::min(r.len, " + buffer +
                    ".data.size() - std::min(r.begin, " + buffer +
                    ".data.size()));\n";
            code += "                if (" + plan.name + "Write) std::memcpy(" +
                    buffer + ".data.data() + r.begin, reinterpret_cast<const " +
                    plan.mappedType + "*>(" + plan.name + "Host), " +
                    plan.name + "Write * sizeof(" + plan.mappedType + "));\n";
        }
        code += "            })) return " + appName + ".finish();\n";
        if (!snapshot.empty())
            code += IndentCodeBlock(TrimCopy(snapshot), "        ");
        code += "            loopState.advance(static_cast<" +
                scalarPlan->mappedType + ">(" + loopStep + "));\n";
        code += "        }\n";
        code += IndentCodeBlock(BuildHostWriteBackOnlyCode(), "    ");
        return {MakeRuntimeBlock(
            FlowGeneratedCodeRole::MulticardLaunch,
            "pointwise.scalar.compact",
            std::move(code))};
    }

    std::string BuildHostVectorWriteBackForPlan(const HostParamPlan& plan,
                                                const std::string& bufferName) const {
        auto EmitWriteBackForAlias = [&](const HostTensorAliasInfo& aliasInfo) {
            const std::string rootSourceVar = aliasInfo.rootContainerName;
            const std::string writeBackOffsetExpr =
                aliasInfo.offsetExpr.empty() || aliasInfo.offsetExpr == "0"
                    ? "static_cast<size_t>(0)"
                    : "static_cast<size_t>(" + aliasInfo.offsetExpr + ")";
            const bool sourceMatchesMappedType =
                !aliasInfo.elementType.empty() &&
                MapToStdType(aliasInfo.elementType) == plan.mappedType;
            return BuildDirectVectorWriteBack(
                rootSourceVar, bufferName, writeBackOffsetExpr,
                plan.mappedType, plan.name + "Final", plan.useFloatBitBridge,
                false, false, "", "", sourceMatchesMappedType,
                "            ");
        };

        if (UsesLoopCarriedVectorSchedule() &&
            plan.name == loopCarriedVectorSchedule.inputParamName &&
            !loopCarriedVectorSchedule.inputArgName.empty()) {
            auto it = hostPreludeCollection.tensorAliases.find(
                BuildHostTensorAliasKey(
                    loopCarriedVectorSchedule.ownerFunctionName,
                    TrimCopy(loopCarriedVectorSchedule.inputArgName)));
            if (it != hostPreludeCollection.tensorAliases.end()) {
                const HostTensorAliasInfo& aliasInfo = it->second;
                if (aliasInfo.valid && !aliasInfo.rootContainerName.empty()) {
                    return EmitWriteBackForAlias(aliasInfo);
                }
            }
        }

        for (const auto& aliasEntry : hostPreludeCollection.tensorAliases) {
            const HostTensorAliasInfo& aliasInfo = aliasEntry.second;
            if (!aliasInfo.valid ||
                aliasInfo.ownerExpr != index ||
                aliasInfo.runtimeBufferName != bufferName ||
                aliasInfo.rootContainerName.empty()) {
                continue;
            }
            return EmitWriteBackForAlias(aliasInfo);
        }
        return "";
    }

    std::vector<dacppTranslator::ascendc::FlowGeneratedCodeBlock>
    BuildCompactLoopCarriedMulticardRuntimeBlocks(
        const MulticardAnalysis& mc) {
        using dacppTranslator::ascendc::FlowGeneratedCodeRole;
        const std::string& appName = AppName();
        const std::string& routeName = RouteName();
        const HostParamPlan* input =
            FindHostParamPlanByName(loopCarriedVectorSchedule.inputParamName);
        const HostParamPlan* output =
            FindHostParamPlanByName(loopCarriedVectorSchedule.outputParamName);
        if (!input || !output) {
            return {MakeRuntimeBlock(
                FlowGeneratedCodeRole::MulticardLaunch,
                "loop-carried.error",
                "    " + appName +
                ".fail(dacpp_host::Stage::HostFlow, 1, \"loop binding\");\n"
                "    return " + appName + ".finish();\n")};
        }
        const std::string exprId = std::to_string(index);
        const std::string inputBuf = HostBufferName(input->name);
        const std::string outputBuf = HostBufferName(output->name);
        const uint32_t align = hasVectorTilingPlan
            ? vectorTilingPlan.tileAlignElements : 1;
        const uint32_t maxTile = hasVectorTilingPlan
            ? vectorTilingPlan.maxTileElements : 1;
        auto RankLocals = [&]() {
            std::string body;
            body += "const dacpp_host::RankArgs &rank = ranks[rankId];\n";
            body += "const dacpp_host::Tile tile = dacpp_host::makeTile(" +
                    appName + ", rank.len, blockDim" + exprId + ", " +
                    std::to_string(align) + ", " + std::to_string(maxTile) +
                    ");\n";
            for (const HostParamPlan* plan :
                 std::vector<const HostParamPlan*>{input, output}) {
                const std::string buffer = HostBufferName(plan->name);
                const int halos = plan == input
                    ? plan->leftHalo + plan->rightHalo : 0;
                body += "const size_t " + plan->name +
                        "Count = ((static_cast<size_t>(tile.len) + " +
                        std::to_string(halos) + " + " + buffer +
                        ".alignElems - 1) / " + buffer + ".alignElems) * " +
                        buffer + ".alignElems;\n";
                body += "const size_t " + plan->name + "Bytes = " +
                        plan->name + "Count * sizeof(" + plan->mappedType +
                        ");\n";
                body += "uint8_t *" + plan->name + "Host = " + routeName +
                        ".host(rankId, " +
                        std::to_string(plan->bufferId) + ", " + plan->name +
                        "Bytes);\n";
                body += "uint8_t *" + plan->name +
                        "Device = " + routeName + ".device(rankId, " +
                        std::to_string(plan->bufferId) + ", " + plan->name +
                        "Bytes);\n";
                body += "if (!" + plan->name + "Host || !" + plan->name +
                        "Device) { " + routeName + ".fail(rankId, "
                        "dacpp_host::Stage::Alloc, 1); return; }\n";
            }
            return body;
        };

        std::string code;
        const std::string commonDecls = TrimCopy(GetCommonDecls());
        if (!commonDecls.empty()) code += IndentCodeBlock(commonDecls, "    ");
        code += "        const size_t len = std::min(static_cast<size_t>("
                "kernelLength" + exprId + "), " + inputBuf +
                ".logicalElems > " + std::to_string(input->rightHalo) +
                " ? " + inputBuf + ".logicalElems - " +
                std::to_string(input->rightHalo) + " : 0);\n";
        code += "        const dacpp_host::RankSpan ranks = "
                "dacpp_host::splitRanks(len, " + routeName + ".size());\n";
        code += "        const size_t maxRankLen = "
                "(len + ranks.size() - 1) / ranks.size();\n";
        code += "        const dacpp_host::Tile slabTile = dacpp_host::makeTile("
                + appName + ", maxRankLen, blockDim" + exprId + ", " +
                std::to_string(align) + ", " + std::to_string(maxTile) + ");\n";
        code += "        const size_t inputStride = ((slabTile.len + " +
                std::to_string(input->leftHalo + input->rightHalo) + " + " +
                inputBuf + ".alignElems - 1) / " + inputBuf +
                ".alignElems) * " + inputBuf + ".alignElems * sizeof(" +
                input->mappedType + ");\n";
        code += "        const size_t outputStride = ((slabTile.len + " +
                outputBuf + ".alignElems - 1) / " + outputBuf +
                ".alignElems) * " + outputBuf + ".alignElems * sizeof(" +
                output->mappedType + ");\n";
        code += "        const dacpp_host::BufferSpec specs[] = {\n";
        code += "            {" + std::to_string(input->bufferId) +
                ", dacpp_host::Access::ReadWrite, " +
                std::to_string(KernelElementBytes(input->mappedType)) + ", " +
                std::to_string(input->leftHalo) + ", " +
                std::to_string(input->rightHalo) +
                ", false, dacpp_host::Distribution::Sharded, inputStride, \"" +
                input->name + "\"},\n";
        code += "            {" + std::to_string(output->bufferId) +
                ", dacpp_host::Access::Write, " +
                std::to_string(KernelElementBytes(output->mappedType)) +
                ", 0, 0, false, dacpp_host::Distribution::Sharded, "
                "outputStride, \"" + output->name + "\"},\n";
        code += "        };\n";
        code += "        if (!" + routeName + ".prepare(specs, 2)) return " +
                appName + ".finish();\n";
        code += "        dacpp_host::TemporalControl loopState;\n";
        code += "        if (!" + routeName +
                ".run(static_cast<uint32_t>(ranks.size()), "
                "[&](uint32_t rankId, aclrtContext ctx, aclrtStream stream) {\n";
        code += "            (void)ctx;\n";
        code += IndentCodeBlock(TrimCopy(RankLocals()), "        ");
        code += "            std::memset(" + input->name + "Host, 0, " +
                input->name + "Bytes);\n";
        code += "            const size_t copyEnd = std::min(rank.begin + "
                "rank.len + " + std::to_string(input->rightHalo) + ", " +
                inputBuf + ".data.size());\n";
        code += "            if (copyEnd > rank.begin) std::memcpy(" +
                input->name + "Host, " + inputBuf +
                ".data.data() + rank.begin, (copyEnd - rank.begin) * sizeof(" +
                input->mappedType + "));\n";
        code += "            " + routeName + ".copyIn(rankId, " +
                input->name + "Device, " + input->name + "Bytes, " +
                input->name + "Host, " + input->name +
                "Bytes, stream);\n";
        code += "            " + routeName + ".sync(rankId, stream);\n";
        code += "        })) return " + appName + ".finish();\n";
        code += "        std::vector<" + output->mappedType + "> tails("
                "ranks.size(), static_cast<" + output->mappedType +
                ">(0));\n";
        code += "        for (uint32_t " +
                loopCarriedVectorSchedule.loopIndexVar +
                " = static_cast<uint32_t>(" +
                loopCarriedVectorSchedule.loopBeginExpr + "); " +
                loopCarriedVectorSchedule.loopIndexVar +
                " < static_cast<uint32_t>(" +
                loopCarriedVectorSchedule.loopUpperBoundExpr + "); ++" +
                loopCarriedVectorSchedule.loopIndexVar + ") {\n";
        code += "            loopState.beginGeneration();\n";
        code += "            if (!" + routeName +
                ".run(static_cast<uint32_t>(ranks.size()), "
                "[&](uint32_t rankId, aclrtContext ctx, aclrtStream stream) {\n";
        code += "                (void)ctx;\n";
        code += IndentCodeBlock(TrimCopy(RankLocals()), "            ");
        code += "                if (!rank.len) return;\n";
        code += "                " + routeName + ".launch(rankId, " +
                GetLaunchWrapperName() + "(blockDim" +
                exprId + ", stream, " + input->name + "Device, " +
                output->name + "Device, tile.len, blockDim" + exprId +
                ", tile.size, tile.count));\n";
        code += "                " + routeName + ".copyOut(rankId, " +
                output->name + "Host, sizeof(" + output->mappedType + "), " +
                output->name + "Device + (rank.len - 1) * sizeof(" +
                output->mappedType + "), sizeof(" + output->mappedType +
                "), stream);\n";
        code += "                if (!" + routeName +
                ".sync(rankId, stream)) return;\n";
        code += "                tails[rankId] = *reinterpret_cast<const " +
                output->mappedType + "*>(" + output->name + "Host);\n";
        code += "            })) return " + appName + ".finish();\n";
        code += "            loopState.beginApply();\n";
        code += "            if (!" + routeName +
                ".run(static_cast<uint32_t>(ranks.size()), "
                "[&](uint32_t rankId, aclrtContext ctx, aclrtStream stream) {\n";
        code += "                (void)ctx;\n";
        code += IndentCodeBlock(TrimCopy(RankLocals()), "            ");
        code += "                if (!rank.len) return;\n";
        code += "                const size_t shift = rank.len - "
                "(rank.begin + rank.len >= len ? 1 : 0);\n";
        code += "                if (shift) " + routeName +
                ".copyDevice(rankId, " + input->name + "Device + sizeof(" +
                input->mappedType + "), shift * sizeof(" +
                output->mappedType + "), " + output->name +
                "Device, shift * sizeof(" + output->mappedType +
                "), stream);\n";
        code += "                if (rankId == 0) " + routeName +
                ".copyDevice(rankId, " + input->name + "Device, sizeof(" +
                input->mappedType + "), " + output->name +
                "Device, sizeof(" + output->mappedType + "), stream);\n";
        code += "                else { *reinterpret_cast<" + input->mappedType +
                "*>(" + input->name + "Host) = static_cast<" +
                input->mappedType +
                ">(tails[rankId - 1]); " + routeName +
                ".copyIn(rankId, " + input->name + "Device, sizeof(" +
                input->mappedType + "), " + input->name +
                "Host, sizeof(" + input->mappedType + "), stream); }\n";
        code += "                " + routeName + ".sync(rankId, stream);\n";
        code += "            })) return " + appName + ".finish();\n";
        code += "        }\n";
        code += "        loopState.requestStop();\n";
        code += "        if (!" + routeName +
                ".run(static_cast<uint32_t>(ranks.size()), "
                "[&](uint32_t rankId, aclrtContext ctx, aclrtStream stream) {\n";
        code += "            (void)ctx;\n";
        code += IndentCodeBlock(TrimCopy(RankLocals()), "        ");
        code += "            " + routeName + ".copyOut(rankId, " +
                input->name + "Host, " + input->name + "Bytes, " +
                input->name + "Device, " + input->name +
                "Bytes, stream);\n";
        code += "            if (!" + routeName +
                ".sync(rankId, stream)) return;\n";
        code += "            if (rank.len) std::memcpy(" + inputBuf +
                ".data.data() + rank.begin, " + input->name +
                "Host, rank.len * sizeof(" + input->mappedType + "));\n";
        code += "        })) return " + appName + ".finish();\n";
        code += BuildHostVectorWriteBackForPlan(*input, inputBuf);
        return {MakeRuntimeBlock(
            FlowGeneratedCodeRole::MulticardLaunch,
            "loop-carried.compact",
            std::move(code))};
    }

    std::vector<dacppTranslator::ascendc::FlowGeneratedCodeBlock>
    BuildMulticardRuntimeBlocks() {
        const MulticardAnalysis mc = AnalyzeMulticardPlan();
        if (!mc.enabled) {
            return {};
        }
        if (UsesLinearOutputShardSchedule()) {
            return BuildDirectLinearMulticardRuntimeBlocks();
        }
        if (UsesPointwiseScalarLoopSchedule()) {
            return BuildCompactPointwiseScalarLoopMulticardRuntimeBlocks(mc);
        }
        if (UsesLoopCarriedVectorSchedule()) {
            return BuildCompactLoopCarriedMulticardRuntimeBlocks(mc);
        }
        return BuildDirectMulticardRuntimeBlocks(mc);
    }

    std::vector<dacppTranslator::ascendc::FlowGeneratedCodeBlock>
    BuildRuntimeBlocks(
        const ASCENDC_TEMPLATE::TemporalLoopSchedule* temporalSpec,
        const std::string& temporalLaunchEntry,
        bool dacInsideForLoop) {
        using dacppTranslator::ascendc::FlowGeneratedCodeBlock;
        using dacppTranslator::ascendc::FlowGeneratedCodeRole;

        (void)dacInsideForLoop;
        std::vector<FlowGeneratedCodeBlock> blocks;
        const bool temporalRecognized =
            temporalSpec && temporalSpec->recognized;
        const bool usePointwiseScalarLoopSchedule =
            pointwiseScalarLoopSchedule.recognized &&
            dominantFamily == dacppTranslator::ascendc_mlir::CalcSemanticFamily::PointwiseFamily &&
            transferPolicy == dacppTranslator::ascendc_mlir::AscendCTransferPolicy::PointwiseDirect;

        if (temporalRecognized || usePointwiseScalarLoopSchedule) {
            // 特殊调度（temporal / pointwise scalar loop）：launch wrapper 内部已
            // 把 H2D copy / kernel launch / D2H copy 编织成一个时间步循环，无法在
            // 不破坏调度语义的前提下细分 H2D / launch / D2H。拆分为：CommonDecl
            // + TemporalLoopBody + DeviceFree。
            const std::string commonDecls = TrimCopy(GetCommonDecls());
            const std::string scheduleNpuCode = temporalRecognized
                                                    ? GetNpuCode(temporalSpec, temporalLaunchEntry)
                                                    : GetNpuCode();
            const std::string scheduleNpuCodeTrim = TrimCopy(scheduleNpuCode);
            const std::string scheduleCpuCodeTrim = TrimCopy(GetCpuCode());
            {
                std::string code = "    {\n";
                if (!commonDecls.empty()) {
                    code += IndentCodeBlock(commonDecls, "    ");
                }
                blocks.push_back(MakeRuntimeBlock(
                    FlowGeneratedCodeRole::CommonDecl,
                    temporalRecognized ? "decl.temporal" : "decl.pointwise",
                    std::move(code)));
            }
            {
                std::string code = "    #ifndef ASCENDC_CPU_DEBUG\n";
                if (!scheduleNpuCodeTrim.empty()) {
                    code += IndentCodeBlock(scheduleNpuCodeTrim, "    ");
                }
                code += "    #else\n";
                if (!scheduleCpuCodeTrim.empty()) {
                    code += IndentCodeBlock(scheduleCpuCodeTrim, "    ");
                }
                code += "    #endif\n";
                blocks.push_back(MakeRuntimeBlock(
                    FlowGeneratedCodeRole::TemporalLoopBody,
                    temporalRecognized ? "schedule.temporal" : "schedule.pointwise",
                    std::move(code)));
            }
            blocks.push_back(MakeRuntimeBlock(
                FlowGeneratedCodeRole::DeviceFree,
                temporalRecognized ? "free.temporal" : "free.pointwise",
                "    }\n"));
            return blocks;
        }

        // 常规 lowering 把 GetNpuCode 中的各阶段切到独立 role-block：
        //   CommonDecl       — { 开 + 基础参数与数据定义（含已嵌入的注释 header）
        //   DeviceAlloc      — NPU 执行流注释 + ptrs + aclrtMalloc
        //   HostToDeviceCopy — npuMemcpysIn
        //   KernelLaunch     — kernel_do(...) + aclrtSynchronizeStream
        //   DeviceToHostCopy — npuMemcpysOut（含 D2H 后的 resize/cast loop）
        //   DeviceFree       — npuFrees + }
        //
        // 缩进契约：legacy `BuildHostExecBlock` 做 TrimCopy + IndentCodeBlock("    ")，
        // 让首行落在 4-space、后续含 4-space 原始缩进的行落在 8-space。本处复刻该模式 —
        // 每个 role 的 body 起头若有可识别注释/小标题，把它放在去掉前导空格的位置，由
        // IndentCodeBlock 统一补齐；data members（npuPtrs/npuAllocs/...）保留它们的
        // 4-space 原始缩进。
        const std::string commonDeclsTrimmed = TrimCopy(GetCommonDecls());
        const std::string& npuPtrsRaw    = npuPtrs;    // 已含 4-space 行首
        const std::string& npuAllocsRaw  = npuAllocs;
        const std::string& npuInRaw      = npuMemcpysIn;
        const std::string& npuOutRaw     = npuMemcpysOut;
        const std::string& npuFreesRaw   = npuFrees;

        // CommonDecl
        {
            std::string code = "    {\n";
            if (!commonDeclsTrimmed.empty()) {
                code += IndentCodeBlock(commonDeclsTrimmed, "    ");
            }
            blocks.push_back(MakeRuntimeBlock(
                FlowGeneratedCodeRole::CommonDecl, "decl", std::move(code)));
        }

        // DeviceAlloc —— NPU 执行流注释 + ptrs + mallocs。NPU 注释放在去掉前导空格
        // 的位置，让 IndentCodeBlock 补到 4-space；ptrs/allocs 自带 4-space → 8-space。
        {
            std::string body = "//  算子 " + kernelName + " NPU 执行流 \n";
            body += "#ifndef ASCENDC_CPU_DEBUG\n";
            if (!npuPtrsRaw.empty())   body += npuPtrsRaw;
            if (!npuAllocsRaw.empty()) body += npuAllocsRaw;
            body += "#else\n";
            body += cpuAllocs;
            body += "#endif\n";
            blocks.push_back(MakeRuntimeBlock(
                FlowGeneratedCodeRole::DeviceAlloc, "alloc",
                IndentCodeBlock(body, "    ")));
        }

        // HostToDeviceCopy
        if (!TrimCopy(npuInRaw).empty() || !TrimCopy(cpuMemcpysIn).empty()) {
            std::string body = "#ifndef ASCENDC_CPU_DEBUG\n" + npuInRaw +
                               "#else\n" + cpuMemcpysIn + "#endif\n";
            blocks.push_back(MakeRuntimeBlock(
                FlowGeneratedCodeRole::HostToDeviceCopy, "h2d",
                IndentCodeBlock(body, "    ")));
        }

        // KernelLaunch —— 真实 launch 调用 + stream sync，仅此而已。
        // 给 launch 调用本身打 4-space 前缀，再交 IndentCodeBlock 加 4-space → 8-space。
        {
            std::string body = "#ifndef ASCENDC_CPU_DEBUG\n";
            body +=
                "    " + DeviceName() + ".launch(" + kernelName +
                "_do(blockDim" + std::to_string(index) + ", " +
                StreamName() + ", " + npuArgs + "));\n";
            body += "    " + DeviceName() + ".sync();\n";
            body += "#else\n";
            body += "    ICPU_RUN_KF(" + kernelName + ", blockDim" +
                    std::to_string(index) + ", " + cpuArgs + ");\n";
            body += "#endif\n";
            blocks.push_back(MakeRuntimeBlock(
                FlowGeneratedCodeRole::KernelLaunch, "launch",
                IndentCodeBlock(body, "    ")));
        }

        // DeviceToHostCopy
        if (!TrimCopy(npuOutRaw).empty() || !TrimCopy(cpuMemcpysOut).empty()) {
            std::string body = "#ifndef ASCENDC_CPU_DEBUG\n" + npuOutRaw +
                               "#else\n" + cpuMemcpysOut + "#endif\n";
            blocks.push_back(MakeRuntimeBlock(
                FlowGeneratedCodeRole::DeviceToHostCopy, "d2h",
                IndentCodeBlock(body, "    ")));
        }

        // DeviceFree —— frees 之后关闭作用域 `    }`。
        {
            std::string code = "    #ifndef ASCENDC_CPU_DEBUG\n";
            if (!TrimCopy(npuFreesRaw).empty()) {
                code += IndentCodeBlock(npuFreesRaw, "    ");
            }
            code += "    #else\n";
            if (!TrimCopy(cpuFrees).empty()) {
                code += IndentCodeBlock(cpuFrees, "    ");
            }
            code += "    #endif\n";
            code += "    }\n";
            blocks.push_back(MakeRuntimeBlock(
                FlowGeneratedCodeRole::DeviceFree, "free", std::move(code)));
        }
        return blocks;
    }
};


// ===== Cube lowering =====
// 矩阵算子代码生成器 (Cube 矩阵计算框架) 
class CubeOperatorGenerator {
private:
    struct CubeRuntimeBindingSlot {
        std::string logicalName;
        int shellParamIndex = -1;
        RuntimeParamBindingInfo runtimeInfo;
        bool hasRuntimeVector = false;
        bool hasScalarExpr = false;
        std::string sourceVar;
        std::string scalarExpr;
    };

    Expression* expr;
    int index;
    clang::Rewriter* rewriter;
    const TranslationPolicy& policy;
    std::string sourceText;

    std::string kernelName;
    std::string generatedKernelFileName;

    bool hasPlanCubeMatmulStep = false;
    dacppTranslator::ascendc_mlir::AscendCPlanStep cubeMatmulStep;

    //  动态解析的算子属性 
    std::string typeA = "half", typeB = "half", typeC = "float";
    std::string userTypeA = "float", userTypeB = "float", userTypeC = "float";
    std::string nameA = "a", nameB = "b", nameC = "c";
    std::string transA = "false", transB = "false";
    std::string dimM = "16", dimN = "16", dimK = "16"; // 默认兜底维度
    bool accumulateToDst = true;
    bool hasInitialDst = true;
    bool cubeAccumulateSemantic = true;
    bool cubeUseBiasAccumulate = false;

    dacppTranslator::ascendc_mlir::CalcShellSummary shellSummary;
    std::vector<std::string> resolvedShellCallArgs;
    HostPreludeCollection hostPreludeCollection;
    std::string hostPreludeCode;
    dacppTranslator::ascendc::HostNameAllocator hostNames;
    std::string tilingName;
    std::string shapeTilingName;

    CubeRuntimeBindingSlot bindingA;
    CubeRuntimeBindingSlot bindingB;
    CubeRuntimeBindingSlot bindingC;

    const std::string& AppName() const {
        return gCubeRuntimeNames.session;
    }

    const std::string& RouteName() const {
        return gCubeRuntimeNames.routeObject;
    }

    const std::string& DeviceName() const {
        return gCubeRuntimeNames.device;
    }

    const std::string& StreamName() const {
        return gCubeRuntimeNames.stream;
    }

    dacppTranslator::ascendc::FlowBufferId
    RuntimeBufferId(const std::string& logicalName) const {
        return gRuntimeBufferIds.get(
            dacppTranslator::ascendc::FlowRouteKind::Cube,
            "cube:" + logicalName);
    }

    static uint32_t StagingBytes(const std::string& typeName) {
        if (typeName == "uint16_t" || typeName == "int16_t" ||
            typeName == "half") return 2;
        if (typeName == "uint8_t" || typeName == "int8_t" ||
            typeName == "char") return 1;
        if (typeName == "double" || typeName == "uint64_t" ||
            typeName == "int64_t") return 8;
        return 4;
    }

    bool IsTransposedA() const { return transA == "true"; }
    bool IsTransposedB() const { return transB == "true"; }

    std::string BuildToStagingExpr(const std::string& valueExpr,
                                   const std::string& userType,
                                   const std::string& stagingType) const {
        const std::string userHostType = MapToCubeUserHostType(userType);
        const bool fpToHalfBits =
            stagingType == "uint16_t" && (userHostType == "float" || userHostType == "double");
        if (fpToHalfBits) {
            return "static_cast<uint16_t>(dacpp_host::fp32ToFp16("
                   "static_cast<float>(" + valueExpr + ")))";
        }
        return "static_cast<" + stagingType + ">(" + valueExpr + ")";
    }

    std::string BuildFromStagingExpr(const std::string& valueExpr,
                                     const std::string& userType,
                                     const std::string& stagingType) const {
        const std::string userHostType = MapToCubeUserHostType(userType);
        const bool halfBitsToFp =
            stagingType == "uint16_t" && (userHostType == "float" || userHostType == "double");
        if (halfBitsToFp) {
            return "static_cast<" + userHostType +
                   ">(dacpp_host::fp16ToFp32(" + valueExpr + "))";
        }
        return "static_cast<" + userHostType + ">(" + valueExpr + ")";
    }

    std::string GetEmittedKernelName() const {
        return policy.enableMulticard ? (kernelName + "_n") : kernelName;
    }

    static int FindShellParamIndexByName(Shell* shell, const std::string& paramName) {
        if (!shell || paramName.empty()) {
            return -1;
        }
        for (int i = 0; i < shell->getNumParams(); ++i) {
            ShellParam* shellParam = shell->getShellParam(i);
            if (!shellParam) {
                continue;
            }
            if (shellParam->getName() == paramName) {
                return i;
            }
        }
        return -1;
    }

    void TryFillMissingDimsFromShellShapes(Shell* shell) {
        auto FindShape2D = [&](const std::string& paramName, int& d0, int& d1) -> bool {
            d0 = 0;
            d1 = 0;
            int idx = FindShellParamIndexByName(shell, paramName);
            if (idx < 0) {
                return false;
            }
            ShellParam* p = shell->getShellParam(idx);
            if (!p || p->getDim() < 2) {
                return false;
            }
            int s0 = p->getShape(0);
            int s1 = p->getShape(1);
            if (s0 <= 0 || s1 <= 0) {
                return false;
            }
            d0 = s0;
            d1 = s1;
            return true;
        };

        int a0 = 0;
        int a1 = 0;
        int b0 = 0;
        int b1 = 0;
        if (FindShape2D(nameA, a0, a1)) {
            int rows = a0;
            int cols = a1;
            if (IsTransposedA()) {
                std::swap(rows, cols);
            }
            if (!IsPositiveLength(dimM)) {
                dimM = std::to_string(rows);
            }
            if (!IsPositiveLength(dimK)) {
                dimK = std::to_string(cols);
            }
        }
        if (FindShape2D(nameB, b0, b1)) {
            int rows = b0;
            int cols = b1;
            if (IsTransposedB()) {
                std::swap(rows, cols);
            }
            if (!IsPositiveLength(dimK)) {
                dimK = std::to_string(rows);
            }
            if (!IsPositiveLength(dimN)) {
                dimN = std::to_string(cols);
            }
        }
    }

    void TryFillMissingDimsFromSourceFallback() {
        std::string aDim1;
        std::string aDim2;
        std::string bDim1;
        std::string bDim2;
        ExtractMatrixDimsFromSource(sourceText, nameA, aDim1, aDim2);
        ExtractMatrixDimsFromSource(sourceText, nameB, bDim1, bDim2);

        if (!aDim1.empty() && !aDim2.empty()) {
            std::string mExpr = aDim1;
            std::string kExpr = aDim2;
            if (IsTransposedA()) {
                std::swap(mExpr, kExpr);
            }
            if (!IsPositiveLength(dimM) && IsPositiveLength(mExpr)) {
                dimM = mExpr;
            }
            if (!IsPositiveLength(dimK) && IsPositiveLength(kExpr)) {
                dimK = kExpr;
            }
        }
        if (!bDim1.empty() && !bDim2.empty()) {
            std::string kExpr = bDim1;
            std::string nExpr = bDim2;
            if (IsTransposedB()) {
                std::swap(kExpr, nExpr);
            }
            if (!IsPositiveLength(dimK) && IsPositiveLength(kExpr)) {
                dimK = kExpr;
            }
            if (!IsPositiveLength(dimN) && IsPositiveLength(nExpr)) {
                dimN = nExpr;
            }
        }
    }

    void BindOneRuntimeSlot(Shell* shell,
                            const clang::Stmt* bindingScopeRoot,
                            const std::function<std::string(const std::string&)>& materializeScalarExpr,
                            std::unordered_set<std::string>& ioCapturedSourceVars,
                            CubeRuntimeBindingSlot& ioSlot) {
        ioSlot.runtimeInfo = RuntimeParamBindingInfo();
        ioSlot.shellParamIndex = FindShellParamIndexByName(shell, ioSlot.logicalName);

        std::vector<std::string> candidates;
        if (ioSlot.shellParamIndex >= 0 &&
            ioSlot.shellParamIndex < static_cast<int>(resolvedShellCallArgs.size())) {
            std::string argExpr = TrimCopy(resolvedShellCallArgs[ioSlot.shellParamIndex]);
            if (IsSimpleIdentifierText(argExpr)) {
                candidates.push_back(argExpr);
            }
        }
        if (!ioSlot.logicalName.empty()) {
            candidates.push_back(ioSlot.logicalName);
        }

        std::unordered_set<std::string> seen;
        for (const std::string& candidate : candidates) {
            std::string name = TrimCopy(candidate);
            if (name.empty() || seen.count(name) > 0) {
                continue;
            }
            seen.insert(name);
            TryBindSourceCandidateFromAst(
                bindingScopeRoot,
                name,
                materializeScalarExpr,
                ioSlot.runtimeInfo,
                ioCapturedSourceVars);
            if (ioSlot.runtimeInfo.sourceIsStdVector || ioSlot.runtimeInfo.hasScalarSourceExpr) {
                break;
            }
        }

        if (ioSlot.shellParamIndex >= 0) {
            ShellParam* shellParam = shell ? shell->getShellParam(ioSlot.shellParamIndex) : nullptr;
            std::string shapeLength;
            if (TryGetShapeTotalLength(shellParam, shapeLength)) {
                ioSlot.runtimeInfo.hasShapeLength = true;
                ioSlot.runtimeInfo.shapeLength = shapeLength;
            }
        }

        const bool hasRootStdVector =
            ioSlot.runtimeInfo.bindingResolved &&
            ioSlot.runtimeInfo.bindingKind == RuntimeBindingNodeKind::StdVector &&
            !ioSlot.runtimeInfo.rootSourceVar.empty();

        ioSlot.hasRuntimeVector = hasRootStdVector;
        ioSlot.sourceVar = hasRootStdVector
            ? ioSlot.runtimeInfo.rootSourceVar
            : (ioSlot.runtimeInfo.sourceIsStdVector ? ioSlot.runtimeInfo.sourceVar : "");

        ioSlot.hasScalarExpr = ioSlot.runtimeInfo.hasScalarSourceExpr && !ioSlot.runtimeInfo.scalarSourceExpr.empty();
        ioSlot.scalarExpr = ioSlot.hasScalarExpr ? ioSlot.runtimeInfo.scalarSourceExpr : "";
    }

    void EmitRuntimeInitForOperand(std::string& ioCode,
                                   const std::string& dataPtr,
                                   const std::string& elemCountVar,
                                   const std::string& userType,
                                   const std::string& stagingType,
                                   const CubeRuntimeBindingSlot& slot,
                                   bool allowScalarFill) const {
        if (slot.hasRuntimeVector && !slot.sourceVar.empty()) {
            ioCode += "    {\n";
            ioCode += "        size_t runtimeCount = std::min(" + elemCountVar + ", " + slot.sourceVar + ".size());\n";
            ioCode += "        for (size_t i = 0; i < runtimeCount; ++i) {\n";
            ioCode += "            " + dataPtr + "[i] = " +
                BuildToStagingExpr(slot.sourceVar + "[i]", userType, stagingType) + ";\n";
            ioCode += "        }\n";
            ioCode += "    }\n";
            return;
        }

        if (allowScalarFill && slot.hasScalarExpr) {
            ioCode += "    {\n";
            ioCode += "        " + stagingType + " scalarValue = " +
                BuildToStagingExpr(slot.scalarExpr, userType, stagingType) + ";\n";
            ioCode += "        for (size_t i = 0; i < " + elemCountVar + "; ++i) { " + dataPtr + "[i] = scalarValue; }\n";
            ioCode += "    }\n";
            return;
        }

        ioCode += "    // Runtime binding not resolved for " + slot.logicalName + ", keep zero-initialized staging buffer.\n";
    }

    static bool IsTruthyEnv(const char* value) {
        if (!value) {
            return false;
        }
        std::string text(value);
        std::transform(text.begin(), text.end(), text.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return text == "1" || text == "true" || text == "on" || text == "yes";
    }

    static bool ResolveCubeBiasAccumulateSupport() {
        // 优先使用显式开关；默认 conservative fallback（Host 侧回加 C_init）。
        if (IsTruthyEnv(std::getenv("DACPP_ASCENDC_CUBE_BIAS_ACCUM"))) {
            return true;
        }
        return false;
    }
public:
    CubeOperatorGenerator(Expression* e,
                          int idx,
                          clang::Rewriter* rw,
                          const std::string& srcText,
                          const TranslationPolicy& policy,
                          std::string kernelFileName)
        : expr(e),
          index(idx),
          rewriter(rw),
          policy(policy),
          sourceText(srcText),
          generatedKernelFileName(std::move(kernelFileName)) {
        kernelName = "cube_kernel_" + std::to_string(index);
        const clang::BinaryOperator* dacExpr = expr ? expr->getDacExpr() : nullptr;
        const clang::FunctionDecl* owner = FindEnclosingFunction(dacExpr);
        if (owner && gRuntimeBindingTUDecl) {
            for (const std::string& name :
                 dacppTranslator::ascendc::CollectHostVisibleIdentifiers(
                     *owner, *gRuntimeBindingTUDecl)) {
                hostNames.reserve(name);
            }
        }
        for (const std::string& name :
             {gCubeRuntimeNames.session, gCubeRuntimeNames.rankCount,
              gCubeRuntimeNames.routeObject, gCubeRuntimeNames.device,
              gCubeRuntimeNames.stream}) {
            hostNames.reserve(name);
        }
        hostNames.pushScope("expr." + std::to_string(index));
        const std::string suffix =
            index == 0 ? std::string() : std::to_string(index + 1);
        tilingName = hostNames.get(
            dacppTranslator::ascendc::HostNameRole::HelperFunction,
            "makeTiling" + suffix).text;
        shapeTilingName = hostNames.get(
            dacppTranslator::ascendc::HostNameRole::HelperFunction,
            "makeShapeTiling" + suffix).text;
        if (!hostNames.ok() && gHostNameError.empty()) {
            gHostNameError = hostNames.error();
        }
    }

    void SetCubeMatmulStep(const dacppTranslator::ascendc_mlir::AscendCPlanStep& step) {
        hasPlanCubeMatmulStep = true;
        cubeMatmulStep = step;
    }

    void BindPlanStep() {
        Shell* shell = expr->getShell();
        if (!shell) {
            return;
        }

        std::vector<ShellParam*> inParams;
        std::vector<ShellParam*> outParams;
        for (int j = 0; j < shell->getNumParams(); j++) {
            ShellParam* param = shell->getShellParam(j);
            if (!param) {
                continue;
            }
            const int rw = static_cast<int>(param->getRw());
            if (rw == 0) {
                inParams.push_back(param);
            } else if (rw == 1 || rw == 2) {
                outParams.push_back(param);
            }
        }

        if (hasPlanCubeMatmulStep && cubeMatmulStep.hasCubeMatmulStepPayload) {
            const auto& payload = cubeMatmulStep.cubeMatmulStepPayload;
            nameA = payload.srcA.empty() ? nameA : payload.srcA;
            nameB = payload.srcB.empty() ? nameB : payload.srcB;
            nameC = payload.dst.empty() ? nameC : payload.dst;
            accumulateToDst = payload.accumulateToDst;
            hasInitialDst = payload.hasInitialDst;

            userTypeA = payload.userTypeA.empty() ? userTypeA : payload.userTypeA;
            userTypeB = payload.userTypeB.empty() ? userTypeB : payload.userTypeB;
            userTypeC = payload.userTypeC.empty() ? userTypeC : payload.userTypeC;
            typeA = payload.kernelTypeA.empty() ? MapToCubeKernelType(userTypeA, false) : payload.kernelTypeA;
            typeB = payload.kernelTypeB.empty() ? MapToCubeKernelType(userTypeB, false) : payload.kernelTypeB;
            typeC = payload.kernelTypeC.empty() ? MapToCubeKernelType(userTypeC, true) : payload.kernelTypeC;

            if (!payload.dimM.empty()) {
                dimM = payload.dimM;
            }
            if (!payload.dimN.empty()) {
                dimN = payload.dimN;
            }
            if (!payload.dimK.empty()) {
                dimK = payload.dimK;
            }
            transA = payload.transA ? "true" : "false";
            transB = payload.transB ? "true" : "false";
        } else {
            if (inParams.size() >= 2) {
                nameA = inParams[0]->getName();
                userTypeA = inParams[0]->getBasicType();
                typeA = MapToCubeKernelType(userTypeA, false);
                nameB = inParams[1]->getName();
                userTypeB = inParams[1]->getBasicType();
                typeB = MapToCubeKernelType(userTypeB, false);
            }
            if (outParams.size() >= 1) {
                nameC = outParams[0]->getName();
                userTypeC = outParams[0]->getBasicType();
                typeC = MapToCubeKernelType(userTypeC, true);
            }

            dimM.clear();
            dimN.clear();
            dimK.clear();

            Calc* calc = expr->getCalc();
            if (calc && calc->getCalcLoc()) {
                clang::SourceRange range = calc->getCalcLoc()->getSourceRange();
                std::string rawCalcExpr = clang::Lexer::getSourceText(
                    clang::CharSourceRange::getTokenRange(range),
                    rewriter->getSourceMgr(), rewriter->getLangOpts()).str();
                if (rawCalcExpr.find(nameA + ".T") != std::string::npos ||
                    rawCalcExpr.find("transpose(" + nameA + ")") != std::string::npos) {
                    transA = "true";
                }
                if (rawCalcExpr.find(nameB + ".T") != std::string::npos ||
                    rawCalcExpr.find("transpose(" + nameB + ")") != std::string::npos) {
                    transB = "true";
                }
            }
        }

        bindingA.logicalName = nameA;
        bindingB.logicalName = nameB;
        bindingC.logicalName = nameC;
    }

    void ResolveRuntimeBindings() {
        Shell* shell = expr->getShell();
        if (!shell) {
            return;
        }

        shellSummary = dacppTranslator::ascendc_mlir::BuildCalcShellSummary(shell);
        resolvedShellCallArgs.clear();
        hostPreludeCollection = HostPreludeCollection();
        hostPreludeCode.clear();
        bindingA = CubeRuntimeBindingSlot();
        bindingB = CubeRuntimeBindingSlot();
        bindingC = CubeRuntimeBindingSlot();
        bindingA.logicalName = nameA;
        bindingB.logicalName = nameB;
        bindingC.logicalName = nameC;

        const clang::BinaryOperator* dacExpr = expr->getDacExpr();
        gRuntimeBindingCurrentDacExprLoc = dacExpr ? dacExpr->getExprLoc() : clang::SourceLocation();
        gRuntimeBindingCurrentOwnerFunction = FindEnclosingFunction(dacExpr);
        const clang::Stmt* bindingScopeRoot =
            gRuntimeBindingCurrentOwnerFunction ? gRuntimeBindingCurrentOwnerFunction->getBody() : nullptr;

        resolvedShellCallArgs = ResolveShellCallArgumentsFromAst(dacExpr, shell);

        auto MaterializeScalar = [&](const std::string& rawExpr) {
            return MaterializeScalarExpr(sourceText, rawExpr);
        };

        std::unordered_set<std::string> capturedSourceVars;
        BindOneRuntimeSlot(shell, bindingScopeRoot, MaterializeScalar, capturedSourceVars, bindingA);
        BindOneRuntimeSlot(shell, bindingScopeRoot, MaterializeScalar, capturedSourceVars, bindingB);
        BindOneRuntimeSlot(shell, bindingScopeRoot, MaterializeScalar, capturedSourceVars, bindingC);

        std::unordered_set<std::string> rootSourceVars = capturedSourceVars;
        for (const CubeRuntimeBindingSlot* slot : {&bindingA, &bindingB, &bindingC}) {
            if (slot->runtimeInfo.bindingResolved &&
                slot->runtimeInfo.bindingKind == RuntimeBindingNodeKind::StdVector &&
                !slot->runtimeInfo.rootSourceVar.empty()) {
                rootSourceVars.insert(slot->runtimeInfo.rootSourceVar);
            }
        }

        BuildHostPreludeCollection(
            gRuntimeBindingCurrentOwnerFunction,
            gRuntimeBindingCurrentDacExprLoc,
            rootSourceVars,
            hostPreludeCollection);
        hostPreludeCode = BuildHostPreludeCodeFromCollection(hostPreludeCollection);
    }

    void ParseParameters() {
        BindPlanStep();
        ResolveRuntimeBindings();

        Shell* shell = expr->getShell();
        if (!shell) {
            return;
        }

        // M/N/K 来源优先级：plan payload -> shellSummary/shell shape -> 源码 fallback
        TryFillMissingDimsFromShellShapes(shell);
        TryFillMissingDimsFromSourceFallback();
        if (!IsPositiveLength(dimM)) {
            dimM = "1";
        }
        if (!IsPositiveLength(dimN)) {
            dimN = "1";
        }
        if (!IsPositiveLength(dimK)) {
            dimK = "1";
        }

        cubeAccumulateSemantic = accumulateToDst && hasInitialDst;
        cubeUseBiasAccumulate = cubeAccumulateSemantic && ResolveCubeBiasAccumulateSupport();

        if (policy.debugOutput) {
            std::cout << "提取到矩阵维度 M=" << dimM << ", N=" << dimN
                      << ", K=" << dimK << std::endl;
        }
        if (cubeAccumulateSemantic && policy.debugOutput) {
            std::cout << "[CubeAccumulate] mode="
                      << (cubeUseBiasAccumulate ? "kernel-bias" : "host-post-add")
                      << std::endl;
        }
    }

    void GenerateKernelFile() {
        // Kernel 由模板统一发射，host 参数初始化由 runtime binding 提供。
        const std::string emittedKernelName = GetEmittedKernelName();
        std::string kernelCode = ASCENDC_TEMPLATE::CodeGen_CubeKernelFile(
            emittedKernelName,
            typeA,
            typeB,
            typeC,
            nameA,
            nameB,
            nameC,
            transA,
            transB,
            cubeUseBiasAccumulate ? "true" : "false"
        );

        WriteOutputFile(generatedKernelFileName, kernelCode, "Cube Kernel侧成功生成: ");
    }

    std::string GetTilingFunc() { 
        if (!IsPositiveLength(dimM)) {
            dimM = "1";
        }
        if (!IsPositiveLength(dimN)) {
            dimN = "1";
        }
        if (!IsPositiveLength(dimK)) {
            dimK = "1";
        }

        std::string tilingDtypeA = MapToTilingDtype(typeA);
        std::string tilingDtypeB = MapToTilingDtype(typeB);
        std::string tilingDtypeC = MapToTilingDtype(typeC);

        return ASCENDC_TEMPLATE::CodeGen_CubeTilingFunc(
            tilingName,
            shapeTilingName,
            dimM,
            dimN,
            dimK,
            tilingDtypeA,
            tilingDtypeB,
            tilingDtypeC,
            transA,
            transB,
            cubeUseBiasAccumulate ? "true" : "false"
        );
    }
    
    std::string GetExternDoDecl() {
        return "#include \"aclrtlaunch_" + GetEmittedKernelName() + ".h\"\n"; 
    }
    
    std::string GetExternCpuDecl() { 
        return "extern \"C\" void " + kernelName + "(uint8_t *a, uint8_t *b, uint8_t *c, uint8_t *workspace, uint8_t *tiling);\n"; 
    }

    std::string GetHostPreludeCode() const { return hostPreludeCode; }
    const HostPreludeCollection& GetHostPreludeCollection() const { return hostPreludeCollection; }

    //  单独提供一个 Tiling 声明的接口 
    std::string GetTilingDecl() {
        return "extern \"C\" uint8_t *" + tilingName +
               "(const char *socVersion);\n"
               "extern \"C\" uint8_t *" + shapeTilingName +
               "(const char *socVersion, int M, int N, int K, int usedCoreNum);\n";
    }

    const std::string& GetTilingName() const { return tilingName; }
    const std::string& GetShapeTilingName() const { return shapeTilingName; }

    std::string GetCommonDecls() { 
        const std::string stagingTypeA = MapToCubeDeviceStagingType(typeA);
        const std::string stagingTypeB = MapToCubeDeviceStagingType(typeB);
        const std::string stagingTypeC = MapToCubeDeviceStagingType(typeC);

        std::string code;
        code += "    //  算子 " + kernelName + " 基础参数与数据定义（plan + runtime binding）\n";
        code += "    const char *socVersion = SOC_VERSION;\n";
        code += "    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);\n";
        code += "    uint32_t blockDim" + std::to_string(index) + " = " + std::to_string(policy.defaultCoreNum) + ";\n";
        code += "\n";
        code += "    const size_t mElemCount = static_cast<size_t>(" + dimM + ");\n";
        code += "    const size_t nElemCount = static_cast<size_t>(" + dimN + ");\n";
        code += "    const size_t kElemCount = static_cast<size_t>(" + dimK + ");\n";
        code += "    const size_t aElemCount = mElemCount * kElemCount;\n";
        code += "    const size_t bElemCount = kElemCount * nElemCount;\n";
        code += "    const size_t cElemCount = mElemCount * nElemCount;\n";
        code += "\n";
        code += "    size_t aFileSize = aElemCount * sizeof(" + stagingTypeA + ");\n";
        code += "    size_t bFileSize = bElemCount * sizeof(" + stagingTypeB + ");\n";
        code += "    size_t cFileSize = cElemCount * sizeof(" + stagingTypeC + ");\n";
        code += "    size_t tilingFileSize = sizeof(TCubeTiling);\n";
        code += "\n";
        code += "    size_t userWorkspaceSize = 0;\n";
        code += "    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());\n";
        code += "    size_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;\n";
        code += "\n";
        code += "    " + stagingTypeA + " *" + nameA + "Data = (" + stagingTypeA + "*)malloc(aFileSize);\n";
        code += "    " + stagingTypeB + " *" + nameB + "Data = (" + stagingTypeB + "*)malloc(bFileSize);\n";
        code += "    " + stagingTypeC + " *" + nameC + "Data = (" + stagingTypeC + "*)malloc(cFileSize);\n";
        code += "\n";
        code += "    memset(" + nameA + "Data, 0, aFileSize);\n";
        code += "    memset(" + nameB + "Data, 0, bFileSize);\n";
        code += "    memset(" + nameC + "Data, 0, cFileSize);\n";
        code += "\n";

        EmitRuntimeInitForOperand(code, nameA + "Data", "aElemCount", userTypeA, stagingTypeA, bindingA, true);
        EmitRuntimeInitForOperand(code, nameB + "Data", "bElemCount", userTypeB, stagingTypeB, bindingB, true);
        if (accumulateToDst && hasInitialDst) {
            EmitRuntimeInitForOperand(code, nameC + "Data", "cElemCount", userTypeC, stagingTypeC, bindingC, false);
        } else {
            code += "    // Destination starts from zero (no initial dst required by plan).\n";
        }

        return code;
    }
    
    // ========================================================
    // Cube runtime tensor alias 构造。
    // ========================================================

    dacppTranslator::ascendc::HostTensorAliasInfo
    BuildCubeRuntimeAliasForOperand(
        const std::string& opName,
        const std::string& opUserType,
        const std::vector<std::string>& shapeExprs,
        const std::string& ownerName) const {
        dacppTranslator::ascendc::HostTensorAliasInfo info;
        info.valid = true;
        info.ownerFunctionName = ownerName;
        info.aliasName = opName;
        info.rootContainerName = opName;
        info.rootDataExpr = opName + "Data";
        info.runtimeBufferName = opName + "Data";
        info.runtimeGenerated = true;
        info.ownerExpr = index;
        info.rank = 2;
        info.currentDim = 0;
        info.elementType = MapToCubeUserHostType(opUserType);
        info.shapeExprs = shapeExprs;
        info.strideExprs = dacppTranslator::ascendc::BuildRowMajorStrideExprs(shapeExprs);
        return info;
    }

    dacppTranslator::ascendc::HostTensorAliasInfo
    BuildCubeRuntimeAliasForA(const std::string& ownerName) const {
        return BuildCubeRuntimeAliasForOperand(
            nameA, userTypeA, {"mElemCount", "kElemCount"}, ownerName);
    }

    dacppTranslator::ascendc::HostTensorAliasInfo
    BuildCubeRuntimeAliasForB(const std::string& ownerName) const {
        return BuildCubeRuntimeAliasForOperand(
            nameB, userTypeB, {"kElemCount", "nElemCount"}, ownerName);
    }

    dacppTranslator::ascendc::HostTensorAliasInfo
    BuildCubeRuntimeAliasForC(const std::string& ownerName) const {
        return BuildCubeRuntimeAliasForOperand(
            nameC, userTypeC, {"mElemCount", "nElemCount"}, ownerName);
    }

    std::vector<dacppTranslator::ascendc::HostTensorAliasInfo>
    BuildCubeRuntimeTensorAliases(const std::string& ownerName) const {
        std::vector<dacppTranslator::ascendc::HostTensorAliasInfo> aliases;
        aliases.push_back(BuildCubeRuntimeAliasForA(ownerName));
        aliases.push_back(BuildCubeRuntimeAliasForB(ownerName));
        aliases.push_back(BuildCubeRuntimeAliasForC(ownerName));
        return aliases;
    }

    // ========================================================
    // Cube 路径的 descriptor 输出。
    // ========================================================

    dacppTranslator::ascendc::FlowKernelLaunch BuildCubeKernelLaunchDescriptor() const {
        dacppTranslator::ascendc::FlowKernelLaunch launch;
        const std::string emittedKernelName = GetEmittedKernelName();
        launch.route = dacppTranslator::ascendc::FlowRouteKind::Cube;
        launch.expressionIndex = index;
        launch.kernelFunctionName = emittedKernelName;
        launch.kernelArtifactFileName = generatedKernelFileName;
        launch.blockDimExpr = "blockDim" + std::to_string(index);
        launch.tilingDataSymbol = tilingName;

        // paramOrder = matmul A/B/C 三操作数（与 cube kernel
        // launch ABI 对齐）。scalarParams 在 Cube 路径下没有 broadcast scalar 概念，
        // 留空。
        if (!nameA.empty()) launch.paramOrder.push_back(nameA);
        if (!nameB.empty()) launch.paramOrder.push_back(nameB);
        if (!nameC.empty()) launch.paramOrder.push_back(nameC);
        auto AddRuntimeBuffer = [&](const std::string& logicalName,
                                    const std::string& elementType,
                                    const std::string& countExpr,
                                    const std::string& bytesExpr,
                                    dacppTranslator::ascendc::FlowBufferAccess access,
                                    dacppTranslator::ascendc::FlowMulticardParamDistribution distribution) {
            dacppTranslator::ascendc::FlowRuntimeBuffer buffer;
            buffer.id = RuntimeBufferId(logicalName);
            buffer.logicalName = logicalName;
            buffer.elementTypeText = elementType;
            buffer.logicalCountExpr = countExpr;
            buffer.byteCountExpr = bytesExpr;
            buffer.hostStrideExpr = bytesExpr;
            buffer.access = access;
            buffer.distribution = distribution;
            launch.runtimeBuffers.push_back(std::move(buffer));
        };
        AddRuntimeBuffer(
            nameA, MapToCubeDeviceStagingType(typeA), "aElemCount",
            "aFileSize", dacppTranslator::ascendc::FlowBufferAccess::Read,
            dacppTranslator::ascendc::FlowMulticardParamDistribution::Sharded);
        AddRuntimeBuffer(
            nameB, MapToCubeDeviceStagingType(typeB), "bElemCount",
            "bFileSize", dacppTranslator::ascendc::FlowBufferAccess::Read,
            dacppTranslator::ascendc::FlowMulticardParamDistribution::Replicated);
        AddRuntimeBuffer(
            nameC, MapToCubeDeviceStagingType(typeC), "cElemCount",
            "cFileSize", dacppTranslator::ascendc::FlowBufferAccess::ReadWrite,
            dacppTranslator::ascendc::FlowMulticardParamDistribution::Sharded);
        AddRuntimeBuffer(
            "tiling", "uint8_t", "tilingFileSize", "tilingFileSize",
            dacppTranslator::ascendc::FlowBufferAccess::Read,
            dacppTranslator::ascendc::FlowMulticardParamDistribution::Replicated);
        AddRuntimeBuffer(
            "workspace", "uint8_t", "workspaceSize", "workspaceSize",
            dacppTranslator::ascendc::FlowBufferAccess::Write,
            dacppTranslator::ascendc::FlowMulticardParamDistribution::Sharded);
        if (policy.enableMulticard) {
            launch.multicardEnabled = true;
            launch.rankLaunchFunctionName = emittedKernelName;
            launch.perRankParamOrder = launch.paramOrder;
            launch.multicard.enabled = true;
            launch.multicard.rankDimExpr = std::to_string(policy.requestedRankDim);
            launch.multicard.requestedRankDimExpr = launch.multicard.rankDimExpr;
            launch.multicard.activeRankDimExpr =
                "min(requestedRanks,shardExtent)";
            launch.multicard.coreNumExpr = std::to_string(policy.defaultCoreNum);
            launch.multicard.deviceMapExpr = "rankId->deviceId=rankId";
            launch.multicard.rankDimEnvName = policy.rankDimEnvName.empty()
                                                  ? "DACPP_ASCENDC_RANK_DIM"
                                                  : policy.rankDimEnvName;
            launch.multicard.shardDim = "M";
            launch.multicard.shardAxis = "M";
            launch.multicard.logicalLengthExpr = dimM;
            launch.multicard.alignedLengthExpr = "rank.localM";
            launch.multicard.strategy = "cube-m-shard";
            launch.multicard.replicatedParams = nameB;
            launch.multicard.shardedParams = nameA + "," + nameC;
            launch.multicard.inputReplicated = true;
            launch.multicard.outputSharded = true;
            launch.multicard.resourceScope = "route";
            launch.multicard.workerLifetime = "persistent-route";
            launch.multicard.copyMode = "async-stream";
            launch.multicard.communication =
                dacppTranslator::ascendc::FlowMulticardCommunicationKind::None;
            for (const std::string& paramName :
                 std::vector<std::string>{nameA, nameB, nameC}) {
                dacppTranslator::ascendc::FlowMulticardParamPartition partition;
                partition.name = paramName;
                partition.bufferId = RuntimeBufferId(paramName);
                partition.distribution = paramName == nameB
                    ? dacppTranslator::ascendc::FlowMulticardParamDistribution::Replicated
                    : dacppTranslator::ascendc::FlowMulticardParamDistribution::Sharded;
                partition.shardAxis = paramName == nameB ? "-" : "M";
                partition.readWriteMode = paramName == nameC ? 1 : 0;
                partition.deviceResident = paramName == nameB;
                launch.multicard.paramPartitions.push_back(std::move(partition));
            }
            launch.launchDescriptorText =
                "multicard rankDim=" + launch.multicard.rankDimExpr +
                " shard=M strategy=cube-m-shard rankLaunch=" +
                launch.rankLaunchFunctionName;
        }
        return launch;
    }

    std::string BuildCubeDeviceAllocCode() const {
        std::string code;
        code += "    //  算子 " + kernelName + " NPU 执行流\n";
        if (cubeAccumulateSemantic && !cubeUseBiasAccumulate) {
            code += "    std::vector<float> " + nameC +
                    "InitSnapshot(cElemCount, 0.0f);\n";
            code += "    for (size_t i = 0; i < cElemCount; ++i) {\n";
            code += "        " + nameC +
                    "InitSnapshot[i] = static_cast<float>(" + nameC +
                    "Data[i]);\n";
            code += "    }\n\n";
        }
        code += "    uint8_t *" + nameA + "Host = nullptr, *" + nameA + "Device = nullptr;\n";
        code += "    " + DeviceName() + ".allocate(" + nameA +
                "Host, " + nameA + "Device, aFileSize);\n";
        code += "    uint8_t *" + nameB + "Host = nullptr, *" + nameB + "Device = nullptr;\n";
        code += "    " + DeviceName() + ".allocate(" + nameB +
                "Host, " + nameB + "Device, bFileSize);\n";
        code += "    uint8_t *workspaceDevice = nullptr;\n";
        code += "    " + DeviceName() + ".allocate(workspaceDevice, workspaceSize);\n";
        code += "    uint8_t *" + nameC + "Host = nullptr, *" + nameC + "Device = nullptr;\n";
        code += "    uint8_t *tilingHost = nullptr, *tilingDevice = nullptr;\n";
        code += "    " + DeviceName() +
                ".allocate(tilingHost, tilingDevice, tilingFileSize);\n";
        code += "    " + DeviceName() + ".allocate(" + nameC +
                "Host, " + nameC + "Device, cFileSize);\n";
        return code;
    }

    std::string BuildCubeHostToDeviceCopyCode() const {
        const std::string emittedKernelName = GetEmittedKernelName();
        std::string code;
        code += "    std::memcpy(" + nameA + "Host, " + nameA + "Data, aFileSize);\n";
        code += "    " + DeviceName() + ".copyIn(" + nameA +
                "Device, aFileSize, " + nameA + "Host, aFileSize);\n";
        code += "    std::memcpy(" + nameB + "Host, " + nameB + "Data, bFileSize);\n";
        code += "    " + DeviceName() + ".copyIn(" + nameB +
                "Device, bFileSize, " + nameB + "Host, bFileSize);\n";
        code += "    uint8_t* generatedTiling = " + tilingName +
                "(socVersion);\n";
        code += "    std::memcpy(tilingHost, generatedTiling, tilingFileSize);\n";
        code += "    " + DeviceName() +
                ".copyIn(tilingDevice, tilingFileSize, tilingHost, tilingFileSize);\n";
        return code;
    }

    std::string BuildCubeKernelLaunchCode() const {
        const std::string bdVar = "blockDim" + std::to_string(index);
        const std::string emittedKernelName = GetEmittedKernelName();
        std::string code;
        code += "    " + DeviceName() + ".launch(ACLRT_LAUNCH_KERNEL(" +
                emittedKernelName + ")(" + bdVar +
                ", " + StreamName() + ", " +
                nameA + "Device, " + nameB + "Device, " + nameC +
                "Device, workspaceDevice, tilingDevice));\n";
        code += "    " + DeviceName() + ".sync();\n";
        return code;
    }

    std::string BuildCubeDeviceToHostCopyCode() const {
        const std::string stagingTypeC = MapToCubeDeviceStagingType(typeC);
        std::string code;
        code += "    " + DeviceName() + ".copyOut(" + nameC +
                "Host, cFileSize, " + nameC + "Device, cFileSize);\n";
        code += "    std::memcpy(" + nameC + "Data, " + nameC + "Host, cFileSize);\n";
        if (cubeAccumulateSemantic && !cubeUseBiasAccumulate) {
            code += "    for (size_t i = 0; i < cElemCount; ++i) {\n";
            code += "        " + nameC + "Data[i] = static_cast<float>(" +
                    nameC + "Data[i]) + " + nameC +
                    "InitSnapshot[i];\n";
            code += "    }\n";
        }
        if (bindingC.hasRuntimeVector && !bindingC.sourceVar.empty()) {
            code += "    if (" + bindingC.sourceVar + ".size() < cElemCount) { " +
                bindingC.sourceVar + ".resize(cElemCount); }\n";
            code += "    for (size_t i = 0; i < cElemCount; ++i) {\n";
            code += "        " + bindingC.sourceVar + "[i] = " +
                BuildFromStagingExpr(nameC + "Data[i]", userTypeC, stagingTypeC) + ";\n";
            code += "    }\n";
        }
        return code;
    }

    std::string BuildCubeDeviceFreeCode() const {
        std::string code;
        code += "    " + DeviceName() + ".release(" + nameA +
                "Host, " + nameA + "Device);\n";
        code += "    " + DeviceName() + ".release(" + nameB +
                "Host, " + nameB + "Device);\n";
        code += "    " + DeviceName() + ".release(" + nameC +
                "Host, " + nameC + "Device);\n";
        code += "    " + DeviceName() + ".release(workspaceDevice);\n";
        code += "    " + DeviceName() +
                ".release(tilingHost, tilingDevice);\n";
        code += "    std::free(generatedTiling);\n";
        return code;
    }

    std::vector<dacppTranslator::ascendc::FlowGeneratedCodeBlock>
    BuildDirectCubeMulticardRuntimeBlocks() {
        using dacppTranslator::ascendc::FlowGeneratedCodeBlock;
        using dacppTranslator::ascendc::FlowGeneratedCodeRole;
        const std::string& appName = AppName();
        const std::string& routeName = RouteName();

        const std::string exprId = std::to_string(index);
        const std::string stagingTypeA = MapToCubeDeviceStagingType(typeA);
        const std::string stagingTypeB = MapToCubeDeviceStagingType(typeB);
        const std::string stagingTypeC = MapToCubeDeviceStagingType(typeC);
        const std::string commonDecls = TrimCopy(GetCommonDecls());
        const std::string emittedKernelName = GetEmittedKernelName();
        std::string code;
        if (!commonDecls.empty()) {
            code += IndentCodeBlock(commonDecls, "    ");
        }

        if (cubeAccumulateSemantic && !cubeUseBiasAccumulate) {
            code += "        std::vector<float> " + nameC +
                    "InitSnapshot(cElemCount, 0.0f);\n";
            code += "        for (size_t i = 0; i < cElemCount; ++i) {\n";
            code += "            " + nameC +
                    "InitSnapshot[i] = static_cast<float>(" + nameC +
                    "Data[i]);\n";
            code += "        }\n";
        }

        code += "        const dacpp_host::RankSpan ranks = "
                "dacpp_host::splitRanks(mElemCount, " + routeName +
                ".size());\n";
        code += "        size_t rankAHostStride = static_cast<size_t>(1);\n";
        code += "        size_t rankCHostStride = static_cast<size_t>(1);\n";
        code += "        for (size_t id = 0; id < ranks.size(); ++id) {\n";
        code += "            const dacpp_host::RankArgs &r = ranks[id];\n";
        code += "            const size_t rankM = r.len;\n";
        code += "            rankAHostStride = std::max("
                "rankAHostStride, rankM * kElemCount * sizeof(" +
                stagingTypeA + "));\n";
        code += "            rankCHostStride = std::max("
                "rankCHostStride, rankM * nElemCount * sizeof(" +
                stagingTypeC + "));\n";
        code += "        }\n";
        code += "        const dacpp_host::BufferSpec specs[] = {\n";
        code += "            {" + std::to_string(RuntimeBufferId(nameA)) +
                ", dacpp_host::Access::Read, " +
                std::to_string(StagingBytes(stagingTypeA)) +
                ", 0, 0, false, dacpp_host::Distribution::Sharded, rankAHostStride, \"" +
                nameA + "\"},\n";
        code += "            {" + std::to_string(RuntimeBufferId(nameB)) +
                ", dacpp_host::Access::Read, " +
                std::to_string(StagingBytes(stagingTypeB)) +
                ", 0, 0, true, dacpp_host::Distribution::Replicated, bFileSize, \"" +
                nameB + "\"},\n";
        code += "            {" + std::to_string(RuntimeBufferId(nameC)) +
                ", dacpp_host::Access::ReadWrite, " +
                std::to_string(StagingBytes(stagingTypeC)) +
                ", 0, 0, false, dacpp_host::Distribution::Sharded, rankCHostStride, \"" +
                nameC + "\"},\n";
        code += "            {" + std::to_string(RuntimeBufferId("tiling")) +
                ", dacpp_host::Access::Read, 1, 0, 0, true, "
                "dacpp_host::Distribution::Replicated, tilingFileSize, \"tiling\"},\n";
        code += "            {" + std::to_string(RuntimeBufferId("workspace")) +
                ", dacpp_host::Access::Write, 1, 0, 0, false, "
                "dacpp_host::Distribution::DeviceOnly, workspaceSize, \"workspace\"},\n";
        code += "        };\n";
        code += "        if (!" + routeName +
                ".prepare(specs, sizeof(specs) / sizeof(specs[0]))) return " +
                appName + ".finish();\n";
        const std::string runRankName = "runRank";
        code += "        auto " + runRankName +
                " = [&](uint32_t id, aclrtContext ctx, aclrtStream stream) {\n";
        code += "                (void)ctx;\n";
        code += "                const dacpp_host::RankArgs &r = ranks[id];\n";
        code += "                const size_t rankM = r.len;\n";
        code += "                const size_t rankAElems = rankM * kElemCount;\n";
        code += "                const size_t rankCElems = rankM * nElemCount;\n";
        code += "                const size_t rankABytes = rankAElems * sizeof(" +
                stagingTypeA + ");\n";
        code += "                const size_t rankCBytes = rankCElems * sizeof(" +
                stagingTypeC + ");\n";
        code += "                const size_t rankAOffset = r.begin * kElemCount;\n";
        code += "                const size_t rankCOffset = r.begin * nElemCount;\n";
        code += "                uint8_t *rankAHost = " + routeName + ".host(id, " + std::to_string(RuntimeBufferId(nameA)) + ", rankABytes);\n";
        code += "                uint8_t *rankADevice = " + routeName + ".device(id, " + std::to_string(RuntimeBufferId(nameA)) + ", rankABytes);\n";
        code += "                uint8_t *rankBHost = " + routeName + ".host(id, " + std::to_string(RuntimeBufferId(nameB)) + ", bFileSize);\n";
        code += "                uint8_t *rankBDevice = " + routeName + ".device(id, " + std::to_string(RuntimeBufferId(nameB)) + ", bFileSize);\n";
        code += "                uint8_t *rankCHost = " + routeName + ".host(id, " + std::to_string(RuntimeBufferId(nameC)) + ", rankCBytes);\n";
        code += "                uint8_t *rankCDevice = " + routeName + ".device(id, " + std::to_string(RuntimeBufferId(nameC)) + ", rankCBytes);\n";
        code += "                uint8_t *workspaceDevice = " + routeName + ".device(id, " + std::to_string(RuntimeBufferId("workspace")) + ", workspaceSize);\n";
        code += "                uint8_t *tilingHost = " + routeName + ".host(id, " + std::to_string(RuntimeBufferId("tiling")) + ", tilingFileSize);\n";
        code += "                uint8_t *tilingDevice = " + routeName + ".device(id, " + std::to_string(RuntimeBufferId("tiling")) + ", tilingFileSize);\n";
        code += "                if (!rankAHost || !rankADevice || !rankBHost || !rankBDevice || !rankCHost || !rankCDevice || !workspaceDevice || !tilingHost || !tilingDevice) { " + routeName + ".fail(id, dacpp_host::Stage::Alloc, 1); return; }\n";
        code += "                std::memcpy(rankAHost, " + nameA +
                "Data + rankAOffset, rankABytes);\n";
        code += "                std::memcpy(rankBHost, " + nameB +
                "Data, bFileSize);\n";
        code += "                " + routeName +
                ".copyIn(id, rankADevice, rankABytes, rankAHost, rankABytes, stream);\n";
        code += "                " + routeName +
                ".copyIn(id, rankBDevice, bFileSize, rankBHost, bFileSize, stream);\n";
        code += "                uint8_t *generatedTiling = " +
                shapeTilingName +
                "(socVersion, static_cast<int>(rankM), static_cast<int>(nElemCount), static_cast<int>(kElemCount), static_cast<int>(blockDim" +
                exprId + "));\n";
        code += "                std::memcpy(tilingHost, generatedTiling, tilingFileSize);\n";
        code += "                std::free(generatedTiling);\n";
        code += "                " + routeName +
                ".copyIn(id, tilingDevice, tilingFileSize, tilingHost, tilingFileSize, stream);\n";
        code += "                " + routeName +
                ".launch(id, ACLRT_LAUNCH_KERNEL(" + emittedKernelName +
                ")(blockDim" + exprId +
                ", stream, rankADevice, rankBDevice, rankCDevice, workspaceDevice, tilingDevice));\n";
        code += "                " + routeName +
                ".copyOut(id, rankCHost, rankCBytes, rankCDevice, rankCBytes, stream);\n";
        code += "                " + routeName + ".sync(id, stream);\n";
        code += "                std::memcpy(" + nameC +
                "Data + rankCOffset, rankCHost, rankCBytes);\n";
        code += "        };\n";
        code += "        if (!" + routeName +
                ".run(static_cast<uint32_t>(ranks.size())" +
                ", " + runRankName + ")) return " + appName + ".finish();\n";

        if (cubeAccumulateSemantic && !cubeUseBiasAccumulate) {
            code += "        for (size_t i = 0; i < cElemCount; ++i) {\n";
            code += "            " + nameC + "Data[i] = static_cast<float>(" +
                    nameC + "Data[i]) + " + nameC + "InitSnapshot[i];\n";
            code += "        }\n";
        }
        if (bindingC.hasRuntimeVector && !bindingC.sourceVar.empty()) {
            code += "        if (" + bindingC.sourceVar +
                    ".size() < cElemCount) { " + bindingC.sourceVar +
                    ".resize(cElemCount); }\n";
            code += "        for (size_t i = 0; i < cElemCount; ++i) {\n";
            code += "            " + bindingC.sourceVar + "[i] = " +
                    BuildFromStagingExpr(
                        nameC + "Data[i]", userTypeC, stagingTypeC) +
                    ";\n";
            code += "        }\n";
        }

        return {MakeFlowCodeBlock(
            FlowGeneratedCodeRole::MulticardLaunch,
            "cube.multicard.runtime",
            std::move(code),
            index)};
    }

    std::vector<dacppTranslator::ascendc::FlowGeneratedCodeBlock>
    BuildCubeRuntimeBlocks() {
        using dacppTranslator::ascendc::FlowGeneratedCodeBlock;
        using dacppTranslator::ascendc::FlowGeneratedCodeRole;

        std::vector<FlowGeneratedCodeBlock> blocks;
        const std::string commonDecls = TrimCopy(GetCommonDecls());

        auto makeBlock = [this](FlowGeneratedCodeRole role,
                                std::string label,
                                std::string code) {
            return MakeFlowCodeBlock(role, std::move(label), std::move(code), index);
        };

        {
            std::string code = "    {\n";
            if (!commonDecls.empty()) {
                code += IndentCodeBlock(commonDecls, "    ");
            }
            blocks.push_back(makeBlock(
                FlowGeneratedCodeRole::CommonDecl, "decl", std::move(code)));
        }
        blocks.push_back(makeBlock(
            FlowGeneratedCodeRole::DeviceAlloc, "alloc",
            IndentCodeBlock(BuildCubeDeviceAllocCode(), "    ")));
        blocks.push_back(makeBlock(
            FlowGeneratedCodeRole::HostToDeviceCopy, "h2d",
            IndentCodeBlock(BuildCubeHostToDeviceCopyCode(), "    ")));
        blocks.push_back(makeBlock(
            FlowGeneratedCodeRole::KernelLaunch, "launch",
            IndentCodeBlock(BuildCubeKernelLaunchCode(), "    ")));
        blocks.push_back(makeBlock(
            FlowGeneratedCodeRole::DeviceToHostCopy, "d2h",
            IndentCodeBlock(BuildCubeDeviceToHostCopyCode(), "    ")));
        {
            std::string code = IndentCodeBlock(BuildCubeDeviceFreeCode(), "    ");
            code += "    }\n";
            blocks.push_back(makeBlock(
                FlowGeneratedCodeRole::DeviceFree, "free", std::move(code)));
        }
        return blocks;
    }
};

static unsigned GetExpressionSourceOrder(Expression* expr, clang::Rewriter* rewriter) {
    if (!expr || !rewriter) {
        return std::numeric_limits<unsigned>::max();
    }
    if (const clang::BinaryOperator* dacExpr = expr->getDacExpr()) {
        clang::SourceLocation exprLoc =
            rewriter->getSourceMgr().getExpansionLoc(dacExpr->getExprLoc());
        if (exprLoc.isValid()) {
            return rewriter->getSourceMgr().getFileOffset(exprLoc);
        }
    }
    return std::numeric_limits<unsigned>::max();
}

static AscendCExpressionArtifact LowerCubeExpressionToArtifact(
    Expression* expr,
    int expressionIndex,
    clang::Rewriter* rewriter,
    const std::string& sourceText,
    const TranslationPolicy& policy,
    const std::string& kernelFileName,
    const dacppTranslator::ascendc_mlir::AscendCPlanStep& cubeMatmulStep) {
    AscendCExpressionArtifact artifact;
    artifact.routeKind = AscendCRouteKind::Cube;
    artifact.runtimeNames = gCubeRuntimeNames;
    PopulateArtifactRuntimeContract(artifact, policy);
    artifact.expressionIndex = expressionIndex;
    artifact.sourceOrder = GetExpressionSourceOrder(expr, rewriter);
    artifact.kernelFileName = kernelFileName;

    CubeOperatorGenerator cubeGen(
        expr, expressionIndex, rewriter, sourceText, policy, kernelFileName);
    cubeGen.SetCubeMatmulStep(cubeMatmulStep);
    cubeGen.ParseParameters();
    cubeGen.GenerateKernelFile();

    artifact.tilingFunc = cubeGen.GetTilingFunc() + "\n";
    artifact.hostPreludeCode = TrimCopy(cubeGen.GetHostPreludeCode());
    artifact.externDoDecl = cubeGen.GetExternDoDecl();
    artifact.tilingDecl = cubeGen.GetTilingDecl();
    for (const std::string& text :
         {cubeGen.GetTilingName(), cubeGen.GetShapeTilingName()}) {
        dacppTranslator::ascendc::FlowGeneratedName name;
        name.text = text;
        name.kind =
            dacppTranslator::ascendc::FlowGeneratedNameKind::FrameworkLocal;
        name.ownerExpr = expressionIndex;
        artifact.generatedNames.push_back(std::move(name));
    }

    // 写入 FlowKernelLaunch + FlowGeneratedCodeBlock descriptor。
    artifact.kernelLaunch = cubeGen.BuildCubeKernelLaunchDescriptor();
    artifact.generatedBlocks = policy.enableMulticard
        ? cubeGen.BuildDirectCubeMulticardRuntimeBlocks()
        : cubeGen.BuildCubeRuntimeBlocks();

    // 构造 runtime tensor alias，让 HostFlowBuilder 能解析 matC.print() 等。
    std::string ownerName;
    const clang::BinaryOperator* dacExpr = expr ? expr->getDacExpr() : nullptr;
    const clang::FunctionDecl* ownerFn = FindEnclosingFunction(dacExpr);
    if (ownerFn) {
        ownerName = ownerFn->getNameAsString();
    }
    artifact.runtimeTensorAliases = cubeGen.BuildCubeRuntimeTensorAliases(ownerName);

    return artifact;
}

static AscendCExpressionArtifact LowerVectorExpressionToArtifact(
    Expression* expr,
    int expressionIndex,
    clang::Rewriter* rewriter,
    const std::string& sourceText,
    const TranslationPolicy& policy,
    const std::string& kernelFileName,
    const ASCENDC_TEMPLATE::TemporalLoopSchedule& temporalSchedule,
    bool& temporalScheduleApplied,
    bool& temporalBridgeBuildFailed,
    std::string& temporalBridgeFailureReason) {
    AscendCExpressionArtifact artifact;
    artifact.routeKind = AscendCRouteKind::Vector;
    artifact.runtimeNames = gVectorRuntimeNames;
    PopulateArtifactRuntimeContract(artifact, policy);
    artifact.expressionIndex = expressionIndex;
    artifact.sourceOrder = GetExpressionSourceOrder(expr, rewriter);
    artifact.kernelFileName = kernelFileName;

    OperatorGenerator generator(
        expr, expressionIndex, rewriter, sourceText, policy, kernelFileName);
    generator.ParseParameters();
    generator.GenerateKernelFile();

    const bool useTemporalForCurrentExpr =
        temporalSchedule.recognized && temporalSchedule.expressionIndex == expressionIndex;

    artifact.hostPreludeCode = TrimCopy(generator.GetHostPreludeCode());
    artifact.externDoDecl = generator.GetExternDoDecl();

    const std::string commonDecls = TrimCopy(generator.GetCommonDecls());

    std::string temporalLaunchEntry;
    if (useTemporalForCurrentExpr) {
        temporalScheduleApplied = true;
        artifact.temporalScheduleApplied = true;
        const std::string wrapperName = generator.GetTemporalWrapperName();
        const std::string wrapperDef = generator.BuildTemporalWrapperFunction(wrapperName, temporalSchedule);
        if (wrapperDef.empty()) {
            temporalBridgeBuildFailed = true;
            temporalBridgeFailureReason =
                "structured schedule missing tensor-kernel bridge for expression #" +
                std::to_string(expressionIndex);
            std::cerr << "[Host侧][Error] " << temporalBridgeFailureReason << std::endl;
            return artifact;
        }
        artifact.hostPreludeCode += "\n" + wrapperDef + "\n";
        temporalLaunchEntry = wrapperName;
        dacppTranslator::ascendc::FlowGeneratedName generatedName;
        generatedName.text = wrapperName;
        generatedName.kind =
            dacppTranslator::ascendc::FlowGeneratedNameKind::FrameworkLocal;
        generatedName.ownerExpr = expressionIndex;
        artifact.generatedNames.push_back(std::move(generatedName));
    }

    // 用 AST parent 链识别 DAC 是否被宿主 ForStmt 包裹，触发 split 路径。
    // （pre-loop alloc / in-loop launch / post-loop free）。
    // 注意：仅 ForStmt 触发 split。WhileStmt/DoStmt 形态由 pointwiseScalarLoopSchedule
    // 的 absorbsOwningWhile 在 GetNpuCode 内部吸收为 temporal wrapper，host body 路径
    // 走 non-split + temporalScheduleApplied=true（builder 跳过外层 host loop）。
    bool dacInsideForLoop = false;
    if (gRuntimeBindingTUDecl) {
        const clang::BinaryOperator* dacExpr = expr ? expr->getDacExpr() : nullptr;
        if (dacExpr) {
            clang::ASTContext& astCtx = gRuntimeBindingTUDecl->getASTContext();
            const clang::Stmt* cur = dacExpr;
            for (int depth = 0; depth < 64 && cur; ++depth) {
                const auto parents = astCtx.getParents(*cur);
                if (parents.empty()) break;
                const clang::Stmt* parentStmt = parents[0].get<clang::Stmt>();
                if (!parentStmt) break;
                if (llvm::isa<clang::ForStmt>(parentStmt)) {
                    dacInsideForLoop = true;
                    break;
                }
                cur = parentStmt;
            }
        }
    }

    if (generator.AbsorbsOwningWhile()) {
        artifact.temporalScheduleApplied = true;
    }
    if (policy.enableMulticard && generator.AbsorbsOwningFor()) {
        artifact.temporalScheduleApplied = true;
    }
    generator.AssignRuntimeBufferIds();
    artifact.kernelLaunch = generator.BuildKernelLaunchDescriptor();
    if (policy.enableMulticard) {
        if (useTemporalForCurrentExpr) {
            artifact.generatedBlocks.push_back(MakeFlowCodeBlock(
                dacppTranslator::ascendc::FlowGeneratedCodeRole::MulticardLaunch,
                "multicard.temporal.unsupported",
                "        " + gVectorRuntimeNames.session +
                    ".fail(dacpp_host::Stage::HostFlow, 1, "
                    "\"multicard temporal schedule\");\n",
                expressionIndex));
        } else {
            artifact.generatedBlocks = generator.BuildMulticardRuntimeBlocks();
        }
    } else {
        artifact.generatedBlocks = generator.BuildRuntimeBlocks(
            useTemporalForCurrentExpr ? &temporalSchedule : nullptr,
            temporalLaunchEntry,
            dacInsideForLoop && !useTemporalForCurrentExpr);
    }

    // setup/cleanup 索引：dac_for 拆分路径下 alloc 是 setup、free 是 cleanup；
    // 普通路径下 emitter 按 generatedBlocks 自身顺序依次发射，无需显式分组。
    if (dacInsideForLoop && !useTemporalForCurrentExpr &&
        artifact.generatedBlocks.size() == 3) {
        artifact.setupBlockIndices = {0};        // DeviceAlloc
        artifact.cleanupBlockIndices = {2};      // DeviceFree
    }

    return artifact;
}


} // namespace

//  主翻译调度流
void dacppTranslator::ascendc::RewriteAscendC(DacppFile* dacppFile, clang::Rewriter* rewriter) {
    std::cout << "DACPP -> AscendC 翻译" << std::endl;
    gRuntimeBufferIds.reset();
    AscendCRewriteContext context(dacppFile, rewriter);
    RuntimeBindingContextGuard runtimeBindingGuard(context);
    gHostNameError.clear();
    const clang::FunctionDecl* hostOwner = nullptr;
    if (gRuntimeBindingTUDecl) {
        for (const clang::Decl* decl : gRuntimeBindingTUDecl->decls()) {
            const auto* fn = llvm::dyn_cast<clang::FunctionDecl>(decl);
            if (fn && fn->getNameAsString() == "main" && fn->hasBody()) {
                hostOwner = fn;
                break;
            }
        }
    }
    gVectorRuntimeNames = AllocateRuntimeNames(
        dacppTranslator::ascendc::FlowRouteKind::Vector,
        hostOwner, gRuntimeBindingTUDecl);
    gCubeRuntimeNames = AllocateRuntimeNames(
        dacppTranslator::ascendc::FlowRouteKind::Cube,
        hostOwner, gRuntimeBindingTUDecl);
    const std::string& sourceText = context.sourceText;
    const TranslationPolicy& policy = context.policy;
    if (!policy.valid) {
        if (rewriter) {
            rewriter->getSourceMgr().getDiagnostics().Report(
                clang::diag::err_fe_error_backend)
                << policy.firstConfigError;
        }
        return;
    }
    if (!gHostNameError.empty()) {
        if (rewriter) {
            rewriter->getSourceMgr().getDiagnostics().Report(
                clang::diag::err_fe_error_backend) << gHostNameError;
        }
        return;
    }
    const dacppTranslator::ascendc_mlir::CalcMlirPipelineOptions& mlirOptions =
        context.mlirOptions;
    std::cout << "[AscendC topology] cores-per-device=" << policy.defaultCoreNum
              << " (" << policy.coreNumSource << ")"
              << ", multicard=" << (policy.enableMulticard ? "on" : "off")
              << ", requested-ranks=" << policy.requestedRankDim
              << " (" << policy.rankDimSource << ")"
              << ", topology=" << policy.requestedRankDim << "x"
              << policy.defaultCoreNum << std::endl;

    ASCENDC_TEMPLATE::TemporalLoopSchedule temporalSchedule;
    TryExtractTemporalLoopScheduleFromAst(dacppFile, temporalSchedule);
    if (temporalSchedule.recognized && policy.debugOutput) {
        std::cout << "[Host侧] TemporalLoopSchedule recognized: "
                  << "expr#" << temporalSchedule.expressionIndex
                  << ", loop(index=" << temporalSchedule.loopIndexVar
                  << ", begin=" << temporalSchedule.loopBeginExpr
                  << ", end=" << temporalSchedule.loopUpperBoundExpr
                  << ", step=" << temporalSchedule.loopStepExpr
                  << ")"
                  << ", state(tensor=" << temporalSchedule.stateTensorName
                  << ", in=" << temporalSchedule.inputSliceVar << "@t" << temporalSchedule.inputTimeOffset
                  << ", out=" << temporalSchedule.outputSliceVar << "@t" << temporalSchedule.outputTimeOffset
                  << ")"
                  << ", writeback(loopIdx=" << temporalSchedule.writebackLoopIndexVar
                  << ", range=[" << temporalSchedule.writebackRangeBeginExpr
                  << "," << temporalSchedule.writebackRangeEndExpr
                  << "], dstSpatialOffset=" << temporalSchedule.writebackDstSpatialOffset
                  << ", dstTimeOffset=" << temporalSchedule.writebackDstTimeOffset
                  << ", srcSpatialOffset=" << temporalSchedule.writebackSrcSpatialOffset
                  << ")"
                  << ", boundary=" << temporalSchedule.boundaryUpdateSummary
                  << std::endl;
    } else if (temporalSchedule.requireNoFallback) {
        std::cerr << "[Host侧][Error] TemporalLoopSchedule detect failed: "
                  << (temporalSchedule.failureReason.empty() ? "unknown reason" : temporalSchedule.failureReason)
                  << std::endl;
    }
    bool temporalScheduleApplied = false;
    bool temporalBridgeBuildFailed = false;
    std::string temporalBridgeFailureReason;

    //  Vector 相关累加器
    std::string vecExternDoDecls = "";
    std::unordered_set<std::string> vecPreludeSeen;
    std::string vecPreludeCode = "";
    bool hasVectorOp = false;

    //  Cube 相关累加器
    std::string cubeExternDoDecls = "";
    std::string cubeTilingFuncs = "";
    std::string cubeTilingDecls = "";
    std::unordered_set<std::string> cubePreludeSeen;
    std::string cubePreludeCode = "";
    bool hasMatrixOp = false;

    // 统一的 artifact 列表替代旧 vecRouteBlocks/cubeRouteBlocks，
    // 让 HostFlowBuilder 一次性构造 plan，再用 routeFilter 过滤为两路发射。
    std::vector<AscendCExpressionArtifact> allArtifacts;
    int generatedKernelFileIndex = 0;

    for(int i = 0; i < dacppFile->getNumExpression(); i++) {
        Expression* expr = dacppFile->getExpression(i);
        
        Shell* shell = expr->getShell();
        const bool isCubeCandidate = ShouldUseCubeRoute(expr, rewriter);
        std::string calcName = expr->getCalc() ? expr->getCalc()->getName() : "Unknown";

        bool hasCubeMatmulStep = false;
        dacppTranslator::ascendc_mlir::AscendCPlanStep cubeMatmulStep;
        std::string cubeFailureReason;
        if (isCubeCandidate && (!expr->getCalc() || !shell)) {
            std::cerr << "[CubeRoute][Error] " << calcName
                      << ": cube candidate is missing calc or shell information" << std::endl;
            return;
        }
        if (isCubeCandidate && expr->getCalc() && shell) {
            std::vector<std::string> cubeShellParamOrder;
            std::vector<std::string> cubeInLocalVars;
            std::vector<std::string> cubeOutLocalVars;
            dacppTranslator::ascendc_mlir::CalcShellSummary cubeShellSummary =
                dacppTranslator::ascendc_mlir::BuildCalcShellSummary(shell);

            for (int sj = 0; sj < shell->getNumParams(); ++sj) {
                ShellParam* shellParam = shell->getShellParam(sj);
                if (!shellParam) {
                    continue;
                }
                cubeShellParamOrder.push_back(shellParam->getName());
                int rw = shellParam->getRw();
                if (rw == 0 || rw == 2) {
                    cubeInLocalVars.push_back(shellParam->getName());
                }
                if (rw == 1 || rw == 2) {
                    cubeOutLocalVars.push_back(shellParam->getName());
                }
            }

            dacppTranslator::ascendc_mlir::CalcAstToMlirOptions lowerOptions;
            lowerOptions.verifyModule = true;
            dacppTranslator::ascendc_mlir::CalcAstToMlirResult lowerResult =
                dacppTranslator::ascendc_mlir::LowerCalcAstToMlir(expr->getCalc(), lowerOptions);

            if (mlirOptions.dumpMlir) {
                llvm::outs() << "[ascendc-mlir] dump mlir module (cube route):\n";
                if (lowerResult.lowered && !lowerResult.moduleText.empty()) {
                    llvm::outs() << lowerResult.moduleText;
                    if (lowerResult.moduleText.back() != '\n') {
                        llvm::outs() << "\n";
                    }
                } else {
                    llvm::outs() << "module {\n}\n";
                }
                if (!lowerResult.diagnostics.empty()) {
                    llvm::outs() << "[ascendc-mlir] diagnostics: " << lowerResult.diagnostics << "\n";
                }
                if (lowerResult.lowered) {
                    llvm::outs() << "[ascendc-mlir] verify: "
                                 << (lowerResult.verified ? "pass" : "fail") << "\n";
                }
            }

            if (lowerResult.lowered) {
                dacppTranslator::ascendc_mlir::CalcAnalysisOptions analysisOptions;
                analysisOptions.shellSummary = &cubeShellSummary;
                analysisOptions.paramOrder = &cubeShellParamOrder;
                dacppTranslator::ascendc_mlir::CalcAnalysisResult analysisResult =
                    dacppTranslator::ascendc_mlir::AnalyzeCalcMlirModule(
                        lowerResult.artifact,
                        analysisOptions);
                if (!lowerResult.diagnostics.empty()) {
                    analysisResult.notes.push_back("lowering diagnostics: " + lowerResult.diagnostics);
                }

                dacppTranslator::ascendc_mlir::AscendCPlanBuildOptions planOptions;
                planOptions.shellSummary = &cubeShellSummary;
                dacppTranslator::ascendc_mlir::AscendCPlan plan =
                    dacppTranslator::ascendc_mlir::BuildAscendCPlan(
                        analysisResult,
                        lowerResult.artifact,
                        cubeShellParamOrder,
                        cubeInLocalVars,
                        cubeOutLocalVars,
                        planOptions);

                if (mlirOptions.dumpAscendCPlan) {
                    llvm::outs() << "[ascendc-mlir] dump ascendc semantic plan (cube route):\n";
                    llvm::outs() << dacppTranslator::ascendc_mlir::DumpAscendCPlan(plan);
                    llvm::outs() << "[ascendc-mlir] dump calc expression graph (cube route):\n";
                    llvm::outs() << dacppTranslator::ascendc_mlir::RenderCalcExprGraph(analysisResult.exprGraph);
                }

                hasCubeMatmulStep = TryExtractCubeMatmulStepFromPlan(plan, cubeMatmulStep);
                if (!hasCubeMatmulStep) {
                    cubeFailureReason = "cube semantic plan did not produce CubeMatmul step";
                    if (!plan.steps.empty()) {
                        cubeFailureReason += "; first-step-kind=" +
                            std::string(dacppTranslator::ascendc_mlir::ToString(plan.steps.front().kind));
                        if (!plan.steps.front().reason.empty()) {
                            cubeFailureReason += "; reason=" + plan.steps.front().reason;
                        }
                    }
                }
            } else {
                cubeFailureReason = lowerResult.diagnostics.empty()
                    ? "cube MLIR lowering failed"
                    : ("cube MLIR lowering failed: " + lowerResult.diagnostics);
            }

            if (!hasCubeMatmulStep) {
                std::cerr << "[CubeRoute][Error] " << calcName << ": "
                          << (cubeFailureReason.empty() ? "unknown cube pipeline failure" : cubeFailureReason)
                          << std::endl;
                return;
            }
        }

        //  分发执行 
        if (hasCubeMatmulStep) {
            hasMatrixOp = true;
            std::cout << "[Router] Cube Core: " << calcName << std::endl;

            const std::string kernelFileName =
                BuildGeneratedKernelFileName(generatedKernelFileIndex++);
            AscendCExpressionArtifact cubeArtifact = LowerCubeExpressionToArtifact(
                expr,
                i,
                rewriter,
                sourceText,
                policy,
                kernelFileName,
                cubeMatmulStep);
            cubeTilingFuncs += cubeArtifact.tilingFunc;
            std::string cubePrelude = TrimCopy(cubeArtifact.hostPreludeCode);
            if (!cubePrelude.empty() && cubePreludeSeen.insert(cubePrelude).second) {
                cubePreludeCode += cubePrelude + "\n\n";
            }
            cubeExternDoDecls += cubeArtifact.externDoDecl;
            cubeTilingDecls += cubeArtifact.tilingDecl;
            allArtifacts.push_back(std::move(cubeArtifact));
            continue;
        }

        hasVectorOp = true;
        std::cout << "[Router] Vector Core: " << calcName << std::endl;
        
        const std::string kernelFileName =
            BuildGeneratedKernelFileName(generatedKernelFileIndex++);
        AscendCExpressionArtifact vectorArtifact = LowerVectorExpressionToArtifact(
            expr,
            i,
            rewriter,
            sourceText,
            policy,
            kernelFileName,
            temporalSchedule,
            temporalScheduleApplied,
            temporalBridgeBuildFailed,
            temporalBridgeFailureReason);

        std::string prelude = TrimCopy(vectorArtifact.hostPreludeCode);
        if (!prelude.empty() && vecPreludeSeen.insert(prelude).second) {
            vecPreludeCode += prelude + "\n\n";
        }
        vecExternDoDecls += vectorArtifact.externDoDecl;
        allArtifacts.push_back(std::move(vectorArtifact));
    }

    if (!gHostNameError.empty()) {
        rewriter->getSourceMgr().getDiagnostics().Report(
            clang::diag::err_fe_error_backend) << gHostNameError;
        return;
    }

    // lowering 全部结束后一次性构造 AscendCFlowPlan。
    dacppTranslator::ascendc::HostFlowBuildInput hostFlowInput;
    hostFlowInput.dacppFile = dacppFile;
    hostFlowInput.tuDecl = gRuntimeBindingTUDecl;
    hostFlowInput.sourceManager = gRuntimeBindingSourceManager;
    hostFlowInput.langOptions = gRuntimeBindingLangOptions;
    hostFlowInput.expressionArtifacts = &allArtifacts;
    dacppTranslator::ascendc::AscendCFlowPlan hostFlowPlan =
        dacppTranslator::ascendc::BuildHostFlowPlanFromAst(hostFlowInput);
    const dacppTranslator::ascendc::HostRuntimeRequirements hostRuntimeReq =
        dacppTranslator::ascendc::CollectHostRuntimeRequirements(
            allArtifacts, hostFlowPlan, policy);
    hostFlowPlan.runtimeFeatureMask = hostRuntimeReq.features;
    if (hostRuntimeReq.packaging ==
        dacppTranslator::ascendc::HostRuntimePackaging::Compact) {
        WriteOutputFile(
            "dacpp_ascendc_host_runtime.h",
            dacppTranslator::ascendc::EmitHostRuntimeHeader(hostRuntimeReq));
    }

    // dump plan：恒写 ascendc_flow_plan.txt 供 atest_diff.sh 比对；命令行开关同步到 stdout。
    {
        const std::string flowPlanText =
            dacppTranslator::ascendc::DumpAscendCFlowPlan(hostFlowPlan);
        WriteOutputFile("ascendc_flow_plan.txt", flowPlanText);
        if (mlirOptions.dumpFlowPlan) {
            std::cout << "[ascendc-flow-plan] dump AscendCFlowPlan:\n" << flowPlanText;
        }
    }

    // 写完 plan dump 后做主流程错误检查。若 builder 在 plan 上
    // 累积了 Error 级 diagnostic 或 plan.partial==true，主流程必须把两路 host
    // 文件都进入 BuildHostGenerationErrorBody——kernel/tiling 不受影响（已生成）。
    bool hostFlowHasBuilderError = hostFlowPlan.partial;
    std::vector<std::string> hostFlowBuilderErrors;
    for (const auto& diag : hostFlowPlan.diagnostics) {
        if (diag.severity ==
            dacppTranslator::ascendc::FlowDiagnosticSeverity::Error) {
            hostFlowHasBuilderError = true;
            std::string line;
            if (!diag.code.empty()) line += diag.code + ": ";
            line += diag.message;
            if (diag.location.isValid()) {
                line += " (" + diag.location.file + ":" +
                        std::to_string(diag.location.line) + ")";
            }
            hostFlowBuilderErrors.push_back(std::move(line));
        }
    }
    if (hostFlowHasBuilderError) {
        std::cerr << "[Host侧][HostFlow] builder reported "
                  << hostFlowBuilderErrors.size()
                  << " error diagnostic(s); host file will be a generation error body"
                  << std::endl;
        if (!hostFlowBuilderErrors.empty()) {
            std::cerr << "[Host侧][HostFlow] "
                      << hostFlowBuilderErrors.front() << std::endl;
        }
    }

    // 一次性的"按 route 过滤后渲染"：每个 host 文件各发射一次。
    // emission 不允许半成品输出——失败时返回完整的 host generation error body，
    // 由上层模板直接当作 main body 写入。
    auto RenderForRoute = [&](dacppTranslator::ascendc::FlowRouteKind route) -> std::string {
        const auto* runtimeNames = hostFlowPlan.findRouteRuntimeNames(route);
        if (!runtimeNames) {
            return BuildHostGenerationErrorBody(
                "[Host侧][HostFlow] missing route runtime names",
                route == dacppTranslator::ascendc::FlowRouteKind::Cube
                    ? gCubeRuntimeNames.session
                    : gVectorRuntimeNames.session);
        }
        dacppTranslator::ascendc::HostFlowEmitOptions emitOpts;
        emitOpts.routeFilter = route;
        dacppTranslator::ascendc::HostFlowEmitResult emitResult =
            dacppTranslator::ascendc::RenderHostFlowBody(hostFlowPlan, emitOpts);
        if (!emitResult.diagnostics.empty()) {
            std::cerr << "[Host侧][HostFlow] "
                      << emitResult.diagnostics.front();
            if (emitResult.diagnostics.size() > 1) {
                std::cerr << " (+" << emitResult.diagnostics.size() - 1
                          << " more)";
            }
            std::cerr << std::endl;
        }
        if (!emitResult.ok || hostFlowHasBuilderError) {
            std::string errMsg = "[Host侧][HostFlow] emission failed";
            if (!hostFlowBuilderErrors.empty()) {
                errMsg = hostFlowBuilderErrors.front();
            } else if (!emitResult.diagnostics.empty()) {
                errMsg = emitResult.diagnostics.front();
            }
            return BuildHostGenerationErrorBody(
                errMsg, runtimeNames->session);
        }
        return emitResult.body;
    };

    int generatedHostFileIndex = 0;

    // 组装 Vector Host 代码
    if (hasVectorOp) {
        const std::string vecRenderedBody =
            RenderForRoute(dacppTranslator::ascendc::FlowRouteKind::Vector);
        std::string vectorHostCode;
        bool temporalGenerationFailed = false;
        std::string temporalFailureReason;

        if (temporalSchedule.requireNoFallback && !temporalSchedule.recognized) {
            temporalGenerationFailed = true;
            temporalFailureReason = temporalSchedule.failureReason.empty()
                ? "structured schedule missing"
                : temporalSchedule.failureReason;
        } else if (temporalSchedule.recognized && !temporalScheduleApplied) {
            temporalGenerationFailed = true;
            temporalFailureReason = "structured schedule missing matched expression index";
        } else if (temporalBridgeBuildFailed) {
            temporalGenerationFailed = true;
            temporalFailureReason = temporalBridgeFailureReason.empty()
                ? "structured schedule missing tensor-kernel bridge"
                : temporalBridgeFailureReason;
        }

        if (temporalGenerationFailed) {
            std::cerr << "[Host侧][Error] Temporal structured schedule missing: "
                      << temporalFailureReason << std::endl;
            vectorHostCode = ASCENDC_TEMPLATE::CodeGen_VectorHostFile(
                vecExternDoDecls,
                vecPreludeCode,
                ASCENDC_TEMPLATE::CodeGen_VectorMainBody(
                    BuildHostGenerationErrorBody(
                        "[Host侧][Error] Temporal structured schedule missing: " +
                        temporalFailureReason,
                        gVectorRuntimeNames.session),
                    gVectorRuntimeNames.session),
                hostRuntimeReq);
        } else {
            // 注意：vecRenderedBody 在 HostFlow 失败时已经是 BuildHostGenerationErrorBody
            // 的返回值（见 RenderForRoute），主流程在这里不再做额外的失败分支。
            std::string vectorMainBody =
                ASCENDC_TEMPLATE::CodeGen_VectorMainBody(
                    vecRenderedBody, gVectorRuntimeNames.session);
            vectorHostCode = ASCENDC_TEMPLATE::CodeGen_VectorHostFile(
                vecExternDoDecls,
                vecPreludeCode,
                vectorMainBody,
                hostRuntimeReq);
        }

        WriteOutputFile(BuildGeneratedHostFileName(generatedHostFileIndex++), vectorHostCode,
                        "[Host侧] 成功生成 Vector 执行流: ");
    }
    //  组装 Cube Tiling 独立文件
    if (hasMatrixOp) {
        // 使用新模板包装所有的 Tiling Funcs
        std::string tilingFileCode = ASCENDC_TEMPLATE::CodeGen_CubeTilingFile(cubeTilingFuncs);
        WriteOutputFile("tiling.cpp", tilingFileCode,
                        "[Tiling侧] 成功生成 Tiling 计算流: ");
    }
    //  组装 Cube Host 代码
    if (hasMatrixOp) {
        const std::string cubeRenderedBody =
            RenderForRoute(dacppTranslator::ascendc::FlowRouteKind::Cube);
        // 注意：cubeRenderedBody 在 HostFlow 失败时已经是 BuildHostGenerationErrorBody
        // 的返回值（见 RenderForRoute），主流程不再做额外的 mixedRouteConflict 分支。
        std::string cubeMainBody =
            ASCENDC_TEMPLATE::CodeGen_CubeMainBody(
                cubeRenderedBody, gCubeRuntimeNames.session);
        std::string cubeHostCode = ASCENDC_TEMPLATE::CodeGen_CubeHostFile(
            cubeExternDoDecls,
            cubeTilingDecls,
            cubeMainBody,
            cubePreludeCode,
            hostRuntimeReq);

        WriteOutputFile(BuildGeneratedHostFileName(generatedHostFileIndex++), cubeHostCode,
                        "[Host侧] 成功生成 Cube 执行流: ");
    }

    // ascendc_flow_plan.txt / --ascendc-dump-flow-plan 在新流程顶部已写出。
}
