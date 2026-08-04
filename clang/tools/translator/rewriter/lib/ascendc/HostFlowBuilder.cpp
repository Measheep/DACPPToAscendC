// HostFlowBuilder：从 Clang AST 直接构造 AscendCFlowPlan。
//
// 本文件不保存用户源码 Stmt 原文：所有 FlowStep 都通过 AST/descriptor 描述。
//
// Step 2 提供 scaffold：扫描 owner function body，识别承载 DAC `<->` 的语句，为其插入
// FlowKernelLaunch；其它语句先落 hostflow.unclassified_stmt。
//
// Step 3 把"其它语句"分类到结构化 FlowStep：
//   - BuildHostInputFromVarDecl       : 原始/常量/数组变量声明
//   - BuildTensorBufferFromVarDecl    : dacpp::Vector/Matrix/Tensor 与 std::vector
//   - BuildHelperInvocationFromCall   : 主体为函数调用的 ExprStmt
//   - BuildHostLoopFromStmt           : ForStmt / WhileStmt / DoStmt（递归分类 body）
//   - BuildHostBranchFromIf           : IfStmt（递归分类 then/else）
//   - BuildHostOutputFromStmt         : print/cout/printf
//   - BuildTensorAssignFromBinaryOperator : tensor[i] = expr / scalar self-update
//   - BuildDiagnosticForUnsupportedStmt   : fallback → FlowDiagnostic(Error)
//
// 同时 AST-based 重写并迁移 driver 中的语义辅助：
//   LooksLikeHostOutputStmtText / CollectTextuallyReferencedPreludeDecls。

#include "ascendc/HostFlowBuilder.h"

#include <algorithm>
#include <cctype>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include "DacppStructure.h"
#include "ascendc/AscendCArtifacts.h"
#include "ascendc/HostCodeNames.h"
#include "ascendc/AscendCRuntimeBinding.h"
#include "ascendc/AscendCSourceUtils.h"
#include "ascendc/AscendCTensorAlias.h"

namespace dacppTranslator {
namespace ascendc {

namespace {

// ============================================================
// 通用 AST / source-text 助手
// ============================================================

bool IsStmtContainingLoc(const clang::Stmt* stmt,
                         const clang::SourceManager& sm,
                         clang::SourceLocation loc) {
    if (!stmt) return false;
    clang::SourceLocation begin = sm.getExpansionLoc(stmt->getBeginLoc());
    clang::SourceLocation end = sm.getExpansionLoc(stmt->getEndLoc());
    loc = sm.getExpansionLoc(loc);
    if (begin.isInvalid() || end.isInvalid() || loc.isInvalid()) return false;
    return !sm.isBeforeInTranslationUnit(loc, begin) &&
           !sm.isBeforeInTranslationUnit(end, loc);
}

static void VisitAllFunctionsInTU(
    const clang::TranslationUnitDecl* tu,
    const std::function<bool(const clang::FunctionDecl*)>& visitor) {
    if (!tu) return;
    std::function<void(const clang::DeclContext*)> walk =
        [&](const clang::DeclContext* dc) {
            if (!dc) return;
            for (const clang::Decl* decl : dc->decls()) {
                if (!decl) continue;
                if (const auto* fn = llvm::dyn_cast<clang::FunctionDecl>(decl)) {
                    if (fn->hasBody() && visitor(fn)) return;
                }
                if (const auto* child = llvm::dyn_cast<clang::DeclContext>(decl)) {
                    walk(child);
                }
            }
        };
    walk(tu);
}

const clang::FunctionDecl* FindEnclosingFunction(
    const clang::BinaryOperator* dacExpr,
    const clang::TranslationUnitDecl* tuDecl,
    const clang::SourceManager& sm) {
    if (!dacExpr || !tuDecl) return nullptr;

    clang::SourceLocation exprLoc = sm.getExpansionLoc(dacExpr->getExprLoc());
    if (exprLoc.isInvalid()) return nullptr;

    const clang::FunctionDecl* best = nullptr;
    unsigned bestSpan = std::numeric_limits<unsigned>::max();

    VisitAllFunctionsInTU(tuDecl, [&](const clang::FunctionDecl* fn) {
        const clang::Stmt* body = fn->getBody();
        if (!body) return false;
        clang::SourceLocation begin = sm.getExpansionLoc(body->getBeginLoc());
        clang::SourceLocation end = sm.getExpansionLoc(body->getEndLoc());
        if (begin.isInvalid() || end.isInvalid() ||
            !sm.isWrittenInSameFile(begin, exprLoc) ||
            !sm.isWrittenInSameFile(end, exprLoc) ||
            sm.isBeforeInTranslationUnit(exprLoc, begin) ||
            sm.isBeforeInTranslationUnit(end, exprLoc)) return false;
        unsigned beginOff = sm.getFileOffset(begin);
        unsigned endOff = sm.getFileOffset(end);
        if (endOff >= beginOff) {
            unsigned span = endOff - beginOff;
            if (!best || span < bestSpan) {
                best = fn;
                bestSpan = span;
            }
        }
        return false;
    });
    return best;
}

FlowSourceLoc MakeSourceLoc(clang::SourceLocation loc, const clang::SourceManager& sm) {
    FlowSourceLoc out;
    if (loc.isInvalid()) return out;
    clang::SourceLocation expansion = sm.getExpansionLoc(loc);
    if (expansion.isInvalid()) return out;
    auto presumed = sm.getPresumedLoc(expansion);
    if (!presumed.isValid()) return out;
    if (const char* fname = presumed.getFilename()) out.file = fname;
    out.line = static_cast<int>(presumed.getLine());
    out.column = static_cast<int>(presumed.getColumn());
    return out;
}

// 用 Lexer 取出 SourceRange 对应的源码片段；空字符串表示不可用。
std::string GetSourceTextRange(clang::SourceRange range,
                               const clang::SourceManager& sm,
                               const clang::LangOptions& lo) {
    if (range.isInvalid()) return {};
    clang::SourceLocation b = sm.getExpansionLoc(range.getBegin());
    clang::SourceLocation e = sm.getExpansionLoc(range.getEnd());
    if (b.isInvalid() || e.isInvalid()) return {};
    clang::CharSourceRange cr = clang::CharSourceRange::getTokenRange(b, e);
    llvm::StringRef text = clang::Lexer::getSourceText(cr, sm, lo);
    if (text.empty()) return {};
    return text.str();
}

std::string GetExprSourceText(const clang::Expr* expr,
                              const clang::SourceManager& sm,
                              const clang::LangOptions& lo) {
    if (!expr) return {};
    return GetSourceTextRange(expr->getSourceRange(), sm, lo);
}

using dacppTranslator::ascendc::TrimCopy;

std::string CollapseWS(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    bool prevSpace = false;
    for (char c : s) {
        if (c == '\n' || c == '\r' || c == '\t' || c == ' ') {
            if (!prevSpace) out.push_back(' ');
            prevSpace = true;
        } else {
            out.push_back(c);
            prevSpace = false;
        }
    }
    return TrimCopy(out);
}

std::string GetQualTypeText(clang::QualType qt, const clang::LangOptions& lo) {
    if (qt.isNull()) return {};
    clang::PrintingPolicy pp(lo);
    pp.SuppressTagKeyword = false;
    pp.SuppressScope = false;
    pp.Bool = true;
    return qt.getAsString(pp);
}

// peel 掉 paren / ImplicitCast / MaterializeTemporary 等，拿到"语义本体"。
const clang::Expr* StripParenImpCasts(const clang::Expr* e) {
    if (!e) return nullptr;
    return e->IgnoreParenImpCasts();
}

// ============================================================
// host helper 函数识别（与 driver::IsHostSideHelperFunction 等价的 AST 判定）
// ============================================================

bool IsHostSideHelperFunction(const clang::FunctionDecl* fn) {
    if (!fn || !fn->hasBody() || fn->isImplicit()) return false;
    if (llvm::isa<clang::CXXMethodDecl>(fn)) return false;
    const std::string name = fn->getNameAsString();
    if (name.empty() || name == "main") return false;
    if (name.find("shell") != std::string::npos ||
        name.find("calc")  != std::string::npos) {
        return false;
    }
    return true;
}

// ============================================================
// 输出语义（cout / printf / .print()）AST 识别
// ============================================================

bool IsStdStreamRef(const clang::Expr* expr) {
    expr = StripParenImpCasts(expr);
    if (!expr) return false;
    if (const auto* ref = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
        const std::string name = ref->getNameInfo().getAsString();
        if (name == "cout" || name == "cerr" || name == "clog") return true;
    }
    return false;
}

class HostOutputDetector
    : public clang::RecursiveASTVisitor<HostOutputDetector> {
public:
    bool found = false;
    FlowHostOutputKind kind = FlowHostOutputKind::Unknown;

    bool VisitCallExpr(clang::CallExpr* call) {
        if (!call) return true;
        if (const clang::FunctionDecl* fn = call->getDirectCallee()) {
            const std::string name = fn->getNameAsString();
            if (name == "printf" || name == "fprintf" || name == "puts" ||
                name == "snprintf" || name == "vprintf") {
                found = true;
                kind = FlowHostOutputKind::PrintfCall;
                return false;
            }
        }
        return true;
    }

    bool VisitCXXMemberCallExpr(clang::CXXMemberCallExpr* call) {
        if (!call) return true;
        if (const clang::CXXMethodDecl* method = call->getMethodDecl()) {
            const std::string name = method->getNameAsString();
            if (name == "print") {
                found = true;
                kind = FlowHostOutputKind::TensorPrint;
                return false;
            }
        }
        return true;
    }

    bool VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr* call) {
        if (!call) return true;
        if (call->getOperator() == clang::OO_LessLess && call->getNumArgs() >= 1) {
            // operator<<(std::ostream&, ...) — 检测左操作数是否最终归到 cout/cerr。
            const clang::Expr* base = call->getArg(0);
            while (base) {
                base = StripParenImpCasts(base);
                if (IsStdStreamRef(base)) {
                    found = true;
                    kind = FlowHostOutputKind::StdOutStream;
                    return false;
                }
                if (const auto* nested =
                        llvm::dyn_cast<clang::CXXOperatorCallExpr>(base)) {
                    if (nested->getOperator() == clang::OO_LessLess &&
                        nested->getNumArgs() >= 1) {
                        base = nested->getArg(0);
                        continue;
                    }
                }
                break;
            }
        }
        return true;
    }
};

bool LooksLikeHostOutputStmtAst(const clang::Stmt* stmt,
                                FlowHostOutputKind* outKind) {
    if (!stmt) return false;
    HostOutputDetector d;
    d.TraverseStmt(const_cast<clang::Stmt*>(stmt));
    if (d.found && outKind) *outKind = d.kind;
    return d.found;
}

// ============================================================
// 类型识别（dacpp tensor / std::vector / 切片）
// ============================================================

bool LooksLikeDacppContainerType(clang::QualType qt, const clang::LangOptions& lo,
                                 std::string* outShortKind) {
    if (qt.isNull()) return false;
    const std::string name = GetQualTypeText(qt, lo);
    auto contains = [&](const char* n) { return name.find(n) != std::string::npos; };
    if (contains("dacpp::Vector") || contains("Vector<")) {
        if (outShortKind) *outShortKind = "Vector";
        return true;
    }
    if (contains("dacpp::Matrix") || contains("Matrix<")) {
        if (outShortKind) *outShortKind = "Matrix";
        return true;
    }
    if (contains("dacpp::Tensor") || contains("Tensor<")) {
        if (outShortKind) *outShortKind = "Tensor";
        return true;
    }
    return false;
}

bool LooksLikeStdVectorType(clang::QualType qt, const clang::LangOptions& lo) {
    if (qt.isNull()) return false;
    const std::string name = GetQualTypeText(qt, lo);
    return name.find("std::vector") != std::string::npos ||
           name.find("vector<") != std::string::npos;
}

// 取 VarDecl 的 declared element type：
//   - T[N] / T[] / T[expr]                          → T
//   - std::vector<T> / dacpp::Vector<T> / 等类模板    → 模板第一个类型实参 T
//   - 其余                                            → type 自身
clang::QualType GetVarElementType(const clang::VarDecl* vd) {
    if (!vd) return {};
    clang::QualType qt = vd->getType();
    if (qt.isNull()) return qt;
    if (const auto* arr = vd->getASTContext().getAsArrayType(qt)) {
        return arr->getElementType();
    }
    // 抽取 ClassTemplateSpecializationDecl 的首个 Type 模板实参。
    if (const clang::CXXRecordDecl* rd = qt->getAsCXXRecordDecl()) {
        if (const auto* spec =
                llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(rd)) {
            const clang::TemplateArgumentList& args = spec->getTemplateArgs();
            for (unsigned i = 0; i < args.size(); ++i) {
                const clang::TemplateArgument& ta = args[i];
                if (ta.getKind() == clang::TemplateArgument::Type) {
                    return ta.getAsType();
                }
            }
        }
    }
    return qt;
}

// 提取数组维度表达式文本（VLA / 常量数组 / IncompleteArray 都各占一种）。
std::string GetArrayExtentExpr(const clang::VarDecl* vd,
                               const clang::SourceManager& sm,
                               const clang::LangOptions& lo) {
    if (!vd) return {};
    clang::QualType qt = vd->getType();
    if (qt.isNull()) return {};
    if (const auto* cat =
            llvm::dyn_cast_or_null<clang::ConstantArrayType>(
                vd->getASTContext().getAsArrayType(qt))) {
        llvm::SmallString<32> buf;
        cat->getSize().toString(buf, /*radix=*/10, /*signed=*/false,
                                /*formatAsCLiteral=*/false);
        return std::string(buf.str());
    }
    if (const auto* vat =
            llvm::dyn_cast_or_null<clang::VariableArrayType>(
                vd->getASTContext().getAsArrayType(qt))) {
        if (const clang::Expr* sz = vat->getSizeExpr()) {
            return CollapseWS(GetExprSourceText(sz, sm, lo));
        }
    }
    return {};
}

// 取 init expr 文本（裸的 init / "= init" 都剥掉）
std::string GetInitializerText(const clang::VarDecl* vd,
                               const clang::SourceManager& sm,
                               const clang::LangOptions& lo) {
    if (!vd || !vd->hasInit()) return {};
    const clang::Expr* init = vd->getInit();
    if (!init) return {};
    return CollapseWS(GetExprSourceText(init, sm, lo));
}

// ============================================================
// 容器/数据流相关：DAC 表达式索引
// ============================================================

struct ExprEntry {
    int expressionIndex = -1;
    const AscendCExpressionArtifact* artifact = nullptr;
    const clang::BinaryOperator* dacExpr = nullptr;
    const clang::FunctionDecl* ownerFn = nullptr;
    clang::SourceLocation exprLoc;
};

struct ClassifyContext {
    AscendCFlowPlan& plan;
    const HostFlowBuildInput& input;
    const clang::SourceManager& sm;
    const clang::LangOptions& lo;
    const std::vector<const ExprEntry*>& fnEntries;
    const clang::FunctionDecl* ownerFn = nullptr;
    // host 端 tensor alias 映射：CollectLiveTensorAliasesInFunction 填充。
    const std::unordered_map<std::string, HostTensorAliasInfo>* aliasMap = nullptr;
    std::size_t* tensorPrintCounter = nullptr;  // 给 BuildTensorViewPrintPlan 用的唯一索引
    std::size_t* tensorAssignCounter = nullptr; // 给 BuildTensorAssignmentPlan 用的唯一索引
    // 内联展开映射：在 main() 遍历过程中遇到 callee 命中此集合时，把 callee body 递归
    // 当作 main 的 stmt 序列处理（DAC 在 callee 内时让 main 体得以完整发射）。
    const std::unordered_set<const clang::FunctionDecl*>* inlineExpansionTargets = nullptr;
};

// 是否包含某个 DAC `<->` 表达式的位置（用于 KernelLaunch 替换）。
const ExprEntry* MatchDacEntryForStmt(const ClassifyContext& cx,
                                      const clang::Stmt* stmt) {
    if (!stmt) return nullptr;
    for (const ExprEntry* e : cx.fnEntries) {
        if (!e->exprLoc.isValid()) continue;
        if (IsStmtContainingLoc(stmt, cx.sm, e->exprLoc)) return e;
    }
    return nullptr;
}



// ============================================================
// 分类器：每个返回 step id；返回 kInvalidFlowStepId 代表已加诊断、调用方不再继续。
// ============================================================

FlowStepId BuildDiagnosticForUnsupportedStmt(ClassifyContext& cx,
                                              const clang::Stmt* stmt,
                                              const char* code,
                                              std::string message) {
    FlowDiagnostic d;
    d.severity = FlowDiagnosticSeverity::Error;
    d.code = code ? code : "hostflow.unsupported_stmt";
    d.message = message.empty()
                    ? "Unsupported host stmt in HostFlow AST builder"
                    : std::move(message);
    if (stmt) d.location = MakeSourceLoc(stmt->getBeginLoc(), cx.sm);
    cx.plan.partial = true;
    return cx.plan.addDiagnostic(std::move(d));
}

// 原 VarDecl 整段源码捕获已删除——builder 仅按 VarDecl 字段
// (type/name/initializer/storageClass...) 结构化重建。

FlowStepId BuildHostInputFromVarDecl(ClassifyContext& cx,
                                     const clang::VarDecl* vd) {
    if (!vd) return kInvalidFlowStepId;

    FlowHostInput hi;
    hi.name = vd->getNameAsString();
    hi.declTypeText = GetQualTypeText(vd->getType(), cx.lo);
    hi.initializerText = GetInitializerText(vd, cx.sm, cx.lo);
    hi.isConst = vd->getType().isConstQualified();
    hi.isConstexpr = vd->isConstexpr();
    hi.isHostGlobal = vd->hasGlobalStorage() && !vd->isLocalVarDecl();
    hi.arrayExtentExpr = GetArrayExtentExpr(vd, cx.sm, cx.lo);
    // 不缓存整条 VarDecl 原文；emitter 按结构化字段重建。

    switch (vd->getStorageClass()) {
        case clang::SC_Static:   hi.storageClass = "static";   break;
        case clang::SC_Extern:   hi.storageClass = "extern";   break;
        case clang::SC_Register: hi.storageClass = "register"; break;
        case clang::SC_Auto:     hi.storageClass = "auto";     break;
        default:                 hi.storageClass.clear();      break;
    }

    hi.location = MakeSourceLoc(vd->getBeginLoc(), cx.sm);
    return cx.plan.addHostInput(std::move(hi));
}

const clang::Stmt* GetLoopBody(const clang::Stmt* stmt) {
    if (!stmt) return nullptr;
    if (const auto* fs = llvm::dyn_cast<clang::ForStmt>(stmt)) return fs->getBody();
    if (const auto* ws = llvm::dyn_cast<clang::WhileStmt>(stmt)) return ws->getBody();
    if (const auto* ds = llvm::dyn_cast<clang::DoStmt>(stmt)) return ds->getBody();
    return nullptr;
}

// 剥掉 InitListExpr 外层的隐式包装（ExprWithCleanups / CXXBindTemporaryExpr /
// MaterializeTemporaryExpr / CXXFunctionalCastExpr 等），定位真正的初始化表达式。
const clang::Expr* UnwrapInitExpr(const clang::Expr* e) {
    while (e) {
        const clang::Expr* prev = e;
        e = e->IgnoreImplicit();
        if (const auto* w = llvm::dyn_cast<clang::ExprWithCleanups>(e)) {
            e = w->getSubExpr();
            continue;
        }
        if (const auto* b = llvm::dyn_cast<clang::CXXBindTemporaryExpr>(e)) {
            e = b->getSubExpr();
            continue;
        }
        if (const auto* m = llvm::dyn_cast<clang::MaterializeTemporaryExpr>(e)) {
            e = m->getSubExpr();
            continue;
        }
        if (e == prev) break;
    }
    return e;
}

FlowStepId BuildTensorBufferFromVarDecl(ClassifyContext& cx,
                                        const clang::VarDecl* vd) {
    if (!vd) return kInvalidFlowStepId;

    std::string shortKind;
    if (LooksLikeDacppContainerType(vd->getType(), cx.lo, &shortKind)) {
        // dacpp::Vector/Matrix/Tensor 声明被翻译为 tensor alias 元数据，
        // 不在 host main body 中重新发射。
        return kInvalidFlowStepId;
    }
    if (!LooksLikeStdVectorType(vd->getType(), cx.lo)) {
        return BuildDiagnosticForUnsupportedStmt(
            cx, vd->getInit() ? vd->getInit() : nullptr,
            "hostflow.unsupported_stmt",
            "VarDecl `" + vd->getNameAsString() +
                "` is not recognized as tensor buffer");
    }
    // 跳过未被任何后续 stmt 引用的 std::vector 声明。
    if (!vd->isReferenced()) {
        return kInvalidFlowStepId;
    }

    FlowTensorBuffer tb;
    tb.name = vd->getNameAsString();
    tb.location = MakeSourceLoc(vd->getBeginLoc(), cx.sm);
    tb.dacppContainerText.clear();  // std::vector

    clang::QualType elem = GetVarElementType(vd);
    if (!elem.isNull()) tb.elementTypeText = GetQualTypeText(elem, cx.lo);

    // 结构化捕获三种 std::vector 初始化形式。
    if (vd->hasInit()) {
        const clang::Expr* init = UnwrapInitExpr(vd->getInit());
        bool classified = false;

        if (const auto* ctor =
                llvm::dyn_cast_or_null<clang::CXXConstructExpr>(init)) {
            const unsigned nargs = ctor->getNumArgs();
            // 跳过末尾的 CXXDefaultArgExpr（allocator 默认参数等）。
            unsigned effArgs = nargs;
            while (effArgs > 0 &&
                   llvm::isa<clang::CXXDefaultArgExpr>(
                       ctor->getArg(effArgs - 1))) {
                --effArgs;
            }

            if (effArgs == 1) {
                const clang::Expr* arg0 =
                    UnwrapInitExpr(ctor->getArg(0));
                if (const auto* il =
                        llvm::dyn_cast_or_null<clang::InitListExpr>(arg0)) {
                    // std::vector<T> v{a, b, c}
                    tb.origin = FlowBufferOrigin::StdVectorCtor;
                    for (unsigned i = 0; i < il->getNumInits(); ++i) {
                        tb.initListElems.push_back(CollapseWS(
                            GetExprSourceText(il->getInit(i), cx.sm, cx.lo)));
                    }
                    classified = true;
                } else if (arg0 &&
                           LooksLikeStdVectorType(arg0->getType(), cx.lo)) {
                    // std::vector<T> v = other / v(other)
                    tb.origin = FlowBufferOrigin::StdVectorCtor;
                    tb.copyFromVectorName =
                        CollapseWS(GetExprSourceText(arg0, cx.sm, cx.lo));
                    tb.sourceVectorName = tb.copyFromVectorName;
                    classified = true;
                } else if (arg0) {
                    // std::vector<T> v(n)
                    tb.origin = FlowBufferOrigin::StdVectorCtor;
                    tb.ctorArgs.push_back(
                        CollapseWS(GetExprSourceText(arg0, cx.sm, cx.lo)));
                    classified = true;
                }
            } else if (effArgs >= 2) {
                // std::vector<T> v(n, init) — 取前 effArgs 个非 default-arg。
                tb.origin = FlowBufferOrigin::StdVectorCtor;
                for (unsigned i = 0; i < effArgs; ++i) {
                    const clang::Expr* a = ctor->getArg(i);
                    tb.ctorArgs.push_back(
                        CollapseWS(GetExprSourceText(a, cx.sm, cx.lo)));
                }
                classified = true;
            } else {
                // 0 个有效参数：默认构造（罕见）。视为空初始化。
                tb.origin = FlowBufferOrigin::StdVectorCtor;
                classified = true;
            }
        } else if (const auto* il =
                       llvm::dyn_cast_or_null<clang::InitListExpr>(init)) {
            // 直接 InitListExpr 兜底
            tb.origin = FlowBufferOrigin::StdVectorCtor;
            for (unsigned i = 0; i < il->getNumInits(); ++i) {
                tb.initListElems.push_back(CollapseWS(
                    GetExprSourceText(il->getInit(i), cx.sm, cx.lo)));
            }
            classified = true;
        }

        if (!classified) {
            return BuildDiagnosticForUnsupportedStmt(
                cx, vd->getInit(), "hostflow.unsupported_stmt",
                "std::vector `" + vd->getNameAsString() +
                    "` has unrecognized initializer form");
        }
    }

    return cx.plan.addTensorBuffer(std::move(tb));
}

FlowHelperRole ClassifyHelperByName(const std::string& nameRaw) {
    std::string name;
    name.reserve(nameRaw.size());
    for (char c : nameRaw)
        name.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));

    auto contains = [&](std::initializer_list<const char*> needles) {
        for (const char* n : needles)
            if (name.find(n) != std::string::npos) return true;
        return false;
    };

    if (contains({"tile", "block_size", "blockdim", "compute_block"}))
        return FlowHelperRole::TileSizeHint;
    if (contains({"print", "dump", "stats", "report", "log"}))
        return FlowHelperRole::PrintLike;
    if (contains({"init", "initialize", "load", "read_input",
                  "build_input", "fill", "prepare"}))
        return FlowHelperRole::InitOnly;
    if (contains({"compute", "calculate", "calc", "run", "execute",
                  "process", "solve", "update", "step",
                  "dft", "fft", "convolve", "transform"}))
        return FlowHelperRole::OutputMutating;
    return FlowHelperRole::Unknown;
}

FlowStepId BuildHelperInvocationFromCall(ClassifyContext& cx,
                                          const clang::CallExpr* call,
                                          const std::string& resultTarget) {
    if (!call) return kInvalidFlowStepId;

    FlowHelperInvocation h;
    if (const clang::FunctionDecl* fn = call->getDirectCallee()) {
        h.functionName = fn->getNameAsString();
        h.role = ClassifyHelperByName(h.functionName);
        h.returnsValue = !fn->getReturnType().isNull() &&
                         !fn->getReturnType()->isVoidType();
    }

    // CXXMemberCallExpr：把 receiver 文本拼到 functionName，让 emitter 直接发
    // `receiver.method(args...)`，避免引入额外字段。
    if (const auto* mc = llvm::dyn_cast<clang::CXXMemberCallExpr>(call)) {
        const clang::Expr* recv = mc->getImplicitObjectArgument();
        if (recv && !h.functionName.empty()) {
            const std::string recvText = CollapseWS(
                GetExprSourceText(StripParenImpCasts(recv), cx.sm, cx.lo));
            if (!recvText.empty()) {
                h.functionName = recvText + "." + h.functionName;
            }
        }
    }

    for (unsigned i = 0; i < call->getNumArgs(); ++i) {
        const clang::Expr* arg = call->getArg(i);
        if (llvm::isa<clang::CXXDefaultArgExpr>(arg)) continue;
        h.argTexts.push_back(CollapseWS(GetExprSourceText(arg, cx.sm, cx.lo)));
    }
    h.resultTargetName = resultTarget;
    h.location = MakeSourceLoc(call->getBeginLoc(), cx.sm);
    return cx.plan.addHelperInvocation(std::move(h));
}

// 找到 stmt 子树内最外层的 operator<< CXXOperatorCallExpr。
const clang::CXXOperatorCallExpr* FindOutermostStreamShift(
    const clang::Stmt* stmt) {
    if (!stmt) return nullptr;
    class V : public clang::RecursiveASTVisitor<V> {
    public:
        const clang::CXXOperatorCallExpr* found = nullptr;
        bool VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr* op) {
            if (op && op->getOperator() == clang::OO_LessLess) {
                found = op;
                return false;
            }
            return true;
        }
    };
    V v;
    v.TraverseStmt(const_cast<clang::Stmt*>(stmt));
    return v.found;
}

// 把 operator<< 链拆成 (streamName, operands)，并识别 std::endl / std::flush。
// 失败时返回 false（不属于 cout/cerr/clog 的链）。
bool ExtractStdStreamChainFromStmt(const clang::Stmt* stmt,
                                   const clang::SourceManager& sm,
                                   const clang::LangOptions& lo,
                                   std::string& outStreamName,
                                   std::vector<std::string>& outOperands) {
    const clang::CXXOperatorCallExpr* root = FindOutermostStreamShift(stmt);
    if (!root) return false;

    std::vector<std::string> ops;  // RHS 自外向内收集，最后反转
    const clang::Expr* cur = root;
    while (true) {
        const clang::Expr* stripped = StripParenImpCasts(cur);
        const auto* op = llvm::dyn_cast_or_null<clang::CXXOperatorCallExpr>(stripped);
        if (!op || op->getOperator() != clang::OO_LessLess ||
            op->getNumArgs() < 2) {
            cur = stripped;
            break;
        }
        const clang::Expr* rhs = StripParenImpCasts(op->getArg(1));
        std::string rhsText = CollapseWS(GetExprSourceText(rhs, sm, lo));
        ops.push_back(std::move(rhsText));
        cur = op->getArg(0);
    }

    cur = StripParenImpCasts(cur);
    if (!IsStdStreamRef(cur)) return false;

    if (const auto* dr = llvm::dyn_cast_or_null<clang::DeclRefExpr>(cur)) {
        const std::string nm = dr->getNameInfo().getAsString();
        outStreamName = "std::" + nm;
    } else {
        outStreamName = CollapseWS(GetExprSourceText(cur, sm, lo));
    }
    std::reverse(ops.begin(), ops.end());
    outOperands = std::move(ops);
    return true;
}

// 从 stmt 子树中找 printf / fprintf / puts / snprintf / vprintf 调用，
// 把 callee 名与 args 文本抽出来。
bool ExtractPrintfCallFromStmt(const clang::Stmt* stmt,
                               const clang::SourceManager& sm,
                               const clang::LangOptions& lo,
                               std::string& outCallee,
                               std::vector<std::string>& outArgs) {
    if (!stmt) return false;
    class V : public clang::RecursiveASTVisitor<V> {
    public:
        const clang::CallExpr* found = nullptr;
        bool VisitCallExpr(clang::CallExpr* c) {
            if (!c) return true;
            if (const clang::FunctionDecl* fn = c->getDirectCallee()) {
                const std::string nm = fn->getNameAsString();
                if (nm == "printf" || nm == "fprintf" || nm == "puts" ||
                    nm == "snprintf" || nm == "vprintf") {
                    found = c;
                    return false;
                }
            }
            return true;
        }
    };
    V v;
    v.TraverseStmt(const_cast<clang::Stmt*>(stmt));
    if (!v.found) return false;

    if (const clang::FunctionDecl* fn = v.found->getDirectCallee()) {
        outCallee = fn->getNameAsString();
    }
    outArgs.clear();
    for (unsigned i = 0; i < v.found->getNumArgs(); ++i) {
        outArgs.push_back(CollapseWS(
            GetExprSourceText(v.found->getArg(i), sm, lo)));
    }
    return !outCallee.empty();
}

// 试着从 `.print()` 的 CXXMemberCallExpr 拿到 base 表达式文本（tensor alias 名）。
std::string ExtractTensorPrintBaseText(const clang::Stmt* stmt,
                                       const clang::SourceManager& sm,
                                       const clang::LangOptions& lo) {
    if (!stmt) return {};
    // 查找 stmt 子树内的第一个 .print() member call。
    class FindPrintCall
        : public clang::RecursiveASTVisitor<FindPrintCall> {
    public:
        const clang::CXXMemberCallExpr* found = nullptr;
        bool VisitCXXMemberCallExpr(clang::CXXMemberCallExpr* call) {
            if (!call) return true;
            if (const clang::CXXMethodDecl* m = call->getMethodDecl()) {
                if (m->getNameAsString() == "print") {
                    found = call;
                    return false;
                }
            }
            return true;
        }
    };
    FindPrintCall v;
    v.TraverseStmt(const_cast<clang::Stmt*>(stmt));
    if (!v.found) return {};
    const clang::Expr* base = v.found->getImplicitObjectArgument();
    if (!base) return {};
    return TrimCopy(GetExprSourceText(StripParenImpCasts(base), sm, lo));
}

// 输出语义：cout / printf / .print()
FlowStepId BuildHostOutputFromStmt(ClassifyContext& cx,
                                   const clang::Stmt* stmt,
                                   FlowHostOutputKind kind) {
    FlowHostOutput o;
    o.kind = kind;
    o.location = MakeSourceLoc(stmt->getBeginLoc(), cx.sm);

    switch (kind) {
        case FlowHostOutputKind::TensorPrint: {
            // alias 解析成功 → generatedCode；失败 → 硬失败 diagnostic。
            const std::string baseText =
                ExtractTensorPrintBaseText(stmt, cx.sm, cx.lo);
            if (cx.aliasMap && !baseText.empty()) {
                HostTensorAliasInfo aliasInfo;
                const std::string ownerName =
                    cx.ownerFn ? cx.ownerFn->getNameAsString() : std::string();
                if (TryResolveHostTensorAliasFromText(
                        baseText, ownerName, *cx.aliasMap, aliasInfo) &&
                    aliasInfo.valid) {
                    const std::size_t uniq =
                        cx.tensorPrintCounter ? (*cx.tensorPrintCounter)++ : 0;
                    TensorAliasAccessPlan plan = BuildTensorViewPrintPlan(
                        aliasInfo, std::to_string(uniq));
                    std::string materialized = EmitTensorAliasAccessPlan(plan);
                    if (!materialized.empty()) {
                        o.targetExprText = baseText;
                        o.generatedCode = std::move(materialized);
                        break;
                    }
                }
            }
            return BuildDiagnosticForUnsupportedStmt(
                cx, stmt, "hostflow.tensor_print_unresolved",
                "tensor `.print()` lowering failed for `" + baseText + "`");
        }
        case FlowHostOutputKind::StdOutStream: {
            std::string streamName;
            std::vector<std::string> operands;
            if (!ExtractStdStreamChainFromStmt(
                    stmt, cx.sm, cx.lo, streamName, operands)) {
                return BuildDiagnosticForUnsupportedStmt(
                    cx, stmt, "hostflow.stdstream_unresolved",
                    "std::cout/cerr operator<< chain not resolvable");
            }
            if (cx.aliasMap) {
                const std::string ownerName =
                    cx.ownerFn ? cx.ownerFn->getNameAsString() : std::string();
                for (std::string& operand : operands) {
                    HostTensorAliasInfo aliasInfo;
                    if (TryResolveHostTensorAliasFromText(
                            operand, ownerName, *cx.aliasMap, aliasInfo) &&
                        aliasInfo.valid &&
                        aliasInfo.rank == 0) {
                        TensorAliasAccessPlan plan =
                            BuildTensorScalarReadPlan(aliasInfo);
                        std::string materialized =
                            EmitTensorAliasAccessPlan(plan);
                        if (!materialized.empty()) {
                            operand = std::move(materialized);
                        }
                    }
                }
            }
            o.streamName = std::move(streamName);
            o.streamOperands = std::move(operands);
            break;
        }
        case FlowHostOutputKind::PrintfCall: {
            std::string callee;
            std::vector<std::string> args;
            if (!ExtractPrintfCallFromStmt(
                    stmt, cx.sm, cx.lo, callee, args)) {
                return BuildDiagnosticForUnsupportedStmt(
                    cx, stmt, "hostflow.printf_unresolved",
                    "printf-like call not resolvable");
            }
            o.printfCallee = std::move(callee);
            o.printfArgs = std::move(args);
            break;
        }
        default:
            // 其它 kind 不应通过 BuildHostOutputFromStmt 入口；视为破坏性 fallback。
            return BuildDiagnosticForUnsupportedStmt(
                cx, stmt, "hostflow.unsupported_stmt",
                "BuildHostOutputFromStmt called with unexpected kind");
    }
    return cx.plan.addHostOutput(std::move(o));
}

// tensor[i] = expr / scalar+= 等
FlowStepId BuildTensorAssignFromBinaryOperator(
    ClassifyContext& cx, const clang::BinaryOperator* binop) {
    if (!binop) return kInvalidFlowStepId;

    FlowHostOutput o;
    o.location = MakeSourceLoc(binop->getBeginLoc(), cx.sm);

    const clang::Expr* lhs = StripParenImpCasts(binop->getLHS());
    const clang::Expr* rhs = StripParenImpCasts(binop->getRHS());
    if (lhs) o.targetExprText = CollapseWS(GetExprSourceText(lhs, cx.sm, cx.lo));
    if (rhs) o.valueExprText  = CollapseWS(GetExprSourceText(rhs, cx.sm, cx.lo));

    auto isSubscriptLike = [](const clang::Expr* e) {
        if (!e) return false;
        if (llvm::isa<clang::ArraySubscriptExpr>(e)) return true;
        if (const auto* op = llvm::dyn_cast<clang::CXXOperatorCallExpr>(e)) {
            return op->getOperator() == clang::OO_Subscript;
        }
        return false;
    };
    const bool lhsIsSubscript = isSubscriptLike(lhs);
    const bool isCompound = binop->isCompoundAssignmentOp();
    const bool looksLikeTensor =
        o.targetExprText.find("_tensor") != std::string::npos;

    if (isCompound) {
        o.kind = FlowHostOutputKind::ScalarSelfUpdate;
        switch (binop->getOpcode()) {
            case clang::BO_AddAssign: o.compoundOp = "+="; break;
            case clang::BO_SubAssign: o.compoundOp = "-="; break;
            case clang::BO_MulAssign: o.compoundOp = "*="; break;
            case clang::BO_DivAssign: o.compoundOp = "/="; break;
            case clang::BO_RemAssign: o.compoundOp = "%="; break;
            case clang::BO_AndAssign: o.compoundOp = "&="; break;
            case clang::BO_OrAssign:  o.compoundOp = "|="; break;
            case clang::BO_XorAssign: o.compoundOp = "^="; break;
            case clang::BO_ShlAssign: o.compoundOp = "<<="; break;
            case clang::BO_ShrAssign: o.compoundOp = ">>="; break;
            default:                  o.compoundOp = binop->getOpcodeStr().str(); break;
        }
    } else if (lhsIsSubscript) {
        o.kind = looksLikeTensor ? FlowHostOutputKind::TensorAssign
                                 : FlowHostOutputKind::VectorElemWrite;
    } else {
        // 非 subscript / 非 compound 的赋值不走 Unknown 兜底，
        // ClassifyTopLevelStmt 早先已把 `x = helper(...)` 改写为 HelperInvocation，
        // 这里只剩破坏性场景。
        return BuildDiagnosticForUnsupportedStmt(
            cx, binop, "hostflow.unsupported_stmt",
            "non-subscript scalar assignment to `" + o.targetExprText +
                "` is not supported as HostOutput");
    }

    // 尝试 tensor alias 重写：LHS 解析为 alias 时把 `aliasA[i] = aliasB[i-1]` 等
    // 改写为 `(rootA.data())[...] = (rootB.data())[...]`。
    if (!isCompound && cx.aliasMap) {
        const std::string ownerName =
            cx.ownerFn ? cx.ownerFn->getNameAsString() : std::string();
        HostTensorAliasInfo lhsInfo;
        const bool lhsResolved =
            TryResolveHostTensorAliasFromText(
                o.targetExprText, ownerName, *cx.aliasMap, lhsInfo) &&
            lhsInfo.valid;
        if (lhsResolved) {
            HostTensorAliasInfo rhsInfo;
            const bool rhsResolved =
                rhs && TryResolveHostTensorAliasFromText(
                    o.valueExprText, ownerName, *cx.aliasMap, rhsInfo) &&
                rhsInfo.valid;
            const std::size_t uniq =
                cx.tensorAssignCounter ? (*cx.tensorAssignCounter)++ : 0;
            TensorAliasAccessPlan plan = BuildTensorAssignmentPlan(
                lhsInfo,
                rhsResolved ? &rhsInfo : nullptr,
                rhsResolved ? std::string() : o.valueExprText,
                uniq);
            std::string materialized = EmitTensorAliasAccessPlan(plan);
            if (materialized.empty()) {
                return BuildDiagnosticForUnsupportedStmt(
                    cx, binop, "hostflow.tensor_assign_unresolved",
                    "tensor alias assignment lowering failed for `" +
                        o.targetExprText + "`");
            }
            o.generatedCode = std::move(materialized);
            o.kind = FlowHostOutputKind::TensorAssign;
        } else if (looksLikeTensor && lhsIsSubscript) {
            // 目标看起来是 tensor alias 但 aliasMap 无法解析 → 硬失败。
            return BuildDiagnosticForUnsupportedStmt(
                cx, binop, "hostflow.tensor_assign_unresolved",
                "tensor-like target `" + o.targetExprText +
                    "` could not be resolved to a host alias");
        }
    }
    return cx.plan.addHostOutput(std::move(o));
}

// 前向声明：HostLoop / HostBranch 需要递归。
FlowStepId ClassifyTopLevelStmt(ClassifyContext& cx, const clang::Stmt* stmt);
void ClassifyBody(ClassifyContext& cx, const clang::Stmt* body,
                  std::vector<FlowStepId>& outIds);

FlowStepId BuildHostLoopFromStmt(ClassifyContext& cx, const clang::Stmt* stmt) {
    // Temporal schedule path：lowering 已将用户 host 侧 while/for 替换为 translator
    // 内置的 temporal wrapper（在 launch body 内），且 wrapper 已吸收 body 内的
    // scalar+= / tensor 赋值等更新。HostFlowBuilder 只需把 DAC stmt 单独挂出来，
    // 其余 body stmts 视为已被 launch block 包含、跳过发射。
    if (MatchDacEntryForStmt(cx, stmt) != nullptr) {
        for (const ExprEntry* e : cx.fnEntries) {
            if (!e->artifact) continue;
            if (!e->artifact->temporalScheduleApplied) continue;
            if (!e->exprLoc.isValid()) continue;
            if (!IsStmtContainingLoc(stmt, cx.sm, e->exprLoc)) continue;
            const clang::Stmt* loopBody = GetLoopBody(stmt);
            FlowStepId last = kInvalidFlowStepId;
            auto handleStmt = [&](const clang::Stmt* sub) {
                if (!sub) return;
                // 仅 classify 承载 DAC 的 stmt；其余被 temporal wrapper 吸收。
                if (IsStmtContainingLoc(sub, cx.sm, e->exprLoc)) {
                    last = ClassifyTopLevelStmt(cx, sub);
                }
            };
            if (const auto* cs = llvm::dyn_cast_or_null<clang::CompoundStmt>(loopBody)) {
                for (const clang::Stmt* sub : cs->body()) handleStmt(sub);
            } else if (loopBody) {
                handleStmt(loopBody);
            }
            return last;
        }
    }

    FlowHostLoop loop;
    loop.location = MakeSourceLoc(stmt->getBeginLoc(), cx.sm);
    loop.containsKernelLaunch = MatchDacEntryForStmt(cx, stmt) != nullptr;
    const clang::Stmt* body = nullptr;

    if (const auto* fs = llvm::dyn_cast<clang::ForStmt>(stmt)) {
        loop.kind = FlowHostLoopKind::For;
        if (const clang::Stmt* init = fs->getInit()) {
            if (const auto* declStmt = llvm::dyn_cast<clang::DeclStmt>(init)) {
                if (declStmt->isSingleDecl()) {
                    if (const auto* vd =
                            llvm::dyn_cast<clang::VarDecl>(declStmt->getSingleDecl())) {
                        loop.initIsDecl = true;
                        loop.initTypeText = vd->getType().getAsString();
                        loop.initVarName = vd->getNameAsString();
                        loop.inductionVar = loop.initVarName;
                        if (vd->getStorageClass() == clang::SC_Static)
                            loop.initStorageClass = "static";
                        if (vd->hasInit())
                            loop.initValueExpr = CollapseWS(GetExprSourceText(vd->getInit(), cx.sm, cx.lo));
                    }
                }
            } else if (const auto* binOp = llvm::dyn_cast<clang::BinaryOperator>(init)) {
                if (binOp->isAssignmentOp()) {
                    loop.initIsAssignment = true;
                    loop.initAssignTargetExpr = CollapseWS(GetExprSourceText(binOp->getLHS(), cx.sm, cx.lo));
                    loop.initAssignValueExpr = CollapseWS(GetExprSourceText(binOp->getRHS(), cx.sm, cx.lo));
                    loop.inductionVar = loop.initAssignTargetExpr;
                }
            }
        }
        if (const clang::Expr* cond = fs->getCond())
            loop.upperBoundExpr = CollapseWS(GetExprSourceText(cond, cx.sm, cx.lo));
        if (const clang::Expr* inc = fs->getInc())
            loop.stepExpr = CollapseWS(GetExprSourceText(inc, cx.sm, cx.lo));
        body = fs->getBody();
    } else if (const auto* ws = llvm::dyn_cast<clang::WhileStmt>(stmt)) {
        loop.kind = FlowHostLoopKind::While;
        if (const clang::Expr* cond = ws->getCond())
            loop.conditionText = CollapseWS(GetExprSourceText(cond, cx.sm, cx.lo));
        body = ws->getBody();
    } else if (const auto* ds = llvm::dyn_cast<clang::DoStmt>(stmt)) {
        loop.kind = FlowHostLoopKind::DoWhile;
        if (const clang::Expr* cond = ds->getCond())
            loop.conditionText = CollapseWS(GetExprSourceText(cond, cx.sm, cx.lo));
        body = ds->getBody();
    } else {
        return BuildDiagnosticForUnsupportedStmt(
            cx, stmt, "hostflow.unsupported_stmt",
            "BuildHostLoopFromStmt called with non-loop stmt");
    }

    // dac_for split path：如果 loop 内的 DAC 表达式 artifact 拆出了 pre-loop alloc /
    // post-loop free（Step 5 split path），把这些 DeviceAlloc 块作为独立的
    // FlowGeneratedCodeBlock step 提前插入 plan，让 top-level walk 把它们发射在
    // for-loop 之前；同样在 body 之后插 DeviceFree 块。DAC handler 会跳过同一份
    // 块的重复入栈。
    std::vector<const AscendCExpressionArtifact*> splitArtifactsInBody;
    if (loop.containsKernelLaunch) {
        for (const ExprEntry* e : cx.fnEntries) {
            if (!e->artifact) continue;
            if (e->artifact->setupBlockIndices.empty() &&
                e->artifact->cleanupBlockIndices.empty()) {
                continue;
            }
            if (!e->exprLoc.isValid()) continue;
            if (!IsStmtContainingLoc(stmt, cx.sm, e->exprLoc)) continue;
            splitArtifactsInBody.push_back(e->artifact);
        }
    }

    // 1. pre-loop：发射所有 DeviceAlloc role 的 generatedBlocks。
    for (const AscendCExpressionArtifact* art : splitArtifactsInBody) {
        for (const FlowGeneratedCodeBlock& blk : art->generatedBlocks) {
            if (blk.role != FlowGeneratedCodeRole::DeviceAlloc) continue;
            FlowGeneratedCodeBlock copy = blk;
            cx.plan.addGeneratedCodeBlock(std::move(copy));
        }
    }

    // 2. 占位：先 addHostLoop 拿到 stepId，再填 bodyStepIds。
    const FlowStepId selfId = cx.plan.addHostLoop(FlowHostLoop{loop});
    std::vector<FlowStepId> bodyIds;
    ClassifyBody(cx, body, bodyIds);
    // 把 bodyIds 写回 pool
    const FlowStep& step = cx.plan.steps[selfId];
    if (step.payloadIndex != static_cast<std::size_t>(-1) &&
        step.payloadIndex < cx.plan.hostLoops.size()) {
        cx.plan.hostLoops[step.payloadIndex].bodyStepIds = std::move(bodyIds);
    }

    // 3. post-loop：发射所有 DeviceFree role 的 generatedBlocks。
    for (const AscendCExpressionArtifact* art : splitArtifactsInBody) {
        for (const FlowGeneratedCodeBlock& blk : art->generatedBlocks) {
            if (blk.role != FlowGeneratedCodeRole::DeviceFree) continue;
            FlowGeneratedCodeBlock copy = blk;
            cx.plan.addGeneratedCodeBlock(std::move(copy));
        }
    }
    return selfId;
}

FlowStepId BuildHostBranchFromIf(ClassifyContext& cx,
                                  const clang::IfStmt* ifs) {
    if (!ifs) return kInvalidFlowStepId;
    FlowHostBranch br;
    br.location = MakeSourceLoc(ifs->getBeginLoc(), cx.sm);
    br.containsKernelLaunch = MatchDacEntryForStmt(cx, ifs) != nullptr;
    // 不缓存整段 if 源码；emitter 按 conditionText + then/else step 重建。
    if (const clang::Expr* cond = ifs->getCond())
        br.conditionText = CollapseWS(GetExprSourceText(cond, cx.sm, cx.lo));

    const FlowStepId selfId = cx.plan.addHostBranch(FlowHostBranch{br});
    std::vector<FlowStepId> thenIds, elseIds;
    ClassifyBody(cx, ifs->getThen(), thenIds);
    ClassifyBody(cx, ifs->getElse(), elseIds);

    const FlowStep& step = cx.plan.steps[selfId];
    if (step.payloadIndex != static_cast<std::size_t>(-1) &&
        step.payloadIndex < cx.plan.hostBranches.size()) {
        cx.plan.hostBranches[step.payloadIndex].thenStepIds = std::move(thenIds);
        cx.plan.hostBranches[step.payloadIndex].elseStepIds = std::move(elseIds);
    }
    return selfId;
}

// ============================================================
// Cleanup（delete / free / aclrtFree / aclrtDestroyStream / aclFinalize）
// ============================================================
FlowCleanupKind ClassifyCleanupCall(const clang::CallExpr* call) {
    if (!call) return FlowCleanupKind::Unknown;
    const clang::FunctionDecl* fn = call->getDirectCallee();
    if (!fn) return FlowCleanupKind::Unknown;
    const std::string name = fn->getNameAsString();
    if (name == "aclFinalize")          return FlowCleanupKind::AclFinalize;
    if (name == "aclrtDestroyStream")   return FlowCleanupKind::HostStreamDestroy;
    if (name == "aclrtFree")            return FlowCleanupKind::DeviceFree;
    if (name == "free")                 return FlowCleanupKind::HostFree;
    return FlowCleanupKind::Unknown;
}

FlowStepId BuildCleanupFromStmt(ClassifyContext& cx, const clang::Stmt* stmt,
                                 FlowCleanupKind kind) {
    FlowCleanupStep c;
    c.kind = kind;
    c.location = MakeSourceLoc(stmt->getBeginLoc(), cx.sm);

    // targetExprText：CXXDeleteExpr 取操作数；已知 cleanup CallExpr 取首参；其余留空。
    if (const auto* del = llvm::dyn_cast<clang::CXXDeleteExpr>(stmt)) {
        if (const clang::Expr* arg = del->getArgument()) {
            c.targetExprText = CollapseWS(GetExprSourceText(arg, cx.sm, cx.lo));
        }
    } else if (const auto* call = llvm::dyn_cast<clang::CallExpr>(stmt)) {
        if (call->getNumArgs() >= 1) {
            c.targetExprText =
                CollapseWS(GetExprSourceText(call->getArg(0), cx.sm, cx.lo));
        }
    }

    // AclFinalize 无参；其它 cleanup 必须能识别 target，否则硬失败。
    if (c.targetExprText.empty() && kind != FlowCleanupKind::AclFinalize) {
        return BuildDiagnosticForUnsupportedStmt(
            cx, stmt, "hostflow.cleanup_target_unresolved",
            "cleanup target not resolvable");
    }
    return cx.plan.addCleanup(std::move(c));
}

// ============================================================
// 顶层 stmt 分类
// ============================================================

// VarDecl 单独抽出，因为 DeclStmt 可能声明多个变量。
FlowStepId ClassifyVarDecl(ClassifyContext& cx, const clang::VarDecl* vd) {
    if (!vd) return kInvalidFlowStepId;
    clang::QualType qt = vd->getType();
    if (LooksLikeDacppContainerType(qt, cx.lo, nullptr) ||
        LooksLikeStdVectorType(qt, cx.lo)) {
        return BuildTensorBufferFromVarDecl(cx, vd);
    }
    return BuildHostInputFromVarDecl(cx, vd);
}

FlowStepId ClassifyDeclStmt(ClassifyContext& cx, const clang::DeclStmt* ds) {
    if (!ds) return kInvalidFlowStepId;
    FlowStepId last = kInvalidFlowStepId;
    for (const clang::Decl* d : ds->decls()) {
        if (const auto* vd = llvm::dyn_cast_or_null<clang::VarDecl>(d)) {
            last = ClassifyVarDecl(cx, vd);
        }
    }
    return last;
}

FlowStepId ClassifyTopLevelStmt(ClassifyContext& cx, const clang::Stmt* stmt) {
    if (!stmt) return kInvalidFlowStepId;

    // 顶层 ExprStmt 可能被 Clang 包成 ExprWithCleanups（带临时变量的表达式），
    // 或 CXXBindTemporaryExpr 等隐式包装。剥掉这一层，否则下面的 dyn_cast<CallExpr>
    // 类型检查无法命中真正的 CallExpr / CXXMemberCallExpr / BinaryOperator。
    while (true) {
        if (const auto* w =
                llvm::dyn_cast<clang::ExprWithCleanups>(stmt)) {
            stmt = w->getSubExpr();
            continue;
        }
        if (const auto* b =
                llvm::dyn_cast<clang::CXXBindTemporaryExpr>(stmt)) {
            stmt = b->getSubExpr();
            continue;
        }
        break;
    }
    if (!stmt) return kInvalidFlowStepId;

    // 0. ReturnStmt：legacy host body 由模板提供 return 0;，跳过用户的 return。
    if (llvm::isa<clang::ReturnStmt>(stmt)) {
        return kInvalidFlowStepId;
    }

    // 控制流必须先于 DAC 识别——外层 for/while/if 可能包含 DAC，DAC 应作为 body 内
    // 的一条 stmt 单独发射（KernelLaunch），不能被外层 stmt 直接吞掉。
    if (llvm::isa<clang::ForStmt>(stmt) ||
        llvm::isa<clang::WhileStmt>(stmt) ||
        llvm::isa<clang::DoStmt>(stmt)) {
        return BuildHostLoopFromStmt(cx, stmt);
    }
    if (const auto* ifs = llvm::dyn_cast<clang::IfStmt>(stmt)) {
        return BuildHostBranchFromIf(cx, ifs);
    }
    if (const auto* cs = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
        FlowStepId last = kInvalidFlowStepId;
        for (const clang::Stmt* sub : cs->body()) {
            last = ClassifyTopLevelStmt(cx, sub);
        }
        return last;
    }

    // 1. DAC `<->`：把 artifact.generatedBlocks 装到 plan.steps，
    //    再 emit 一个绑定到这些 step id 的 FlowKernelLaunch。
    if (const ExprEntry* e = MatchDacEntryForStmt(cx, stmt)) {
        std::vector<FlowStepId> commonDeclIds, allocIds, copyInIds, copyOutIds, freeIds;
        // split-path（dac_for）：DeviceAlloc / DeviceFree 已由外层 FlowHostLoop
        // hoist 到循环外发射，这里跳过避免重复。
        const bool splitPath =
            e->artifact && (!e->artifact->setupBlockIndices.empty() ||
                            !e->artifact->cleanupBlockIndices.empty());
        if (e->artifact) {
            for (const FlowGeneratedCodeBlock& blk : e->artifact->generatedBlocks) {
                if (splitPath &&
                    (blk.role == FlowGeneratedCodeRole::DeviceAlloc ||
                     blk.role == FlowGeneratedCodeRole::DeviceFree)) {
                    continue;
                }
                FlowGeneratedCodeBlock copy = blk;
                if (copy.route == FlowRouteKind::Unknown) {
                    copy.route = e->artifact->kernelLaunch.route;
                }
                const FlowStepId id = cx.plan.addGeneratedCodeBlock(std::move(copy));
                switch (blk.role) {
                    case FlowGeneratedCodeRole::CommonDecl:
                        commonDeclIds.push_back(id);
                        break;
                    case FlowGeneratedCodeRole::DeviceAlloc:
                        allocIds.push_back(id);
                        break;
                    case FlowGeneratedCodeRole::HostToDeviceCopy:
                        copyInIds.push_back(id);
                        break;
                    case FlowGeneratedCodeRole::DeviceToHostCopy:
                        copyOutIds.push_back(id);
                        break;
                    case FlowGeneratedCodeRole::DeviceFree:
                        freeIds.push_back(id);
                        break;
                    case FlowGeneratedCodeRole::KernelLaunch:
                    case FlowGeneratedCodeRole::TemporalLoopBody:
                    case FlowGeneratedCodeRole::MulticardRouteSetup:
                    case FlowGeneratedCodeRole::MulticardLaunch:
                    case FlowGeneratedCodeRole::MulticardRouteCleanup:
                    case FlowGeneratedCodeRole::MulticardRuntime:
                    case FlowGeneratedCodeRole::StreamCreate:
                    case FlowGeneratedCodeRole::StreamDestroy:
                    case FlowGeneratedCodeRole::AclRuntimeInit:
                    case FlowGeneratedCodeRole::AclRuntimeFinalize:
                    case FlowGeneratedCodeRole::AuxiliaryPrelude:
                    case FlowGeneratedCodeRole::Unknown:
                        // KernelLaunch / 其它非生命周期块：步骤已入 plan.steps，
                        // emitter 通过 ownerExpr+role 主动拾取，无需挂在专门列表上。
                        break;
                }
            }
        }

        FlowKernelLaunch launch;
        if (e->artifact) {
            launch = e->artifact->kernelLaunch;  // 复制 lowering 给出的 descriptor
        }
        launch.expressionIndex = e->expressionIndex;
        launch.location = MakeSourceLoc(stmt->getBeginLoc(), cx.sm);
        launch.commonDeclStepIds = std::move(commonDeclIds);
        launch.allocStepIds = std::move(allocIds);
        launch.copyInStepIds = std::move(copyInIds);
        launch.copyOutStepIds = std::move(copyOutIds);
        launch.freeStepIds = std::move(freeIds);
        return cx.plan.addKernelLaunch(std::move(launch));
    }

    // 3. DeclStmt（host input / tensor buffer）
    if (const auto* ds = llvm::dyn_cast<clang::DeclStmt>(stmt)) {
        return ClassifyDeclStmt(cx, ds);
    }

    // 4. Cleanup：delete / delete[] / 已知 cleanup 函数调用
    if (llvm::isa<clang::CXXDeleteExpr>(stmt)) {
        return BuildCleanupFromStmt(cx, stmt, FlowCleanupKind::HostDelete);
    }
    if (const auto* call = llvm::dyn_cast<clang::CallExpr>(stmt)) {
        FlowCleanupKind ck = ClassifyCleanupCall(call);
        if (ck != FlowCleanupKind::Unknown) {
            return BuildCleanupFromStmt(cx, stmt, ck);
        }
    }

    // 5. Output：cout / printf / .print()
    {
        FlowHostOutputKind okind = FlowHostOutputKind::Unknown;
        if (LooksLikeHostOutputStmtAst(stmt, &okind)) {
            return BuildHostOutputFromStmt(cx, stmt, okind);
        }
    }

    // 6. tensor[i] = ... / scalar+= / x = helper(...)
    if (const auto* binop = llvm::dyn_cast<clang::BinaryOperator>(stmt)) {
        if (binop->isAssignmentOp() || binop->isCompoundAssignmentOp()) {
            // `x = helper(...)` 形式（非复合赋值）改写为
            // HelperInvocation，把 LHS 文本作为 resultTargetName。
            if (binop->isAssignmentOp() && !binop->isCompoundAssignmentOp()) {
                const clang::Expr* rhs = StripParenImpCasts(binop->getRHS());
                if (const auto* call =
                        llvm::dyn_cast_or_null<clang::CallExpr>(rhs)) {
                    if (const clang::FunctionDecl* callee =
                            call->getDirectCallee()) {
                        if (IsHostSideHelperFunction(callee)) {
                            const clang::Expr* lhs =
                                StripParenImpCasts(binop->getLHS());
                            const std::string target = CollapseWS(
                                GetExprSourceText(lhs, cx.sm, cx.lo));
                            return BuildHelperInvocationFromCall(
                                cx, call, target);
                        }
                    }
                }
            }
            return BuildTensorAssignFromBinaryOperator(cx, binop);
        }
    }

    // 7. 顶层 helper 调用：含 host-side helper 函数调用，以及 method 调用（如
    //    vec.push_back(...)）等 - 都按 HelperInvocation 收集，emitter 按
    //    functionName/argTexts 重建调用语句。
    if (const auto* call = llvm::dyn_cast<clang::CallExpr>(stmt)) {
        // 跨函数内联：若 callee 是某个 DAC owner（main 调 calculateDecay 的场景），
        // 用 callee body 替代该 call site，让 main 与 callee 体作为整体发射。
        if (cx.inlineExpansionTargets) {
            const clang::FunctionDecl* callee = call->getDirectCallee();
            if (callee) {
                const clang::FunctionDecl* canonical = callee->getCanonicalDecl();
                const clang::FunctionDecl* def = callee->getDefinition();
                if (cx.inlineExpansionTargets->count(canonical) ||
                    (def && cx.inlineExpansionTargets->count(def))) {
                    const clang::FunctionDecl* target = def ? def : callee;
                    if (target && target->hasBody()) {
                        if (const auto* body = llvm::dyn_cast<clang::CompoundStmt>(
                                target->getBody())) {
                            FlowStepId last = kInvalidFlowStepId;
                            for (const clang::Stmt* sub : body->body()) {
                                last = ClassifyTopLevelStmt(cx, sub);
                            }
                            return last;
                        }
                    }
                }
            }
        }
        return BuildHelperInvocationFromCall(cx, call, /*resultTarget=*/"");
    }

    // 8. NullStmt / 单纯分号 — 静默忽略
    if (llvm::isa<clang::NullStmt>(stmt)) {
        return kInvalidFlowStepId;
    }

    // 未识别的 top-level host stmt 走 FlowDiagnostic(Error)，
    // 让 driver 显式失败而不是源码回放。诊断里附 stmtClass 名便于排查。
    const char* stmtKindRaw = stmt->getStmtClassName();
    const std::string stmtKind = stmtKindRaw ? stmtKindRaw : "?";
    return BuildDiagnosticForUnsupportedStmt(
        cx, stmt, "hostflow.unsupported_stmt",
        "unsupported top-level host stmt (stmtClass=" + stmtKind + ")");
}

void ClassifyBody(ClassifyContext& cx, const clang::Stmt* body,
                  std::vector<FlowStepId>& outIds) {
    if (!body) return;
    if (const auto* cs = llvm::dyn_cast<clang::CompoundStmt>(body)) {
        for (const clang::Stmt* sub : cs->body()) {
            FlowStepId id = ClassifyTopLevelStmt(cx, sub);
            if (id != kInvalidFlowStepId) outIds.push_back(id);
        }
        return;
    }
    FlowStepId id = ClassifyTopLevelStmt(cx, body);
    if (id != kInvalidFlowStepId) outIds.push_back(id);
}

// ============================================================
// Metadata 与入口
// ============================================================

void FillMetadata(AscendCFlowPlan& plan, const HostFlowBuildInput& in) {
    plan.builderRan = true;
    if (in.sourceFileHint) plan.sourceFileHint = *in.sourceFileHint;
    if (in.dacppFile) plan.totalExpressions = in.dacppFile->getNumExpression();
    if (in.expressionArtifacts) {
        for (const auto& a : *in.expressionArtifacts) {
            if (a.routeKind == AscendCRouteKind::Vector) plan.vectorExpressionCount++;
            else if (a.routeKind == AscendCRouteKind::Cube) plan.cubeExpressionCount++;
        }
    }
}

std::vector<ExprEntry> CollectExprEntries(AscendCFlowPlan& plan,
                                          const HostFlowBuildInput& in,
                                          const clang::SourceManager& sm) {
    std::vector<ExprEntry> entries;
    if (!in.dacppFile || !in.expressionArtifacts) return entries;

    std::unordered_map<int, const AscendCExpressionArtifact*> byExpr;
    byExpr.reserve(in.expressionArtifacts->size());
    for (const auto& a : *in.expressionArtifacts) {
        byExpr[a.expressionIndex] = &a;
    }

    for (int i = 0; i < in.dacppFile->getNumExpression(); ++i) {
        Expression* e = in.dacppFile->getExpression(i);
        if (!e) continue;
        auto it = byExpr.find(i);
        if (it == byExpr.end()) continue;

        ExprEntry ent;
        ent.expressionIndex = i;
        ent.artifact = it->second;
        ent.dacExpr = e->getDacExpr();
        ent.ownerFn = FindEnclosingFunction(ent.dacExpr, in.tuDecl, sm);
        ent.exprLoc = ent.dacExpr ? sm.getExpansionLoc(ent.dacExpr->getExprLoc())
                                  : clang::SourceLocation();

        if (!ent.ownerFn) {
            FlowDiagnostic d;
            d.severity = FlowDiagnosticSeverity::Error;
            d.code = "hostflow.no_owner_fn";
            d.message = "Expression " + std::to_string(i) +
                        " has no enclosing FunctionDecl";
            d.relatedExpr = i;
            d.location = MakeSourceLoc(ent.exprLoc, sm);
            plan.addDiagnostic(std::move(d));
            plan.partial = true;
            continue;
        }
        entries.push_back(std::move(ent));
    }

    std::sort(entries.begin(), entries.end(),
              [](const ExprEntry& a, const ExprEntry& b) {
                  if (a.artifact && b.artifact &&
                      a.artifact->sourceOrder != b.artifact->sourceOrder) {
                      return a.artifact->sourceOrder < b.artifact->sourceOrder;
                  }
                  return a.expressionIndex < b.expressionIndex;
              });

    return entries;
}

const FlowKernelLaunch* FindFirstLaunchForRoute(
    const HostFlowBuildInput& in,
    FlowRouteKind route) {
    if (!in.expressionArtifacts) return nullptr;
    for (const AscendCExpressionArtifact& artifact : *in.expressionArtifacts) {
        if (artifact.kernelLaunch.route == route) return &artifact.kernelLaunch;
    }
    return nullptr;
}

void AddHostRuntimeSetupBlocks(AscendCFlowPlan& plan,
                               const HostFlowBuildInput& in) {
    for (FlowRouteKind route : {FlowRouteKind::Vector, FlowRouteKind::Cube}) {
        const FlowKernelLaunch* launch = FindFirstLaunchForRoute(in, route);
        if (!launch) continue;
        const FlowRouteRuntimeNames* names =
            plan.findRouteRuntimeNames(route);
        if (!names) continue;

        FlowGeneratedCodeBlock block;
        block.label = route == FlowRouteKind::Vector
            ? "vector.route.setup" : "cube.route.setup";
        block.route = route;
        if (launch->multicardEnabled) {
            const FlowMulticardConfig& config = launch->multicard;
            const std::string requested = !config.requestedRankDimExpr.empty()
                ? config.requestedRankDimExpr
                : (!config.rankDimExpr.empty() ? config.rankDimExpr : "8");
            const std::string rankEnv = config.rankDimEnvName.empty()
                ? "DACPP_ASCENDC_RANK_DIM" : config.rankDimEnvName;
            block.role = FlowGeneratedCodeRole::MulticardRouteSetup;
            block.code =
                "    const uint32_t " + names->rankCount +
                " = dacpp_host::readRankCount(" + names->session + ", \"" +
                rankEnv + "\", " + requested + ", static_cast<size_t>(0));\n"
                "    dacpp_host::Route " + names->routeObject + "(" +
                names->session + ", " + names->rankCount + ");\n"
                "    if (!" + names->routeObject + ".ok()) return " +
                names->session + ".finish();\n";
        } else {
            block.role = FlowGeneratedCodeRole::AclRuntimeInit;
            block.code =
                "#ifndef ASCENDC_CPU_DEBUG\n"
                "    dacpp_host::Device " + names->device + "(" +
                names->session + ", 0);\n"
                "    if (!" + names->device + ".ok()) return " +
                names->session + ".finish();\n"
                "    aclrtStream " + names->stream + " = " +
                names->device + ".stream();\n"
                "#endif\n";
        }
        plan.addGeneratedCodeBlock(std::move(block));
    }
}

void AddHostRuntimeCleanupBlocks(AscendCFlowPlan& plan,
                                 const HostFlowBuildInput& in) {
    for (FlowRouteKind route : {FlowRouteKind::Vector, FlowRouteKind::Cube}) {
        const FlowKernelLaunch* launch = FindFirstLaunchForRoute(in, route);
        if (!launch) continue;
        const FlowRouteRuntimeNames* names =
            plan.findRouteRuntimeNames(route);
        if (!names) continue;
        FlowGeneratedCodeBlock block;
        block.role = launch->multicardEnabled
            ? FlowGeneratedCodeRole::MulticardRouteCleanup
            : FlowGeneratedCodeRole::AclRuntimeFinalize;
        block.label = route == FlowRouteKind::Vector
            ? "vector.route.cleanup" : "cube.route.cleanup";
        block.route = route;
        block.code = launch->multicardEnabled
            ? "    " + names->routeObject + ".close();\n"
            : "#ifndef ASCENDC_CPU_DEBUG\n    " + names->device +
              ".close();\n#endif\n";
        plan.addGeneratedCodeBlock(std::move(block));
    }
}

void FinalizeRuntimeBuffersAndFeatures(AscendCFlowPlan& plan) {
    std::unordered_map<std::string, FlowBufferId> ids;
    std::unordered_map<int, FlowBufferId> nextByRoute;
    for (const FlowKernelLaunch& launch : plan.kernelLaunches) {
        FlowBufferId& next = nextByRoute[static_cast<int>(launch.route)];
        for (const FlowRuntimeBuffer& buffer : launch.runtimeBuffers) {
            if (buffer.id == kInvalidFlowBufferId) continue;
            next = std::max(next, static_cast<FlowBufferId>(buffer.id + 1));
            ids.emplace(
                std::to_string(static_cast<int>(launch.route)) + ":" +
                    buffer.logicalName,
                buffer.id);
        }
    }
    auto IdFor = [&](FlowRouteKind route, const std::string& logicalKey) {
        const std::string key =
            std::to_string(static_cast<int>(route)) + ":" + logicalKey;
        auto found = ids.find(key);
        if (found != ids.end()) return found->second;
        FlowBufferId& next = nextByRoute[static_cast<int>(route)];
        const FlowBufferId id = next++;
        ids.emplace(key, id);
        return id;
    };

    for (FlowKernelLaunch& launch : plan.kernelLaunches) {
        if (launch.route == FlowRouteKind::Vector) {
            plan.runtimeFeatureMask |=
                static_cast<uint32_t>(HostRuntimeFeature::CheckedMath) |
                static_cast<uint32_t>(HostRuntimeFeature::VectorTiling);
        } else if (launch.route == FlowRouteKind::Cube) {
            plan.runtimeFeatureMask |=
                static_cast<uint32_t>(HostRuntimeFeature::CubeTiling) |
                static_cast<uint32_t>(HostRuntimeFeature::Fp16Bridge);
        }
        plan.runtimeFeatureMask |= static_cast<uint32_t>(
            launch.multicardEnabled ? HostRuntimeFeature::MulticardRoute
                                    : HostRuntimeFeature::SingleCard);
        if (launch.multicard.persistentWorkers) {
            plan.runtimeFeatureMask |=
                static_cast<uint32_t>(HostRuntimeFeature::PersistentLoop);
        }

        for (FlowMulticardParamPartition& partition :
             launch.multicard.paramPartitions) {
            if (partition.bufferId == kInvalidFlowBufferId)
                partition.bufferId = IdFor(launch.route, partition.name);
            for (FlowRuntimeBuffer& buffer : launch.runtimeBuffers) {
                if (buffer.logicalName == partition.name) {
                    buffer.id = partition.bufferId;
                    buffer.distribution = partition.distribution;
                    break;
                }
            }
        }
        for (FlowRuntimeBuffer& buffer : launch.runtimeBuffers) {
            if (buffer.id == kInvalidFlowBufferId) {
                buffer.id = IdFor(launch.route, buffer.logicalName);
            }
        }
    }
}

void AddMulticardInternalReuseEdges(AscendCFlowPlan& plan) {
    if (plan.kernelLaunches.size() < 2) return;
    std::unordered_map<FlowExprIndex, std::vector<FlowBufferId>>
        staticReuseCandidates;
    for (FlowKernelLaunch& launch : plan.kernelLaunches) {
        if (!launch.multicardEnabled) continue;
        for (FlowRuntimeBuffer& buffer : launch.runtimeBuffers) {
            if (buffer.producerResident) {
                staticReuseCandidates[launch.expressionIndex].push_back(
                    buffer.id);
                buffer.producerResident = false;
            }
        }
    }
    auto FindLaunchStep = [&](FlowExprIndex expr) {
        for (std::size_t id = 0; id < plan.steps.size(); ++id) {
            const FlowStep& step = plan.steps[id];
            if (step.kind != FlowStepKind::KernelLaunch ||
                step.payloadIndex >= plan.kernelLaunches.size()) continue;
            if (plan.kernelLaunches[step.payloadIndex].expressionIndex == expr) {
                return static_cast<FlowStepId>(id);
            }
        }
        return kInvalidFlowStepId;
    };
    auto HasUnsafeInterveningMutation = [&](FlowStepId begin, FlowStepId end) {
        if (begin == kInvalidFlowStepId || end == kInvalidFlowStepId ||
            begin >= end) return true;
        for (FlowStepId id = begin + 1; id < end; ++id) {
            const FlowStep& step = plan.steps[id];
            switch (step.kind) {
                case FlowStepKind::HostOutput:
                    if (step.payloadIndex < plan.hostOutputs.size()) {
                        const FlowHostOutputKind kind =
                            plan.hostOutputs[step.payloadIndex].kind;
                        if (kind == FlowHostOutputKind::TensorAssign ||
                            kind == FlowHostOutputKind::ScalarSelfUpdate ||
                            kind == FlowHostOutputKind::VectorElemWrite) {
                            return true;
                        }
                    }
                    break;
                case FlowStepKind::HelperInvocation:
                    if (step.payloadIndex >= plan.helperInvocations.size() ||
                        plan.helperInvocations[step.payloadIndex].role !=
                            FlowHelperRole::PrintLike) {
                        return true;
                    }
                    break;
                case FlowStepKind::HostLoop:
                case FlowStepKind::HostBranch:
                case FlowStepKind::Cleanup:
                case FlowStepKind::Diagnostic:
                case FlowStepKind::Raw:
                    return true;
                default:
                    break;
            }
        }
        return false;
    };
    auto FindRuntimeBuffer = [](const FlowKernelLaunch& launch,
                                FlowBufferId id) -> const FlowRuntimeBuffer* {
        for (const FlowRuntimeBuffer& buffer : launch.runtimeBuffers) {
            if (buffer.id == id) return &buffer;
        }
        return nullptr;
    };
    for (std::size_t i = 1; i < plan.kernelLaunches.size(); ++i) {
        const FlowKernelLaunch& producer = plan.kernelLaunches[i - 1];
        FlowKernelLaunch& consumer = plan.kernelLaunches[i];
        if (!producer.multicardEnabled || !consumer.multicardEnabled ||
            producer.route != consumer.route ||
            producer.multicard.strategy != "record-field-pointwise" ||
            consumer.multicard.strategy != "record-field-pointwise" ||
            producer.multicard.shardDim != consumer.multicard.shardDim ||
            producer.multicard.shardAxis != consumer.multicard.shardAxis ||
            producer.multicard.requestedRankDimExpr !=
                consumer.multicard.requestedRankDimExpr ||
            producer.multicard.rankDimEnvName !=
                consumer.multicard.rankDimEnvName) {
            continue;
        }
        if (HasUnsafeInterveningMutation(
                FindLaunchStep(producer.expressionIndex),
                FindLaunchStep(consumer.expressionIndex))) {
            continue;
        }
        for (const FlowMulticardParamPartition& out :
                 producer.multicard.paramPartitions) {
            if ((out.readWriteMode != 1 && out.readWriteMode != 2) ||
                !out.deviceResident ||
                out.bufferId == kInvalidFlowBufferId) {
                continue;
            }
            for (const FlowMulticardParamPartition& in :
                     consumer.multicard.paramPartitions) {
                if ((in.readWriteMode != 0 && in.readWriteMode != 2) ||
                    in.bufferId != out.bufferId ||
                    in.distribution != out.distribution ||
                    in.shardAxis != out.shardAxis) {
                    continue;
                }
                const FlowRuntimeBuffer* produced =
                    FindRuntimeBuffer(producer, out.bufferId);
                const FlowRuntimeBuffer* consumed =
                    FindRuntimeBuffer(consumer, in.bufferId);
                if (!produced || !consumed ||
                    produced->elementTypeText != consumed->elementTypeText ||
                    produced->distribution != consumed->distribution ||
                    produced->leftHalo != consumed->leftHalo ||
                    produced->rightHalo != consumed->rightHalo) {
                    continue;
                }
                FlowExprDataflowEdge edge;
                edge.producerExpr = producer.expressionIndex;
                edge.consumerExpr = consumer.expressionIndex;
                edge.bufferName = out.name;
                edge.kind = FlowExprDataflowKind::InternalReuse;
                plan.addDataflowEdge(std::move(edge));
                if (std::find(consumer.internalReuseBufferIds.begin(),
                              consumer.internalReuseBufferIds.end(),
                              out.bufferId) ==
                    consumer.internalReuseBufferIds.end()) {
                    consumer.internalReuseBufferIds.push_back(out.bufferId);
                }
                for (FlowRuntimeBuffer& buffer : consumer.runtimeBuffers) {
                    if (buffer.id == out.bufferId) {
                        buffer.producerResident = true;
                    }
                }
            }
        }
    }
    for (const auto& entry : staticReuseCandidates) {
        const FlowKernelLaunch* launch = nullptr;
        for (const FlowKernelLaunch& candidate : plan.kernelLaunches) {
            if (candidate.expressionIndex == entry.first) {
                launch = &candidate;
                break;
            }
        }
        if (!launch) continue;
        for (FlowBufferId id : entry.second) {
            if (std::find(launch->internalReuseBufferIds.begin(),
                          launch->internalReuseBufferIds.end(), id) !=
                launch->internalReuseBufferIds.end()) {
                continue;
            }
            FlowDiagnostic diagnostic;
            diagnostic.severity = FlowDiagnosticSeverity::Error;
            diagnostic.code = "hostflow.static_reuse_unproven";
            diagnostic.message =
                "static resident reuse candidate failed plan proof for "
                "expression #" + std::to_string(entry.first) +
                ", buffer #" + std::to_string(id);
            plan.addDiagnostic(std::move(diagnostic));
            plan.partial = true;
        }
    }
}

// 递归扫描 step id 列表，把紧跟 KernelLaunch 的 TensorPrint（命中
// runtime alias name）挂到该 launch 的 postLaunchStepIds。
void AttachPostLaunchStepsInList(
    AscendCFlowPlan& plan,
    const std::vector<FlowStepId>& stepIds,
    const std::unordered_map<int, std::vector<std::string>>& runtimeAliasNamesByExpr) {

    for (std::size_t i = 0; i < stepIds.size(); ++i) {
        FlowStepId sid = stepIds[i];
        if (sid >= plan.steps.size()) continue;
        const FlowStep& s = plan.steps[sid];

        if (s.kind == FlowStepKind::HostLoop &&
            s.payloadIndex < plan.hostLoops.size()) {
            AttachPostLaunchStepsInList(
                plan, plan.hostLoops[s.payloadIndex].bodyStepIds,
                runtimeAliasNamesByExpr);
            continue;
        }
        if (s.kind == FlowStepKind::HostBranch &&
            s.payloadIndex < plan.hostBranches.size()) {
            const FlowHostBranch& br = plan.hostBranches[s.payloadIndex];
            AttachPostLaunchStepsInList(plan, br.thenStepIds, runtimeAliasNamesByExpr);
            AttachPostLaunchStepsInList(plan, br.elseStepIds, runtimeAliasNamesByExpr);
            continue;
        }

        if (s.kind != FlowStepKind::KernelLaunch ||
            s.payloadIndex >= plan.kernelLaunches.size()) {
            continue;
        }

        FlowKernelLaunch& launch = plan.kernelLaunches[s.payloadIndex];
        auto aliasIt = runtimeAliasNamesByExpr.find(launch.expressionIndex);
        if (aliasIt == runtimeAliasNamesByExpr.end()) continue;
        const std::vector<std::string>& aliasNames = aliasIt->second;

        for (std::size_t j = i + 1; j < stepIds.size(); ++j) {
            FlowStepId fid = stepIds[j];
            if (fid >= plan.steps.size()) continue;
            const FlowStep& following = plan.steps[fid];
            if (following.kind == FlowStepKind::KernelLaunch) break;

            if (following.kind == FlowStepKind::HostOutput &&
                following.payloadIndex < plan.hostOutputs.size()) {
                const FlowHostOutput& output =
                    plan.hostOutputs[following.payloadIndex];
                if (output.kind != FlowHostOutputKind::TensorPrint) continue;
                for (const std::string& name : aliasNames) {
                    if (output.targetExprText == name) {
                        launch.postLaunchStepIds.push_back(fid);
                        break;
                    }
                }
            }
        }
    }
}

} // namespace

AscendCFlowPlan BuildHostFlowPlanFromAst(const HostFlowBuildInput& in) {
    AscendCFlowPlan plan;
    FillMetadata(plan, in);

    if (in.mixedRouteConflict) {
        FlowDiagnostic d;
        d.severity = FlowDiagnosticSeverity::Error;
        d.code = "hostflow.mixed_route_conflict";
        d.message = in.mixedRouteReason.empty()
                        ? "mixed Vector/Cube route order conflict"
                        : in.mixedRouteReason;
        plan.addDiagnostic(std::move(d));
        plan.partial = true;
    }

    if (!in.dacppFile || !in.tuDecl || !in.sourceManager ||
        !in.langOptions || !in.expressionArtifacts) {
        FlowDiagnostic d;
        d.severity = FlowDiagnosticSeverity::Error;
        d.code = "hostflow.missing_input";
        d.message = "BuildHostFlowPlanFromAst missing required input fields";
        plan.addDiagnostic(std::move(d));
        plan.partial = true;
        return plan;
    }

    const clang::SourceManager& sm = *in.sourceManager;
    const clang::LangOptions& lo = *in.langOptions;

    for (const AscendCExpressionArtifact& artifact :
         *in.expressionArtifacts) {
        const FlowRouteRuntimeNames& names = artifact.runtimeNames;
        if (names.route == FlowRouteKind::Unknown ||
            names.session.empty()) {
            continue;
        }
        if (!plan.findRouteRuntimeNames(names.route)) {
            plan.routeRuntimeNames.push_back(names);
            for (const FlowGeneratedName& generated :
                 artifact.generatedNames) {
                plan.addGeneratedName(generated);
            }
        }
    }

    // Route ownership blocks surround the AST-derived HostFlow. They are
    // route-filtered by the emitter and therefore execute exactly once per
    // generated Vector/Cube host file, independent of launch count.
    AddHostRuntimeSetupBlocks(plan, in);

    std::vector<ExprEntry> entries = CollectExprEntries(plan, in, sm);

    std::vector<const clang::FunctionDecl*> ownerOrder;
    std::unordered_map<const clang::FunctionDecl*, std::vector<const ExprEntry*>> entriesByFn;
    for (const ExprEntry& e : entries) {
        if (entriesByFn.find(e.ownerFn) == entriesByFn.end()) {
            ownerOrder.push_back(e.ownerFn);
        }
        entriesByFn[e.ownerFn].push_back(&e);
    }

    // 跨函数内联：如果某个 owner function 不是 main，并且能在 TU 内找到 main，把 main
    // 作为 effective owner（让 main 的 host-setup 代码也出现在 host body 里），并把
    // 原 owner 加入 inlineExpansionTargets — main 中调用该 owner 的 call site 会被
    // 替换为 owner body 的递归分类。
    const clang::FunctionDecl* mainFn = nullptr;
    if (in.tuDecl) {
        VisitAllFunctionsInTU(in.tuDecl, [&](const clang::FunctionDecl* fn) {
            if (fn->getNameAsString() == "main") { mainFn = fn; return true; }
            return false;
        });
    }

    // 每个 owner function 各自的 tensor alias 命名空间与计数器（跨函数不共享）。
    for (const clang::FunctionDecl* fn : ownerOrder) {
        if (!fn || !fn->hasBody()) continue;
        const clang::FunctionDecl* effectiveFn = fn;
        std::unordered_set<const clang::FunctionDecl*> inlineTargets;
        if (mainFn && fn != mainFn && fn->getNameAsString() != "main") {
            effectiveFn = mainFn;
            inlineTargets.insert(fn->getCanonicalDecl());
            if (const clang::FunctionDecl* def = fn->getDefinition()) {
                inlineTargets.insert(def);
            }
        }
        const auto* body = llvm::dyn_cast<clang::CompoundStmt>(effectiveFn->getBody());
        if (!body) continue;

        // 收集本 effective owner（main 或原 owner）内的 tensor alias。原 owner（如
        // calculateDecay）的 alias 在 inline 展开时另外按需收集（合并到同一个 map）。
        std::unordered_map<std::string, HostTensorAliasInfo> aliasMap;
        TensorAliasResolverContext resolverCtx;
        resolverCtx.sourceManager = &sm;
        resolverCtx.langOptions = &lo;
        CollectLiveTensorAliasesInFunction(effectiveFn, resolverCtx, aliasMap);
        if (effectiveFn != fn) {
            CollectLiveTensorAliasesInFunction(fn, resolverCtx, aliasMap);
        }

        // Runtime alias 覆盖 — 把 lowering 生成的 staging buffer
        // alias 合并到 aliasMap，优先级高于源码 ctor 推断结果。
        if (in.expressionArtifacts) {
            const std::string ownerName =
                effectiveFn ? effectiveFn->getNameAsString() : std::string();
            for (const auto& artifact : *in.expressionArtifacts) {
                for (const HostTensorAliasInfo& rtAlias : artifact.runtimeTensorAliases) {
                    if (!rtAlias.valid) continue;
                    const std::string key =
                        BuildHostTensorAliasKey(ownerName, rtAlias.aliasName);
                    aliasMap.insert_or_assign(key, rtAlias);
                }
            }
        }

        std::size_t tensorPrintCounter = 0;
        std::size_t tensorAssignCounter = 0;

        ClassifyContext cx{plan, in, sm, lo, entriesByFn[fn],
                           fn, &aliasMap, &tensorPrintCounter, &tensorAssignCounter,
                           inlineTargets.empty() ? nullptr : &inlineTargets};
        for (const clang::Stmt* stmt : body->body()) {
            ClassifyTopLevelStmt(cx, stmt);
        }
    }

    FinalizeRuntimeBuffersAndFeatures(plan);
    AddMulticardInternalReuseEdges(plan);

    // Post-launch attachment — 把紧跟 KernelLaunch 的、使用
    // runtime alias 的 HostOutput（如 tensor print）挂到 postLaunchStepIds，
    // 让 emitter 在 D2H copy 之后、free 之前发射。
    if (in.expressionArtifacts) {
        std::unordered_map<int, std::vector<std::string>> runtimeAliasNamesByExpr;
        for (const auto& a : *in.expressionArtifacts) {
            for (const HostTensorAliasInfo& rtAlias : a.runtimeTensorAliases) {
                if (rtAlias.valid) {
                    runtimeAliasNamesByExpr[a.expressionIndex].push_back(
                        rtAlias.aliasName);
                }
            }
        }
        if (!runtimeAliasNamesByExpr.empty()) {
            std::vector<FlowStepId> topLevelIds;
            topLevelIds.reserve(plan.steps.size());
            for (std::size_t k = 0; k < plan.steps.size(); ++k) {
                topLevelIds.push_back(k);
            }
            AttachPostLaunchStepsInList(plan, topLevelIds, runtimeAliasNamesByExpr);
        }
    }

    AddHostRuntimeCleanupBlocks(plan, in);

    return plan;
}

} // namespace ascendc
} // namespace dacppTranslator
