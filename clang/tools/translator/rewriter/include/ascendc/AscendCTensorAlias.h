#ifndef TRANSLATOR_REWRITER_ASCENDC_TENSOR_ALIAS_H
#define TRANSLATOR_REWRITER_ASCENDC_TENSOR_ALIAS_H

#include <string>
#include <unordered_map>
#include <vector>

#include "ascendc/HostTensorAliasInfo.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
class Expr;
class FunctionDecl;
class LangOptions;
class SourceManager;
}

namespace dacppTranslator {
namespace ascendc {

struct TensorAliasResolverContext {
    const clang::SourceManager* sourceManager = nullptr;
    const clang::LangOptions* langOptions = nullptr;
};

std::vector<std::string> SplitTopLevelByCommaPreserveNesting(const std::string& text);
int ExtractTensorRankFromTypeName(const std::string& typeName);
std::string ExtractTensorElementTypeFromTypeName(const std::string& typeName);
std::string MakeParenthesizedExpr(const std::string& expr);
std::string BuildSumExpr(const std::string& lhs, const std::string& rhs);
std::string BuildProductExpr(const std::string& lhs, const std::string& rhs);
std::vector<std::string> BuildRowMajorStrideExprs(const std::vector<std::string>& shapeExprs);
std::string StripBalancedOuterParens(const std::string& rawText);

bool TryApplyTensorIndexText(const HostTensorAliasInfo& baseInfo,
                             llvm::StringRef rawIndexText,
                             HostTensorAliasInfo& outInfo);

bool TryResolveHostTensorAliasFromText(
    llvm::StringRef rawExprText,
    llvm::StringRef ownerFunctionName,
    const std::unordered_map<std::string, HostTensorAliasInfo>& aliasMap,
    HostTensorAliasInfo& outInfo);

void MergeHostTensorAliasMaps(
    std::unordered_map<std::string, HostTensorAliasInfo>& ioDst,
    const std::unordered_map<std::string, HostTensorAliasInfo>& src);

void CollectLiveTensorAliasesInFunction(
    const clang::FunctionDecl* ownerFn,
    const TensorAliasResolverContext& context,
    std::unordered_map<std::string, HostTensorAliasInfo>& ioAliases);

// ============================================================
// TensorAliasAccessPlan
// ============================================================
//
// 把 driver 之前 emit-string 风格的 tensor alias 发射函数改成"先 plan、再 emit"模式。
// Plan 是纯数据：包含已解析的 HostTensorAliasInfo 与发射用的局部命名 token，
// emitter 据此生成 translator-generated 代码（与已删除的 replay 文本路径完全解耦）。

enum class TensorAliasAccessKind {
    Unknown,
    ScalarRead,   // rank-0 标量读
    ViewPrint,    // tensor view 的 host 端可打印 materialization
    Assignment,   // tensor alias 之间或 tensor=标量 的赋值
};

struct TensorAliasAccessPlan {
    TensorAliasAccessKind kind = TensorAliasAccessKind::Unknown;
    bool valid = false;

    // LHS / RHS alias 信息：ScalarRead/ViewPrint 只用 lhsInfo。
    HostTensorAliasInfo lhsInfo;
    HostTensorAliasInfo rhsInfo;
    bool rhsIsAlias = false;
    std::string rhsScalarExpr;   // Assignment 且 rhs 非 alias 时的标量 RHS 文本

    // 命名 token：避免不同 alias step 之间发射的临时变量名冲突。
    std::string uniqueSuffix;    // ViewPrint
    std::size_t uniqueIndex = 0; // Assignment
};

// Plan builders：纯函数；校验输入后置 valid。
TensorAliasAccessPlan BuildTensorScalarReadPlan(const HostTensorAliasInfo& aliasInfo);
TensorAliasAccessPlan BuildTensorViewPrintPlan(
    const HostTensorAliasInfo& aliasInfo,
    const std::string& uniqueSuffix);
TensorAliasAccessPlan BuildTensorAssignmentPlan(
    const HostTensorAliasInfo& lhsInfo,
    const HostTensorAliasInfo* rhsInfo,
    const std::string& rhsScalarExpr,
    std::size_t uniqueIndex);

// Emit-from-plan dispatcher：返回 translator 生成的源码字符串。
// 任何用户源码 Stmt 原文都不允许出现在这里——只产 ACL/runtime/scalar-read 等生成代码。
std::string EmitTensorAliasAccessPlan(const TensorAliasAccessPlan& plan);

// 旧字符串接口的 plan-wrapper：保留供 driver 在 Step 8 拆除前继续使用。
std::string EmitTensorScalarRead(const HostTensorAliasInfo& aliasInfo);
std::string EmitTensorViewPrint(
    const HostTensorAliasInfo& aliasInfo,
    const std::string& uniqueSuffix);
std::string EmitTensorAssignment(
    const HostTensorAliasInfo& lhsInfo,
    const HostTensorAliasInfo* rhsInfo,
    const std::string& rhsScalarExpr,
    std::size_t uniqueIndex);

// Plan emit 用到的底层 expr builder（公开给 Step 7 emitter）。
std::string BuildHostTensorIndexedAccessExpr(
    const HostTensorAliasInfo& aliasInfo,
    const std::vector<std::string>& indexExprs);
std::string MaybeCastTensorAssignmentRhs(
    const std::string& rhsExpr,
    const HostTensorAliasInfo& lhsInfo,
    const HostTensorAliasInfo* rhsInfo);

} // namespace ascendc
} // namespace dacppTranslator

#endif // TRANSLATOR_REWRITER_ASCENDC_TENSOR_ALIAS_H
