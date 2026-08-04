/*
把 Shell 参数语义显式化。

- 解析每个参数的读写属性、rank、shape。
- 归一化 split/window 信息（含 halo）。
- 识别广播标量参数。
*/
#include "ascendc_mlir/CalcShellInfo.h"

#include <algorithm>
#include <sstream>
#include <unordered_set>

#include "clang/AST/RecursiveASTVisitor.h"
#include "Shell.h"

namespace dacppTranslator {
namespace ascendc_mlir {
namespace {

// 功能：检查Is Empty Brace Index Expr相关条件。
static bool IsEmptyBraceIndexExpr(const clang::Expr* expr) {
    if (!expr) {
        return false;
    }

    expr = expr->IgnoreParenImpCasts();
    if (const auto* initList = clang::dyn_cast<clang::InitListExpr>(expr)) {
        return initList->getNumInits() == 0;
    }
    if (const auto* stdInit = clang::dyn_cast<clang::CXXStdInitializerListExpr>(expr)) {
        return IsEmptyBraceIndexExpr(stdInit->getSubExpr());
    }
    return false;
}

// 功能：尝试处理Try Extract Base Identifier相关逻辑。
static bool TryExtractBaseIdentifier(const clang::Expr* expr, std::string& outName) {
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

    if (const auto* memberExpr = clang::dyn_cast<clang::MemberExpr>(expr)) {
        return TryExtractBaseIdentifier(memberExpr->getBase(), outName);
    }

    if (const auto* subscriptExpr = clang::dyn_cast<clang::ArraySubscriptExpr>(expr)) {
        return TryExtractBaseIdentifier(subscriptExpr->getBase(), outName);
    }

    if (const auto* opCall = clang::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
        if (opCall->getOperator() == clang::OverloadedOperatorKind::OO_Subscript &&
            opCall->getNumArgs() > 0) {
            return TryExtractBaseIdentifier(opCall->getArg(0), outName);
        }
    }

    return false;
}

class BroadcastScalarParamDetector
    : public clang::RecursiveASTVisitor<BroadcastScalarParamDetector> {
public:
    std::unordered_set<std::string> params;

    // 功能：处理Visit CXXOperator Call Expr相关操作。
    bool VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr* opCall) {
        if (!opCall || opCall->getOperator() != clang::OverloadedOperatorKind::OO_Subscript ||
            opCall->getNumArgs() < 2) {
            return true;
        }

        if (!IsEmptyBraceIndexExpr(opCall->getArg(1))) {
            return true;
        }

        std::string baseName;
        if (TryExtractBaseIdentifier(opCall->getArg(0), baseName) && !baseName.empty()) {
            params.insert(baseName);
        }
        return true;
    }

    // 功能：处理Visit Array Subscript Expr相关操作。
    bool VisitArraySubscriptExpr(clang::ArraySubscriptExpr* expr) {
        if (!expr || !IsEmptyBraceIndexExpr(expr->getIdx())) {
            return true;
        }

        std::string baseName;
        if (TryExtractBaseIdentifier(expr->getBase(), baseName) && !baseName.empty()) {
            params.insert(baseName);
        }
        return true;
    }
};

// 功能：收集Collect Broadcast Scalar Param Names相关数据。
static std::unordered_set<std::string> CollectBroadcastScalarParamNames(dacppTranslator::Shell* shell) {
    std::unordered_set<std::string> result;
    if (!shell || !shell->getShellLoc() || !shell->getShellLoc()->hasBody()) {
        return result;
    }

    BroadcastScalarParamDetector detector;
    detector.TraverseStmt(const_cast<clang::Stmt*>(shell->getShellLoc()->getBody()));
    return detector.params;
}

// 功能：为Build Window Info From Split构建所需数据。
CalcShellWindowInfo BuildWindowInfoFromSplit(Split* split) {
    CalcShellWindowInfo info;
    if (!split) {
        info.isScalar = true;
        info.splitKind = "Split";
        return info;
    }

    info.dimIdx = split->getDimIdx();
    info.splitKind = split->type.empty() ? "Split" : split->type;

    if (auto* regular = dynamic_cast<RegularSplit*>(split)) {
        info.splitSize = regular->getSplitSize();
        info.splitStride = regular->getSplitStride();
        // RegularSplit(size, stride) on 1D window implies right overlap when size > stride.
        info.leftHalo = 0;
        info.rightHalo = std::max(0, info.splitSize - info.splitStride);
        info.isWindow = true;
        return info;
    }

    if (dynamic_cast<IndexSplit*>(split)) {
        info.splitSize = 1;
        info.splitStride = 1;
        info.leftHalo = 0;
        info.rightHalo = 0;
        info.isWindow = true;
        return info;
    }

    info.splitSize = 1;
    info.splitStride = 1;
    info.isScalar = true;
    return info;
}

// 功能：检查Has Window On Dim相关条件。
bool HasWindowOnDim(const CalcShellParamInfo& paramInfo, int dimIdx) {
    for (const CalcShellWindowInfo& window : paramInfo.windows) {
        if (window.isWindow && window.dimIdx == dimIdx) {
            return true;
        }
    }
    return false;
}

// 功能：处理Append Shape Derived Windows相关操作。
void AppendShapeDerivedWindows(CalcShellParamInfo& paramInfo) {
    for (size_t i = 0; i < paramInfo.shape.size(); ++i) {
        const int dimSize = paramInfo.shape[i];
        if (dimSize <= 0) {
            continue;
        }
        const int dimIdx = static_cast<int>(i);
        if (HasWindowOnDim(paramInfo, dimIdx)) {
            continue;
        }
        CalcShellWindowInfo shapeWindow;
        shapeWindow.dimIdx = dimIdx;
        shapeWindow.splitSize = dimSize;
        shapeWindow.splitStride = dimSize;
        shapeWindow.leftHalo = 0;
        shapeWindow.rightHalo = 0;
        shapeWindow.isScalar = false;
        shapeWindow.isWindow = true;
        shapeWindow.splitKind = "Shape";
        paramInfo.windows.push_back(shapeWindow);
    }
}

} // namespace

// 建立Calc Shell信息的相关总结。
CalcShellSummary BuildCalcShellSummary(dacppTranslator::Shell* shell) {
    CalcShellSummary summary;
    if (!shell) {
        return summary;
    }

    const int paramCount = shell->getNumShellParams();
    const std::unordered_set<std::string> broadcastNames = CollectBroadcastScalarParamNames(shell);
    summary.params.reserve(paramCount > 0 ? static_cast<size_t>(paramCount) : 0U);

    for (int i = 0; i < paramCount; ++i) {
        ShellParam* param = shell->getShellParam(i);
        if (!param) {
            continue;
        }

        CalcShellParamInfo paramInfo;
        paramInfo.name = param->getName();
        paramInfo.rw = static_cast<int>(param->getRw());
        paramInfo.rank = std::max(0, param->getDim());
        for (int dimIdx = 0; dimIdx < paramInfo.rank; ++dimIdx) {
            paramInfo.shape.push_back(param->getShape(dimIdx));
        }
        paramInfo.isBroadcastScalar = broadcastNames.count(paramInfo.name) > 0;

        const int splitCount = param->getNumSplit();
        if (splitCount <= 0) {
            AppendShapeDerivedWindows(paramInfo);
            if (paramInfo.windows.empty()) {
                CalcShellWindowInfo scalarInfo;
                scalarInfo.isScalar = true;
                scalarInfo.splitKind = "Split";
                paramInfo.windows.push_back(scalarInfo);
                paramInfo.isScalar = true;
            } else {
                paramInfo.isWindow = true;
                paramInfo.isScalar = false;
            }
            summary.params.push_back(paramInfo);
            continue;
        }

        bool hasWindow = false;
        bool hasPlainSplit = false;
        bool hasIndexedOrRegularSplit = false;
        for (int k = 0; k < splitCount; ++k) {
            Split* split = param->getSplit(k);
            CalcShellWindowInfo window = BuildWindowInfoFromSplit(split);
            if (window.isWindow) {
                hasWindow = true;
            }
            if (split) {
                const std::string splitType = split->type;
                if (splitType == "Split") {
                    hasPlainSplit = true;
                }
                if (splitType == "IndexSplit" || splitType == "RegularSplit") {
                    hasIndexedOrRegularSplit = true;
                }
            }
            paramInfo.windows.push_back(window);
        }

        AppendShapeDerivedWindows(paramInfo);

        hasWindow = false;
        for (const CalcShellWindowInfo& window : paramInfo.windows) {
            if (window.isWindow) {
                hasWindow = true;
                break;
            }
        }

        paramInfo.isWindow = hasWindow;
        paramInfo.isScalar = (paramInfo.rank <= 0) ? !hasWindow : false;
        if (!paramInfo.isBroadcastScalar && hasPlainSplit && !hasIndexedOrRegularSplit) {
            paramInfo.isBroadcastScalar = true;
        }
        summary.params.push_back(paramInfo);
    }

    summary.valid = !summary.params.empty();
    return summary;
}

// 输出shell信息
std::string DumpCalcShellSummary(const CalcShellSummary& summary) {
    std::ostringstream oss;
    if (!summary.valid) {
        oss << "[ascendc-mlir] shell-summary: <empty>\n";
        return oss.str();
    }

    oss << "[ascendc-mlir] shell-summary:\n";
    for (const CalcShellParamInfo& param : summary.params) {
        oss << "  - " << param.name << ": rw=" << param.rw
            << ", rank=" << param.rank;
        if (!param.shape.empty()) {
            oss << ", shape=[";
            for (size_t i = 0; i < param.shape.size(); ++i) {
                if (i > 0) {
                    oss << ",";
                }
                oss << param.shape[i];
            }
            oss << "]";
        }

        bool emittedWindow = false;
        for (const CalcShellWindowInfo& window : param.windows) {
            if (!window.isWindow) {
                continue;
            }
            oss << ", dimIdx=" << window.dimIdx
                << ", size=" << window.splitSize
                << ",stride=" << window.splitStride
                << ",leftHalo=" << window.leftHalo
                << ",rightHalo=" << window.rightHalo;
            emittedWindow = true;
        }

        if (!emittedWindow) {
            oss << (param.rank > 0 ? ", shape-only" : ", scalar");
        }
        if (param.isBroadcastScalar) {
            oss << ", broadcast-scalar";
        }
        oss << "\n";
    }
    return oss.str();
}

} // namespace ascendc_mlir
} // namespace dacppTranslator
