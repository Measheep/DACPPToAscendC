#include "ascendc_mlir/CalcMlirUtils.h"

#include <algorithm>
#include <cmath>
#include <sstream>

#if defined(DACPP_TRANSLATOR_HAS_MLIR)
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Value.h"
#endif

namespace dacppTranslator {
namespace ascendc_mlir {

#if defined(DACPP_TRANSLATOR_HAS_MLIR)

bool IsArgMemref(mlir::Value memref) {
    auto arg = memref.dyn_cast<mlir::BlockArgument>();
    if (!arg) {
        return false;
    }

    mlir::Block* owner = arg.getOwner();
    if (!owner) {
        return false;
    }

    auto func = llvm::dyn_cast<mlir::func::FuncOp>(owner->getParentOp());
    if (!func || &func.getBody().front() != owner) {
        return false;
    }

    return memref.getType().isa<mlir::MemRefType>();
}

std::string ValueToken(mlir::Value value) {
    if (auto arg = value.dyn_cast<mlir::BlockArgument>()) {
        return "%arg" + std::to_string(static_cast<unsigned>(arg.getArgNumber()));
    }
    return "%value";
}

std::string ResolveTensorNameFromValue(mlir::Value value,
                                       const std::vector<std::string>& paramOrder,
                                       const CalcShellSummary* shellSummary) {
    if (auto arg = value.dyn_cast<mlir::BlockArgument>()) {
        size_t idx = static_cast<size_t>(arg.getArgNumber());
        return ResolveArgIndexToTensorName(idx, paramOrder, shellSummary);
    }
    return ValueToken(value);
}

#endif

bool TryParseArgToken(const std::string& token, size_t& outIndex) {
    if (token.size() < 5 || token.compare(0, 4, "%arg") != 0) {
        return false;
    }
    std::string suffix = token.substr(4);
    if (suffix.empty()) {
        return false;
    }
    for (char c : suffix) {
        if (c < '0' || c > '9') {
            return false;
        }
    }
    outIndex = static_cast<size_t>(std::stoul(suffix));
    return true;
}

const CalcShellParamInfo* FindShellParam(const CalcShellSummary* shellSummary,
                                         const std::string& tensor) {
    if (!shellSummary) {
        return nullptr;
    }
    for (const CalcShellParamInfo& param : shellSummary->params) {
        if (param.name == tensor) {
            return &param;
        }
    }
    return nullptr;
}

int ExprPrecedence(CalcExprNodeKind kind) {
    switch (kind) {
    case CalcExprNodeKind::Add:
    case CalcExprNodeKind::Sub:
        return 1;
    case CalcExprNodeKind::Mul:
    case CalcExprNodeKind::Div:
        return 2;
    default:
        return 3;
    }
}

std::string FormatFloatConst(double value) {
    if (std::abs(value - std::round(value)) < 1e-12) {
        return std::to_string(static_cast<long long>(std::llround(value)));
    }
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

std::string ResolveArgIndexToTensorName(size_t argIdx,
                                        const std::vector<std::string>& paramOrder,
                                        const CalcShellSummary* shellSummary) {
    if (argIdx < paramOrder.size() && !paramOrder[argIdx].empty()) {
        return paramOrder[argIdx];
    }
    if (shellSummary && argIdx < shellSummary->params.size() &&
        !shellSummary->params[argIdx].name.empty()) {
        return shellSummary->params[argIdx].name;
    }
    return "%arg" + std::to_string(argIdx);
}

std::string ResolveTensorNameFromToken(const std::string& token,
                                       const std::vector<std::string>& paramOrder,
                                       const CalcShellSummary* shellSummary) {
    size_t idx = 0;
    if (TryParseArgToken(token, idx)) {
        return ResolveArgIndexToTensorName(idx, paramOrder, shellSummary);
    }
    return token;
}

void InferHaloAndLengths(const CalcShellSummary* shellSummary,
                         const std::string& tensor,
                         const std::set<int>& offsets,
                         int& outLeftHalo,
                         int& outRightHalo,
                         int& outValidLength,
                         int& outCopyLength) {
    outLeftHalo = 0;
    outRightHalo = 0;
    outValidLength = 0;
    outCopyLength = 0;

    for (int offset : offsets) {
        if (offset < 0) {
            outLeftHalo = std::max(outLeftHalo, -offset);
        } else {
            outRightHalo = std::max(outRightHalo, offset);
        }
    }

    const CalcShellParamInfo* param = FindShellParam(shellSummary, tensor);
    if (!param) {
        return;
    }

    for (const CalcShellWindowInfo& window : param->windows) {
        if (!window.isWindow) {
            continue;
        }
        outLeftHalo = std::max(outLeftHalo, window.leftHalo);
        outRightHalo = std::max(outRightHalo, window.rightHalo);
        if (outValidLength <= 0 && window.splitStride > 0) {
            outValidLength = window.splitStride;
        }
        if (outCopyLength <= 0 && window.splitSize > 0) {
            outCopyLength = window.splitSize;
        }
    }

    if (outValidLength > 0 && outCopyLength <= 0) {
        outCopyLength = outValidLength + outLeftHalo + outRightHalo;
    }
}

CalcBoundaryPolicy BuildCalcBoundaryPolicy(const std::string& tensor,
                                           const CalcShellSummary* shellSummary) {
    const CalcShellParamInfo* param = FindShellParam(shellSummary, tensor);
    if (!param) {
        return CalcBoundaryPolicy::Clamp;
    }
    if (param->isScalar) {
        return CalcBoundaryPolicy::Clamp;
    }

    bool hasWindow = false;
    bool hasHalo = false;
    for (const CalcShellWindowInfo& window : param->windows) {
        if (!window.isWindow) {
            continue;
        }
        hasWindow = true;
        if (window.leftHalo > 0 || window.rightHalo > 0 ||
            window.splitSize > window.splitStride) {
            hasHalo = true;
            break;
        }
    }

    if (hasHalo) {
        return CalcBoundaryPolicy::RequireHalo;
    }
    if (hasWindow) {
        return CalcBoundaryPolicy::Clamp;
    }
    return CalcBoundaryPolicy::Replicate;
}

bool ContainsTensorLoad(const CalcExprGraph& graph,
                        int nodeId,
                        const std::string& tensor) {
    if (!graph.ValidNodeId(nodeId)) {
        return false;
    }

    std::unordered_set<int> visiting;
    std::vector<int> stack;
    stack.push_back(nodeId);

    while (!stack.empty()) {
        int current = stack.back();
        stack.pop_back();
        if (!graph.ValidNodeId(current)) {
            continue;
        }
        if (visiting.find(current) != visiting.end()) {
            continue;
        }
        visiting.insert(current);

        const CalcExprNode& node = graph.nodes[static_cast<size_t>(current)];
        if (node.kind == CalcExprNodeKind::Load && node.isArgTensor) {
            const std::string tensorKey =
                node.canonicalTensor.empty() ? node.token : node.canonicalTensor;
            if (tensorKey == tensor) {
                return true;
            }
        }
        for (int inId : node.inputs) {
            stack.push_back(inId);
        }
    }

    return false;
}

bool TryGetShape2DFromShell(const CalcShellSummary* shellSummary,
                            const std::string& tensor,
                            int& outDim0,
                            int& outDim1,
                            std::string* outDim0Source,
                            std::string* outDim1Source) {
    outDim0 = 0;
    outDim1 = 0;
    if (outDim0Source) {
        outDim0Source->clear();
    }
    if (outDim1Source) {
        outDim1Source->clear();
    }

    const CalcShellParamInfo* param = FindShellParam(shellSummary, tensor);
    if (!param) {
        return false;
    }

    if (param->shape.size() >= 2) {
        outDim0 = param->shape[0];
        outDim1 = param->shape[1];
        if (outDim0Source) {
            *outDim0Source = "shell:" + tensor + ".shape[0]";
        }
        if (outDim1Source) {
            *outDim1Source = "shell:" + tensor + ".shape[1]";
        }
        if (outDim0 > 0 && outDim1 > 0) {
            return true;
        }
    }

    std::vector<CalcShellWindowInfo> windows;
    for (const CalcShellWindowInfo& window : param->windows) {
        if (window.dimIdx >= 0 && window.splitSize > 0) {
            windows.push_back(window);
        }
    }
    if (windows.size() < 2) {
        return false;
    }

    std::sort(windows.begin(), windows.end(),
              [](const CalcShellWindowInfo& a, const CalcShellWindowInfo& b) {
                  return a.dimIdx < b.dimIdx;
              });

    outDim0 = windows[0].splitSize;
    outDim1 = windows[1].splitSize;
    if (outDim0Source) {
        *outDim0Source = "shell:" + tensor + ".dim" +
                         std::to_string(windows[0].dimIdx) + ".splitSize";
    }
    if (outDim1Source) {
        *outDim1Source = "shell:" + tensor + ".dim" +
                         std::to_string(windows[1].dimIdx) + ".splitSize";
    }
    return outDim0 > 0 && outDim1 > 0;
}

std::string RenderExprCommon(const CalcExprGraph& graph,
                             int nodeId,
                             ExprNodeRenderer renderer,
                             bool& ok,
                             std::unordered_set<int>& visiting,
                             int parentPrec) {
    if (!ok) {
        return "?";
    }
    if (!graph.ValidNodeId(nodeId)) {
        ok = false;
        return "?";
    }
    if (visiting.find(nodeId) != visiting.end()) {
        ok = false;
        return "<cycle>";
    }

    visiting.insert(nodeId);
    const CalcExprNode& node = graph.nodes[static_cast<size_t>(nodeId)];

    if (renderer) {
        std::optional<std::string> override = renderer(nodeId, node, ok);
        if (override.has_value()) {
            visiting.erase(nodeId);
            return override.value();
        }
    }

    const int prec = ExprPrecedence(node.kind);

    auto RenderChild = [&](size_t idx, int childParentPrec) -> std::string {
        if (idx >= node.inputs.size()) {
            ok = false;
            return "?";
        }
        return RenderExprCommon(graph, node.inputs[idx], renderer, ok,
                                visiting, childParentPrec);
    };

    std::string out;
    switch (node.kind) {
    case CalcExprNodeKind::Load:
        out = node.token.empty() ? "load" : node.token;
        break;
    case CalcExprNodeKind::Const:
        out = node.hasConstValue
                  ? FormatFloatConst(node.constValue)
                  : (node.token.empty() ? "const" : node.token);
        break;
    case CalcExprNodeKind::Add: {
        std::string lhs = RenderChild(0, prec);
        std::string rhs = RenderChild(1, prec);
        out = lhs + "+" + rhs;
        break;
    }
    case CalcExprNodeKind::Sub: {
        std::string lhs = RenderChild(0, prec);
        std::string rhs = RenderChild(1, prec + 1);
        out = lhs + "-" + rhs;
        break;
    }
    case CalcExprNodeKind::Mul: {
        std::string lhs = RenderChild(0, prec);
        std::string rhs = RenderChild(1, prec);
        if (lhs == "1") {
            out = rhs;
        } else if (rhs == "1") {
            out = lhs;
        } else {
            out = lhs + "*" + rhs;
        }
        break;
    }
    case CalcExprNodeKind::Div: {
        std::string lhs = RenderChild(0, prec);
        std::string rhs = RenderChild(1, prec + 1);
        out = lhs + "/" + rhs;
        break;
    }
    case CalcExprNodeKind::Exp:
        out = "exp(" + RenderChild(0, 0) + ")";
        break;
    case CalcExprNodeKind::Min:
        out = "min(" + RenderChild(0, 0) + "," + RenderChild(1, 0) + ")";
        break;
    case CalcExprNodeKind::Max:
        out = "max(" + RenderChild(0, 0) + "," + RenderChild(1, 0) + ")";
        break;
    case CalcExprNodeKind::Clamp:
        out = "clamp(" + RenderChild(0, 0) + ")";
        break;
    case CalcExprNodeKind::Store:
    case CalcExprNodeKind::Unknown:
        ok = false;
        out = "?";
        break;
    }

    visiting.erase(nodeId);
    if ((node.kind == CalcExprNodeKind::Add || node.kind == CalcExprNodeKind::Sub ||
         node.kind == CalcExprNodeKind::Mul || node.kind == CalcExprNodeKind::Div) &&
        prec < parentPrec) {
        return "(" + out + ")";
    }
    return out;
}

} // namespace ascendc_mlir
} // namespace dacppTranslator
