/*
模式识别器 and 语义分析引擎。
CalcExprGraph.cpp已经把底层的 MLIR 指令抽象成了一张纯数学的“表达式计算图”。
但 Ascend NPU是向量化处理器，它不能像 CPU 那样随意执行标量指令，它需要按特定的计算模板来下发指令。
代码的作用就是扫描计算图，识别出这段代码到底属于哪种“高级计算语义”，从而决定后续生成哪种 AscendC 硬件加速代码。

*/
#include "ascendc_mlir/CalcAnalysis.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ascendc_mlir/CalcExprGraph.h"
#include "ascendc_mlir/CalcMlirUtils.h"
#include "ascendc_mlir/CalcShellInfo.h"

#if defined(DACPP_TRANSLATOR_HAS_MLIR)
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#endif

namespace dacppTranslator {
namespace ascendc_mlir {
namespace {

struct ExprStats {
    bool hasExp = false;
    bool hasMul = false;
    bool hasAddSub = false;
    bool hasDiv = false;
    bool hasMin = false;
    bool hasMax = false;
    bool hasClamp = false;
    bool hasUnknown = false;

    bool hasUnknownOffset = false;
    bool hasNonZeroOffset = false;
    bool allOffsetsZero = true;
    bool hasSelfRead = false;
    size_t windowBaseLoadCount = 0;

    size_t argLoadCount = 0;
    std::unordered_map<std::string, std::set<int>> offsetsByTensor;
    std::vector<std::string> loadTensorOrder;
};

// 遍历计算图，填充 ExprStats 结构体
void CollectStatsDfs(const CalcExprGraph& graph,
                     int nodeId,
                     const std::string& storeToken,
                     std::unordered_set<int>& visited,
                     ExprStats& out) {
    if (nodeId < 0 || static_cast<size_t>(nodeId) >= graph.nodes.size()) {
        return;
    }
    if (visited.find(nodeId) != visited.end()) {
        return;
    }
    visited.insert(nodeId);

    const CalcExprNode& node = graph.nodes[static_cast<size_t>(nodeId)];
    switch (node.kind) {
    case CalcExprNodeKind::Load:
        if (node.isArgTensor || !node.canonicalTensor.empty()) {
            const std::string tensorKey =
                node.canonicalTensor.empty() ? node.token : node.canonicalTensor;
            ++out.argLoadCount;
            if (node.isWindowBase) {
                ++out.windowBaseLoadCount;
            }
            if (out.offsetsByTensor.find(tensorKey) == out.offsetsByTensor.end()) {
                out.loadTensorOrder.push_back(tensorKey);
            }
            if (node.offsetKnown) {
                out.offsetsByTensor[tensorKey].insert(node.offset);
                if (node.offset != 0) {
                    out.hasNonZeroOffset = true;
                    out.allOffsetsZero = false;
                }
            } else {
                out.hasUnknownOffset = true;
                out.allOffsetsZero = false;
            }
            if (!storeToken.empty() && tensorKey == storeToken) {
                out.hasSelfRead = true;
            }
        }
        break;
    case CalcExprNodeKind::Exp:
        out.hasExp = true;
        break;
    case CalcExprNodeKind::Mul:
        out.hasMul = true;
        break;
    case CalcExprNodeKind::Add:
    case CalcExprNodeKind::Sub:
        out.hasAddSub = true;
        break;
    case CalcExprNodeKind::Div:
        out.hasDiv = true;
        break;
    case CalcExprNodeKind::Min:
        out.hasMin = true;
        break;
    case CalcExprNodeKind::Max:
        out.hasMax = true;
        break;
    case CalcExprNodeKind::Clamp:
        out.hasClamp = true;
        break;
    case CalcExprNodeKind::Unknown:
        out.hasUnknown = true;
        break;
    case CalcExprNodeKind::Const:
    case CalcExprNodeKind::Store:
        break;
    }

    for (int inId : node.inputs) {
        CollectStatsDfs(graph, inId, storeToken, visited, out);
    }
}

// 功能：为Build Affine Inputs构建所需数据。
std::vector<AffineAccess> BuildAffineInputs(const std::string& tensor,
                                            const std::set<int>& offsets) {
    std::vector<AffineAccess> out;
    for (int offset : offsets) {
        AffineAccess access;
        access.tensor = tensor;
        access.offset = offset;
        access.coeff = 1.0;
        out.push_back(access);
    }
    return out;
}

// 功能：处理Pick Primary Tensor相关逻辑。
std::string PickPrimaryTensor(const ExprStats& stats,
                              std::set<int>& outOffsets) {
    std::string primary;
    size_t bestCount = 0;
    for (const auto& entry : stats.offsetsByTensor) {
        if (entry.second.size() > bestCount) {
            bestCount = entry.second.size();
            primary = entry.first;
            outOffsets = entry.second;
        }
    }

    if (!primary.empty()) {
        return primary;
    }

    for (const std::string& token : stats.loadTensorOrder) {
        auto it = stats.offsetsByTensor.find(token);
        if (it != stats.offsetsByTensor.end()) {
            outOffsets = it->second;
            return token;
        }
    }

    return "";
}

// 功能：为Build Diag Suffix构建所需数据。
std::string BuildDiagSuffix(const ExprStats& stats) {
    std::ostringstream oss;
    oss << "diag(exp=" << (stats.hasExp ? 1 : 0)
        << ",mul=" << (stats.hasMul ? 1 : 0)
        << ",min=" << (stats.hasMin ? 1 : 0)
        << ",offsetNZ=" << (stats.hasNonZeroOffset ? 1 : 0)
        << ",offsetUnknown=" << (stats.hasUnknownOffset ? 1 : 0)
        << ",windowBaseLoads=" << stats.windowBaseLoadCount
        << ",argTensors=" << stats.offsetsByTensor.size()
        << ",argLoads=" << stats.argLoadCount << ")";
    return oss.str();
}

// 功能：为Build Elementwise Inputs构建所需数据。
std::vector<AffineAccess> BuildElementwiseInputs(const ExprStats& stats) {
    std::vector<AffineAccess> out;
    for (const std::string& token : stats.loadTensorOrder) {
        auto it = stats.offsetsByTensor.find(token);
        if (it == stats.offsetsByTensor.end()) {
            continue;
        }
        if (it->second.find(0) == it->second.end()) {
            continue;
        }

        AffineAccess access;
        access.tensor = token;
        access.offset = 0;
        access.coeff = 1.0;
        out.push_back(access);
    }
    return out;
}

// 功能：处理Count Additive Terms For Tensor相关逻辑。
size_t CountAdditiveTermsForTensor(const CalcExprGraph& graph,
                                   int nodeId,
                                   const std::string& tensor) {
    if (nodeId < 0 || static_cast<size_t>(nodeId) >= graph.nodes.size()) {
        return 0;
    }

    const CalcExprNode& node = graph.nodes[static_cast<size_t>(nodeId)];
    if ((node.kind == CalcExprNodeKind::Add || node.kind == CalcExprNodeKind::Sub) && !node.inputs.empty()) {
        size_t total = 0;
        for (int inId : node.inputs) {
            total += CountAdditiveTermsForTensor(graph, inId, tensor);
        }
        return total;
    }

    return ContainsTensorLoad(graph, nodeId, tensor) ? 1 : 0;
}

// 功能：检查Can Lower Direct Pointwise Expr Dfs相关条件。
bool CanLowerDirectPointwiseExprDfs(const CalcExprGraph& graph,
                                    int nodeId,
                                    std::unordered_set<int>& visiting) {
    if (nodeId < 0 || static_cast<size_t>(nodeId) >= graph.nodes.size()) {
        return false;
    }
    if (visiting.find(nodeId) != visiting.end()) {
        return false;
    }
    visiting.insert(nodeId);

    const CalcExprNode& node = graph.nodes[static_cast<size_t>(nodeId)];
    switch (node.kind) {
    case CalcExprNodeKind::Load:
        if ((!node.isArgTensor && node.canonicalTensor.empty()) ||
            !node.offsetKnown || node.offset != 0) {
            visiting.erase(nodeId);
            return false;
        }
        visiting.erase(nodeId);
        return true;
    case CalcExprNodeKind::Const:
        visiting.erase(nodeId);
        return true;
    case CalcExprNodeKind::Exp:
    case CalcExprNodeKind::Clamp:
        if (node.inputs.size() != 1) {
            visiting.erase(nodeId);
            return false;
        }
        {
            const bool ok = CanLowerDirectPointwiseExprDfs(graph, node.inputs.front(), visiting);
            visiting.erase(nodeId);
            return ok;
        }
    case CalcExprNodeKind::Add:
    case CalcExprNodeKind::Sub:
    case CalcExprNodeKind::Mul:
    case CalcExprNodeKind::Div:
    case CalcExprNodeKind::Min:
    case CalcExprNodeKind::Max:
        if (node.inputs.size() < 2) {
            visiting.erase(nodeId);
            return false;
        }
        for (int inId : node.inputs) {
            if (!CanLowerDirectPointwiseExprDfs(graph, inId, visiting)) {
                visiting.erase(nodeId);
                return false;
            }
        }
        visiting.erase(nodeId);
        return true;
    case CalcExprNodeKind::Store:
    case CalcExprNodeKind::Unknown:
        visiting.erase(nodeId);
        return false;
    }

    visiting.erase(nodeId);
    return false;
}

// 功能：检查Can Lower Direct Pointwise Expr相关条件。
bool CanLowerDirectPointwiseExpr(const CalcExprGraph& graph,
                                 int nodeId) {
    std::unordered_set<int> visiting;
    return CanLowerDirectPointwiseExprDfs(graph, nodeId, visiting);
}

// 功能：收集Collect Tensor Offsets Dfs相关数据。
void CollectTensorOffsetsDfs(const CalcExprGraph& graph,
                             int nodeId,
                             const std::string& tensor,
                             std::unordered_set<int>& visiting,
                             std::set<int>& outOffsets) {
    if (nodeId < 0 || static_cast<size_t>(nodeId) >= graph.nodes.size()) {
        return;
    }
    if (visiting.find(nodeId) != visiting.end()) {
        return;
    }
    visiting.insert(nodeId);

    const CalcExprNode& node = graph.nodes[static_cast<size_t>(nodeId)];
    if (node.kind == CalcExprNodeKind::Load && (node.isArgTensor || !node.canonicalTensor.empty())) {
        const std::string tensorKey =
            node.canonicalTensor.empty() ? node.token : node.canonicalTensor;
        if (tensorKey == tensor && node.offsetKnown) {
            outOffsets.insert(node.offset);
        }
    }

    for (int inId : node.inputs) {
        CollectTensorOffsetsDfs(graph, inId, tensor, visiting, outOffsets);
    }
}

// 功能：收集Collect Tensor Offsets相关数据。
std::set<int> CollectTensorOffsets(const CalcExprGraph& graph,
                                   int nodeId,
                                   const std::string& tensor) {
    std::set<int> offsets;
    std::unordered_set<int> visiting;
    CollectTensorOffsetsDfs(graph, nodeId, tensor, visiting, offsets);
    return offsets;
}

// 功能：检查Is Commutative Node相关条件。
bool IsCommutativeNode(CalcExprNodeKind kind) {
    return kind == CalcExprNodeKind::Add ||
           kind == CalcExprNodeKind::Mul ||
           kind == CalcExprNodeKind::Min ||
           kind == CalcExprNodeKind::Max;
}

// 功能：为Build Canonical Sub Expr Signature构建所需数据。
std::string BuildCanonicalSubExprSignature(const CalcExprGraph& graph,
                                           int nodeId,
                                           const std::string& tensor,
                                           bool ignoreTensorOffsets,
                                           std::unordered_set<int>& visiting) {
    if (nodeId < 0 || static_cast<size_t>(nodeId) >= graph.nodes.size()) {
        return "?";
    }
    if (visiting.find(nodeId) != visiting.end()) {
        return "<cycle>";
    }
    visiting.insert(nodeId);

    const CalcExprNode& node = graph.nodes[static_cast<size_t>(nodeId)];
    std::string self;
    switch (node.kind) {
    case CalcExprNodeKind::Load: {
        const std::string tensorKey =
            node.canonicalTensor.empty() ? node.token : node.canonicalTensor;
        if (!tensor.empty() && tensorKey == tensor) {
            if (ignoreTensorOffsets) {
                self = "Load(" + tensor + ",*)";
            } else if (node.offsetKnown) {
                self = "Load(" + tensor + "," + std::to_string(node.offset) + ")";
            } else {
                self = "Load(" + tensor + ",?)";
            }
        } else {
            self = "Load(" + tensorKey + ")";
        }
        break;
    }
    case CalcExprNodeKind::Const:
        if (node.hasConstValue) {
            std::ostringstream oss;
            oss << node.constValue;
            self = "Const(" + oss.str() + ")";
        } else {
            self = "Const(" + node.token + ")";
        }
        break;
    case CalcExprNodeKind::Add:
        self = "Add";
        break;
    case CalcExprNodeKind::Sub:
        self = "Sub";
        break;
    case CalcExprNodeKind::Mul:
        self = "Mul";
        break;
    case CalcExprNodeKind::Div:
        self = "Div";
        break;
    case CalcExprNodeKind::Exp:
        self = "Exp";
        break;
    case CalcExprNodeKind::Min:
        self = "Min";
        break;
    case CalcExprNodeKind::Max:
        self = "Max";
        break;
    case CalcExprNodeKind::Clamp:
        self = "Clamp";
        break;
    case CalcExprNodeKind::Store:
        self = "Store(" + node.token + ")";
        break;
    case CalcExprNodeKind::Unknown:
        self = "Unknown(" + node.token + ")";
        break;
    }

    if (node.inputs.empty()) {
        visiting.erase(nodeId);
        return self;
    }

    std::vector<std::string> parts;
    for (int inId : node.inputs) {
        parts.push_back(BuildCanonicalSubExprSignature(
            graph, inId, tensor, ignoreTensorOffsets, visiting));
    }
    if (IsCommutativeNode(node.kind)) {
        std::sort(parts.begin(), parts.end());
    }

    std::string out = self + "(";
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i != 0) {
            out += ",";
        }
        out += parts[i];
    }
    out += ")";

    visiting.erase(nodeId);
    return out;
}

// 功能：为Build Canonical Sub Expr Signature构建所需数据。
std::string BuildCanonicalSubExprSignature(const CalcExprGraph& graph,
                                           int nodeId,
                                           const std::string& tensor,
                                           bool ignoreTensorOffsets) {
    std::unordered_set<int> visiting;
    return BuildCanonicalSubExprSignature(graph, nodeId, tensor, ignoreTensorOffsets, visiting);
}

// 功能：尝试处理Try Find Windowed Mirror Sub Expr相关逻辑。
bool TryFindWindowedMirrorSubExpr(const CalcExprGraph& graph,
                                  int rootNodeId,
                                  const std::string& tensor,
                                  int& outShiftedNodeId,
                                  int& outCenterNodeId,
                                  int& outDiffNodeId) {
    outShiftedNodeId = -1;
    outCenterNodeId = -1;
    outDiffNodeId = -1;

    if (rootNodeId < 0 || static_cast<size_t>(rootNodeId) >= graph.nodes.size()) {
        return false;
    }

    std::vector<int> stack;
    std::unordered_set<int> visited;
    stack.push_back(rootNodeId);

    while (!stack.empty()) {
        int nodeId = stack.back();
        stack.pop_back();

        if (nodeId < 0 || static_cast<size_t>(nodeId) >= graph.nodes.size()) {
            continue;
        }
        if (visited.find(nodeId) != visited.end()) {
            continue;
        }
        visited.insert(nodeId);

        const CalcExprNode& node = graph.nodes[static_cast<size_t>(nodeId)];
        if (node.kind == CalcExprNodeKind::Sub && node.inputs.size() >= 2) {
            const int lhsId = node.inputs[0];
            const int rhsId = node.inputs[1];
            std::set<int> lhsOffsets = CollectTensorOffsets(graph, lhsId, tensor);
            std::set<int> rhsOffsets = CollectTensorOffsets(graph, rhsId, tensor);
            if (lhsOffsets.size() == 1 && rhsOffsets.size() == 1) {
                const int lhsOffset = *lhsOffsets.begin();
                const int rhsOffset = *rhsOffsets.begin();
                if (lhsOffset != rhsOffset) {
                    std::string lhsSig =
                        BuildCanonicalSubExprSignature(graph, lhsId, tensor, true);
                    std::string rhsSig =
                        BuildCanonicalSubExprSignature(graph, rhsId, tensor, true);
                    if (!lhsSig.empty() && lhsSig == rhsSig) {
                        outDiffNodeId = nodeId;
                        if (lhsOffset != 0 && rhsOffset == 0) {
                            outShiftedNodeId = lhsId;
                            outCenterNodeId = rhsId;
                        } else if (rhsOffset != 0 && lhsOffset == 0) {
                            outShiftedNodeId = rhsId;
                            outCenterNodeId = lhsId;
                        } else {
                            outShiftedNodeId = lhsId;
                            outCenterNodeId = rhsId;
                        }
                        return true;
                    }
                }
            }
        }

        for (int inId : node.inputs) {
            stack.push_back(inId);
        }
    }

    return false;
}

// 功能：检查Is Scalar Param相关条件。
bool IsScalarParam(const CalcShellSummary* shellSummary,
                   const std::string& tensor) {
    const CalcShellParamInfo* param = FindShellParam(shellSummary, tensor);
    return param && (param->isScalar || param->isBroadcastScalar);
}

#if defined(DACPP_TRANSLATOR_HAS_MLIR)

static const std::vector<std::string> kEmptyParamOrder;

std::string ResolveTensorNameForAnalysis(mlir::Value value,
                                         const CalcAnalysisOptions& options) {
    return ResolveTensorNameFromValue(value,
        options.paramOrder ? *options.paramOrder : kEmptyParamOrder,
        options.shellSummary);
}

// 功能：检查Is Input Tensor Param相关条件。
bool IsInputTensorParam(const CalcAnalysisOptions& options,
                        const std::string& tensor) {
    if (!options.shellSummary) {
        return true;
    }
    const CalcShellParamInfo* param = FindShellParam(options.shellSummary, tensor);
    if (!param || param->isScalar) {
        return false;
    }
    return param->rw == 0 || param->rw == 2;
}

// 功能：检查Is Output Tensor Param相关条件。
bool IsOutputTensorParam(const CalcAnalysisOptions& options,
                         const std::string& tensor) {
    if (!options.shellSummary) {
        return true;
    }
    const CalcShellParamInfo* param = FindShellParam(options.shellSummary, tensor);
    if (!param || param->isScalar) {
        return false;
    }
    return param->rw == 1 || param->rw == 2;
}

bool TryGetShape2DForAnalysis(const CalcAnalysisOptions& options,
                              const std::string& tensor,
                              int& outD0, int& outD1) {
    return TryGetShape2DFromShell(options.shellSummary, tensor, outD0, outD1);
}

// 功能：尝试处理Try Recognize Cube Matmul Expr Graph Pattern相关逻辑。
bool TryRecognizeCubeMatmulExprGraphPattern(const CalcExprGraph& graph,
                                            const CalcAnalysisOptions& options,
                                            const CalcExprNode& storeNode,
                                            CubeMatmulPayload& outPayload,
                                            std::string& outReason) {
    outPayload = CubeMatmulPayload{};
    outReason.clear();

    if (storeNode.inputs.empty()) {
        outReason = "store node has no rhs";
        return false;
    }

    const int rhsId = storeNode.inputs.front();
    if (rhsId < 0 || static_cast<size_t>(rhsId) >= graph.nodes.size()) {
        outReason = "store rhs is invalid";
        return false;
    }

    const CalcExprNode& rhsNode = graph.nodes[static_cast<size_t>(rhsId)];
    if (rhsNode.kind != CalcExprNodeKind::Add || rhsNode.inputs.size() != 2) {
        outReason = "rhs is not add(dst, mul)";
        return false;
    }

    auto ResolveNode = [&](int nodeId) -> const CalcExprNode* {
        if (nodeId < 0 || static_cast<size_t>(nodeId) >= graph.nodes.size()) {
            return nullptr;
        }
        return &graph.nodes[static_cast<size_t>(nodeId)];
    };

    const CalcExprNode* dstLoadNode = nullptr;
    const CalcExprNode* mulNode = nullptr;
    for (int inputId : rhsNode.inputs) {
        const CalcExprNode* node = ResolveNode(inputId);
        if (!node) {
            continue;
        }
        const std::string canonical = !node->canonicalTensor.empty() ? node->canonicalTensor : node->token;
        if (node->kind == CalcExprNodeKind::Load && canonical == storeNode.token &&
            node->offsetKnown && node->offset == 0) {
            dstLoadNode = node;
            continue;
        }
        if (node->kind == CalcExprNodeKind::Mul && node->inputs.size() == 2) {
            mulNode = node;
        }
    }

    if (!dstLoadNode || !mulNode) {
        outReason = "add tree is missing dst self-load or mul branch";
        return false;
    }

    const CalcExprNode* lhsNode = ResolveNode(mulNode->inputs[0]);
    const CalcExprNode* rhsMulNode = ResolveNode(mulNode->inputs[1]);
    if (!lhsNode || !rhsMulNode ||
        lhsNode->kind != CalcExprNodeKind::Load ||
        rhsMulNode->kind != CalcExprNodeKind::Load) {
        outReason = "mul branch is not load(A) * load(B)";
        return false;
    }

    const std::string srcA = !lhsNode->canonicalTensor.empty() ? lhsNode->canonicalTensor : lhsNode->token;
    const std::string srcB = !rhsMulNode->canonicalTensor.empty() ? rhsMulNode->canonicalTensor : rhsMulNode->token;
    if (srcA.empty() || srcB.empty() || srcA == srcB || srcA == storeNode.token || srcB == storeNode.token) {
        outReason = "mul branch tensors are not two distinct inputs";
        return false;
    }
    if (!IsInputTensorParam(options, srcA) || !IsInputTensorParam(options, srcB) ||
        !IsOutputTensorParam(options, storeNode.token)) {
        outReason = "expr-graph tensors do not map to input/output matrix params";
        return false;
    }

    int a0 = 0, a1 = 0, b0 = 0, b1 = 0, c0 = 0, c1 = 0;
    const bool hasA = TryGetShape2DForAnalysis(options, srcA, a0, a1);
    const bool hasB = TryGetShape2DForAnalysis(options, srcB, b0, b1);
    const bool hasC = TryGetShape2DForAnalysis(options, storeNode.token, c0, c1);
    if (hasA && hasB && hasC && !(a1 == b0 && c0 == a0 && c1 == b1)) {
        outReason = "expr-graph shape check failed for A(M,K), B(K,N), C(M,N)";
        return false;
    }

    outPayload.srcA = srcA;
    outPayload.srcB = srcB;
    outPayload.dst = storeNode.token;
    outPayload.accumulateToDst = true;
    outPayload.hasInitialDst = true;
    outPayload.logicalInputType = "float";
    outPayload.kernelInputType = "half";
    outPayload.kernelOutputType = "float";
    outReason = "matched cube matmul via expr-graph pattern: dst += A[i] * B[i]";
    return true;
}

// 功能：检查Is Zero Index Like相关条件。
bool IsZeroIndexLike(mlir::Value index) {
    if (!index) {
        return false;
    }
    if (auto constIdx = index.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
        return constIdx.value() == 0;
    }
    if (auto constOp = index.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
            return intAttr.getInt() == 0;
        }
    }
    if (auto castOp = index.getDefiningOp<mlir::arith::IndexCastOp>()) {
        return IsZeroIndexLike(castOp.getIn());
    }
    return false;
}

// 功能：检查Is Loop Iv Like相关条件。
bool IsLoopIvLike(mlir::Value index,
                  mlir::Value iv) {
    if (!index || !iv) {
        return false;
    }
    if (index == iv) {
        return true;
    }
    if (auto castOp = index.getDefiningOp<mlir::arith::IndexCastOp>()) {
        return IsLoopIvLike(castOp.getIn(), iv);
    }
    return false;
}

// 功能：生成或转换Normalize Mlir Type相关结果。
std::string NormalizeMlirType(mlir::Type type) {
    if (!type) {
        return "unknown";
    }
    if (type.isF16()) return "float16";
    if (type.isF32()) return "float";
    if (type.isF64()) return "double";
    if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(type)) {
        switch (intTy.getWidth()) {
        case 8: return intTy.isUnsigned() ? "uint8_t" : "int8_t";
        case 16: return intTy.isUnsigned() ? "uint16_t" : "int16_t";
        case 32: return intTy.isUnsigned() ? "uint32_t" : "int32_t";
        case 64: return intTy.isUnsigned() ? "uint64_t" : "int64_t";
        default: break;
        }
    }
    return "unknown";
}

// 功能：处理Match Cube Matmul Loop Pattern相关逻辑。
bool MatchCubeMatmulLoopPattern(mlir::scf::ForOp loopOp,
                                const CalcAnalysisOptions& options,
                                CubeMatmulPayload& outPayload,
                                std::string& outReason) {
    outPayload = CubeMatmulPayload{};
    outReason.clear();

    mlir::Block* body = loopOp.getBody();
    if (!body) {
        outReason = "for-loop has no body";
        return false;
    }

    for (mlir::Operation& op : body->getOperations()) {
        if (auto nestedFor = llvm::dyn_cast<mlir::scf::ForOp>(op)) {
            if (nestedFor != loopOp) {
                outReason = "nested reduction loop detected";
                return false;
            }
        }
    }

    for (mlir::Operation& op : body->getOperations()) {
        auto storeOp = llvm::dyn_cast<mlir::memref::StoreOp>(op);
        if (!storeOp) {
            continue;
        }
        if (storeOp.getIndices().size() != 1 || !IsZeroIndexLike(storeOp.getIndices().front())) {
            continue;
        }

        mlir::Value dstMemref = storeOp.getMemRef();
        std::string dstName = ResolveTensorNameForAnalysis(dstMemref, options);
        if (!IsOutputTensorParam(options, dstName)) {
            continue;
        }

        mlir::Value storeValue = storeOp.getValueToStore();
        mlir::Value dstOldValue;
        mlir::Value mulValue;

        if (storeValue.getDefiningOp<mlir::arith::MulFOp>() ||
            storeValue.getDefiningOp<mlir::arith::MulIOp>()) {
            outReason = "detected pure matmul store (C=A*B); dot-product accumulation requires dst += A[i]*B[i]";
            continue;
        }

        if (auto addf = storeValue.getDefiningOp<mlir::arith::AddFOp>()) {
            mlir::Value lhs = addf.getLhs();
            mlir::Value rhs = addf.getRhs();
            auto lhsLoad = lhs.getDefiningOp<mlir::memref::LoadOp>();
            auto rhsLoad = rhs.getDefiningOp<mlir::memref::LoadOp>();
            if (lhsLoad && lhsLoad.getMemRef() == dstMemref) {
                dstOldValue = lhs;
                mulValue = rhs;
            } else if (rhsLoad && rhsLoad.getMemRef() == dstMemref) {
                dstOldValue = rhs;
                mulValue = lhs;
            } else {
                continue;
            }
        } else if (auto addi = storeValue.getDefiningOp<mlir::arith::AddIOp>()) {
            mlir::Value lhs = addi.getLhs();
            mlir::Value rhs = addi.getRhs();
            auto lhsLoad = lhs.getDefiningOp<mlir::memref::LoadOp>();
            auto rhsLoad = rhs.getDefiningOp<mlir::memref::LoadOp>();
            if (lhsLoad && lhsLoad.getMemRef() == dstMemref) {
                dstOldValue = lhs;
                mulValue = rhs;
            } else if (rhsLoad && rhsLoad.getMemRef() == dstMemref) {
                dstOldValue = rhs;
                mulValue = lhs;
            } else {
                continue;
            }
        } else {
            continue;
        }

        auto dstLoad = dstOldValue.getDefiningOp<mlir::memref::LoadOp>();
        if (!dstLoad || dstLoad.getIndices().size() != 1 ||
            !IsZeroIndexLike(dstLoad.getIndices().front())) {
            continue;
        }

        mlir::Value lhsMul;
        mlir::Value rhsMul;
        if (auto mulf = mulValue.getDefiningOp<mlir::arith::MulFOp>()) {
            lhsMul = mulf.getLhs();
            rhsMul = mulf.getRhs();
        } else if (auto muli = mulValue.getDefiningOp<mlir::arith::MulIOp>()) {
            lhsMul = muli.getLhs();
            rhsMul = muli.getRhs();
        } else {
            continue;
        }

        auto loadA = lhsMul.getDefiningOp<mlir::memref::LoadOp>();
        auto loadB = rhsMul.getDefiningOp<mlir::memref::LoadOp>();
        if (!loadA || !loadB) {
            continue;
        }
        if (loadA.getMemRef() == dstMemref || loadB.getMemRef() == dstMemref ||
            loadA.getMemRef() == loadB.getMemRef()) {
            continue;
        }
        if (loadA.getIndices().size() != 1 || loadB.getIndices().size() != 1) {
            continue;
        }

        if (!IsLoopIvLike(loadA.getIndices().front(), loopOp.getInductionVar()) ||
            !IsLoopIvLike(loadB.getIndices().front(), loopOp.getInductionVar())) {
            continue;
        }

        std::string srcA = ResolveTensorNameForAnalysis(loadA.getMemRef(), options);
        std::string srcB = ResolveTensorNameForAnalysis(loadB.getMemRef(), options);
        if (!IsInputTensorParam(options, srcA) || !IsInputTensorParam(options, srcB)) {
            continue;
        }

        int a0 = 0, a1 = 0, b0 = 0, b1 = 0, c0 = 0, c1 = 0;
        const bool hasA = TryGetShape2DForAnalysis(options, srcA, a0, a1);
        const bool hasB = TryGetShape2DForAnalysis(options, srcB, b0, b1);
        const bool hasC = TryGetShape2DForAnalysis(options, dstName, c0, c1);
        if (hasA && hasB && hasC) {
            if (!(a1 == b0 && c0 == a0 && c1 == b1)) {
                outReason = "shape check failed for A(M,K), B(K,N), C(M,N)";
                continue;
            }
        }

        outPayload.srcA = srcA;
        outPayload.srcB = srcB;
        outPayload.dst = dstName;
        // 单独处理 dotProduct[0] += ...：该路径必须保留累加语义。
        outPayload.accumulateToDst = true;
        outPayload.hasInitialDst = true;

        mlir::Type elemType = loadA.getType();
        outPayload.logicalInputType = NormalizeMlirType(elemType);
        outPayload.kernelInputType = (outPayload.logicalInputType == "float") ? "half" : outPayload.logicalInputType;
        outPayload.kernelOutputType = "float";

        outReason = "matched single-loop dot-product reduction pattern: dst += A[i] * B[i] (accumulateToDst=true)";
        return true;
    }

    outReason = "no loop body matched cube matmul reduction pattern";
    return false;
}

// 功能：尝试处理Try Recognize Cube Matmul Reduction相关逻辑。
bool TryRecognizeCubeMatmulReduction(const CalcMlirArtifact& artifact,
                                     const CalcAnalysisOptions& options,
                                     const std::string& storeTarget,
                                     CubeMatmulPayload& outPayload,
                                     std::string& outReason) {
    outPayload = CubeMatmulPayload{};
    outReason.clear();

    if (!artifact.valid || !artifact.module) {
        outReason = "mlir artifact is invalid";
        return false;
    }

    mlir::ModuleOp module = const_cast<mlir::ModuleOp&>(artifact.module);
    if (!module) {
        outReason = "mlir module is null";
        return false;
    }

    bool matchedTarget = false;
    std::string lastReason = "no scf.for reduction loop found";
    module.walk([&](mlir::scf::ForOp forOp) {
        if (matchedTarget) {
            return;
        }
        CubeMatmulPayload payload;
        std::string reason;
        if (!MatchCubeMatmulLoopPattern(forOp, options, payload, reason)) {
            lastReason = reason;
            return;
        }
        if (!storeTarget.empty() && payload.dst != storeTarget) {
            lastReason = "matched matmul loop but destination does not match current store target";
            return;
        }
        outPayload = payload;
        outReason = reason;
        matchedTarget = true;
    });

    if (!matchedTarget) {
        outReason = lastReason;
    }
    return matchedTarget;
}

// Try to extract a constant integer value from an MLIR Value (index or integer type).
// 功能：尝试处理Try Get Constant Int相关逻辑。
static bool TryGetConstantInt(mlir::Value v, int64_t& outVal) {
    if (!v) return false;
    if (auto constIdx = v.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
        outVal = constIdx.value();
        return true;
    }
    if (auto constOp = v.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
            outVal = intAttr.getInt();
            return true;
        }
    }
    if (auto castOp = v.getDefiningOp<mlir::arith::IndexCastOp>()) {
        return TryGetConstantInt(castOp.getIn(), outVal);
    }
    return false;
}

// Match the linear reduction pattern:
//   %acc = alloca
//   store bias[0] -> %acc          (bias init, may be outside or inside loop)
//   scf.for i = 0 to K step 1:
//     %old = load %acc[0]
//     %a   = load input[i]
//     %b   = load weight[i]
//     %mul = %a * %b
//     %add = %old + %mul
//     store %add -> %acc[0]
//   %result = load %acc[0]
//   store %result -> output[0]
// 功能：尝试处理Try Match Linear Reduction相关逻辑。
bool TryMatchLinearReduction(const CalcMlirArtifact& artifact,
                             const CalcAnalysisOptions& options,
                             CalcSemanticFinding& outFinding) {
    outFinding = CalcSemanticFinding{};

    if (!artifact.valid || !artifact.module) {
        return false;
    }

    mlir::ModuleOp module = const_cast<mlir::ModuleOp&>(artifact.module);
    if (!module) {
        return false;
    }

    // Step 1: find a local alloca that is used as accumulator.
    // It must have exactly one scf.for that writes acc += input[i]*weight[i],
    // and exactly one store to an output arg after the loop.
    bool found = false;

    module.walk([&](mlir::memref::AllocaOp allocaOp) {
        if (found) return;

        mlir::Value accMemref = allocaOp.getResult();

        // Collect all stores to this alloca.
        llvm::SmallVector<mlir::memref::StoreOp, 4> storesToAcc;
        for (mlir::Operation* user : accMemref.getUsers()) {
            if (auto storeOp = llvm::dyn_cast<mlir::memref::StoreOp>(user)) {
                if (storeOp.getMemRef() == accMemref) {
                    storesToAcc.push_back(storeOp);
                }
            }
        }
        if (storesToAcc.size() < 2) return;

        // Find the bias-init store (outside any scf.for) and the loop store.
        mlir::memref::StoreOp biasInitStore;
        mlir::scf::ForOp reductionLoop;
        std::string inputName, weightName;
        int K = 0;

        for (auto storeOp : storesToAcc) {
            // Check if this store is inside a scf.for.
            mlir::Operation* parent = storeOp->getParentOp();
            if (llvm::isa<mlir::scf::ForOp>(parent)) {
                // This is the loop-body accumulation store.
                auto forOp = llvm::cast<mlir::scf::ForOp>(parent);

                // Verify pattern: store (load acc[0]) + (load input[iv]) * (load weight[iv])
                mlir::Value storeVal = storeOp.getValueToStore();
                mlir::Value accOldVal, mulVal;

                auto addf = storeVal.getDefiningOp<mlir::arith::AddFOp>();
                if (!addf) return;

                mlir::Value lhs = addf.getLhs();
                mlir::Value rhs = addf.getRhs();
                auto lhsLoad = lhs.getDefiningOp<mlir::memref::LoadOp>();
                auto rhsLoad = rhs.getDefiningOp<mlir::memref::LoadOp>();

                if (lhsLoad && lhsLoad.getMemRef() == accMemref) {
                    accOldVal = lhs;
                    mulVal = rhs;
                } else if (rhsLoad && rhsLoad.getMemRef() == accMemref) {
                    accOldVal = rhs;
                    mulVal = lhs;
                } else {
                    return;
                }

                // Verify acc load is at index 0.
                auto accLoad = accOldVal.getDefiningOp<mlir::memref::LoadOp>();
                if (!accLoad || accLoad.getIndices().size() != 1 ||
                    !IsZeroIndexLike(accLoad.getIndices().front())) {
                    return;
                }

                // Verify mul = input[iv] * weight[iv].
                auto mulf = mulVal.getDefiningOp<mlir::arith::MulFOp>();
                if (!mulf) return;

                auto loadA = mulf.getLhs().getDefiningOp<mlir::memref::LoadOp>();
                auto loadB = mulf.getRhs().getDefiningOp<mlir::memref::LoadOp>();
                if (!loadA || !loadB) return;
                if (loadA.getMemRef() == accMemref || loadB.getMemRef() == accMemref) return;
                if (loadA.getIndices().size() != 1 || loadB.getIndices().size() != 1) return;

                mlir::Value iv = forOp.getInductionVar();
                if (!IsLoopIvLike(loadA.getIndices().front(), iv) ||
                    !IsLoopIvLike(loadB.getIndices().front(), iv)) {
                    return;
                }

                std::string nameA = ResolveTensorNameForAnalysis(loadA.getMemRef(), options);
                std::string nameB = ResolveTensorNameForAnalysis(loadB.getMemRef(), options);
                if (nameA.empty() || nameB.empty()) return;

                // Extract trip count K from for bounds.
                int64_t lb = 0, ub = 0, step = 1;
                if (!TryGetConstantInt(forOp.getLowerBound(), lb) ||
                    !TryGetConstantInt(forOp.getUpperBound(), ub) ||
                    !TryGetConstantInt(forOp.getStep(), step) ||
                    step != 1 || lb != 0 || ub <= 0) {
                    return;
                }

                reductionLoop = forOp;
                inputName = nameA;
                weightName = nameB;
                K = static_cast<int>(ub);
            } else {
                // Candidate bias-init store (outside loop).
                biasInitStore = storeOp;
            }
        }

        if (!reductionLoop || K <= 0) return;

        // Step 2: find the store to output arg that reads from acc.
        // output[0] = load acc[0]
        std::string outputName;
        bool foundOutputStore = false;
        for (mlir::Operation* user : accMemref.getUsers()) {
            auto loadOp = llvm::dyn_cast<mlir::memref::LoadOp>(user);
            if (!loadOp) continue;
            if (loadOp.getIndices().size() != 1 ||
                !IsZeroIndexLike(loadOp.getIndices().front())) continue;

            // This load's result should be stored to an output arg.
            for (mlir::Operation* loadUser : loadOp.getResult().getUsers()) {
                auto storeOp = llvm::dyn_cast<mlir::memref::StoreOp>(loadUser);
                if (!storeOp) continue;
                if (storeOp.getIndices().size() != 1 ||
                    !IsZeroIndexLike(storeOp.getIndices().front())) continue;

                std::string dstName = ResolveTensorNameForAnalysis(storeOp.getMemRef(), options);
                if (!IsOutputTensorParam(options, dstName)) continue;

                outputName = dstName;
                foundOutputStore = true;
                break;
            }
            if (foundOutputStore) break;
        }

        if (!foundOutputStore || outputName.empty()) return;

        // Step 3: identify bias tensor from the bias-init store.
        std::string biasName;
        bool hasBias = false;
        if (biasInitStore) {
            mlir::Value biasVal = biasInitStore.getValueToStore();
            auto biasLoad = biasVal.getDefiningOp<mlir::memref::LoadOp>();
            if (biasLoad) {
                std::string candidate = ResolveTensorNameForAnalysis(biasLoad.getMemRef(), options);
                if (!candidate.empty() && candidate != inputName && candidate != weightName) {
                    biasName = candidate;
                    hasBias = true;
                }
            }
        }

        // Build the finding.
        outFinding.category = CalcSemanticCategory::LinearReduction;
        outFinding.family = CalcSemanticFamily::LinearReductionFamily;
        outFinding.target = outputName;
        outFinding.hasLinearReductionPayload = true;
        outFinding.linearReductionPayload.inputTensor = inputName;
        outFinding.linearReductionPayload.weightTensor = weightName;
        outFinding.linearReductionPayload.biasTensor = biasName;
        outFinding.linearReductionPayload.outputTensor = outputName;
        outFinding.linearReductionPayload.K = K;
        outFinding.linearReductionPayload.hasBias = hasBias;
        outFinding.reason = "linear reduction: output[0] = bias[0] + sum(input[i]*weight[i], i=0.." +
                            std::to_string(K) + ")";
        found = true;
    });

    return found;
}

#else

// 功能：尝试处理Try Match Linear Reduction相关逻辑。
bool TryMatchLinearReduction(const CalcMlirArtifact&,
                             const CalcAnalysisOptions&,
                             CalcSemanticFinding& outFinding) {
    outFinding = CalcSemanticFinding{};
    return false;
}

// 功能：尝试处理Try Recognize Cube Matmul Reduction相关逻辑。
bool TryRecognizeCubeMatmulReduction(const CalcMlirArtifact&,
                                     const CalcAnalysisOptions&,
                                     const std::string&,
                                     CubeMatmulPayload& outPayload,
                                     std::string& outReason) {
    outPayload = CubeMatmulPayload{};
    outReason = "mlir backend unavailable";
    return false;
}

#endif

// 功能：处理Subtree Contains Node相关逻辑。
bool SubtreeContainsNode(const CalcExprGraph& graph,
                         int rootNodeId,
                         int probeNodeId,
                         std::unordered_set<int>& visiting) {
    if (rootNodeId < 0 || static_cast<size_t>(rootNodeId) >= graph.nodes.size()) {
        return false;
    }
    if (rootNodeId == probeNodeId) {
        return true;
    }
    if (visiting.find(rootNodeId) != visiting.end()) {
        return false;
    }
    visiting.insert(rootNodeId);

    const CalcExprNode& node = graph.nodes[static_cast<size_t>(rootNodeId)];
    for (int inId : node.inputs) {
        if (SubtreeContainsNode(graph, inId, probeNodeId, visiting)) {
            return true;
        }
    }
    return false;
}

// 功能：处理Subtree Contains Node相关逻辑。
bool SubtreeContainsNode(const CalcExprGraph& graph,
                         int rootNodeId,
                         int probeNodeId) {
    std::unordered_set<int> visiting;
    return SubtreeContainsNode(graph, rootNodeId, probeNodeId, visiting);
}

enum class StructuredExprMode {
    Map,
    Update,
    Post,
};

struct StructuredExprRenderConfig {
    std::string srcTensor;
    int centerOffset = 0;
    int shiftOffset = 0;
    int centerMappedNodeId = -1;
    int shiftedMappedNodeId = -1;
    int updateNodeId = -1;
};

std::string RenderStructuredExpr(const CalcExprGraph& graph,
                                 int nodeId,
                                 StructuredExprMode mode,
                                 const StructuredExprRenderConfig& config,
                                 bool& ok) {
    std::unordered_set<int> visiting;
    ExprNodeRenderer renderer = [&](int nid, const CalcExprNode& node,
                                     bool& okRef) -> std::optional<std::string> {
        if (mode == StructuredExprMode::Update) {
            if (nid == config.centerMappedNodeId) return std::string("CenterMapped");
            if (nid == config.shiftedMappedNodeId) return std::string("ShiftedMapped");
        }
        if (mode == StructuredExprMode::Post && nid == config.updateNodeId) {
            return std::string("UpdateResult");
        }
        if (node.kind != CalcExprNodeKind::Load) return std::nullopt;

        const std::string tensorKey =
            node.canonicalTensor.empty() ? node.token : node.canonicalTensor;
        if (mode == StructuredExprMode::Map) {
            return (tensorKey == config.srcTensor) ? std::string("RawInput") : tensorKey;
        }
        if (mode == StructuredExprMode::Update) {
            if (tensorKey == config.srcTensor) {
                if (!node.offsetKnown) { okRef = false; return std::string("?"); }
                if (node.offset == config.centerOffset) return std::string("CenterRaw");
                if (node.offset == config.shiftOffset) return std::string("ShiftedRaw");
                okRef = false;
                return std::string("?");
            }
            return tensorKey;
        }
        okRef = false;
        return std::string("?");
    };
    return RenderExprCommon(graph, nodeId, renderer, ok, visiting, 0);
}

enum class MultiViewExprMode {
    Main,
    Post,
};

struct MultiViewExprRenderConfig {
    std::string srcTensor;
    std::map<int, size_t> viewIndexByOffset;
    std::unordered_map<std::string, size_t> scalarIndexByTensor;
    int mainExprNodeId = -1;
};

std::string RenderMultiViewExpr(const CalcExprGraph& graph,
                                int nodeId,
                                MultiViewExprMode mode,
                                const MultiViewExprRenderConfig& config,
                                bool& ok) {
    std::unordered_set<int> visiting;
    ExprNodeRenderer renderer = [&](int nid, const CalcExprNode& node,
                                     bool& okRef) -> std::optional<std::string> {
        if (mode == MultiViewExprMode::Post && nid == config.mainExprNodeId) {
            return std::string("MainExpr");
        }
        if (node.kind != CalcExprNodeKind::Load) return std::nullopt;

        const std::string tensorKey =
            node.canonicalTensor.empty() ? node.token : node.canonicalTensor;
        if (tensorKey == config.srcTensor) {
            if (!node.offsetKnown) { okRef = false; return std::string("?"); }
            auto it = config.viewIndexByOffset.find(node.offset);
            if (it == config.viewIndexByOffset.end()) { okRef = false; return std::string("?"); }
            return std::string("View") + std::to_string(it->second);
        }
        auto scalarIt = config.scalarIndexByTensor.find(tensorKey);
        if (scalarIt == config.scalarIndexByTensor.end()) { okRef = false; return std::string("?"); }
        return std::string("Scalar") + std::to_string(scalarIt->second);
    };
    return RenderExprCommon(graph, nodeId, renderer, ok, visiting, 0);
}

// 功能：收集Collect Multi View Loads Dfs相关数据。
void CollectMultiViewLoadsDfs(const CalcExprGraph& graph,
                              int nodeId,
                              const std::string& srcTensor,
                              const CalcShellSummary* shellSummary,
                              std::unordered_set<int>& visiting,
                              std::set<int>& outViewOffsets,
                              std::vector<std::string>& outScalarInputs,
                              std::unordered_set<std::string>& seenScalarInputs,
                              bool& outHasUnknownSrcOffset,
                              bool& outHasNonScalarInput,
                              bool& outHasInvalidScalarOffset) {
    if (nodeId < 0 || static_cast<size_t>(nodeId) >= graph.nodes.size()) {
        return;
    }
    if (visiting.find(nodeId) != visiting.end()) {
        return;
    }
    visiting.insert(nodeId);

    const CalcExprNode& node = graph.nodes[static_cast<size_t>(nodeId)];
    if (node.kind == CalcExprNodeKind::Load && (node.isArgTensor || !node.canonicalTensor.empty())) {
        const std::string tensorKey =
            node.canonicalTensor.empty() ? node.token : node.canonicalTensor;
        if (tensorKey == srcTensor) {
            if (node.offsetKnown) {
                outViewOffsets.insert(node.offset);
            } else {
                outHasUnknownSrcOffset = true;
            }
        } else {
            bool isScalar = shellSummary ? IsScalarParam(shellSummary, tensorKey) : node.offsetKnown;
            if (!isScalar) {
                outHasNonScalarInput = true;
            }
            if (node.offsetKnown && node.offset != 0) {
                outHasInvalidScalarOffset = true;
            }
            if (seenScalarInputs.find(tensorKey) == seenScalarInputs.end()) {
                seenScalarInputs.insert(tensorKey);
                outScalarInputs.push_back(tensorKey);
            }
        }
    }

    for (int inputId : node.inputs) {
        CollectMultiViewLoadsDfs(graph,
                                 inputId,
                                 srcTensor,
                                 shellSummary,
                                 visiting,
                                 outViewOffsets,
                                 outScalarInputs,
                                 seenScalarInputs,
                                 outHasUnknownSrcOffset,
                                 outHasNonScalarInput,
                                 outHasInvalidScalarOffset);
    }
}

// 识别典型的 系数模板 (Coefficient Stencil) 计算，如：out[i] = c0 * A[i-1] + c1 * A[i] + c2 * A[i+1]。
bool TryExtractStructuredMultiViewExprPattern(const CalcExprGraph& graph,
                                              int rhsRootNodeId,
                                              const std::string& srcTensor,
                                              const std::string& dstTensor,
                                              const CalcShellSummary* shellSummary,
                                              StructuredMultiViewExprPattern& outPattern,
                                              std::string& outReason) {
    outPattern = StructuredMultiViewExprPattern{};
    outReason.clear();

    if (rhsRootNodeId < 0 || static_cast<size_t>(rhsRootNodeId) >= graph.nodes.size()) {
        outReason = "invalid RHS root node";
        return false;
    }

    std::set<int> viewOffsets;
    std::vector<std::string> scalarInputs;
    std::unordered_set<std::string> seenScalarInputs;
    bool hasUnknownSrcOffset = false;
    bool hasNonScalarInput = false;
    bool hasInvalidScalarOffset = false;
    {
        std::unordered_set<int> visiting;
        CollectMultiViewLoadsDfs(graph,
                                 rhsRootNodeId,
                                 srcTensor,
                                 shellSummary,
                                 visiting,
                                 viewOffsets,
                                 scalarInputs,
                                 seenScalarInputs,
                                 hasUnknownSrcOffset,
                                 hasNonScalarInput,
                                 hasInvalidScalarOffset);
    }

    if (hasUnknownSrcOffset) {
        outReason = "multi-view source load has unknown offset";
        return false;
    }
    if (viewOffsets.size() < 3) {
        outReason = "multi-view source offsets are insufficient";
        return false;
    }
    if (hasNonScalarInput) {
        outReason = "multi-view expression contains non-scalar side input";
        return false;
    }
    if (hasInvalidScalarOffset) {
        outReason = "scalar input participates with non-zero offset";
        return false;
    }

    int mainRootNodeId = rhsRootNodeId;
    int postRootNodeId = -1;
    const CalcExprNode& rhsRoot = graph.nodes[static_cast<size_t>(rhsRootNodeId)];
    if (rhsRoot.kind == CalcExprNodeKind::Clamp && !rhsRoot.inputs.empty() &&
        ContainsTensorLoad(graph, rhsRoot.inputs[0], srcTensor)) {
        mainRootNodeId = rhsRoot.inputs[0];
        postRootNodeId = rhsRootNodeId;
    } else if (rhsRoot.kind == CalcExprNodeKind::Max && rhsRoot.inputs.size() >= 2) {
        const bool lhsHasSrc = ContainsTensorLoad(graph, rhsRoot.inputs[0], srcTensor);
        const bool rhsHasSrc = ContainsTensorLoad(graph, rhsRoot.inputs[1], srcTensor);
        if (lhsHasSrc != rhsHasSrc) {
            mainRootNodeId = lhsHasSrc ? rhsRoot.inputs[0] : rhsRoot.inputs[1];
            postRootNodeId = rhsRootNodeId;
        }
    }

    MultiViewExprRenderConfig config;
    config.srcTensor = srcTensor;
    size_t idx = 0;
    for (int offset : viewOffsets) {
        config.viewIndexByOffset[offset] = idx;
        ++idx;
    }
    for (size_t scalarIdx = 0; scalarIdx < scalarInputs.size(); ++scalarIdx) {
        config.scalarIndexByTensor[scalarInputs[scalarIdx]] = scalarIdx;
    }
    config.mainExprNodeId = mainRootNodeId;

    bool okMain = true;
    std::string mainExpr = RenderMultiViewExpr(graph,
                                               mainRootNodeId,
                                               MultiViewExprMode::Main,
                                               config,
                                               okMain);
    if (!okMain || mainExpr.empty()) {
        outReason = "failed to render structured main expression";
        return false;
    }

    bool hasPost = postRootNodeId >= 0;
    std::string postExpr;
    if (hasPost) {
        bool okPost = true;
        postExpr = RenderMultiViewExpr(graph,
                                       postRootNodeId,
                                       MultiViewExprMode::Post,
                                       config,
                                       okPost);
        if (!okPost || postExpr.empty()) {
            outReason = "failed to render structured post expression";
            return false;
        }
    }

    outPattern.srcTensor = srcTensor;
    outPattern.dstTensor = dstTensor;
    outPattern.viewOffsets.assign(viewOffsets.begin(), viewOffsets.end());
    outPattern.scalarInputs = scalarInputs;
    outPattern.mainExpr = mainExpr;
    outPattern.hasPost = hasPost;
    outPattern.postExpr = postExpr;
    outPattern.boundaryPolicy = BuildCalcBoundaryPolicy(srcTensor, shellSummary);
    InferHaloAndLengths(shellSummary,
                   srcTensor,
                   viewOffsets,
                   outPattern.leftHalo,
                   outPattern.rightHalo,
                   outPattern.validLength,
                   outPattern.copyLength);

    std::ostringstream reason;
    reason << "structured multi-view expr recognized"
           << "; views=" << outPattern.viewOffsets.size()
           << "; scalarInputs=" << outPattern.scalarInputs.size()
           << "; mainNode#" << mainRootNodeId;
    if (hasPost) {
        reason << ", postNode#" << postRootNodeId;
    }
    outReason = reason.str();
    return true;
}

// 功能：检查Is Direct Tensor View相关条件。
bool IsDirectTensorView(const CalcExprGraph& graph,
                        int nodeId,
                        const std::string& tensor,
                        int expectedOffset) {
    if (nodeId < 0 || static_cast<size_t>(nodeId) >= graph.nodes.size()) {
        return false;
    }
    const CalcExprNode& node = graph.nodes[static_cast<size_t>(nodeId)];
    if (node.kind != CalcExprNodeKind::Load) {
        return false;
    }
    const std::string tensorKey = node.canonicalTensor.empty() ? node.token : node.canonicalTensor;
    return tensorKey == tensor && node.offsetKnown && node.offset == expectedOffset;
}

// 功能：尝试处理Try Extract Single Tensor Offset相关逻辑。
bool TryExtractSingleTensorOffset(const CalcExprGraph& graph,
                                  int nodeId,
                                  const std::string& tensor,
                                  int& outOffset) {
    std::set<int> offsets = CollectTensorOffsets(graph, nodeId, tensor);
    if (offsets.size() != 1) {
        return false;
    }
    outOffset = *offsets.begin();
    return true;
}

// 功能：尝试处理Try Extract Structured Shifted Map Update Pattern相关逻辑。
bool TryExtractStructuredShiftedMapUpdatePattern(const CalcExprGraph& graph,
                                                 int rhsRootNodeId,
                                                 const std::string& srcTensor,
                                                 const std::string& dstTensor,
                                                 const CalcShellSummary* shellSummary,
                                                 StructuredShiftedMapUpdatePattern& outPattern,
                                                 std::string& outReason) {
    outPattern = StructuredShiftedMapUpdatePattern{};
    outReason.clear();

    int shiftedNodeId = -1;
    int centerNodeId = -1;
    int diffNodeId = -1;
    if (!TryFindWindowedMirrorSubExpr(graph,
                                      rhsRootNodeId,
                                      srcTensor,
                                      shiftedNodeId,
                                      centerNodeId,
                                      diffNodeId)) {
        return false;
    }

    int centerOffset = 0;
    int shiftOffset = 0;
    if (!TryExtractSingleTensorOffset(graph, centerNodeId, srcTensor, centerOffset) ||
        !TryExtractSingleTensorOffset(graph, shiftedNodeId, srcTensor, shiftOffset)) {
        return false;
    }

    if (centerOffset != 0 && shiftOffset == 0) {
        std::swap(centerNodeId, shiftedNodeId);
        std::swap(centerOffset, shiftOffset);
    }
    if (centerOffset != 0 || shiftOffset == 0) {
        return false;
    }

    const std::string centerSig =
        BuildCanonicalSubExprSignature(graph, centerNodeId, srcTensor, true);
    const std::string shiftedSig =
        BuildCanonicalSubExprSignature(graph, shiftedNodeId, srcTensor, true);
    if (centerSig.empty() || centerSig != shiftedSig) {
        return false;
    }

    int updateRootNodeId = rhsRootNodeId;
    int postRootNodeId = -1;
    if (rhsRootNodeId >= 0 && static_cast<size_t>(rhsRootNodeId) < graph.nodes.size()) {
        const CalcExprNode& rhsRoot = graph.nodes[static_cast<size_t>(rhsRootNodeId)];
        if (rhsRoot.kind == CalcExprNodeKind::Clamp && !rhsRoot.inputs.empty() &&
            SubtreeContainsNode(graph, rhsRoot.inputs[0], diffNodeId)) {
            updateRootNodeId = rhsRoot.inputs[0];
            postRootNodeId = rhsRootNodeId;
        } else if (rhsRoot.kind == CalcExprNodeKind::Max && rhsRoot.inputs.size() >= 2) {
            const bool lhsHasDiff = SubtreeContainsNode(graph, rhsRoot.inputs[0], diffNodeId);
            const bool rhsHasDiff = SubtreeContainsNode(graph, rhsRoot.inputs[1], diffNodeId);
            if (lhsHasDiff != rhsHasDiff) {
                updateRootNodeId = lhsHasDiff ? rhsRoot.inputs[0] : rhsRoot.inputs[1];
                postRootNodeId = rhsRootNodeId;
            }
        }
    }

    StructuredExprRenderConfig config;
    config.srcTensor = srcTensor;
    config.centerOffset = centerOffset;
    config.shiftOffset = shiftOffset;
    config.centerMappedNodeId = centerNodeId;
    config.shiftedMappedNodeId = shiftedNodeId;
    config.updateNodeId = updateRootNodeId;

    const bool directCenter = IsDirectTensorView(graph, centerNodeId, srcTensor, centerOffset);
    const bool directShift = IsDirectTensorView(graph, shiftedNodeId, srcTensor, shiftOffset);
    const bool hasMap = !(directCenter && directShift);

    std::string mapExpr = "RawInput";
    if (hasMap) {
        bool okMap = true;
        mapExpr = RenderStructuredExpr(graph,
                                       centerNodeId,
                                       StructuredExprMode::Map,
                                       config,
                                       okMap);
        if (!okMap || mapExpr.empty()) {
            return false;
        }
    }

    bool okUpdate = true;
    std::string updateExpr = RenderStructuredExpr(graph,
                                                  updateRootNodeId,
                                                  StructuredExprMode::Update,
                                                  config,
                                                  okUpdate);
    if (!okUpdate || updateExpr.empty()) {
        return false;
    }

    std::string postExpr;
    bool hasPost = postRootNodeId >= 0;
    if (hasPost) {
        bool okPost = true;
        postExpr = RenderStructuredExpr(graph,
                                        postRootNodeId,
                                        StructuredExprMode::Post,
                                        config,
                                        okPost);
        if (!okPost || postExpr.empty()) {
            return false;
        }
    }

    outPattern.srcTensor = srcTensor;
    outPattern.dstTensor = dstTensor;
    outPattern.shiftOffset = shiftOffset;
    outPattern.hasMap = hasMap;
    outPattern.hasPost = hasPost;
    outPattern.boundaryPolicy = BuildCalcBoundaryPolicy(srcTensor, shellSummary);
    outPattern.mapExpr = mapExpr;
    outPattern.updateExpr = updateExpr;
    outPattern.postExpr = postExpr;

    std::ostringstream reason;
    reason << "structured shifted-map-update recognized"
           << "; mapNode#" << centerNodeId
           << ", diffNode#" << diffNodeId
           << ", updateNode#" << updateRootNodeId;
    if (hasPost) {
        reason << ", postNode#" << postRootNodeId;
    }
    outReason = reason.str();
    return true;
}

// 功能：尝试处理Try Build Pointwise Family Finding相关逻辑。
bool TryBuildPointwiseFamilyFinding(const CalcSemanticFinding& seed,
                                    CalcSemanticFinding& outFinding) {
    switch (seed.category) {
    case CalcSemanticCategory::ElementwiseBinary:
    case CalcSemanticCategory::DecayExpMul:
    case CalcSemanticCategory::PointwiseExpr:
    case CalcSemanticCategory::Clamp:
        break;
    default:
        return false;
    }

    outFinding = seed;
    outFinding.family = CalcSemanticFamily::PointwiseFamily;
    outFinding.hasPointwisePayload = true;
    outFinding.pointwisePayload.operation = seed.category;

    outFinding.hasFamilyPayload = true;
    outFinding.familyPayload = CalcFamilyPayload{};
    outFinding.familyPayload.dstTensor = seed.target;
    outFinding.familyPayload.mainExpr = ToString(seed.category);
    outFinding.familyPayload.boundaryPolicy = CalcBoundaryPolicy::Clamp;
    for (const AffineAccess& access : seed.inputs) {
        if (outFinding.familyPayload.srcTensor.empty()) {
            outFinding.familyPayload.srcTensor = access.tensor;
            outFinding.familyPayload.boundaryPolicy = BuildCalcBoundaryPolicy(access.tensor, nullptr);
        }
        outFinding.familyPayload.offsets.push_back(access.offset);
    }
    std::sort(outFinding.familyPayload.offsets.begin(), outFinding.familyPayload.offsets.end());
    outFinding.familyPayload.offsets.erase(
        std::unique(outFinding.familyPayload.offsets.begin(), outFinding.familyPayload.offsets.end()),
        outFinding.familyPayload.offsets.end());
    return true;
}

void PopulateFamilyPayloadFromPattern(const StructuredMultiViewExprPattern& pattern,
                                      const std::string& target,
                                      bool loopCarried,
                                      CalcFamilyPayload& out) {
    out = CalcFamilyPayload{};
    out.srcTensor = pattern.srcTensor;
    out.dstTensor = pattern.dstTensor.empty() ? target : pattern.dstTensor;
    out.offsets = pattern.viewOffsets;
    out.scalarInputs = pattern.scalarInputs;
    out.mainExpr = pattern.mainExpr;
    out.hasPostExpr = pattern.hasPost;
    out.postExpr = pattern.postExpr;
    out.boundaryPolicy = pattern.boundaryPolicy;
    out.loopCarried = loopCarried;
    out.leftHalo = pattern.leftHalo;
    out.rightHalo = pattern.rightHalo;
    out.validLength = pattern.validLength;
    out.copyLength = pattern.copyLength;
}

// 功能：尝试处理Try Build Spatial Stencil Family Finding相关逻辑。
bool TryBuildSpatialStencilFamilyFinding(const CalcSemanticFinding& seed,
                                         CalcSemanticFinding& outFinding) {
    if (seed.category == CalcSemanticCategory::StructuredShiftedMapUpdate &&
        seed.hasStructuredShiftedMapUpdate) {
        outFinding = seed;
        outFinding.family = CalcSemanticFamily::SpatialStencilFamily;
        outFinding.hasFamilyPayload = true;
        outFinding.familyPayload = CalcFamilyPayload{};

        const StructuredShiftedMapUpdatePattern& pattern = seed.structuredShiftedMapUpdate;
        outFinding.familyPayload.srcTensor = pattern.srcTensor;
        outFinding.familyPayload.dstTensor = pattern.dstTensor.empty() ? seed.target : pattern.dstTensor;
        outFinding.familyPayload.offsets = {0, pattern.shiftOffset};
        outFinding.familyPayload.mainExpr = pattern.updateExpr;
        outFinding.familyPayload.hasPostExpr = pattern.hasPost;
        outFinding.familyPayload.postExpr = pattern.postExpr;
        outFinding.familyPayload.boundaryPolicy = pattern.boundaryPolicy;
        outFinding.familyPayload.loopCarried = false;
        outFinding.familyPayload.leftHalo = pattern.shiftOffset < 0 ? -pattern.shiftOffset : 0;
        outFinding.familyPayload.rightHalo = pattern.shiftOffset > 0 ? pattern.shiftOffset : 0;
        return true;
    }

    if (seed.category == CalcSemanticCategory::CoeffStencil &&
        seed.hasStructuredMultiViewExpr &&
        seed.structuredMultiViewExpr.scalarInputs.empty()) {
        outFinding = seed;
        outFinding.family = CalcSemanticFamily::SpatialStencilFamily;
        PopulateFamilyPayloadFromPattern(seed.structuredMultiViewExpr, seed.target,
                                         false, outFinding.familyPayload);
        outFinding.hasFamilyPayload = true;
        return true;
    }

    if (seed.category == CalcSemanticCategory::WindowedPointwise ||
        seed.category == CalcSemanticCategory::ShiftedElementwise) {
        outFinding = seed;
        outFinding.family = CalcSemanticFamily::SpatialStencilFamily;
        outFinding.hasFamilyPayload = true;
        outFinding.familyPayload = CalcFamilyPayload{};
        outFinding.familyPayload.dstTensor = seed.target;
        for (const AffineAccess& access : seed.inputs) {
            if (outFinding.familyPayload.srcTensor.empty()) {
                outFinding.familyPayload.srcTensor = access.tensor;
            }
            outFinding.familyPayload.offsets.push_back(access.offset);
        }
        std::sort(outFinding.familyPayload.offsets.begin(), outFinding.familyPayload.offsets.end());
        outFinding.familyPayload.offsets.erase(
            std::unique(outFinding.familyPayload.offsets.begin(), outFinding.familyPayload.offsets.end()),
            outFinding.familyPayload.offsets.end());
        outFinding.familyPayload.mainExpr = ToString(seed.category);
        return true;
    }

    return false;
}

// 功能：尝试处理Try Build Temporal Stencil Family Finding相关逻辑。
bool TryBuildTemporalStencilFamilyFinding(const CalcSemanticFinding& seed,
                                          CalcSemanticFinding& outFinding) {
    if (seed.category != CalcSemanticCategory::CoeffStencil ||
        !seed.hasStructuredMultiViewExpr) {
        return false;
    }

    const StructuredMultiViewExprPattern& pattern = seed.structuredMultiViewExpr;
    if (pattern.scalarInputs.empty()) {
        return false;
    }

    outFinding = seed;
    outFinding.family = CalcSemanticFamily::TemporalStencilFamily;
    PopulateFamilyPayloadFromPattern(pattern, seed.target,
                                     true, outFinding.familyPayload);
    outFinding.hasFamilyPayload = true;
    return true;
}

// 功能：尝试处理Try Build Cube Matmul Family Finding相关逻辑。
bool TryBuildCubeMatmulFamilyFinding(const CalcSemanticFinding& seed,
                                     CalcSemanticFinding& outFinding) {
    if (seed.category != CalcSemanticCategory::MatmulReduction ||
        !seed.hasCubeMatmulPayload) {
        return false;
    }

    outFinding = seed;
    outFinding.family = CalcSemanticFamily::CubeMatmulFamily;
    outFinding.hasPointwisePayload = false;
    outFinding.hasFamilyPayload = false;
    outFinding.familyPayload = CalcFamilyPayload{};
    return true;
}

} // namespace

// 功能：渲染To String相关文本。
const char* ToString(CalcSemanticCategory category) {
    switch (category) {
    case CalcSemanticCategory::StructuredShiftedMapUpdate:
        return "structured-shifted-map-update";
    case CalcSemanticCategory::ElementwiseBinary:
        return "elementwise-binary";
    case CalcSemanticCategory::DecayExpMul:
        return "decay-exp-mul";
    case CalcSemanticCategory::CoeffStencil:
        return "coeff-stencil";
    case CalcSemanticCategory::WindowedPointwise:
        return "windowed-pointwise";
    case CalcSemanticCategory::ShiftedElementwise:
        return "shifted-elementwise";
    case CalcSemanticCategory::PointwiseExpr:
        return "pointwise-expr";
    case CalcSemanticCategory::Clamp:
        return "clamp";
    case CalcSemanticCategory::MatmulReduction:
        return "matmul-reduction";
    case CalcSemanticCategory::Reduction:
        return "reduction";
    case CalcSemanticCategory::LinearReduction:
        return "linear-reduction";
    case CalcSemanticCategory::ScalarFallback:
        return "scalar-fallback";
    }
    return "unknown";
}

// 功能：渲染To String相关文本。
const char* ToString(CalcSemanticFamily family) {
    switch (family) {
    case CalcSemanticFamily::PointwiseFamily:
        return "PointwiseFamily";
    case CalcSemanticFamily::SpatialStencilFamily:
        return "SpatialStencilFamily";
    case CalcSemanticFamily::TemporalStencilFamily:
        return "TemporalStencilFamily";
    case CalcSemanticFamily::CubeMatmulFamily:
        return "CubeMatmulFamily";
    case CalcSemanticFamily::ReductionFamily:
        return "ReductionFamily";
    case CalcSemanticFamily::LinearReductionFamily:
        return "LinearReductionFamily";
    case CalcSemanticFamily::ScalarFallbackFamily:
        return "ScalarFallbackFamily";
    }
    return "ScalarFallbackFamily";
}

// 功能：渲染To String相关文本。
const char* ToString(CalcBoundaryPolicy policy) {
    switch (policy) {
    case CalcBoundaryPolicy::Replicate:
        return "Replicate";
    case CalcBoundaryPolicy::Clamp:
        return "Clamp";
    case CalcBoundaryPolicy::RequireHalo:
        return "RequireHalo";
    }
    return "Clamp";
}

/*
启发式匹配模型：
1.构建图： 调用 BuildCalcExprGraph 拿到表达式图。
2.遍历输出： 找到图中所有的 Store 节点（代表往内存写结果的操作）。
3.特征收集： 对每个 Store 节点等号右边（RHS）的计算子图进行深度优先遍历（DFS），提取统计特征。
4.模式匹配： 根据收集到的特征，按优先级从高到低尝试套用预定义的模板。
5.族系归类与参数推导： 将匹配到的具体模式归类为更宏观的Semantic Family，并计算 NPU Tiling所需的Halo大小。
*/
CalcAnalysisResult AnalyzeCalcMlirModule(const CalcMlirArtifact& artifact,
                                         const CalcAnalysisOptions& options) {
    CalcAnalysisResult result;
    result.analyzed = true;

    // Try the linear-reduction pattern first, before building the expr-graph.
    // The expr-graph truncates local accumulators (multiSourceMemrefs), so this
    // pattern must be matched directly on the MLIR module.
    {
        CalcSemanticFinding linearFinding;
        if (TryMatchLinearReduction(artifact, options, linearFinding)) {
            result.findings.push_back(std::move(linearFinding));
            return result;
        }
    }

    result.exprGraph = BuildCalcExprGraph(artifact, options.shellSummary, options.paramOrder);

    if (!result.exprGraph.valid) {
        CalcSemanticFinding finding;
        finding.category = CalcSemanticCategory::ScalarFallback;
        finding.target = "module";
        finding.reason = "expression graph unavailable";
        result.findings.push_back(std::move(finding));

        for (const std::string& note : result.exprGraph.notes) {
            result.notes.push_back(note);
        }
        return result;
    }

    for (int storeNodeId : result.exprGraph.storeNodeIds) {
        if (storeNodeId < 0 || static_cast<size_t>(storeNodeId) >= result.exprGraph.nodes.size()) {
            continue;
        }

        const CalcExprNode& storeNode = result.exprGraph.nodes[static_cast<size_t>(storeNodeId)];
        if (storeNode.inputs.empty()) {
            CalcSemanticFinding finding;
            finding.category = CalcSemanticCategory::ScalarFallback;
            finding.target = storeNode.token;
            finding.reason = "store node has no RHS expression";
            result.findings.push_back(std::move(finding));
            continue;
        }

        int rhsId = storeNode.inputs.front();
        if (rhsId < 0 || static_cast<size_t>(rhsId) >= result.exprGraph.nodes.size()) {
            CalcSemanticFinding finding;
            finding.category = CalcSemanticCategory::ScalarFallback;
            finding.target = storeNode.token;
            finding.reason = "store RHS node is invalid";
            result.findings.push_back(std::move(finding));
            continue;
        }

        ExprStats stats;
        std::unordered_set<int> visited;
        CollectStatsDfs(result.exprGraph, rhsId, storeNode.token, visited, stats);

        const CalcExprNode& rhsRoot = result.exprGraph.nodes[static_cast<size_t>(rhsId)];
        std::set<int> primaryOffsets;
        std::string primaryTensor = PickPrimaryTensor(stats, primaryOffsets);

        CalcSemanticFinding finding;
        finding.target = storeNode.token;

        if (rhsRoot.kind == CalcExprNodeKind::Clamp ||
            (rhsRoot.kind == CalcExprNodeKind::Max && primaryTensor == storeNode.token)) {
            finding.category = CalcSemanticCategory::Clamp;
            if (!primaryTensor.empty()) {
                finding.inputs = BuildAffineInputs(primaryTensor, std::set<int>{0});
            }
            finding.reason = "root expression is clamp/max-to-bound; " + BuildDiagSuffix(stats);
            result.findings.push_back(std::move(finding));
            continue;
        }

        if (stats.hasSelfRead) {
            CubeMatmulPayload matmulPayload;
            std::string matmulReason;
            if (TryRecognizeCubeMatmulReduction(
                    artifact,
                    options,
                    storeNode.token,
                    matmulPayload,
                    matmulReason) ||
                // 功能：尝试处理Try Recognize Cube Matmul Expr Graph Pattern相关逻辑。
                TryRecognizeCubeMatmulExprGraphPattern(
                    result.exprGraph,
                    options,
                    storeNode,
                    matmulPayload,
                    matmulReason)) {
                finding.category = CalcSemanticCategory::MatmulReduction;
                finding.hasCubeMatmulPayload = true;
                finding.cubeMatmulPayload = matmulPayload;
                finding.inputs.push_back(AffineAccess{matmulPayload.srcA, 0, 1.0});
                finding.inputs.push_back(AffineAccess{matmulPayload.srcB, 0, 1.0});
                finding.reason = matmulReason + "; " + BuildDiagSuffix(stats);
            } else {
                finding.category = CalcSemanticCategory::Reduction;
                finding.reason = "expression graph reads the same tensor as store target; " +
                                 matmulReason + "; " + BuildDiagSuffix(stats);
            }
            result.findings.push_back(std::move(finding));
            continue;
        }

        const size_t weightedTermCount =
            primaryTensor.empty() ? 0 : CountAdditiveTermsForTensor(result.exprGraph, rhsId, primaryTensor);

        if (!primaryTensor.empty() && primaryOffsets.size() >= 3 && weightedTermCount >= 3) {
            StructuredMultiViewExprPattern structuredPattern;
            std::string structuredReason;
            if (TryExtractStructuredMultiViewExprPattern(
                    result.exprGraph,
                    rhsId,
                    primaryTensor,
                    storeNode.token,
                    options.shellSummary,
                    structuredPattern,
                    structuredReason)) {
                finding.category = CalcSemanticCategory::CoeffStencil;
                finding.inputs = BuildAffineInputs(
                    primaryTensor,
                    std::set<int>(structuredPattern.viewOffsets.begin(), structuredPattern.viewOffsets.end()));
                finding.hasStructuredMultiViewExpr = true;
                finding.structuredMultiViewExpr = structuredPattern;
                finding.reason = structuredReason + "; " + BuildDiagSuffix(stats);
            } else {
                finding.category = CalcSemanticCategory::ScalarFallback;
                finding.reason = "coeff-stencil candidate failed structured extraction: " +
                                 structuredReason + "; " + BuildDiagSuffix(stats);
            }
            result.findings.push_back(std::move(finding));
            continue;
        }

        if (!primaryTensor.empty() && primaryOffsets.size() >= 2 && stats.hasNonZeroOffset &&
            primaryOffsets.find(0) != primaryOffsets.end()) {
            StructuredShiftedMapUpdatePattern structuredPattern;
            std::string structuredReason;
            if (TryExtractStructuredShiftedMapUpdatePattern(
                    result.exprGraph,
                    rhsId,
                    primaryTensor,
                    storeNode.token,
                    options.shellSummary,
                    structuredPattern,
                    structuredReason)) {
                finding.category = CalcSemanticCategory::StructuredShiftedMapUpdate;
                finding.hasStructuredShiftedMapUpdate = true;
                finding.structuredShiftedMapUpdate = structuredPattern;
                finding.inputs = BuildAffineInputs(primaryTensor,
                                                   std::set<int>{0, structuredPattern.shiftOffset});
                finding.reason = structuredReason + "; " + BuildDiagSuffix(stats);
                result.findings.push_back(std::move(finding));
                continue;
            }
        }

        if (!primaryTensor.empty() && primaryOffsets.size() >= 2 && stats.hasNonZeroOffset &&
            (stats.hasMul || stats.hasDiv)) {
            int shiftedSubExprRoot = -1;
            int centerSubExprRoot = -1;
            int diffSubExprRoot = -1;
            bool hasMirroredPair = TryFindWindowedMirrorSubExpr(
                result.exprGraph,
                rhsId,
                primaryTensor,
                shiftedSubExprRoot,
                centerSubExprRoot,
                diffSubExprRoot);

            finding.category = CalcSemanticCategory::WindowedPointwise;
            finding.inputs = BuildAffineInputs(primaryTensor, primaryOffsets);
            std::ostringstream reason;
            reason << "windowed pointwise expression preserved";
            if (hasMirroredPair) {
                reason << "; q(shifted)#" << shiftedSubExprRoot
                       << ", q(center)#" << centerSubExprRoot
                       << ", diff#" << diffSubExprRoot
                       << ", outer#" << rhsId;
            } else {
                reason << "; mirrored q-branches unresolved, keep full outer expr#" << rhsId;
            }
            reason << "; " << BuildDiagSuffix(stats);
            finding.reason = reason.str();
            result.findings.push_back(std::move(finding));
            continue;
        }

        if (stats.offsetsByTensor.size() == 1 && stats.hasNonZeroOffset) {
            std::set<int> shifted;
            for (int off : primaryOffsets) {
                if (off != 0) {
                    shifted.insert(off);
                }
            }
            if (shifted.empty()) {
                shifted.insert(*primaryOffsets.begin());
            }
            finding.category = CalcSemanticCategory::ShiftedElementwise;
            finding.inputs = BuildAffineInputs(primaryTensor, shifted);
            finding.reason = "single-input shifted window expression chain; " + BuildDiagSuffix(stats);
            result.findings.push_back(std::move(finding));
            continue;
        }

        if (stats.offsetsByTensor.size() == 2 && !stats.hasUnknownOffset && stats.allOffsetsZero &&
            (stats.hasAddSub || stats.hasMul || stats.hasDiv || stats.hasMax)) {
            finding.category = CalcSemanticCategory::ElementwiseBinary;
            finding.inputs = BuildElementwiseInputs(stats);
            finding.reason = "aligned two-tensor elementwise graph; " + BuildDiagSuffix(stats);
            result.findings.push_back(std::move(finding));
            continue;
        }

        if (stats.hasExp && stats.hasMul && stats.offsetsByTensor.size() >= 2) {
            finding.category = CalcSemanticCategory::DecayExpMul;
            finding.inputs = BuildElementwiseInputs(stats);
            finding.reason = "graph contains exp-mul decay chain with multi-input tensors; " + BuildDiagSuffix(stats);
            result.findings.push_back(std::move(finding));
            continue;
        }

        if (!stats.hasUnknown && !stats.hasUnknownOffset && stats.allOffsetsZero &&
            CanLowerDirectPointwiseExpr(result.exprGraph, rhsId) &&
            (stats.hasAddSub || stats.hasMul || stats.hasDiv ||
             stats.hasMin || stats.hasMax || stats.hasClamp || stats.hasExp)) {
            finding.category = CalcSemanticCategory::PointwiseExpr;
            finding.inputs = BuildElementwiseInputs(stats);
            finding.reason = "direct pointwise expression tree preserved; " + BuildDiagSuffix(stats);
            result.findings.push_back(std::move(finding));
            continue;
        }

        finding.category = CalcSemanticCategory::ScalarFallback;
        finding.reason = "graph pattern is not mapped to an optimized semantic class; " + BuildDiagSuffix(stats);
        result.findings.push_back(std::move(finding));
    }

    for (CalcSemanticFinding& finding : result.findings) {
        CalcSemanticFinding upgraded;
        if (TryBuildCubeMatmulFamilyFinding(finding, upgraded) ||
            
            TryBuildTemporalStencilFamilyFinding(finding, upgraded) ||
            
            TryBuildSpatialStencilFamilyFinding(finding, upgraded) ||
            
            TryBuildPointwiseFamilyFinding(finding, upgraded)) {
            finding = std::move(upgraded);
            continue;
        }

        if (finding.category == CalcSemanticCategory::MatmulReduction) {
            finding.family = CalcSemanticFamily::CubeMatmulFamily;
        } else if (finding.category == CalcSemanticCategory::Reduction) {
            finding.family = CalcSemanticFamily::ReductionFamily;
        } else {
            finding.family = CalcSemanticFamily::ScalarFallbackFamily;
        }
        finding.hasPointwisePayload = false;
        finding.hasFamilyPayload = false;
        if (finding.family != CalcSemanticFamily::CubeMatmulFamily) {
            finding.hasCubeMatmulPayload = false;
            finding.cubeMatmulPayload = CubeMatmulPayload{};
        }
    }

    if (result.findings.empty()) {
        CalcSemanticFinding finding;
        finding.category = CalcSemanticCategory::ScalarFallback;
        finding.family = CalcSemanticFamily::ScalarFallbackFamily;
        finding.target = "module";
        finding.reason = "expression graph has no store node";
        result.findings.push_back(std::move(finding));
    }

    for (const std::string& note : result.exprGraph.notes) {
        result.notes.push_back(note);
    }

    return result;
}

// 功能：渲染Render Calc Analysis Plan相关文本。
std::string RenderCalcAnalysisPlan(const CalcAnalysisResult& result) {
    std::ostringstream oss;
    oss << "plan:\n";
    for (size_t i = 0; i < result.findings.size(); ++i) {
        const CalcSemanticFinding& finding = result.findings[i];
        oss << "  - target: " << finding.target << "\n";
        oss << "    class: " << ToString(finding.category) << "\n";
        oss << "    family: " << ToString(finding.family) << "\n";
        if (!finding.inputs.empty()) {
            oss << "    inputs:";
            for (const AffineAccess& input : finding.inputs) {
                oss << " (tensor=" << input.tensor
                    << ", offset=" << input.offset
                    << ", coeff=" << input.coeff << ")";
            }
            oss << "\n";
        }
        if (finding.hasFamilyPayload) {
            const CalcFamilyPayload& payload = finding.familyPayload;
            oss << "    payload: src=" << payload.srcTensor
                << " dst=" << payload.dstTensor
                << " offsets=[";
            for (size_t j = 0; j < payload.offsets.size(); ++j) {
                if (j != 0) {
                    oss << ",";
                }
                oss << payload.offsets[j];
            }
            oss << "] scalarInputs=[";
            for (size_t j = 0; j < payload.scalarInputs.size(); ++j) {
                if (j != 0) {
                    oss << ",";
                }
                oss << payload.scalarInputs[j];
            }
            oss << "] boundary=" << ToString(payload.boundaryPolicy)
                << " loop-carried=" << (payload.loopCarried ? "true" : "false")
                << " span=(L" << payload.leftHalo
                << ",R" << payload.rightHalo;
            if (payload.validLength > 0) {
                oss << ",valid=" << payload.validLength;
            }
            if (payload.copyLength > 0) {
                oss << ",copy=" << payload.copyLength;
            }
            oss << ")\n";
            if (!payload.mainExpr.empty()) {
                oss << "    mainExpr: " << payload.mainExpr << "\n";
            }
            if (payload.hasPostExpr) {
                oss << "    postExpr: " << payload.postExpr << "\n";
            }
        }
        if (finding.hasCubeMatmulPayload) {
            const CubeMatmulPayload& payload = finding.cubeMatmulPayload;
            oss << "    matmul: srcA=" << payload.srcA
                << " srcB=" << payload.srcB
                << " dst=" << payload.dst
                << " accumulateToDst=" << (payload.accumulateToDst ? "true" : "false")
                << " hasInitialDst=" << (payload.hasInitialDst ? "true" : "false")
                << " logicalInputType=" << payload.logicalInputType
                << " kernelInputType=" << payload.kernelInputType
                << " kernelOutputType=" << payload.kernelOutputType << "\n";
        }
        if (finding.category == CalcSemanticCategory::StructuredShiftedMapUpdate &&
            finding.hasStructuredShiftedMapUpdate) {
            const StructuredShiftedMapUpdatePattern& pattern = finding.structuredShiftedMapUpdate;
            oss << "    structured: src=" << pattern.srcTensor
                << " dst=" << pattern.dstTensor
                << " shift=" << pattern.shiftOffset
                << " hasMap=" << (pattern.hasMap ? 1 : 0)
                << " hasPost=" << (pattern.hasPost ? 1 : 0)
                << " boundary=" << ToString(pattern.boundaryPolicy) << "\n";
            oss << "    mapExpr: " << pattern.mapExpr << "\n";
            oss << "    updateExpr: " << pattern.updateExpr << "\n";
            if (pattern.hasPost) {
                oss << "    postExpr: " << pattern.postExpr << "\n";
            }
        }
        if (finding.category == CalcSemanticCategory::CoeffStencil &&
            finding.hasStructuredMultiViewExpr) {
            const StructuredMultiViewExprPattern& pattern = finding.structuredMultiViewExpr;
            oss << "    structuredMultiView: src=" << pattern.srcTensor
                << " dst=" << pattern.dstTensor
                << " views=[";
            for (size_t j = 0; j < pattern.viewOffsets.size(); ++j) {
                if (j != 0) {
                    oss << ",";
                }
                oss << pattern.viewOffsets[j];
            }
            oss << "] scalarInputs=[";
            for (size_t j = 0; j < pattern.scalarInputs.size(); ++j) {
                if (j != 0) {
                    oss << ",";
                }
                oss << pattern.scalarInputs[j];
            }
            oss << "] boundary=" << ToString(pattern.boundaryPolicy)
                << " span=(L" << pattern.leftHalo
                << ",R" << pattern.rightHalo;
            if (pattern.validLength > 0) {
                oss << ",valid=" << pattern.validLength;
            }
            if (pattern.copyLength > 0) {
                oss << ",copy=" << pattern.copyLength;
            }
            oss << ")\n";
            oss << "    mainExpr: " << pattern.mainExpr << "\n";
            if (pattern.hasPost) {
                oss << "    postExpr: " << pattern.postExpr << "\n";
            }
        }
        oss << "    reason: " << finding.reason << "\n";
    }

    if (!result.notes.empty()) {
        oss << "notes:\n";
        for (const std::string& note : result.notes) {
            oss << "  - " << note << "\n";
        }
    }

    oss << RenderCalcExprGraph(result.exprGraph);
    return oss.str();
}

} // namespace ascendc_mlir
} // namespace dacppTranslator
