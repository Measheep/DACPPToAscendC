/*
执行计划构建器.
它负责把这些抽象的数学模式，转化为针对华为 Ascend NPU 硬件特性的具体执行步骤.
它决定了数据该怎么搬运、边界该怎么处理、需要多大的内存重叠区，以及最终应该调用 AscendC 的哪类底层向量 API。
*/
#include "ascendc_mlir/AscendCPlan.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "ascendc_mlir/CalcMlirUtils.h"
#include "ascendc_mlir/CalcShellInfo.h"

#if defined(DACPP_TRANSLATOR_HAS_MLIR)
#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#endif

namespace dacppTranslator {
namespace ascendc_mlir {
namespace {

#if defined(DACPP_TRANSLATOR_HAS_MLIR)

struct StoreStmt {
    mlir::Value value;
    mlir::Value memref;
    std::string token;
    std::string resolvedToken;
};

// 功能：查找Find Store For Target相关数据。
const StoreStmt* FindStoreForTarget(const std::vector<StoreStmt>& stores,
                                    const std::string& targetToken) {
    for (const StoreStmt& store : stores) {
        if (store.resolvedToken == targetToken || store.token == targetToken) {
            return &store;
        }
    }

    if (!stores.empty()) {
        return &stores.front();
    }
    return nullptr;
}

// 功能：收集Collect Loaded Arg Memrefs相关数据。
std::vector<mlir::Value> CollectLoadedArgMemrefs(mlir::Value root,
                                                  llvm::DenseSet<mlir::Operation*>& visiting,
                                                  llvm::DenseSet<mlir::Value>& seen) {
    std::vector<mlir::Value> out;
    mlir::Operation* op = root.getDefiningOp();
    if (!op || visiting.contains(op)) {
        return out;
    }
    visiting.insert(op);

    if (auto load = llvm::dyn_cast<mlir::memref::LoadOp>(op)) {
        mlir::Value memref = load.getMemRef();
        if (IsArgMemref(memref) && !seen.contains(memref)) {
            seen.insert(memref);
            out.push_back(memref);
        }
    }

    for (mlir::Value operand : op->getOperands()) {
        std::vector<mlir::Value> child = CollectLoadedArgMemrefs(operand, visiting, seen);
        out.insert(out.end(), child.begin(), child.end());
    }

    return out;
}

// 功能：收集Collect Loaded Arg Memrefs相关数据。
std::vector<mlir::Value> CollectLoadedArgMemrefs(mlir::Value root) {
    llvm::DenseSet<mlir::Operation*> visiting;
    llvm::DenseSet<mlir::Value> seen;
    return CollectLoadedArgMemrefs(root, visiting, seen);
}

// 功能：提取或解析Resolve Loaded Memref相关信息。
mlir::Value ResolveLoadedMemref(mlir::Value value,
                                llvm::DenseSet<mlir::Operation*>& visiting) {
    mlir::Operation* op = value.getDefiningOp();
    if (!op || visiting.contains(op)) {
        return mlir::Value();
    }
    visiting.insert(op);

    if (auto load = llvm::dyn_cast<mlir::memref::LoadOp>(op)) {
        return load.getMemRef();
    }

    for (mlir::Value operand : op->getOperands()) {
        mlir::Value memref = ResolveLoadedMemref(operand, visiting);
        if (memref) {
            return memref;
        }
    }

    return mlir::Value();
}

// 功能：提取或解析Resolve Loaded Memref相关信息。
mlir::Value ResolveLoadedMemref(mlir::Value value) {
    llvm::DenseSet<mlir::Operation*> visiting;
    return ResolveLoadedMemref(value, visiting);
}

// 功能：检查Is Binary Arith Op相关条件。
bool IsBinaryArithOp(mlir::Operation* op) {
    return llvm::isa<mlir::arith::AddFOp, mlir::arith::AddIOp,
                     mlir::arith::SubFOp, mlir::arith::SubIOp,
                     mlir::arith::MulFOp, mlir::arith::MulIOp,
                     mlir::arith::DivFOp, mlir::arith::DivSIOp>(op);
}

// 功能：查找Find Root Binary Value相关数据。
mlir::Value FindRootBinaryValue(mlir::Value root,
                                llvm::DenseSet<mlir::Operation*>& visiting) {
    mlir::Operation* op = root.getDefiningOp();
    if (!op || visiting.contains(op)) {
        return mlir::Value();
    }
    visiting.insert(op);

    if (IsBinaryArithOp(op)) {
        return root;
    }

    for (mlir::Value operand : op->getOperands()) {
        mlir::Value nested = FindRootBinaryValue(operand, visiting);
        if (nested) {
            return nested;
        }
    }

    return mlir::Value();
}

// 功能：查找Find Root Binary Value相关数据。
mlir::Value FindRootBinaryValue(mlir::Value root) {
    llvm::DenseSet<mlir::Operation*> visiting;
    return FindRootBinaryValue(root, visiting);
}

// 功能：处理To Binary Kind相关逻辑。
AscendCBinaryOpKind ToBinaryKind(mlir::Operation* op) {
    if (llvm::isa<mlir::arith::AddFOp, mlir::arith::AddIOp>(op)) return AscendCBinaryOpKind::Add;
    if (llvm::isa<mlir::arith::SubFOp, mlir::arith::SubIOp>(op)) return AscendCBinaryOpKind::Sub;
    if (llvm::isa<mlir::arith::MulFOp, mlir::arith::MulIOp>(op)) return AscendCBinaryOpKind::Mul;
    if (llvm::isa<mlir::arith::DivFOp, mlir::arith::DivSIOp>(op)) return AscendCBinaryOpKind::Div;
    return AscendCBinaryOpKind::Unknown;
}

#endif

// 生成最简单的for循环。
std::string BuildDefaultFallbackLoopBody(const std::vector<std::string>& inLocalVars,
                                         const std::vector<std::string>& outLocalVars) {
    (void)inLocalVars;
    std::string body;
    if (!outLocalVars.empty()) {
        body += outLocalVars[0] + "Local.SetValue(i, " + outLocalVars[0] + "Local.GetValue(i));";
        return body;
    }
    body += "/* no writable output tensor, keep empty fallback */";
    return body;
}

// scalar标量回退。
AscendCPlanStep MakeScalarFallbackStep(const std::string& reason,
                                       const std::vector<std::string>& inLocalVars,
                                       const std::vector<std::string>& outLocalVars) {
    AscendCPlanStep step;
    step.kind = AscendCPlanStepKind::ScalarFallback;
    step.family = CalcSemanticFamily::ScalarFallbackFamily;
    step.transferPolicy = AscendCTransferPolicy::ScalarFallback;
    step.reason = reason;
    step.loopBody = BuildDefaultFallbackLoopBody(inLocalVars, outLocalVars);
    return step;
}

// 功能：处理Attach Family Payload To Step相关操作。
void AttachFamilyPayloadToStep(const CalcSemanticFinding& finding,
                               AscendCPlanStep& step) {
    step.family = finding.family;
    step.hasPointwisePayload = finding.hasPointwisePayload;
    step.pointwisePayload = finding.pointwisePayload;
    step.hasFamilyPayload = finding.hasFamilyPayload;
    step.familyPayload = finding.familyPayload;
    step.hasCubeMatmulPayload = finding.hasCubeMatmulPayload;
    step.cubeMatmulPayload = finding.cubeMatmulPayload;
}

AscendCBoundaryPolicy ConvertBoundaryPolicy(CalcBoundaryPolicy policy) {
    switch (policy) {
    case CalcBoundaryPolicy::Replicate:
        return AscendCBoundaryPolicy::Replicate;
    case CalcBoundaryPolicy::Clamp:
        return AscendCBoundaryPolicy::Clamp;
    case CalcBoundaryPolicy::RequireHalo:
        return AscendCBoundaryPolicy::RequireHalo;
    }
    return AscendCBoundaryPolicy::Clamp;
}

AscendCBoundaryPolicy BuildBoundaryPolicyForPlan(const std::string& tensor,
                                                  const CalcShellSummary* shellSummary) {
    return ConvertBoundaryPolicy(BuildCalcBoundaryPolicy(tensor, shellSummary));
}

std::string ExprToString(const CalcExprGraph& graph, int nodeId) {
    bool ok = true;
    std::unordered_set<int> visiting;
    return RenderExprCommon(graph, nodeId, nullptr, ok, visiting, 0);
}

// 功能：处理Merge Coeff Product相关操作。
std::string MergeCoeffProduct(const std::string& a, const std::string& b) {
    if (a == "1") {
        return b;
    }
    if (b == "1") {
        return a;
    }
    return a + "*" + b;
}

// 功能：处理Merge Coeff Add相关操作。
std::string MergeCoeffAdd(const std::string& a, const std::string& b) {
    if (a.empty()) {
        return b;
    }
    if (b.empty()) {
        return a;
    }
    if (!b.empty() && b[0] == '-') {
        return a + b;
    }
    return a + "+" + b;
}

// 功能：处理Apply Sign相关操作。
std::string ApplySign(const std::string& coeffExpr, int sign) {
    if (sign >= 0) {
        return coeffExpr;
    }
    if (coeffExpr == "1") {
        return "-1";
    }
    return "-(" + coeffExpr + ")";
}

// 功能：尝试处理Try Extract Tensor Term相关逻辑。
bool TryExtractTensorTerm(const CalcExprGraph& graph,
                          int nodeId,
                          const std::string& tensor,
                          int& outOffset,
                          std::string& outCoeff) {
    if (nodeId < 0 || static_cast<size_t>(nodeId) >= graph.nodes.size()) {
        return false;
    }

    const CalcExprNode& node = graph.nodes[static_cast<size_t>(nodeId)];
    if (node.kind == CalcExprNodeKind::Load && node.isArgTensor && node.token == tensor) {
        outOffset = node.offsetKnown ? node.offset : 0;
        outCoeff = "1";
        return true;
    }

    if (node.kind == CalcExprNodeKind::Mul && node.inputs.size() >= 2) {
        int lhsOffset = 0;
        std::string lhsCoeff;
        bool lhsTerm = TryExtractTensorTerm(graph, node.inputs[0], tensor, lhsOffset, lhsCoeff);
        bool rhsContainsTensor = ContainsTensorLoad(graph, node.inputs[1], tensor);

        if (lhsTerm && !rhsContainsTensor) {
            outOffset = lhsOffset;
            outCoeff = MergeCoeffProduct(lhsCoeff, ExprToString(graph, node.inputs[1]));
            return true;
        }

        int rhsOffset = 0;
        std::string rhsCoeff;
        bool rhsTerm = TryExtractTensorTerm(graph, node.inputs[1], tensor, rhsOffset, rhsCoeff);
        bool lhsContainsTensor = ContainsTensorLoad(graph, node.inputs[0], tensor);

        if (rhsTerm && !lhsContainsTensor) {
            outOffset = rhsOffset;
            outCoeff = MergeCoeffProduct(rhsCoeff, ExprToString(graph, node.inputs[0]));
            return true;
        }
    }

    return false;
}

// 功能：收集Collect Window Coeff Terms相关数据。
void CollectWindowCoeffTerms(const CalcExprGraph& graph,
                             int nodeId,
                             const std::string& tensor,
                             int sign,
                             std::map<int, std::string>& coeffByOffset) {
    if (nodeId < 0 || static_cast<size_t>(nodeId) >= graph.nodes.size()) {
        return;
    }

    const CalcExprNode& node = graph.nodes[static_cast<size_t>(nodeId)];
    if (node.kind == CalcExprNodeKind::Add && node.inputs.size() >= 2) {
        CollectWindowCoeffTerms(graph, node.inputs[0], tensor, sign, coeffByOffset);
        CollectWindowCoeffTerms(graph, node.inputs[1], tensor, sign, coeffByOffset);
        return;
    }

    if (node.kind == CalcExprNodeKind::Sub && node.inputs.size() >= 2) {
        CollectWindowCoeffTerms(graph, node.inputs[0], tensor, sign, coeffByOffset);
        CollectWindowCoeffTerms(graph, node.inputs[1], tensor, -sign, coeffByOffset);
        return;
    }

    int offset = 0;
    std::string coeffExpr;
    if (!TryExtractTensorTerm(graph, nodeId, tensor, offset, coeffExpr)) {
        return;
    }

    coeffExpr = ApplySign(coeffExpr, sign);
    coeffByOffset[offset] = MergeCoeffAdd(coeffByOffset[offset], coeffExpr);
}

// 功能：查找Find Store Node相关数据。
const CalcExprNode* FindStoreNode(const CalcExprGraph& graph,
                                  const std::string& target) {
    for (int storeId : graph.storeNodeIds) {
        if (storeId < 0 || static_cast<size_t>(storeId) >= graph.nodes.size()) {
            continue;
        }
        const CalcExprNode& node = graph.nodes[static_cast<size_t>(storeId)];
        if (node.token == target) {
            return &node;
        }
    }
    return nullptr;
}

// 功能：收集Collect Input Load Node Ids Dfs相关数据。
void CollectInputLoadNodeIdsDfs(const CalcExprGraph& graph,
                                int nodeId,
                                const std::string& preferredTensor,
                                std::unordered_set<int>& visited,
                                std::unordered_set<int>& seenLoadIds,
                                std::vector<int>& outLoadIds) {
    if (nodeId < 0 || static_cast<size_t>(nodeId) >= graph.nodes.size()) {
        return;
    }
    if (visited.find(nodeId) != visited.end()) {
        return;
    }
    visited.insert(nodeId);

    const CalcExprNode& node = graph.nodes[static_cast<size_t>(nodeId)];
    if (node.kind == CalcExprNodeKind::Load && node.isArgTensor) {
        if (preferredTensor.empty() || node.token == preferredTensor) {
            if (seenLoadIds.find(nodeId) == seenLoadIds.end()) {
                seenLoadIds.insert(nodeId);
                outLoadIds.push_back(nodeId);
            }
        }
    }

    for (int inputId : node.inputs) {
        CollectInputLoadNodeIdsDfs(graph,
                                   inputId,
                                   preferredTensor,
                                   visited,
                                   seenLoadIds,
                                   outLoadIds);
    }
}

// 功能：收集Collect Input Load Node Ids相关数据。
std::vector<int> CollectInputLoadNodeIds(const CalcExprGraph& graph,
                                         int rootNodeId,
                                         const std::string& preferredTensor) {
    std::vector<int> loadIds;
    std::unordered_set<int> visited;
    std::unordered_set<int> seenLoadIds;
    CollectInputLoadNodeIdsDfs(graph,
                               rootNodeId,
                               preferredTensor,
                               visited,
                               seenLoadIds,
                               loadIds);
    return loadIds;
}

#if defined(DACPP_TRANSLATOR_HAS_MLIR)

AscendCPlanStep BuildClampStep(const CalcSemanticFinding& finding,
                               const std::vector<StoreStmt>& stores,
                               const std::vector<std::string>& shellParamOrder,
                               const AscendCPlanBuildOptions& options,
                               const std::vector<std::string>& inLocalVars,
                               const std::vector<std::string>& outLocalVars);

// 处理最简单的 C = A + B 等二元运算。
AscendCPlanStep BuildElementwiseStep(const CalcSemanticFinding& finding,
                                     const std::vector<StoreStmt>& stores,
                                     const std::vector<std::string>& shellParamOrder,
                                     const AscendCPlanBuildOptions& options,
                                     const std::vector<std::string>& inLocalVars,
                                     const std::vector<std::string>& outLocalVars) {
    const StoreStmt* store = FindStoreForTarget(stores, finding.target);
    if (!store) {
        return MakeScalarFallbackStep("elementwise plan build failed: store not found", inLocalVars, outLocalVars);
    }

    mlir::Value binaryValue = FindRootBinaryValue(store->value);
    if (!binaryValue) {
        return MakeScalarFallbackStep("elementwise plan build failed: binary op root not found", inLocalVars, outLocalVars);
    }

    mlir::Operation* binOp = binaryValue.getDefiningOp();
    if (!binOp || binOp->getNumOperands() < 2) {
        return MakeScalarFallbackStep("elementwise plan build failed: malformed binary op", inLocalVars, outLocalVars);
    }

    mlir::Value lhs = ResolveLoadedMemref(binOp->getOperand(0));
    mlir::Value rhs = ResolveLoadedMemref(binOp->getOperand(1));
    if (!lhs || !rhs) {
        return MakeScalarFallbackStep("elementwise plan build failed: source tensors unresolved", inLocalVars, outLocalVars);
    }

    AscendCPlanStep step;
    step.kind = AscendCPlanStepKind::ElementwiseBinary;
    step.transferPolicy = AscendCTransferPolicy::PointwiseDirect;
    step.binaryOp = ToBinaryKind(binOp);
    step.dst = store->resolvedToken;
    step.srcA = ResolveTensorNameFromValue(lhs, shellParamOrder, options.shellSummary);
    step.srcB = ResolveTensorNameFromValue(rhs, shellParamOrder, options.shellSummary);
    step.boundaryPolicy = BuildBoundaryPolicyForPlan(step.srcA, options.shellSummary);
    step.reason = finding.reason;
    AttachFamilyPayloadToStep(finding, step);
    return step;
}

// 功能：为Build Decay Step构建所需数据。
AscendCPlanStep BuildDecayStep(const CalcSemanticFinding& finding,
                               const std::vector<StoreStmt>& stores,
                               const std::vector<std::string>& shellParamOrder,
                               const AscendCPlanBuildOptions& options,
                               const std::vector<std::string>& inLocalVars,
                               const std::vector<std::string>& outLocalVars) {
    const StoreStmt* store = FindStoreForTarget(stores, finding.target);
    if (!store) {
        return MakeScalarFallbackStep("decay plan build failed: store not found", inLocalVars, outLocalVars);
    }

    std::vector<mlir::Value> argLoads = CollectLoadedArgMemrefs(store->value);
    if (argLoads.size() < 3) {
        return MakeScalarFallbackStep("decay plan build failed: expect at least 3 loaded arg tensors", inLocalVars, outLocalVars);
    }

    AscendCPlanStep step;
    step.kind = AscendCPlanStepKind::DecayExpMul;
    step.transferPolicy = AscendCTransferPolicy::PointwiseDirect;
    step.dst = store->resolvedToken;
    step.srcA = ResolveTensorNameFromValue(argLoads[0], shellParamOrder, options.shellSummary);
    step.srcB = ResolveTensorNameFromValue(argLoads[1], shellParamOrder, options.shellSummary);
    step.srcC = ResolveTensorNameFromValue(argLoads[2], shellParamOrder, options.shellSummary);
    step.boundaryPolicy = BuildBoundaryPolicyForPlan(step.srcA, options.shellSummary);
    step.reason = finding.reason;
    AttachFamilyPayloadToStep(finding, step);

    step.scalarTensor = step.srcC;
    if (step.hasFamilyPayload && !step.srcC.empty()) {
        auto& scalarInputs = step.familyPayload.scalarInputs;
        const bool exists = std::find(scalarInputs.begin(), scalarInputs.end(), step.srcC) != scalarInputs.end();
        if (!exists) {
            scalarInputs.push_back(step.srcC);
        }
        if (step.familyPayload.dstTensor.empty()) {
            step.familyPayload.dstTensor = step.dst;
        }
    }
    return step;
}

// 处理更复杂的逐点数学公式
AscendCPlanStep BuildDirectPointwiseExprStep(const CalcSemanticFinding& finding,
                                             const std::vector<StoreStmt>& stores,
                                             const std::vector<std::string>& shellParamOrder,
                                             const AscendCPlanBuildOptions& options,
                                             const CalcAnalysisResult& analysis,
                                             const std::vector<std::string>& inLocalVars,
                                             const std::vector<std::string>& outLocalVars) {
    const StoreStmt* store = FindStoreForTarget(stores, finding.target);
    if (!store) {
        return MakeScalarFallbackStep("pointwise expr plan build failed: store not found", inLocalVars, outLocalVars);
    }

    const CalcExprNode* storeNode = FindStoreNode(analysis.exprGraph, store->resolvedToken);
    if (!storeNode || storeNode->inputs.empty()) {
        return MakeScalarFallbackStep(
            "pointwise expr plan build failed: expression root not found",
            inLocalVars,
            outLocalVars);
    }

    const int rhsRootNodeId = storeNode->inputs.front();
    std::vector<int> inputNodeIds = CollectInputLoadNodeIds(analysis.exprGraph, rhsRootNodeId, std::string());
    if (inputNodeIds.empty()) {
        return MakeScalarFallbackStep(
            "pointwise expr plan build failed: input loads unresolved",
            inLocalVars,
            outLocalVars);
    }

    AscendCPlanStep step;
    step.kind = AscendCPlanStepKind::PointwiseExpr;
    step.transferPolicy = AscendCTransferPolicy::PointwiseDirect;
    step.dst = store->resolvedToken;
    if (!finding.inputs.empty()) {
        step.srcA = ResolveTensorNameFromToken(finding.inputs.front().tensor, shellParamOrder, options.shellSummary);
        if (finding.inputs.size() > 1) {
            step.srcB = ResolveTensorNameFromToken(finding.inputs[1].tensor, shellParamOrder, options.shellSummary);
        }
    }
    if (step.srcA.empty() && !inputNodeIds.empty()) {
        const CalcExprNode& firstLoad = analysis.exprGraph.nodes[static_cast<size_t>(inputNodeIds.front())];
        const std::string tensorKey =
            firstLoad.canonicalTensor.empty() ? firstLoad.token : firstLoad.canonicalTensor;
        step.srcA = tensorKey;
    }
    step.boundaryPolicy = BuildBoundaryPolicyForPlan(step.srcA, options.shellSummary);
    step.reason = finding.reason;
    step.inputNodeIds = inputNodeIds;
    step.exprRootNodeId = rhsRootNodeId;
    step.exprRole = "pointwise-expr";
    AttachFamilyPayloadToStep(finding, step);
    return step;
}

// 处理高级模板运算（如各种滤波、差分）
AscendCPlanStep BuildCoeffStencilStep(const CalcSemanticFinding& finding,
                                      const std::vector<StoreStmt>& stores,
                                      const std::vector<std::string>& shellParamOrder,
                                      const AscendCPlanBuildOptions& options,
                                      const CalcAnalysisResult& analysis,
                                      const std::vector<std::string>& inLocalVars,
                                      const std::vector<std::string>& outLocalVars) {
    (void)analysis;

    if (finding.family != CalcSemanticFamily::SpatialStencilFamily) {
        return MakeScalarFallbackStep(
            "coeff-stencil plan build failed: only SpatialStencilFamily can use BuildCoeffStencilStep",
            inLocalVars,
            outLocalVars);
    }

    if (!finding.hasStructuredMultiViewExpr) {
        return MakeScalarFallbackStep(
            "coeff-stencil plan build failed: finding has no structured multi-view payload",
            inLocalVars,
            outLocalVars);
    }

    const StructuredMultiViewExprPattern& pattern = finding.structuredMultiViewExpr;
    const StoreStmt* store = FindStoreForTarget(stores, finding.target);
    if (!store) {
        return MakeScalarFallbackStep("coeff-stencil plan build failed: store not found", inLocalVars, outLocalVars);
    }

    AscendCPlanStep step;
    step.kind = AscendCPlanStepKind::CoeffStencil;
    step.transferPolicy = AscendCTransferPolicy::SpatialWindow;
    step.structuredMultiViewExpr = pattern;
    step.dst = store->resolvedToken.empty()
        ? ResolveTensorNameFromToken(pattern.dstTensor, shellParamOrder, options.shellSummary)
        : store->resolvedToken;
    step.srcA = ResolveTensorNameFromToken(pattern.srcTensor, shellParamOrder, options.shellSummary);

    for (int offset : pattern.viewOffsets) {
        step.windowOffsets.push_back(offset);
    }

    for (size_t i = 0; i < pattern.viewOffsets.size(); ++i) {
        step.windowCoeffs.push_back("1");
    }

    if (!pattern.scalarInputs.empty()) {
        step.scalarTensor = ResolveTensorNameFromToken(
            pattern.scalarInputs.front(),
            shellParamOrder,
            options.shellSummary);
    }
    step.srcB = step.scalarTensor;

    step.stencilRadius = 0;
    for (int offset : pattern.viewOffsets) {
        step.stencilRadius = std::max(step.stencilRadius, std::abs(offset));
    }
    step.leftHalo = pattern.leftHalo;
    step.rightHalo = pattern.rightHalo;
    step.validLength = pattern.validLength;
    step.copyLength = pattern.copyLength;
    step.boundaryPolicy = ConvertBoundaryPolicy(pattern.boundaryPolicy);
    step.reason = finding.reason;
    AttachFamilyPayloadToStep(finding, step);
    return step;
}

// 功能：为Build Pointwise Family Step构建所需数据。
AscendCPlanStep BuildPointwiseFamilyStep(const CalcSemanticFinding& finding,
                                         const std::vector<StoreStmt>& stores,
                                         const std::vector<std::string>& shellParamOrder,
                                         const AscendCPlanBuildOptions& options,
                                         const CalcAnalysisResult& analysis,
                                         const std::vector<std::string>& inLocalVars,
                                         const std::vector<std::string>& outLocalVars) {
    if (finding.family != CalcSemanticFamily::PointwiseFamily) {
        return MakeScalarFallbackStep(
            "pointwise family builder received non-pointwise finding",
            inLocalVars,
            outLocalVars);
    }

    switch (finding.category) {
    case CalcSemanticCategory::ElementwiseBinary:
        return BuildElementwiseStep(
            finding,
            stores,
            shellParamOrder,
            options,
            inLocalVars,
            outLocalVars);
    case CalcSemanticCategory::DecayExpMul:
        return BuildDecayStep(
            finding,
            stores,
            shellParamOrder,
            options,
            inLocalVars,
            outLocalVars);
    case CalcSemanticCategory::PointwiseExpr:
        return BuildDirectPointwiseExprStep(
            finding,
            stores,
            shellParamOrder,
            options,
            analysis,
            inLocalVars,
            outLocalVars);
    case CalcSemanticCategory::Clamp:
        return BuildClampStep(
            finding,
            stores,
            shellParamOrder,
            options,
            inLocalVars,
            outLocalVars);
    default:
        return MakeScalarFallbackStep(
            "pointwise family builder cannot lower this category: " +
                std::string(ToString(finding.category)),
            inLocalVars,
            outLocalVars);
    }
}

// 功能：为Build Temporal Stencil Family Step构建所需数据。
AscendCPlanStep BuildTemporalStencilFamilyStep(const CalcSemanticFinding& finding,
                                               const std::vector<StoreStmt>& stores,
                                               const std::vector<std::string>& shellParamOrder,
                                               const AscendCPlanBuildOptions& options,
                                               const std::vector<std::string>& inLocalVars,
                                               const std::vector<std::string>& outLocalVars) {
    if (finding.family != CalcSemanticFamily::TemporalStencilFamily) {
        return MakeScalarFallbackStep(
            "temporal family builder received non-temporal finding",
            inLocalVars,
            outLocalVars);
    }

    if (!finding.hasStructuredMultiViewExpr) {
        return MakeScalarFallbackStep(
            "temporal family build failed: missing structured multi-view payload",
            inLocalVars,
            outLocalVars);
    }

    const StoreStmt* store = FindStoreForTarget(stores, finding.target);
    if (!store) {
        return MakeScalarFallbackStep(
            "temporal family build failed: store not found",
            inLocalVars,
            outLocalVars);
    }

    const StructuredMultiViewExprPattern& pattern = finding.structuredMultiViewExpr;
    AscendCPlanStep step;
    step.kind = AscendCPlanStepKind::CoeffStencil;
    step.transferPolicy = AscendCTransferPolicy::TemporalLoopCarried;
    step.structuredMultiViewExpr = pattern;
    step.dst = store->resolvedToken.empty()
        ? ResolveTensorNameFromToken(pattern.dstTensor, shellParamOrder, options.shellSummary)
        : store->resolvedToken;
    step.srcA = ResolveTensorNameFromToken(pattern.srcTensor, shellParamOrder, options.shellSummary);

    for (int offset : pattern.viewOffsets) {
        step.windowOffsets.push_back(offset);
        step.windowCoeffs.push_back("1");
    }
    if (!pattern.scalarInputs.empty()) {
        step.scalarTensor = ResolveTensorNameFromToken(
            pattern.scalarInputs.front(),
            shellParamOrder,
            options.shellSummary);
        step.srcB = step.scalarTensor;
    }
    step.stencilRadius = 0;
    for (int offset : pattern.viewOffsets) {
        step.stencilRadius = std::max(step.stencilRadius, std::abs(offset));
    }
    step.leftHalo = pattern.leftHalo;
    step.rightHalo = pattern.rightHalo;
    step.validLength = pattern.validLength;
    step.copyLength = pattern.copyLength;
    step.boundaryPolicy = ConvertBoundaryPolicy(pattern.boundaryPolicy);
    step.reason = finding.reason;
    AttachFamilyPayloadToStep(finding, step);
    return step;
}

// 功能：为Build Cube Matmul Step构建所需数据。
AscendCPlanStep BuildCubeMatmulStep(const CalcSemanticFinding& finding,
                                    const std::vector<StoreStmt>& stores,
                                    const std::vector<std::string>& shellParamOrder,
                                    const AscendCPlanBuildOptions& options,
                                    const std::vector<std::string>& inLocalVars,
                                    const std::vector<std::string>& outLocalVars) {
    if (finding.family != CalcSemanticFamily::CubeMatmulFamily) {
        return MakeScalarFallbackStep(
            "cube-matmul family builder received non-cube-matmul finding",
            inLocalVars,
            outLocalVars);
    }

    if (!finding.hasCubeMatmulPayload) {
        return MakeScalarFallbackStep(
            "cube-matmul family build failed: missing cube matmul payload",
            inLocalVars,
            outLocalVars);
    }

    const CubeMatmulPayload& payload = finding.cubeMatmulPayload;
    const StoreStmt* store = FindStoreForTarget(stores, finding.target);

    AscendCPlanStep step;
    step.kind = AscendCPlanStepKind::CubeMatmul;
    step.transferPolicy = AscendCTransferPolicy::CubeMatmul;
    step.srcA = ResolveTensorNameFromToken(payload.srcA, shellParamOrder, options.shellSummary);
    step.srcB = ResolveTensorNameFromToken(payload.srcB, shellParamOrder, options.shellSummary);
    if (store && !store->resolvedToken.empty()) {
        step.dst = store->resolvedToken;
    } else {
        step.dst = ResolveTensorNameFromToken(payload.dst, shellParamOrder, options.shellSummary);
    }
    step.srcC = step.dst;
    step.boundaryPolicy = AscendCBoundaryPolicy::Clamp;
    step.reason = finding.reason;
    AttachFamilyPayloadToStep(finding, step);
    step.hasCubeMatmulPayload = true;
    step.cubeMatmulPayload = payload;
    step.cubeMatmulPayload.srcA = step.srcA;
    step.cubeMatmulPayload.srcB = step.srcB;
    step.cubeMatmulPayload.dst = step.dst;

    CubeMatmulStepPayload stepPayload;
    stepPayload.srcA = step.srcA;
    stepPayload.srcB = step.srcB;
    stepPayload.dst = step.dst;
    stepPayload.accumulateToDst = payload.accumulateToDst;
    stepPayload.hasInitialDst = payload.hasInitialDst || payload.accumulateToDst;
    stepPayload.transA = false;
    stepPayload.transB = false;

    stepPayload.userTypeA = payload.logicalInputType.empty() ? "float" : payload.logicalInputType;
    stepPayload.userTypeB = payload.logicalInputType.empty() ? "float" : payload.logicalInputType;
    stepPayload.userTypeC = payload.kernelOutputType.empty() ? stepPayload.userTypeA : payload.kernelOutputType;
    stepPayload.kernelTypeA = payload.kernelInputType.empty() ? stepPayload.userTypeA : payload.kernelInputType;
    stepPayload.kernelTypeB = payload.kernelInputType.empty() ? stepPayload.userTypeB : payload.kernelInputType;
    stepPayload.kernelTypeC = payload.kernelOutputType.empty() ? stepPayload.userTypeC : payload.kernelOutputType;

    auto AssignDimIfEmpty = [&](std::string& dim,
                                std::string& dimSource,
                                const std::string& value,
                                const std::string& source) {
        if (value.empty()) {
            return;
        }
        if (dim.empty()) {
            dim = value;
            dimSource = source;
            return;
        }
        if (dim != value) {
            if (!dimSource.empty()) {
                dimSource += "|";
            }
            dimSource += source + "(mismatch:" + value + ")";
        }
    };

    int a0 = 0;
    int a1 = 0;
    int b0 = 0;
    int b1 = 0;
    int c0 = 0;
    int c1 = 0;
    std::string a0Source;
    std::string a1Source;
    std::string b0Source;
    std::string b1Source;
    std::string c0Source;
    std::string c1Source;

    const bool hasA = TryGetShape2DFromShell(options.shellSummary, step.srcA,
                                               a0, a1, &a0Source, &a1Source);
    const bool hasB = TryGetShape2DFromShell(options.shellSummary, step.srcB,
                                               b0, b1, &b0Source, &b1Source);
    const bool hasC = TryGetShape2DFromShell(options.shellSummary, step.dst,
                                               c0, c1, &c0Source, &c1Source);

    if (hasA) {
        AssignDimIfEmpty(stepPayload.dimM, stepPayload.dimMSource,
                         std::to_string(a0), a0Source);
        AssignDimIfEmpty(stepPayload.dimK, stepPayload.dimKSource,
                         std::to_string(a1), a1Source);
    }
    if (hasB) {
        AssignDimIfEmpty(stepPayload.dimK, stepPayload.dimKSource,
                         std::to_string(b0), b0Source);
        AssignDimIfEmpty(stepPayload.dimN, stepPayload.dimNSource,
                         std::to_string(b1), b1Source);
    }
    if (hasC) {
        AssignDimIfEmpty(stepPayload.dimM, stepPayload.dimMSource,
                         std::to_string(c0), c0Source);
        AssignDimIfEmpty(stepPayload.dimN, stepPayload.dimNSource,
                         std::to_string(c1), c1Source);
    }

    if (stepPayload.dimM.empty()) {
        stepPayload.dimMSource = "unresolved";
    }
    if (stepPayload.dimN.empty()) {
        stepPayload.dimNSource = "unresolved";
    }
    if (stepPayload.dimK.empty()) {
        stepPayload.dimKSource = "unresolved";
    }

    step.hasCubeMatmulStepPayload = true;
    step.cubeMatmulStepPayload = stepPayload;
    if (stepPayload.accumulateToDst) {
        if (!step.reason.empty()) {
            step.reason += "; ";
        }
        step.reason += "preserve dot-product accumulation semantics (A*B + C_init)";
    }
    return step;
}

// 功能：为Build Structured Shifted Map Update Step构建所需数据。
AscendCPlanStep BuildStructuredShiftedMapUpdateStep(const CalcSemanticFinding& finding,
                                                    const std::vector<StoreStmt>& stores,
                                                    const std::vector<std::string>& shellParamOrder,
                                                    const AscendCPlanBuildOptions& options,
                                                    const std::vector<std::string>& inLocalVars,
                                                    const std::vector<std::string>& outLocalVars) {
    if (!finding.hasStructuredShiftedMapUpdate) {
        return MakeScalarFallbackStep(
            "structured-shifted-map-update plan build failed: finding has no structured payload",
            inLocalVars,
            outLocalVars);
    }

    const StructuredShiftedMapUpdatePattern& pattern = finding.structuredShiftedMapUpdate;
    AscendCPlanStep step;
    step.kind = AscendCPlanStepKind::StructuredShiftedMapUpdate;
    step.transferPolicy = AscendCTransferPolicy::SpatialWindow;
    step.structuredShiftedMapUpdate = pattern;

    const StoreStmt* store = FindStoreForTarget(stores, finding.target);
    if (store) {
        step.dst = store->resolvedToken;
    }
    if (step.dst.empty()) {
        step.dst = !pattern.dstTensor.empty()
            ? ResolveTensorNameFromToken(pattern.dstTensor, shellParamOrder, options.shellSummary)
            : finding.target;
    }

    step.srcA = ResolveTensorNameFromToken(pattern.srcTensor, shellParamOrder, options.shellSummary);
    step.boundaryPolicy = ConvertBoundaryPolicy(pattern.boundaryPolicy);

    step.windowOffsets.push_back(0);
    step.windowOffsets.push_back(pattern.shiftOffset);
    step.windowCoeffs.push_back("1");
    step.windowCoeffs.push_back("1");

    if (pattern.shiftOffset < 0) {
        step.leftHalo = std::max(step.leftHalo, -pattern.shiftOffset);
    } else if (pattern.shiftOffset > 0) {
        step.rightHalo = std::max(step.rightHalo, pattern.shiftOffset);
    }
    step.stencilRadius = std::abs(pattern.shiftOffset);
    step.reason = finding.reason;
    AttachFamilyPayloadToStep(finding, step);
    return step;
}

// 处理类似 B[i] = A[i+1] + 5 这种带简单偏移的运算。
std::vector<AscendCPlanStep> BuildShiftedElementwiseStep(const CalcSemanticFinding& finding,
                                                         const std::vector<StoreStmt>& stores,
                                                         const std::vector<std::string>& shellParamOrder,
                                                         const AscendCPlanBuildOptions& options,
                                                         const CalcAnalysisResult& analysis,
                                                         const std::vector<std::string>& inLocalVars,
                                                         const std::vector<std::string>& outLocalVars) {
    std::vector<AscendCPlanStep> steps;

    const StoreStmt* store = FindStoreForTarget(stores, finding.target);
    if (!store) {
        steps.push_back(MakeScalarFallbackStep("shifted-elementwise plan build failed: store not found", inLocalVars, outLocalVars));
        return steps;
    }

    const CalcExprNode* storeNode = FindStoreNode(analysis.exprGraph, store->resolvedToken);
    if (!storeNode || storeNode->inputs.empty()) {
        steps.push_back(MakeScalarFallbackStep(
            "shifted-elementwise plan build failed: expression root not found",
            inLocalVars,
            outLocalVars));
        return steps;
    }

    const int rhsRootNodeId = storeNode->inputs.front();
    if (rhsRootNodeId < 0 || static_cast<size_t>(rhsRootNodeId) >= analysis.exprGraph.nodes.size()) {
        steps.push_back(MakeScalarFallbackStep(
            "shifted-elementwise plan build failed: invalid expression root node",
            inLocalVars,
            outLocalVars));
        return steps;
    }

    std::string srcTensor;
    std::set<int> shiftOffsets;
    if (!finding.inputs.empty()) {
        srcTensor = ResolveTensorNameFromToken(finding.inputs.front().tensor, shellParamOrder, options.shellSummary);
        for (const AffineAccess& input : finding.inputs) {
            shiftOffsets.insert(input.offset);
        }
    }

    std::vector<int> inputNodeIds = CollectInputLoadNodeIds(analysis.exprGraph, rhsRootNodeId, srcTensor);
    if (inputNodeIds.empty()) {
        // 如果未命中主张量，降级收集 RHS 的全部输入 load 节点。
        inputNodeIds = CollectInputLoadNodeIds(analysis.exprGraph, rhsRootNodeId, std::string());
    }

    for (int nodeId : inputNodeIds) {
        const CalcExprNode& node = analysis.exprGraph.nodes[static_cast<size_t>(nodeId)];
        if (node.offsetKnown) {
            shiftOffsets.insert(node.offset);
        }
    }

    AscendCPlanStep windowLoad;
    windowLoad.kind = AscendCPlanStepKind::WindowLoad;
    windowLoad.transferPolicy = AscendCTransferPolicy::SpatialWindow;
    windowLoad.dst = store->resolvedToken;
    windowLoad.srcA = srcTensor;
    windowLoad.boundaryPolicy = BuildBoundaryPolicyForPlan(srcTensor, options.shellSummary);
    windowLoad.reason = "window load derived from expression-graph input loads";
    windowLoad.inputNodeIds = inputNodeIds;
    windowLoad.exprRootNodeId = rhsRootNodeId;
    windowLoad.exprRole = "window-load";
    for (int offset : shiftOffsets) {
        windowLoad.windowOffsets.push_back(offset);
        windowLoad.windowCoeffs.push_back("1");
    }
    int radius = 0;
    for (int offset : windowLoad.windowOffsets) {
        int absOffset = offset < 0 ? -offset : offset;
        if (absOffset > radius) {
            radius = absOffset;
        }
    }
    windowLoad.stencilRadius = radius;
    InferHaloAndLengths(options.shellSummary,
                        windowLoad.srcA,
                        shiftOffsets,
                        windowLoad.leftHalo,
                        windowLoad.rightHalo,
                        windowLoad.validLength,
                        windowLoad.copyLength);
    AttachFamilyPayloadToStep(finding, windowLoad);
    steps.push_back(windowLoad);

    AscendCPlanStep pointwiseExpr;
    pointwiseExpr.kind = AscendCPlanStepKind::PointwiseExpr;
    pointwiseExpr.transferPolicy = AscendCTransferPolicy::SpatialWindow;
    pointwiseExpr.dst = store->resolvedToken;
    pointwiseExpr.srcA = srcTensor;
    pointwiseExpr.boundaryPolicy = BuildBoundaryPolicyForPlan(srcTensor, options.shellSummary);
    pointwiseExpr.reason = finding.reason;
    pointwiseExpr.inputNodeIds = inputNodeIds;
    pointwiseExpr.exprRootNodeId = rhsRootNodeId;
    pointwiseExpr.exprRole = "expr-root";
    pointwiseExpr.leftHalo = windowLoad.leftHalo;
    pointwiseExpr.rightHalo = windowLoad.rightHalo;
    pointwiseExpr.validLength = windowLoad.validLength;
    pointwiseExpr.copyLength = windowLoad.copyLength;
    AttachFamilyPayloadToStep(finding, pointwiseExpr);
    steps.push_back(pointwiseExpr);

    const CalcExprNode& rhsRoot = analysis.exprGraph.nodes[static_cast<size_t>(rhsRootNodeId)];
    if (rhsRoot.kind == CalcExprNodeKind::Clamp) {
        AscendCPlanStep clampZero;
        clampZero.kind = AscendCPlanStepKind::ClampZero;
        clampZero.transferPolicy = AscendCTransferPolicy::SpatialWindow;
        clampZero.dst = store->resolvedToken;
        clampZero.srcA = store->resolvedToken;
        clampZero.boundaryPolicy = pointwiseExpr.boundaryPolicy;
        clampZero.reason = "preserve clamp-zero semantics from expression root";
        clampZero.inputNodeIds = inputNodeIds;
        clampZero.exprRootNodeId = rhsRootNodeId;
        clampZero.exprRole = "clamp-zero";
        clampZero.leftHalo = pointwiseExpr.leftHalo;
        clampZero.rightHalo = pointwiseExpr.rightHalo;
        clampZero.validLength = pointwiseExpr.validLength;
        clampZero.copyLength = pointwiseExpr.copyLength;
        AttachFamilyPayloadToStep(finding, clampZero);
        steps.push_back(clampZero);
    }

    return steps;
}

// 边界独立处理。
AscendCPlanStep BuildClampStep(const CalcSemanticFinding& finding,
                               const std::vector<StoreStmt>& stores,
                               const std::vector<std::string>& shellParamOrder,
                               const AscendCPlanBuildOptions& options,
                               const std::vector<std::string>& inLocalVars,
                               const std::vector<std::string>& outLocalVars) {
    const StoreStmt* store = FindStoreForTarget(stores, finding.target);
    if (!store) {
        return MakeScalarFallbackStep("clamp plan build failed: store not found", inLocalVars, outLocalVars);
    }

    AscendCPlanStep step;
    step.kind = AscendCPlanStepKind::Clamp;
    step.transferPolicy = AscendCTransferPolicy::PointwiseDirect;
    step.tempTensor = "Clamp";
    step.dst = store->resolvedToken;
    if (!finding.inputs.empty()) {
        step.srcA = ResolveTensorNameFromToken(finding.inputs.front().tensor, shellParamOrder, options.shellSummary);
    } else {
        step.srcA = step.dst;
    }
    step.boundaryPolicy = BuildBoundaryPolicyForPlan(step.srcA, options.shellSummary);
    step.reason = finding.reason;
    AttachFamilyPayloadToStep(finding, step);
    return step;
}

// 功能：为Build Linear Reduction Step构建所需数据。
AscendCPlanStep BuildLinearReductionStep(const CalcSemanticFinding& finding,
                                         const std::vector<std::string>& inLocalVars,
                                         const std::vector<std::string>& outLocalVars) {
    AscendCPlanStep step;
    step.kind = AscendCPlanStepKind::LinearReduction;
    step.family = CalcSemanticFamily::LinearReductionFamily;
    step.transferPolicy = AscendCTransferPolicy::ScalarFallback;
    step.reason = finding.reason;

    if (finding.hasLinearReductionPayload) {
        const LinearReductionPayload& p = finding.linearReductionPayload;
        step.hasLinearReductionPayload = true;
        step.linearReductionPayload.inputTensor = p.inputTensor;
        step.linearReductionPayload.weightTensor = p.weightTensor;
        step.linearReductionPayload.biasTensor = p.biasTensor;
        step.linearReductionPayload.outputTensor = p.outputTensor;
        step.linearReductionPayload.K = p.K;
        step.linearReductionPayload.hasBias = p.hasBias;
        step.srcA = p.inputTensor;
        step.srcB = p.weightTensor;
        step.srcC = p.biasTensor;
        step.dst = p.outputTensor;
    }

    (void)inLocalVars;
    (void)outLocalVars;
    return step;
}

#endif

} // namespace

// 功能：渲染To String相关文本。
const char* ToString(AscendCPlanStepKind kind) {
    switch (kind) {
    case AscendCPlanStepKind::StructuredShiftedMapUpdate:
        return "StructuredShiftedMapUpdate";
    case AscendCPlanStepKind::ElementwiseBinary:
        return "ElementwiseBinary";
    case AscendCPlanStepKind::DecayExpMul:
        return "DecayExpMul";
    case AscendCPlanStepKind::CoeffStencil:
        return "CoeffStencil";
    case AscendCPlanStepKind::ShiftedElementwise:
        return "ShiftedElementwise";
    case AscendCPlanStepKind::WindowLoad:
        return "WindowLoad";
    case AscendCPlanStepKind::PointwiseExpr:
        return "PointwiseExpr";
    case AscendCPlanStepKind::ClampZero:
        return "ClampZero";
    case AscendCPlanStepKind::Shift:
        return "Shift";
    case AscendCPlanStepKind::Q:
        return "Q";
    case AscendCPlanStepKind::ShiftedQ:
        return "ShiftedQ";
    case AscendCPlanStepKind::Diff:
        return "Diff";
    case AscendCPlanStepKind::Scale:
        return "Scale";
    case AscendCPlanStepKind::Sub:
        return "Sub";
    case AscendCPlanStepKind::Clamp:
        return "Clamp";
    case AscendCPlanStepKind::CubeMatmul:
        return "CubeMatmul";
    case AscendCPlanStepKind::LinearReduction:
        return "LinearReduction";
    case AscendCPlanStepKind::ScalarFallback:
        return "ScalarFallback";
    }
    return "Unknown";
}

// 功能：渲染To String相关文本。
const char* ToString(AscendCBinaryOpKind kind) {
    switch (kind) {
    case AscendCBinaryOpKind::Add:
        return "Add";
    case AscendCBinaryOpKind::Sub:
        return "Sub";
    case AscendCBinaryOpKind::Mul:
        return "Mul";
    case AscendCBinaryOpKind::Div:
        return "Div";
    case AscendCBinaryOpKind::Unknown:
        return "Unknown";
    }
    return "Unknown";
}

// 功能：渲染To String相关文本。
const char* ToString(AscendCBoundaryPolicy policy) {
    switch (policy) {
    case AscendCBoundaryPolicy::Replicate:
        return "Replicate";
    case AscendCBoundaryPolicy::Clamp:
        return "Clamp";
    case AscendCBoundaryPolicy::RequireHalo:
        return "RequireHalo";
    }
    return "Clamp";
}

// 功能：渲染To String相关文本。
const char* ToString(AscendCTransferPolicy policy) {
    switch (policy) {
    case AscendCTransferPolicy::PointwiseDirect:
        return "PointwiseDirect";
    case AscendCTransferPolicy::SpatialWindow:
        return "SpatialWindow";
    case AscendCTransferPolicy::TemporalLoopCarried:
        return "TemporalLoopCarried";
    case AscendCTransferPolicy::BroadcastScalar:
        return "BroadcastScalar";
    case AscendCTransferPolicy::CubeMatmul:
        return "CubeMatmul";
    case AscendCTransferPolicy::ScalarFallback:
        return "ScalarFallback";
    }
    return "ScalarFallback";
}

// 功能：为Build Ascend CPlan构建所需数据。
AscendCPlan BuildAscendCPlan(const CalcAnalysisResult& analysis,
                             const CalcMlirArtifact& artifact,
                             const std::vector<std::string>& shellParamOrder,
                             const std::vector<std::string>& inLocalVars,
                             const std::vector<std::string>& outLocalVars,
                             const AscendCPlanBuildOptions& options) {
    AscendCPlan plan;
    plan.reusable = true;

    for (const std::string& note : analysis.notes) {
        plan.notes.push_back(note);
    }

#if defined(DACPP_TRANSLATOR_HAS_MLIR)
    std::vector<StoreStmt> stores;
    if (artifact.valid && artifact.module) {
        mlir::ModuleOp module = const_cast<mlir::ModuleOp&>(artifact.module);
        if (!module) {
            plan.notes.push_back("mlir artifact module operation is null for plan building");
        } else {
            module.walk([&](mlir::memref::StoreOp storeOp) {
                if (!IsArgMemref(storeOp.getMemRef())) {
                    return;
                }

                StoreStmt stmt;
                stmt.value = storeOp.getValueToStore();
                stmt.memref = storeOp.getMemRef();
                stmt.token = ValueToken(storeOp.getMemRef());
                stmt.resolvedToken = ResolveTensorNameFromValue(storeOp.getMemRef(), shellParamOrder, options.shellSummary);
                stores.push_back(std::move(stmt));
            });
        }
    } else {
        plan.notes.push_back("mlir artifact is invalid for plan building");
    }

    for (const CalcSemanticFinding& finding : analysis.findings) {
        switch (finding.family) {
        case CalcSemanticFamily::PointwiseFamily:
            plan.steps.push_back(BuildPointwiseFamilyStep(
                finding,
                stores,
                shellParamOrder,
                options,
                analysis,
                inLocalVars,
                outLocalVars));
            break;
        case CalcSemanticFamily::SpatialStencilFamily:
            if (finding.category == CalcSemanticCategory::StructuredShiftedMapUpdate) {
                plan.steps.push_back(BuildStructuredShiftedMapUpdateStep(
                    finding,
                    stores,
                    shellParamOrder,
                    options,
                    inLocalVars,
                    outLocalVars));
            } else if (finding.category == CalcSemanticCategory::CoeffStencil) {
                plan.steps.push_back(BuildCoeffStencilStep(
                    finding,
                    stores,
                    shellParamOrder,
                    options,
                    analysis,
                    inLocalVars,
                    outLocalVars));
            } else if (finding.category == CalcSemanticCategory::WindowedPointwise ||
                       finding.category == CalcSemanticCategory::ShiftedElementwise) {
                std::vector<AscendCPlanStep> staged = BuildShiftedElementwiseStep(
                    finding,
                    stores,
                    shellParamOrder,
                    options,
                    analysis,
                    inLocalVars,
                    outLocalVars);
                for (AscendCPlanStep& step : staged) {
                    plan.steps.push_back(std::move(step));
                }
            } else {
                plan.steps.push_back(MakeScalarFallbackStep(
                    "spatial family category is not lowerable: " +
                        std::string(ToString(finding.category)),
                    inLocalVars,
                    outLocalVars));
            }
            break;
        case CalcSemanticFamily::TemporalStencilFamily:
            plan.steps.push_back(BuildTemporalStencilFamilyStep(
                finding,
                stores,
                shellParamOrder,
                options,
                inLocalVars,
                outLocalVars));
            break;
        case CalcSemanticFamily::CubeMatmulFamily:
            plan.steps.push_back(BuildCubeMatmulStep(
                finding,
                stores,
                shellParamOrder,
                options,
                inLocalVars,
                outLocalVars));
            break;
        case CalcSemanticFamily::ReductionFamily:
            plan.steps.push_back(MakeScalarFallbackStep(
                "reduction currently falls back to scalar emit path: " + finding.reason,
                inLocalVars,
                outLocalVars));
            break;
        case CalcSemanticFamily::LinearReductionFamily:
            plan.steps.push_back(BuildLinearReductionStep(finding, inLocalVars, outLocalVars));
            break;
        case CalcSemanticFamily::ScalarFallbackFamily:
            plan.steps.push_back(MakeScalarFallbackStep(
                finding.reason,
                inLocalVars,
                outLocalVars));
            break;
        }
    }
#else
    (void)artifact;
    (void)options;
    for (const CalcSemanticFinding& finding : analysis.findings) {
        plan.steps.push_back(MakeScalarFallbackStep(finding.reason, inLocalVars, outLocalVars));
    }
#endif

    if (plan.steps.empty()) {
        plan.steps.push_back(MakeScalarFallbackStep(
            "analysis produced no semantic finding; use scalar fallback",
            inLocalVars, outLocalVars));
    }

    return plan;
}

// 功能：渲染Dump Ascend CPlan相关文本。
std::string DumpAscendCPlan(const AscendCPlan& plan) {
    std::ostringstream oss;
    oss << "AscendCPlan(reusable=" << (plan.reusable ? "true" : "false") << ")\n";
    for (size_t i = 0; i < plan.steps.size(); ++i) {
        const AscendCPlanStep& step = plan.steps[i];
        oss << "  [" << i << "] kind=" << ToString(step.kind)
            << " family=" << ToString(step.family)
            << " transfer=" << ToString(step.transferPolicy);

        if (step.kind == AscendCPlanStepKind::ElementwiseBinary) {
            oss << " op=" << ToString(step.binaryOp);
        }
        if (step.kind == AscendCPlanStepKind::StructuredShiftedMapUpdate) {
            oss << " shift=" << step.structuredShiftedMapUpdate.shiftOffset
                << " hasMap=" << (step.structuredShiftedMapUpdate.hasMap ? "true" : "false")
                << " hasPost=" << (step.structuredShiftedMapUpdate.hasPost ? "true" : "false");
        }
        if (step.kind == AscendCPlanStepKind::CoeffStencil &&
            !step.structuredMultiViewExpr.mainExpr.empty()) {
            oss << " views=[";
            for (size_t j = 0; j < step.structuredMultiViewExpr.viewOffsets.size(); ++j) {
                if (j != 0) {
                    oss << ",";
                }
                oss << step.structuredMultiViewExpr.viewOffsets[j];
            }
            oss << "]";
        }
        if (step.kind == AscendCPlanStepKind::CubeMatmul &&
            step.hasCubeMatmulPayload) {
            oss << " matmul(accumulateToDst="
                << (step.cubeMatmulPayload.accumulateToDst ? "true" : "false")
                << ",hasInitialDst="
                << (step.cubeMatmulPayload.hasInitialDst ? "true" : "false")
                << ")";
        }

        if (!step.dst.empty()) {
            oss << " dst=" << step.dst;
        }
        if (!step.srcA.empty()) {
            oss << " srcA=" << step.srcA;
        }
        if (!step.srcB.empty()) {
            oss << " srcB=" << step.srcB;
        }
        if (!step.srcC.empty()) {
            oss << " srcC=" << step.srcC;
        }
        if (!step.tempTensor.empty()) {
            oss << " temp=" << step.tempTensor;
        }
        if (step.leftHalo > 0 || step.rightHalo > 0) {
            oss << " halo=(L" << step.leftHalo << ",R" << step.rightHalo << ")";
        }
        if (step.validLength > 0) {
            oss << " validLength=" << step.validLength;
        }
        if (step.copyLength > 0) {
            oss << " copyLength=" << step.copyLength;
        }
        if (!step.inputNodeIds.empty()) {
            oss << " inputNodeIds=[";
            for (size_t j = 0; j < step.inputNodeIds.size(); ++j) {
                if (j != 0) {
                    oss << ",";
                }
                oss << step.inputNodeIds[j];
            }
            oss << "]";
        }
        if (step.exprRootNodeId >= 0) {
            oss << " exprRoot=" << step.exprRootNodeId;
        }
        if (!step.exprRole.empty()) {
            oss << " exprRole=" << step.exprRole;
        }

        if (!step.windowOffsets.empty()) {
            oss << " windows=[";
            for (size_t j = 0; j < step.windowOffsets.size(); ++j) {
                if (j != 0) {
                    oss << ",";
                }
                oss << step.windowOffsets[j];
            }
            oss << "]";
        }

        if (!step.windowCoeffs.empty()) {
            oss << " coeffs=[";
            for (size_t j = 0; j < step.windowCoeffs.size(); ++j) {
                if (j != 0) {
                    oss << ",";
                }
                oss << step.windowCoeffs[j];
            }
            oss << "]";
        }

        if (!step.scalarTensor.empty()) {
            oss << " scalarTensor=" << step.scalarTensor;
        }
        if (std::abs(step.scalarImm) > 1e-12) {
            oss << " scalarImm=" << step.scalarImm;
        }
        if (step.hasFamilyPayload) {
            const CalcFamilyPayload& payload = step.familyPayload;
            oss << " payload.src=" << payload.srcTensor
                << " payload.dst=" << payload.dstTensor
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
            oss << "] loop-carried=" << (payload.loopCarried ? "true" : "false");
            if (!payload.mainExpr.empty()) {
                oss << " mainExpr=" << payload.mainExpr;
            }
            if (payload.hasPostExpr) {
                oss << " postExpr=" << payload.postExpr;
            }
        }
        if (step.hasCubeMatmulPayload) {
            const CubeMatmulPayload& payload = step.cubeMatmulPayload;
            oss << " matmul.srcA=" << payload.srcA
                << " matmul.srcB=" << payload.srcB
                << " matmul.dst=" << payload.dst
                << " matmul.logicalInputType=" << payload.logicalInputType
                << " matmul.kernelInputType=" << payload.kernelInputType
                << " matmul.kernelOutputType=" << payload.kernelOutputType;
        }
        if (step.hasCubeMatmulStepPayload) {
            const CubeMatmulStepPayload& payload = step.cubeMatmulStepPayload;
            oss << " cube.srcA=" << payload.srcA
                << " cube.srcB=" << payload.srcB
                << " cube.dst=" << payload.dst
                << " cube.accumulateToDst=" << (payload.accumulateToDst ? "true" : "false")
                << " cube.M=" << payload.dimM
                << " cube.N=" << payload.dimN
                << " cube.K=" << payload.dimK
                << " cube.M.source=" << payload.dimMSource
                << " cube.N.source=" << payload.dimNSource
                << " cube.K.source=" << payload.dimKSource
                << " cube.transA=" << (payload.transA ? "true" : "false")
                << " cube.transB=" << (payload.transB ? "true" : "false")
                << " cube.userTypeA=" << payload.userTypeA
                << " cube.userTypeB=" << payload.userTypeB
                << " cube.userTypeC=" << payload.userTypeC
                << " cube.kernelTypeA=" << payload.kernelTypeA
                << " cube.kernelTypeB=" << payload.kernelTypeB
                << " cube.kernelTypeC=" << payload.kernelTypeC;
        }
        if (step.kind == AscendCPlanStepKind::StructuredShiftedMapUpdate) {
            oss << " mapExpr=" << step.structuredShiftedMapUpdate.mapExpr
                << " updateExpr=" << step.structuredShiftedMapUpdate.updateExpr;
            if (step.structuredShiftedMapUpdate.hasPost) {
                oss << " postExpr=" << step.structuredShiftedMapUpdate.postExpr;
            }
        }
        if (step.kind == AscendCPlanStepKind::CoeffStencil &&
            !step.structuredMultiViewExpr.mainExpr.empty()) {
            if (!step.structuredMultiViewExpr.scalarInputs.empty()) {
                oss << " scalarInputs=[";
                for (size_t j = 0; j < step.structuredMultiViewExpr.scalarInputs.size(); ++j) {
                    if (j != 0) {
                        oss << ",";
                    }
                    oss << step.structuredMultiViewExpr.scalarInputs[j];
                }
                oss << "]";
            }
            oss << " mainExpr=" << step.structuredMultiViewExpr.mainExpr;
            if (step.structuredMultiViewExpr.hasPost) {
                oss << " postExpr=" << step.structuredMultiViewExpr.postExpr;
            }
        }
        oss << " boundary=" << ToString(step.boundaryPolicy);

        if (!step.reason.empty()) {
            oss << " reason=\"" << step.reason << "\"";
        }
        oss << "\n";
    }

    if (!plan.notes.empty()) {
        oss << "  notes:\n";
        for (const std::string& note : plan.notes) {
            oss << "    - " << note << "\n";
        }
    }

    return oss.str();
}

} // namespace ascendc_mlir
} // namespace dacppTranslator
