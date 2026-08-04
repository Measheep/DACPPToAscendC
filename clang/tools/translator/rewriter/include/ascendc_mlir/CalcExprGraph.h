#ifndef TRANSLATOR_REWRITER_ASCENDC_MLIR_CALC_EXPR_GRAPH_H
#define TRANSLATOR_REWRITER_ASCENDC_MLIR_CALC_EXPR_GRAPH_H

#include <string>
#include <vector>

#include "ascendc_mlir/CalcAstToMlir.h"

namespace dacppTranslator {
namespace ascendc_mlir {

struct CalcShellSummary;

enum class CalcExprNodeKind {
    Load,
    Const,
    Add,
    Sub,
    Mul,
    Div,
    Exp,
    Min,
    Max,
    Clamp,
    Store,
    Unknown,
};

struct CalcExprNode {
    int id = -1;
    CalcExprNodeKind kind = CalcExprNodeKind::Unknown;
    std::string token;
    std::vector<int> inputs;

    bool isArgTensor = false;
    bool isWindowBase = false;
    bool offsetKnown = false;
    int offset = 0;
    std::string canonicalTensor;

    bool hasConstValue = false;
    double constValue = 0.0;
};

struct CalcExprGraph {
    bool valid = false;
    std::vector<CalcExprNode> nodes;
    std::vector<int> storeNodeIds;
    std::vector<std::string> notes;

    bool ValidNodeId(int id) const {
        return id >= 0 && static_cast<size_t>(id) < nodes.size();
    }
    const CalcExprNode* GetNode(int id) const {
        return ValidNodeId(id) ? &nodes[static_cast<size_t>(id)] : nullptr;
    }
};

CalcExprGraph BuildCalcExprGraph(const CalcMlirArtifact& artifact,
                                 const CalcShellSummary* shellSummary = nullptr,
                                 const std::vector<std::string>* paramOrder = nullptr);

const char* ToString(CalcExprNodeKind kind);
std::string RenderCalcExprGraph(const CalcExprGraph& graph);

} // namespace ascendc_mlir
} // namespace dacppTranslator

#endif // TRANSLATOR_REWRITER_ASCENDC_MLIR_CALC_EXPR_GRAPH_H
