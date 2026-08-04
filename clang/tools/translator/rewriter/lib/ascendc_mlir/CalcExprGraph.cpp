/*
语义提炼器
在将 AST 降级为 MLIR 后，虽然代码变成了标准的中间表示，
但 MLIR 的操作（Op）通常非常底层且繁杂（包含了大量类型转换、索引计算等）。
这份代码的作用是：扫描 MLIR 模块，从所有的内存写入（Store）操作开始往回溯，
将底层的 MLIR 指令流提取成一个高层级的、纯数学的“表达式计算图”（Expression Graph/DAG）。
这个计算图摒弃了底层的控制流和类型细节，只关注“数据是怎么算出来的”，
这对于后续识别计算模式（比如是普通的逐点运算 Pointwise，还是带偏移的模板运算 Stencil）至关重要。
*/
#include "ascendc_mlir/CalcExprGraph.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "ascendc_mlir/CalcMlirUtils.h"
#include "ascendc_mlir/CalcShellInfo.h"

#if defined(DACPP_TRANSLATOR_HAS_MLIR)
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#endif

namespace dacppTranslator {
namespace ascendc_mlir {
namespace {

#if defined(DACPP_TRANSLATOR_HAS_MLIR)
//表示一次内存写入操作。记录了要写入的值（value）、目标内存（memref）以及写入的索引/坐标（indices）。
struct StoreStmt {
    mlir::Value value;
    mlir::Value memref;
    std::vector<mlir::Value> indices;
};
//表示一个数组索引表达式。它试图将复杂的索引计算简化为 base + offset（基准变量 + 静态偏移）的形式。
struct IndexExpr {
    bool known = false;
    mlir::Value base;
    long offset = 0;
};

static const std::vector<std::string> kEmptyParamOrder;

// 递归地评估一个 MLIR 索引计算。
// 它识别循环的归纳变量（loopIvs，如 i），
// 并沿着加减法（AddIOp, SubIOp）和常数（ConstantIndexOp）追踪。
IndexExpr EvalIndexExpr(mlir::Value value,
                        const llvm::DenseSet<mlir::Value>& loopIvs,
                        llvm::DenseSet<mlir::Value>& visiting) {
    IndexExpr unknown;
    if (!value) {
        return unknown;
    }

    if (loopIvs.contains(value)) {
        return IndexExpr{true, value, 0};
    }

    if (visiting.contains(value)) {
        return unknown;
    }
    visiting.insert(value);

    if (auto cidx = value.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
        visiting.erase(value);
        return IndexExpr{true, mlir::Value(), static_cast<long>(cidx.value())};
    }

    if (auto cint = value.getDefiningOp<mlir::arith::ConstantIntOp>()) {
        visiting.erase(value);
        return IndexExpr{true, mlir::Value(), static_cast<long>(cint.value())};
    }

    if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>()) {
        IndexExpr inner = EvalIndexExpr(cast.getIn(), loopIvs, visiting);
        visiting.erase(value);
        return inner;
    }

    if (auto add = value.getDefiningOp<mlir::arith::AddIOp>()) {
        IndexExpr lhs = EvalIndexExpr(add.getLhs(), loopIvs, visiting);
        IndexExpr rhs = EvalIndexExpr(add.getRhs(), loopIvs, visiting);
        visiting.erase(value);

        if (!lhs.known || !rhs.known) {
            return unknown;
        }
        if (lhs.base && rhs.base && lhs.base != rhs.base) {
            return unknown;
        }

        mlir::Value base = lhs.base ? lhs.base : rhs.base;
        return IndexExpr{true, base, lhs.offset + rhs.offset};
    }

    if (auto sub = value.getDefiningOp<mlir::arith::SubIOp>()) {
        IndexExpr lhs = EvalIndexExpr(sub.getLhs(), loopIvs, visiting);
        IndexExpr rhs = EvalIndexExpr(sub.getRhs(), loopIvs, visiting);
        visiting.erase(value);

        if (!lhs.known || !rhs.known) {
            return unknown;
        }
        if (rhs.base) {
            return unknown;
        }

        return IndexExpr{true, lhs.base, lhs.offset - rhs.offset};
    }

    visiting.erase(value);
    return unknown;
}

// 功能：处理Eval Index Expr相关逻辑。
IndexExpr EvalIndexExpr(mlir::Value value,
                        const llvm::DenseSet<mlir::Value>& loopIvs) {
    llvm::DenseSet<mlir::Value> visiting;
    return EvalIndexExpr(value, loopIvs, visiting);
}

// 对比“写入目标的索引”和“读取数据源的索引”。
// 如果把数据写到 A[i]，但数据来源于 B[i+2]，这个函数就会捕捉到相对偏移量 offset = 2。
// 这是后续生成 AscendC 数据搬移（Data Copy）或向量偏移指令的关键前提。
void CaptureOffset(const StoreStmt& store,
                   mlir::memref::LoadOp load,
                   const llvm::DenseSet<mlir::Value>& loopIvs,
                   bool& offsetKnown,
                   int& offset) {
    offsetKnown = false;
    offset = 0;

    if (!load || store.indices.empty() || load.getIndices().empty()) {
        return;
    }

    IndexExpr lhs = EvalIndexExpr(store.indices.front(), loopIvs);
    IndexExpr rhs = EvalIndexExpr(load.getIndices().front(), loopIvs);
    if (!lhs.known || !rhs.known) {
        return;
    }
    if (lhs.base != rhs.base) {
        return;
    }

    offsetKnown = true;
    offset = static_cast<int>(rhs.offset - lhs.offset);
}

// 尝试从不同的 MLIR 常量节点（浮点、整数、索引，甚至被强制类型转换过的常量）中提取出一个通用的 double 类型的标量值。
bool TryGetConstScalar(mlir::Value value, double& outValue) {
    if (!value) {
        return false;
    }

    if (auto cf = value.getDefiningOp<mlir::arith::ConstantFloatOp>()) {
        outValue = cf.value().convertToDouble();
        return true;
    }
    if (auto ci = value.getDefiningOp<mlir::arith::ConstantIntOp>()) {
        outValue = static_cast<double>(ci.value());
        return true;
    }
    if (auto cidx = value.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
        outValue = static_cast<double>(cidx.value());
        return true;
    }
    if (auto cast = value.getDefiningOp<mlir::arith::SIToFPOp>()) {
        return TryGetConstScalar(cast.getIn(), outValue);
    }
    if (auto cast = value.getDefiningOp<mlir::arith::ExtSIOp>()) {
        return TryGetConstScalar(cast.getIn(), outValue);
    }
    if (auto cast = value.getDefiningOp<mlir::arith::ExtUIOp>()) {
        return TryGetConstScalar(cast.getIn(), outValue);
    }
    if (auto cast = value.getDefiningOp<mlir::arith::TruncIOp>()) {
        return TryGetConstScalar(cast.getIn(), outValue);
    }
    if (auto cast = value.getDefiningOp<mlir::arith::ExtFOp>()) {
        return TryGetConstScalar(cast.getIn(), outValue);
    }
    if (auto cast = value.getDefiningOp<mlir::arith::TruncFOp>()) {
        return TryGetConstScalar(cast.getIn(), outValue);
    }
    if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>()) {
        return TryGetConstScalar(cast.getIn(), outValue);
    }

    return false;
}

// 功能：检查Is Zero Value相关条件。
bool IsZeroValue(mlir::Value value) {
    double c = 0.0;
    if (!TryGetConstScalar(value, c)) {
        return false;
    }
    return std::abs(c) < 1e-12;
}

// 列出了一系列被视为“直接透传”的类型转换操作（如整型扩展、浮点截断）。
// 在构建高层纯数学计算图时，这些底层类型转换节点会被直接跳过（折叠）。
bool IsPassthroughCast(mlir::Operation* op) {
    return llvm::isa<mlir::arith::SIToFPOp,
                     mlir::arith::ExtSIOp,
                     mlir::arith::ExtUIOp,
                     mlir::arith::TruncIOp,
                     mlir::arith::ExtFOp,
                     mlir::arith::TruncFOp,
                     mlir::arith::IndexCastOp>(op);
}

class ExprGraphBuilder {
public:
    // 功能：处理Expr Graph Builder相关逻辑。
    ExprGraphBuilder(const llvm::DenseSet<mlir::Value>& loopIvs,
                     const llvm::DenseMap<mlir::Value, mlir::Value>& memrefSourceValue,
                     const std::vector<std::string>* paramOrder,
                     const CalcShellSummary* shellSummary,
                     CalcExprGraph& graph)
                : loopIvs_(loopIvs),
                    // 功能：处理memref Source Value相关逻辑。
                    memrefSourceValue_(memrefSourceValue),
                    // 功能：处理param Order相关逻辑。
                    paramOrder_(paramOrder),
                    // 功能：处理shell Summary相关逻辑。
                    shellSummary_(shellSummary),
                    // 功能：处理graph相关逻辑。
                    graph_(graph) {}

    // 接收一个内存写入操作，从等号右边的值开始触发递归构建。
    int BuildForStoreRhs(const StoreStmt& store) {
        currentStore_ = &store;
        int id = BuildValueNode(store.value);
        currentStore_ = nullptr;
        return id;
    }

private:
    // 功能：处理Add Node相关逻辑。
    int AddNode(CalcExprNode node) {
        node.id = static_cast<int>(graph_.nodes.size());
        graph_.nodes.push_back(std::move(node));
        return graph_.nodes.back().id;
    }

    // 核心递归函数。
    int BuildValueNode(mlir::Value value) {
        if (!value) {
            CalcExprNode node;
            node.kind = CalcExprNodeKind::Unknown;
            node.token = "<null>";
            return AddNode(std::move(node));
        }

        auto it = valueNodeMap_.find(value);
        if (it != valueNodeMap_.end()) {
            return it->second;
        }

        mlir::Operation* op = value.getDefiningOp();
        if (!op) {
            CalcExprNode node;
            node.kind = CalcExprNodeKind::Unknown;
            node.token = ValueToken(value);
            int id = AddNode(std::move(node));
            valueNodeMap_[value] = id;
            return id;
        }

        if (IsPassthroughCast(op) && op->getNumOperands() == 1) {
            int foldedId = BuildValueNode(op->getOperand(0));
            valueNodeMap_[value] = foldedId;
            return foldedId;
        }

        CalcExprNode node;

        if (auto load = llvm::dyn_cast<mlir::memref::LoadOp>(op)) {
            if (!IsArgMemref(load.getMemRef())) {
                auto it = memrefSourceValue_.find(load.getMemRef());
                if (it != memrefSourceValue_.end() && it->second && it->second != value) {
                    int foldedId = BuildValueNode(it->second);
                    valueNodeMap_[value] = foldedId;
                    return foldedId;
                }
            }

            node.kind = CalcExprNodeKind::Load;
            node.isArgTensor = IsArgMemref(load.getMemRef());
            node.token = ResolveTensorNameFromValue(load.getMemRef(),
                paramOrder_ ? *paramOrder_ : kEmptyParamOrder, shellSummary_);
            if (node.isArgTensor) {
                node.canonicalTensor = node.token;
            }
            if (currentStore_) {
                CaptureOffset(*currentStore_, load, loopIvs_, node.offsetKnown, node.offset);
            }
            node.isWindowBase = node.isArgTensor && node.offsetKnown;
            int id = AddNode(std::move(node));
            valueNodeMap_[value] = id;
            return id;
        }

        if (auto cst = llvm::dyn_cast<mlir::arith::ConstantOp>(op)) {
            node.kind = CalcExprNodeKind::Const;
            double scalar = 0.0;
            if (TryGetConstScalar(cst.getResult(), scalar)) {
                node.hasConstValue = true;
                node.constValue = scalar;
                std::ostringstream oss;
                oss << scalar;
                node.token = oss.str();
            } else {
                node.token = cst.getValue().getAbstractAttribute().getDialect().getNamespace().str();
            }
            int id = AddNode(std::move(node));
            valueNodeMap_[value] = id;
            return id;
        }

        if (llvm::isa<mlir::arith::AddFOp, mlir::arith::AddIOp>(op)) {
            node.kind = CalcExprNodeKind::Add;
        } else if (llvm::isa<mlir::arith::SubFOp, mlir::arith::SubIOp>(op)) {
            node.kind = CalcExprNodeKind::Sub;
        } else if (llvm::isa<mlir::arith::MulFOp, mlir::arith::MulIOp>(op)) {
            node.kind = CalcExprNodeKind::Mul;
        } else if (llvm::isa<mlir::arith::DivFOp, mlir::arith::DivSIOp>(op)) {
            node.kind = CalcExprNodeKind::Div;
        } else if (llvm::isa<mlir::math::ExpOp, mlir::math::Exp2Op>(op)) {
            node.kind = CalcExprNodeKind::Exp;
        } else if (auto select = llvm::dyn_cast<mlir::arith::SelectOp>(op)) {
            auto FinalizeBinarySelect = [&](CalcExprNodeKind kind,
                                            mlir::Value lhs,
                                            mlir::Value rhs) {
                node.kind = kind;
                node.inputs.push_back(BuildValueNode(lhs));
                node.inputs.push_back(BuildValueNode(rhs));
                int id = AddNode(std::move(node));
                valueNodeMap_[value] = id;
                return id;
            };

            auto FinalizeClamp = [&](mlir::Value nonZeroBranch) {
                node.kind = CalcExprNodeKind::Clamp;
                node.inputs.push_back(BuildValueNode(nonZeroBranch));
                int id = AddNode(std::move(node));
                valueNodeMap_[value] = id;
                return id;
            };

            if (auto cmpf = select.getCondition().getDefiningOp<mlir::arith::CmpFOp>()) {
                const auto pred = cmpf.getPredicate();
                if (select.getTrueValue() == cmpf.getLhs() &&
                    select.getFalseValue() == cmpf.getRhs()) {
                    if (pred == mlir::arith::CmpFPredicate::OLT ||
                        pred == mlir::arith::CmpFPredicate::OLE) {
                        return FinalizeBinarySelect(CalcExprNodeKind::Min, cmpf.getLhs(), cmpf.getRhs());
                    }
                    if (pred == mlir::arith::CmpFPredicate::OGT ||
                        pred == mlir::arith::CmpFPredicate::OGE) {
                        const bool lhsIsZero = IsZeroValue(cmpf.getLhs());
                        const bool rhsIsZero = IsZeroValue(cmpf.getRhs());
                        if (lhsIsZero || rhsIsZero) {
                            return FinalizeClamp(lhsIsZero ? cmpf.getRhs() : cmpf.getLhs());
                        }
                        return FinalizeBinarySelect(CalcExprNodeKind::Max, cmpf.getLhs(), cmpf.getRhs());
                    }
                }
            }

            if (auto cmpi = select.getCondition().getDefiningOp<mlir::arith::CmpIOp>()) {
                const auto pred = cmpi.getPredicate();
                if (select.getTrueValue() == cmpi.getLhs() &&
                    select.getFalseValue() == cmpi.getRhs()) {
                    if (pred == mlir::arith::CmpIPredicate::slt ||
                        pred == mlir::arith::CmpIPredicate::sle ||
                        pred == mlir::arith::CmpIPredicate::ult ||
                        pred == mlir::arith::CmpIPredicate::ule) {
                        return FinalizeBinarySelect(CalcExprNodeKind::Min, cmpi.getLhs(), cmpi.getRhs());
                    }
                    if (pred == mlir::arith::CmpIPredicate::sgt ||
                        pred == mlir::arith::CmpIPredicate::sge ||
                        pred == mlir::arith::CmpIPredicate::ugt ||
                        pred == mlir::arith::CmpIPredicate::uge) {
                        const bool lhsIsZero = IsZeroValue(cmpi.getLhs());
                        const bool rhsIsZero = IsZeroValue(cmpi.getRhs());
                        if (lhsIsZero || rhsIsZero) {
                            return FinalizeClamp(lhsIsZero ? cmpi.getRhs() : cmpi.getLhs());
                        }
                        return FinalizeBinarySelect(CalcExprNodeKind::Max, cmpi.getLhs(), cmpi.getRhs());
                    }
                }
            }

            node.kind = CalcExprNodeKind::Unknown;
            node.token = op->getName().getStringRef().str();
        } else if (llvm::isa<mlir::arith::MaximumFOp, mlir::arith::MaxSIOp, mlir::arith::MaxUIOp>(op)) {
            node.kind = CalcExprNodeKind::Max;
        } else {
            node.kind = CalcExprNodeKind::Unknown;
            node.token = op->getName().getStringRef().str();
        }

        for (mlir::Value operand : op->getOperands()) {
            node.inputs.push_back(BuildValueNode(operand));
        }

        int id = AddNode(std::move(node));
        valueNodeMap_[value] = id;
        return id;
    }

    const llvm::DenseSet<mlir::Value>& loopIvs_;
    const llvm::DenseMap<mlir::Value, mlir::Value>& memrefSourceValue_;
    const std::vector<std::string>* paramOrder_ = nullptr;
    const CalcShellSummary* shellSummary_;
    CalcExprGraph& graph_;

    const StoreStmt* currentStore_ = nullptr;
    llvm::DenseMap<mlir::Value, int> valueNodeMap_;
};

#endif

} // namespace

// 功能：渲染To String相关文本。
const char* ToString(CalcExprNodeKind kind) {
    switch (kind) {
    case CalcExprNodeKind::Load:
        return "Load";
    case CalcExprNodeKind::Const:
        return "Const";
    case CalcExprNodeKind::Add:
        return "Add";
    case CalcExprNodeKind::Sub:
        return "Sub";
    case CalcExprNodeKind::Mul:
        return "Mul";
    case CalcExprNodeKind::Div:
        return "Div";
    case CalcExprNodeKind::Exp:
        return "Exp";
    case CalcExprNodeKind::Min:
        return "Min";
    case CalcExprNodeKind::Max:
        return "Max";
    case CalcExprNodeKind::Clamp:
        return "Clamp";
    case CalcExprNodeKind::Store:
        return "Store";
    case CalcExprNodeKind::Unknown:
        return "Unknown";
    }
    return "Unknown";
}

// 总调度函数。
// 1. 遍历 MLIR 模块，收集所有的 for 循环变量（loopIvs）。
// 2. 扫描所有的 memref.store 写入操作。如果写入的是局部内存（非 Arg），它会记录下来用于后续的局部内存转发；
// 如果写入的是函数的输出参数，它会将这些 store 作为计算图的终点（根节点）存入 stores 列表。
// 3. 实例化 ExprGraphBuilder，遍历收集到的每一个全局 store，驱动其构建整个计算图。
CalcExprGraph BuildCalcExprGraph(const CalcMlirArtifact& artifact,
                                 const CalcShellSummary* shellSummary,
                                 const std::vector<std::string>* paramOrder) {
    CalcExprGraph graph;

#if defined(DACPP_TRANSLATOR_HAS_MLIR)
    if (!artifact.valid || !artifact.module) {
        graph.notes.push_back("mlir artifact unavailable for expression graph");
        return graph;
    }

    mlir::ModuleOp module = const_cast<mlir::ModuleOp&>(artifact.module);
    if (!module) {
        graph.notes.push_back("mlir module operation is null");
        return graph;
    }

    llvm::DenseSet<mlir::Value> loopIvs;
    module.walk([&](mlir::scf::ForOp forOp) {
        loopIvs.insert(forOp.getInductionVar());
    });

    std::vector<StoreStmt> stores;
    llvm::DenseMap<mlir::Value, mlir::Value> memrefSourceValue;
    llvm::DenseSet<mlir::Value> multiSourceMemrefs;
    module.walk([&](mlir::memref::StoreOp storeOp) {
        mlir::Value memref = storeOp.getMemRef();
        if (!IsArgMemref(memref)) {
            if (multiSourceMemrefs.contains(memref)) {
                return;
            }
            auto it = memrefSourceValue.find(memref);
            if (it == memrefSourceValue.end()) {
                memrefSourceValue[memref] = storeOp.getValueToStore();
            } else if (it->second != storeOp.getValueToStore()) {
                memrefSourceValue.erase(memref);
                multiSourceMemrefs.insert(memref);
            }
            return;
        }

        StoreStmt stmt;
        stmt.value = storeOp.getValueToStore();
        stmt.memref = memref;
        stmt.indices.assign(storeOp.getIndices().begin(), storeOp.getIndices().end());
        stores.push_back(std::move(stmt));
    });

    if (stores.empty()) {
        graph.notes.push_back("no memref.store on argument tensors");
        return graph;
    }

    ExprGraphBuilder builder(loopIvs, memrefSourceValue, paramOrder, shellSummary, graph);
    for (const StoreStmt& store : stores) {
        int rhsId = builder.BuildForStoreRhs(store);

        CalcExprNode storeNode;
        storeNode.kind = CalcExprNodeKind::Store;
        storeNode.token = ResolveTensorNameFromValue(store.memref,
            paramOrder ? *paramOrder : kEmptyParamOrder, shellSummary);
        storeNode.isArgTensor = true;
        storeNode.canonicalTensor = storeNode.token;
        storeNode.inputs.push_back(rhsId);
        int storeId = static_cast<int>(graph.nodes.size());
        storeNode.id = storeId;
        graph.nodes.push_back(std::move(storeNode));
        graph.storeNodeIds.push_back(storeId);
    }

    graph.valid = !graph.storeNodeIds.empty();
#else
    (void)artifact;
    (void)shellSummary;
    graph.notes.push_back("MLIR is not enabled in this build");
#endif

    return graph;
}

// 调试输出函数。将构建好的图结构（节点、边、偏移量、依赖关系）以纯文本的层级格式打印出来，供开发者在终端调试查看。
std::string RenderCalcExprGraph(const CalcExprGraph& graph) {
    std::ostringstream oss;
    oss << "expr-graph(valid=" << (graph.valid ? "true" : "false") << ")\n";
    for (const CalcExprNode& node : graph.nodes) {
        oss << "  [" << node.id << "] kind=" << ToString(node.kind);
        if (!node.token.empty()) {
            oss << " token=" << node.token;
        }
        if (node.kind == CalcExprNodeKind::Load && node.offsetKnown) {
            oss << " offset=" << node.offset;
        }
        if (node.kind == CalcExprNodeKind::Load && node.isWindowBase) {
            oss << " windowBase=1";
        }
        if (!node.canonicalTensor.empty()) {
            oss << " canonical=" << node.canonicalTensor;
        }
        if (node.kind == CalcExprNodeKind::Const && node.hasConstValue) {
            oss << " value=" << node.constValue;
        }
        if (!node.inputs.empty()) {
            oss << " inputs=";
            for (size_t i = 0; i < node.inputs.size(); ++i) {
                if (i != 0) {
                    oss << ",";
                }
                oss << node.inputs[i];
            }
        }
        oss << "\n";
    }

    if (!graph.notes.empty()) {
        oss << "  notes:\n";
        for (const std::string& note : graph.notes) {
            oss << "    - " << note << "\n";
        }
    }
    return oss.str();
}

} // namespace ascendc_mlir
} // namespace dacppTranslator
