/*
代码生成与API发射。
读取上一步生成的AscendCPlan，并将其翻译成最终可以被华为昇腾 NPU 编译器识别并执行的底层 C++ 向量 API 源码。
*/
#include "ascendc_mlir/AscendCEmitter.h"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace dacppTranslator {
namespace ascendc_mlir {
namespace {

enum class StructuredTokenKind {
    End,
    Ident,
    Number,
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
    Comma,
    Invalid,
};

struct StructuredToken {
    StructuredTokenKind kind = StructuredTokenKind::Invalid;
    std::string text;
    double number = 0.0;
};

enum class StructuredExprKind {
    Var,
    Const,
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Min,
    Max,
    Exp,
    Clamp,
};

struct StructuredExprNode {
    StructuredExprKind kind = StructuredExprKind::Const;
    std::string ident;
    double value = 0.0;
    StructuredExprNode* lhs = nullptr;
    StructuredExprNode* rhs = nullptr;
};
// 一个手写的递归下降解析器（词法+语法分析）。它能把公式字符串解析成一棵微型抽象语法树
class StructuredExprParser {
public:
    explicit StructuredExprParser(const std::string& source)
        : source_(source) {
        Next();
    }

    StructuredExprNode* Parse(bool& ok) {
        ok = true;
        StructuredExprNode* root = ParseExpression(ok);
        if (!ok || !root) {
            return nullptr;
        }
        if (current_.kind != StructuredTokenKind::End) {
            ok = false;
            return nullptr;
        }
        return root;
    }

private:
    // 功能：为Make Node构建所需数据。
    StructuredExprNode* MakeNode(StructuredExprKind kind) {
        nodes_.push_back(std::make_unique<StructuredExprNode>());
        nodes_.back()->kind = kind;
        return nodes_.back().get();
    }

    // 功能：处理Skip Spaces相关逻辑。
    void SkipSpaces() {
        while (pos_ < source_.size() && std::isspace(static_cast<unsigned char>(source_[pos_]))) {
            ++pos_;
        }
    }

    // 功能：处理Read Token相关逻辑。
    StructuredToken ReadToken() {
        SkipSpaces();
        if (pos_ >= source_.size()) {
            return StructuredToken{StructuredTokenKind::End, "", 0.0};
        }

        const char ch = source_[pos_];
        switch (ch) {
        case '+':
            ++pos_;
            return StructuredToken{StructuredTokenKind::Plus, "+", 0.0};
        case '-':
            ++pos_;
            return StructuredToken{StructuredTokenKind::Minus, "-", 0.0};
        case '*':
            ++pos_;
            return StructuredToken{StructuredTokenKind::Star, "*", 0.0};
        case '/':
            ++pos_;
            return StructuredToken{StructuredTokenKind::Slash, "/", 0.0};
        case '(':
            ++pos_;
            return StructuredToken{StructuredTokenKind::LParen, "(", 0.0};
        case ')':
            ++pos_;
            return StructuredToken{StructuredTokenKind::RParen, ")", 0.0};
        case ',':
            ++pos_;
            return StructuredToken{StructuredTokenKind::Comma, ",", 0.0};
        default:
            break;
        }

        if (std::isdigit(static_cast<unsigned char>(ch)) || ch == '.') {
            const size_t begin = pos_;
            char* endPtr = nullptr;
            double value = std::strtod(source_.c_str() + pos_, &endPtr);
            if (endPtr == source_.c_str() + pos_) {
                ++pos_;
                return StructuredToken{StructuredTokenKind::Invalid, std::string(1, ch), 0.0};
            }
            pos_ += static_cast<size_t>(endPtr - (source_.c_str() + pos_));
            return StructuredToken{StructuredTokenKind::Number, source_.substr(begin, pos_ - begin), value};
        }

        if (std::isalpha(static_cast<unsigned char>(ch)) || ch == '_') {
            const size_t begin = pos_;
            ++pos_;
            while (pos_ < source_.size()) {
                const char c = source_[pos_];
                if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
                    break;
                }
                ++pos_;
            }
            return StructuredToken{StructuredTokenKind::Ident, source_.substr(begin, pos_ - begin), 0.0};
        }

        ++pos_;
        return StructuredToken{StructuredTokenKind::Invalid, std::string(1, ch), 0.0};
    }

    // 功能：处理Next相关逻辑。
    void Next() {
        current_ = ReadToken();
    }

    // 功能：处理Accept相关逻辑。
    bool Accept(StructuredTokenKind kind) {
        if (current_.kind != kind) {
            return false;
        }
        Next();
        return true;
    }

    // 功能：提取或解析Parse Expression相关信息。
    StructuredExprNode* ParseExpression(bool& ok) {
        StructuredExprNode* lhs = ParseTerm(ok);
        if (!ok || !lhs) {
            return nullptr;
        }

        while (current_.kind == StructuredTokenKind::Plus ||
               current_.kind == StructuredTokenKind::Minus) {
            const StructuredTokenKind op = current_.kind;
            Next();
            StructuredExprNode* rhs = ParseTerm(ok);
            if (!ok || !rhs) {
                return nullptr;
            }
            StructuredExprNode* node = MakeNode(
                op == StructuredTokenKind::Plus ? StructuredExprKind::Add : StructuredExprKind::Sub);
            node->lhs = lhs;
            node->rhs = rhs;
            lhs = node;
        }

        return lhs;
    }

    // 功能：提取或解析Parse Term相关信息。
    StructuredExprNode* ParseTerm(bool& ok) {
        StructuredExprNode* lhs = ParseFactor(ok);
        if (!ok || !lhs) {
            return nullptr;
        }

        while (current_.kind == StructuredTokenKind::Star ||
               current_.kind == StructuredTokenKind::Slash) {
            const StructuredTokenKind op = current_.kind;
            Next();
            StructuredExprNode* rhs = ParseFactor(ok);
            if (!ok || !rhs) {
                return nullptr;
            }
            StructuredExprNode* node = MakeNode(
                op == StructuredTokenKind::Star ? StructuredExprKind::Mul : StructuredExprKind::Div);
            node->lhs = lhs;
            node->rhs = rhs;
            lhs = node;
        }

        return lhs;
    }

    // 功能：提取或解析Parse Factor相关信息。
    StructuredExprNode* ParseFactor(bool& ok) {
        if (current_.kind == StructuredTokenKind::Plus) {
            Next();
            return ParseFactor(ok);
        }
        if (current_.kind == StructuredTokenKind::Minus) {
            Next();
            StructuredExprNode* inner = ParseFactor(ok);
            if (!ok || !inner) {
                return nullptr;
            }
            StructuredExprNode* node = MakeNode(StructuredExprKind::Neg);
            node->lhs = inner;
            return node;
        }

        if (current_.kind == StructuredTokenKind::LParen) {
            Next();
            StructuredExprNode* inner = ParseExpression(ok);
            if (!ok || !inner || !Accept(StructuredTokenKind::RParen)) {
                ok = false;
                return nullptr;
            }
            return inner;
        }

        if (current_.kind == StructuredTokenKind::Number) {
            StructuredExprNode* node = MakeNode(StructuredExprKind::Const);
            node->value = current_.number;
            Next();
            return node;
        }

        if (current_.kind == StructuredTokenKind::Ident) {
            const std::string ident = current_.text;
            Next();
            if (current_.kind != StructuredTokenKind::LParen) {
                StructuredExprNode* node = MakeNode(StructuredExprKind::Var);
                node->ident = ident;
                return node;
            }

            Next();
            StructuredExprNode* firstArg = ParseExpression(ok);
            if (!ok || !firstArg) {
                return nullptr;
            }

            if (ident == "exp" || ident == "clamp") {
                if (!Accept(StructuredTokenKind::RParen)) {
                    ok = false;
                    return nullptr;
                }
                StructuredExprNode* node = MakeNode(
                    ident == "exp" ? StructuredExprKind::Exp : StructuredExprKind::Clamp);
                node->lhs = firstArg;
                return node;
            }

            if (ident == "max" || ident == "min") {
                if (!Accept(StructuredTokenKind::Comma)) {
                    ok = false;
                    return nullptr;
                }
                StructuredExprNode* secondArg = ParseExpression(ok);
                if (!ok || !secondArg || !Accept(StructuredTokenKind::RParen)) {
                    ok = false;
                    return nullptr;
                }
                StructuredExprNode* node = MakeNode(
                    ident == "min" ? StructuredExprKind::Min : StructuredExprKind::Max);
                node->lhs = firstArg;
                node->rhs = secondArg;
                return node;
            }

            ok = false;
            return nullptr;
        }

        ok = false;
        return nullptr;
    }

    std::string source_;
    size_t pos_ = 0;
    StructuredToken current_;
    std::vector<std::unique_ptr<StructuredExprNode>> nodes_;
};

// 功能：检查Is Const Node相关条件。
bool IsConstNode(const StructuredExprNode* node, double& outValue) {
    if (!node || node->kind != StructuredExprKind::Const) {
        return false;
    }
    outValue = node->value;
    return true;
}

// 提前运行常量计算。
bool FoldStructuredExprNode(StructuredExprNode* node, double* outConstValue = nullptr) {
    if (!node) {
        return false;
    }

    auto markConst = [&](double value) {
        node->kind = StructuredExprKind::Const;
        node->value = value;
        node->ident.clear();
        node->lhs = nullptr;
        node->rhs = nullptr;
        if (outConstValue) {
            *outConstValue = value;
        }
        return true;
    };

    switch (node->kind) {
    case StructuredExprKind::Const:
        if (outConstValue) {
            *outConstValue = node->value;
        }
        return true;
    case StructuredExprKind::Var:
        return false;
    case StructuredExprKind::Neg: {
        double lhs = 0.0;
        if (FoldStructuredExprNode(node->lhs, &lhs)) {
            return markConst(-lhs);
        }
        return false;
    }
    case StructuredExprKind::Exp: {
        double lhs = 0.0;
        if (FoldStructuredExprNode(node->lhs, &lhs)) {
            return markConst(std::exp(lhs));
        }
        return false;
    }
    case StructuredExprKind::Clamp: {
        double lhs = 0.0;
        if (FoldStructuredExprNode(node->lhs, &lhs)) {
            return markConst(std::max(lhs, 0.0));
        }
        return false;
    }
    case StructuredExprKind::Add:
    case StructuredExprKind::Sub:
    case StructuredExprKind::Mul:
    case StructuredExprKind::Div:
    case StructuredExprKind::Min:
    case StructuredExprKind::Max:
        break;
    }

    double lhs = 0.0;
    double rhs = 0.0;
    const bool lhsConst = FoldStructuredExprNode(node->lhs, &lhs);
    const bool rhsConst = FoldStructuredExprNode(node->rhs, &rhs);
    if (!lhsConst || !rhsConst) {
        return false;
    }

    switch (node->kind) {
    case StructuredExprKind::Add:
        return markConst(lhs + rhs);
    case StructuredExprKind::Sub:
        return markConst(lhs - rhs);
    case StructuredExprKind::Mul:
        return markConst(lhs * rhs);
    case StructuredExprKind::Div:
        if (std::abs(rhs) <= 1e-12) {
            return false;
        }
        return markConst(lhs / rhs);
    case StructuredExprKind::Min:
        return markConst(std::min(lhs, rhs));
    case StructuredExprKind::Max:
        return markConst(std::max(lhs, rhs));
    default:
        break;
    }
    return false;
}

// 缓存分配器
int RequiredScratchCount(const StructuredExprNode* node) {
    if (!node) {
        return 0;
    }

    switch (node->kind) {
    case StructuredExprKind::Var:
    case StructuredExprKind::Const:
        return 0;
    case StructuredExprKind::Neg:
    case StructuredExprKind::Exp:
        return RequiredScratchCount(node->lhs);
    case StructuredExprKind::Clamp:
        return std::max(1, RequiredScratchCount(node->lhs));
    case StructuredExprKind::Add:
    case StructuredExprKind::Sub:
    case StructuredExprKind::Mul:
    case StructuredExprKind::Div: {
        double c = 0.0;
        const bool lhsConst = IsConstNode(node->lhs, c);
        const bool rhsConst = IsConstNode(node->rhs, c);
        if (node->kind == StructuredExprKind::Add || node->kind == StructuredExprKind::Mul) {
            if (lhsConst && !rhsConst) {
                return RequiredScratchCount(node->rhs);
            }
            if (rhsConst && !lhsConst) {
                return RequiredScratchCount(node->lhs);
            }
        }
        if (node->kind == StructuredExprKind::Sub || node->kind == StructuredExprKind::Div) {
            if (rhsConst && !lhsConst) {
                return RequiredScratchCount(node->lhs);
            }
        }

        const int lhsNeed = RequiredScratchCount(node->lhs);
        const int rhsNeed = RequiredScratchCount(node->rhs);
        return std::max(lhsNeed, 1 + rhsNeed);
    }
    case StructuredExprKind::Min: {
        const int lhsNeed = RequiredScratchCount(node->lhs);
        const int rhsNeed = RequiredScratchCount(node->rhs);
        return std::max(lhsNeed, 1 + rhsNeed);
    }
    case StructuredExprKind::Max: {
        const int lhsNeed = RequiredScratchCount(node->lhs);
        const int rhsNeed = RequiredScratchCount(node->rhs);
        return std::max(lhsNeed, 1 + rhsNeed);
    }
    }
    return 0;
}

// 功能：渲染Format Float Literal相关文本。
std::string FormatFloatLiteral(double value) {
    if (std::isnan(value)) {
        return "(std::numeric_limits<float>::quiet_NaN())";
    }
    if (!std::isfinite(value)) {
        return value > 0.0
            ? "(std::numeric_limits<float>::infinity())"
            : "(-std::numeric_limits<float>::infinity())";
    }

    const float f = static_cast<float>(value);
    if (f == 0.0f) {
        return "0.0f";
    }

    std::ostringstream oss;
    oss << std::setprecision(std::numeric_limits<float>::max_digits10)
        << std::defaultfloat
        << f;
    std::string text = oss.str();

    if (!text.empty() && text[0] == '.') {
        text = "0" + text;
    } else if (text.size() > 1 && text[0] == '-' && text[1] == '.') {
        text = "-0" + text.substr(1);
    }

    const bool hasDot = text.find('.') != std::string::npos;
    const bool hasExp = text.find('e') != std::string::npos || text.find('E') != std::string::npos;
    if (!hasDot && !hasExp) {
        text += ".0";
    }
    if (!hasDot && hasExp) {
        size_t expPos = text.find('e');
        if (expPos == std::string::npos) {
            expPos = text.find('E');
        }
        if (expPos != std::string::npos) {
            text.insert(expPos, ".0");
        }
    }

    if (text == "-0.0" || text == "-0.0e+0" || text == "-0.0e0") {
        text = "0.0";
    }
    return text + "f";
}

// 功能：检查Is Integral Literal相关条件。
bool IsIntegralLiteral(double value) {
    return std::isfinite(value) && std::abs(value - std::llround(value)) <= 1e-12;
}

// 功能：渲染Format Scalar Literal相关文本。
std::string FormatScalarLiteral(double value) {
    if (IsIntegralLiteral(value)) {
        return std::to_string(static_cast<long long>(std::llround(value)));
    }
    return FormatFloatLiteral(value);
}

// 功能：为Build Typed Immediate Expr构建所需数据。
std::string BuildTypedImmediateExpr(const std::string& localTensor, double value) {
    return "static_cast<decltype(+" + localTensor + ".GetValue(0))>(" +
           FormatScalarLiteral(value) + ")";
}

// 功能：生成或转换Emit Tensor Copy相关结果。
void EmitTensorCopy(std::string& code,
                    const std::string& dst,
                    const std::string& src,
                    const std::string& tileLengthVar) {
    if (dst == src) {
        return;
    }
    code += "        AscendC::Adds(" + dst + ", " + src + ", " +
            BuildTypedImmediateExpr(dst, 0.0) + ", " + tileLengthVar + ");\n";
}

// 功能：生成或转换Emit Const Fill相关结果。
void EmitConstFill(std::string& code,
                   const std::string& dst,
                   double value,
                   const std::string& tileLengthVar) {
    code += "        AscendC::Duplicate(" + dst + ", " + BuildTypedImmediateExpr(dst, value) +
            ", " + tileLengthVar + ");\n";
}

struct StructuredExprCompileContext {
    std::string tileLengthVar;
    std::unordered_map<std::string, std::string> vars;
        std::unordered_map<std::string, std::string> scalarVars;
    std::vector<std::string> scratchLocals;
    std::vector<bool> scratchInUse;
    std::string scalarDeclCode;
    std::string code;
    bool ok = true;

    // 功能：处理Structured Expr Compile Context相关逻辑。
    explicit StructuredExprCompileContext(std::string tileLength,
                                          const std::unordered_map<std::string, std::string>& bindings,
                                                                                    const std::unordered_map<std::string, std::string>& scalarBindings,
                                          int scratchCount)
        : tileLengthVar(std::move(tileLength)),
          // 功能：处理vars相关逻辑。
          vars(bindings),
                    // 功能：处理scalar Vars相关逻辑。
                    scalarVars(scalarBindings),
          // 功能：处理scratch Locals相关逻辑。
          scratchLocals(static_cast<size_t>(scratchCount)),
          // 功能：处理scratch In Use相关逻辑。
          scratchInUse(static_cast<size_t>(scratchCount), false) {
        for (int i = 0; i < scratchCount; ++i) {
            scratchLocals[static_cast<size_t>(i)] =
                "calcBuffer" + std::to_string(i + 2) + "Local";
        }
    }

    // 功能：处理Acquire Scratch相关逻辑。
    int AcquireScratch() {
        for (size_t i = 0; i < scratchInUse.size(); ++i) {
            if (!scratchInUse[i]) {
                scratchInUse[i] = true;
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    // 功能：处理Release Scratch相关逻辑。
    void ReleaseScratch(int index) {
        if (index < 0 || static_cast<size_t>(index) >= scratchInUse.size()) {
            return;
        }
        scratchInUse[static_cast<size_t>(index)] = false;
    }
};

// 功能：生成或转换Emit Scalar Temp Decl相关结果。
std::string EmitScalarTempDecl(StructuredExprCompileContext& ctx,
                               const std::string& scalarExpr) {
    static int globalScalarTempCounter = 0;
    const std::string tempName = "scalarExprTmp" + std::to_string(globalScalarTempCounter++);
    ctx.scalarDeclCode += "        const auto " + tempName + " = (" + scalarExpr + ");\n";
    return tempName;
}

// 功能：生成或转换Emit Scalar Lowered Immediate相关结果。
std::string EmitScalarLoweredImmediate(StructuredExprCompileContext& ctx,
                                       double value) {
    return EmitScalarTempDecl(ctx, FormatScalarLiteral(value));
}

bool EmitStructuredExprNode(const StructuredExprNode* node,
                            const std::string& targetLocal,
                            StructuredExprCompileContext& ctx);

static bool EmitConstScalarOp(const StructuredExprNode* childNode,
                               double constValue,
                               const char* apiName,
                               const std::string& targetLocal,
                               StructuredExprCompileContext& ctx) {
    if (!EmitStructuredExprNode(childNode, targetLocal, ctx)) {
        return false;
    }
    const std::string scalarTemp = EmitScalarLoweredImmediate(ctx, constValue);
    ctx.code += "        AscendC::" + std::string(apiName) + "(" + targetLocal +
                ", " + targetLocal + ", static_cast<decltype(+" + targetLocal +
                ".GetValue(0))>(" + scalarTemp + "), " + ctx.tileLengthVar + ");\n";
    return true;
}

static bool EmitExprScalarOp(const StructuredExprNode* childNode,
                              const std::string& scalarExpr,
                              const char* apiName,
                              const std::string& targetLocal,
                              StructuredExprCompileContext& ctx) {
    if (childNode && !EmitStructuredExprNode(childNode, targetLocal, ctx)) {
        return false;
    }
    const std::string scalarTemp = EmitScalarTempDecl(ctx, scalarExpr);
    ctx.code += "        AscendC::" + std::string(apiName) + "(" + targetLocal +
                ", " + targetLocal + ", static_cast<decltype(+" + targetLocal +
                ".GetValue(0))>((" + scalarTemp + ")), " + ctx.tileLengthVar + ");\n";
    return true;
}

// 功能：尝试处理Try Build Scalar Expr相关逻辑。
bool TryBuildScalarExpr(const StructuredExprNode* node,
                        const StructuredExprCompileContext& ctx,
                        std::string& outExpr) {
    outExpr.clear();
    if (!node) {
        return false;
    }

    switch (node->kind) {
    case StructuredExprKind::Const:
        outExpr = FormatScalarLiteral(node->value);
        return true;
    case StructuredExprKind::Var: {
        auto it = ctx.scalarVars.find(node->ident);
        if (it == ctx.scalarVars.end() || it->second.empty()) {
            return false;
        }
        outExpr = it->second;
        return true;
    }
    case StructuredExprKind::Neg: {
        std::string inner;
        if (!TryBuildScalarExpr(node->lhs, ctx, inner)) {
            return false;
        }
        outExpr = "-(" + inner + ")";
        return true;
    }
    case StructuredExprKind::Add:
    case StructuredExprKind::Sub:
    case StructuredExprKind::Mul:
    case StructuredExprKind::Div:
    case StructuredExprKind::Min:
    case StructuredExprKind::Max:
        break;
    case StructuredExprKind::Exp:
    case StructuredExprKind::Clamp:
        return false;
    }

    std::string lhs;
    std::string rhs;
    if (!TryBuildScalarExpr(node->lhs, ctx, lhs) ||
        !TryBuildScalarExpr(node->rhs, ctx, rhs)) {
        return false;
    }

    switch (node->kind) {
    case StructuredExprKind::Add:
        outExpr = "(" + lhs + ")+(" + rhs + ")";
        return true;
    case StructuredExprKind::Sub:
        outExpr = "(" + lhs + ")-(" + rhs + ")";
        return true;
    case StructuredExprKind::Mul:
        outExpr = "(" + lhs + ")*(" + rhs + ")";
        return true;
    case StructuredExprKind::Div:
        outExpr = "(" + lhs + ")/(" + rhs + ")";
        return true;
    case StructuredExprKind::Min:
        outExpr = "std::min((" + lhs + "),(" + rhs + "))";
        return true;
    case StructuredExprKind::Max:
        outExpr = "std::max((" + lhs + "),(" + rhs + "))";
        return true;
    default:
        break;
    }
    return false;
}

// 核心生成逻辑。
// 递归遍历微型语法树，自动向代码字符串中插入API，并妥善管理临时 Buffer 的申请和释放。
bool EmitStructuredExprNode(const StructuredExprNode* node,
                            const std::string& targetLocal,
                            StructuredExprCompileContext& ctx) {
    if (!node || !ctx.ok) {
        ctx.ok = false;
        return false;
    }

    switch (node->kind) {
    case StructuredExprKind::Var: {
        auto it = ctx.vars.find(node->ident);
        if (it != ctx.vars.end() && !it->second.empty()) {
            EmitTensorCopy(ctx.code, targetLocal, it->second, ctx.tileLengthVar);
            return true;
        }

        auto scalarIt = ctx.scalarVars.find(node->ident);
        if (scalarIt == ctx.scalarVars.end() || scalarIt->second.empty()) {
            ctx.ok = false;
            return false;
        }
        EmitConstFill(ctx.code, targetLocal, 0.0, ctx.tileLengthVar);
        const std::string scalarTemp = EmitScalarTempDecl(ctx, scalarIt->second);
        ctx.code += "        AscendC::Adds(" + targetLocal + ", " + targetLocal +
                    ", static_cast<decltype(+" + targetLocal + ".GetValue(0))>((" +
                    scalarTemp + ")), " + ctx.tileLengthVar + ");\n";
        return true;
    }
    case StructuredExprKind::Const:
        EmitConstFill(ctx.code, targetLocal, node->value, ctx.tileLengthVar);
        return true;
    case StructuredExprKind::Neg:
        if (!EmitStructuredExprNode(node->lhs, targetLocal, ctx)) {
            return false;
        }
        ctx.code += "        AscendC::Muls(" + targetLocal + ", " + targetLocal + ", " +
                    BuildTypedImmediateExpr(targetLocal, -1.0) + ", " + ctx.tileLengthVar + ");\n";
        return true;
    case StructuredExprKind::Exp:
        if (!EmitStructuredExprNode(node->lhs, targetLocal, ctx)) {
            return false;
        }
        ctx.code += "        AscendC::Exp(" + targetLocal + ", " + targetLocal + ", " + ctx.tileLengthVar + ");\n";
        return true;
    case StructuredExprKind::Clamp: {
        if (!EmitStructuredExprNode(node->lhs, targetLocal, ctx)) {
            return false;
        }
        int scratchIndex = ctx.AcquireScratch();
        if (scratchIndex < 0) {
            ctx.ok = false;
            return false;
        }
        const std::string scratchLocal = ctx.scratchLocals[static_cast<size_t>(scratchIndex)];
        EmitConstFill(ctx.code, scratchLocal, 0.0, ctx.tileLengthVar);
        ctx.code += "        AscendC::Max(" + targetLocal + ", " + targetLocal + ", " + scratchLocal + ", " + ctx.tileLengthVar + ");\n";
        ctx.ReleaseScratch(scratchIndex);
        return true;
    }
    case StructuredExprKind::Add:
    case StructuredExprKind::Sub:
    case StructuredExprKind::Mul:
    case StructuredExprKind::Div:
    case StructuredExprKind::Min:
    case StructuredExprKind::Max:
        break;
    }

    double lhsConst = 0.0;
    double rhsConst = 0.0;
    const bool hasLhsConst = IsConstNode(node->lhs, lhsConst);
    const bool hasRhsConst = IsConstNode(node->rhs, rhsConst);

    if (node->kind == StructuredExprKind::Add) {
        if (hasLhsConst && !hasRhsConst)
            return EmitConstScalarOp(node->rhs, lhsConst, "Adds", targetLocal, ctx);
        if (hasRhsConst && !hasLhsConst)
            return EmitConstScalarOp(node->lhs, rhsConst, "Adds", targetLocal, ctx);
    }
    if (node->kind == StructuredExprKind::Sub && hasRhsConst && !hasLhsConst)
        return EmitConstScalarOp(node->lhs, -rhsConst, "Adds", targetLocal, ctx);
    if (node->kind == StructuredExprKind::Mul) {
        if (hasLhsConst && !hasRhsConst)
            return EmitConstScalarOp(node->rhs, lhsConst, "Muls", targetLocal, ctx);
        if (hasRhsConst && !hasLhsConst)
            return EmitConstScalarOp(node->lhs, rhsConst, "Muls", targetLocal, ctx);
    }
    if (node->kind == StructuredExprKind::Div && hasRhsConst && !hasLhsConst && std::abs(rhsConst) > 1e-12)
        return EmitConstScalarOp(node->lhs, 1.0 / rhsConst, "Muls", targetLocal, ctx);
    if (node->kind == StructuredExprKind::Min) {
        if (hasLhsConst && !hasRhsConst)
            return EmitConstScalarOp(node->rhs, lhsConst, "Mins", targetLocal, ctx);
        if (hasRhsConst && !hasLhsConst)
            return EmitConstScalarOp(node->lhs, rhsConst, "Mins", targetLocal, ctx);
    }

    std::string lhsScalarExpr;
    std::string rhsScalarExpr;
    const bool lhsIsScalarExpr = TryBuildScalarExpr(node->lhs, ctx, lhsScalarExpr);
    const bool rhsIsScalarExpr = TryBuildScalarExpr(node->rhs, ctx, rhsScalarExpr);

    if (node->kind == StructuredExprKind::Mul) {
        if (lhsIsScalarExpr && !rhsIsScalarExpr)
            return EmitExprScalarOp(node->rhs, lhsScalarExpr, "Muls", targetLocal, ctx);
        if (rhsIsScalarExpr && !lhsIsScalarExpr)
            return EmitExprScalarOp(node->lhs, rhsScalarExpr, "Muls", targetLocal, ctx);
    }
    if (node->kind == StructuredExprKind::Add) {
        if (lhsIsScalarExpr && !rhsIsScalarExpr)
            return EmitExprScalarOp(node->rhs, lhsScalarExpr, "Adds", targetLocal, ctx);
        if (rhsIsScalarExpr && !lhsIsScalarExpr)
            return EmitExprScalarOp(node->lhs, rhsScalarExpr, "Adds", targetLocal, ctx);
    }
    if (node->kind == StructuredExprKind::Sub) {
        if (rhsIsScalarExpr && !lhsIsScalarExpr)
            return EmitExprScalarOp(node->lhs, "-(" + rhsScalarExpr + ")", "Adds", targetLocal, ctx);
        if (lhsIsScalarExpr && !rhsIsScalarExpr) {
            if (!EmitStructuredExprNode(node->rhs, targetLocal, ctx))
                return false;
            ctx.code += "        AscendC::Muls(" + targetLocal + ", " + targetLocal + ", " +
                        BuildTypedImmediateExpr(targetLocal, -1.0) + ", " + ctx.tileLengthVar + ");\n";
            return EmitExprScalarOp(nullptr, lhsScalarExpr, "Adds", targetLocal, ctx);
        }
    }
    if (node->kind == StructuredExprKind::Div && rhsIsScalarExpr && !lhsIsScalarExpr)
        return EmitExprScalarOp(node->lhs, FormatFloatLiteral(1.0) + "/(" + rhsScalarExpr + ")", "Muls", targetLocal, ctx);
    if (node->kind == StructuredExprKind::Min) {
        if (lhsIsScalarExpr && !rhsIsScalarExpr)
            return EmitExprScalarOp(node->rhs, lhsScalarExpr, "Mins", targetLocal, ctx);
        if (rhsIsScalarExpr && !lhsIsScalarExpr)
            return EmitExprScalarOp(node->lhs, rhsScalarExpr, "Mins", targetLocal, ctx);
    }

    if (!EmitStructuredExprNode(node->lhs, targetLocal, ctx)) {
        return false;
    }

    int scratchIndex = ctx.AcquireScratch();
    if (scratchIndex < 0) {
        ctx.ok = false;
        return false;
    }
    const std::string scratchLocal = ctx.scratchLocals[static_cast<size_t>(scratchIndex)];
    if (!EmitStructuredExprNode(node->rhs, scratchLocal, ctx)) {
        ctx.ReleaseScratch(scratchIndex);
        return false;
    }

    const char* api = nullptr;
    switch (node->kind) {
    case StructuredExprKind::Add:
        api = "AscendC::Add";
        break;
    case StructuredExprKind::Sub:
        api = "AscendC::Sub";
        break;
    case StructuredExprKind::Mul:
        api = "AscendC::Mul";
        break;
    case StructuredExprKind::Div:
        api = "AscendC::Div";
        break;
    case StructuredExprKind::Min:
        api = "AscendC::Min";
        break;
    case StructuredExprKind::Max:
        api = "AscendC::Max";
        break;
    default:
        break;
    }
    if (!api) {
        ctx.ReleaseScratch(scratchIndex);
        ctx.ok = false;
        return false;
    }

    ctx.code += std::string("        ") + api + "(" + targetLocal + ", " + targetLocal + ", " +
                scratchLocal + ", " + ctx.tileLengthVar + ");\n";
    ctx.ReleaseScratch(scratchIndex);
    return true;
}

// 功能：生成或转换Emit Structured Expr Internal相关结果。
std::string EmitStructuredExprInternal(const std::string& expr,
                                       const std::unordered_map<std::string, std::string>& bindings,
                                       const std::unordered_map<std::string, std::string>& scalarBindings,
                                       const std::string& targetLocal,
                                       const std::string& tileLengthVar,
                                       int* outScratchBufferCount) {
    if (outScratchBufferCount) {
        *outScratchBufferCount = 0;
    }

    StructuredExprParser parser(expr);
    bool parseOk = false;
    StructuredExprNode* root = parser.Parse(parseOk);
    if (!parseOk || !root) {
        return std::string();
    }

    FoldStructuredExprNode(root);

    const int scratchCount = RequiredScratchCount(root);
    if (outScratchBufferCount) {
        *outScratchBufferCount = scratchCount;
    }

    StructuredExprCompileContext ctx(tileLengthVar, bindings, scalarBindings, scratchCount);
    if (!EmitStructuredExprNode(root, targetLocal, ctx) || !ctx.ok) {
        return std::string();
    }

    return ctx.scalarDeclCode + ctx.code;
}

// 功能：为Build Structured Scalar Bindings构建所需数据。
std::unordered_map<std::string, std::string> BuildStructuredScalarBindings(
    const StructuredMultiViewExprPattern& pattern) {
    std::unordered_map<std::string, std::string> scalarBindings;
    for (size_t i = 0; i < pattern.scalarInputs.size(); ++i) {
        scalarBindings["Scalar" + std::to_string(i)] = "scalar" + std::to_string(i) + "Value";
    }
    return scalarBindings;
}

// 功能：处理To Vector Api相关逻辑。
const char* ToVectorApi(AscendCBinaryOpKind kind) {
    switch (kind) {
    case AscendCBinaryOpKind::Add:
        return "AscendC::Add";
    case AscendCBinaryOpKind::Sub:
        return "AscendC::Sub";
    case AscendCBinaryOpKind::Mul:
        return "AscendC::Mul";
    case AscendCBinaryOpKind::Div:
        return "AscendC::Div";
    case AscendCBinaryOpKind::Unknown:
        return nullptr;
    }
    return nullptr;
}

// 功能：处理Local Tensor相关逻辑。
std::string LocalTensor(const std::string& tensor) {
    if (tensor.empty()) {
        return std::string();
    }
    return tensor + "Local";
}

// 功能：生成或转换Emit Reason Comment相关结果。
std::string EmitReasonComment(const char* prefix,
                              const AscendCPlanStep& step,
                              const AscendCEmitterOptions& options) {
    if (!options.emitReasonComments || step.reason.empty()) {
        return std::string();
    }
    return std::string("        // ") + prefix + ": " + step.reason + "\n";
}

// 针对滑窗操作（Window Load）生成边界处理 C++ 代码。
std::string EmitBoundaryShiftPatch(const AscendCPlanStep& step,
                                   const std::string& tileLengthVar,
                                   const AscendCEmitterOptions& options) {
    (void)tileLengthVar;
    (void)options;

    std::string code;
    switch (step.boundaryPolicy) {
    case AscendCBoundaryPolicy::Replicate:
        code += "            if (boundaryIndex < 0 || boundaryIndex >= copyLength) {\n";
        code += "                boundaryIndex = center;\n";
        code += "            }\n";
        code += "            if (boundaryIndex < 0) { boundaryIndex = 0; }\n";
        code += "            if (boundaryIndex >= copyLength) { boundaryIndex = copyLength - 1; }\n";
        return code;
    case AscendCBoundaryPolicy::Clamp:
        code += "            if (boundaryIndex < 0) { boundaryIndex = 0; }\n";
        code += "            if (boundaryIndex >= copyLength) { boundaryIndex = copyLength - 1; }\n";
        return code;
    case AscendCBoundaryPolicy::RequireHalo:
        code += "            if (boundaryIndex < 0 || boundaryIndex >= copyLength) {\n";
        code += "                // RequireHalo policy violation: defensively fall back to center sample.\n";
        code += "                boundaryIndex = center;\n";
        code += "            }\n";
        code += "            if (boundaryIndex < 0) { boundaryIndex = 0; }\n";
        code += "            if (boundaryIndex >= copyLength) { boundaryIndex = copyLength - 1; }\n";
        return code;
    }

    return std::string();
}

// 功能：生成或转换Emit Boundary Window Patch相关结果。
std::string EmitBoundaryWindowPatch(const AscendCPlanStep& step,
                                    const std::string& tileLengthVar,
                                    const AscendCEmitterOptions& options) {
    (void)tileLengthVar;
    (void)options;
    std::string code;
    code += "        // boundary policy: ";
    code += ToString(step.boundaryPolicy);
    code += " (weighted-window stage does not apply emitter-side edge patch)\n";
    return code;
}

// 功能：生成或转换Emit Tensor Copy Api相关结果。
std::string EmitTensorCopyApi(const std::string& dstLocal,
                              const std::string& srcLocal,
                              const std::string& tileLengthVar) {
    std::string code;
    code += "        AscendC::Adds(" + dstLocal + ", " + srcLocal + ", " +
            BuildTypedImmediateExpr(dstLocal, 0.0) + ", " + tileLengthVar + ");\n";
    return code;
}

// 功能：尝试处理Try Get Const Node Value相关逻辑。
bool TryGetConstNodeValue(const CalcExprGraph& graph,
                          int nodeId,
                          double& outValue) {
    if (nodeId < 0 || static_cast<size_t>(nodeId) >= graph.nodes.size()) {
        return false;
    }

    const CalcExprNode& node = graph.nodes[static_cast<size_t>(nodeId)];
    if (node.kind != CalcExprNodeKind::Const || !node.hasConstValue) {
        return false;
    }

    outValue = node.constValue;
    return true;
}

// 功能：处理To Expr Api相关逻辑。
const char* ToExprApi(CalcExprNodeKind kind) {
    switch (kind) {
    case CalcExprNodeKind::Add:
        return "AscendC::Add";
    case CalcExprNodeKind::Sub:
        return "AscendC::Sub";
    case CalcExprNodeKind::Mul:
        return "AscendC::Mul";
    case CalcExprNodeKind::Div:
        return "AscendC::Div";
    case CalcExprNodeKind::Min:
        return "AscendC::Min";
    case CalcExprNodeKind::Max:
        return "AscendC::Max";
    default:
        return nullptr;
    }
}

// 功能：提取或解析Resolve Load Local Tensor相关信息。
std::string ResolveLoadLocalTensor(const CalcExprNode& node,
                                   int nodeId,
                                   const std::unordered_map<int, std::string>& loadNodeLocalTensors,
                                   const std::string& defaultLocal) {
    auto it = loadNodeLocalTensors.find(nodeId);
    if (it != loadNodeLocalTensors.end() && !it->second.empty()) {
        return it->second;
    }
    if (!node.canonicalTensor.empty()) {
        return LocalTensor(node.canonicalTensor);
    }
    if (!node.token.empty() && node.token != "%value") {
        return LocalTensor(node.token);
    }
    return defaultLocal;
}

} // namespace

// 功能：生成或转换Emit Structured Map Expr相关结果。
std::string EmitStructuredMapExpr(const StructuredShiftedMapUpdatePattern& pattern,
                                  bool shiftedInput,
                                  const std::string& tileLengthVar,
                                  int* outScratchBufferCount) {
    const std::string expr = pattern.mapExpr.empty() ? "RawInput" : pattern.mapExpr;
    std::unordered_map<std::string, std::string> bindings;
    bindings["RawInput"] = shiftedInput ? "srcShiftLocal" : "srcLocal";

    const std::string targetLocal = shiftedInput ? "mapShiftLocal" : "mapCenterLocal";
    return EmitStructuredExprInternal(expr,
                                      bindings,
                                      {},
                                      targetLocal,
                                      tileLengthVar,
                                      outScratchBufferCount);
}

// 功能：生成或转换Emit Structured Update Expr相关结果。
std::string EmitStructuredUpdateExpr(const StructuredShiftedMapUpdatePattern& pattern,
                                     bool emitPostExpr,
                                     const std::string& tileLengthVar,
                                     int* outScratchBufferCount) {
    if (emitPostExpr) {
        if (!pattern.hasPost || pattern.postExpr.empty()) {
            if (outScratchBufferCount) {
                *outScratchBufferCount = 0;
            }
            return std::string();
        }
        std::unordered_map<std::string, std::string> bindings;
        bindings["UpdateResult"] = "dstLocal";
        return EmitStructuredExprInternal(pattern.postExpr,
                                          bindings,
                                          {},
                                          "dstLocal",
                                          tileLengthVar,
                                          outScratchBufferCount);
    }

    if (pattern.updateExpr.empty()) {
        if (outScratchBufferCount) {
            *outScratchBufferCount = 0;
        }
        return std::string();
    }

    std::string canonicalUpdateExpr;
    canonicalUpdateExpr.reserve(pattern.updateExpr.size());
    for (char c : pattern.updateExpr) {
        if (!std::isspace(static_cast<unsigned char>(c))) {
            canonicalUpdateExpr.push_back(c);
        }
    }

    if (canonicalUpdateExpr == "ShiftedMapped-CenterMapped") {
        if (outScratchBufferCount) {
            *outScratchBufferCount = 0;
        }
        return "        AscendC::Sub(dstLocal, mapShiftLocal, mapCenterLocal, " + tileLengthVar + ");\n";
    }
    if (canonicalUpdateExpr == "CenterMapped-ShiftedMapped") {
        if (outScratchBufferCount) {
            *outScratchBufferCount = 0;
        }
        return "        AscendC::Sub(dstLocal, mapCenterLocal, mapShiftLocal, " + tileLengthVar + ");\n";
    }

    std::unordered_map<std::string, std::string> bindings;
    bindings["CenterRaw"] = "srcLocal";
    bindings["ShiftedRaw"] = "srcShiftLocal";
    bindings["CenterMapped"] = "mapCenterLocal";
    bindings["ShiftedMapped"] = "mapShiftLocal";
    return EmitStructuredExprInternal(pattern.updateExpr,
                                      bindings,
                                      {},
                                      "dstLocal",
                                      tileLengthVar,
                                      outScratchBufferCount);
}

// 功能：生成或转换Emit Structured Multi View Main Expr相关结果。
std::string EmitStructuredMultiViewMainExpr(const StructuredMultiViewExprPattern& pattern,
                                            const std::string& tileLengthVar,
                                            int* outScratchBufferCount) {
    if (pattern.mainExpr.empty()) {
        if (outScratchBufferCount) {
            *outScratchBufferCount = 0;
        }
        return std::string();
    }

    std::unordered_map<std::string, std::string> tensorBindings;
    for (size_t i = 0; i < pattern.viewOffsets.size(); ++i) {
        tensorBindings["View" + std::to_string(i)] = "view" + std::to_string(i) + "Local";
    }

    const auto scalarBindings = BuildStructuredScalarBindings(pattern);

    return EmitStructuredExprInternal(pattern.mainExpr,
                                      tensorBindings,
                                      scalarBindings,
                                      "dstLocal",
                                      tileLengthVar,
                                      outScratchBufferCount);
}

// 功能：生成或转换Emit Structured Multi View Post Expr相关结果。
std::string EmitStructuredMultiViewPostExpr(const StructuredMultiViewExprPattern& pattern,
                                            const std::string& tileLengthVar,
                                            int* outScratchBufferCount) {
    if (!pattern.hasPost || pattern.postExpr.empty()) {
        if (outScratchBufferCount) {
            *outScratchBufferCount = 0;
        }
        return std::string();
    }

    std::unordered_map<std::string, std::string> tensorBindings;
    tensorBindings["MainExpr"] = "dstLocal";
    for (size_t i = 0; i < pattern.viewOffsets.size(); ++i) {
        tensorBindings["View" + std::to_string(i)] = "view" + std::to_string(i) + "Local";
    }

    const auto scalarBindings = BuildStructuredScalarBindings(pattern);

    return EmitStructuredExprInternal(pattern.postExpr,
                                      tensorBindings,
                                      scalarBindings,
                                      "dstLocal",
                                      tileLengthVar,
                                      outScratchBufferCount);
}

// 它接收 CalcExprGraph 和根节点，采用后序遍历的方式，将图结构直接映射为 AscendC API 调用链。
std::string EmitExprTreeToAscendC(const CalcExprGraph& graph,
                                  int rootNodeId,
                                  const std::string& dstLocalTensor,
                                  const std::string& auxLocalTensor,
                                  const std::string& tileLengthVar,
                                  const std::unordered_map<int, std::string>& loadNodeLocalTensors,
                                  bool* outSuccess) {
    if (outSuccess) {
        *outSuccess = false;
    }

    if (rootNodeId < 0 || static_cast<size_t>(rootNodeId) >= graph.nodes.size()) {
        return std::string();
    }
    if (dstLocalTensor.empty() || auxLocalTensor.empty() || dstLocalTensor == auxLocalTensor) {
        return std::string();
    }

    std::string code;
    auto emitNode = [&](auto&& self,
                        int nodeId,
                        const std::string& outLocal,
                        const std::string& auxLocal) -> bool {
        if (nodeId < 0 || static_cast<size_t>(nodeId) >= graph.nodes.size()) {
            return false;
        }

        const CalcExprNode& node = graph.nodes[static_cast<size_t>(nodeId)];
        switch (node.kind) {
        case CalcExprNodeKind::Load: {
            std::string srcLocal = ResolveLoadLocalTensor(
                node, nodeId, loadNodeLocalTensors, auxLocal);
            if (srcLocal.empty()) {
                return false;
            }
            if (srcLocal != outLocal) {
                code += EmitTensorCopyApi(outLocal, srcLocal, tileLengthVar);
            }
            return true;
        }
        case CalcExprNodeKind::Const: {
            if (!node.hasConstValue) {
                return false;
            }
            code += "        AscendC::Duplicate(" + outLocal + ", " +
                    BuildTypedImmediateExpr(outLocal, node.constValue) + ", " + tileLengthVar + ");\n";
            return true;
        }
        case CalcExprNodeKind::Clamp: {
            if (node.inputs.empty()) {
                return false;
            }
            if (!self(self, node.inputs.front(), outLocal, auxLocal)) {
                return false;
            }
            code += "        AscendC::Duplicate(" + auxLocal + ", " +
                    BuildTypedImmediateExpr(auxLocal, 0.0) + ", " + tileLengthVar + ");\n";
            code += "        AscendC::Max(" + outLocal + ", " + outLocal + ", " + auxLocal + ", " + tileLengthVar + ");\n";
            return true;
        }
        case CalcExprNodeKind::Add:
        case CalcExprNodeKind::Sub:
        case CalcExprNodeKind::Mul:
        case CalcExprNodeKind::Div:
        case CalcExprNodeKind::Min:
        case CalcExprNodeKind::Max: {
            if (node.inputs.size() < 2) {
                return false;
            }

            if (node.kind == CalcExprNodeKind::Mul) {
                double c = 0.0;
                if (TryGetConstNodeValue(graph, node.inputs[0], c)) {
                    if (!self(self, node.inputs[1], outLocal, auxLocal)) {
                        return false;
                    }
                    code += "        AscendC::Muls(" + outLocal + ", " + outLocal + ", " +
                            BuildTypedImmediateExpr(outLocal, c) + ", " + tileLengthVar + ");\n";
                    return true;
                }
                if (TryGetConstNodeValue(graph, node.inputs[1], c)) {
                    if (!self(self, node.inputs[0], outLocal, auxLocal)) {
                        return false;
                    }
                    code += "        AscendC::Muls(" + outLocal + ", " + outLocal + ", " +
                            BuildTypedImmediateExpr(outLocal, c) + ", " + tileLengthVar + ");\n";
                    return true;
                }
            }

            if (node.kind == CalcExprNodeKind::Add) {
                double c = 0.0;
                if (TryGetConstNodeValue(graph, node.inputs[0], c)) {
                    if (!self(self, node.inputs[1], outLocal, auxLocal)) {
                        return false;
                    }
                    code += "        AscendC::Adds(" + outLocal + ", " + outLocal + ", " +
                            BuildTypedImmediateExpr(outLocal, c) + ", " + tileLengthVar + ");\n";
                    return true;
                }
                if (TryGetConstNodeValue(graph, node.inputs[1], c)) {
                    if (!self(self, node.inputs[0], outLocal, auxLocal)) {
                        return false;
                    }
                    code += "        AscendC::Adds(" + outLocal + ", " + outLocal + ", " +
                            BuildTypedImmediateExpr(outLocal, c) + ", " + tileLengthVar + ");\n";
                    return true;
                }
            }

            if (node.kind == CalcExprNodeKind::Sub) {
                double c = 0.0;
                if (TryGetConstNodeValue(graph, node.inputs[1], c)) {
                    if (!self(self, node.inputs[0], outLocal, auxLocal)) {
                        return false;
                    }
                    code += "        AscendC::Adds(" + outLocal + ", " + outLocal + ", " +
                            BuildTypedImmediateExpr(outLocal, -c) + ", " + tileLengthVar + ");\n";
                    return true;
                }
                if (TryGetConstNodeValue(graph, node.inputs[0], c)) {
                    if (!self(self, node.inputs[1], outLocal, auxLocal)) {
                        return false;
                    }
                    code += "        AscendC::Muls(" + outLocal + ", " + outLocal + ", " +
                            BuildTypedImmediateExpr(outLocal, -1.0) + ", " + tileLengthVar + ");\n";
                    code += "        AscendC::Adds(" + outLocal + ", " + outLocal + ", " +
                            BuildTypedImmediateExpr(outLocal, c) + ", " + tileLengthVar + ");\n";
                    return true;
                }
            }

            if (node.kind == CalcExprNodeKind::Div) {
                double c = 0.0;
                if (TryGetConstNodeValue(graph, node.inputs[1], c) && std::abs(c) > 1e-12) {
                    if (!self(self, node.inputs[0], outLocal, auxLocal)) {
                        return false;
                    }
                    code += "        AscendC::Muls(" + outLocal + ", " + outLocal + ", " +
                            BuildTypedImmediateExpr(outLocal, 1.0 / c) + ", " + tileLengthVar + ");\n";
                    return true;
                }
            }

            if (node.kind == CalcExprNodeKind::Min) {
                double c = 0.0;
                if (TryGetConstNodeValue(graph, node.inputs[0], c)) {
                    if (!self(self, node.inputs[1], outLocal, auxLocal)) {
                        return false;
                    }
                    code += "        AscendC::Mins(" + outLocal + ", " + outLocal + ", " +
                            BuildTypedImmediateExpr(outLocal, c) + ", " + tileLengthVar + ");\n";
                    return true;
                }
                if (TryGetConstNodeValue(graph, node.inputs[1], c)) {
                    if (!self(self, node.inputs[0], outLocal, auxLocal)) {
                        return false;
                    }
                    code += "        AscendC::Mins(" + outLocal + ", " + outLocal + ", " +
                            BuildTypedImmediateExpr(outLocal, c) + ", " + tileLengthVar + ");\n";
                    return true;
                }
            }

            if (!self(self, node.inputs[0], outLocal, auxLocal)) {
                return false;
            }
            if (!self(self, node.inputs[1], auxLocal, outLocal)) {
                return false;
            }

            const char* api = ToExprApi(node.kind);
            if (!api) {
                return false;
            }

            code += std::string("        ") + api + "(" + outLocal + ", " + outLocal + ", " + auxLocal + ", " + tileLengthVar + ");\n";
            return true;
        }
        case CalcExprNodeKind::Exp:
        case CalcExprNodeKind::Store:
        case CalcExprNodeKind::Unknown:
            return false;
        }

        return false;
    };

    bool ok = emitNode(emitNode, rootNodeId, dstLocalTensor, auxLocalTensor);
    if (outSuccess) {
        *outSuccess = ok;
    }
    return ok ? code : std::string();
}

// Emit the Compute body for a linear reduction:
//   output[0] = bias[0] + sum(input[i] * weight[i], i=0..K-1)
// Uses AscendC::Mul (element-wise) + AscendC::ReduceSum, then adds bias.
// inputLocal is reused as the Mul destination (it's already been read).
// 功能：生成或转换Emit Linear Reduction相关结果。
std::string EmitLinearReduction(const AscendCPlanStep& step,
                                const std::string& /*tileLengthVar*/,
                                const AscendCEmitterOptions& options) {
    std::string code;
    code += EmitReasonComment("linear-reduction", step, options);

    if (!step.hasLinearReductionPayload) {
        return EmitScalarFallback(step, "tileLength");
    }

    const LinearReductionStepPayload& p = step.linearReductionPayload;
    const std::string inputLocal  = p.inputTensor  + "Local";
    const std::string weightLocal = p.weightTensor + "Local";
    const std::string outputLocal = p.outputTensor + "Local";
    const std::string biasLocal   = p.biasTensor   + "Local";
    const std::string K           = std::to_string(p.K);

    // Step 1: element-wise multiply input * weight -> reuse inputLocal as scratch.
    code += "        AscendC::Mul(" + inputLocal + ", " + inputLocal + ", " +
            weightLocal + ", " + K + ");\n";

    // Step 2: reduce sum -> output[0] = sum(inputLocal[0..K-1]).
    // ReduceSum(dst, src, work, len): work must be a separate buffer.
    // We use weightLocal as the work buffer (already consumed).
    code += "        AscendC::ReduceSum(" + outputLocal + ", " + inputLocal + ", " +
            weightLocal + ", " + K + ");\n";

    // Step 3: add bias to output[0].
    if (p.hasBias && !p.biasTensor.empty()) {
        code += "        float biasScalar = " + biasLocal + ".GetValue(0);\n";
        code += "        " + outputLocal + ".SetValue(0, " + outputLocal +
                ".GetValue(0) + biasScalar);\n";
    }

    return code;
}

// fallback
std::string EmitScalarFallback(const AscendCPlanStep& step,
                               const std::string& tileLengthVar) {
    std::string code;
    code += EmitReasonComment("scalar fallback", step, AscendCEmitterOptions{});
    code += "        for (int32_t i = 0; i < static_cast<int32_t>(" + tileLengthVar + "); ++i) {\n";
    code += "            if (i >= 0 && i < static_cast<int32_t>(" + tileLengthVar + ")) {\n";
    if (!step.loopBody.empty()) {
        code += "                " + step.loopBody + "\n";
    } else if (!step.dst.empty() && !step.srcA.empty()) {
        code += "                " + step.dst + "Local.SetValue(i, " + step.srcA + "Local.GetValue(i));\n";
    } else if (!step.dst.empty()) {
        code += "                " + step.dst + "Local.SetValue(i, 0);\n";
    } else {
        code += "                // fallback has no addressable tensors\n";
    }
    code += "            }\n";
    code += "        }\n";
    return code;
}

// 处理纯向量逐点运算。
std::string EmitPointwiseStage(const AscendCPlanStep& step,
                               const std::string& tileLengthVar,
                               const AscendCEmitterOptions& options) {
    std::string code;
    code += EmitReasonComment("pointwise", step, options);

    std::string dst = LocalTensor(step.dst);
    std::string srcA = LocalTensor(step.srcA);
    std::string srcB = LocalTensor(step.srcB);

    if (dst.empty()) {
        dst = srcA;
    }
    if (srcA.empty()) {
        srcA = dst;
    }

    if (step.kind == AscendCPlanStepKind::WindowLoad ||
        step.kind == AscendCPlanStepKind::Shift) {
        if (dst.empty() || srcA.empty()) {
            return EmitScalarFallback(step, tileLengthVar);
        }

        int shift = 0;
        for (int offset : step.windowOffsets) {
            if (offset != 0) {
                shift = offset;
                break;
            }
        }

        const int leftHalo = std::max(0, step.leftHalo);
        const int rightHalo = std::max(0, step.rightHalo);
        code += "        // window load from source tensor with overlap-aware indexing\n";
        code += "        const int32_t validLength = static_cast<int32_t>(" + tileLengthVar + ");\n";
        if (!step.srcA.empty()) {
            code += "        int32_t copyLength = static_cast<int32_t>(" + step.srcA + "CopyLength);\n";
        } else if (step.copyLength > 0) {
            code += "        int32_t copyLength = " + std::to_string(step.copyLength) + ";\n";
        } else {
            code += "        int32_t copyLength = validLength + " + std::to_string(leftHalo + rightHalo) + ";\n";
        }
        code += "        if (copyLength <= 0) { copyLength = 1; }\n";
        code += "        for (int32_t i = 0; i < validLength; ++i) {\n";
        code += "            int32_t center = i + " + std::to_string(leftHalo) + ";\n";
        code += "            if (center < 0) { center = 0; }\n";
        code += "            if (center >= copyLength) { center = copyLength - 1; }\n";
        code += "            int32_t boundaryIndex = center + " + std::to_string(shift) + ";\n";
        code += EmitBoundaryShiftPatch(step, tileLengthVar, options);
        code += "            " + dst + ".SetValue(i, " + srcA + ".GetValue(boundaryIndex));\n";
        code += "        }\n";
        return code;
    }

    if (step.kind == AscendCPlanStepKind::PointwiseExpr ||
        step.kind == AscendCPlanStepKind::ShiftedElementwise) {
        if (!options.exprGraph || step.exprRootNodeId < 0) {
            return EmitScalarFallback(step, tileLengthVar);
        }

        std::string aux = srcB;
        if (aux.empty() || aux == dst) {
            aux = srcA;
        }
        if (aux.empty() || aux == dst) {
            return EmitScalarFallback(step, tileLengthVar);
        }

        std::unordered_map<int, std::string> loadMap;
        for (int nodeId : step.inputNodeIds) {
            if (nodeId < 0 || static_cast<size_t>(nodeId) >= options.exprGraph->nodes.size()) {
                continue;
            }
            const CalcExprNode& node = options.exprGraph->nodes[static_cast<size_t>(nodeId)];
            if (node.kind != CalcExprNodeKind::Load) {
                continue;
            }

            std::string local = srcA;
            if (node.offsetKnown && node.offset != 0 && !dst.empty()) {
                local = dst;
            } else if (!node.canonicalTensor.empty()) {
                if (!step.srcA.empty() && node.canonicalTensor == step.srcA && !srcA.empty()) {
                    local = srcA;
                } else if (!step.dst.empty() && node.canonicalTensor == step.dst && !dst.empty()) {
                    local = dst;
                } else {
                    local = LocalTensor(node.canonicalTensor);
                }
            }
            if (!local.empty()) {
                loadMap[nodeId] = local;
            }
        }

        bool ok = false;
        std::string exprCode = EmitExprTreeToAscendC(
            *options.exprGraph,
            step.exprRootNodeId,
            dst,
            aux,
            tileLengthVar,
            loadMap,
            &ok);
        if (!ok || exprCode.empty()) {
            return EmitScalarFallback(step, tileLengthVar);
        }
        code += exprCode;
        return code;
    }

    if (step.kind == AscendCPlanStepKind::Diff ||
        step.kind == AscendCPlanStepKind::Sub) {
        if (srcB.empty()) {
            srcB = srcA;
        }
        code += "        AscendC::Sub(" + dst + ", " + srcA + ", " + srcB + ", " + tileLengthVar + ");\n";
        return code;
    }

    if (step.kind == AscendCPlanStepKind::Scale) {
        if (!srcB.empty()) {
            code += "        AscendC::Mul(" + dst + ", " + srcA + ", " + srcB + ", " + tileLengthVar + ");\n";
        } else if (std::abs(step.scalarImm) > 1e-12) {
            code += "        AscendC::Muls(" + dst + ", " + srcA + ", " + FormatFloatLiteral(step.scalarImm) + ", " + tileLengthVar + ");\n";
        } else {
            code += EmitTensorCopyApi(dst, srcA, tileLengthVar);
        }
        return code;
    }

    if (step.kind == AscendCPlanStepKind::ElementwiseBinary) {
        const char* api = ToVectorApi(step.binaryOp);
        if (!api) {
            return EmitScalarFallback(step, tileLengthVar);
        }
        if (srcB.empty()) {
            srcB = srcA;
        }
        code += std::string("        ") + api + "(" + dst + ", " + srcA + ", " + srcB + ", " + tileLengthVar + ");\n";
        return code;
    }

    code += EmitTensorCopyApi(dst, srcA, tileLengthVar);
    return code;
}

// 功能：生成或转换Emit Weighted Window Stage相关结果。
std::string EmitWeightedWindowStage(const AscendCPlanStep& step,
                                    const std::string& tileLengthVar,
                                    const AscendCEmitterOptions& options) {
    std::string code;
    code += EmitReasonComment("weighted-window", step, options);

    std::string dst = LocalTensor(step.dst);
    std::string src = LocalTensor(step.srcA);
    std::string scalar = LocalTensor(!step.scalarTensor.empty() ? step.scalarTensor : step.srcB);

    if (dst.empty() || src.empty()) {
        return EmitScalarFallback(step, tileLengthVar);
    }

    if (!scalar.empty()) {
        // API-chain body for weighted window style (Mul/Muls/Sub/Add).
        code += "        AscendC::Mul(" + dst + ", " + src + ", " + scalar + ", " + tileLengthVar + ");\n";
        code += "        AscendC::Muls(" + dst + ", " + dst + ", 2.0f, " + tileLengthVar + ");\n";
        code += "        AscendC::Sub(" + dst + ", " + src + ", " + dst + ", " + tileLengthVar + ");\n";
        code += "        AscendC::Mul(" + dst + ", " + dst + ", " + scalar + ", " + tileLengthVar + ");\n";
        code += "        AscendC::Add(" + dst + ", " + dst + ", " + src + ", " + tileLengthVar + ");\n";
    } else {
        code += EmitTensorCopyApi(dst, src, tileLengthVar);
        code += "        AscendC::Sub(" + dst + ", " + dst + ", " + src + ", " + tileLengthVar + ");\n";
        code += "        AscendC::Add(" + dst + ", " + dst + ", " + src + ", " + tileLengthVar + ");\n";
    }

    code += EmitBoundaryWindowPatch(step, tileLengthVar, options);
    return code;
}

// 处理模板/卷积运算
std::string EmitCoeffStencil(const AscendCPlanStep& step,
                             const std::string& tileLengthVar,
                             const AscendCEmitterOptions& options) {
    if (step.kind != AscendCPlanStepKind::CoeffStencil ||
        step.structuredMultiViewExpr.mainExpr.empty()) {
        return EmitScalarFallback(step, tileLengthVar);
    }

    std::string code;
    code += EmitReasonComment("multi-view", step, options);

    size_t scalarCount = step.structuredMultiViewExpr.scalarInputs.size();
    if (step.hasFamilyPayload) {
        scalarCount = std::max(scalarCount, step.familyPayload.scalarInputs.size());
    }

    for (size_t i = 0; i < scalarCount; ++i) {
        std::string scalarToken;
        if (i < step.structuredMultiViewExpr.scalarInputs.size()) {
            scalarToken = step.structuredMultiViewExpr.scalarInputs[i];
        } else if (step.hasFamilyPayload && i < step.familyPayload.scalarInputs.size()) {
            scalarToken = step.familyPayload.scalarInputs[i];
        }
        std::string scalarLocal = LocalTensor(scalarToken);
        if (scalarLocal.empty()) {
            scalarLocal = "scalar" + std::to_string(i) + "Local";
        }
        code += "        const auto scalar" + std::to_string(i) + "Value = static_cast<float>(" +
                scalarLocal + ".GetValue(0));\n";
    }

    int mainScratch = 0;
    int postScratch = 0;
    std::string mainCode = EmitStructuredMultiViewMainExpr(
        step.structuredMultiViewExpr,
        tileLengthVar,
        &mainScratch);
    std::string postCode = EmitStructuredMultiViewPostExpr(
        step.structuredMultiViewExpr,
        tileLengthVar,
        &postScratch);

    if (mainCode.empty()) {
        return EmitScalarFallback(step, tileLengthVar);
    }

    code += mainCode;
    if (!postCode.empty()) {
        code += postCode;
    }
    return code;
}

// 指令组合。（有点过于针对化了，感觉不是很好）
std::string EmitDecayExpMulScalarFastPath(const AscendCPlanStep& step,
                                          const std::string& tileLengthVar,
                                          const AscendCEmitterOptions& options) {
    std::string code;
    code += EmitReasonComment("decay-scalar-fast-path", step, options);

    const std::string dst = LocalTensor(step.dst);
    const std::string srcA = LocalTensor(step.srcA);
    const std::string srcB = LocalTensor(step.srcB);

    std::string scalarTensor = step.srcC;
    if (step.hasFamilyPayload && !step.familyPayload.scalarInputs.empty()) {
        scalarTensor = step.familyPayload.scalarInputs.front();
    }

    if (dst.empty() || srcA.empty() || srcB.empty()) {
        return EmitScalarFallback(step, tileLengthVar);
    }

    std::string scalarExpr;
    if (!scalarTensor.empty()) {
        scalarExpr = LocalTensor(scalarTensor) + ".GetValue(0)";
    } else {
        scalarExpr = FormatFloatLiteral(step.scalarImm);
    }

    static int decayScalarExprTempCounter = 0;
    const std::string scalarTempName =
        "decayScalarExprTmp" + std::to_string(decayScalarExprTempCounter++);
    code += "        const auto " + scalarTempName + " = (" + scalarExpr + ");\n";
    code += "        AscendC::Muls(" + srcB + ", " + srcB + ", (-" + scalarTempName + "), " + tileLengthVar + ");\n";
    code += "        AscendC::Exp(" + srcB + ", " + srcB + ", " + tileLengthVar + ");\n";
    code += "        AscendC::Mul(" + dst + ", " + srcA + ", " + srcB + ", " + tileLengthVar + ");\n";
    return code;
}

// 功能：生成或转换Emit Shifted Elementwise相关结果。
std::string EmitShiftedElementwise(const AscendCPlanStep& step,
                                   const std::string& tileLengthVar,
                                   const AscendCEmitterOptions& options) {
    return EmitPointwiseStage(step, tileLengthVar, options);
}

// 处理边界截断
std::string EmitClamp(const AscendCPlanStep& step,
                      const std::string& tileLengthVar,
                      const AscendCEmitterOptions& options) {
    std::string code;
    code += EmitReasonComment("clamp", step, options);

    std::string dst = LocalTensor(step.dst);
    std::string src = LocalTensor(step.srcA.empty() ? step.dst : step.srcA);
    std::string scratch = LocalTensor(step.srcB);

    if (dst.empty() || src.empty()) {
        return EmitScalarFallback(step, tileLengthVar);
    }

    if (scratch.empty() || scratch == dst) {
        if (src != dst) {
            scratch = src;
        } else {
            // Last-resort narrow scalar patch when only one tensor is available.
            if (options.allowScalarBoundaryFallback) {
                code += "        for (int32_t i = 0; i < static_cast<int32_t>(" + tileLengthVar + "); ++i) {\n";
                code += "            float clampV = static_cast<float>(" + dst + ".GetValue(i));\n";
                code += "            if (clampV < 0.0f) { clampV = 0.0f; }\n";
                code += "            " + dst + ".SetValue(i, clampV);\n";
                code += "        }\n";
                return code;
            }
            return EmitScalarFallback(step, tileLengthVar);
        }
    }

    if (src != dst) {
        code += EmitTensorCopyApi(dst, src, tileLengthVar);
    }

    code += "        AscendC::Muls(" + scratch + ", " + scratch + ", 0.0f, " + tileLengthVar + ");\n";
    code += "        AscendC::Max(" + dst + ", " + dst + ", " + scratch + ", " + tileLengthVar + ");\n";
    return code;
}

// 功能：生成或转换Emit Ascend CFrom Plan相关结果。
std::string EmitAscendCFromPlan(const AscendCPlan& plan,
                                const std::string& tileLengthVar,
                                const AscendCEmitterOptions& options) {
    std::string code;

    std::string primaryDst;
    std::string primarySrc;
    for (const AscendCPlanStep& step : plan.steps) {
        if (primaryDst.empty() && !step.dst.empty()) {
            primaryDst = step.dst;
        }
        if (primarySrc.empty() && !step.srcA.empty()) {
            primarySrc = step.srcA;
        }
    }
    if (primaryDst.empty()) {
        primaryDst = primarySrc;
    }
    if (primarySrc.empty()) {
        primarySrc = primaryDst;
    }

    std::unordered_map<std::string, std::string> tempMap;
    if (!primaryDst.empty()) {
        tempMap["Shift"] = primaryDst;
        tempMap["Q"] = primaryDst;
        tempMap["Diff"] = primaryDst;
        tempMap["Scale"] = primaryDst;
        tempMap["Sub"] = primaryDst;
        tempMap["Clamp"] = primaryDst;
    }
    if (!primarySrc.empty()) {
        tempMap["ShiftedQ"] = primarySrc;
    }

    auto ResolveRef = [&](const std::string& token) -> std::string {
        if (token.empty()) {
            return std::string();
        }
        auto it = tempMap.find(token);
        if (it != tempMap.end()) {
            return it->second;
        }
        return token;
    };

    for (const AscendCPlanStep& rawStep : plan.steps) {
        AscendCPlanStep step = rawStep;

        step.dst = ResolveRef(step.dst);
        step.srcA = ResolveRef(step.srcA);
        step.srcB = ResolveRef(step.srcB);
        step.srcC = ResolveRef(step.srcC);
        step.scalarTensor = ResolveRef(step.scalarTensor);

        if (!step.tempTensor.empty()) {
            std::string mapped = ResolveRef(step.tempTensor);
            if (mapped.empty()) {
                mapped = primaryDst;
            }
            if (!mapped.empty()) {
                tempMap[step.tempTensor] = mapped;
            }
            if (step.dst.empty()) {
                step.dst = mapped;
            }
        }

        const bool isStencilTrait =
            (step.kind == AscendCPlanStepKind::CoeffStencil ||
             step.kind == AscendCPlanStepKind::Stencil ||
             !step.structuredMultiViewExpr.mainExpr.empty() ||
             step.transferPolicy == AscendCTransferPolicy::SpatialWindow ||
             step.transferPolicy == AscendCTransferPolicy::TemporalLoopCarried);

        const bool isPointwiseTrait =
            (step.kind == AscendCPlanStepKind::ElementwiseBinary ||
             step.kind == AscendCPlanStepKind::DecayExpMul ||
             step.kind == AscendCPlanStepKind::ShiftedElementwise ||
             step.kind == AscendCPlanStepKind::WindowLoad ||
             step.kind == AscendCPlanStepKind::PointwiseExpr ||
             step.kind == AscendCPlanStepKind::Shift ||
             step.kind == AscendCPlanStepKind::Q ||
             step.kind == AscendCPlanStepKind::ShiftedQ ||
             step.kind == AscendCPlanStepKind::Diff ||
             step.kind == AscendCPlanStepKind::Scale ||
             step.kind == AscendCPlanStepKind::Sub);

        if (step.family == CalcSemanticFamily::PointwiseFamily &&
            step.kind == AscendCPlanStepKind::DecayExpMul) {
            code += EmitDecayExpMulScalarFastPath(step, tileLengthVar, options);
            continue;
        }

        if ((step.family == CalcSemanticFamily::SpatialStencilFamily ||
             step.family == CalcSemanticFamily::TemporalStencilFamily) &&
            isStencilTrait) {
            code += EmitCoeffStencil(step, tileLengthVar, options);
            continue;
        }

        if (step.kind == AscendCPlanStepKind::Clamp) {
            if (step.srcB.empty() && !primarySrc.empty() && primarySrc != step.dst) {
                step.srcB = primarySrc;
            }
            code += EmitClamp(step, tileLengthVar, options);
            continue;
        }

        if (isPointwiseTrait || step.family == CalcSemanticFamily::PointwiseFamily) {
            code += EmitPointwiseStage(step, tileLengthVar, options);
            continue;
        }

        if (step.kind == AscendCPlanStepKind::LinearReduction) {
            code += EmitLinearReduction(step, tileLengthVar, options);
            continue;
        }

        code += EmitScalarFallback(step, tileLengthVar);
    }

    return code;
}

} // namespace ascendc_mlir
} // namespace dacppTranslator
