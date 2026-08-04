/*
职责：把 Calc 的 Clang AST 降级为 MLIR。

- 支持赋值、复合赋值、for、if、数组/下标访问。
- 支持部分 inline call 与 const 折叠。
- 生成可验证的 MLIR module。
*/
#include "ascendc_mlir/CalcAstToMlir.h"
#include "ascendc_mlir/CalcContext.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Calc.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#if defined(DACPP_TRANSLATOR_HAS_MLIR)
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#endif

namespace dacppTranslator {
namespace ascendc_mlir {

#if defined(DACPP_TRANSLATOR_HAS_MLIR)
namespace {

struct AccessPath {
    std::string base;
    std::vector<const clang::Expr*> indices;
    std::vector<std::string> fields;
};

struct SymbolInfo {
    enum class Kind {
        MemRef,
        ScalarAlloca,
        LoopIndex,
    } kind = Kind::MemRef;
    mlir::Value value;
    mlir::Type elementType;
};

struct LValueInfo {
    bool valid = false;
    mlir::Value memref;
    std::vector<mlir::Value> indices;
    mlir::Type elementType;
};

// 把clang类型转化为string字符串
static std::string TypeToString(clang::QualType qt) {
    return qt.getCanonicalType().getAsString();
}

class CalcAstLowering {
public:
    // 初始化
    CalcAstLowering(const Calc* calc, const CalcAstToMlirOptions& options)
        : calc(calc), options(options), context(std::make_shared<mlir::MLIRContext>()), builder(context.get()) {
        if (this->options.maxInlineDepth == 0) {
            this->options.maxInlineDepth = 1;
        }
        mlir::DialectRegistry registry;
        registry.insert<mlir::func::FuncDialect>();
        registry.insert<mlir::arith::ArithDialect>();
        registry.insert<mlir::memref::MemRefDialect>();
        registry.insert<mlir::scf::SCFDialect>();
        registry.insert<mlir::math::MathDialect>();
        context->appendDialectRegistry(registry);
        context->loadAllAvailableDialects();
        module = mlir::ModuleOp::create(UnknownLoc());
    }

    // 功能：执行Run流程。
    CalcAstToMlirResult Run() {
        CalcAstToMlirResult result;
        result.mlirAvailable = true;

        if (!calc) {
            result.diagnostics = "calc is null";
            return result;
        }

        const clang::FunctionDecl* funcDecl = calc->getCalcFunctionDecl();
        if (!funcDecl) {
            result.diagnostics = "calc function decl is null";
            return result;
        }

        astContext = &funcDecl->getASTContext();

        if (!LowerFunction(funcDecl)) {
            if (resultDiagnostics.empty()) {
                resultDiagnostics = "failed to lower calc function";
            }
            result.diagnostics = resultDiagnostics;
            result.loweringDiagnostics = loweringDiagnostics;
            return result;
        }

        std::string moduleText;
        llvm::raw_string_ostream os(moduleText);
        module.print(os);
        os.flush();

        result.lowered = true;
        result.moduleText = moduleText;

        if (options.verifyModule) {
            if (mlir::failed(mlir::verify(module))) {
                result.verified = false;
                if (resultDiagnostics.empty()) {
                    resultDiagnostics = "mlir::verify failed";
                }
            } else {
                result.verified = true;
            }
        } else {
            result.verified = true;
        }

        result.diagnostics = resultDiagnostics;
        result.loweringDiagnostics = loweringDiagnostics;
        result.artifact.valid = true;
        result.artifact.context = context;
        result.artifact.module = module;
        return result;
    }

private:
    const Calc* calc = nullptr;
    CalcAstToMlirOptions options;
    std::shared_ptr<mlir::MLIRContext> context;
    mlir::OpBuilder builder;
    mlir::ModuleOp module;
    const clang::ASTContext* astContext = nullptr;
    std::unordered_map<std::string, SymbolInfo> symbols;
    std::unordered_set<const clang::ValueDecl*> activeDeclInits;
    unsigned declInlineDepth = 0;
    unsigned helperInlineDepth = 0;
    std::string resultDiagnostics;
    std::vector<CalcLoweringDiagnostic> loweringDiagnostics;

    // 功能：处理Unknown Loc相关逻辑。
    mlir::Location UnknownLoc() {
        return mlir::UnknownLoc::get(context.get());
    }

    // 功能：处理Report Unsupported相关逻辑。
    bool ReportUnsupported(const clang::Stmt* node, llvm::StringRef reason) {
        CalcLoweringDiagnostic diagnostic;
        diagnostic.message = reason.str();
        diagnostic.nodeKind = node ? node->getStmtClassName() : "Unknown";
        loweringDiagnostics.push_back(diagnostic);

        if (!resultDiagnostics.empty()) {
            resultDiagnostics += "; ";
        }
        resultDiagnostics += "unsupported(" + diagnostic.nodeKind + "): " + diagnostic.message;
        return false;
    }

    // 功能：检查Contains Token相关条件。
    bool ContainsToken(const std::string& text, llvm::StringRef token) {
        return text.find(token.str()) != std::string::npos;
    }

    // 通过字符串名称推断对应的 MLIR 标量类型
    mlir::Type InferElementTypeFromString(const std::string& typeStr) {
        if (ContainsToken(typeStr, "double")) return builder.getF64Type();
        if (ContainsToken(typeStr, "float")) return builder.getF32Type();
        if (ContainsToken(typeStr, "long long") || ContainsToken(typeStr, "int64")) return builder.getI64Type();
        if (ContainsToken(typeStr, "short")) return builder.getI16Type();
        if (ContainsToken(typeStr, "char")) return builder.getI8Type();
        if (ContainsToken(typeStr, "bool")) return builder.getI1Type();
        return builder.getI32Type();
    }

    // 标量类型映射到MLIR类型。
    mlir::Type LowerScalarType(clang::QualType qt) {
        qt = qt.getCanonicalType();

        if (qt->isBooleanType()) {
            return builder.getI1Type();
        }
        if (qt->isSpecificBuiltinType(clang::BuiltinType::Float)) {
            return builder.getF32Type();
        }
        if (qt->isSpecificBuiltinType(clang::BuiltinType::Double)) {
            return builder.getF64Type();
        }
        if (qt->isSpecificBuiltinType(clang::BuiltinType::LongDouble)) {
            return builder.getF64Type();
        }

        if (qt->isIntegerType() && astContext) {
            unsigned width = astContext->getTypeSize(qt);
            if (width == 0) {
                width = 32;
            }
            if (width <= 8) return builder.getI8Type();
            if (width <= 16) return builder.getI16Type();
            if (width <= 32) return builder.getI32Type();
            return builder.getI64Type();
        }

        return InferElementTypeFromString(TypeToString(qt));
    }

    // 翻译函数参数类型。
    mlir::Type LowerParamType(clang::QualType qt) {
        qt = qt.getCanonicalType();
        if (const auto* refType = qt->getAs<clang::ReferenceType>()) {
            qt = refType->getPointeeType().getCanonicalType();
        }

        if (const auto* ptrType = qt->getAs<clang::PointerType>()) {
            mlir::Type elemTy = LowerScalarType(ptrType->getPointeeType());
            return mlir::MemRefType::get({mlir::ShapedType::kDynamic}, elemTy);
        }

        const std::string typeStr = TypeToString(qt);
        if (ContainsToken(typeStr, "dacpp::Matrix")) {
            mlir::Type elemTy = InferElementTypeFromString(typeStr);
            return mlir::MemRefType::get(
                {mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic}, elemTy);
        }
        if (ContainsToken(typeStr, "dacpp::Vector") || ContainsToken(typeStr, "dacpp::Tensor")) {
            mlir::Type elemTy = InferElementTypeFromString(typeStr);
            return mlir::MemRefType::get({mlir::ShapedType::kDynamic}, elemTy);
        }

        return LowerScalarType(qt);
    }

    // 功能：检查Is Float Type相关条件。
    bool IsFloatType(mlir::Type ty) {
        return ty.isa<mlir::FloatType>();
    }

    // 功能：检查Is Index Type相关条件。
    bool IsIndexType(mlir::Type ty) {
        return ty.isa<mlir::IndexType>();
    }

    // 功能：检查Is Integer Type相关条件。
    bool IsIntegerType(mlir::Type ty) {
        return ty.isa<mlir::IntegerType>();
    }

    // 功能：处理Get Integer Like Type相关逻辑。
    mlir::Type GetIntegerLikeType(mlir::Type ty) {
        if (IsIndexType(ty)) {
            return builder.getI64Type();
        }
        if (IsIntegerType(ty)) {
            return ty;
        }
        return builder.getI32Type();
    }

    // 快速生成一个 MLIR 中的常量索引值（index 类型），常用于数组寻址和循环计数。
    mlir::Value ConstIndex(int64_t v) {
        return builder.create<mlir::arith::ConstantIndexOp>(UnknownLoc(), v);
    }

    // 根据传入的类型，生成一个对应的“零”值（如整数 0，浮点数 0.0）。
    mlir::Value BuildZero(mlir::Type ty) {
        if (!ty) {
            return mlir::Value();
        }
        if (IsIndexType(ty)) {
            return ConstIndex(0);
        }
        if (auto intTy = ty.dyn_cast<mlir::IntegerType>()) {
            auto attr = builder.getIntegerAttr(intTy, 0);
            return builder.create<mlir::arith::ConstantOp>(UnknownLoc(), intTy, attr);
        }
        if (auto floatTy = ty.dyn_cast<mlir::FloatType>()) {
            auto attr = builder.getFloatAttr(floatTy, 0.0);
            return builder.create<mlir::arith::ConstantOp>(UnknownLoc(), floatTy, attr);
        }
        return mlir::Value();
    }

    // 极其重要的强制类型转换函数。
    mlir::Value CastValueTo(mlir::Value value, mlir::Type dstType) {
        if (!value || !dstType) {
            return value;
        }

        mlir::Type srcType = value.getType();
        if (srcType == dstType) {
            return value;
        }

        if (dstType.isa<mlir::IndexType>()) {
            if (srcType.isa<mlir::IndexType>()) {
                return value;
            }
            if (srcType.isa<mlir::IntegerType>()) {
                return builder.create<mlir::arith::IndexCastOp>(UnknownLoc(), dstType, value);
            }
            if (srcType.isa<mlir::FloatType>()) {
                mlir::Value tmp = builder.create<mlir::arith::FPToSIOp>(UnknownLoc(), builder.getI64Type(), value);
                return builder.create<mlir::arith::IndexCastOp>(UnknownLoc(), dstType, tmp);
            }
            return value;
        }

        if (srcType.isa<mlir::IndexType>() && dstType.isa<mlir::IntegerType>()) {
            return builder.create<mlir::arith::IndexCastOp>(UnknownLoc(), dstType, value);
        }

        if (srcType.isa<mlir::IndexType>() && dstType.isa<mlir::FloatType>()) {
            mlir::Value tmp = builder.create<mlir::arith::IndexCastOp>(UnknownLoc(), builder.getI64Type(), value);
            return builder.create<mlir::arith::SIToFPOp>(UnknownLoc(), dstType, tmp);
        }

        if (auto srcFloat = srcType.dyn_cast<mlir::FloatType>()) {
            if (auto dstFloat = dstType.dyn_cast<mlir::FloatType>()) {
                if (srcFloat.getWidth() < dstFloat.getWidth()) {
                    return builder.create<mlir::arith::ExtFOp>(UnknownLoc(), dstType, value);
                }
                return builder.create<mlir::arith::TruncFOp>(UnknownLoc(), dstType, value);
            }
            if (dstType.isa<mlir::IntegerType>()) {
                return builder.create<mlir::arith::FPToSIOp>(UnknownLoc(), dstType, value);
            }
        }

        if (auto srcInt = srcType.dyn_cast<mlir::IntegerType>()) {
            if (auto dstInt = dstType.dyn_cast<mlir::IntegerType>()) {
                if (srcInt.getWidth() < dstInt.getWidth()) {
                    return builder.create<mlir::arith::ExtSIOp>(UnknownLoc(), dstType, value);
                }
                if (srcInt.getWidth() > dstInt.getWidth()) {
                    return builder.create<mlir::arith::TruncIOp>(UnknownLoc(), dstType, value);
                }
                return value;
            }
            if (dstType.isa<mlir::FloatType>()) {
                return builder.create<mlir::arith::SIToFPOp>(UnknownLoc(), dstType, value);
            }
        }

        return value;
    }

    // 实现类似 C++ 中的“算术类型提升”规则。
    mlir::Type MergeArithmeticType(mlir::Type lhsTy, mlir::Type rhsTy) {
        if (!lhsTy) return rhsTy;
        if (!rhsTy) return lhsTy;

        if (IsFloatType(lhsTy) || IsFloatType(rhsTy)) {
            bool useF64 = lhsTy.isF64() || rhsTy.isF64();
            return useF64 ? builder.getF64Type() : builder.getF32Type();
        }

        if (IsIndexType(lhsTy) || IsIndexType(rhsTy)) {
            return builder.getI64Type();
        }

        unsigned lhsWidth = lhsTy.cast<mlir::IntegerType>().getWidth();
        unsigned rhsWidth = rhsTy.cast<mlir::IntegerType>().getWidth();
        unsigned width = std::max(lhsWidth, rhsWidth);
        if (width <= 1) return builder.getI1Type();
        if (width <= 8) return builder.getI8Type();
        if (width <= 16) return builder.getI16Type();
        if (width <= 32) return builder.getI32Type();
        return builder.getI64Type();
    }

    // to bool的转换，非零数值转为true，零值或不支持的类型转为false。
    mlir::Value EnsureBool(mlir::Value v) {
        if (!v) {
            return BuildZero(builder.getI1Type());
        }

        mlir::Type ty = v.getType();
        if (ty.isInteger(1)) {
            return v;
        }

        if (IsFloatType(ty)) {
            mlir::Value zero = BuildZero(ty);
            return builder.create<mlir::arith::CmpFOp>(
                UnknownLoc(), mlir::arith::CmpFPredicate::ONE, v, zero);
        }

        mlir::Type intTy = GetIntegerLikeType(ty);
        mlir::Value casted = CastValueTo(v, intTy);
        mlir::Value zero = BuildZero(intTy);
        return builder.create<mlir::arith::CmpIOp>(
            UnknownLoc(), mlir::arith::CmpIPredicate::ne, casted, zero);
    }

    // 功能：处理Ensure Index相关逻辑。
    mlir::Value EnsureIndex(mlir::Value v) {
        if (!v) {
            return ConstIndex(0);
        }
        if (v.getType().isa<mlir::IndexType>()) {
            return v;
        }
        return CastValueTo(v, builder.getIndexType());
    }

    // 尝试从外部上下文（如 N，M 这样的全局常量）中解析并生成一个 MLIR 常量节点。
    bool TryResolveConstFromContext(llvm::StringRef name,
                                    mlir::Type preferredType,
                                    mlir::Value& out) {
        if (!options.contextSummary) {
            return false;
        }

        auto it = options.contextSummary->numericConsts.find(name.str());
        if (it == options.contextSummary->numericConsts.end()) {
            return false;
        }

        const double numericValue = it->second;
        if (preferredType && preferredType.isa<mlir::FloatType>()) {
            auto floatTy = preferredType.cast<mlir::FloatType>();
            auto attr = builder.getFloatAttr(floatTy, numericValue);
            out = builder.create<mlir::arith::ConstantOp>(UnknownLoc(), floatTy, attr);
            return true;
        }

        if (preferredType && preferredType.isa<mlir::IndexType>()) {
            out = ConstIndex(static_cast<int64_t>(std::llround(numericValue)));
            return true;
        }

        if (preferredType && preferredType.isa<mlir::IntegerType>()) {
            auto intTy = preferredType.cast<mlir::IntegerType>();
            out = builder.create<mlir::arith::ConstantIntOp>(
                UnknownLoc(), static_cast<int64_t>(std::llround(numericValue)), intTy.getWidth());
            return true;
        }

        auto attr = builder.getFloatAttr(builder.getF64Type(), numericValue);
        out = builder.create<mlir::arith::ConstantOp>(UnknownLoc(), builder.getF64Type(), attr);
        if (preferredType) {
            out = CastValueTo(out, preferredType);
        }
        return true;
    }

    // 找代码调用的函数是否属于外部注入的“纯内联辅助函数”（Pure Inline Function）。
    bool TryFindPureInlineFnInfo(llvm::StringRef calleeName,
                                 const CalcInlineFnInfo*& outInfo) {
        outInfo = nullptr;
        if (!options.contextSummary) {
            return false;
        }

        auto findDirect = [&](llvm::StringRef key) {
            auto it = options.contextSummary->pureInlineFns.find(key.str());
            if (it == options.contextSummary->pureInlineFns.end()) {
                return false;
            }
            outInfo = &it->second;
            return true;
        };

        if (findDirect(calleeName)) {
            return true;
        }

        size_t scopePos = calleeName.str().rfind("::");
        if (scopePos != std::string::npos) {
            return findDirect(calleeName.substr(scopePos + 2));
        }

        return false;
    }

    // 查找参数绑定信息，主要处理复杂 C++ 结构体在底层被拆解为多个物理指针的情况。
    const CalcParamBinding* FindParamBinding(llvm::StringRef logicalName) const {
        if (!options.paramBindings) {
            return nullptr;
        }
        for (const auto& binding : *options.paramBindings) {
            if (binding.logicalName == logicalName) {
                return &binding;
            }
        }
        return nullptr;
    }

    // 用于处理形如 struct_obj.field 这样的结构体/记录字段访问，将其映射回正确的底层物理张量类型和名称。
    mlir::Type LowerBoundRecordFieldParamType(clang::QualType qt,
                                              llvm::StringRef fieldType) {
        qt = qt.getCanonicalType();
        if (const auto* refType = qt->getAs<clang::ReferenceType>()) {
            qt = refType->getPointeeType().getCanonicalType();
        }

        mlir::Type elemTy = InferElementTypeFromString(fieldType.str());
        if (const auto* ptrType = qt->getAs<clang::PointerType>()) {
            (void)ptrType;
            return mlir::MemRefType::get({mlir::ShapedType::kDynamic}, elemTy);
        }

        const std::string typeStr = TypeToString(qt);
        if (ContainsToken(typeStr, "dacpp::Matrix")) {
            return mlir::MemRefType::get(
                {mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic}, elemTy);
        }
        if (ContainsToken(typeStr, "dacpp::Vector") || ContainsToken(typeStr, "dacpp::Tensor")) {
            return mlir::MemRefType::get({mlir::ShapedType::kDynamic}, elemTy);
        }

        return elemTy;
    }

    // 功能：提取或解析Resolve Access Path Base相关信息。
    bool ResolveAccessPathBase(const AccessPath& access,
                               std::string& resolvedBase,
                               mlir::Type* resolvedFieldType,
                               const clang::Stmt* nodeForDiag) {
        resolvedBase = access.base;
        if (resolvedFieldType) {
            *resolvedFieldType = mlir::Type();
        }
        if (access.fields.empty()) {
            return true;
        }

        if (access.fields.size() != 1) {
            return ReportUnsupported(nodeForDiag, "nested record field access is not supported");
        }

        const CalcParamBinding* binding = FindParamBinding(access.base);
        if (!binding) {
            return ReportUnsupported(
                nodeForDiag,
                "record field access has no param binding: " + access.base + "." + access.fields.front());
        }

        for (const auto& field : binding->fields) {
            if (field.fieldName != access.fields.front()) {
                continue;
            }
            resolvedBase = field.physicalName;
            if (resolvedFieldType) {
                *resolvedFieldType = InferElementTypeFromString(field.fieldType);
            }
            return true;
        }

        return ReportUnsupported(
            nodeForDiag,
            "record field binding not found: " + access.base + "." + access.fields.front());
    }

    // 功能：为Build Numeric Literal构建所需数据。
    mlir::Value BuildNumericLiteral(double value, mlir::Type preferredType) {
        mlir::Type useTy = preferredType;
        if (!useTy) {
            useTy = builder.getF64Type();
        }

        if (useTy.isa<mlir::FloatType>()) {
            auto floatTy = useTy.cast<mlir::FloatType>();
            auto attr = builder.getFloatAttr(floatTy, value);
            return builder.create<mlir::arith::ConstantOp>(UnknownLoc(), floatTy, attr);
        }

        if (useTy.isa<mlir::IndexType>()) {
            return ConstIndex(static_cast<int64_t>(std::llround(value)));
        }

        if (useTy.isa<mlir::IntegerType>()) {
            auto intTy = useTy.cast<mlir::IntegerType>();
            return builder.create<mlir::arith::ConstantIntOp>(
                UnknownLoc(), static_cast<int64_t>(std::llround(value)), intTy.getWidth());
        }

        auto attr = builder.getFloatAttr(builder.getF64Type(), value);
        return builder.create<mlir::arith::ConstantOp>(UnknownLoc(), builder.getF64Type(), attr);
    }

    // 这里实现了一个迷你的 C++ 算术表达式解析器。
    bool LowerInlineExprText(const std::string& exprText,
                             const std::unordered_map<std::string, mlir::Value>& locals,
                             mlir::Type preferredType,
                             unsigned inlineDepth,
                             mlir::Value& out) {
        struct InlineExprParser {
            CalcAstLowering* self;
            llvm::StringRef expr;
            const std::unordered_map<std::string, mlir::Value>& locals;
            mlir::Type preferredType;
            unsigned inlineDepth;
            size_t pos = 0;
            bool ok = true;

            // 功能：处理Skip Spaces相关逻辑。
            void SkipSpaces() {
                while (pos < expr.size() && std::isspace(static_cast<unsigned char>(expr[pos]))) {
                    ++pos;
                }
            }

            // 功能：提取或解析Parse Identifier相关信息。
            std::string ParseIdentifier() {
                if (pos >= expr.size() ||
                    (!std::isalpha(static_cast<unsigned char>(expr[pos])) && expr[pos] != '_')) {
                    ok = false;
                    return "";
                }

                size_t begin = pos++;
                while (pos < expr.size() &&
                       (std::isalnum(static_cast<unsigned char>(expr[pos])) || expr[pos] == '_')) {
                    ++pos;
                }

                return expr.substr(begin, pos - begin).str();
            }

            // 功能：提取或解析Parse Number相关信息。
            mlir::Value ParseNumber() {
                const char* beginPtr = expr.data() + pos;
                char* endPtr = nullptr;
                double value = std::strtod(beginPtr, &endPtr);
                if (endPtr == beginPtr) {
                    ok = false;
                    return mlir::Value();
                }

                pos += static_cast<size_t>(endPtr - beginPtr);
                while (pos < expr.size() && std::isalpha(static_cast<unsigned char>(expr[pos]))) {
                    ++pos;
                }
                return self->BuildNumericLiteral(value, preferredType);
            }

            // 功能：提取或解析Parse Factor相关信息。
            mlir::Value ParseFactor() {
                SkipSpaces();
                if (!ok || pos >= expr.size()) {
                    ok = false;
                    return mlir::Value();
                }

                if (expr[pos] == '+') {
                    ++pos;
                    return ParseFactor();
                }

                if (expr[pos] == '-') {
                    ++pos;
                    mlir::Value sub = ParseFactor();
                    if (!ok || !sub) {
                        return mlir::Value();
                    }
                    if (self->IsFloatType(sub.getType())) {
                        return self->builder.create<mlir::arith::NegFOp>(self->UnknownLoc(), sub);
                    }
                    return self->builder.create<mlir::arith::SubIOp>(
                        self->UnknownLoc(), self->BuildZero(sub.getType()), sub);
                }

                if (expr[pos] == '(') {
                    ++pos;
                    mlir::Value inside = ParseExpression();
                    SkipSpaces();
                    if (pos >= expr.size() || expr[pos] != ')') {
                        ok = false;
                        return mlir::Value();
                    }
                    ++pos;
                    return inside;
                }

                if (std::isdigit(static_cast<unsigned char>(expr[pos])) || expr[pos] == '.') {
                    return ParseNumber();
                }

                if (std::isalpha(static_cast<unsigned char>(expr[pos])) || expr[pos] == '_') {
                    std::string ident = ParseIdentifier();
                    if (!ok) {
                        return mlir::Value();
                    }

                    SkipSpaces();
                    if (pos < expr.size() && expr[pos] == '(') {
                        ++pos;
                        mlir::Value arg = ParseExpression();
                        SkipSpaces();
                        if (pos >= expr.size() || expr[pos] != ')') {
                            ok = false;
                            return mlir::Value();
                        }
                        ++pos;

                        mlir::Value callValue;
                        if (!self->LowerPureInlineCallFromExprText(
                                ident, arg, preferredType, inlineDepth + 1, callValue)) {
                            ok = false;
                            return mlir::Value();
                        }
                        return callValue;
                    }

                    auto localIt = locals.find(ident);
                    if (localIt != locals.end()) {
                        return localIt->second;
                    }

                    mlir::Value constValue;
                    if (self->TryResolveConstFromContext(ident, mlir::Type(), constValue)) {
                        return constValue;
                    }

                    if (ident == "M_PI") {
                        return self->BuildNumericLiteral(3.14159265358979323846, preferredType);
                    }
                    if (ident == "M_E") {
                        return self->BuildNumericLiteral(2.71828182845904523536, preferredType);
                    }

                    ok = false;
                    return mlir::Value();
                }

                ok = false;
                return mlir::Value();
            }

            // 功能：提取或解析Parse Term相关信息。
            mlir::Value ParseTerm() {
                mlir::Value lhs = ParseFactor();
                while (ok) {
                    SkipSpaces();
                    if (pos >= expr.size() || (expr[pos] != '*' && expr[pos] != '/')) {
                        break;
                    }
                    char op = expr[pos++];
                    mlir::Value rhs = ParseFactor();
                    if (!ok || !lhs || !rhs) {
                        ok = false;
                        return mlir::Value();
                    }

                    lhs = self->LowerBinaryArithmetic(
                        op == '*' ? clang::BO_Mul : clang::BO_Div,
                        lhs,
                        rhs,
                        preferredType);
                }
                return lhs;
            }

            // 功能：提取或解析Parse Expression相关信息。
            mlir::Value ParseExpression() {
                mlir::Value lhs = ParseTerm();
                while (ok) {
                    SkipSpaces();
                    if (pos >= expr.size() || (expr[pos] != '+' && expr[pos] != '-')) {
                        break;
                    }
                    char op = expr[pos++];
                    mlir::Value rhs = ParseTerm();
                    if (!ok || !lhs || !rhs) {
                        ok = false;
                        return mlir::Value();
                    }

                    lhs = self->LowerBinaryArithmetic(
                        op == '+' ? clang::BO_Add : clang::BO_Sub,
                        lhs,
                        rhs,
                        preferredType);
                }
                return lhs;
            }
        };

        InlineExprParser parser{this, exprText, locals, preferredType, inlineDepth};
        mlir::Value parsed = parser.ParseExpression();
        parser.SkipSpaces();
        if (!parser.ok || parser.pos != parser.expr.size() || !parsed) {
            return false;
        }

        out = preferredType ? CastValueTo(parsed, preferredType) : parsed;
        return true;
    }

    // 翻译对纯内联函数的调用。
    bool LowerPureInlineCallFromExprText(llvm::StringRef calleeName,
                                         mlir::Value arg,
                                         mlir::Type preferredType,
                                         unsigned inlineDepth,
                                         mlir::Value& out) {
        if (inlineDepth > options.maxInlineDepth) {
            return false;
        }

        const CalcInlineFnInfo* fnInfo = nullptr;
        if (!TryFindPureInlineFnInfo(calleeName, fnInfo) || !fnInfo) {
            return false;
        }
        if (fnInfo->paramNames.size() != 1 || fnInfo->returnExpr.empty()) {
            return false;
        }

        std::unordered_map<std::string, mlir::Value> locals;
        locals[fnInfo->paramNames[0]] = arg;
        return LowerInlineExprText(fnInfo->returnExpr, locals, preferredType, inlineDepth, out);
    }

    // 功能：生成或转换Lower Pure Inline Call相关结果。
    bool LowerPureInlineCall(const clang::CallExpr* callExpr,
                             llvm::StringRef calleeName,
                             mlir::Type preferredType,
                             unsigned inlineDepth,
                             mlir::Value& out) {
        if (!callExpr || inlineDepth > options.maxInlineDepth) {
            return false;
        }

        const CalcInlineFnInfo* fnInfo = nullptr;
        if (!TryFindPureInlineFnInfo(calleeName, fnInfo) || !fnInfo) {
            return false;
        }

        const clang::FunctionDecl* helperFn = fnInfo->functionDecl;
        const auto* body = helperFn ? llvm::dyn_cast_or_null<clang::CompoundStmt>(helperFn->getBody()) : nullptr;
        if (!helperFn || !body) {
            if (callExpr->getNumArgs() == 1) {
                mlir::Value arg;
                if (!LowerExprStrict(callExpr->getArg(0), arg)) {
                    return false;
                }
                return LowerPureInlineCallFromExprText(calleeName, arg, preferredType, inlineDepth, out);
            }
            return false;
        }

        if (callExpr->getNumArgs() != helperFn->getNumParams()) {
            return false;
        }

        std::unordered_set<std::string> touchedNames;
        std::unordered_map<std::string, SymbolInfo> savedSymbols;

        auto RestoreSymbols = [&]() {
            for (const std::string& name : touchedNames) {
                auto it = savedSymbols.find(name);
                if (it != savedSymbols.end()) {
                    symbols[name] = it->second;
                } else {
                    symbols.erase(name);
                }
            }
        };

        auto BindScalarValue = [&](const std::string& name,
                                   mlir::Value value,
                                   mlir::Type scalarTy) -> bool {
            if (name.empty() || !value || !scalarTy) {
                return false;
            }

            if (touchedNames.insert(name).second) {
                auto oldIt = symbols.find(name);
                if (oldIt != symbols.end()) {
                    savedSymbols.emplace(name, oldIt->second);
                }
            }

            mlir::Value casted = CastValueTo(value, scalarTy);
            mlir::MemRefType slotTy = mlir::MemRefType::get({1}, scalarTy);
            mlir::Value slot = builder.create<mlir::memref::AllocaOp>(UnknownLoc(), slotTy);
            builder.create<mlir::memref::StoreOp>(UnknownLoc(), casted, slot, mlir::ValueRange{ConstIndex(0)});
            symbols[name] = SymbolInfo{SymbolInfo::Kind::ScalarAlloca, slot, scalarTy};
            return true;
        };

        for (unsigned i = 0; i < helperFn->getNumParams(); ++i) {
            const clang::ParmVarDecl* paramDecl = helperFn->getParamDecl(i);
            if (!paramDecl) {
                RestoreSymbols();
                return false;
            }

            std::string paramName = paramDecl->getNameAsString();
            if (paramName.empty()) {
                paramName = "arg" + std::to_string(i);
            }

            mlir::Type paramTy = LowerScalarType(paramDecl->getType());
            mlir::Value argValue;
            if (!LowerExprStrict(callExpr->getArg(i), argValue, paramTy)) {
                RestoreSymbols();
                return false;
            }

            if (!BindScalarValue(paramName, argValue, paramTy)) {
                RestoreSymbols();
                return false;
            }
        }

        const unsigned previousHelperDepth = helperInlineDepth;
        helperInlineDepth = inlineDepth;

        bool sawReturn = false;
        for (const clang::Stmt* stmt : body->body()) {
            if (!stmt || llvm::isa<clang::NullStmt>(stmt)) {
                continue;
            }

            if (const auto* declStmt = llvm::dyn_cast<clang::DeclStmt>(stmt)) {
                for (const clang::Decl* decl : declStmt->decls()) {
                    const auto* varDecl = llvm::dyn_cast<clang::VarDecl>(decl);
                    if (!varDecl || !varDecl->hasInit()) {
                        helperInlineDepth = previousHelperDepth;
                        RestoreSymbols();
                        return false;
                    }

                    std::string localName = varDecl->getNameAsString();
                    if (localName.empty()) {
                        helperInlineDepth = previousHelperDepth;
                        RestoreSymbols();
                        return false;
                    }

                    mlir::Type localTy = LowerScalarType(varDecl->getType());
                    mlir::Value initValue;
                    if (!LowerExprStrict(varDecl->getInit(), initValue, localTy) ||
                        !BindScalarValue(localName, initValue, localTy)) {
                        helperInlineDepth = previousHelperDepth;
                        RestoreSymbols();
                        return false;
                    }
                }
                continue;
            }

            if (const auto* returnStmt = llvm::dyn_cast<clang::ReturnStmt>(stmt)) {
                if (sawReturn || !returnStmt->getRetValue()) {
                    helperInlineDepth = previousHelperDepth;
                    RestoreSymbols();
                    return false;
                }

                mlir::Type returnTy = preferredType ? preferredType : LowerScalarType(helperFn->getReturnType());
                if (!LowerExprStrict(returnStmt->getRetValue(), out, returnTy)) {
                    helperInlineDepth = previousHelperDepth;
                    RestoreSymbols();
                    return false;
                }
                if (preferredType) {
                    out = CastValueTo(out, preferredType);
                }
                sawReturn = true;
                continue;
            }

            helperInlineDepth = previousHelperDepth;
            RestoreSymbols();
            return false;
        }

        helperInlineDepth = previousHelperDepth;
        RestoreSymbols();
        return sawReturn;
    }

    // 将 C++ 的复杂寻址表达式拆解为 AccessPath 结构体。
    bool CollectAccessPath(const clang::Expr* expr, AccessPath& out) {
        if (!expr) {
            return false;
        }

        expr = expr->IgnoreParenImpCasts();

        if (const auto* ref = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
            out.base = ref->getDecl()->getNameAsString();
            return true;
        }

        if (const auto* sub = llvm::dyn_cast<clang::ArraySubscriptExpr>(expr)) {
            if (!CollectAccessPath(sub->getBase(), out)) {
                return false;
            }
            out.indices.push_back(sub->getIdx());
            return true;
        }

        if (const auto* member = llvm::dyn_cast<clang::MemberExpr>(expr)) {
            if (!CollectAccessPath(member->getBase(), out)) {
                return false;
            }
            out.fields.push_back(member->getMemberDecl()->getNameAsString());
            return true;
        }

        if (const auto* call = llvm::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
            if (call->getOperator() == clang::OverloadedOperatorKind::OO_Subscript &&
                call->getNumArgs() >= 2) {
                if (!CollectAccessPath(call->getArg(0), out)) {
                    return false;
                }
                out.indices.push_back(call->getArg(1));
                return true;
            }
        }

        return false;
    }

    // 将上一步拆解出的 AccessPath 转化为实际的 MLIR 读取操作。
    bool LoadFromAccessPathStrict(const AccessPath& access,
                                  mlir::Value& out,
                                  mlir::Type preferredType,
                                  const clang::Stmt* nodeForDiag = nullptr) {
        std::string resolvedBase;
        mlir::Type fieldType;
        if (!ResolveAccessPathBase(access, resolvedBase, &fieldType, nodeForDiag)) {
            return false;
        }

        mlir::Type effectivePreferredType = preferredType ? preferredType : fieldType;

        auto it = symbols.find(resolvedBase);
        if (it == symbols.end()) {
            if (access.indices.empty() &&
                access.fields.empty() &&
                // 功能：尝试处理Try Resolve Const From Context相关逻辑。
                TryResolveConstFromContext(resolvedBase, effectivePreferredType, out)) {
                return true;
            }
            return ReportUnsupported(nodeForDiag, "unknown symbol reference: " + resolvedBase);
        }

        const SymbolInfo& sym = it->second;
        if (sym.kind == SymbolInfo::Kind::LoopIndex) {
            out = CastValueTo(sym.value, effectivePreferredType ? effectivePreferredType : sym.value.getType());
            return true;
        }

        if (sym.kind == SymbolInfo::Kind::ScalarAlloca) {
            mlir::Value loaded = builder.create<mlir::memref::LoadOp>(
                UnknownLoc(), sym.value, mlir::ValueRange{ConstIndex(0)});
            out = effectivePreferredType ? CastValueTo(loaded, effectivePreferredType) : loaded;
            return true;
        }

        if (sym.kind == SymbolInfo::Kind::MemRef) {
            auto memTy = sym.value.getType().dyn_cast<mlir::MemRefType>();
            if (!memTy) {
                return ReportUnsupported(nodeForDiag, "symbol is not a memref: " + resolvedBase);
            }

            std::vector<mlir::Value> idxValues;
            idxValues.reserve(access.indices.size());
            for (const clang::Expr* idxExpr : access.indices) {
                mlir::Value idx;
                if (!LowerExprStrict(idxExpr, idx, builder.getI64Type())) {
                    return false;
                }
                idxValues.push_back(EnsureIndex(idx));
            }

            const int64_t rank = memTy.getRank();
            while (static_cast<int64_t>(idxValues.size()) < rank) {
                idxValues.push_back(ConstIndex(0));
            }
            if (rank >= 0 && static_cast<int64_t>(idxValues.size()) > rank) {
                idxValues.resize(rank);
            }

            mlir::Value loaded = builder.create<mlir::memref::LoadOp>(UnknownLoc(), sym.value, idxValues);
            out = effectivePreferredType ? CastValueTo(loaded, effectivePreferredType) : loaded;
            return true;
        }

        return ReportUnsupported(nodeForDiag, "unsupported symbol kind for access: " + resolvedBase);
    }

    // 功能：处理Load From Access Path相关逻辑。
    mlir::Value LoadFromAccessPath(const AccessPath& access, mlir::Type preferredType) {
        mlir::Value out;
        if (!LoadFromAccessPathStrict(access, out, preferredType, nullptr)) {
            return BuildZero(preferredType ? preferredType : builder.getF64Type());
        }
        return out;
    }

    //  构建左值信息。
    bool BuildLValueStrict(const clang::Expr* expr, LValueInfo& out) {
        if (!expr) {
            return ReportUnsupported(expr, "null lvalue expression");
        }

        expr = expr->IgnoreParenImpCasts();

        AccessPath access;
        if (CollectAccessPath(expr, access) && !access.base.empty()) {
            std::string resolvedBase;
            mlir::Type resolvedFieldType;
            if (!ResolveAccessPathBase(access, resolvedBase, &resolvedFieldType, expr)) {
                return false;
            }

            auto it = symbols.find(resolvedBase);
            if (it == symbols.end()) {
                return ReportUnsupported(expr, "unknown lvalue symbol: " + resolvedBase);
            }

            const SymbolInfo& sym = it->second;
            if (sym.kind == SymbolInfo::Kind::MemRef) {
                auto memTy = sym.value.getType().dyn_cast<mlir::MemRefType>();
                if (!memTy) {
                    return ReportUnsupported(expr, "lvalue symbol is not memref: " + resolvedBase);
                }

                out.valid = true;
                out.memref = sym.value;
                out.elementType = resolvedFieldType ? resolvedFieldType : memTy.getElementType();

                for (const clang::Expr* idxExpr : access.indices) {
                    mlir::Value idx;
                    if (!LowerExprStrict(idxExpr, idx, builder.getI64Type())) {
                        return false;
                    }
                    out.indices.push_back(EnsureIndex(idx));
                }
                while (static_cast<int64_t>(out.indices.size()) < memTy.getRank()) {
                    out.indices.push_back(ConstIndex(0));
                }
                if (memTy.getRank() >= 0 && static_cast<int64_t>(out.indices.size()) > memTy.getRank()) {
                    out.indices.resize(memTy.getRank());
                }
                return true;
            }

            if (sym.kind == SymbolInfo::Kind::ScalarAlloca) {
                out.valid = true;
                out.memref = sym.value;
                out.indices.push_back(ConstIndex(0));
                out.elementType = sym.elementType;
                return true;
            }

            return ReportUnsupported(expr, "unsupported lvalue symbol kind: " + access.base);
        }

        if (const auto* ref = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
            auto it = symbols.find(ref->getDecl()->getNameAsString());
            if (it != symbols.end() && it->second.kind == SymbolInfo::Kind::ScalarAlloca) {
                out.valid = true;
                out.memref = it->second.value;
                out.indices.push_back(ConstIndex(0));
                out.elementType = it->second.elementType;
                return true;
            }
        }

        return ReportUnsupported(expr, "expression is not assignable lvalue");
    }

    // 翻译变量引用表达式。决定是去内存读取这个变量、使用常量，还是展开枚举值。
    bool LowerDeclRefExpr(const clang::DeclRefExpr* declRef,
                          mlir::Value& out,
                          mlir::Type preferredType) {
        const std::string name = declRef->getDecl()->getNameAsString();
        auto it = symbols.find(name);
        if (it != symbols.end()) {
            return LoadFromAccessPathStrict({name, {}, {}}, out, preferredType, declRef);
        }

        if (TryResolveConstFromContext(name, preferredType, out)) {
            return true;
        }

        if (const auto* varDecl = llvm::dyn_cast<clang::VarDecl>(declRef->getDecl())) {
            if (varDecl->hasInit()) {
                if (declInlineDepth >= options.maxInlineDepth) {
                    return ReportUnsupported(declRef,
                                             "inline decl-ref depth exceeded for: " + name);
                }
                if (activeDeclInits.count(varDecl) > 0) {
                    return ReportUnsupported(declRef,
                                             "cyclic decl-ref init detected: " + name);
                }

                activeDeclInits.insert(varDecl);
                ++declInlineDepth;
                bool lowered = LowerExprStrict(varDecl->getInit(), out, preferredType);
                --declInlineDepth;
                activeDeclInits.erase(varDecl);
                return lowered;
            }
        }

        if (const auto* enumConst = llvm::dyn_cast<clang::EnumConstantDecl>(declRef->getDecl())) {
            llvm::APSInt val = enumConst->getInitVal();
            mlir::Type intTy = preferredType ? preferredType : builder.getI64Type();
            if (intTy.isa<mlir::FloatType>()) {
                intTy = builder.getI64Type();
            }

            if (intTy.isa<mlir::IndexType>()) {
                out = CastValueTo(ConstIndex(val.getSExtValue()), preferredType ? preferredType : builder.getIndexType());
                return true;
            }

            if (!intTy.isa<mlir::IntegerType>()) {
                intTy = builder.getI64Type();
            }

            mlir::Value cst = builder.create<mlir::arith::ConstantIntOp>(
                UnknownLoc(), val.getSExtValue(), intTy.cast<mlir::IntegerType>().getWidth());
            out = CastValueTo(cst, preferredType ? preferredType : cst.getType());
            return true;
        }

        return ReportUnsupported(declRef, "unknown variable reference: " + name);
    }

    // 翻译加减乘除与取模（+, -, *, /, %）。
    // 它会自动调用前面提到的 MergeArithmeticType 进行类型对齐，然后生成诸如 arith.addf 或 arith.addi 的 MLIR 操作。
    mlir::Value LowerBinaryArithmetic(clang::BinaryOperatorKind opcode,
                                      mlir::Value lhs,
                                      mlir::Value rhs,
                                      mlir::Type preferredType) {
        mlir::Type mergedTy = MergeArithmeticType(lhs.getType(), rhs.getType());
        lhs = CastValueTo(lhs, mergedTy);
        rhs = CastValueTo(rhs, mergedTy);

        switch (opcode) {
        case clang::BO_Add:
            if (IsFloatType(mergedTy)) return builder.create<mlir::arith::AddFOp>(UnknownLoc(), lhs, rhs);
            return builder.create<mlir::arith::AddIOp>(UnknownLoc(), lhs, rhs);
        case clang::BO_Sub:
            if (IsFloatType(mergedTy)) return builder.create<mlir::arith::SubFOp>(UnknownLoc(), lhs, rhs);
            return builder.create<mlir::arith::SubIOp>(UnknownLoc(), lhs, rhs);
        case clang::BO_Mul:
            if (IsFloatType(mergedTy)) return builder.create<mlir::arith::MulFOp>(UnknownLoc(), lhs, rhs);
            return builder.create<mlir::arith::MulIOp>(UnknownLoc(), lhs, rhs);
        case clang::BO_Div:
            if (IsFloatType(mergedTy)) return builder.create<mlir::arith::DivFOp>(UnknownLoc(), lhs, rhs);
            return builder.create<mlir::arith::DivSIOp>(UnknownLoc(), lhs, rhs);
        case clang::BO_Rem:
            if (IsFloatType(mergedTy)) return builder.create<mlir::arith::RemFOp>(UnknownLoc(), lhs, rhs);
            return builder.create<mlir::arith::RemSIOp>(UnknownLoc(), lhs, rhs);
        default:
            break;
        }

        mlir::Value zero = BuildZero(preferredType ? preferredType : mergedTy);
        return zero;
    }

    // 翻译比较操作符（<, <=, >, >=, ==, !=），生成 arith.cmpf 或 arith.cmpi 节点。
    mlir::Value LowerBinaryCompare(clang::BinaryOperatorKind opcode,
                                   mlir::Value lhs,
                                   mlir::Value rhs) {
        mlir::Type mergedTy = MergeArithmeticType(lhs.getType(), rhs.getType());
        lhs = CastValueTo(lhs, mergedTy);
        rhs = CastValueTo(rhs, mergedTy);

        if (IsFloatType(mergedTy)) {
            mlir::arith::CmpFPredicate pred = mlir::arith::CmpFPredicate::OEQ;
            switch (opcode) {
            case clang::BO_LT: pred = mlir::arith::CmpFPredicate::OLT; break;
            case clang::BO_LE: pred = mlir::arith::CmpFPredicate::OLE; break;
            case clang::BO_GT: pred = mlir::arith::CmpFPredicate::OGT; break;
            case clang::BO_GE: pred = mlir::arith::CmpFPredicate::OGE; break;
            case clang::BO_EQ: pred = mlir::arith::CmpFPredicate::OEQ; break;
            case clang::BO_NE: pred = mlir::arith::CmpFPredicate::ONE; break;
            default: break;
            }
            return builder.create<mlir::arith::CmpFOp>(UnknownLoc(), pred, lhs, rhs);
        }

        mlir::arith::CmpIPredicate pred = mlir::arith::CmpIPredicate::eq;
        switch (opcode) {
        case clang::BO_LT: pred = mlir::arith::CmpIPredicate::slt; break;
        case clang::BO_LE: pred = mlir::arith::CmpIPredicate::sle; break;
        case clang::BO_GT: pred = mlir::arith::CmpIPredicate::sgt; break;
        case clang::BO_GE: pred = mlir::arith::CmpIPredicate::sge; break;
        case clang::BO_EQ: pred = mlir::arith::CmpIPredicate::eq; break;
        case clang::BO_NE: pred = mlir::arith::CmpIPredicate::ne; break;
        default: break;
        }
        return builder.create<mlir::arith::CmpIOp>(UnknownLoc(), pred, lhs, rhs);
    }

    // 翻译函数调用。内置支持了数学函数 exp, min (映射为 arith.select), max，
    // 对于其他函数尝试调用 LowerPureInlineCall，如果都不行则报错。
    bool LowerCallExpr(const clang::CallExpr* callExpr,
                       mlir::Value& out,
                       mlir::Type preferredType) {
        std::string calleeName;
        if (const clang::FunctionDecl* direct = callExpr->getDirectCallee()) {
            calleeName = direct->getQualifiedNameAsString();
        }

        auto containsName = [&](llvm::StringRef token) {
            return calleeName.find(token.str()) != std::string::npos;
        };

        if (containsName("exp") && callExpr->getNumArgs() >= 1) {
            mlir::Type outTy = preferredType && IsFloatType(preferredType) ? preferredType : builder.getF64Type();
            mlir::Value arg;
            if (!LowerExprStrict(callExpr->getArg(0), arg, outTy)) {
                return false;
            }
            arg = CastValueTo(arg, outTy);
            out = builder.create<mlir::math::ExpOp>(UnknownLoc(), arg);
            return true;
        }

        if ((containsName("fmin") || containsName("min")) && callExpr->getNumArgs() >= 2) {
            mlir::Value lhs;
            mlir::Value rhs;
            if (!LowerExprStrict(callExpr->getArg(0), lhs) ||
                !LowerExprStrict(callExpr->getArg(1), rhs)) {
                return false;
            }
            mlir::Type merged = MergeArithmeticType(lhs.getType(), rhs.getType());
            lhs = CastValueTo(lhs, merged);
            rhs = CastValueTo(rhs, merged);
            mlir::Value cond;
            if (IsFloatType(merged)) {
                cond = builder.create<mlir::arith::CmpFOp>(UnknownLoc(), mlir::arith::CmpFPredicate::OLT, lhs, rhs);
            } else {
                cond = builder.create<mlir::arith::CmpIOp>(UnknownLoc(), mlir::arith::CmpIPredicate::slt, lhs, rhs);
            }
            out = builder.create<mlir::arith::SelectOp>(UnknownLoc(), cond, lhs, rhs);
            if (preferredType) {
                out = CastValueTo(out, preferredType);
            }
            return true;
        }

        if ((containsName("fmax") || containsName("max")) && callExpr->getNumArgs() >= 2) {
            mlir::Value lhs;
            mlir::Value rhs;
            if (!LowerExprStrict(callExpr->getArg(0), lhs) ||
                !LowerExprStrict(callExpr->getArg(1), rhs)) {
                return false;
            }
            mlir::Type merged = MergeArithmeticType(lhs.getType(), rhs.getType());
            lhs = CastValueTo(lhs, merged);
            rhs = CastValueTo(rhs, merged);
            mlir::Value cond;
            if (IsFloatType(merged)) {
                cond = builder.create<mlir::arith::CmpFOp>(UnknownLoc(), mlir::arith::CmpFPredicate::OGT, lhs, rhs);
            } else {
                cond = builder.create<mlir::arith::CmpIOp>(UnknownLoc(), mlir::arith::CmpIPredicate::sgt, lhs, rhs);
            }
            out = builder.create<mlir::arith::SelectOp>(UnknownLoc(), cond, lhs, rhs);
            if (preferredType) {
                out = CastValueTo(out, preferredType);
            }
            return true;
        }

        if (!calleeName.empty()) {
            const unsigned nextInlineDepth = helperInlineDepth + 1;
            if (LowerPureInlineCall(callExpr, calleeName, preferredType, nextInlineDepth, out)) {
                return true;
            }
        }

        if (calleeName.empty()) {
            return ReportUnsupported(callExpr, "unsupported indirect call expression");
        }
        return ReportUnsupported(callExpr, "unsupported function call: " + calleeName);
    }

    // 表达式降级的总调度中枢。
    // 根据传入 AST Expr 的具体子类型（字面量、一元操作符如 !x 或 x++、二元操作符、函数调用等），
    // 派发到上面不同的具体处理函数中去。
    bool LowerExprStrict(const clang::Expr* expr,
                         mlir::Value& out,
                         mlir::Type preferredType = mlir::Type()) {
        if (!expr) {
            return ReportUnsupported(expr, "null expression");
        }

        expr = expr->IgnoreParenImpCasts();

        if (const auto* intLit = llvm::dyn_cast<clang::IntegerLiteral>(expr)) {
            int64_t v = intLit->getValue().getSExtValue();
            mlir::Type intTy = preferredType && IsIntegerType(preferredType) ? preferredType : builder.getI64Type();
            if (intTy.isa<mlir::IndexType>()) {
                out = ConstIndex(v);
                return true;
            }
            out = builder.create<mlir::arith::ConstantIntOp>(
                UnknownLoc(), v, intTy.cast<mlir::IntegerType>().getWidth());
            if (preferredType) {
                out = CastValueTo(out, preferredType);
            }
            return true;
        }

        if (const auto* floatLit = llvm::dyn_cast<clang::FloatingLiteral>(expr)) {
            mlir::Type outTy = preferredType && IsFloatType(preferredType) ? preferredType : builder.getF64Type();
            double v = floatLit->getValueAsApproximateDouble();
            auto attr = builder.getFloatAttr(outTy.cast<mlir::FloatType>(), v);
            out = builder.create<mlir::arith::ConstantOp>(UnknownLoc(), outTy, attr);
            return true;
        }

        if (const auto* boolLit = llvm::dyn_cast<clang::CXXBoolLiteralExpr>(expr)) {
            out = builder.create<mlir::arith::ConstantIntOp>(UnknownLoc(), boolLit->getValue() ? 1 : 0, 1);
            return true;
        }

        AccessPath access;
        if (CollectAccessPath(expr, access) && !access.base.empty()) {
            const bool isSimpleRef = access.indices.empty() && access.fields.empty();
            const bool hasSymbol = symbols.find(access.base) != symbols.end();
            const bool hasContextConst =
                options.contextSummary &&
                options.contextSummary->numericConsts.find(access.base) !=
                    options.contextSummary->numericConsts.end();

            if (!isSimpleRef || hasSymbol || hasContextConst) {
                if (!LoadFromAccessPathStrict(access, out, preferredType, expr)) {
                    return false;
                }
                if (preferredType) {
                    out = CastValueTo(out, preferredType);
                }
                return true;
            }
        }

        if (const auto* declRef = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
            return LowerDeclRefExpr(declRef, out, preferredType);
        }

        if (const auto* unaryOp = llvm::dyn_cast<clang::UnaryOperator>(expr)) {
            mlir::Value sub;
            if (!LowerExprStrict(unaryOp->getSubExpr(), sub)) {
                return false;
            }
            switch (unaryOp->getOpcode()) {
            case clang::UO_Minus:
                if (IsFloatType(sub.getType())) {
                    out = builder.create<mlir::arith::NegFOp>(UnknownLoc(), sub);
                    return true;
                }
                out = builder.create<mlir::arith::SubIOp>(UnknownLoc(), BuildZero(sub.getType()), sub);
                return true;
            case clang::UO_LNot:
                out = builder.create<mlir::arith::XOrIOp>(
                    UnknownLoc(), EnsureBool(sub),
                    builder.create<mlir::arith::ConstantIntOp>(UnknownLoc(), 1, 1));
                return true;
            case clang::UO_PostInc:
            case clang::UO_PreInc:
            case clang::UO_PostDec:
            case clang::UO_PreDec: {
                LValueInfo lv;
                if (!BuildLValueStrict(unaryOp->getSubExpr(), lv) || !lv.valid) {
                    return ReportUnsupported(unaryOp, "increment/decrement requires assignable lvalue");
                }
                mlir::Value oldV = builder.create<mlir::memref::LoadOp>(UnknownLoc(), lv.memref, lv.indices);
                int64_t step = (unaryOp->getOpcode() == clang::UO_PostDec || unaryOp->getOpcode() == clang::UO_PreDec) ? -1 : 1;
                mlir::Value stepV = builder.create<mlir::arith::ConstantIntOp>(
                    UnknownLoc(), step, GetIntegerLikeType(oldV.getType()).cast<mlir::IntegerType>().getWidth());
                mlir::Value newV;
                if (IsFloatType(oldV.getType())) {
                    mlir::Value stepF = CastValueTo(stepV, oldV.getType());
                    newV = builder.create<mlir::arith::AddFOp>(UnknownLoc(), oldV, stepF);
                } else {
                    mlir::Value oldI = CastValueTo(oldV, GetIntegerLikeType(oldV.getType()));
                    mlir::Value stepI = CastValueTo(stepV, oldI.getType());
                    newV = builder.create<mlir::arith::AddIOp>(UnknownLoc(), oldI, stepI);
                    newV = CastValueTo(newV, oldV.getType());
                }
                builder.create<mlir::memref::StoreOp>(UnknownLoc(), newV, lv.memref, lv.indices);
                out = (unaryOp->isPostfix() ? oldV : newV);
                return true;
            }
            default:
                return ReportUnsupported(unaryOp, "unsupported unary operator in expression");
            }
        }

        if (const auto* callExpr = llvm::dyn_cast<clang::CallExpr>(expr)) {
            return LowerCallExpr(callExpr, out, preferredType);
        }

        if (const auto* binaryOp = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
            clang::BinaryOperatorKind opcode = binaryOp->getOpcode();

            if (binaryOp->isLogicalOp()) {
                mlir::Value lhs;
                mlir::Value rhs;
                if (!LowerExprStrict(binaryOp->getLHS(), lhs) ||
                    !LowerExprStrict(binaryOp->getRHS(), rhs)) {
                    return false;
                }
                lhs = EnsureBool(lhs);
                rhs = EnsureBool(rhs);
                if (opcode == clang::BO_LAnd) {
                    out = builder.create<mlir::arith::AndIOp>(UnknownLoc(), lhs, rhs);
                    return true;
                }
                out = builder.create<mlir::arith::OrIOp>(UnknownLoc(), lhs, rhs);
                return true;
            }

            if (binaryOp->isComparisonOp()) {
                mlir::Value lhs;
                mlir::Value rhs;
                if (!LowerExprStrict(binaryOp->getLHS(), lhs) ||
                    !LowerExprStrict(binaryOp->getRHS(), rhs)) {
                    return false;
                }
                out = LowerBinaryCompare(opcode, lhs, rhs);
                return true;
            }

            if (binaryOp->isAdditiveOp() || binaryOp->isMultiplicativeOp()) {
                mlir::Value lhs;
                mlir::Value rhs;
                if (!LowerExprStrict(binaryOp->getLHS(), lhs) ||
                    !LowerExprStrict(binaryOp->getRHS(), rhs)) {
                    return false;
                }
                out = LowerBinaryArithmetic(opcode, lhs, rhs, preferredType);
                if (preferredType) {
                    out = CastValueTo(out, preferredType);
                }
                return true;
            }

            return ReportUnsupported(binaryOp, "unsupported binary operator in expression");
        }

        if (const auto* ctorExpr = llvm::dyn_cast<clang::CXXConstructExpr>(expr)) {
            if (ctorExpr->getNumArgs() > 0) {
                return LowerExprStrict(ctorExpr->getArg(0), out, preferredType);
            }
            return ReportUnsupported(ctorExpr, "unsupported constructor expression without arguments");
        }

        return ReportUnsupported(expr, "unsupported expression kind");
    }

    // 功能：生成或转换Lower Expr相关结果。
    mlir::Value LowerExpr(const clang::Expr* expr, mlir::Type preferredType = mlir::Type()) {
        mlir::Value out;
        if (!LowerExprStrict(expr, out, preferredType)) {
            if (preferredType) {
                return BuildZero(preferredType);
            }
            return BuildZero(builder.getF64Type());
        }
        return out;
    }

    // 处理赋值和复合赋值（=, +=, -= 等）
    bool LowerAssignmentLike(const clang::BinaryOperator* binOp) {
        LValueInfo lhs;
        if (!BuildLValueStrict(binOp->getLHS(), lhs) || !lhs.valid) {
            return ReportUnsupported(binOp, "assignment lhs is not assignable");
        }

        mlir::Value rhs;
        if (binOp->isCompoundAssignmentOp()) {
            mlir::Value oldV = builder.create<mlir::memref::LoadOp>(UnknownLoc(), lhs.memref, lhs.indices);
            mlir::Value rhsV;
            if (!LowerExprStrict(binOp->getRHS(), rhsV)) {
                return false;
            }

            clang::BinaryOperatorKind actual = clang::BO_Assign;
            switch (binOp->getOpcode()) {
            case clang::BO_AddAssign: actual = clang::BO_Add; break;
            case clang::BO_SubAssign: actual = clang::BO_Sub; break;
            case clang::BO_MulAssign: actual = clang::BO_Mul; break;
            case clang::BO_DivAssign: actual = clang::BO_Div; break;
            default: actual = clang::BO_Assign; break;
            }

            if (actual == clang::BO_Assign) {
                rhs = rhsV;
            } else {
                rhs = LowerBinaryArithmetic(actual, oldV, rhsV, lhs.elementType);
            }
        } else {
            if (!LowerExprStrict(binOp->getRHS(), rhs, lhs.elementType)) {
                return false;
            }
        }

        rhs = CastValueTo(rhs, lhs.elementType);
        builder.create<mlir::memref::StoreOp>(UnknownLoc(), rhs, lhs.memref, lhs.indices);
        return true;
    }

    // 基于 AST 的模式匹配函数。
    // 这三个函数专门用来解析标准的、可预测的 C++ 数组遍历循环模式，
    // 从中提取出：归纳变量名（比如 i），下界（如 0），上界（如 N），以及步长（如 1）。
    bool ParseForLoopIVName(const clang::Stmt* initStmt,
                            std::string& ivName,
                            const clang::Expr*& lowerExpr) {
        lowerExpr = nullptr;
        if (!initStmt) {
            return false;
        }

        if (const auto* declStmt = llvm::dyn_cast<clang::DeclStmt>(initStmt)) {
            if (declStmt->isSingleDecl()) {
                if (const auto* varDecl = llvm::dyn_cast<clang::VarDecl>(declStmt->getSingleDecl())) {
                    ivName = varDecl->getNameAsString();
                    lowerExpr = varDecl->getInit();
                    return !ivName.empty();
                }
            }
        }

        if (const auto* expr = llvm::dyn_cast<clang::Expr>(initStmt)) {
            if (const auto* bin = llvm::dyn_cast<clang::BinaryOperator>(expr->IgnoreParenImpCasts())) {
                if (bin->isAssignmentOp()) {
                    if (const auto* lhsRef = llvm::dyn_cast<clang::DeclRefExpr>(bin->getLHS()->IgnoreParenImpCasts())) {
                        ivName = lhsRef->getDecl()->getNameAsString();
                        lowerExpr = bin->getRHS();
                        return !ivName.empty();
                    }
                }
            }
        }

        return false;
    }

    // 同上2
    bool ParseForLoopCond(const clang::Expr* condExpr,
                          const std::string& ivName,
                          const clang::Expr*& upperExpr,
                          bool& inclusiveUpper,
                          bool& isAscending) {
        upperExpr = nullptr;
        inclusiveUpper = false;
        isAscending = true;

        if (!condExpr) {
            return false;
        }

        const auto* bin = llvm::dyn_cast<clang::BinaryOperator>(condExpr->IgnoreParenImpCasts());
        if (!bin || !bin->isComparisonOp()) {
            return false;
        }

        const auto* lhsRef = llvm::dyn_cast<clang::DeclRefExpr>(bin->getLHS()->IgnoreParenImpCasts());
        const auto* rhsRef = llvm::dyn_cast<clang::DeclRefExpr>(bin->getRHS()->IgnoreParenImpCasts());

        clang::BinaryOperatorKind op = bin->getOpcode();
        if (lhsRef && lhsRef->getDecl()->getNameAsString() == ivName) {
            upperExpr = bin->getRHS();
            if (op == clang::BO_LT) {
                inclusiveUpper = false;
                isAscending = true;
                return true;
            }
            if (op == clang::BO_LE) {
                inclusiveUpper = true;
                isAscending = true;
                return true;
            }
            if (op == clang::BO_GT) {
                inclusiveUpper = false;
                isAscending = false;
                return true;
            }
            if (op == clang::BO_GE) {
                inclusiveUpper = true;
                isAscending = false;
                return true;
            }
            return false;
        }

        if (rhsRef && rhsRef->getDecl()->getNameAsString() == ivName) {
            upperExpr = bin->getLHS();
            if (op == clang::BO_GT) {
                inclusiveUpper = false;
                isAscending = true;
                return true;
            }
            if (op == clang::BO_GE) {
                inclusiveUpper = true;
                isAscending = true;
                return true;
            }
            if (op == clang::BO_LT) {
                inclusiveUpper = false;
                isAscending = false;
                return true;
            }
            if (op == clang::BO_LE) {
                inclusiveUpper = true;
                isAscending = false;
                return true;
            }
        }

        return false;
    }

    // 同上3
    bool ParseForLoopStep(const clang::Expr* incExpr,
                          const std::string& ivName,
                          int64_t& stepValue,
                          bool& isAscending) {
        stepValue = 1;

        if (!incExpr) {
            return true;
        }

        const clang::Expr* stripped = incExpr->IgnoreParenImpCasts();
        if (const auto* unary = llvm::dyn_cast<clang::UnaryOperator>(stripped)) {
            if (const auto* ref = llvm::dyn_cast<clang::DeclRefExpr>(unary->getSubExpr()->IgnoreParenImpCasts())) {
                if (ref->getDecl()->getNameAsString() != ivName) {
                    return false;
                }
                if (unary->getOpcode() == clang::UO_PreInc || unary->getOpcode() == clang::UO_PostInc) {
                    stepValue = 1;
                    isAscending = true;
                    return true;
                }
                if (unary->getOpcode() == clang::UO_PreDec || unary->getOpcode() == clang::UO_PostDec) {
                    stepValue = -1;
                    isAscending = false;
                    return true;
                }
            }
        }

        if (const auto* bin = llvm::dyn_cast<clang::BinaryOperator>(stripped)) {
            if (const auto* lhsRef = llvm::dyn_cast<clang::DeclRefExpr>(bin->getLHS()->IgnoreParenImpCasts())) {
                if (lhsRef->getDecl()->getNameAsString() != ivName) {
                    return false;
                }

                if (const auto* rhsInt = llvm::dyn_cast<clang::IntegerLiteral>(bin->getRHS()->IgnoreParenImpCasts())) {
                    int64_t rhsVal = rhsInt->getValue().getSExtValue();
                    if (bin->getOpcode() == clang::BO_AddAssign) {
                        stepValue = rhsVal;
                        isAscending = rhsVal >= 0;
                        return true;
                    }
                    if (bin->getOpcode() == clang::BO_SubAssign) {
                        stepValue = -rhsVal;
                        isAscending = rhsVal <= 0;
                        return true;
                    }
                }
            }
        }

        return false;
    }

    // 将 C++ 的 for 循环翻译为 MLIR 结构化的 scf.for 循环。
    bool LowerForStmt(const clang::ForStmt* forStmt) {
        std::string ivName;
        const clang::Expr* lowerExprNode = nullptr;
        if (!ParseForLoopIVName(forStmt->getInit(), ivName, lowerExprNode)) {
            return ReportUnsupported(forStmt, "unsupported for-loop init form");
        }

        const clang::Expr* upperExprNode = nullptr;
        bool inclusiveUpper = false;
        bool ascendingFromCond = true;
        if (!ParseForLoopCond(forStmt->getCond(), ivName, upperExprNode, inclusiveUpper, ascendingFromCond)) {
            return ReportUnsupported(forStmt, "unsupported for-loop condition form");
        }

        int64_t stepValue = 1;
        bool ascendingFromStep = true;
        if (!ParseForLoopStep(forStmt->getInc(), ivName, stepValue, ascendingFromStep)) {
            return ReportUnsupported(forStmt, "unsupported for-loop step form");
        }

        if (stepValue == 0) {
            return ReportUnsupported(forStmt, "for-loop step must not be zero");
        }

        if (!(ascendingFromCond && ascendingFromStep && stepValue > 0)) {
            return ReportUnsupported(forStmt, "unsupported non-ascending for-loop");
        }

        mlir::Value lowerRaw;
        mlir::Value upperRaw;
        if (!LowerExprStrict(lowerExprNode, lowerRaw, builder.getI64Type()) ||
            !LowerExprStrict(upperExprNode, upperRaw, builder.getI64Type())) {
            return false;
        }

        mlir::Value lower = EnsureIndex(lowerRaw);
        mlir::Value upper = EnsureIndex(upperRaw);
        mlir::Value step = EnsureIndex(builder.create<mlir::arith::ConstantIntOp>(UnknownLoc(), stepValue, 64));

        if (inclusiveUpper) {
            upper = builder.create<mlir::arith::AddIOp>(UnknownLoc(), upper, ConstIndex(1));
        }

        mlir::scf::ForOp loop = builder.create<mlir::scf::ForOp>(UnknownLoc(), lower, upper, step);

        auto oldIter = symbols.find(ivName);
        bool hadOld = oldIter != symbols.end();
        SymbolInfo oldValue;
        if (hadOld) {
            oldValue = oldIter->second;
        }

        symbols[ivName] = SymbolInfo{SymbolInfo::Kind::LoopIndex, loop.getInductionVar(), builder.getIndexType()};

        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(loop.getBody());
            if (!LowerStmtStrict(forStmt->getBody())) {
                return false;
            }
        }

        if (hadOld) {
            symbols[ivName] = oldValue;
        } else {
            symbols.erase(ivName);
        }

        return true;
    }

    // 处理局部变量声明。它会在 MLIR 中生成一个大小为 1 的局部内存分配（memref.alloca），并将初始值存入其中。
    bool LowerDeclStmt(const clang::DeclStmt* declStmt) {
        for (const clang::Decl* decl : declStmt->decls()) {
            const auto* varDecl = llvm::dyn_cast<clang::VarDecl>(decl);
            if (!varDecl) {
                continue;
            }

            std::string name = varDecl->getNameAsString();
            if (name.empty()) {
                continue;
            }

            mlir::Type varType = LowerParamType(varDecl->getType());

            if (auto memTy = varType.dyn_cast<mlir::MemRefType>()) {
                // 先处理 "引用到已有 memref" 的简单别名场景。
                if (varDecl->hasInit()) {
                    AccessPath access;
                    if (CollectAccessPath(varDecl->getInit()->IgnoreParenImpCasts(), access)) {
                        auto it = symbols.find(access.base);
                        if (it != symbols.end() && it->second.kind == SymbolInfo::Kind::MemRef) {
                            symbols[name] = it->second;
                            continue;
                        }
                    }
                }

                // 复杂场景暂时不生成本地 memref，直接记录诊断并跳过。
                (void)memTy;
                return ReportUnsupported(varDecl->getInit(), "unsupported local non-scalar declaration: " + name);
            }

            mlir::MemRefType slotType = mlir::MemRefType::get({1}, varType);
            mlir::Value slot = builder.create<mlir::memref::AllocaOp>(UnknownLoc(), slotType);
            symbols[name] = SymbolInfo{SymbolInfo::Kind::ScalarAlloca, slot, varType};

            mlir::Value initVal;
            if (varDecl->hasInit()) {
                if (!LowerExprStrict(varDecl->getInit(), initVal, varType)) {
                    return false;
                }
            } else {
                initVal = BuildZero(varType);
            }
            initVal = CastValueTo(initVal, varType);
            builder.create<mlir::memref::StoreOp>(UnknownLoc(), initVal, slot, mlir::ValueRange{ConstIndex(0)});
        }
        return true;
    }

    // 将 C++ 的 if-else 翻译为 MLIR 的 scf.if，包含条件判断和块展开。
    bool LowerIfStmt(const clang::IfStmt* ifStmt) {
        mlir::Value condRaw;
        if (!LowerExprStrict(ifStmt->getCond(), condRaw)) {
            return false;
        }
        mlir::Value cond = EnsureBool(condRaw);
        bool hasElse = ifStmt->getElse() != nullptr;
        mlir::scf::IfOp ifOp = builder.create<mlir::scf::IfOp>(UnknownLoc(), cond, hasElse);

        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
            if (!LowerStmtStrict(ifStmt->getThen())) {
                return false;
            }
        }

        if (hasElse) {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
            if (!LowerStmtStrict(ifStmt->getElse())) {
                return false;
            }
        }

        return true;
    }

    // 语句降级的总调度中枢。
    bool LowerStmtStrict(const clang::Stmt* stmt) {
        if (!stmt) {
            return true;
        }

        if (const auto* compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
            for (const clang::Stmt* bodyStmt : compound->body()) {
                if (!LowerStmtStrict(bodyStmt)) {
                    return false;
                }
            }
            return true;
        }

        if (const auto* declStmt = llvm::dyn_cast<clang::DeclStmt>(stmt)) {
            return LowerDeclStmt(declStmt);
        }

        if (const auto* ifStmt = llvm::dyn_cast<clang::IfStmt>(stmt)) {
            return LowerIfStmt(ifStmt);
        }

        if (const auto* forStmt = llvm::dyn_cast<clang::ForStmt>(stmt)) {
            return LowerForStmt(forStmt);
        }

        if (const auto* returnStmt = llvm::dyn_cast<clang::ReturnStmt>(stmt)) {
            (void)returnStmt;
            return true;
        }

        if (const auto* expr = llvm::dyn_cast<clang::Expr>(stmt)) {
            if (const auto* bin = llvm::dyn_cast<clang::BinaryOperator>(expr->IgnoreParenImpCasts())) {
                if (bin->isAssignmentOp() || bin->isCompoundAssignmentOp()) {
                    return LowerAssignmentLike(bin);
                }
            }
            mlir::Value ignored;
            return LowerExprStrict(expr, ignored);
        }

        if (llvm::isa<clang::NullStmt>(stmt)) {
            return true;
        }

        return ReportUnsupported(stmt, "unsupported statement kind");
    }

    // 功能：生成或转换Lower Stmt相关结果。
    void LowerStmt(const clang::Stmt* stmt) {
        (void)LowerStmtStrict(stmt);
    }

    // 翻译整个计算函数。
    bool LowerFunction(const clang::FunctionDecl* funcDecl) {
        struct LoweredFunctionArg {
            std::string symbolName;
            mlir::Type type;
        };

        std::vector<LoweredFunctionArg> loweredArgs;
        for (unsigned i = 0; i < funcDecl->getNumParams(); ++i) {
            const clang::ParmVarDecl* paramDecl = funcDecl->getParamDecl(i);
            std::string name = paramDecl->getNameAsString();
            if (name.empty()) {
                name = "arg" + std::to_string(i);
            }

            if (const CalcParamBinding* binding = FindParamBinding(name)) {
                bool emittedBindingArgs = false;
                for (const auto& field : binding->fields) {
                    loweredArgs.push_back(
                        {field.physicalName,
                         LowerBoundRecordFieldParamType(paramDecl->getType(), field.fieldType)});
                    emittedBindingArgs = true;
                }
                if (emittedBindingArgs) {
                    continue;
                }
            }

            loweredArgs.push_back({name, LowerParamType(paramDecl->getType())});
        }

        std::vector<mlir::Type> argTypes;
        argTypes.reserve(loweredArgs.size());
        for (const auto& arg : loweredArgs) {
            argTypes.push_back(arg.type);
        }

        mlir::FunctionType funcTy = builder.getFunctionType(argTypes, {});
        std::string fnName = funcDecl->getNameAsString();
        mlir::func::FuncOp funcOp = mlir::func::FuncOp::create(UnknownLoc(), fnName, funcTy);

        mlir::Block* entry = funcOp.addEntryBlock();
        builder.setInsertionPointToStart(entry);

        symbols.clear();
        for (unsigned i = 0; i < loweredArgs.size(); ++i) {
            const std::string& name = loweredArgs[i].symbolName;
            mlir::Value argVal = entry->getArgument(i);
            if (argVal.getType().isa<mlir::MemRefType>()) {
                mlir::Type elemTy = argVal.getType().cast<mlir::MemRefType>().getElementType();
                symbols[name] = SymbolInfo{SymbolInfo::Kind::MemRef, argVal, elemTy};
            } else {
                // 标量参数统一降到单元素 memref，便于赋值语义。
                mlir::Type scalarTy = argVal.getType();
                mlir::MemRefType slotTy = mlir::MemRefType::get({1}, scalarTy);
                mlir::Value slot = builder.create<mlir::memref::AllocaOp>(UnknownLoc(), slotTy);
                builder.create<mlir::memref::StoreOp>(UnknownLoc(), argVal, slot, mlir::ValueRange{ConstIndex(0)});
                symbols[name] = SymbolInfo{SymbolInfo::Kind::ScalarAlloca, slot, scalarTy};
            }
        }

        const clang::Stmt* bodyStmt = calc->getCalcBodyStmt();
        if (!bodyStmt) {
            bodyStmt = funcDecl->getBody();
        }
        if (!bodyStmt) {
            resultDiagnostics = "calc body stmt is null";
            return false;
        }

        if (!LowerStmtStrict(bodyStmt)) {
            return false;
        }

        if (!entry->empty() && !entry->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
            builder.create<mlir::func::ReturnOp>(UnknownLoc());
        }

        module.push_back(funcOp);
        return true;
    }
};

} // namespace
#endif

// 实例借口，
CalcAstToMlirResult LowerCalcAstToMlir(const Calc* calc,
                                       const CalcAstToMlirOptions& options) {
#if defined(DACPP_TRANSLATOR_HAS_MLIR)
    CalcAstLowering lowering(calc, options);
    return lowering.Run();
#else
    (void)calc;
    (void)options;
    CalcAstToMlirResult result;
    result.mlirAvailable = false;
    result.lowered = false;
    result.verified = false;
    result.diagnostics = "MLIR is not enabled in this build";
    return result;
#endif
}

} // namespace ascendc_mlir
} // namespace dacppTranslator
