#include <string>

#include "clang/AST/Attr.h"
#include "llvm/ADT/StringExtras.h"

#include "DacppStructure.h"
#include "ASTParse.h"


using namespace clang;

namespace {

static bool TryExtractTensorShapeFromExpr(const Expr* expr,
                                          const ASTContext& astContext,
                                          std::vector<int>& outShape);
static bool TryExtractTensorShapeFromVarDecl(const VarDecl* varDecl,
                                             std::vector<int>& outShape);

static const Expr* StripShapeExprWrappers(const Expr* expr) {
    const Expr* cur = expr;
    while (cur) {
        if (const auto* castExpr = dyn_cast<ImplicitCastExpr>(cur)) {
            cur = castExpr->getSubExpr();
            continue;
        }
        if (const auto* bindExpr = dyn_cast<CXXBindTemporaryExpr>(cur)) {
            cur = bindExpr->getSubExpr();
            continue;
        }
        if (const auto* materializeExpr = dyn_cast<MaterializeTemporaryExpr>(cur)) {
            cur = materializeExpr->getSubExpr();
            continue;
        }
        if (const auto* cleanupsExpr = dyn_cast<ExprWithCleanups>(cur)) {
            cur = cleanupsExpr->getSubExpr();
            continue;
        }
        if (const auto* parenExpr = dyn_cast<ParenExpr>(cur)) {
            cur = parenExpr->getSubExpr();
            continue;
        }
        break;
    }
    return cur;
}

static bool TryEvaluateIntExpr(const Expr* expr,
                               const ASTContext& astContext,
                               int& outValue) {
    outValue = 0;
    const Expr* stripped = StripShapeExprWrappers(expr);
    if (!stripped) {
        return false;
    }

    Expr::EvalResult evalResult;
    if (stripped->EvaluateAsInt(evalResult, astContext)) {
        outValue = static_cast<int>(evalResult.Val.getInt().getExtValue());
        return true;
    }

    if (const auto* unary = dyn_cast<UnaryOperator>(stripped)) {
        int inner = 0;
        if (TryEvaluateIntExpr(unary->getSubExpr(), astContext, inner)) {
            outValue = (unary->getOpcode() == UO_Minus) ? -inner : inner;
            return true;
        }
    }

    return false;
}

static bool TryExtractTensorRank(clang::QualType type, int& outRank) {
    outRank = 0;
    while (!type.isNull()) {
        if (const auto* refType = type->getAs<ReferenceType>()) {
            type = refType->getPointeeType();
            continue;
        }
        if (const auto* elaborated = dyn_cast<ElaboratedType>(type.getTypePtr())) {
            type = elaborated->desugar();
            continue;
        }
        if (const auto* typedefType = dyn_cast<TypedefType>(type.getTypePtr())) {
            type = typedefType->desugar();
            continue;
        }
        break;
    }

    if (const auto* recordDecl = type->getAsCXXRecordDecl()) {
        if (const auto* classSpec = dyn_cast<ClassTemplateSpecializationDecl>(recordDecl)) {
            const ClassTemplateDecl* tmplDecl = classSpec->getSpecializedTemplate();
            const auto args = classSpec->getTemplateArgs().asArray();
            if (tmplDecl && tmplDecl->getNameAsString() == "Tensor" && args.size() >= 2) {
                const TemplateArgument& rankArg = args[1];
                if (rankArg.getKind() == TemplateArgument::Integral) {
                    outRank = static_cast<int>(rankArg.getAsIntegral().getExtValue());
                    if (outRank > 0) {
                        return true;
                    }
                }
            }
        }
    }

    const auto* specType = type->getAs<TemplateSpecializationType>();
    if (!specType) {
        type = type.getCanonicalType();
        specType = type->getAs<TemplateSpecializationType>();
    }
    if (!specType) {
        return false;
    }

    const TemplateDecl* tmplDecl = specType->getTemplateName().getAsTemplateDecl();
    const auto args = specType->template_arguments();
    if (!tmplDecl || tmplDecl->getNameAsString() != "Tensor" || args.size() < 2) {
        return false;
    }

    const TemplateArgument& rankArg = args[1];
    if (rankArg.getKind() != TemplateArgument::Integral) {
        return false;
    }

    outRank = static_cast<int>(rankArg.getAsIntegral().getExtValue());
    return outRank > 0;
}

static bool TryCollectShapeVectorFromExpr(const Expr* expr,
                                          const ASTContext& astContext,
                                          std::vector<int>& outShape) {
    const Expr* stripped = StripShapeExprWrappers(expr);
    if (!stripped) {
        return false;
    }

    if (const auto* stdInit = dyn_cast<CXXStdInitializerListExpr>(stripped)) {
        return TryCollectShapeVectorFromExpr(stdInit->getSubExpr(), astContext, outShape);
    }

    if (const auto* initList = dyn_cast<InitListExpr>(stripped)) {
        std::vector<int> candidate;
        for (const Stmt* child : initList->children()) {
            const Expr* childExpr = dyn_cast_or_null<Expr>(child);
            if (!childExpr) {
                continue;
            }
            int dim = 0;
            if (!TryEvaluateIntExpr(childExpr, astContext, dim)) {
                return false;
            }
            candidate.push_back(dim);
        }
        if (candidate.empty()) {
            return false;
        }
        outShape = std::move(candidate);
        return true;
    }

    if (const auto* parenList = dyn_cast<CXXParenListInitExpr>(stripped)) {
        for (const Stmt* child : parenList->children()) {
            const Expr* childExpr = dyn_cast_or_null<Expr>(child);
            if (!childExpr) {
                continue;
            }
            if (TryCollectShapeVectorFromExpr(childExpr, astContext, outShape) &&
                !outShape.empty()) {
                return true;
            }
        }
        return false;
    }

    if (const auto* construct = dyn_cast<CXXConstructExpr>(stripped)) {
        for (unsigned int i = 0; i < construct->getNumArgs(); ++i) {
            std::vector<int> candidate;
            if (TryCollectShapeVectorFromExpr(construct->getArg(i), astContext, candidate) &&
                !candidate.empty()) {
                outShape = std::move(candidate);
                return true;
            }
        }
        return false;
    }

    if (const auto* declRef = dyn_cast<DeclRefExpr>(stripped)) {
        const auto* refVar = dyn_cast_or_null<VarDecl>(declRef->getDecl());
        if (refVar && refVar->getInit()) {
            return TryCollectShapeVectorFromExpr(refVar->getInit(), astContext, outShape);
        }
    }

    return false;
}

static bool TryExtractStaticExtentFromExpr(const Expr* expr,
                                           const ASTContext& astContext,
                                           int& outExtent) {
    outExtent = 0;
    const Expr* stripped = StripShapeExprWrappers(expr);
    if (!stripped) {
        return false;
    }

    if (TryEvaluateIntExpr(stripped, astContext, outExtent) && outExtent >= 0) {
        return true;
    }

    if (const auto* stdInit = dyn_cast<CXXStdInitializerListExpr>(stripped)) {
        return TryExtractStaticExtentFromExpr(stdInit->getSubExpr(), astContext, outExtent);
    }

    if (const auto* initList = dyn_cast<InitListExpr>(stripped)) {
        outExtent = static_cast<int>(initList->getNumInits());
        return outExtent >= 0;
    }

    if (const auto* parenList = dyn_cast<CXXParenListInitExpr>(stripped)) {
        for (const Stmt* child : parenList->children()) {
            const Expr* childExpr = dyn_cast_or_null<Expr>(child);
            if (!childExpr) {
                continue;
            }
            if (TryExtractStaticExtentFromExpr(childExpr, astContext, outExtent)) {
                return true;
            }
        }
        return false;
    }

    if (const auto* construct = dyn_cast<CXXConstructExpr>(stripped)) {
        if (construct->getNumArgs() >= 1 &&
            TryEvaluateIntExpr(construct->getArg(0), astContext, outExtent) &&
            outExtent >= 0) {
            return true;
        }
        for (unsigned int i = 0; i < construct->getNumArgs(); ++i) {
            if (TryExtractStaticExtentFromExpr(construct->getArg(i), astContext, outExtent)) {
                return true;
            }
        }
        return false;
    }

    if (const auto* declRef = dyn_cast<DeclRefExpr>(stripped)) {
        const auto* refVar = dyn_cast_or_null<VarDecl>(declRef->getDecl());
        if (refVar && refVar->getInit()) {
            return TryExtractStaticExtentFromExpr(refVar->getInit(), astContext, outExtent);
        }
    }

    return false;
}

static bool TryCollectIndexVectorFromExpr(const Expr* expr,
                                          const ASTContext& astContext,
                                          std::vector<int>& outValues) {
    outValues.clear();
    const Expr* stripped = StripShapeExprWrappers(expr);
    if (!stripped) {
        return false;
    }

    if (const auto* stdInit = dyn_cast<CXXStdInitializerListExpr>(stripped)) {
        return TryCollectIndexVectorFromExpr(stdInit->getSubExpr(), astContext, outValues);
    }

    if (const auto* initList = dyn_cast<InitListExpr>(stripped)) {
        for (const Stmt* child : initList->children()) {
            const Expr* childExpr = dyn_cast_or_null<Expr>(child);
            if (!childExpr) {
                continue;
            }
            int value = 0;
            if (!TryEvaluateIntExpr(childExpr, astContext, value)) {
                return false;
            }
            outValues.push_back(value);
        }
        return true;
    }

    return TryCollectShapeVectorFromExpr(stripped, astContext, outValues);
}

static bool TryExtractTensorSliceShape(const Expr* expr,
                                       const ASTContext& astContext,
                                       std::vector<int>& outShape) {
    outShape.clear();
    const Expr* stripped = StripShapeExprWrappers(expr);
    const auto* opCall = dyn_cast_or_null<CXXOperatorCallExpr>(stripped);
    if (!opCall || opCall->getOperator() != OO_Subscript || opCall->getNumArgs() < 2) {
        return false;
    }

    std::vector<int> baseShape;
    if (!TryExtractTensorShapeFromExpr(opCall->getArg(0), astContext, baseShape) ||
        baseShape.empty()) {
        return false;
    }

    std::vector<int> indices;
    if (!TryCollectIndexVectorFromExpr(opCall->getArg(1), astContext, indices)) {
        return false;
    }

    outShape = baseShape;
    if (indices.empty()) {
        return true;
    }

    const int targetDim = 0;
    if (targetDim >= static_cast<int>(outShape.size())) {
        return false;
    }

    if (indices.size() == 1) {
        outShape[targetDim] = 1;
        return true;
    }

    const int start = indices[0];
    const int end = indices[1];
    const int stride = (indices.size() >= 3) ? indices[2] : 1;
    if (stride <= 0 || end < start) {
        return false;
    }

    const int extent = (end > start) ? ((end - start - 1) / stride + 1) : 0;
    outShape[targetDim] = extent;
    return true;
}

static bool TryExtractTensorShapeFromInit(const Expr* initExpr,
                                          const ASTContext& astContext,
                                          int tensorRank,
                                          std::vector<int>& outShape) {
    outShape.clear();
    const Expr* stripped = StripShapeExprWrappers(initExpr);
    if (!stripped) {
        return false;
    }

    if (const auto* declRef = dyn_cast<DeclRefExpr>(stripped)) {
        const auto* refVar = dyn_cast_or_null<VarDecl>(declRef->getDecl());
        return refVar && TryExtractTensorShapeFromVarDecl(refVar, outShape) && !outShape.empty();
    }

    if (TryExtractTensorSliceShape(stripped, astContext, outShape) && !outShape.empty()) {
        return true;
    }

    if (const auto* construct = dyn_cast<CXXConstructExpr>(stripped)) {
        if (tensorRank > 1) {
            if (construct->getNumArgs() >= 1 &&
                TryCollectShapeVectorFromExpr(construct->getArg(0), astContext, outShape) &&
                !outShape.empty()) {
                return true;
            }
        } else {
            int extent = 0;
            if (construct->getNumArgs() >= 1 &&
                TryExtractStaticExtentFromExpr(construct->getArg(0), astContext, extent) &&
                extent > 0) {
                outShape.push_back(extent);
                return true;
            }
        }

        for (unsigned int i = 0; i < construct->getNumArgs(); ++i) {
            std::vector<int> candidate;
            if (TryExtractTensorShapeFromInit(construct->getArg(i), astContext, tensorRank, candidate) &&
                !candidate.empty()) {
                outShape = std::move(candidate);
                return true;
            }
        }
        return false;
    }

    if (tensorRank > 1) {
        return TryCollectShapeVectorFromExpr(stripped, astContext, outShape) && !outShape.empty();
    }

    int extent = 0;
    if (TryExtractStaticExtentFromExpr(stripped, astContext, extent) && extent > 0) {
        outShape.push_back(extent);
        return true;
    }
    return false;
}

static bool TryExtractTensorShapeFromExpr(const Expr* expr,
                                          const ASTContext& astContext,
                                          std::vector<int>& outShape) {
    outShape.clear();
    if (!expr) {
        return false;
    }

    int tensorRank = 0;
    if (!TryExtractTensorRank(expr->getType(), tensorRank)) {
        return false;
    }

    return TryExtractTensorShapeFromInit(expr, astContext, tensorRank, outShape) &&
           !outShape.empty();
}

static bool TryExtractTensorShapeFromVarDecl(const VarDecl* varDecl,
                                             std::vector<int>& outShape) {
    outShape.clear();
    if (!varDecl || !varDecl->getInit()) {
        return false;
    }
    int tensorRank = 0;
    if (!TryExtractTensorRank(varDecl->getType(), tensorRank)) {
        return false;
    }
    return TryExtractTensorShapeFromInit(varDecl->getInit(),
                                         varDecl->getASTContext(),
                                         tensorRank,
                                         outShape) &&
           !outShape.empty();
}

} // namespace


/**
 * 存储数据关联计算表达式信息类实现
 */
dacppTranslator::Expression::Expression() {
}

void dacppTranslator::Expression::setShell(Shell* shell) {
    this->shell = shell;
}

dacppTranslator::Shell* dacppTranslator::Expression::getShell() {
    return shell;
}

void dacppTranslator::Expression::setCalc(Calc* calc) {
    this->calc = calc;
}

dacppTranslator::Calc* dacppTranslator::Expression::getCalc() {
    return calc;
}

void dacppTranslator::Expression::setDacExpr(const clang::BinaryOperator* dacExpr) {
    this->dacExpr = dacExpr;
}

const clang::BinaryOperator* dacppTranslator::Expression::getDacExpr() {
    return dacExpr;
}

bool dacppTranslator::Expression::shellLHS_p(const BinaryOperator *dacExpr) {
  bool found_p = false;
  Expr *LHS;
  const CXXBindTemporaryExpr *BTE;
  const CallExpr *Call;
  const ImplicitCastExpr *ICE;
  const DeclRefExpr *DRE;
  const FunctionDecl *FD;

  do {
    LHS = dacExpr->getLHS();
    BTE = dyn_cast<CXXBindTemporaryExpr>(LHS);
    if (!BTE)
      break;

    Call = dyn_cast<CallExpr>(BTE->getSubExpr());
    if (!Call)
      break;

    ICE = dyn_cast<ImplicitCastExpr>(Call->getCallee());
    if (!ICE)
      break;

    DRE = dyn_cast<DeclRefExpr>(ICE->getSubExpr());
    if (!DRE)
      break;

    FD = dyn_cast<FunctionDecl>(DRE->getDecl());
    if (!FD)
      break;

    if (!FD->getDescribedFunctionTemplate() &&
        !FD->isFunctionTemplateSpecialization()) {
      if (FD->hasAttrs()) {
        const AttrVec &Attrs = FD->getAttrs();
        for (auto *A : Attrs) {
          std::string BSBuf;
          llvm::raw_string_ostream BSStream(BSBuf);
          A->printPretty(BSStream, PrintingPolicy(LangOptions()));
          if (!strcmp(BSBuf.c_str(), "shell")) {
            found_p = true;
            break;
          }
        }
      }
    }
  } while (0);
  return found_p;
}

/**
 * 存储DACPP文件信息类实现
 */
dacppTranslator::DacppFile::DacppFile() {
}

void dacppTranslator::DacppFile::setExpression(const BinaryOperator* dacExpr) {
    // 获取 DAC 数据关联表达式左值
    Expr* dacExprLHS = dacppTranslator::Expression::shellLHS_p (dacExpr) ?  dacExpr->getLHS() : dacExpr->getRHS();
    CallExpr* shellCall = getNode<CallExpr>(dacExprLHS);
    if (!shellCall) {
        return;
    }

    const FunctionDecl* shellFuncDecl = shellCall->getDirectCallee();
    const unsigned int shellParamCount =
        shellFuncDecl ? shellFuncDecl->getNumParams() : shellCall->getNumArgs();

    // 获取实参的形状
    // 这里的实参目前指的是 DACPP:Tensor
    // TODO 
    // Tensor的初始化用到了两个std::vector，如果在生成之后对其进行了push_back，则不能得到正确的形状，会出现bug
    std::vector<std::vector<int>> shapes(shellParamCount);
    const unsigned int argCount = shellCall->getNumArgs();
    for(unsigned int paramsCount = 0; paramsCount < argCount && paramsCount < shellParamCount; paramsCount++) {
        Expr* curExpr = shellCall->getArg(paramsCount);
        DeclRefExpr* declRefExpr = nullptr;
        if(isa<DeclRefExpr>(curExpr)) {
            declRefExpr = dyn_cast<DeclRefExpr>(curExpr);
        } else if(isa<ImplicitCastExpr>(curExpr)) {
            declRefExpr = getNode<DeclRefExpr>(curExpr);
        } else {
            // 带切片的Tensor
            while(getNode<CXXOperatorCallExpr>(curExpr)) {
                curExpr = getNode<CXXOperatorCallExpr>(curExpr);
            }
            int count = 0;
            for(Stmt::child_iterator it = curExpr->child_begin(); it != curExpr->child_end(); it++, count++) {
                if(count != 1) {
                    continue;
                }
                declRefExpr = getNode<DeclRefExpr>(*it);
            }
        }

        if (!declRefExpr || !declRefExpr->getDecl()) {
            continue;
        }

        VarDecl* sourceVarDecl = dyn_cast<VarDecl>(declRefExpr->getDecl());
        if (!sourceVarDecl || !sourceVarDecl->getInit()) {
            continue;
        }

        TryExtractTensorShapeFromVarDecl(sourceVarDecl, shapes[paramsCount]);
    }
    Shell* shell = new Shell();
    Calc* calc = new Calc();
    Expression* expr = new Expression();
    expr->setDacExpr(dacExpr);
    expr->setShell(shell);
    expr->setCalc(calc);
    shell->setFather(expr);
    calc->setFather(expr);
    shell->parseShell(dacExpr, shapes);
    calc->parseCalc(dacExpr);
    exprs.push_back(expr);
}

dacppTranslator::Expression* dacppTranslator::DacppFile::getExpression(int idx) {
    return exprs[idx];
}

int dacppTranslator::DacppFile::getNumExpression() {
    return exprs.size();
}

void dacppTranslator::DacppFile::setMainFuncLoc(const FunctionDecl* mainFuncLoc) {
    this->mainFuncLoc = mainFuncLoc;
}

const FunctionDecl* dacppTranslator::DacppFile::getMainFuncLoc() {
    return mainFuncLoc;
}

void dacppTranslator::printParserResult(DacppFile* dacppFile) {
    if (!dacppFile) {
        std::cerr << "错误：DacppFile 对象为空！" << std::endl;
        return;
    }

    std::cout << "\n========== DACPP Parser 中间产物输出 ==========\n";
    int numExprs = dacppFile->getNumExpression();
    std::cout << "共找到 " << numExprs << " 个数据关联计算表达式 (Expression)。\n";

    for (int i = 0; i < numExprs; ++i) {
        Expression* expr = dacppFile->getExpression(i);
        std::cout << "\n--- Expression [" << i << "] ---\n";

        // 1. 打印 Shell 信息
        Shell* shell = expr->getShell();
        if (shell) {
            std::cout << "[Shell 信息]\n";
            std::cout << "  函数名: " << shell->getName() << "\n";
            int numParams = shell->getNumParams();
            std::cout << "  参数数量: " << numParams << "\n";
            
            for (int j = 0; j < numParams; ++j) {
                Param* param = shell->getShellParam(j);
                std::cout << "    - 参数 " << j << ": " << param->getName() 
                          << " | 类型: " << param->getType() 
                          << " | 读写属性: " << (param->getRw() ? "输出(Write)" : "输入(Read)") << "\n";
                
                // 打印参数的维度/形状
                std::cout << "      形状(Shape): [ ";
                for(int d = 0; d < param->getDim(); d++) {
                    std::cout << param->getShape(d) << " ";
                }
                std::cout << "]\n";
            }
        }

        // 2. 打印 Calc 信息
        Calc* calc = expr->getCalc();
        if (calc) {
            std::cout << "\n[Calc 信息]\n";
            std::cout << "  函数名: " << calc->getName() << "\n";
            int numParams = calc->getNumParams();
            std::cout << "  参数数量: " << numParams << "\n";
            
            for (int j = 0; j < numParams; ++j) {
                Param* param = calc->getParam(j);
                std::cout << "    - 参数 " << j << ": " << param->getName() 
                          << " | 基本类型: " << param->getBasicType() << "\n";
            }
            
            // 打印解析提取出的计算代码块
            if (calc->getNumBody() > 0) {
                std::cout << "  代码体 (Body Blocks):\n";
                for (int b = 0; b < calc->getNumBody(); ++b) {
                     std::cout << "    ---- Block " << b << " ----\n";
                     std::cout << calc->getBody(b) << "\n";
                }
            }
        }
    }
    std::cout << "===============================================\n\n";
}
