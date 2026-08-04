#ifndef TRANSLATOR_PARSER_DACPPSTRUCTURE_H
#define TRANSLATOR_PARSER_DACPPSTRUCTURE_H


#include <string>
#include <vector>
#include <iostream>

#include "clang/AST/AST.h"

#include "Split.h"
#include "Param.h"
#include "Shell.h"
#include "Calc.h"
#include "Dacfor.h"


using namespace clang;


namespace dacppTranslator {


/**
 * 存储数据关联计算表达式信息
 */
class Expression {
private:
    Shell* shell; // 数据关联表达式对应的shell
    Calc* calc; // 数据关联计算表达式对应的calc
    const clang::BinaryOperator* dacExpr; // AST中数据关联计算表达式节点的位置

public:
    Expression();

    void setShell(Shell* shell);
    Shell* getShell();

    void setCalc(Calc* calc);
    Calc* getCalc();

    void setDacExpr(const clang::BinaryOperator* dacExpr);
    const clang::BinaryOperator* getDacExpr();
    const clang::BinaryOperator* getDacExpr() const { return dacExpr; }

    static bool shellLHS_p(const BinaryOperator *dacExpr);
};


/**
 * 存储DACPP文件信息
 */
class DacppFile {
public:
    std::vector<const clang::BinaryOperator*> dacExprs;
private:
    std::vector<Expression*> exprs; // 数据关联计算表达式
    const FunctionDecl* mainFuncLoc; // AST中主函数节点位置
    clang::TranslationUnitDecl* decl;
    ControlBlock* block;
    const CallExpr* FS;
    // SourceRange forRange;



public:
    const FunctionDecl* node;
    DacppFile();

    void setTranslationUnitDecl(clang::TranslationUnitDecl* decl) {
        this->decl = decl;
    }

    clang::TranslationUnitDecl* getTranslationUnitDecl() {
        return decl;
    }
    
    void setExpression(const BinaryOperator* dacExpr);
    Expression* getExpression(int idx);
    int getNumExpression();

    void setMainFuncLoc(const FunctionDecl* mainFuncLoc);
    const FunctionDecl* getMainFuncLoc();

    void setBlock(ControlBlock* block) {
        this->block = block;
    }
    ControlBlock* getBlock() {
        return block;
    }
    void setForStmt(const CallExpr* FS) {
        this->FS = FS;
    }
    const CallExpr* getForStmt() {
        return FS;
    }
    // void setForRange(SourceRange range) { forRange = range; }
    // SourceRange getForRange() const { return forRange; }

};

void printParserResult(DacppFile* dacppFile);

}

#endif
