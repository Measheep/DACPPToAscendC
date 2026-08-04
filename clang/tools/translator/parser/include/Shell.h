#ifndef TRANSLATOR_PARSER_SHELL_H
#define TRANSLATOR_PARSER_SHELL_H


#include <string>
#include <vector>

#include "clang/AST/AST.h"

#include "Param.h"
#include "Split.h"

using namespace clang;


namespace dacppTranslator {


class Expression;

typedef struct _tagBINDINFO
    {
    int icls;
    VNode *v;
    std::string offset;
    } 	BINDINFO;

/**
 * 存储划分结构信息
 */
class Shell {
// rewriter_->ReplaceText(shellFunc->getSourceRange(), "");
private:
    std::string name; // 函数名
    std::vector<Param*> params; // 参数列表
    std::vector<Split*> splits; // 划分列表
    std::vector<ShellParam*> shellParams; // 划分结构参数
    Expression* father; // 所属的数据关联计算表达式
    FunctionDecl* shellLoc; // AST中Shell节点的位置

public:
    Shell();
    ~Shell();

    void setName(std::string name);
    std::string getName();

    void setParam(Param* param);
    Param* getParam(int idx);
    int getNumParams();

    void setSplit(Split* split);
    Split* getSplit(int idx);
    int getNumSplits();

    void setShellParam(ShellParam* param);
    ShellParam* getShellParam(int idx);
    int getNumShellParams();

    void setFather(Expression* expr);
    Expression* getFather();

    void setShellLoc(FunctionDecl* expr);
    FunctionDecl* getShellLoc();

    void parseShell(const BinaryOperator* dacExpr, std::vector<std::vector<int>> shapes);
    void GetBindInfo(std::vector<BINDINFO> *pbindInfo) ;
    dacppTranslator::Split *search_symbol(VNode *v);

    ALGraph *G;
};


} // namespace dacppTranslator

#endif