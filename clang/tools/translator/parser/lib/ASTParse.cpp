#include <string>
#include <vector>

#include "ASTParse.h"

using namespace clang;

// 获得划分信息，将划分相关的变量名和节点保存到name和splits中
void dacppTranslator::getSplitExpr(Expr* curExpr, std::string& name, std::vector<Expr*>& splits) {
    if(!curExpr) { return; }
    CXXOperatorCallExpr* splitExpr = nullptr;
    if(!isa<CXXOperatorCallExpr>(curExpr)) {
        splitExpr = dacppTranslator::getNode<CXXOperatorCallExpr>(curExpr); 
    }
    else {
        splitExpr = dyn_cast<CXXOperatorCallExpr>(curExpr);
    }
    getSplitExpr(getNode<CXXOperatorCallExpr>(splitExpr), name, splits);
    int count = 0;
    for(Stmt::child_iterator it = splitExpr->child_begin(); it != splitExpr->child_end(); it++) {
        if(count == 1 && !getNode<CXXOperatorCallExpr>(splitExpr)) {
            DeclRefExpr* tmp = isa<DeclRefExpr>(*it) ? dyn_cast<DeclRefExpr>(*it) : getNodeBFS<DeclRefExpr>(*it);
            name =  tmp->getNameInfo().getAsString();
        } else if(count == 2) {
            splits.push_back(dyn_cast<Expr>(*it));
        }
        count++;
    }
}


// 判断数据的输入输出属性
bool dacppTranslator::inputOrOutput(std::string dataType) {
    std::string pre = "const";
    std::string::size_type idx;
    idx = dataType.find(pre);
    if(idx == std::string::npos) {
        return true;
    }
    else {
        return false;
    }
}
