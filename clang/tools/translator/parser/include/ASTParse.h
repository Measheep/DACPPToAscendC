#ifndef ASTPARSE_H
#define ASTPARSE_H

#include <vector>
#include <queue>

#include "clang/AST/AST.h"

#include "Param.h"

using namespace clang;

namespace dacppTranslator {

// 从结点中找到特定结点，深度优先搜索
template <typename NodeType>
NodeType* getNode(Stmt* curStmt) {
    if(!curStmt) return nullptr;
    for(Stmt::child_iterator it = curStmt->child_begin(); it != curStmt->child_end(); it++) {
        if(isa<NodeType>(*it)) return dyn_cast<NodeType>(*it);
        if(getNode<NodeType>(*it)) {
            return getNode<NodeType>(*it);
        }
    }
    return nullptr;
}

// 从结点中找到特定结点，广度优先搜索
template <typename NodeType>
NodeType* getNodeBFS(Stmt* curStmt) {
    std::queue<Stmt*> q;
    q.push(curStmt);
    while(!q.empty()) {
        Stmt* cur = q.front();
        q.pop();
        for(Stmt::child_iterator it = cur->child_begin(); it != cur->child_end(); it++) {
            if(isa<NodeType>(*it)) {
                return dyn_cast<NodeType>(*it);
            }
            q.push(*it);
        }
    }
    return nullptr;
}



void getSplitExpr(Expr* curExpr, std::string& name, std::vector<Expr*>& splits);

// 判断数据的输入输出属性
bool inputOrOutput(std::string dataType);

}

#endif
