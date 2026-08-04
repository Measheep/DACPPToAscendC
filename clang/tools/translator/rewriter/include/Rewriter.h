#ifndef TRANSLATOR_REWRITER_REWRITER_H
#define TRANSLATOR_REWRITER_REWRITER_H

#include "clang/AST/AST.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "DacppStructure.h"
#include "ASTParse.h"
#include <string>

namespace dacppTranslator {

class Rewriter {
private:
    clang::Rewriter* rewriter;
    DacppFile* dacppFile;
    bool debug = false;

public:
    void setRewriter(clang::Rewriter* rewriter) {
        this->rewriter = rewriter;
    }

    void setDacppFile(DacppFile* dacppFile) {
        this->dacppFile = dacppFile;
    }

    void setDebug(bool enabled) {
        debug = enabled;
    }

    void rewriteDac_AscendC();

    void rewriteMain() {
        for (std::size_t exprCount = 0; exprCount < dacppFile->dacExprs.size();
             ++exprCount) {
            if (debug) {
                llvm::outs() << "dacExprCount: " << exprCount << "\n";
            }
            const BinaryOperator* dacExpr = dacppFile->dacExprs[exprCount];
            const CallExpr* FS;
            if(dacppFile->getBlock()){
                FS = dacppFile->getForStmt();
            }
            CallExpr* shellCall = dacppTranslator::getNode<CallExpr>(dacExpr->getLHS());
            DeclRefExpr* declRefExpr;
            if(isa<DeclRefExpr>(dacExpr->getRHS())) {
                declRefExpr = dyn_cast<DeclRefExpr>(dacExpr->getRHS());
            }
            else {
                declRefExpr = dacppTranslator::getNode<DeclRefExpr>(dacExpr->getRHS());
            }
            std::string str;
            llvm::raw_string_ostream rso(str);
            clang::LangOptions langOpts;
            langOpts.CPlusPlus = true; // 启用C++选项（根据需要配置）
            clang::PrintingPolicy policy(langOpts);
            shellCall->printPretty(rso, nullptr, policy);
            std::string code = rso.str();
            code.replace(code.find(shellCall->getDirectCallee()->getNameAsString()), 
            shellCall->getDirectCallee()->getNameAsString().size(), shellCall->getDirectCallee()->getNameAsString()
             + "_" + declRefExpr->getDecl()->getNameAsString());
            if(dacppFile->getBlock()){
                rewriter->ReplaceText(FS->getSourceRange(), code);
            }else{
                rewriter->ReplaceText(dacExpr->getSourceRange(), code);
            }
        }
    }
};

// namespace dacppTranslator
}

# endif
