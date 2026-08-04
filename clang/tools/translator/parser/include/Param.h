#ifndef TRANSLATOR_PARSER_PARAM_H
#define TRANSLATOR_PARSER_PARAM_H

#include <string>
#include <vector>
#include "clang/AST/Type.h"

#include "Split.h"

namespace dacppTranslator {

/*
    参数
*/
class Param {

private:
    bool rw; // 读写属性
    std::string name; // 参数名
    std::vector<int> shape; // 参数形状

public:
    Param();

    void setRw(bool rw);
    bool getRw();

    void setType(clang::QualType newType);
    std::string getType();
    std::string getBasicType();

    void setName(std::string name);
    std::string getName();

    void __attribute__(( unused, deprecated )) setShape(int size) ;
    void __attribute__(( unused, deprecated )) setShape(int idx, int size);
    int __attribute__(( unused, deprecated )) getShape(int idx);
    int __attribute__(( unused, deprecated )) getDim();

    /* 维度：通过tensor的第二个模版参数获取。  */
    int dim;

    /* 参数类型、参数基本类型。  */
    clang::QualType newType, BasicType;
};


/*
    划分结构参数
*/
class ShellParam : public Param {

private:
    std::vector<Split*> splits; // 划分组

public:
    ShellParam();
    
    void setSplit(Split* split);
    Split* getSplit(int idx);
    int getNumSplit();

};

}

#endif
