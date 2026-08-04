#ifndef TRANSLATOR_PARSER_SPLIT_H
#define TRANSLATOR_PARSER_SPLIT_H

#include <string>

typedef struct ALGraph ALGraph;
typedef struct VNode VNode;

namespace dacppTranslator {

/*
    划分父类
*/
class Split {

private:
    std::string id; // 划分标识
    int dimIdx; // 划分作用维度
    int splitNumber;

public:
    std::string type;
    Split(Split *parent);
    virtual ~Split() {}
    Split(Split *parent, std::string id, int dimIdx);

    void setId(std::string id);
    std::string getId();

    void setDimIdx(int dimIdx);
    int getDimIdx();

    void setSplitNumber(int splitNumber);
    int getSplitNumber();

    VNode *v;
    Split *parent;

};

/*
    降维划分
*/
class IndexSplit : public Split {

private:
    int splitNumber; // 划分总份数

public:
    IndexSplit(Split *parent);
    IndexSplit(Split *parent, std::string id, int dimIdx, int splitNumber);

    void setSplitNumber(int splitNumber);
    int getSplitNumber();

    std::string toString(){
        return "id: " + getId() + "\n" +  // 使用父类的 getter
               "dimIdx: " + std::to_string(getDimIdx()) + "\n" +
               "splitNumber: " + std::to_string(splitNumber) + "\n";
    }
};
/*
    规则分区划分
*/
class RegularSplit : public Split {

private:
    int splitSize; // 划分大小
    int splitStride; // 划分步长
    int splitNumber; // 划分总份数

public:
    RegularSplit(Split *parent);
    RegularSplit(Split *parent, std::string id, int dimIdx, int splitSize, int splitStride, int splitNumber);

    void setSplitSize(int splitSize);
    int getSplitSize();

    void setSplitStride(int splitStride);
    int getSplitStride();

    void setSplitNumber(int splitNumber);
    int getSplitNumber();

    std::string toString(){
        return "id: " + getId() + "\n" +  // 使用父类的 getter
               "dimIdx: " + std::to_string(getDimIdx()) + "\n" +
               "splitSize: " + std::to_string(splitSize) + "\n" +
               "splitStride: " + std::to_string(splitStride) + "\n" +
               "splitNumber: " + std::to_string(splitNumber) + "\n";
    }
};

}

#endif