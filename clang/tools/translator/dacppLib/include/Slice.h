#ifndef SLICE_H
#define SLICE_H

#include <string>
#include <vector>
#include <set>
#include <any>

namespace dacpp {


// 降维算子
class index {
public:
    const index operator+(const int &R)const;
    const index operator-(const int &R)const;
};


// 分区算子
class split {
private:
    int size_;
    int stride_;

public:
    split(int size, int stride) : size_(size), stride_(stride) {}
    const split operator+(const int &R)const;
    const split operator-(const int &R)const;
}; 


// 保形算子
class conformal {};


struct Slice {
    bool isindex_;
    bool isAll_;
    int start_;
    int end_;
    int stride_;

    Slice() : isindex_(false), isAll_(true) {}
    Slice(int idx) : isindex_(false), isAll_(false), start_(idx), end_(idx + 1), stride_(1) {}
    Slice(int start, int end, int stride = 1) : isindex_(false), isAll_(false), start_(start), end_(end), stride_(stride) {}
    Slice(index i) : isindex_(true) {}
};


void binding (index, index);
void binding (index, split);
void binding (split, index);
void binding (split, split);


} // namespace dacpp

#endif