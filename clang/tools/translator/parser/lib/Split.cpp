#include <string>

#include "Split.h"

/*
    划分父类
*/
dacppTranslator::Split::Split(Split *parent) {
    this->parent = parent;
}

dacppTranslator::Split::Split(Split *parent, std::string id, int dimIdx) {
    this->id = id;
    this->dimIdx = dimIdx;
    this->parent = parent;
}

void dacppTranslator::Split::setId(std::string id) {
    this->id = id;
}

std::string dacppTranslator::Split::getId() {
    return id;
}

void dacppTranslator::Split::setDimIdx(int dimIdx) {
    this->dimIdx = dimIdx;
}

int dacppTranslator::Split::getDimIdx() {
    return dimIdx;
}

void dacppTranslator::Split::setSplitNumber(int splitNumber) {
    this->splitNumber = splitNumber;
}

int dacppTranslator::Split::getSplitNumber() {
    return splitNumber;
}

/*
    降维划分
*/
dacppTranslator::IndexSplit::IndexSplit(Split *parent): Split(parent) {
}

dacppTranslator::IndexSplit::IndexSplit(Split *parent, std::string id, int dimIdx, int splitNumber) : Split(parent, id, dimIdx) {
    this->splitNumber = splitNumber;
}

void dacppTranslator::IndexSplit::setSplitNumber(int splitNumber) {
    this->splitNumber = splitNumber;
}

int dacppTranslator::IndexSplit::getSplitNumber() {
    return splitNumber;
}

/*
    规则分区划分
*/
dacppTranslator::RegularSplit::RegularSplit(Split *parent): Split(parent) {
}

dacppTranslator::RegularSplit::RegularSplit(Split *parent, std::string id, int dimIdx, int splitSize, int splitStride, int splitNumber) : Split(parent, id, dimIdx) {
    this->splitSize = splitSize;
    this->splitStride = splitStride;
    this->splitNumber = splitNumber;
}

void dacppTranslator::RegularSplit::setSplitSize(int splitSize) {
    this->splitSize = splitSize;
}

int dacppTranslator::RegularSplit::getSplitSize() {
    return splitSize;
}

void dacppTranslator::RegularSplit::setSplitStride(int splitStride) {
    this->splitStride = splitStride;
}

int dacppTranslator::RegularSplit::getSplitStride() {
    return splitStride;
}

void dacppTranslator::RegularSplit::setSplitNumber(int splitNumber) {
    this->splitNumber = splitNumber;
}

int dacppTranslator::RegularSplit::getSplitNumber() {
    return splitNumber;
}