#include "Dacfor.h"

namespace dacppTranslator {

// Dacfor 实现

ParamControl::ParamControl()
    : paramA_(nullptr), paramB_(nullptr), operation_(SemanticOperation::None) {}

void ParamControl::setParamA(Param* param) {
    paramA_ = param;
}

void ParamControl::setParamB(Param* param) {
    paramB_ = param;
}

void ParamControl::setOperation(SemanticOperation op) {
    operation_ = op;
}

Param* ParamControl::getParamA(){
    return paramA_;
}

Param* ParamControl::getParamB(){
    return paramB_;
}


SemanticOperation ParamControl::getOperation() const {
    return operation_;
}

// ControlBlock 实现


void ControlBlock::addParamControl(const std::shared_ptr<ParamControl>& control) {
    paramControls_.push_back(control);
}

void ControlBlock::setLoopBound(std::string count) {
    loopBound_ = count;
}

const std::vector<std::shared_ptr<ParamControl>>& ControlBlock::getParamControls() const {
    return paramControls_;
}

std::string ControlBlock::getLoopBound() const {
    return loopBound_;
}

} // namespace dacppTranslator
