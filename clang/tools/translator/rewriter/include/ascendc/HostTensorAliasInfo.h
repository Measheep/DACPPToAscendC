#ifndef TRANSLATOR_REWRITER_ASCENDC_HOST_TENSOR_ALIAS_INFO_H
#define TRANSLATOR_REWRITER_ASCENDC_HOST_TENSOR_ALIAS_INFO_H

// HostTensorAliasInfo：host 侧 tensor alias 的 rank / shape / stride / offset 描述。
//
// 从 AscendCArtifacts.h 抽出到独立头，在 driver / HostFlowBuilder / HostFlowEmitter
// 之间共享。

#include <string>
#include <vector>

namespace dacppTranslator {
namespace ascendc {

struct HostTensorAliasInfo {
    bool valid = false;
    bool isView = false;
    std::string ownerFunctionName;
    std::string aliasName;
    std::string rootContainerName;
    std::string rootDataExpr;
    std::string elementType;
    std::string offsetExpr = "0";
    int rank = 0;
    int currentDim = 0;
    std::vector<std::string> shapeExprs;
    std::vector<std::string> strideExprs;

    int ownerExpr = -1;
    bool runtimeGenerated = false;
    std::string runtimeBufferName;
};

} // namespace ascendc
} // namespace dacppTranslator

#endif // TRANSLATOR_REWRITER_ASCENDC_HOST_TENSOR_ALIAS_INFO_H
