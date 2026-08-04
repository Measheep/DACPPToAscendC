#ifndef TRANSLATOR_REWRITER_ASCENDC_MLIR_CALC_SHELL_INFO_H
#define TRANSLATOR_REWRITER_ASCENDC_MLIR_CALC_SHELL_INFO_H

#include <string>
#include <vector>

namespace dacppTranslator {
class Shell;

namespace ascendc_mlir {

struct CalcShellWindowInfo {
    int dimIdx = -1;
    int splitSize = 1;
    int splitStride = 1;
    int leftHalo = 0;
    int rightHalo = 0;
    bool isScalar = false;
    bool isWindow = false;
    std::string splitKind;
};

struct CalcShellParamInfo {
    std::string name;
    int rw = 0;
    int rank = 0;
    std::vector<int> shape;
    bool isScalar = false;
    bool isWindow = false;
    bool isBroadcastScalar = false;
    std::vector<CalcShellWindowInfo> windows;
};

struct CalcShellSummary {
    bool valid = false;
    std::vector<CalcShellParamInfo> params;
};

CalcShellSummary BuildCalcShellSummary(dacppTranslator::Shell* shell);
std::string DumpCalcShellSummary(const CalcShellSummary& summary);

} // namespace ascendc_mlir
} // namespace dacppTranslator

#endif // TRANSLATOR_REWRITER_ASCENDC_MLIR_CALC_SHELL_INFO_H
