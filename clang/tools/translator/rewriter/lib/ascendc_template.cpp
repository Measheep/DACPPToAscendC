#include "ascendc_template.h"
#include "ascendc/HostRuntimeEmitter.h"
#include "universal_template.h"

#include <algorithm>
#include <regex>
#include <sstream>

namespace ASCENDC_TEMPLATE {

static bool ExprMentionsSymbol(const std::string& expr, const std::string& symbol) {
    return !expr.empty() && expr.find(symbol) != std::string::npos;
}

static int FindMaxCalcBufferIndexInExpr(const std::string& primaryExpr,
                                        const std::string& secondaryExpr) {
    int maxIndex = -1;
    std::regex bufferRegex(R"(calcBuffer([0-9]+)Local)");
    auto scan = [&](const std::string& text) {
        for (std::sregex_iterator it(text.begin(), text.end(), bufferRegex), end;
             it != end; ++it) {
            int idx = std::stoi((*it)[1].str());
            maxIndex = std::max(maxIndex, idx);
        }
    };
    scan(primaryExpr);
    scan(secondaryExpr);
    return maxIndex;
}

// Kernel file template
const char *KERNEL_FILE_Template = R"~~~(
#include "kernel_operator.h"
using namespace AscendC;

// Vector tiling/resource plan (translation-time selection)
{{TILING_DIAGNOSTICS}}
constexpr int32_t BUFFER_NUM = {{BUFFER_NUM}};
constexpr uint32_t TILE_ALIGN_ELEMENTS = {{TILE_ALIGN_ELEMENTS}};
constexpr uint32_t MAX_TILE_ELEMENTS = {{MAX_TILE_ELEMENTS}};
constexpr uint32_t DACPP_VECTOR_ALIGN_BYTES = {{VECTOR_ALIGN_BYTES}};

//  2. 算子类定义 
class Kernel_{{KERNEL_NAME}} {
public:
    __aicore__ inline Kernel_{{KERNEL_NAME}}() {}
    __aicore__ inline void Init({{GM_ADDR_PARAMS}}) {
        // 运行时动态 Tiling 参数计算
        this->totalLength = totalLength;
        this->useCoreNum = coreNum == 0 ? 1 : coreNum;
        this->tileLength = tileLength;
        this->tileCount = tileCount;
        if (this->totalLength == 0 || this->tileLength == 0 ||
            this->tileLength > MAX_TILE_ELEMENTS ||
            this->tileLength % TILE_ALIGN_ELEMENTS != 0 || this->tileCount == 0) {
            this->active = false;
            return;
        }
        this->blockLength = (totalLength + this->useCoreNum - 1) / this->useCoreNum;
        this->blockStart = this->blockLength * AscendC::GetBlockIdx();
        if (this->blockStart >= this->totalLength) {
            this->active = false;
            return;
        }
        this->blockEnd = this->blockStart + this->blockLength;
        if (this->blockEnd > this->totalLength) {
            this->blockEnd = this->totalLength;
        }
        this->blockLength = this->blockEnd - this->blockStart;
        const uint32_t expectedTileCount =
            (this->blockLength - 1) / this->tileLength + 1;
        if (this->tileCount != expectedTileCount) {
            this->active = false;
            return;
        }
{{INIT_CODE}}
    }
    __aicore__ inline void Process() {
        if (!active) {
            return;
        }
        for (uint32_t progress = 0; progress < tileCount; ++progress) {
            const uint32_t tileOffset = progress * tileLength;
            if (tileOffset >= blockLength) {
                break;
            }
            const uint32_t currentTileLength =
                tileLength < blockLength - tileOffset ? tileLength : blockLength - tileOffset;
            CopyIn(progress, tileOffset, currentTileLength);
            Compute(progress, tileOffset, currentTileLength);
            CopyOut(progress, tileOffset, currentTileLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t tileOffset,
                                  uint32_t currentTileLength) {
{{COPY_IN_CODE}}
    }
    __aicore__ inline void Compute(uint32_t progress, uint32_t tileOffset,
                                   uint32_t currentTileLength) {
{{COMPUTE_CODE}}
    }
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t tileOffset,
                                   uint32_t currentTileLength) {
{{COPY_OUT_CODE}}
    }

private:
    TPipe pipe;
{{TQUE_DECL}}

    //  动态 Tiling 内部变量 
    uint32_t totalLength;
    uint32_t useCoreNum;
    uint32_t blockLength;
    uint32_t blockStart;
    uint32_t blockEnd;
    uint32_t tileLength;
    uint32_t tileCount;
    bool active = true;
};

//  3. 对外暴露的 __aicore__ 函数 
extern "C" __global__ __aicore__ void {{KERNEL_NAME}}({{GM_ADDR_PARAMS}}) {
    Kernel_{{KERNEL_NAME}} op;
    op.Init({{PARAMS_NAMES}});
    op.Process();
}

//  4. Host 端启动 Wrapper 函数 
#ifndef ASCENDC_CPU_DEBUG
// #include "acl/acl.h"
// 宿主机包装器接收统一的运行时 tiling 参数。
extern "C" uint32_t {{KERNEL_NAME}}_do(uint32_t blockDim, void *stream, {{HOST_WRAPPER_PARAMS}}) {
    return {{KERNEL_NAME}}<<<blockDim, nullptr, stream>>>({{PARAMS_NAMES}});
}
#endif
)~~~";

std::string CodeGen_KernelFile(
    std::string kernelName, std::string gmAddrParams, std::string paramsNames,
    std::string hostWrapperParams, std::string initCode,
    std::string copyInCode, std::string computeCode, std::string copyOutCode,
    std::string tQueDecl,
    int vectorComputeAlignBytes,
    uint32_t bufferNum,
    uint32_t tileAlignElements,
    uint32_t maxTileElements,
    std::string tilingDiagnostics) {
    if (vectorComputeAlignBytes <= 0) {
        vectorComputeAlignBytes = 256;
    }

    std::string commentDiagnostics;
    std::istringstream diagnosticLines(tilingDiagnostics);
    for (std::string line; std::getline(diagnosticLines, line);) {
        commentDiagnostics += "// " + line + "\n";
    }
    return UNIVERSAL_TEMPLATE::templateString(KERNEL_FILE_Template, {
        {"{{KERNEL_NAME}}", kernelName},
        {"{{VECTOR_ALIGN_BYTES}}", std::to_string(vectorComputeAlignBytes)},
        {"{{BUFFER_NUM}}", std::to_string(bufferNum)},
        {"{{TILE_ALIGN_ELEMENTS}}", std::to_string(tileAlignElements)},
        {"{{MAX_TILE_ELEMENTS}}", std::to_string(maxTileElements)},
        {"{{TILING_DIAGNOSTICS}}", commentDiagnostics},
        {"{{GM_ADDR_PARAMS}}", gmAddrParams},
        {"{{PARAMS_NAMES}}", paramsNames},
        {"{{HOST_WRAPPER_PARAMS}}", hostWrapperParams},
        {"{{INIT_CODE}}", initCode},
        {"{{COPY_IN_CODE}}", copyInCode},
        {"{{COMPUTE_CODE}}", computeCode},
        {"{{COPY_OUT_CODE}}", copyOutCode},
        {"{{TQUE_DECL}}", tQueDecl}
    });
}

StructuredShiftedMapUpdateFastPathCode CodeGen_StructuredShiftedMapUpdateFastPath(
    std::string srcKernelType,
    std::string dstKernelType,
    std::string srcName,
    std::string dstName,
    int shiftOffset,
    int scratchBufferCount,
    std::string mapCenterCode,
    std::string mapShiftCode,
    std::string updateCode,
    std::string postCode,
    int vectorComputeAlignBytes) {
    StructuredShiftedMapUpdateFastPathCode code;
    if (vectorComputeAlignBytes <= 0) {
        vectorComputeAlignBytes = 256;
    }

    const std::string srcGmName = srcName + "Gm";
    const std::string dstGmName = dstName + "Gm";
    const std::string srcQueueName = "inQueue" + srcName;
    const std::string srcShiftQueueName = srcQueueName + "Shift";
    const std::string dstQueueName = "outQueue" + dstName;
    const std::string srcInputLengthName = srcName + "InputLength";
    const std::string shiftLiteral = std::to_string(shiftOffset);
    const std::string absShiftLiteral = std::to_string(shiftOffset >= 0 ? shiftOffset : -shiftOffset);

    const std::string srcAlignElemsExpr =
        "static_cast<uint32_t>((" + std::to_string(vectorComputeAlignBytes) +
        " + sizeof(" + srcKernelType + ") - 1) / sizeof(" + srcKernelType + "))";
    code.initCode += "        " + srcInputLengthName + " = (totalLength + static_cast<uint32_t>(" + absShiftLiteral + ") + " + srcAlignElemsExpr + " - 1) / " + srcAlignElemsExpr + " * " + srcAlignElemsExpr + ";\n";
    code.initCode += "        uint32_t " + srcName + "Remaining = " + srcInputLengthName + " - blockStart;\n";
    code.initCode += "        " + srcGmName + ".SetGlobalBuffer((__gm__ " + srcKernelType + " *)" + srcName + " + blockStart, " + srcName + "Remaining);\n";
    code.initCode += "        pipe.InitBuffer(" + srcQueueName + ", BUFFER_NUM, tileLength * sizeof(" + srcKernelType + "));\n";
    code.initCode += "        pipe.InitBuffer(" + srcShiftQueueName + ", BUFFER_NUM, tileLength * sizeof(" + srcKernelType + "));\n";
    code.initCode += "        " + dstGmName + ".SetGlobalBuffer((__gm__ " + dstKernelType + " *)" + dstName + " + blockStart, blockLength);\n";
    code.initCode += "        pipe.InitBuffer(" + dstQueueName + ", BUFFER_NUM, tileLength * sizeof(" + dstKernelType + "));\n";

    code.initCode += "        pipe.InitBuffer(calcBuffer0, tileLength * sizeof(" + dstKernelType + "));\n";
    code.initCode += "        pipe.InitBuffer(calcBuffer1, tileLength * sizeof(" + dstKernelType + "));\n";
    for (int i = 0; i < scratchBufferCount; ++i) {
        code.initCode += "        pipe.InitBuffer(calcBuffer" + std::to_string(i + 2) + ", tileLength * sizeof(" + dstKernelType + "));\n";
    }

    code.copyInCode += "        AscendC::LocalTensor<" + srcKernelType + "> srcLocal = " + srcQueueName + ".AllocTensor<" + srcKernelType + ">();\n";
    code.copyInCode += "        AscendC::LocalTensor<" + srcKernelType + "> srcShiftLocal = " + srcShiftQueueName + ".AllocTensor<" + srcKernelType + ">();\n";
    code.copyInCode += "        uint32_t " + srcName + "BaseOffset = tileOffset;\n";
    if (shiftOffset >= 0) {
        code.copyInCode += "        AscendC::DataCopy(srcLocal, " + srcGmName + "[" + srcName + "BaseOffset], currentTileLength);\n";
        code.copyInCode += "        AscendC::DataCopy(srcShiftLocal, " + srcGmName + "[" + srcName + "BaseOffset + " + shiftLiteral + "], currentTileLength);\n";
    } else {
        code.copyInCode += "        AscendC::DataCopy(srcShiftLocal, " + srcGmName + "[" + srcName + "BaseOffset], currentTileLength);\n";
        code.copyInCode += "        AscendC::DataCopy(srcLocal, " + srcGmName + "[" + srcName + "BaseOffset + " + absShiftLiteral + "], currentTileLength);\n";
    }
    code.copyInCode += "        " + srcQueueName + ".EnQue(srcLocal);\n";
    code.copyInCode += "        " + srcShiftQueueName + ".EnQue(srcShiftLocal);\n";

    code.computeCode += "        AscendC::LocalTensor<" + srcKernelType + "> srcLocal = " + srcQueueName + ".DeQue<" + srcKernelType + ">();\n";
    code.computeCode += "        AscendC::LocalTensor<" + srcKernelType + "> srcShiftLocal = " + srcShiftQueueName + ".DeQue<" + srcKernelType + ">();\n";
    code.computeCode += "        AscendC::LocalTensor<" + dstKernelType + "> dstLocal = " + dstQueueName + ".AllocTensor<" + dstKernelType + ">();\n";
    code.computeCode += "        AscendC::LocalTensor<" + dstKernelType + "> mapCenterLocal = calcBuffer0.Get<" + dstKernelType + ">();\n";
    code.computeCode += "        AscendC::LocalTensor<" + dstKernelType + "> mapShiftLocal = calcBuffer1.Get<" + dstKernelType + ">();\n";
    for (int i = 0; i < scratchBufferCount; ++i) {
        const std::string localName = "calcBuffer" + std::to_string(i + 2) + "Local";
        code.computeCode += "        AscendC::LocalTensor<" + dstKernelType + "> " + localName + " = calcBuffer" + std::to_string(i + 2) + ".Get<" + dstKernelType + ">();\n";
        code.computeCode += "        (void)" + localName + ";\n";
    }

    if (mapCenterCode.empty()) {
        code.computeCode += "        AscendC::Muls(mapCenterLocal, srcLocal, 1.0f, currentTileLength);\n";
    } else {
        code.computeCode += mapCenterCode;
    }

    if (mapShiftCode.empty()) {
        code.computeCode += "        AscendC::Muls(mapShiftLocal, srcShiftLocal, 1.0f, currentTileLength);\n";
    } else {
        code.computeCode += mapShiftCode;
    }

    if (updateCode.empty()) {
        code.computeCode += "        AscendC::Sub(dstLocal, mapShiftLocal, mapCenterLocal, currentTileLength);\n";
    } else {
        code.computeCode += updateCode;
    }

    if (!postCode.empty()) {
        code.computeCode += postCode;
    }

    code.computeCode += "        " + dstQueueName + ".EnQue<" + dstKernelType + ">(dstLocal);\n";
    code.computeCode += "        " + srcQueueName + ".FreeTensor(srcLocal);\n";
    code.computeCode += "        " + srcShiftQueueName + ".FreeTensor(srcShiftLocal);\n";

    code.copyOutCode += "        AscendC::LocalTensor<" + dstKernelType + "> dstLocal = " + dstQueueName + ".DeQue<" + dstKernelType + ">();\n";
    code.copyOutCode += "        AscendC::DataCopy(" + dstGmName + "[tileOffset], dstLocal, currentTileLength);\n";
    code.copyOutCode += "        " + dstQueueName + ".FreeTensor(dstLocal);\n";

    code.tQueDecl += "    AscendC::GlobalTensor<" + srcKernelType + "> " + srcGmName + ";\n";
    code.tQueDecl += "    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> " + srcQueueName + ";\n";
    code.tQueDecl += "    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> " + srcShiftQueueName + ";\n";
    code.tQueDecl += "    uint32_t " + srcInputLengthName + " = 0;\n";
    code.tQueDecl += "    AscendC::GlobalTensor<" + dstKernelType + "> " + dstGmName + ";\n";
    code.tQueDecl += "    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> " + dstQueueName + ";\n";
    code.tQueDecl += "    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuffer0;\n";
    code.tQueDecl += "    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuffer1;\n";
    for (int i = 0; i < scratchBufferCount; ++i) {
        code.tQueDecl += "    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuffer" + std::to_string(i + 2) + ";\n";
    }

    return code;
}

StructuredMultiViewExprFastPathCode CodeGen_StructuredMultiViewExprFastPath(
    std::string srcKernelType,
    std::string dstKernelType,
    std::string scalarKernelType,
    std::string srcName,
    std::string dstName,
    std::string scalarName,
    const std::vector<int>& viewOffsets,
    bool hasScalarInput,
    int scratchBufferCount,
    std::string mainExprCode,
    std::string postExprCode,
    int vectorComputeAlignBytes) {
    StructuredMultiViewExprFastPathCode code;
    if (vectorComputeAlignBytes <= 0) {
        vectorComputeAlignBytes = 256;
    }

    std::vector<int> offsets = viewOffsets;
    if (offsets.empty()) {
        offsets.push_back(0);
    }
    const int minOffset = *std::min_element(offsets.begin(), offsets.end());
    const int maxOffset = *std::max_element(offsets.begin(), offsets.end());
    const int spanExtra = maxOffset - minOffset;

    const std::string srcGmName = srcName + "Gm";
    const std::string dstGmName = dstName + "Gm";
    const std::string srcValidLengthName = srcName + "ValidLength";
    const std::string srcGmStartName = srcName + "GmStart";
    const std::string srcGmLengthName = srcName + "GmLength";

    const bool needsCalcBuffer0 = ExprMentionsSymbol(mainExprCode, "calcBuffer0Local") ||
                                  ExprMentionsSymbol(postExprCode, "calcBuffer0Local");
    const bool needsCalcBuffer1 = ExprMentionsSymbol(mainExprCode, "calcBuffer1Local") ||
                                  ExprMentionsSymbol(postExprCode, "calcBuffer1Local");
    int maxExprCalcBuffer = FindMaxCalcBufferIndexInExpr(mainExprCode, postExprCode);
    if (maxExprCalcBuffer < 1 && scratchBufferCount > 0) {
        maxExprCalcBuffer = 1 + scratchBufferCount;
    }

    code.initCode += "        " + srcValidLengthName + " = tileLength;\n";
    code.initCode += "        " + srcGmStartName + " = static_cast<int32_t>(blockStart) + static_cast<int32_t>(" +
                     std::to_string(minOffset) + ");\n";
    code.initCode += "        if (" + srcGmStartName + " < 0) { " + srcGmStartName + " = 0; }\n";
    const std::string srcAlignElemsExpr =
        "static_cast<uint32_t>((" + std::to_string(vectorComputeAlignBytes) +
        " + sizeof(" + srcKernelType + ") - 1) / sizeof(" + srcKernelType + "))";
    code.initCode += "        " + srcGmLengthName + " = (blockLength + static_cast<uint32_t>(" +
                     std::to_string(spanExtra) + ") + " + srcAlignElemsExpr + " - 1) / " +
                     srcAlignElemsExpr + " * " + srcAlignElemsExpr + ";\n";
    code.initCode += "        " + srcGmName + ".SetGlobalBuffer((__gm__ " + srcKernelType + " *)" + srcName +
                     " + " + srcGmStartName + ", " + srcGmLengthName + ");\n";

    for (size_t i = 0; i < offsets.size(); ++i) {
        const std::string viewQueueName = "inQueue" + srcName + "View" + std::to_string(i);
        code.initCode += "        pipe.InitBuffer(" + viewQueueName + ", BUFFER_NUM, " + srcValidLengthName +
                         " * sizeof(" + srcKernelType + "));\n";
    }

    if (hasScalarInput) {
        const std::string scalarGmName = scalarName + "ScalarGm";
        code.initCode += "        " + scalarGmName + ".SetGlobalBuffer((__gm__ " + scalarKernelType + " *)" +
                         scalarName + ", 1);\n";
        code.initCode += "        pipe.InitBuffer(scalar0Buffer, sizeof(" + scalarKernelType + "));\n";
        code.initCode += "        {\n";
        code.initCode += "            AscendC::LocalTensor<" + scalarKernelType + "> scalar0Local = scalar0Buffer.Get<" +
                         scalarKernelType + ">();\n";
        code.initCode += "            AscendC::DataCopy(scalar0Local, " + scalarGmName + "[0], 1);\n";
        code.initCode += "        }\n";
    }

    code.initCode += "        " + dstGmName + ".SetGlobalBuffer((__gm__ " + dstKernelType + " *)" + dstName +
                     " + blockStart, blockLength);\n";
    code.initCode += "        pipe.InitBuffer(outQueue" + dstName + ", BUFFER_NUM, tileLength" +
                     " * sizeof(" + dstKernelType + "));\n";

    if (needsCalcBuffer0) {
        code.initCode += "        pipe.InitBuffer(calcBuffer0, tileLength * sizeof(" + dstKernelType + "));\n";
    }
    if (needsCalcBuffer1) {
        code.initCode += "        pipe.InitBuffer(calcBuffer1, tileLength * sizeof(" + dstKernelType + "));\n";
    }
    for (int bufferIdx = 2; bufferIdx <= maxExprCalcBuffer; ++bufferIdx) {
        const std::string localName = "calcBuffer" + std::to_string(bufferIdx) + "Local";
        const bool usedInExpr = ExprMentionsSymbol(mainExprCode, localName) || ExprMentionsSymbol(postExprCode, localName);
        if (!usedInExpr) {
            continue;
        }
        code.initCode += "        pipe.InitBuffer(calcBuffer" + std::to_string(bufferIdx) + ", tileLength * sizeof(" +
                         dstKernelType + "));\n";
    }

    for (size_t i = 0; i < offsets.size(); ++i) {
        const int offsetShift = offsets[i] - minOffset;
        const std::string viewLocal = "view" + std::to_string(i) + "Local";
        const std::string viewQueueName = "inQueue" + srcName + "View" + std::to_string(i);
        const std::string baseName = srcName + "View" + std::to_string(i) + "BaseOffset";
        code.copyInCode += "        AscendC::LocalTensor<" + srcKernelType + "> " + viewLocal + " = " +
                           viewQueueName + ".AllocTensor<" + srcKernelType + ">();\n";
        code.copyInCode += "        int32_t " + baseName +
                           " = static_cast<int32_t>(tileOffset) + static_cast<int32_t>(" +
                           std::to_string(offsetShift) + ");\n";
        code.copyInCode += "        AscendC::DataCopy(" + viewLocal + ", " + srcGmName + "[" + baseName + "], " +
                           "currentTileLength);\n";
        code.copyInCode += "        " + viewQueueName + ".EnQue(" + viewLocal + ");\n";
    }

    for (size_t i = 0; i < offsets.size(); ++i) {
        const std::string viewLocal = "view" + std::to_string(i) + "Local";
        const std::string viewQueueName = "inQueue" + srcName + "View" + std::to_string(i);
        code.computeCode += "        AscendC::LocalTensor<" + srcKernelType + "> " + viewLocal + " = " +
                            viewQueueName + ".DeQue<" + srcKernelType + ">();\n";
    }
    if (hasScalarInput) {
        code.computeCode += "        float scalar0Value = static_cast<float>(" + scalarName +
                            "ScalarGm.GetValue(0));\n";
    }
    code.computeCode += "        AscendC::LocalTensor<" + dstKernelType + "> dstLocal = outQueue" + dstName +
                        ".AllocTensor<" + dstKernelType + ">();\n";
    if (needsCalcBuffer0) {
        code.computeCode += "        AscendC::LocalTensor<" + dstKernelType + "> calcBuffer0Local = calcBuffer0.Get<" +
                            dstKernelType + ">();\n";
        code.computeCode += "        (void)calcBuffer0Local;\n";
    }
    if (needsCalcBuffer1) {
        code.computeCode += "        AscendC::LocalTensor<" + dstKernelType + "> calcBuffer1Local = calcBuffer1.Get<" +
                            dstKernelType + ">();\n";
        code.computeCode += "        (void)calcBuffer1Local;\n";
    }
    for (int bufferIdx = 2; bufferIdx <= maxExprCalcBuffer; ++bufferIdx) {
        const std::string localName = "calcBuffer" + std::to_string(bufferIdx) + "Local";
        const bool usedInExpr = ExprMentionsSymbol(mainExprCode, localName) || ExprMentionsSymbol(postExprCode, localName);
        if (!usedInExpr) {
            continue;
        }
        code.computeCode += "        AscendC::LocalTensor<" + dstKernelType + "> " + localName + " = calcBuffer" +
                            std::to_string(bufferIdx) + ".Get<" + dstKernelType + ">();\n";
        code.computeCode += "        (void)" + localName + ";\n";
    }

    if (mainExprCode.empty()) {
        code.computeCode += "        AscendC::Muls(dstLocal, view0Local, 1.0f, currentTileLength);\n";
    } else {
        code.computeCode += mainExprCode;
    }
    if (!postExprCode.empty()) {
        code.computeCode += postExprCode;
    }

    code.computeCode += "        outQueue" + dstName + ".EnQue<" + dstKernelType + ">(dstLocal);\n";
    for (size_t i = 0; i < offsets.size(); ++i) {
        const std::string viewLocal = "view" + std::to_string(i) + "Local";
        const std::string viewQueueName = "inQueue" + srcName + "View" + std::to_string(i);
        code.computeCode += "        " + viewQueueName + ".FreeTensor(" + viewLocal + ");\n";
    }

    code.copyOutCode += "        AscendC::LocalTensor<" + dstKernelType + "> dstLocal = outQueue" + dstName +
                        ".DeQue<" + dstKernelType + ">();\n";
    code.copyOutCode += "        AscendC::DataCopy(" + dstGmName + "[tileOffset], dstLocal, currentTileLength);\n";
    code.copyOutCode += "        outQueue" + dstName + ".FreeTensor(dstLocal);\n";

    code.tQueDecl += "    AscendC::GlobalTensor<" + srcKernelType + "> " + srcGmName + ";\n";
    for (size_t i = 0; i < offsets.size(); ++i) {
        const std::string viewQueueName = "inQueue" + srcName + "View" + std::to_string(i);
        code.tQueDecl += "    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> " + viewQueueName + ";\n";
    }
    code.tQueDecl += "    uint32_t " + srcValidLengthName + " = 0;\n";
    code.tQueDecl += "    int32_t " + srcGmStartName + " = 0;\n";
    code.tQueDecl += "    uint32_t " + srcGmLengthName + " = 0;\n";

    if (hasScalarInput) {
        code.tQueDecl += "    AscendC::GlobalTensor<" + scalarKernelType + "> " + scalarName + "ScalarGm;\n";
        code.tQueDecl += "    AscendC::TBuf<AscendC::QuePosition::VECCALC> scalar0Buffer;\n";
    }

    code.tQueDecl += "    AscendC::GlobalTensor<" + dstKernelType + "> " + dstGmName + ";\n";
    code.tQueDecl += "    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue" + dstName + ";\n";
    if (needsCalcBuffer0) {
        code.tQueDecl += "    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuffer0;\n";
    }
    if (needsCalcBuffer1) {
        code.tQueDecl += "    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuffer1;\n";
    }
    for (int bufferIdx = 2; bufferIdx <= maxExprCalcBuffer; ++bufferIdx) {
        const std::string localName = "calcBuffer" + std::to_string(bufferIdx) + "Local";
        const bool usedInExpr = ExprMentionsSymbol(mainExprCode, localName) || ExprMentionsSymbol(postExprCode, localName);
        if (!usedInExpr) {
            continue;
        }
        code.tQueDecl += "    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuffer" + std::to_string(bufferIdx) + ";\n";
    }

    return code;
}

// 1. Cube Kernel 侧模板  NPU 计算流
const char* CUBE_KERNEL_FILE_Template = R"~~~(
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

__aicore__ inline void CopyTiling(TCubeTiling *tiling, GM_ADDR tilingGM) {
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);
    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
}

__aicore__ inline void CalcGMOffset(int blockIdx, const TCubeTiling &tiling, int &offsetA, int &offsetB, int &offsetC,
                                    int &tailM, int &tailN, bool isTransA, bool isTransB) {
    uint32_t mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
    uint32_t mCoreIndx = blockIdx % mSingleBlocks;
    uint32_t nCoreIndx = blockIdx / mSingleBlocks;

    offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
    if (isTransA) offsetA = mCoreIndx * tiling.singleCoreM;

    offsetB = nCoreIndx * tiling.singleCoreN;
    if (isTransB) offsetB = nCoreIndx * tiling.Kb * tiling.singleCoreN;

    offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;

    tailM = tiling.M - mCoreIndx * tiling.singleCoreM;
    tailM = tailM < tiling.singleCoreM ? tailM : tiling.singleCoreM;
    tailN = tiling.N - nCoreIndx * tiling.singleCoreN;
    tailN = tailN < tiling.singleCoreN ? tailN : tiling.singleCoreN;
}

extern "C" __global__ __aicore__ void {{KERNEL_NAME}}(GM_ADDR {{NAME_A}}, GM_ADDR {{NAME_B}}, GM_ADDR {{NAME_C}}, GM_ADDR workspace, GM_ADDR tilingGm)
{
    using A_T = {{TYPE_A}};
    using B_T = {{TYPE_B}};
    using C_T = {{TYPE_C}};
    bool isBias = {{IS_BIAS}};

    AscendC::TPipe pipe;
    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);

    AscendC::GlobalTensor<A_T> aGlobal;
    AscendC::GlobalTensor<B_T> bGlobal;
    AscendC::GlobalTensor<C_T> cGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>({{NAME_A}}), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>({{NAME_B}}), tiling.Ka * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>({{NAME_C}}), tiling.M * tiling.N);

    int offsetA = 0;
    int offsetB = 0;
    int offsetC = 0;
    bool isTransA = {{TRANS_A}};
    bool isTransB = {{TRANS_B}};

    int tailM = 0;
    int tailN = 0;
    CalcGMOffset(GetBlockIdx(), tiling, offsetA, offsetB, offsetC, tailM, tailN, isTransA, isTransB);

    auto gmA = aGlobal[offsetA];
    auto gmB = bGlobal[offsetB];
    auto gmC = cGlobal[offsetC];

    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_T>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_T>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_T>> mm;

    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);

    mm.SetTensorA(gmA, isTransA);
    mm.SetTensorB(gmB, isTransB);
    mm.SetTail(tailM, tailN);
    // Bias 累加能力由本地 SDK 决定，当前默认保守路径（A*B），Host 侧可做回加兜底。
    (void)isBias;
    mm.IterateAll(gmC);
    mm.End();
}
)~~~";

std::string CodeGen_CubeKernelFile(
    std::string kernelName,
    std::string typeA, std::string typeB, std::string typeC,
    std::string nameA, std::string nameB, std::string nameC,
    std::string transA, std::string transB,
    std::string isBias) {

    return UNIVERSAL_TEMPLATE::templateString(CUBE_KERNEL_FILE_Template, {
        {"{{KERNEL_NAME}}", kernelName},
        {"{{TYPE_A}}", typeA},
        {"{{TYPE_B}}", typeB},
        {"{{TYPE_C}}", typeC},
        {"{{NAME_A}}", nameA},
        {"{{NAME_B}}", nameB},
        {"{{NAME_C}}", nameC},
        {"{{TRANS_A}}", transA},
        {"{{TRANS_B}}", transB},
        {"{{IS_BIAS}}", isBias}
    });
}

// 2. Cube Tiling 侧模板 (CPU 切分算法)

// Tiling 单算子生成函数模板
const char* CUBE_TILING_FUNC_Template = R"~~~(
extern "C" uint8_t *{{SHAPE_TILING_NAME}}(
    const char *socVersion,
    int M,
    int N,
    int K,
    int usedCoreNum);

extern "C" uint8_t *{{TILING_NAME}}(const char *socVersion)
{
    int M = {{M}};
    int N = {{N}};
    int K = {{K}};
    int usedCoreNum = 1; // 默认使用1核测试
    return {{SHAPE_TILING_NAME}}(socVersion, M, N, K, usedCoreNum);
}

extern "C" uint8_t *{{SHAPE_TILING_NAME}}(
    const char *socVersion,
    int M,
    int N,
    int K,
    int usedCoreNum)
{
    M = std::max(M, 1);
    N = std::max(N, 1);
    K = std::max(K, 1);
    usedCoreNum = std::max(usedCoreNum, 1);

    TPosition leftPosition = TPosition::GM;
    CubeFormat leftFormat = CubeFormat::ND;
    DataType leftDtype = {{TYPE_A}};
    bool isTransA = {{TRANS_A}};

    TPosition rightPosition = TPosition::GM;
    CubeFormat rightFormat = CubeFormat::ND;
    DataType rightDtype = {{TYPE_B}};
    bool isTransB = {{TRANS_B}};

    TPosition resultPosition = TPosition::GM;
    CubeFormat resultFormat = CubeFormat::ND;
    DataType resultDtype = {{TYPE_C}};

    bool isBias = {{IS_BIAS}};
    int32_t baseM = 16;  // 基础切分
    int32_t baseN = 16;

    optiling::TCubeTiling tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    MultiCoreMatmulTiling tilingApi(*ascendcPlatform);

    tilingApi.SetDim(usedCoreNum);
    tilingApi.SetAType(leftPosition, leftFormat, leftDtype, isTransA);
    tilingApi.SetBType(rightPosition, rightFormat, rightDtype, isTransB);
    tilingApi.SetCType(resultPosition, resultFormat, resultDtype);

    tilingApi.SetOrgShape(M, N, K);
    tilingApi.SetShape(M, N, K);
    tilingApi.SetFixSplit(baseM, baseN, -1);
    tilingApi.SetBias(isBias);
    tilingApi.SetBufferSpace(-1, -1, -1);

    int64_t res = tilingApi.GetTiling(tilingData);
    if (res == -1) {
        std::cout << "gen tiling failed" << std::endl;
    }
    return GetTilingBuf(&tilingData);
}
)~~~";

// Tiling 整体文件外壳模板
const char* CUBE_TILING_FILE_Template = R"~~~(
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <cstdlib>
#include <limits>
#include <stdexcept>

#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
using namespace matmul_tiling;
using namespace std;

uint8_t *GetTilingBuf(optiling::TCubeTiling *tilingData)
{
    uint32_t tilingSize = tilingData->GetDataSize();
    uint8_t *buf = (uint8_t *)malloc(tilingSize);
    tilingData->SaveToBuffer(buf, tilingSize);
    return buf;
}

{{TILING_FUNCS}}
)~~~";

std::string CodeGen_CubeTilingFunc(
    std::string tilingName, std::string shapeTilingName,
    std::string dimM, std::string dimN, std::string dimK,
    std::string typeA, std::string typeB, std::string typeC,
    std::string transA, std::string transB,
    std::string isBias) {

    return UNIVERSAL_TEMPLATE::templateString(CUBE_TILING_FUNC_Template, {
        {"{{TILING_NAME}}", tilingName},
        {"{{SHAPE_TILING_NAME}}", shapeTilingName},
        {"{{M}}", dimM}, {"{{N}}", dimN}, {"{{K}}", dimK},
        {"{{TYPE_A}}", typeA}, {"{{TYPE_B}}", typeB}, {"{{TYPE_C}}", typeC},
        {"{{TRANS_A}}", transA}, {"{{TRANS_B}}", transB},
        {"{{IS_BIAS}}", isBias}
    });
}

std::string CodeGen_CubeTilingFile(std::string tilingFuncs) {
    return UNIVERSAL_TEMPLATE::templateString(CUBE_TILING_FILE_Template, {
        {"{{TILING_FUNCS}}", tilingFuncs}
    });
}

const char* VECTOR_MAIN_BODY_Template = R"~~~(int32_t main(int32_t argc, char *argv[]) {
    dacpp_host::Session {{SESSION_NAME}};
{{HOST_FLOW_BODY}}
    return {{SESSION_NAME}}.finish();
}
)~~~";

const char* VECTOR_HOST_FILE_Template = R"~~~(//  自动生成的 AscendC Vector Host 代码
{{HOST_RUNTIME_INSERTION}}
{{FEATURE_INCLUDES}}
{{VEC_EXTERN_DO_DECLS}}
{{VEC_PRELUDE_CODE}}{{VECTOR_MAIN_BODY}}
)~~~";

const char* CUBE_MAIN_BODY_Template = R"~~~(int32_t main(int32_t argc, char *argv[]) {
    dacpp_host::Session {{SESSION_NAME}};
{{HOST_FLOW_BODY}}
    return {{SESSION_NAME}}.finish();
}
)~~~";

const char* CUBE_HOST_FILE_Template = R"~~~(//  自动生成的 AscendC Cube Host 代码
{{HOST_RUNTIME_INSERTION}}
{{FEATURE_INCLUDES}}
//  包含矩阵 Tiling 所需的核心依赖
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"

{{CUBE_EXTERN_DO_DECLS}}
{{CUBE_TILING_DECLS}}
{{CUBE_PRELUDE_CODE}}{{CUBE_MAIN_BODY}}
)~~~";

static std::string BuildUserHostIncludes(const std::string& code) {
    std::string includes;
    auto AddIfUsed = [&](const char* token, const char* header) {
        if (code.find(token) != std::string::npos) {
            includes += "#include <";
            includes += header;
            includes += ">\n";
        }
    };
    AddIfUsed("std::array", "array");
    AddIfUsed("std::complex", "complex");
    AddIfUsed("std::pair", "utility");
    if (code.find("std::remove_") != std::string::npos ||
        code.find("std::decay") != std::string::npos ||
        code.find("std::is_") != std::string::npos ||
        code.find("std::enable_") != std::string::npos) {
        includes += "#include <type_traits>\n";
    }
    if (code.find("std::setprecision") != std::string::npos ||
        code.find("std::setw") != std::string::npos) {
        includes += "#include <iomanip>\n";
    }
    AddIfUsed("std::time", "ctime");
    return includes;
}

std::string CodeGen_VectorMainBody(std::string hostFlowBody,
                                   std::string sessionName) {
    return UNIVERSAL_TEMPLATE::templateString(VECTOR_MAIN_BODY_Template, {
        {"{{SESSION_NAME}}", sessionName},
        {"{{HOST_FLOW_BODY}}", hostFlowBody}
    });
}

std::string CodeGen_VectorHostFile(
    std::string vecExternDoDecls,
    std::string vecPreludeCode,
    std::string vectorMainBody,
    const dacppTranslator::ascendc::HostRuntimeRequirements& runtimeRequirements) {
    const std::string featureIncludes = BuildUserHostIncludes(
        vecExternDoDecls + vecPreludeCode + vectorMainBody);
    return UNIVERSAL_TEMPLATE::templateString(VECTOR_HOST_FILE_Template, {
        {"{{HOST_RUNTIME_INSERTION}}",
         dacppTranslator::ascendc::EmitHostRuntimeInsertion(
             runtimeRequirements)},
        {"{{VEC_EXTERN_DO_DECLS}}", vecExternDoDecls},
        {"{{VEC_PRELUDE_CODE}}", vecPreludeCode},
        {"{{FEATURE_INCLUDES}}", featureIncludes},
        {"{{VECTOR_MAIN_BODY}}", vectorMainBody}
    });
}

std::string CodeGen_VectorTemporalStencilSchedule(
    const TemporalLoopSchedule& spec,
    std::string launchEntry,
    std::string blockDimVar,
    std::string launchArgs,
    std::string inputTensorName,
    std::string outputTensorName,
    std::string deviceName,
    std::string streamName) {
    if (!spec.recognized || launchEntry.empty()) {
        return "";
    }

    std::string indexVar = spec.loopIndexVar.empty() ? "k" : spec.loopIndexVar;
    std::string loopBegin = spec.loopBeginExpr.empty() ? "0" : spec.loopBeginExpr;
    std::string upperBound = spec.loopUpperBoundExpr.empty() ? "0" : spec.loopUpperBoundExpr;
    std::string stepExpr = spec.loopStepExpr.empty() ? ("++" + indexVar) : spec.loopStepExpr;
    std::string writebackBegin = spec.writebackRangeBeginExpr.empty() ? "1" : spec.writebackRangeBeginExpr;
    std::string writebackEnd = spec.writebackRangeEndExpr.empty() ? "temporalWriteCount" : spec.writebackRangeEndExpr;
    const std::string inputBufName = inputTensorName + "Buf";
    const std::string outputBufName = outputTensorName + "Buf";

    std::string code;
    code += "    // temporal scheduling body\n";
    code += "    for (int " + indexVar + " = " + loopBegin + "; " + indexVar + " < " + upperBound + "; " + stepExpr + ") {\n";
    code += "        " + deviceName + ".launch(" + launchEntry + "(" +
        blockDimVar + ", " + streamName;
    if (!launchArgs.empty()) {
        code += ", " + launchArgs;
    }
    code += "));\n";
    code += "        " + deviceName + ".sync();\n";
    code += "\n";
    code += "        // Pull current step result and apply loop-carried writeback.\n";
    code += "        " + deviceName + ".copyOut(" + outputBufName + ");\n";
    code += "\n";
    code += "        size_t temporalWriteCount = std::min(" + outputBufName + ".logicalElems, " +
        inputBufName + ".logicalElems);\n";
    code += "        for (size_t i = static_cast<size_t>(" + writebackBegin + "); i <= temporalWriteCount; ++i) {\n";
    code += "            long long srcSigned = static_cast<long long>(i) + static_cast<long long>(" + std::to_string(spec.writebackSrcSpatialOffset) + ");\n";
    code += "            if (srcSigned < 0) {\n";
    code += "                continue;\n";
    code += "            }\n";
    code += "            size_t srcIndex = static_cast<size_t>(srcSigned);\n";
    code += "            if (srcIndex >= " + outputBufName + ".logicalElems) {\n";
    code += "                continue;\n";
    code += "            }\n";
    code += "            " + inputBufName + ".data[i] = " + outputBufName + ".data[srcIndex];\n";
    code += "        }\n";
    code += "\n";
    code += "        " + deviceName + ".copyIn(" + inputBufName + ");\n";
    code += "    }\n";
    return code;
}

std::string CodeGen_CubeMainBody(
    std::string hostFlowBody,
    std::string sessionName) {
    return UNIVERSAL_TEMPLATE::templateString(CUBE_MAIN_BODY_Template, {
        {"{{SESSION_NAME}}", sessionName},
        {"{{HOST_FLOW_BODY}}", hostFlowBody}
    });
}

std::string CodeGen_CubeHostFile(
    std::string cubeExternDoDecls,
    std::string cubeTilingDecls,
    std::string cubeMainBody,
    std::string cubePreludeCode,
    const dacppTranslator::ascendc::HostRuntimeRequirements& runtimeRequirements) {
    const std::string featureIncludes = BuildUserHostIncludes(
        cubeExternDoDecls + cubeTilingDecls + cubeMainBody + cubePreludeCode);
    return UNIVERSAL_TEMPLATE::templateString(CUBE_HOST_FILE_Template, {
        {"{{HOST_RUNTIME_INSERTION}}",
         dacppTranslator::ascendc::EmitHostRuntimeInsertion(
             runtimeRequirements)},
        {"{{CUBE_EXTERN_DO_DECLS}}", cubeExternDoDecls},
        {"{{CUBE_TILING_DECLS}}", cubeTilingDecls},
        {"{{CUBE_PRELUDE_CODE}}", cubePreludeCode},
        {"{{FEATURE_INCLUDES}}", featureIncludes},
        {"{{CUBE_MAIN_BODY}}", cubeMainBody}
    });
}

} // namespace ASCENDC_TEMPLATE
