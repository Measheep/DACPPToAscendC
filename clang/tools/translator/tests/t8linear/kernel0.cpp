
#include "kernel_operator.h"
using namespace AscendC;

// Vector tiling/resource plan (translation-time selection)
// UB budget: 131072 bytes
// UB reserve: 16384 bytes
// selected BUFFER_NUM: 2
// tile alignment: 64 elements
// max tile elements: 192
// bytes per tile estimate: 102400
// selection source: auto (default), ub-budget=default, ub-reserve=default
// allocation breakdown:
//   fixed: 1024 bytes
//   linear-weight-tile (input-queue): 98304 bytes
//   linear-bias-tile (input-queue): 1536 bytes
//   linear-output-tile (output-queue): 1536 bytes

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_ALIGN_ELEMENTS = 64;
constexpr uint32_t MAX_TILE_ELEMENTS = 192;
constexpr uint32_t DACPP_VECTOR_ALIGN_BYTES = 256;

//  2. 算子类定义 
class Kernel_vector_kernel_0 {
public:
    __aicore__ inline Kernel_vector_kernel_0() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR weight, GM_ADDR bias, GM_ADDR output, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount) {
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
        inputGm.SetGlobalBuffer((__gm__ float*)input, 64);
        weightGm.SetGlobalBuffer((__gm__ float*)weight + static_cast<size_t>(blockStart) * 64, 64 * blockLength);
        biasGm.SetGlobalBuffer((__gm__ float*)bias + blockStart, blockLength);
        outputGm.SetGlobalBuffer((__gm__ float*)output + blockStart, blockLength);
        pipe.InitBuffer(inQueueinput, BUFFER_NUM, 64 * sizeof(float));
        pipe.InitBuffer(inQueueweight, BUFFER_NUM, 64 * tileLength * sizeof(float));
        pipe.InitBuffer(inQueuebias, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(outQueueoutput, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(calcBuf, 64 * sizeof(float));
        pipe.InitBuffer(calcBufReduce, 64 * sizeof(float));

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
        AscendC::LocalTensor<float> inputLocal = inQueueinput.AllocTensor<float>();
        AscendC::LocalTensor<float> weightLocal = inQueueweight.AllocTensor<float>();
        AscendC::LocalTensor<float> biasLocal = inQueuebias.AllocTensor<float>();
        AscendC::DataCopy(inputLocal, inputGm, 64);
        AscendC::DataCopy(weightLocal, weightGm[static_cast<size_t>(tileOffset) * 64], 64 * currentTileLength);
        inQueueinput.EnQue(inputLocal);
        inQueueweight.EnQue(weightLocal);
        AscendC::DataCopy(biasLocal, biasGm[tileOffset], currentTileLength);
        inQueuebias.EnQue(biasLocal);

    }
    __aicore__ inline void Compute(uint32_t progress, uint32_t tileOffset,
                                   uint32_t currentTileLength) {
        AscendC::LocalTensor<float> inputLocal = inQueueinput.DeQue<float>();
        AscendC::LocalTensor<float> weightLocal = inQueueweight.DeQue<float>();
        AscendC::LocalTensor<float> biasLocal = inQueuebias.DeQue<float>();
        AscendC::LocalTensor<float> outputLocal = outQueueoutput.AllocTensor<float>();
        AscendC::LocalTensor<float> tmpLocal = calcBuf.Get<float>();
        AscendC::LocalTensor<float> reduceLocal = calcBufReduce.Get<float>();
        for (uint32_t i = 0; i < currentTileLength; ++i) {
            AscendC::LocalTensor<float> weightRow = weightLocal[i * 64];
            AscendC::Mul(tmpLocal, inputLocal, weightRow, 64);
            AscendC::WholeReduceSum(reduceLocal, tmpLocal, 64, 1, 1, 1, 1);
            float sum = biasLocal.GetValue(i) + reduceLocal.GetValue(0);
            outputLocal.SetValue(i, sum);
        }
        outQueueoutput.EnQue<float>(outputLocal);
        inQueueinput.FreeTensor(inputLocal);
        inQueueweight.FreeTensor(weightLocal);
        inQueuebias.FreeTensor(biasLocal);

    }
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t tileOffset,
                                   uint32_t currentTileLength) {
        AscendC::LocalTensor<float> outCopy = outQueueoutput.DeQue<float>();
        AscendC::DataCopy(outputGm[tileOffset], outCopy, currentTileLength);
        outQueueoutput.FreeTensor(outCopy);

    }

private:
    TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBufReduce;
    AscendC::GlobalTensor<float> inputGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueinput;
    AscendC::GlobalTensor<float> weightGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueweight;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueuebias;
    AscendC::GlobalTensor<float> outputGm;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueoutput;


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
extern "C" __global__ __aicore__ void vector_kernel_0(GM_ADDR input, GM_ADDR weight, GM_ADDR bias, GM_ADDR output, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount) {
    Kernel_vector_kernel_0 op;
    op.Init(input, weight, bias, output, totalLength, coreNum, tileLength, tileCount);
    op.Process();
}

//  4. Host 端启动 Wrapper 函数 
#ifndef ASCENDC_CPU_DEBUG
// #include "acl/acl.h"
// 宿主机包装器接收统一的运行时 tiling 参数。
extern "C" uint32_t vector_kernel_0_do(uint32_t blockDim, void *stream, uint8_t *input, uint8_t *weight, uint8_t *bias, uint8_t *output, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount) {
    return vector_kernel_0<<<blockDim, nullptr, stream>>>(input, weight, bias, output, totalLength, coreNum, tileLength, tileCount);
}
#endif
