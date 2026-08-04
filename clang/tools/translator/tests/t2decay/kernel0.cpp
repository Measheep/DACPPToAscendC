
#include "kernel_operator.h"
using namespace AscendC;

// Vector tiling/resource plan (translation-time selection)
// UB budget: 131072 bytes
// UB reserve: 16384 bytes
// selected BUFFER_NUM: 2
// tile alignment: 64 elements
// max tile elements: 3584
// bytes per tile estimate: 114688
// selection source: auto (default), ub-budget=default, ub-reserve=default
// allocation breakdown:
//   inQueueN0s (input-queue): 28672 bytes
//   inQueuelambdas (input-queue): 28672 bytes
//   outQueuelocal_A (output-queue): 28672 bytes
//   inQueuet (broadcast-queue): 28672 bytes

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_ALIGN_ELEMENTS = 64;
constexpr uint32_t MAX_TILE_ELEMENTS = 3584;
constexpr uint32_t DACPP_VECTOR_ALIGN_BYTES = 256;

//  2. 算子类定义 
class Kernel_vector_kernel_0 {
public:
    __aicore__ inline Kernel_vector_kernel_0() {}
    __aicore__ inline void Init(GM_ADDR N0s, GM_ADDR lambdas, GM_ADDR local_A, GM_ADDR t, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount) {
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
        N0sGm.SetGlobalBuffer((__gm__ float *)N0s + blockStart, blockLength);
        pipe.InitBuffer(inQueueN0s, BUFFER_NUM, tileLength * sizeof(float));
        lambdasGm.SetGlobalBuffer((__gm__ float *)lambdas + blockStart, blockLength);
        pipe.InitBuffer(inQueuelambdas, BUFFER_NUM, tileLength * sizeof(float));
        local_AGm.SetGlobalBuffer((__gm__ float *)local_A + blockStart, blockLength);
        pipe.InitBuffer(outQueuelocal_A, BUFFER_NUM, tileLength * sizeof(float));
        tGm.SetGlobalBuffer((__gm__ float *)t, 1);
        pipe.InitBuffer(inQueuet, BUFFER_NUM, tileLength * sizeof(float));

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
        AscendC::LocalTensor<float> N0sLocal = inQueueN0s.AllocTensor<float>();
        AscendC::DataCopy(N0sLocal, N0sGm[tileOffset], currentTileLength);
        inQueueN0s.EnQue(N0sLocal);
        AscendC::LocalTensor<float> lambdasLocal = inQueuelambdas.AllocTensor<float>();
        AscendC::DataCopy(lambdasLocal, lambdasGm[tileOffset], currentTileLength);
        inQueuelambdas.EnQue(lambdasLocal);
        AscendC::LocalTensor<float> tLocal = inQueuet.AllocTensor<float>();
        float tScalarValue = tGm.GetValue(0);
        tLocal.SetValue(0, tScalarValue);
        for (int32_t fill = 1; fill < static_cast<int32_t>(currentTileLength); ++fill) {
            tLocal.SetValue(fill, tScalarValue);
        }
        inQueuet.EnQue(tLocal);

    }
    __aicore__ inline void Compute(uint32_t progress, uint32_t tileOffset,
                                   uint32_t currentTileLength) {
        AscendC::LocalTensor<float> N0sLocal = inQueueN0s.DeQue<float>();
        AscendC::LocalTensor<float> lambdasLocal = inQueuelambdas.DeQue<float>();
        AscendC::LocalTensor<float> tLocal = inQueuet.DeQue<float>();
        const auto tScalarValue = tGm.GetValue(0);
        AscendC::LocalTensor<float> local_ALocal = outQueuelocal_A.AllocTensor<float>();
        // decay-scalar-fast-path: graph contains exp-mul decay chain with multi-input tensors; diag(exp=1,mul=1,min=0,offsetNZ=0,offsetUnknown=0,windowBaseLoads=3,argTensors=3,argLoads=3)
        const auto decayScalarExprTmp0 = (tScalarValue);
        AscendC::Muls(lambdasLocal, lambdasLocal, (-decayScalarExprTmp0), currentTileLength);
        AscendC::Exp(lambdasLocal, lambdasLocal, currentTileLength);
        AscendC::Mul(local_ALocal, N0sLocal, lambdasLocal, currentTileLength);
        outQueuelocal_A.EnQue<float>(local_ALocal);
        inQueueN0s.FreeTensor(N0sLocal);
        inQueuelambdas.FreeTensor(lambdasLocal);
        inQueuet.FreeTensor(tLocal);

    }
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t tileOffset,
                                   uint32_t currentTileLength) {
        AscendC::LocalTensor<float> local_ALocal = outQueuelocal_A.DeQue<float>();
        AscendC::DataCopy(local_AGm[tileOffset], local_ALocal, currentTileLength);
        outQueuelocal_A.FreeTensor(local_ALocal);

    }

private:
    TPipe pipe;
    AscendC::GlobalTensor<float> N0sGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueN0s;
    AscendC::GlobalTensor<float> lambdasGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueuelambdas;
    AscendC::GlobalTensor<float> local_AGm;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueuelocal_A;
    AscendC::GlobalTensor<float> tGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueuet;


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
extern "C" __global__ __aicore__ void vector_kernel_0(GM_ADDR N0s, GM_ADDR lambdas, GM_ADDR local_A, GM_ADDR t, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount) {
    Kernel_vector_kernel_0 op;
    op.Init(N0s, lambdas, local_A, t, totalLength, coreNum, tileLength, tileCount);
    op.Process();
}

//  4. Host 端启动 Wrapper 函数 
#ifndef ASCENDC_CPU_DEBUG
// #include "acl/acl.h"
// 宿主机包装器接收统一的运行时 tiling 参数。
extern "C" uint32_t vector_kernel_0_do(uint32_t blockDim, void *stream, uint8_t *N0s, uint8_t *lambdas, uint8_t *local_A, uint8_t *t, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount) {
    return vector_kernel_0<<<blockDim, nullptr, stream>>>(N0s, lambdas, local_A, t, totalLength, coreNum, tileLength, tileCount);
}
#endif
