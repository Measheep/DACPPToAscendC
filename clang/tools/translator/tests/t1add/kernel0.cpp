
#include "kernel_operator.h"
using namespace AscendC;

// Vector tiling/resource plan (translation-time selection)
// UB budget: 131072 bytes
// UB reserve: 16384 bytes
// selected BUFFER_NUM: 2
// tile alignment: 64 elements
// max tile elements: 4736
// bytes per tile estimate: 113664
// selection source: auto (default), ub-budget=default, ub-reserve=default
// allocation breakdown:
//   inQueuevecA (input-queue): 37888 bytes
//   inQueuevecB (input-queue): 37888 bytes
//   outQueuevecC (output-queue): 37888 bytes

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_ALIGN_ELEMENTS = 64;
constexpr uint32_t MAX_TILE_ELEMENTS = 4736;
constexpr uint32_t DACPP_VECTOR_ALIGN_BYTES = 256;

//  2. 算子类定义 
class Kernel_vector_kernel_0 {
public:
    __aicore__ inline Kernel_vector_kernel_0() {}
    __aicore__ inline void Init(GM_ADDR vecA, GM_ADDR vecB, GM_ADDR vecC, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount) {
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
        vecAGm.SetGlobalBuffer((__gm__ int32_t *)vecA + blockStart, blockLength);
        pipe.InitBuffer(inQueuevecA, BUFFER_NUM, tileLength * sizeof(int32_t));
        vecBGm.SetGlobalBuffer((__gm__ int32_t *)vecB + blockStart, blockLength);
        pipe.InitBuffer(inQueuevecB, BUFFER_NUM, tileLength * sizeof(int32_t));
        vecCGm.SetGlobalBuffer((__gm__ int32_t *)vecC + blockStart, blockLength);
        pipe.InitBuffer(outQueuevecC, BUFFER_NUM, tileLength * sizeof(int32_t));

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
        AscendC::LocalTensor<int32_t> vecALocal = inQueuevecA.AllocTensor<int32_t>();
        AscendC::DataCopy(vecALocal, vecAGm[tileOffset], currentTileLength);
        inQueuevecA.EnQue(vecALocal);
        AscendC::LocalTensor<int32_t> vecBLocal = inQueuevecB.AllocTensor<int32_t>();
        AscendC::DataCopy(vecBLocal, vecBGm[tileOffset], currentTileLength);
        inQueuevecB.EnQue(vecBLocal);

    }
    __aicore__ inline void Compute(uint32_t progress, uint32_t tileOffset,
                                   uint32_t currentTileLength) {
        AscendC::LocalTensor<int32_t> vecALocal = inQueuevecA.DeQue<int32_t>();
        AscendC::LocalTensor<int32_t> vecBLocal = inQueuevecB.DeQue<int32_t>();
        AscendC::LocalTensor<int32_t> vecCLocal = outQueuevecC.AllocTensor<int32_t>();
        // pointwise: aligned two-tensor elementwise graph; diag(exp=0,mul=0,min=0,offsetNZ=0,offsetUnknown=0,windowBaseLoads=2,argTensors=2,argLoads=2)
        AscendC::Add(vecCLocal, vecALocal, vecBLocal, currentTileLength);
        outQueuevecC.EnQue<int32_t>(vecCLocal);
        inQueuevecA.FreeTensor(vecALocal);
        inQueuevecB.FreeTensor(vecBLocal);

    }
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t tileOffset,
                                   uint32_t currentTileLength) {
        AscendC::LocalTensor<int32_t> vecCLocal = outQueuevecC.DeQue<int32_t>();
        AscendC::DataCopy(vecCGm[tileOffset], vecCLocal, currentTileLength);
        outQueuevecC.FreeTensor(vecCLocal);

    }

private:
    TPipe pipe;
    AscendC::GlobalTensor<int32_t> vecAGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueuevecA;
    AscendC::GlobalTensor<int32_t> vecBGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueuevecB;
    AscendC::GlobalTensor<int32_t> vecCGm;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueuevecC;


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
extern "C" __global__ __aicore__ void vector_kernel_0(GM_ADDR vecA, GM_ADDR vecB, GM_ADDR vecC, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount) {
    Kernel_vector_kernel_0 op;
    op.Init(vecA, vecB, vecC, totalLength, coreNum, tileLength, tileCount);
    op.Process();
}

//  4. Host 端启动 Wrapper 函数 
#ifndef ASCENDC_CPU_DEBUG
// #include "acl/acl.h"
// 宿主机包装器接收统一的运行时 tiling 参数。
extern "C" uint32_t vector_kernel_0_do(uint32_t blockDim, void *stream, uint8_t *vecA, uint8_t *vecB, uint8_t *vecC, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount) {
    return vector_kernel_0<<<blockDim, nullptr, stream>>>(vecA, vecB, vecC, totalLength, coreNum, tileLength, tileCount);
}
#endif
