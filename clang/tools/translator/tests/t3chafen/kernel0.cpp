
#include "kernel_operator.h"
using namespace AscendC;

// Vector tiling/resource plan (translation-time selection)
// UB budget: 131072 bytes
// UB reserve: 16384 bytes
// selected BUFFER_NUM: 2
// tile alignment: 64 elements
// max tile elements: 3520
// bytes per tile estimate: 112704
// selection source: auto (default), ub-budget=default, ub-reserve=default
// allocation breakdown:
//   inQueuerho (halo-queue): 28224 bytes
//   outQueuediff (output-queue): 28160 bytes
//   structured-extra-view-queues (halo-queue): 28160 bytes
//   structured-scratch (calc-buffer): 28160 bytes

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_ALIGN_ELEMENTS = 64;
constexpr uint32_t MAX_TILE_ELEMENTS = 3520;
constexpr uint32_t DACPP_VECTOR_ALIGN_BYTES = 256;

//  2. 算子类定义 
class Kernel_vector_kernel_0 {
public:
    __aicore__ inline Kernel_vector_kernel_0() {}
    __aicore__ inline void Init(GM_ADDR rho, GM_ADDR diff, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount) {
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
        rhoInputLength = (totalLength + static_cast<uint32_t>(1) + static_cast<uint32_t>((256 + sizeof(float) - 1) / sizeof(float)) - 1) / static_cast<uint32_t>((256 + sizeof(float) - 1) / sizeof(float)) * static_cast<uint32_t>((256 + sizeof(float) - 1) / sizeof(float));
        uint32_t rhoRemaining = rhoInputLength - blockStart;
        rhoGm.SetGlobalBuffer((__gm__ float *)rho + blockStart, rhoRemaining);
        pipe.InitBuffer(inQueuerho, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(inQueuerhoShift, BUFFER_NUM, tileLength * sizeof(float));
        diffGm.SetGlobalBuffer((__gm__ float *)diff + blockStart, blockLength);
        pipe.InitBuffer(outQueuediff, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(calcBuffer0, tileLength * sizeof(float));
        pipe.InitBuffer(calcBuffer1, tileLength * sizeof(float));

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
        AscendC::LocalTensor<float> srcLocal = inQueuerho.AllocTensor<float>();
        AscendC::LocalTensor<float> srcShiftLocal = inQueuerhoShift.AllocTensor<float>();
        uint32_t rhoBaseOffset = tileOffset;
        AscendC::DataCopy(srcLocal, rhoGm[rhoBaseOffset], currentTileLength);
        AscendC::DataCopy(srcShiftLocal, rhoGm[rhoBaseOffset + 1], currentTileLength);
        inQueuerho.EnQue(srcLocal);
        inQueuerhoShift.EnQue(srcShiftLocal);

    }
    __aicore__ inline void Compute(uint32_t progress, uint32_t tileOffset,
                                   uint32_t currentTileLength) {
        AscendC::LocalTensor<float> srcLocal = inQueuerho.DeQue<float>();
        AscendC::LocalTensor<float> srcShiftLocal = inQueuerhoShift.DeQue<float>();
        AscendC::LocalTensor<float> dstLocal = outQueuediff.AllocTensor<float>();
        AscendC::LocalTensor<float> mapCenterLocal = calcBuffer0.Get<float>();
        AscendC::LocalTensor<float> mapShiftLocal = calcBuffer1.Get<float>();
        AscendC::Adds(mapCenterLocal, srcLocal, static_cast<decltype(+mapCenterLocal.GetValue(0))>(0), currentTileLength);
        AscendC::Adds(mapShiftLocal, srcShiftLocal, static_cast<decltype(+mapShiftLocal.GetValue(0))>(0), currentTileLength);
        AscendC::Sub(dstLocal, mapShiftLocal, mapCenterLocal, currentTileLength);
        outQueuediff.EnQue<float>(dstLocal);
        inQueuerho.FreeTensor(srcLocal);
        inQueuerhoShift.FreeTensor(srcShiftLocal);

    }
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t tileOffset,
                                   uint32_t currentTileLength) {
        AscendC::LocalTensor<float> dstLocal = outQueuediff.DeQue<float>();
        AscendC::DataCopy(diffGm[tileOffset], dstLocal, currentTileLength);
        outQueuediff.FreeTensor(dstLocal);

    }

private:
    TPipe pipe;
    AscendC::GlobalTensor<float> rhoGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueuerho;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueuerhoShift;
    uint32_t rhoInputLength = 0;
    AscendC::GlobalTensor<float> diffGm;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueuediff;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuffer0;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuffer1;


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
extern "C" __global__ __aicore__ void vector_kernel_0(GM_ADDR rho, GM_ADDR diff, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount) {
    Kernel_vector_kernel_0 op;
    op.Init(rho, diff, totalLength, coreNum, tileLength, tileCount);
    op.Process();
}

//  4. Host 端启动 Wrapper 函数 
#ifndef ASCENDC_CPU_DEBUG
// #include "acl/acl.h"
// 宿主机包装器接收统一的运行时 tiling 参数。
extern "C" uint32_t vector_kernel_0_do(uint32_t blockDim, void *stream, uint8_t *rho, uint8_t *diff, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount) {
    return vector_kernel_0<<<blockDim, nullptr, stream>>>(rho, diff, totalLength, coreNum, tileLength, tileCount);
}
#endif
