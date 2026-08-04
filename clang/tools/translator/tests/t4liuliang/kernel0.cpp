
#include "kernel_operator.h"
using namespace AscendC;

// Vector tiling/resource plan (translation-time selection)
// UB budget: 131072 bytes
// UB reserve: 16384 bytes
// selected BUFFER_NUM: 2
// tile alignment: 64 elements
// max tile elements: 2816
// bytes per tile estimate: 112704
// selection source: auto (default), ub-budget=default, ub-reserve=default
// allocation breakdown:
//   inQueuerho (halo-queue): 22592 bytes
//   outQueuenew_rho (output-queue): 22528 bytes
//   structured-extra-view-queues (halo-queue): 22528 bytes
//   structured-scratch (calc-buffer): 45056 bytes

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_ALIGN_ELEMENTS = 64;
constexpr uint32_t MAX_TILE_ELEMENTS = 2816;
constexpr uint32_t DACPP_VECTOR_ALIGN_BYTES = 256;

//  2. 算子类定义 
class Kernel_vector_kernel_0 {
public:
    __aicore__ inline Kernel_vector_kernel_0() {}
    __aicore__ inline void Init(GM_ADDR rho, GM_ADDR new_rho, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount) {
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
        new_rhoGm.SetGlobalBuffer((__gm__ float *)new_rho + blockStart, blockLength);
        pipe.InitBuffer(outQueuenew_rho, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(calcBuffer0, tileLength * sizeof(float));
        pipe.InitBuffer(calcBuffer1, tileLength * sizeof(float));
        pipe.InitBuffer(calcBuffer2, tileLength * sizeof(float));
        pipe.InitBuffer(calcBuffer3, tileLength * sizeof(float));

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
        AscendC::LocalTensor<float> dstLocal = outQueuenew_rho.AllocTensor<float>();
        AscendC::LocalTensor<float> mapCenterLocal = calcBuffer0.Get<float>();
        AscendC::LocalTensor<float> mapShiftLocal = calcBuffer1.Get<float>();
        AscendC::LocalTensor<float> calcBuffer2Local = calcBuffer2.Get<float>();
        (void)calcBuffer2Local;
        AscendC::LocalTensor<float> calcBuffer3Local = calcBuffer3.Get<float>();
        (void)calcBuffer3Local;
        const auto scalarExprTmp0 = (30);
        const auto scalarExprTmp1 = (0.0199999996f);
        const auto scalarExprTmp2 = (1);
        AscendC::Adds(mapCenterLocal, srcLocal, static_cast<decltype(+mapCenterLocal.GetValue(0))>(0), currentTileLength);
        AscendC::Muls(mapCenterLocal, mapCenterLocal, static_cast<decltype(+mapCenterLocal.GetValue(0))>(scalarExprTmp0), currentTileLength);
        AscendC::Adds(calcBuffer2Local, srcLocal, static_cast<decltype(+calcBuffer2Local.GetValue(0))>(0), currentTileLength);
        AscendC::Muls(calcBuffer2Local, calcBuffer2Local, static_cast<decltype(+calcBuffer2Local.GetValue(0))>(scalarExprTmp1), currentTileLength);
        AscendC::Muls(calcBuffer2Local, calcBuffer2Local, static_cast<decltype(+calcBuffer2Local.GetValue(0))>(-1), currentTileLength);
        AscendC::Adds(calcBuffer2Local, calcBuffer2Local, static_cast<decltype(+calcBuffer2Local.GetValue(0))>((scalarExprTmp2)), currentTileLength);
        AscendC::Mul(mapCenterLocal, mapCenterLocal, calcBuffer2Local, currentTileLength);
        const auto scalarExprTmp3 = (30);
        const auto scalarExprTmp4 = (0.0199999996f);
        const auto scalarExprTmp5 = (1);
        AscendC::Adds(mapShiftLocal, srcShiftLocal, static_cast<decltype(+mapShiftLocal.GetValue(0))>(0), currentTileLength);
        AscendC::Muls(mapShiftLocal, mapShiftLocal, static_cast<decltype(+mapShiftLocal.GetValue(0))>(scalarExprTmp3), currentTileLength);
        AscendC::Adds(calcBuffer2Local, srcShiftLocal, static_cast<decltype(+calcBuffer2Local.GetValue(0))>(0), currentTileLength);
        AscendC::Muls(calcBuffer2Local, calcBuffer2Local, static_cast<decltype(+calcBuffer2Local.GetValue(0))>(scalarExprTmp4), currentTileLength);
        AscendC::Muls(calcBuffer2Local, calcBuffer2Local, static_cast<decltype(+calcBuffer2Local.GetValue(0))>(-1), currentTileLength);
        AscendC::Adds(calcBuffer2Local, calcBuffer2Local, static_cast<decltype(+calcBuffer2Local.GetValue(0))>((scalarExprTmp5)), currentTileLength);
        AscendC::Mul(mapShiftLocal, mapShiftLocal, calcBuffer2Local, currentTileLength);
        const auto scalarExprTmp6 = (0.00999999978f);
        AscendC::Adds(dstLocal, srcShiftLocal, static_cast<decltype(+dstLocal.GetValue(0))>(0), currentTileLength);
        AscendC::Adds(calcBuffer2Local, mapShiftLocal, static_cast<decltype(+calcBuffer2Local.GetValue(0))>(0), currentTileLength);
        AscendC::Adds(calcBuffer3Local, mapCenterLocal, static_cast<decltype(+calcBuffer3Local.GetValue(0))>(0), currentTileLength);
        AscendC::Sub(calcBuffer2Local, calcBuffer2Local, calcBuffer3Local, currentTileLength);
        AscendC::Muls(calcBuffer2Local, calcBuffer2Local, static_cast<decltype(+calcBuffer2Local.GetValue(0))>(scalarExprTmp6), currentTileLength);
        AscendC::Sub(dstLocal, dstLocal, calcBuffer2Local, currentTileLength);
        AscendC::Duplicate(calcBuffer2Local, static_cast<decltype(+calcBuffer2Local.GetValue(0))>(0), currentTileLength);
        AscendC::Max(dstLocal, dstLocal, calcBuffer2Local, currentTileLength);
        outQueuenew_rho.EnQue<float>(dstLocal);
        inQueuerho.FreeTensor(srcLocal);
        inQueuerhoShift.FreeTensor(srcShiftLocal);

    }
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t tileOffset,
                                   uint32_t currentTileLength) {
        AscendC::LocalTensor<float> dstLocal = outQueuenew_rho.DeQue<float>();
        AscendC::DataCopy(new_rhoGm[tileOffset], dstLocal, currentTileLength);
        outQueuenew_rho.FreeTensor(dstLocal);

    }

private:
    TPipe pipe;
    AscendC::GlobalTensor<float> rhoGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueuerho;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueuerhoShift;
    uint32_t rhoInputLength = 0;
    AscendC::GlobalTensor<float> new_rhoGm;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueuenew_rho;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuffer0;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuffer1;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuffer2;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuffer3;


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
extern "C" __global__ __aicore__ void vector_kernel_0(GM_ADDR rho, GM_ADDR new_rho, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount) {
    Kernel_vector_kernel_0 op;
    op.Init(rho, new_rho, totalLength, coreNum, tileLength, tileCount);
    op.Process();
}

//  4. Host 端启动 Wrapper 函数 
#ifndef ASCENDC_CPU_DEBUG
// #include "acl/acl.h"
// 宿主机包装器接收统一的运行时 tiling 参数。
extern "C" uint32_t vector_kernel_0_do(uint32_t blockDim, void *stream, uint8_t *rho, uint8_t *new_rho, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount) {
    return vector_kernel_0<<<blockDim, nullptr, stream>>>(rho, new_rho, totalLength, coreNum, tileLength, tileCount);
}
#endif
