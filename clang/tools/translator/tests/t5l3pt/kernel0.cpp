
#include "kernel_operator.h"
using namespace AscendC;

// Vector tiling/resource plan (translation-time selection)
// UB budget: 131072 bytes
// UB reserve: 16384 bytes
// selected BUFFER_NUM: 2
// tile alignment: 64 elements
// max tile elements: 2368
// bytes per tile estimate: 113760
// selection source: auto (default), ub-budget=default, ub-reserve=default
// allocation breakdown:
//   fixed: 32 bytes
//   inQueueu_kin (halo-queue): 19008 bytes
//   outQueueu_kout (output-queue): 18944 bytes
//   structured-extra-view-queues (halo-queue): 37888 bytes
//   structured-scratch (calc-buffer): 37888 bytes

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_ALIGN_ELEMENTS = 64;
constexpr uint32_t MAX_TILE_ELEMENTS = 2368;
constexpr uint32_t DACPP_VECTOR_ALIGN_BYTES = 256;

//  2. 算子类定义 
class Kernel_vector_kernel_0 {
public:
    __aicore__ inline Kernel_vector_kernel_0() {}
    __aicore__ inline void Init(GM_ADDR u_kin, GM_ADDR u_kout, GM_ADDR r, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount) {
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
        u_kinValidLength = tileLength;
        u_kinGmStart = static_cast<int32_t>(blockStart) + static_cast<int32_t>(0);
        if (u_kinGmStart < 0) { u_kinGmStart = 0; }
        u_kinGmLength = (blockLength + static_cast<uint32_t>(2) + static_cast<uint32_t>((256 + sizeof(float) - 1) / sizeof(float)) - 1) / static_cast<uint32_t>((256 + sizeof(float) - 1) / sizeof(float)) * static_cast<uint32_t>((256 + sizeof(float) - 1) / sizeof(float));
        u_kinGm.SetGlobalBuffer((__gm__ float *)u_kin + u_kinGmStart, u_kinGmLength);
        pipe.InitBuffer(inQueueu_kinView0, BUFFER_NUM, u_kinValidLength * sizeof(float));
        pipe.InitBuffer(inQueueu_kinView1, BUFFER_NUM, u_kinValidLength * sizeof(float));
        pipe.InitBuffer(inQueueu_kinView2, BUFFER_NUM, u_kinValidLength * sizeof(float));
        rScalarGm.SetGlobalBuffer((__gm__ float *)r, 1);
        pipe.InitBuffer(scalar0Buffer, sizeof(float));
        {
            AscendC::LocalTensor<float> scalar0Local = scalar0Buffer.Get<float>();
            AscendC::DataCopy(scalar0Local, rScalarGm[0], 1);
        }
        u_koutGm.SetGlobalBuffer((__gm__ float *)u_kout + blockStart, blockLength);
        pipe.InitBuffer(outQueueu_kout, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(calcBuffer2, tileLength * sizeof(float));

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
        AscendC::LocalTensor<float> view0Local = inQueueu_kinView0.AllocTensor<float>();
        int32_t u_kinView0BaseOffset = static_cast<int32_t>(tileOffset) + static_cast<int32_t>(0);
        AscendC::DataCopy(view0Local, u_kinGm[u_kinView0BaseOffset], currentTileLength);
        inQueueu_kinView0.EnQue(view0Local);
        AscendC::LocalTensor<float> view1Local = inQueueu_kinView1.AllocTensor<float>();
        int32_t u_kinView1BaseOffset = static_cast<int32_t>(tileOffset) + static_cast<int32_t>(1);
        AscendC::DataCopy(view1Local, u_kinGm[u_kinView1BaseOffset], currentTileLength);
        inQueueu_kinView1.EnQue(view1Local);
        AscendC::LocalTensor<float> view2Local = inQueueu_kinView2.AllocTensor<float>();
        int32_t u_kinView2BaseOffset = static_cast<int32_t>(tileOffset) + static_cast<int32_t>(2);
        AscendC::DataCopy(view2Local, u_kinGm[u_kinView2BaseOffset], currentTileLength);
        inQueueu_kinView2.EnQue(view2Local);

    }
    __aicore__ inline void Compute(uint32_t progress, uint32_t tileOffset,
                                   uint32_t currentTileLength) {
        AscendC::LocalTensor<float> view0Local = inQueueu_kinView0.DeQue<float>();
        AscendC::LocalTensor<float> view1Local = inQueueu_kinView1.DeQue<float>();
        AscendC::LocalTensor<float> view2Local = inQueueu_kinView2.DeQue<float>();
        float scalar0Value = static_cast<float>(rScalarGm.GetValue(0));
        AscendC::LocalTensor<float> dstLocal = outQueueu_kout.AllocTensor<float>();
        AscendC::LocalTensor<float> calcBuffer2Local = calcBuffer2.Get<float>();
        (void)calcBuffer2Local;
        const auto scalarExprTmp3 = (scalar0Value);
        const auto scalarExprTmp4 = ((1)-((2)*(scalar0Value)));
        const auto scalarExprTmp5 = (scalar0Value);
        AscendC::Adds(dstLocal, view0Local, static_cast<decltype(+dstLocal.GetValue(0))>(0), currentTileLength);
        AscendC::Muls(dstLocal, dstLocal, static_cast<decltype(+dstLocal.GetValue(0))>((scalarExprTmp3)), currentTileLength);
        AscendC::Adds(calcBuffer2Local, view1Local, static_cast<decltype(+calcBuffer2Local.GetValue(0))>(0), currentTileLength);
        AscendC::Muls(calcBuffer2Local, calcBuffer2Local, static_cast<decltype(+calcBuffer2Local.GetValue(0))>((scalarExprTmp4)), currentTileLength);
        AscendC::Add(dstLocal, dstLocal, calcBuffer2Local, currentTileLength);
        AscendC::Adds(calcBuffer2Local, view2Local, static_cast<decltype(+calcBuffer2Local.GetValue(0))>(0), currentTileLength);
        AscendC::Muls(calcBuffer2Local, calcBuffer2Local, static_cast<decltype(+calcBuffer2Local.GetValue(0))>((scalarExprTmp5)), currentTileLength);
        AscendC::Add(dstLocal, dstLocal, calcBuffer2Local, currentTileLength);
        outQueueu_kout.EnQue<float>(dstLocal);
        inQueueu_kinView0.FreeTensor(view0Local);
        inQueueu_kinView1.FreeTensor(view1Local);
        inQueueu_kinView2.FreeTensor(view2Local);

    }
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t tileOffset,
                                   uint32_t currentTileLength) {
        AscendC::LocalTensor<float> dstLocal = outQueueu_kout.DeQue<float>();
        AscendC::DataCopy(u_koutGm[tileOffset], dstLocal, currentTileLength);
        outQueueu_kout.FreeTensor(dstLocal);

    }

private:
    TPipe pipe;
    AscendC::GlobalTensor<float> u_kinGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueu_kinView0;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueu_kinView1;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueu_kinView2;
    uint32_t u_kinValidLength = 0;
    int32_t u_kinGmStart = 0;
    uint32_t u_kinGmLength = 0;
    AscendC::GlobalTensor<float> rScalarGm;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scalar0Buffer;
    AscendC::GlobalTensor<float> u_koutGm;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueu_kout;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuffer2;


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
extern "C" __global__ __aicore__ void vector_kernel_0(GM_ADDR u_kin, GM_ADDR u_kout, GM_ADDR r, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount) {
    Kernel_vector_kernel_0 op;
    op.Init(u_kin, u_kout, r, totalLength, coreNum, tileLength, tileCount);
    op.Process();
}

//  4. Host 端启动 Wrapper 函数 
#ifndef ASCENDC_CPU_DEBUG
// #include "acl/acl.h"
// 宿主机包装器接收统一的运行时 tiling 参数。
extern "C" uint32_t vector_kernel_0_do(uint32_t blockDim, void *stream, uint8_t *u_kin, uint8_t *u_kout, uint8_t *r, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount) {
    return vector_kernel_0<<<blockDim, nullptr, stream>>>(u_kin, u_kout, r, totalLength, coreNum, tileLength, tileCount);
}
#endif
