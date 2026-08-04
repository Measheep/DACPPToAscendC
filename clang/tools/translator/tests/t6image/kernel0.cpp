
#include "kernel_operator.h"
using namespace AscendC;

// Vector tiling/resource plan (translation-time selection)
// UB budget: 131072 bytes
// UB reserve: 16384 bytes
// selected BUFFER_NUM: 2
// tile alignment: 64 elements
// max tile elements: 2368
// bytes per tile estimate: 113664
// selection source: auto (default), ub-budget=default, ub-reserve=default
// allocation breakdown:
//   inQueueimageR (input-queue): 18944 bytes
//   inQueueimageG (input-queue): 18944 bytes
//   inQueueimageB (input-queue): 18944 bytes
//   outQueueimage2R (output-queue): 18944 bytes
//   outQueueimage2G (output-queue): 18944 bytes
//   outQueueimage2B (output-queue): 18944 bytes

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_ALIGN_ELEMENTS = 64;
constexpr uint32_t MAX_TILE_ELEMENTS = 2368;
constexpr uint32_t DACPP_VECTOR_ALIGN_BYTES = 256;

//  2. 算子类定义 
class Kernel_vector_kernel_0 {
public:
    __aicore__ inline Kernel_vector_kernel_0() {}
    __aicore__ inline void Init(GM_ADDR imageR, GM_ADDR imageG, GM_ADDR imageB, GM_ADDR image2R, GM_ADDR image2G, GM_ADDR image2B, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount) {
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
        imageRGm.SetGlobalBuffer((__gm__ int32_t *)imageR + blockStart, blockLength);
        pipe.InitBuffer(inQueueimageR, BUFFER_NUM, tileLength * sizeof(int32_t));
        imageGGm.SetGlobalBuffer((__gm__ int32_t *)imageG + blockStart, blockLength);
        pipe.InitBuffer(inQueueimageG, BUFFER_NUM, tileLength * sizeof(int32_t));
        imageBGm.SetGlobalBuffer((__gm__ int32_t *)imageB + blockStart, blockLength);
        pipe.InitBuffer(inQueueimageB, BUFFER_NUM, tileLength * sizeof(int32_t));
        image2RGm.SetGlobalBuffer((__gm__ int32_t *)image2R + blockStart, blockLength);
        pipe.InitBuffer(outQueueimage2R, BUFFER_NUM, tileLength * sizeof(int32_t));
        image2GGm.SetGlobalBuffer((__gm__ int32_t *)image2G + blockStart, blockLength);
        pipe.InitBuffer(outQueueimage2G, BUFFER_NUM, tileLength * sizeof(int32_t));
        image2BGm.SetGlobalBuffer((__gm__ int32_t *)image2B + blockStart, blockLength);
        pipe.InitBuffer(outQueueimage2B, BUFFER_NUM, tileLength * sizeof(int32_t));

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
        AscendC::LocalTensor<int32_t> imageRLocal = inQueueimageR.AllocTensor<int32_t>();
        AscendC::DataCopy(imageRLocal, imageRGm[tileOffset], currentTileLength);
        inQueueimageR.EnQue(imageRLocal);
        AscendC::LocalTensor<int32_t> imageGLocal = inQueueimageG.AllocTensor<int32_t>();
        AscendC::DataCopy(imageGLocal, imageGGm[tileOffset], currentTileLength);
        inQueueimageG.EnQue(imageGLocal);
        AscendC::LocalTensor<int32_t> imageBLocal = inQueueimageB.AllocTensor<int32_t>();
        AscendC::DataCopy(imageBLocal, imageBGm[tileOffset], currentTileLength);
        inQueueimageB.EnQue(imageBLocal);

    }
    __aicore__ inline void Compute(uint32_t progress, uint32_t tileOffset,
                                   uint32_t currentTileLength) {
        AscendC::LocalTensor<int32_t> imageRLocal = inQueueimageR.DeQue<int32_t>();
        AscendC::LocalTensor<int32_t> imageGLocal = inQueueimageG.DeQue<int32_t>();
        AscendC::LocalTensor<int32_t> imageBLocal = inQueueimageB.DeQue<int32_t>();
        AscendC::LocalTensor<int32_t> image2RLocal = outQueueimage2R.AllocTensor<int32_t>();
        AscendC::LocalTensor<int32_t> image2GLocal = outQueueimage2G.AllocTensor<int32_t>();
        AscendC::LocalTensor<int32_t> image2BLocal = outQueueimage2B.AllocTensor<int32_t>();
        // pointwise: direct pointwise expression tree preserved; diag(exp=0,mul=0,min=1,offsetNZ=0,offsetUnknown=0,windowBaseLoads=1,argTensors=1,argLoads=1)
        AscendC::Adds(image2RLocal, imageRLocal, static_cast<decltype(+image2RLocal.GetValue(0))>(0), currentTileLength);
        AscendC::Adds(image2RLocal, image2RLocal, static_cast<decltype(+image2RLocal.GetValue(0))>(50), currentTileLength);
        AscendC::Mins(image2RLocal, image2RLocal, static_cast<decltype(+image2RLocal.GetValue(0))>(255), currentTileLength);
        AscendC::Adds(image2GLocal, imageGLocal, static_cast<decltype(+image2GLocal.GetValue(0))>(0), currentTileLength);
        AscendC::Adds(image2BLocal, imageBLocal, static_cast<decltype(+image2BLocal.GetValue(0))>(0), currentTileLength);
        outQueueimage2R.EnQue<int32_t>(image2RLocal);
        outQueueimage2G.EnQue<int32_t>(image2GLocal);
        outQueueimage2B.EnQue<int32_t>(image2BLocal);
        inQueueimageR.FreeTensor(imageRLocal);
        inQueueimageG.FreeTensor(imageGLocal);
        inQueueimageB.FreeTensor(imageBLocal);

    }
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t tileOffset,
                                   uint32_t currentTileLength) {
        AscendC::LocalTensor<int32_t> image2RLocal = outQueueimage2R.DeQue<int32_t>();
        AscendC::DataCopy(image2RGm[tileOffset], image2RLocal, currentTileLength);
        outQueueimage2R.FreeTensor(image2RLocal);
        AscendC::LocalTensor<int32_t> image2GLocal = outQueueimage2G.DeQue<int32_t>();
        AscendC::DataCopy(image2GGm[tileOffset], image2GLocal, currentTileLength);
        outQueueimage2G.FreeTensor(image2GLocal);
        AscendC::LocalTensor<int32_t> image2BLocal = outQueueimage2B.DeQue<int32_t>();
        AscendC::DataCopy(image2BGm[tileOffset], image2BLocal, currentTileLength);
        outQueueimage2B.FreeTensor(image2BLocal);

    }

private:
    TPipe pipe;
    AscendC::GlobalTensor<int32_t> imageRGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueimageR;
    AscendC::GlobalTensor<int32_t> imageGGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueimageG;
    AscendC::GlobalTensor<int32_t> imageBGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueimageB;
    AscendC::GlobalTensor<int32_t> image2RGm;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueimage2R;
    AscendC::GlobalTensor<int32_t> image2GGm;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueimage2G;
    AscendC::GlobalTensor<int32_t> image2BGm;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueimage2B;


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
extern "C" __global__ __aicore__ void vector_kernel_0(GM_ADDR imageR, GM_ADDR imageG, GM_ADDR imageB, GM_ADDR image2R, GM_ADDR image2G, GM_ADDR image2B, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount) {
    Kernel_vector_kernel_0 op;
    op.Init(imageR, imageG, imageB, image2R, image2G, image2B, totalLength, coreNum, tileLength, tileCount);
    op.Process();
}

//  4. Host 端启动 Wrapper 函数 
#ifndef ASCENDC_CPU_DEBUG
// #include "acl/acl.h"
// 宿主机包装器接收统一的运行时 tiling 参数。
extern "C" uint32_t vector_kernel_0_do(uint32_t blockDim, void *stream, uint8_t *imageR, uint8_t *imageG, uint8_t *imageB, uint8_t *image2R, uint8_t *image2G, uint8_t *image2B, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount) {
    return vector_kernel_0<<<blockDim, nullptr, stream>>>(imageR, imageG, imageB, image2R, image2G, image2B, totalLength, coreNum, tileLength, tileCount);
}
#endif
