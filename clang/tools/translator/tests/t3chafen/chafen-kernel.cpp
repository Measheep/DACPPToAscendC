#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

class KernelChafen {
public:
    __aicore__ inline KernelChafen() {}

    __aicore__ inline void Init(GM_ADDR rho, GM_ADDR diff, uint32_t totalLength) {
        this->totalLength = totalLength;
        this->alignedLength = (totalLength + 8 - 1) / 8 * 8;
        this->readAlignedLength = (totalLength + 1 + 8 - 1) / 8 * 8;

        rhoGm.SetGlobalBuffer((__gm__ float*)rho, this->readAlignedLength);
        diffGm.SetGlobalBuffer((__gm__ float*)diff, this->alignedLength);

        pipe.InitBuffer(inQueueCur, BUFFER_NUM, this->alignedLength * sizeof(float));
        pipe.InitBuffer(inQueueNext, BUFFER_NUM, this->alignedLength * sizeof(float));
        pipe.InitBuffer(outQueueDiff, BUFFER_NUM, this->alignedLength * sizeof(float));
    }

    __aicore__ inline void Process() {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn() {
        LocalTensor<float> curLocal = inQueueCur.AllocTensor<float>();
        LocalTensor<float> nextLocal = inQueueNext.AllocTensor<float>();
        DataCopy(curLocal, rhoGm, this->alignedLength);
        DataCopy(nextLocal, rhoGm[1], this->alignedLength);
        inQueueCur.EnQue(curLocal);
        inQueueNext.EnQue(nextLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> curLocal = inQueueCur.DeQue<float>();
        LocalTensor<float> nextLocal = inQueueNext.DeQue<float>();
        LocalTensor<float> diffLocal = outQueueDiff.AllocTensor<float>();

        Sub(diffLocal, nextLocal, curLocal, this->alignedLength);

        outQueueDiff.EnQue(diffLocal);
        inQueueCur.FreeTensor(curLocal);
        inQueueNext.FreeTensor(nextLocal);
    }

    __aicore__ inline void CopyOut() {
        LocalTensor<float> diffLocal = outQueueDiff.DeQue<float>();
        DataCopy(diffGm, diffLocal, this->alignedLength);
        outQueueDiff.FreeTensor(diffLocal);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueCur;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueNext;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueDiff;
    GlobalTensor<float> rhoGm;
    GlobalTensor<float> diffGm;
    uint32_t totalLength;
    uint32_t alignedLength;
    uint32_t readAlignedLength;
};

extern "C" __global__ __aicore__ void custom_kernel_0(GM_ADDR rho,
                                                       GM_ADDR diff,
                                                       uint32_t totalLength,
                                                       uint32_t coreNum) {
    KernelChafen op;
    op.Init(rho, diff, totalLength);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern void custom_kernel_0_do(uint32_t blockDim,
                               void *stream,
                               uint8_t *rho,
                               uint8_t *diff,
                               uint32_t totalLength,
                               uint32_t coreNum) {
    custom_kernel_0<<<blockDim, nullptr, stream>>>(rho, diff, totalLength, coreNum);
}
#endif
