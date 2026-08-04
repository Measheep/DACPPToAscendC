#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

class KernelLiuliang {
public:
    __aicore__ inline KernelLiuliang() {}
    __aicore__ inline void Init(GM_ADDR rho, GM_ADDR new_rho, uint32_t totalLength) {
        this->totalLength = totalLength;
        
        // 输出侧对齐长度 (32 Bytes 向上对齐)
        this->alignedLength = (totalLength + 8 - 1) / 8 * 8; 
        // 输入侧包含冗余滑窗的安全对齐长度
        this->readAlignedLength = (this->alignedLength + 1 + 8 - 1) / 8 * 8; 

        rhoGm.SetGlobalBuffer((__gm__ float*)rho, this->readAlignedLength);
        newRhoGm.SetGlobalBuffer((__gm__ float*)new_rho, this->alignedLength);

        // 【核心修复】：为 rho[0] 和 rho[1] 各自分配一个专属的队列，确立硬件级的同步屏障
        pipe.InitBuffer(inQueueRho, BUFFER_NUM, this->alignedLength * sizeof(float));
        pipe.InitBuffer(inQueueRhoNext, BUFFER_NUM, this->alignedLength * sizeof(float));
        
        pipe.InitBuffer(calcBufferQCur, this->alignedLength * sizeof(float));
        pipe.InitBuffer(calcBufferQNext, this->alignedLength * sizeof(float));
        pipe.InitBuffer(calcBufferTmp, this->alignedLength * sizeof(float));
        pipe.InitBuffer(outQueueNewRho, BUFFER_NUM, this->alignedLength * sizeof(float));
    }

    __aicore__ inline void Process() {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn() {
        // 1. 发起从 GM 拉取 rho[0] 的 MTE2 异步搬运，并压入队列 A
        LocalTensor<float> rhoLocal = inQueueRho.AllocTensor<float>();
        DataCopy(rhoLocal, rhoGm, this->alignedLength);
        inQueueRho.EnQue(rhoLocal);

        // 2. 发起从 GM 偏移 1 个位置拉取 rho[1] 的 MTE2 异步搬运，并压入队列 B
        LocalTensor<float> rhoNextLocal = inQueueRhoNext.AllocTensor<float>();
        DataCopy(rhoNextLocal, rhoGm[1], this->alignedLength);
        inQueueRhoNext.EnQue(rhoNextLocal);
    }

    __aicore__ inline void Compute() {
        // 【核心屏障】：DeQue 会触发底层的 WaitFlag！
        // VEC 向量计算引擎将在这里严格阻塞，直到 MTE2 引擎真的把两条数据都搬完才放行！
        LocalTensor<float> rhoLocal = inQueueRho.DeQue<float>();
        LocalTensor<float> rhoNext = inQueueRhoNext.DeQue<float>();
        
        LocalTensor<float> newRhoLocal = outQueueNewRho.AllocTensor<float>();

        LocalTensor<float> qCur = calcBufferQCur.Get<float>();
        LocalTensor<float> qNext = calcBufferQNext.Get<float>();
        LocalTensor<float> tmp = calcBufferTmp.Get<float>();

        // 1. 计算 q(rho_cur) = 30 * rho - 0.6 * rho^2
        Muls(qCur, rhoLocal, -0.6f, this->alignedLength);
        Adds(qCur, qCur, 30.0f, this->alignedLength);
        Mul(qCur, qCur, rhoLocal, this->alignedLength);

        // 2. 计算 q(rho_next) = 30 * rhoNext - 0.6 * rhoNext^2
        Muls(qNext, rhoNext, -0.6f, this->alignedLength);
        Adds(qNext, qNext, 30.0f, this->alignedLength);
        Mul(qNext, qNext, rhoNext, this->alignedLength);

        // 3. 计算差分: new_rho = rhoNext - 0.01 * (qNext - qCur)
        Sub(tmp, qNext, qCur, this->alignedLength);
        Muls(tmp, tmp, 0.01f, this->alignedLength);
        Sub(newRhoLocal, rhoNext, tmp, this->alignedLength);

        // 4. 边界夹逼: new_rho = max(0.0, new_rho)
        Muls(tmp, tmp, 0.0f, this->alignedLength);
        Max(newRhoLocal, newRhoLocal, tmp, this->alignedLength);

        outQueueNewRho.EnQue(newRhoLocal);
        
        inQueueRho.FreeTensor(rhoLocal);
        inQueueRhoNext.FreeTensor(rhoNext);
    }

    __aicore__ inline void CopyOut() {
        LocalTensor<float> newRhoLocal = outQueueNewRho.DeQue<float>();
        DataCopy(newRhoGm, newRhoLocal, this->alignedLength);
        outQueueNewRho.FreeTensor(newRhoLocal);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueRho;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueRhoNext; // 新增的同步队列
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueNewRho;
    TBuf<QuePosition::VECCALC> calcBufferQCur, calcBufferQNext, calcBufferTmp;
    GlobalTensor<float> rhoGm, newRhoGm;

    uint32_t totalLength;
    uint32_t alignedLength;
    uint32_t readAlignedLength;
};

extern "C" __global__ __aicore__ void custom_kernel_0(GM_ADDR rho, GM_ADDR new_rho, uint32_t totalLength, uint32_t coreNum) {
    KernelLiuliang op;
    op.Init(rho, new_rho, totalLength);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern void custom_kernel_0_do(uint32_t blockDim, void *stream, uint8_t *rho, uint8_t *new_rho, uint32_t totalLength, uint32_t coreNum) {
    custom_kernel_0<<<blockDim, nullptr, stream>>>(rho, new_rho, totalLength, coreNum);
}
#endif