#include "kernel_operator.h"
using namespace AscendC;

// 1. 编译期常量定义
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t TILE_NUM = 1;

// 2. 算子类定义
class KernelPde3 {
public:
    __aicore__ inline KernelPde3() {}
    
    __aicore__ inline void Init(GM_ADDR u_kin, GM_ADDR u_kout, GM_ADDR r, uint32_t totalLength, uint32_t coreNum) {
        // 运行时动态 Tiling 参数计算（完全按照你的范式）
        this->totalLength = totalLength;
        this->useCoreNum = coreNum == 0 ? 1 : coreNum;
        this->blockLength = (totalLength + this->useCoreNum - 1) / this->useCoreNum;
        this->blockStart = this->blockLength * AscendC::GetBlockIdx();
        if (this->blockStart > this->totalLength) {
            this->blockStart = this->totalLength;
        }
        this->blockEnd = this->blockStart + this->blockLength;
        if (this->blockEnd > this->totalLength) {
            this->blockEnd = this->totalLength;
        }
        this->blockLength = this->blockEnd - this->blockStart;
        this->tileLength = this->blockLength / TILE_NUM / BUFFER_NUM;
        if (this->tileLength == 0) {
            this->tileLength = this->blockLength == 0 ? 1 : this->blockLength;
        }

        // 滑窗大小为3，所以输入总长为 N + 2
        uint32_t kinInputLength = totalLength + static_cast<uint32_t>(2);
        uint32_t kinRemaining = kinInputLength - blockStart;

        // 全局内存映射
        u_kinGm.SetGlobalBuffer((__gm__ float *)u_kin + blockStart, kinRemaining);
        u_koutGm.SetGlobalBuffer((__gm__ float *)u_kout + blockStart, blockLength);
        rGm.SetGlobalBuffer((__gm__ float *)r, 1);

        // 为每个偏移窗口分配独立的输入流水线 Queue
        pipe.InitBuffer(inQueueX0, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(inQueueX1, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, tileLength * sizeof(float));
        // 标量也需要一个独立的 Queue 搬运进 UB
        pipe.InitBuffer(inQueueR, BUFFER_NUM, 8 * sizeof(float)); // 预留足够空间避免越界

        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(calcBuffer, tileLength * sizeof(float));
    }

    __aicore__ inline void Process() {
        constexpr int32_t loopCount = TILE_NUM * BUFFER_NUM;
        for (int32_t progress = 0; progress < loopCount; progress++) {
            CopyIn(progress);
            Compute(progress);
            CopyOut(progress);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        AscendC::LocalTensor<float> x0Local = inQueueX0.AllocTensor<float>();
        AscendC::LocalTensor<float> x1Local = inQueueX1.AllocTensor<float>();
        AscendC::LocalTensor<float> x2Local = inQueueX2.AllocTensor<float>();
        AscendC::LocalTensor<float> rLocal = inQueueR.AllocTensor<float>();

        uint32_t baseOffset = progress * tileLength;
        
        // 绝妙的 MTE2 非对齐 GM 读取：直接从 GM 的不同偏移量加载！
        AscendC::DataCopy(x0Local, u_kinGm[baseOffset], tileLength);
        AscendC::DataCopy(x1Local, u_kinGm[baseOffset + 1], tileLength);
        AscendC::DataCopy(x2Local, u_kinGm[baseOffset + 2], tileLength);
        
        // 读取标量数据 r (这里取长度 1，因为实际只用到 r[0])
        AscendC::DataCopy(rLocal, rGm, 1);

        inQueueX0.EnQue(x0Local);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
        inQueueR.EnQue(rLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        AscendC::LocalTensor<float> x0Local = inQueueX0.DeQue<float>();
        AscendC::LocalTensor<float> x1Local = inQueueX1.DeQue<float>();
        AscendC::LocalTensor<float> x2Local = inQueueX2.DeQue<float>();
        AscendC::LocalTensor<float> rLocal = inQueueR.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<float> tmpLocal = calcBuffer.Get<float>();

        // 提取标量值并计算中间系数
        float r_val = rLocal.GetValue(0);
        float coef_mid = 1.0f - 2.0f * r_val;

        // 此时由于是从不同 Queue 中分别取出的 LocalTensor，起始索引都是 0
        // 所以在这里执行 Vector 指令是完美 32Bytes 对齐的，不会引发异常！
        
        // y = r * u_kin[0]
        AscendC::Muls(yLocal, x0Local, r_val, tileLength);
        
        // tmp = (1 - 2r) * u_kin[1]
        AscendC::Muls(tmpLocal, x1Local, coef_mid, tileLength);
        // y = y + tmp
        AscendC::Add(yLocal, yLocal, tmpLocal, tileLength);
        
        // tmp = r * u_kin[2]
        AscendC::Muls(tmpLocal, x2Local, r_val, tileLength);
        // y = y + tmp
        AscendC::Add(yLocal, yLocal, tmpLocal, tileLength);

        outQueueY.EnQue<float>(yLocal);

        inQueueX0.FreeTensor(x0Local);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
        inQueueR.FreeTensor(rLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(u_koutGm[progress * tileLength], yLocal, tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    AscendC::GlobalTensor<float> u_kinGm;
    AscendC::GlobalTensor<float> u_koutGm;
    AscendC::GlobalTensor<float> rGm;

    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX0;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX1;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX2;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueR;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuffer;

    uint32_t totalLength;
    uint32_t useCoreNum;
    uint32_t blockLength;
    uint32_t blockStart;
    uint32_t blockEnd;
    uint32_t tileLength;
};

// 3. 对外暴露的 __aicore__ 函数
extern "C" __global__ __aicore__ void custom_kernel_pde3(GM_ADDR u_kin, GM_ADDR u_kout, GM_ADDR r, uint32_t totalLength, uint32_t coreNum) {
    KernelPde3 op;
    op.Init(u_kin, u_kout, r, totalLength, coreNum);
    op.Process();
}

// 4. Host 端启动 Wrapper 函数
#ifndef __CCE_KT_TEST__
void custom_kernel_pde3_do(uint32_t blockDim, void *stream, uint8_t *u_kin, uint8_t *u_kout, uint8_t *r, uint32_t totalLength, uint32_t coreNum) {
    custom_kernel_pde3<<<blockDim, nullptr, stream>>>(u_kin, u_kout, r, totalLength, coreNum);
}
#endif