#include "kernel_operator.h"
using namespace AscendC;

// 1. 编译期超参约束声明
constexpr int32_t IN_FEATURES = 64;
constexpr int32_t OUT_FEATURES = 64;

class KernelLinear {
public:
    __aicore__ inline KernelLinear() {}

    // 2. 算子初始化逻辑
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR weight, GM_ADDR bias, GM_ADDR output) {
        // [内存映射] 将外部传入的 Global Memory 映射绑定为可操作的 GlobalTensor
        inputGm.SetGlobalBuffer((__gm__ float*)input, IN_FEATURES);
        weightGm.SetGlobalBuffer((__gm__ float*)weight, IN_FEATURES * OUT_FEATURES);
        biasGm.SetGlobalBuffer((__gm__ float*)bias, OUT_FEATURES);
        outputGm.SetGlobalBuffer((__gm__ float*)output, OUT_FEATURES);

        // [管线分配] 注册内部使用的 TQue 与 TBuf，设置对应的内存容量参数
        pipe.InitBuffer(inQueueInput, 1, IN_FEATURES * sizeof(float)); 
        pipe.InitBuffer(inQueueWeight, 1, IN_FEATURES * OUT_FEATURES * sizeof(float)); 
        pipe.InitBuffer(inQueueBias, 1, OUT_FEATURES * sizeof(float)); 
        pipe.InitBuffer(outQueueOutput, 1, OUT_FEATURES * sizeof(float)); 
        
        // 分配 32Bytes 作为矢量计算 Vector单元 的中间结果存储区
        pipe.InitBuffer(calcBuf, IN_FEATURES * sizeof(float)); 
        // 【优化】新增：分配 32Bytes 作为硬件级 ReduceSum 的接收缓冲
        pipe.InitBuffer(calcBufReduce, IN_FEATURES * sizeof(float)); 
    }

    // 3. 计算引擎核心逻辑
    __aicore__ inline void Process() {
        // 分配局部张量，申请将所需的数据读入片上统一缓冲 (Unified Buffer)
        LocalTensor<float> inputLocal = inQueueInput.AllocTensor<float>();
        LocalTensor<float> weightLocal = inQueueWeight.AllocTensor<float>();
        LocalTensor<float> biasLocal = inQueueBias.AllocTensor<float>();
        LocalTensor<float> outputLocal = outQueueOutput.AllocTensor<float>();
        
        LocalTensor<float> tmpLocal = calcBuf.Get<float>();
        LocalTensor<float> reduceLocal = calcBufReduce.Get<float>(); // 硬件求和接收张量

        // 【数据载入期】(MTE2 指令)
        DataCopy(inputLocal, inputGm, IN_FEATURES);
        DataCopy(weightLocal, weightGm, IN_FEATURES * OUT_FEATURES);
        DataCopy(biasLocal, biasGm, OUT_FEATURES);

        // 【核心计算期】(Vector + Scalar 流水优化版)
        for (int i = 0; i < OUT_FEATURES; ++i) {
            // 切片获取第 i 行权重 [长度 64]
            LocalTensor<float> weightRow = weightLocal[i * IN_FEATURES];

            // 1. 矢量乘法：tmp_buf = input * weight_row
            Mul(tmpLocal, inputLocal, weightRow, IN_FEATURES, 1, {1, 1, 1, 8, 8, 8});

            // 2. 硬件级降维累加 (WholeReduceSum)
            // 【深度优化】：使用硬件规约指令彻底消灭标量 for 循环！
            // 该指令由 V-Core (矢量核心) 闭环执行，将 64 个元素求和并存在 reduceLocal[0] 中。
            // 完美避开了 S-Core 和 V-Core 之间昂贵的同步屏障 (Sync Barrier)。
            WholeReduceSum(reduceLocal, tmpLocal, IN_FEATURES, 1, 1, 1, 1);

            // 3. 极简标量同步：每行计算仅保留 1 次 Read 和 1 次 Write
            float sum = biasLocal.GetValue(i) + reduceLocal.GetValue(0); 
            outputLocal.SetValue(i, sum);
        }

        // 【数据回写期】(MTE3 指令)
        outQueueOutput.EnQue<float>(outputLocal);

        LocalTensor<float> outCopy = outQueueOutput.DeQue<float>();
        DataCopy(outputGm, outCopy, OUT_FEATURES); 

        // 释放句柄资源
        outQueueOutput.FreeTensor(outCopy);
        inQueueInput.FreeTensor(inputLocal);
        inQueueWeight.FreeTensor(weightLocal);
        inQueueBias.FreeTensor(biasLocal);
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> calcBuf;
    TBuf<TPosition::VECCALC> calcBufReduce; // 【优化】规约缓冲声明

    // GM 张量声明
    GlobalTensor<float> inputGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> biasGm;
    GlobalTensor<float> outputGm;

    // UB 队列管理
    TQue<QuePosition::VECIN, 1> inQueueInput;
    TQue<QuePosition::VECIN, 1> inQueueWeight;
    TQue<QuePosition::VECIN, 1> inQueueBias;
    TQue<QuePosition::VECOUT, 1> outQueueOutput;
};

// 4. 内核入口暴露
extern "C" __global__ __aicore__ void custom_kernel_0(GM_ADDR input, GM_ADDR weight, GM_ADDR bias, GM_ADDR output, uint32_t totalLength, uint32_t coreNum) {
    if (GetBlockIdx() != 0) return;
    
    KernelLinear op;
    op.Init(input, weight, bias, output);
    op.Process();
}

#ifndef __CCE_KT_TEST__
// 5. Host 侧调用的启动桩函数 (Stub)
void custom_kernel_0_do(uint32_t blockDim, void* stream, uint8_t* input, uint8_t* weight, uint8_t* bias, uint8_t* output, uint32_t totalLength, uint32_t coreNum) {
    custom_kernel_0<<<blockDim, nullptr, stream>>>(input, weight, bias, output, totalLength, coreNum);
}
#endif