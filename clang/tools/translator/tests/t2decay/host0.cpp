//  自动生成的 AscendC Vector Host 代码
#include "dacpp_ascendc_host_runtime.h"
static_assert(DACPP_ASCENDC_HOST_RUNTIME_VERSION == 19,
              "regenerate Host runtime");


#ifndef ASCENDC_CPU_DEBUG
extern "C" uint32_t vector_kernel_0_do(uint32_t blockDim, void *stream, uint8_t *N0s, uint8_t *lambdas, uint8_t *local_A, uint8_t *t, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount);
#else
extern "C" __global__ __aicore__ void vector_kernel_0(GM_ADDR N0s, GM_ADDR lambdas, GM_ADDR local_A, GM_ADDR t, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount);
#endif

const float dt = 0.1;
const float T = 40.0;
const size_t numIsotopes = 524288;

int32_t main(int32_t argc, char *argv[]) {
    dacpp_host::Session app;
#ifndef ASCENDC_CPU_DEBUG
    dacpp_host::Device dev(app, 0);
    if (!dev.ok()) return app.finish();
    aclrtStream stream = dev.stream();
#endif
    std::vector<float> lambdas(numIsotopes);
    std::vector<float> N0s(numIsotopes, 1000.0);
    for (size_t i = 0; i < numIsotopes; ++i) {
        lambdas[i] = 0.01f + 0.0001f * static_cast<float>(i % 1024);
    }
    size_t numIsotopes = lambdas.size();
    std::vector<float> A(T/dt*numIsotopes, 0.0);
    std::vector<float> t;
    t.push_back(static_cast<float>(0));
    std::vector<float> local_A(numIsotopes, 0.0);
    {
    //  算子 vector_kernel_0 基础参数与数据定义 
        uint32_t blockDim0 = 8;
        dacpp_host::HostBuffer<float> N0sBuf(app, N0s.size(), static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>((256 + sizeof(float) - 1) / sizeof(float)));
            std::memcpy(N0sBuf.data.data(), N0s.data(), std::min(N0sBuf.logicalElems, N0s.size()) * sizeof(float));
        dacpp_host::HostBuffer<float> lambdasBuf(app, lambdas.size(), static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>((256 + sizeof(float) - 1) / sizeof(float)));
            std::memcpy(lambdasBuf.data.data(), lambdas.data(), std::min(lambdasBuf.logicalElems, lambdas.size()) * sizeof(float));
        dacpp_host::ensureSize(local_A, local_A.size());
        dacpp_host::HostBuffer<float> local_ABuf(app, local_A.size(), static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>((256 + sizeof(float) - 1) / sizeof(float)));
        dacpp_host::HostBuffer<float> tBuf(app, static_cast<size_t>(1), static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>((256 + sizeof(float) - 1) / sizeof(float)));
        std::fill_n(tBuf.data.begin(), tBuf.logicalElems, static_cast<float>(t[0]));
        size_t kernelSize0 = static_cast<size_t>(0);
        const size_t coreCount0 = blockDim0 == 0 ? static_cast<size_t>(1) : static_cast<size_t>(blockDim0);
        kernelSize0 = std::max(kernelSize0, N0sBuf.kernelElems);
        kernelSize0 = std::max(kernelSize0, lambdasBuf.kernelElems);
        kernelSize0 = std::max(kernelSize0, local_ABuf.kernelElems);
        uint32_t kernelLength0 = 0;
        const dacpp_host::Tile vectorRuntimeTiling0 = dacpp_host::makeTile(app, kernelSize0, coreCount0, 64, 3584);
        kernelSize0 = static_cast<size_t>(vectorRuntimeTiling0.len);
        kernelLength0 = vectorRuntimeTiling0.len;
        uint32_t tileLength0 = vectorRuntimeTiling0.size;
        uint32_t tileCount0 = vectorRuntimeTiling0.count;
        N0sBuf.kernelElems = std::max(N0sBuf.kernelElems, kernelSize0);
        { size_t withLeftHalo = 0, storage = 0, physical = 0; dacpp_host::checkedAdd(app, N0sBuf.kernelElems, N0sBuf.leftHaloElems, withLeftHalo); dacpp_host::checkedAdd(app, withLeftHalo, N0sBuf.rightHaloElems, storage); dacpp_host::checkedAlign(app, storage, N0sBuf.alignElems, physical); if (app.ok() && N0sBuf.data.size() < physical) N0sBuf.data.resize(physical); }
        lambdasBuf.kernelElems = std::max(lambdasBuf.kernelElems, kernelSize0);
        { size_t withLeftHalo = 0, storage = 0, physical = 0; dacpp_host::checkedAdd(app, lambdasBuf.kernelElems, lambdasBuf.leftHaloElems, withLeftHalo); dacpp_host::checkedAdd(app, withLeftHalo, lambdasBuf.rightHaloElems, storage); dacpp_host::checkedAlign(app, storage, lambdasBuf.alignElems, physical); if (app.ok() && lambdasBuf.data.size() < physical) lambdasBuf.data.resize(physical); }
        local_ABuf.kernelElems = std::max(local_ABuf.kernelElems, kernelSize0);
        { size_t withLeftHalo = 0, storage = 0, physical = 0; dacpp_host::checkedAdd(app, local_ABuf.kernelElems, local_ABuf.leftHaloElems, withLeftHalo); dacpp_host::checkedAdd(app, withLeftHalo, local_ABuf.rightHaloElems, storage); dacpp_host::checkedAlign(app, storage, local_ABuf.alignElems, physical); if (app.ok() && local_ABuf.data.size() < physical) local_ABuf.data.resize(physical); }
    #ifndef ASCENDC_CPU_DEBUG
    //  算子 vector_kernel_0 NPU 执行流 
        dev.allocate(N0sBuf);
        dev.allocate(lambdasBuf);
        dev.allocate(local_ABuf);
        dev.allocate(tBuf);
        dev.copyIn(N0sBuf);
        dev.copyIn(lambdasBuf);
        dev.copyIn(tBuf);
        auto tLoopScalarCurrent = static_cast<float>(static_cast<float>(0));
        while (tLoopScalarCurrent <= static_cast<float>(T)) {
            if (!tBuf.data.empty()) {
                tBuf.data[0] = static_cast<float>(tLoopScalarCurrent);
            }
            dev.copyIn(tBuf);
            dev.launch(vector_kernel_0_do(blockDim0, stream, N0sBuf.device, lambdasBuf.device, local_ABuf.device, tBuf.device, kernelLength0, blockDim0, tileLength0, tileCount0));
            dev.sync();
            dev.copyOut(local_ABuf);
            {
                const long long copyLen100_0 = std::min(static_cast<long long>(static_cast<int>(numIsotopes)), static_cast<long long>(local_A.size()));
                for (long long copyIdx100_0 = 0; copyIdx100_0 < copyLen100_0; ++copyIdx100_0) {
                    (A.data())[static_cast<long long>(((10*tLoopScalarCurrent) * (static_cast<int>(numIsotopes))) + (copyIdx100_0))] = (local_ABuf.data.data())[static_cast<long long>(copyIdx100_0)];
                }
            }
            tLoopScalarCurrent += static_cast<float>(dt);
        }
        dev.copyOut(local_ABuf);
        const size_t local_AOffset = static_cast<size_t>(0);
        std::memcpy(local_A.data() + local_AOffset, local_ABuf.data.data(), local_ABuf.logicalElems * sizeof(float));
        dev.release(N0sBuf);
        dev.release(lambdasBuf);
        dev.release(local_ABuf);
        dev.release(tBuf);
    #else
    //  算子 vector_kernel_0 CPU 执行流 
        uint8_t *cpuGm_N0s = (uint8_t *)AscendC::GmAlloc(N0sBuf.bytes());
        uint8_t *cpuGm_lambdas = (uint8_t *)AscendC::GmAlloc(lambdasBuf.bytes());
        uint8_t *cpuGm_local_A = (uint8_t *)AscendC::GmAlloc(local_ABuf.bytes());
        uint8_t *cpuGm_t = (uint8_t *)AscendC::GmAlloc(tBuf.bytes());
        memcpy(cpuGm_N0s, N0sBuf.data.data(), N0sBuf.bytes());
        memcpy(cpuGm_lambdas, lambdasBuf.data.data(), lambdasBuf.bytes());
        memcpy(cpuGm_t, tBuf.data.data(), tBuf.bytes());
        ICPU_RUN_KF(vector_kernel_0, blockDim0, cpuGm_N0s, cpuGm_lambdas, cpuGm_local_A, cpuGm_t, kernelLength0, blockDim0, tileLength0, tileCount0);
        memcpy(local_ABuf.data.data(), cpuGm_local_A, local_ABuf.bytes());
        const size_t local_AOffset = static_cast<size_t>(0);
        std::memcpy(local_A.data() + local_AOffset, local_ABuf.data.data(), local_ABuf.logicalElems * sizeof(float));
        AscendC::GmFree((void *)cpuGm_N0s);
        AscendC::GmFree((void *)cpuGm_lambdas);
        AscendC::GmFree((void *)cpuGm_local_A);
        AscendC::GmFree((void *)cpuGm_t);
    #endif
    }
    {
        std::cout << "{";
        for (long long dacppPrintI0_0 = 0; dacppPrintI0_0 < static_cast<long long>(static_cast<int>(numIsotopes)); ++dacppPrintI0_0) {
            if (dacppPrintI0_0 != 0) { std::cout << ", "; }
            std::cout << (A.data())[static_cast<long long>(static_cast<int>(numIsotopes)) + dacppPrintI0_0 * static_cast<long long>(1)];
        }
        std::cout << "}";
        std::cout << std::endl;
    }
#ifndef ASCENDC_CPU_DEBUG
    dev.close();
#endif
    return app.finish();
}

