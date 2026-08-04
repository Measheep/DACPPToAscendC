//  自动生成的 AscendC Vector Host 代码
#include "dacpp_ascendc_host_runtime.h"
static_assert(DACPP_ASCENDC_HOST_RUNTIME_VERSION == 19,
              "regenerate Host runtime");


#ifndef ASCENDC_CPU_DEBUG
extern "C" uint32_t vector_kernel_0_do(uint32_t blockDim, void *stream, uint8_t *vecA, uint8_t *vecB, uint8_t *vecC, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount);
#else
extern "C" __global__ __aicore__ void vector_kernel_0(GM_ADDR vecA, GM_ADDR vecB, GM_ADDR vecC, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount);
#endif

int32_t main(int32_t argc, char *argv[]) {
    dacpp_host::Session app;
#ifndef ASCENDC_CPU_DEBUG
    dacpp_host::Device dev(app, 0);
    if (!dev.ok()) return app.finish();
    aclrtStream stream = dev.stream();
#endif
    std::vector<int> vecA(134348800, 1);
    std::vector<int> vecB(134348800, 1);
    std::vector<int> vecC(134348800, 0);
    {
    //  算子 vector_kernel_0 基础参数与数据定义 
        uint32_t blockDim0 = 8;
        dacpp_host::HostBuffer<int32_t> vecABuf(app, vecA.size(), static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>((256 + sizeof(int32_t) - 1) / sizeof(int32_t)));
            std::memcpy(vecABuf.data.data(), vecA.data(), std::min(vecABuf.logicalElems, vecA.size()) * sizeof(int32_t));
        dacpp_host::HostBuffer<int32_t> vecBBuf(app, vecB.size(), static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>((256 + sizeof(int32_t) - 1) / sizeof(int32_t)));
            std::memcpy(vecBBuf.data.data(), vecB.data(), std::min(vecBBuf.logicalElems, vecB.size()) * sizeof(int32_t));
        dacpp_host::ensureSize(vecC, static_cast<size_t>(134348800));
        dacpp_host::HostBuffer<int32_t> vecCBuf(app, vecC.size(), static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>((256 + sizeof(int32_t) - 1) / sizeof(int32_t)));
        size_t kernelSize0 = static_cast<size_t>(0);
        const size_t coreCount0 = blockDim0 == 0 ? static_cast<size_t>(1) : static_cast<size_t>(blockDim0);
        kernelSize0 = std::max(kernelSize0, vecABuf.kernelElems);
        kernelSize0 = std::max(kernelSize0, vecBBuf.kernelElems);
        kernelSize0 = std::max(kernelSize0, vecCBuf.kernelElems);
        uint32_t kernelLength0 = 0;
        const dacpp_host::Tile vectorRuntimeTiling0 = dacpp_host::makeTile(app, kernelSize0, coreCount0, 64, 4736);
        kernelSize0 = static_cast<size_t>(vectorRuntimeTiling0.len);
        kernelLength0 = vectorRuntimeTiling0.len;
        uint32_t tileLength0 = vectorRuntimeTiling0.size;
        uint32_t tileCount0 = vectorRuntimeTiling0.count;
        vecABuf.kernelElems = std::max(vecABuf.kernelElems, kernelSize0);
        { size_t withLeftHalo = 0, storage = 0, physical = 0; dacpp_host::checkedAdd(app, vecABuf.kernelElems, vecABuf.leftHaloElems, withLeftHalo); dacpp_host::checkedAdd(app, withLeftHalo, vecABuf.rightHaloElems, storage); dacpp_host::checkedAlign(app, storage, vecABuf.alignElems, physical); if (app.ok() && vecABuf.data.size() < physical) vecABuf.data.resize(physical); }
        vecBBuf.kernelElems = std::max(vecBBuf.kernelElems, kernelSize0);
        { size_t withLeftHalo = 0, storage = 0, physical = 0; dacpp_host::checkedAdd(app, vecBBuf.kernelElems, vecBBuf.leftHaloElems, withLeftHalo); dacpp_host::checkedAdd(app, withLeftHalo, vecBBuf.rightHaloElems, storage); dacpp_host::checkedAlign(app, storage, vecBBuf.alignElems, physical); if (app.ok() && vecBBuf.data.size() < physical) vecBBuf.data.resize(physical); }
        vecCBuf.kernelElems = std::max(vecCBuf.kernelElems, kernelSize0);
        { size_t withLeftHalo = 0, storage = 0, physical = 0; dacpp_host::checkedAdd(app, vecCBuf.kernelElems, vecCBuf.leftHaloElems, withLeftHalo); dacpp_host::checkedAdd(app, withLeftHalo, vecCBuf.rightHaloElems, storage); dacpp_host::checkedAlign(app, storage, vecCBuf.alignElems, physical); if (app.ok() && vecCBuf.data.size() < physical) vecCBuf.data.resize(physical); }
    //  算子 vector_kernel_0 NPU 执行流 
    #ifndef ASCENDC_CPU_DEBUG
        dev.allocate(vecABuf);
        dev.allocate(vecBBuf);
        dev.allocate(vecCBuf);
    #else
        uint8_t *cpuGm_vecA = (uint8_t *)AscendC::GmAlloc(vecABuf.bytes());
        uint8_t *cpuGm_vecB = (uint8_t *)AscendC::GmAlloc(vecBBuf.bytes());
        uint8_t *cpuGm_vecC = (uint8_t *)AscendC::GmAlloc(vecCBuf.bytes());
    #endif
    #ifndef ASCENDC_CPU_DEBUG
        dev.copyIn(vecABuf);
        dev.copyIn(vecBBuf);
    #else
        memcpy(cpuGm_vecA, vecABuf.data.data(), vecABuf.bytes());
        memcpy(cpuGm_vecB, vecBBuf.data.data(), vecBBuf.bytes());
    #endif
    #ifndef ASCENDC_CPU_DEBUG
        dev.launch(vector_kernel_0_do(blockDim0, stream, vecABuf.device, vecBBuf.device, vecCBuf.device, kernelLength0, blockDim0, tileLength0, tileCount0));
        dev.sync();
    #else
        ICPU_RUN_KF(vector_kernel_0, blockDim0, cpuGm_vecA, cpuGm_vecB, cpuGm_vecC, kernelLength0, blockDim0, tileLength0, tileCount0);
    #endif
    #ifndef ASCENDC_CPU_DEBUG
        dev.copyOut(vecCBuf);
        const size_t vecCOffset = static_cast<size_t>(0);
        std::memcpy(vecC.data() + vecCOffset, vecCBuf.data.data(), vecCBuf.logicalElems * sizeof(int32_t));
    #else
        memcpy(vecCBuf.data.data(), cpuGm_vecC, vecCBuf.bytes());
        const size_t vecCOffset = static_cast<size_t>(0);
        std::memcpy(vecC.data() + vecCOffset, vecCBuf.data.data(), vecCBuf.logicalElems * sizeof(int32_t));
    #endif
    #ifndef ASCENDC_CPU_DEBUG
        dev.release(vecABuf);
        dev.release(vecBBuf);
        dev.release(vecCBuf);
    #else
        AscendC::GmFree((void *)cpuGm_vecA);
        AscendC::GmFree((void *)cpuGm_vecB);
        AscendC::GmFree((void *)cpuGm_vecC);
    #endif
    }
    {
        std::cout << "{";
        for (long long dacppPrintI0_0 = 0; dacppPrintI0_0 < static_cast<long long>(vecC.size()); ++dacppPrintI0_0) {
            if (dacppPrintI0_0 != 0) { std::cout << ", "; }
            std::cout << (vecC.data())[static_cast<long long>(0) + dacppPrintI0_0 * static_cast<long long>(1)];
        }
        std::cout << "}";
        std::cout << std::endl;
    }
#ifndef ASCENDC_CPU_DEBUG
    dev.close();
#endif
    return app.finish();
}

