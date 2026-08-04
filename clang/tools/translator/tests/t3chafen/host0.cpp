//  自动生成的 AscendC Vector Host 代码
#include "dacpp_ascendc_host_runtime.h"
static_assert(DACPP_ASCENDC_HOST_RUNTIME_VERSION == 19,
              "regenerate Host runtime");


#ifndef ASCENDC_CPU_DEBUG
extern "C" uint32_t vector_kernel_0_do(uint32_t blockDim, void *stream, uint8_t *rho, uint8_t *diff, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount);
#else
extern "C" __global__ __aicore__ void vector_kernel_0(GM_ADDR rho, GM_ADDR diff, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount);
#endif

const int WIDTH = 41943040;

void initializeInput(std::vector<float>& rho) {
    for (int i = 0; i <= WIDTH; ++i) {
        rho[i] = static_cast<float>(i % 65536);
    }
}

int32_t main(int32_t argc, char *argv[]) {
    dacpp_host::Session app;
#ifndef ASCENDC_CPU_DEBUG
    dacpp_host::Device dev(app, 0);
    if (!dev.ok()) return app.finish();
    aclrtStream stream = dev.stream();
#endif
    std::vector<float> rho(WIDTH + 1, 0.0f);
    std::vector<float> diff(WIDTH, 0.0f);
    initializeInput(rho);
    {
    //  算子 vector_kernel_0 基础参数与数据定义 
        uint32_t blockDim0 = 8;
        dacpp_host::HostBuffer<float> rhoBuf(app, rho.size(), static_cast<size_t>(0), static_cast<size_t>(1), static_cast<size_t>((256 + sizeof(float) - 1) / sizeof(float)));
            std::memcpy(rhoBuf.data.data(), rho.data(), std::min(rhoBuf.logicalElems, rho.size()) * sizeof(float));
        dacpp_host::ensureSize(diff, static_cast<size_t>(41943040));
        dacpp_host::HostBuffer<float> diffBuf(app, diff.size(), static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>((256 + sizeof(float) - 1) / sizeof(float)));
        size_t kernelSize0 = static_cast<size_t>(0);
        const size_t coreCount0 = blockDim0 == 0 ? static_cast<size_t>(1) : static_cast<size_t>(blockDim0);
        kernelSize0 = std::max(kernelSize0, rhoBuf.kernelElems);
        kernelSize0 = std::max(kernelSize0, diffBuf.kernelElems);
        uint32_t kernelLength0 = 0;
        const dacpp_host::Tile vectorRuntimeTiling0 = dacpp_host::makeTile(app, kernelSize0, coreCount0, 64, 3520);
        kernelSize0 = static_cast<size_t>(vectorRuntimeTiling0.len);
        kernelLength0 = vectorRuntimeTiling0.len;
        uint32_t tileLength0 = vectorRuntimeTiling0.size;
        uint32_t tileCount0 = vectorRuntimeTiling0.count;
        rhoBuf.kernelElems = std::max(rhoBuf.kernelElems, kernelSize0);
        { size_t withLeftHalo = 0, storage = 0, physical = 0; dacpp_host::checkedAdd(app, rhoBuf.kernelElems, rhoBuf.leftHaloElems, withLeftHalo); dacpp_host::checkedAdd(app, withLeftHalo, rhoBuf.rightHaloElems, storage); dacpp_host::checkedAlign(app, storage, rhoBuf.alignElems, physical); if (app.ok() && rhoBuf.data.size() < physical) rhoBuf.data.resize(physical); }
        diffBuf.kernelElems = std::max(diffBuf.kernelElems, kernelSize0);
        { size_t withLeftHalo = 0, storage = 0, physical = 0; dacpp_host::checkedAdd(app, diffBuf.kernelElems, diffBuf.leftHaloElems, withLeftHalo); dacpp_host::checkedAdd(app, withLeftHalo, diffBuf.rightHaloElems, storage); dacpp_host::checkedAlign(app, storage, diffBuf.alignElems, physical); if (app.ok() && diffBuf.data.size() < physical) diffBuf.data.resize(physical); }
    //  算子 vector_kernel_0 NPU 执行流 
    #ifndef ASCENDC_CPU_DEBUG
        dev.allocate(rhoBuf);
        dev.allocate(diffBuf);
    #else
        uint8_t *cpuGm_rho = (uint8_t *)AscendC::GmAlloc(rhoBuf.bytes());
        uint8_t *cpuGm_diff = (uint8_t *)AscendC::GmAlloc(diffBuf.bytes());
    #endif
    #ifndef ASCENDC_CPU_DEBUG
        dev.copyIn(rhoBuf);
    #else
        memcpy(cpuGm_rho, rhoBuf.data.data(), rhoBuf.bytes());
    #endif
    #ifndef ASCENDC_CPU_DEBUG
        dev.launch(vector_kernel_0_do(blockDim0, stream, rhoBuf.device, diffBuf.device, kernelLength0, blockDim0, tileLength0, tileCount0));
        dev.sync();
    #else
        ICPU_RUN_KF(vector_kernel_0, blockDim0, cpuGm_rho, cpuGm_diff, kernelLength0, blockDim0, tileLength0, tileCount0);
    #endif
    #ifndef ASCENDC_CPU_DEBUG
        dev.copyOut(diffBuf);
        const size_t diffOffset = static_cast<size_t>(0);
        std::memcpy(diff.data() + diffOffset, diffBuf.data.data(), diffBuf.logicalElems * sizeof(float));
    #else
        memcpy(diffBuf.data.data(), cpuGm_diff, diffBuf.bytes());
        const size_t diffOffset = static_cast<size_t>(0);
        std::memcpy(diff.data() + diffOffset, diffBuf.data.data(), diffBuf.logicalElems * sizeof(float));
    #endif
    #ifndef ASCENDC_CPU_DEBUG
        dev.release(rhoBuf);
        dev.release(diffBuf);
    #else
        AscendC::GmFree((void *)cpuGm_rho);
        AscendC::GmFree((void *)cpuGm_diff);
    #endif
    }
    {
        std::cout << "{";
        for (long long dacppPrintI0_0 = 0; dacppPrintI0_0 < static_cast<long long>(diff.size()); ++dacppPrintI0_0) {
            if (dacppPrintI0_0 != 0) { std::cout << ", "; }
            std::cout << (diff.data())[static_cast<long long>(0) + dacppPrintI0_0 * static_cast<long long>(1)];
        }
        std::cout << "}";
        std::cout << std::endl;
    }
#ifndef ASCENDC_CPU_DEBUG
    dev.close();
#endif
    return app.finish();
}

