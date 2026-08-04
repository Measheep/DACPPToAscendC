//  自动生成的 AscendC Vector Host 代码
#include "dacpp_ascendc_host_runtime.h"
static_assert(DACPP_ASCENDC_HOST_RUNTIME_VERSION == 19,
              "regenerate Host runtime");


#ifndef ASCENDC_CPU_DEBUG
extern "C" uint32_t vector_kernel_0_do(uint32_t blockDim, void *stream, uint8_t *u_kin, uint8_t *u_kout, uint8_t *r, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount);
#else
extern "C" __global__ __aicore__ void vector_kernel_0(GM_ADDR u_kin, GM_ADDR u_kout, GM_ADDR r, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount);
#endif

int32_t main(int32_t argc, char *argv[]) {
    dacpp_host::Session app;
#ifndef ASCENDC_CPU_DEBUG
    dacpp_host::Device dev(app, 0);
    if (!dev.ok()) return app.finish();
    aclrtStream stream = dev.stream();
#endif
    constexpr const int input_size = 4096;
    std::vector<float> u_data(input_size, 0.0f);
    for (int i = 0; i < input_size; ++i) {
        u_data[i] = static_cast<float>(i);
    }
    std::vector<float> out_data(input_size - 2, 0.0f);
    std::vector<float> r_data({0.25f});
    {
    //  算子 vector_kernel_0 基础参数与数据定义 
        uint32_t blockDim0 = 8;
        dacpp_host::HostBuffer<float> u_kinBuf(app, u_data.size(), static_cast<size_t>(0), static_cast<size_t>(2), static_cast<size_t>((256 + sizeof(float) - 1) / sizeof(float)));
            std::memcpy(u_kinBuf.data.data(), u_data.data(), std::min(u_kinBuf.logicalElems, u_data.size()) * sizeof(float));
        dacpp_host::ensureSize(out_data, static_cast<size_t>(4094));
        dacpp_host::HostBuffer<float> u_koutBuf(app, out_data.size(), static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>((256 + sizeof(float) - 1) / sizeof(float)));
        dacpp_host::HostBuffer<float> rBuf(app, static_cast<size_t>(1), static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>((256 + sizeof(float) - 1) / sizeof(float)));
        std::fill_n(rBuf.data.begin(), rBuf.logicalElems, static_cast<float>(r_data[0]));
        size_t kernelSize0 = static_cast<size_t>(0);
        const size_t coreCount0 = blockDim0 == 0 ? static_cast<size_t>(1) : static_cast<size_t>(blockDim0);
        kernelSize0 = std::max(kernelSize0, u_kinBuf.kernelElems);
        kernelSize0 = std::max(kernelSize0, u_koutBuf.kernelElems);
        uint32_t kernelLength0 = 0;
        const dacpp_host::Tile vectorRuntimeTiling0 = dacpp_host::makeTile(app, kernelSize0, coreCount0, 64, 2368);
        kernelSize0 = static_cast<size_t>(vectorRuntimeTiling0.len);
        kernelLength0 = vectorRuntimeTiling0.len;
        uint32_t tileLength0 = vectorRuntimeTiling0.size;
        uint32_t tileCount0 = vectorRuntimeTiling0.count;
        u_kinBuf.kernelElems = std::max(u_kinBuf.kernelElems, kernelSize0);
        { size_t withLeftHalo = 0, storage = 0, physical = 0; dacpp_host::checkedAdd(app, u_kinBuf.kernelElems, u_kinBuf.leftHaloElems, withLeftHalo); dacpp_host::checkedAdd(app, withLeftHalo, u_kinBuf.rightHaloElems, storage); dacpp_host::checkedAlign(app, storage, u_kinBuf.alignElems, physical); if (app.ok() && u_kinBuf.data.size() < physical) u_kinBuf.data.resize(physical); }
        u_koutBuf.kernelElems = std::max(u_koutBuf.kernelElems, kernelSize0);
        { size_t withLeftHalo = 0, storage = 0, physical = 0; dacpp_host::checkedAdd(app, u_koutBuf.kernelElems, u_koutBuf.leftHaloElems, withLeftHalo); dacpp_host::checkedAdd(app, withLeftHalo, u_koutBuf.rightHaloElems, storage); dacpp_host::checkedAlign(app, storage, u_koutBuf.alignElems, physical); if (app.ok() && u_koutBuf.data.size() < physical) u_koutBuf.data.resize(physical); }
    //  算子 vector_kernel_0 NPU 执行流 
    #ifndef ASCENDC_CPU_DEBUG
        dev.allocate(u_kinBuf);
        dev.allocate(u_koutBuf);
        dev.allocate(rBuf);
    #else
        uint8_t *cpuGm_u_kin = (uint8_t *)AscendC::GmAlloc(u_kinBuf.bytes());
        uint8_t *cpuGm_u_kout = (uint8_t *)AscendC::GmAlloc(u_koutBuf.bytes());
        uint8_t *cpuGm_r = (uint8_t *)AscendC::GmAlloc(rBuf.bytes());
    #endif
    #ifndef ASCENDC_CPU_DEBUG
        dev.copyIn(u_kinBuf);
        dev.copyIn(rBuf);
    #else
        memcpy(cpuGm_u_kin, u_kinBuf.data.data(), u_kinBuf.bytes());
        memcpy(cpuGm_r, rBuf.data.data(), rBuf.bytes());
    #endif
    #ifndef ASCENDC_CPU_DEBUG
        dev.launch(vector_kernel_0_do(blockDim0, stream, u_kinBuf.device, u_koutBuf.device, rBuf.device, kernelLength0, blockDim0, tileLength0, tileCount0));
        dev.sync();
    #else
        ICPU_RUN_KF(vector_kernel_0, blockDim0, cpuGm_u_kin, cpuGm_u_kout, cpuGm_r, kernelLength0, blockDim0, tileLength0, tileCount0);
    #endif
    #ifndef ASCENDC_CPU_DEBUG
        dev.copyOut(u_koutBuf);
        const size_t u_koutOffset = static_cast<size_t>(0);
        std::memcpy(out_data.data() + u_koutOffset, u_koutBuf.data.data(), u_koutBuf.logicalElems * sizeof(float));
    #else
        memcpy(u_koutBuf.data.data(), cpuGm_u_kout, u_koutBuf.bytes());
        const size_t u_koutOffset = static_cast<size_t>(0);
        std::memcpy(out_data.data() + u_koutOffset, u_koutBuf.data.data(), u_koutBuf.logicalElems * sizeof(float));
    #endif
    #ifndef ASCENDC_CPU_DEBUG
        dev.release(u_kinBuf);
        dev.release(u_koutBuf);
        dev.release(rBuf);
    #else
        AscendC::GmFree((void *)cpuGm_u_kin);
        AscendC::GmFree((void *)cpuGm_u_kout);
        AscendC::GmFree((void *)cpuGm_r);
    #endif
    }
    {
        std::cout << "{";
        for (long long dacppPrintI0_0 = 0; dacppPrintI0_0 < static_cast<long long>(out_data.size()); ++dacppPrintI0_0) {
            if (dacppPrintI0_0 != 0) { std::cout << ", "; }
            std::cout << (out_data.data())[static_cast<long long>(0) + dacppPrintI0_0 * static_cast<long long>(1)];
        }
        std::cout << "}";
        std::cout << std::endl;
    }
#ifndef ASCENDC_CPU_DEBUG
    dev.close();
#endif
    return app.finish();
}

