//  自动生成的 AscendC Vector Host 代码
#include "dacpp_ascendc_host_runtime.h"
static_assert(DACPP_ASCENDC_HOST_RUNTIME_VERSION == 19,
              "regenerate Host runtime");


#ifndef ASCENDC_CPU_DEBUG
extern "C" uint32_t vector_kernel_0_do(uint32_t blockDim, void *stream, uint8_t *rho, uint8_t *new_rho, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount);
#else
extern "C" __global__ __aicore__ void vector_kernel_0(GM_ADDR rho, GM_ADDR new_rho, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount);
#endif

const int WIDTH = 128;
const float TIME_STEPS = 256;

void initializeDensity(std::vector<float>& rho) {
    for (int i = 0; i < WIDTH; ++i) {
        if (i < WIDTH / 4) {
            rho[i] = 40; // 高密度区
        } else if (i < 3 * WIDTH / 4) {
            rho[i] = 20; // 中密度区
        } else {
            rho[i] = 10; // 低密度区
        }
    }
}

int32_t main(int32_t argc, char *argv[]) {
    dacpp_host::Session app;
#ifndef ASCENDC_CPU_DEBUG
    dacpp_host::Device dev(app, 0);
    if (!dev.ok()) return app.finish();
    aclrtStream stream = dev.stream();
#endif
    std::vector<float> rho(WIDTH, 0.0);
    std::vector<float> new_rho(WIDTH, 0.0);
    initializeDensity(rho);
    for (int t = 0; t < TIME_STEPS; ++t) {
        {
        //  算子 vector_kernel_0 基础参数与数据定义 
            uint32_t blockDim0 = 8;
            dacpp_host::HostBuffer<float> rhoBuf(app, rho.size(), static_cast<size_t>(0), static_cast<size_t>(1), static_cast<size_t>((256 + sizeof(float) - 1) / sizeof(float)));
                std::memcpy(rhoBuf.data.data(), rho.data(), std::min(rhoBuf.logicalElems, rho.size()) * sizeof(float));
            dacpp_host::ensureSize(new_rho, static_cast<size_t>(126));
            dacpp_host::HostBuffer<float> new_rhoBuf(app, new_rho.size(), static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>((256 + sizeof(float) - 1) / sizeof(float)));
            size_t kernelSize0 = static_cast<size_t>(0);
            const size_t coreCount0 = blockDim0 == 0 ? static_cast<size_t>(1) : static_cast<size_t>(blockDim0);
            kernelSize0 = std::max(kernelSize0, rhoBuf.kernelElems);
            kernelSize0 = std::max(kernelSize0, new_rhoBuf.kernelElems);
            uint32_t kernelLength0 = 0;
            const dacpp_host::Tile vectorRuntimeTiling0 = dacpp_host::makeTile(app, kernelSize0, coreCount0, 64, 2816);
            kernelSize0 = static_cast<size_t>(vectorRuntimeTiling0.len);
            kernelLength0 = vectorRuntimeTiling0.len;
            uint32_t tileLength0 = vectorRuntimeTiling0.size;
            uint32_t tileCount0 = vectorRuntimeTiling0.count;
            rhoBuf.kernelElems = std::max(rhoBuf.kernelElems, kernelSize0);
            { size_t withLeftHalo = 0, storage = 0, physical = 0; dacpp_host::checkedAdd(app, rhoBuf.kernelElems, rhoBuf.leftHaloElems, withLeftHalo); dacpp_host::checkedAdd(app, withLeftHalo, rhoBuf.rightHaloElems, storage); dacpp_host::checkedAlign(app, storage, rhoBuf.alignElems, physical); if (app.ok() && rhoBuf.data.size() < physical) rhoBuf.data.resize(physical); }
            new_rhoBuf.kernelElems = std::max(new_rhoBuf.kernelElems, kernelSize0);
            { size_t withLeftHalo = 0, storage = 0, physical = 0; dacpp_host::checkedAdd(app, new_rhoBuf.kernelElems, new_rhoBuf.leftHaloElems, withLeftHalo); dacpp_host::checkedAdd(app, withLeftHalo, new_rhoBuf.rightHaloElems, storage); dacpp_host::checkedAlign(app, storage, new_rhoBuf.alignElems, physical); if (app.ok() && new_rhoBuf.data.size() < physical) new_rhoBuf.data.resize(physical); }
        //  算子 vector_kernel_0 NPU 执行流 
        #ifndef ASCENDC_CPU_DEBUG
            dev.allocate(rhoBuf);
            dev.allocate(new_rhoBuf);
        #else
            uint8_t *cpuGm_rho = (uint8_t *)AscendC::GmAlloc(rhoBuf.bytes());
            uint8_t *cpuGm_new_rho = (uint8_t *)AscendC::GmAlloc(new_rhoBuf.bytes());
        #endif
        #ifndef ASCENDC_CPU_DEBUG
            dev.copyIn(rhoBuf);
        #else
            memcpy(cpuGm_rho, rhoBuf.data.data(), rhoBuf.bytes());
        #endif
        #ifndef ASCENDC_CPU_DEBUG
            dev.launch(vector_kernel_0_do(blockDim0, stream, rhoBuf.device, new_rhoBuf.device, kernelLength0, blockDim0, tileLength0, tileCount0));
            dev.sync();
        #else
            ICPU_RUN_KF(vector_kernel_0, blockDim0, cpuGm_rho, cpuGm_new_rho, kernelLength0, blockDim0, tileLength0, tileCount0);
        #endif
        #ifndef ASCENDC_CPU_DEBUG
            dev.copyOut(new_rhoBuf);
            const size_t new_rhoOffset = static_cast<size_t>(1);
            std::memcpy(new_rho.data() + new_rhoOffset, new_rhoBuf.data.data(), new_rhoBuf.logicalElems * sizeof(float));
        #else
            memcpy(new_rhoBuf.data.data(), cpuGm_new_rho, new_rhoBuf.bytes());
            const size_t new_rhoOffset = static_cast<size_t>(0);
            std::memcpy(new_rho.data() + new_rhoOffset, new_rhoBuf.data.data(), new_rhoBuf.logicalElems * sizeof(float));
        #endif
        #ifndef ASCENDC_CPU_DEBUG
            dev.release(rhoBuf);
            dev.release(new_rhoBuf);
        #else
            AscendC::GmFree((void *)cpuGm_rho);
            AscendC::GmFree((void *)cpuGm_new_rho);
        #endif
        }
        for (int i = 1; i <= WIDTH-2; i++) {
            (rho.data())[static_cast<long long>(i)] = (new_rho.data())[static_cast<long long>((1) + (i-1))];
        }
        (rho.data())[static_cast<long long>(0)] = (new_rho.data())[static_cast<long long>(1)];
    }
    std::cout << (rho.data())[static_cast<long long>(15)] << std::endl;
#ifndef ASCENDC_CPU_DEBUG
    dev.close();
#endif
    return app.finish();
}

