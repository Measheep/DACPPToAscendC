//  自动生成的 AscendC Vector Host 代码
#include "dacpp_ascendc_host_runtime.h"
static_assert(DACPP_ASCENDC_HOST_RUNTIME_VERSION == 19,
              "regenerate Host runtime");


#ifndef ASCENDC_CPU_DEBUG
extern "C" uint32_t vector_kernel_0_do(uint32_t blockDim, void *stream, uint8_t *input, uint8_t *weight, uint8_t *bias, uint8_t *output, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount);
#else
extern "C" __global__ __aicore__ void vector_kernel_0(GM_ADDR input, GM_ADDR weight, GM_ADDR bias, GM_ADDR output, uint32_t totalLength, uint32_t coreNum, uint32_t tileLength, uint32_t tileCount);
#endif

const int IN_FEATURES = 64;
const int OUT_FEATURES = 1048576;

int32_t main(int32_t argc, char *argv[]) {
    dacpp_host::Session app;
#ifndef ASCENDC_CPU_DEBUG
    dacpp_host::Device dev(app, 0);
    if (!dev.ok()) return app.finish();
    aclrtStream stream = dev.stream();
#endif
    std::vector<float> input_data(IN_FEATURES, 0.0f);
    for (int i = 0; i < IN_FEATURES; ++i) {
        input_data[i] = static_cast<float>(i) * 0.01f;
    }
    std::vector<float> weight_data(OUT_FEATURES * IN_FEATURES, 0.0f);
    for (int idx = 0; idx < OUT_FEATURES * IN_FEATURES; ++idx) {
        weight_data[idx] = static_cast<float>(idx % IN_FEATURES) * 0.001f;
    }
    std::vector<float> bias_data(OUT_FEATURES, 0.1f);
    std::vector<float> output_data(OUT_FEATURES, 0.0f);
    {
    //  算子 vector_kernel_0 基础参数与数据定义 
        uint32_t blockDim0 = 8;
        dacpp_host::HostBuffer<float> inputBuf(app, static_cast<size_t>(1), static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>((256 + sizeof(float) - 1) / sizeof(float)));
        std::fill_n(inputBuf.data.begin(), inputBuf.logicalElems, static_cast<float>(input_data[0]));
        dacpp_host::HostBuffer<float> weightBuf(app, weight_data.size(), static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>((256 + sizeof(float) - 1) / sizeof(float)));
            std::memcpy(weightBuf.data.data(), weight_data.data(), std::min(weightBuf.logicalElems, weight_data.size()) * sizeof(float));
        dacpp_host::HostBuffer<float> biasBuf(app, bias_data.size(), static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>((256 + sizeof(float) - 1) / sizeof(float)));
            std::memcpy(biasBuf.data.data(), bias_data.data(), std::min(biasBuf.logicalElems, bias_data.size()) * sizeof(float));
        dacpp_host::ensureSize(output_data, static_cast<size_t>(1048576));
        dacpp_host::HostBuffer<float> outputBuf(app, output_data.size(), static_cast<size_t>(0), static_cast<size_t>(0), static_cast<size_t>((256 + sizeof(float) - 1) / sizeof(float)));
        size_t kernelSize0 = static_cast<size_t>(0);
        const size_t coreCount0 = blockDim0 == 0 ? static_cast<size_t>(1) : static_cast<size_t>(blockDim0);
        kernelSize0 = std::max(kernelSize0, weightBuf.kernelElems);
        kernelSize0 = std::max(kernelSize0, biasBuf.kernelElems);
        kernelSize0 = std::max(kernelSize0, outputBuf.kernelElems);
        uint32_t kernelLength0 = 0;
        kernelSize0 = outputBuf.logicalElems;
        const dacpp_host::Tile vectorRuntimeTiling0 = dacpp_host::makeTile(app, kernelSize0, coreCount0, 64, 192);
        kernelSize0 = static_cast<size_t>(vectorRuntimeTiling0.len);
        kernelLength0 = vectorRuntimeTiling0.len;
        uint32_t tileLength0 = vectorRuntimeTiling0.size;
        uint32_t tileCount0 = vectorRuntimeTiling0.count;
        inputBuf.data.resize(std::max(inputBuf.data.size(), static_cast<size_t>(64)));
        if (kernelSize0 > std::numeric_limits<size_t>::max() / static_cast<size_t>(64)) { app.fail(dacpp_host::Stage::Config, 2, "linear weight padding overflow"); return app.finish(); }
        weightBuf.data.resize(std::max(weightBuf.data.size(), kernelSize0 * static_cast<size_t>(64)));
        outputBuf.data.resize(std::max(outputBuf.data.size(), kernelSize0));
        biasBuf.data.resize(std::max(biasBuf.data.size(), kernelSize0));
        weightBuf.kernelElems = std::max(weightBuf.kernelElems, kernelSize0);
        { size_t withLeftHalo = 0, storage = 0, physical = 0; dacpp_host::checkedAdd(app, weightBuf.kernelElems, weightBuf.leftHaloElems, withLeftHalo); dacpp_host::checkedAdd(app, withLeftHalo, weightBuf.rightHaloElems, storage); dacpp_host::checkedAlign(app, storage, weightBuf.alignElems, physical); if (app.ok() && weightBuf.data.size() < physical) weightBuf.data.resize(physical); }
        biasBuf.kernelElems = std::max(biasBuf.kernelElems, kernelSize0);
        { size_t withLeftHalo = 0, storage = 0, physical = 0; dacpp_host::checkedAdd(app, biasBuf.kernelElems, biasBuf.leftHaloElems, withLeftHalo); dacpp_host::checkedAdd(app, withLeftHalo, biasBuf.rightHaloElems, storage); dacpp_host::checkedAlign(app, storage, biasBuf.alignElems, physical); if (app.ok() && biasBuf.data.size() < physical) biasBuf.data.resize(physical); }
        outputBuf.kernelElems = std::max(outputBuf.kernelElems, kernelSize0);
        { size_t withLeftHalo = 0, storage = 0, physical = 0; dacpp_host::checkedAdd(app, outputBuf.kernelElems, outputBuf.leftHaloElems, withLeftHalo); dacpp_host::checkedAdd(app, withLeftHalo, outputBuf.rightHaloElems, storage); dacpp_host::checkedAlign(app, storage, outputBuf.alignElems, physical); if (app.ok() && outputBuf.data.size() < physical) outputBuf.data.resize(physical); }
    //  算子 vector_kernel_0 NPU 执行流 
    #ifndef ASCENDC_CPU_DEBUG
        dev.allocate(inputBuf);
        dev.allocate(weightBuf);
        dev.allocate(biasBuf);
        dev.allocate(outputBuf);
    #else
        uint8_t *cpuGm_input = (uint8_t *)AscendC::GmAlloc(inputBuf.bytes());
        uint8_t *cpuGm_weight = (uint8_t *)AscendC::GmAlloc(weightBuf.bytes());
        uint8_t *cpuGm_bias = (uint8_t *)AscendC::GmAlloc(biasBuf.bytes());
        uint8_t *cpuGm_output = (uint8_t *)AscendC::GmAlloc(outputBuf.bytes());
    #endif
    #ifndef ASCENDC_CPU_DEBUG
        dev.copyIn(inputBuf);
        dev.copyIn(weightBuf);
        dev.copyIn(biasBuf);
    #else
        memcpy(cpuGm_input, inputBuf.data.data(), inputBuf.bytes());
        memcpy(cpuGm_weight, weightBuf.data.data(), weightBuf.bytes());
        memcpy(cpuGm_bias, biasBuf.data.data(), biasBuf.bytes());
    #endif
    #ifndef ASCENDC_CPU_DEBUG
        dev.launch(vector_kernel_0_do(blockDim0, stream, inputBuf.device, weightBuf.device, biasBuf.device, outputBuf.device, kernelLength0, blockDim0, tileLength0, tileCount0));
        dev.sync();
    #else
        ICPU_RUN_KF(vector_kernel_0, blockDim0, cpuGm_input, cpuGm_weight, cpuGm_bias, cpuGm_output, kernelLength0, blockDim0, tileLength0, tileCount0);
    #endif
    #ifndef ASCENDC_CPU_DEBUG
        dev.copyOut(outputBuf);
        const size_t outputOffset = static_cast<size_t>(0);
        std::memcpy(output_data.data() + outputOffset, outputBuf.data.data(), outputBuf.logicalElems * sizeof(float));
    #else
        memcpy(outputBuf.data.data(), cpuGm_output, outputBuf.bytes());
        const size_t outputOffset = static_cast<size_t>(0);
        std::memcpy(output_data.data() + outputOffset, outputBuf.data.data(), outputBuf.logicalElems * sizeof(float));
    #endif
    #ifndef ASCENDC_CPU_DEBUG
        dev.release(inputBuf);
        dev.release(weightBuf);
        dev.release(biasBuf);
        dev.release(outputBuf);
    #else
        AscendC::GmFree((void *)cpuGm_input);
        AscendC::GmFree((void *)cpuGm_weight);
        AscendC::GmFree((void *)cpuGm_bias);
        AscendC::GmFree((void *)cpuGm_output);
    #endif
    }
    {
        std::cout << "{";
        for (long long dacppPrintI0_0 = 0; dacppPrintI0_0 < static_cast<long long>(output_data.size()); ++dacppPrintI0_0) {
            if (dacppPrintI0_0 != 0) { std::cout << ", "; }
            std::cout << (output_data.data())[static_cast<long long>(0) + dacppPrintI0_0 * static_cast<long long>(1)];
        }
        std::cout << "}";
        std::cout << std::endl;
    }
#ifndef ASCENDC_CPU_DEBUG
    dev.close();
#endif
    return app.finish();
}

