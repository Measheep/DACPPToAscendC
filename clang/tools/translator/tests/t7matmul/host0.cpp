//  自动生成的 AscendC Cube Host 代码
#include "dacpp_ascendc_host_runtime.h"
static_assert(DACPP_ASCENDC_HOST_RUNTIME_VERSION == 19,
              "regenerate Host runtime");


//  包含矩阵 Tiling 所需的核心依赖
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"

#include "aclrtlaunch_cube_kernel_0.h"

extern "C" uint8_t *makeTiling(const char *socVersion);
extern "C" uint8_t *makeShapeTiling(const char *socVersion, int M, int N, int K, int usedCoreNum);

using namespace std;

int32_t main(int32_t argc, char *argv[]) {
    dacpp_host::Session app;
#ifndef ASCENDC_CPU_DEBUG
    dacpp_host::Device dev(app, 0);
    if (!dev.ok()) return app.finish();
    aclrtStream stream = dev.stream();
#endif
    std::vector<float> dataA(2048*4096, 1.0f);
    std::vector<float> dataB(4096*2048, 1.0f);
    {
    //  算子 cube_kernel_0 基础参数与数据定义（plan + runtime binding）
        const char *socVersion = SOC_VERSION;
        auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
        uint32_t blockDim0 = 8;
    
        const size_t mElemCount = static_cast<size_t>(2048);
        const size_t nElemCount = static_cast<size_t>(2048);
        const size_t kElemCount = static_cast<size_t>(4096);
        const size_t aElemCount = mElemCount * kElemCount;
        const size_t bElemCount = kElemCount * nElemCount;
        const size_t cElemCount = mElemCount * nElemCount;
    
        size_t aFileSize = aElemCount * sizeof(uint16_t);
        size_t bFileSize = bElemCount * sizeof(uint16_t);
        size_t cFileSize = cElemCount * sizeof(float);
        size_t tilingFileSize = sizeof(TCubeTiling);
    
        size_t userWorkspaceSize = 0;
        size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
        size_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;
    
        uint16_t *matAData = (uint16_t*)malloc(aFileSize);
        uint16_t *matBData = (uint16_t*)malloc(bFileSize);
        float *matCData = (float*)malloc(cFileSize);
    
        memset(matAData, 0, aFileSize);
        memset(matBData, 0, bFileSize);
        memset(matCData, 0, cFileSize);
    
        {
            size_t runtimeCount = std::min(aElemCount, dataA.size());
            for (size_t i = 0; i < runtimeCount; ++i) {
                matAData[i] = static_cast<uint16_t>(dacpp_host::fp32ToFp16(static_cast<float>(dataA[i])));
            }
        }
        {
            size_t runtimeCount = std::min(bElemCount, dataB.size());
            for (size_t i = 0; i < runtimeCount; ++i) {
                matBData[i] = static_cast<uint16_t>(dacpp_host::fp32ToFp16(static_cast<float>(dataB[i])));
            }
        }
        // Runtime binding not resolved for matC, keep zero-initialized staging buffer.
        //  算子 cube_kernel_0 NPU 执行流
        std::vector<float> matCInitSnapshot(cElemCount, 0.0f);
        for (size_t i = 0; i < cElemCount; ++i) {
            matCInitSnapshot[i] = static_cast<float>(matCData[i]);
        }
    
        uint8_t *matAHost = nullptr, *matADevice = nullptr;
        dev.allocate(matAHost, matADevice, aFileSize);
        uint8_t *matBHost = nullptr, *matBDevice = nullptr;
        dev.allocate(matBHost, matBDevice, bFileSize);
        uint8_t *workspaceDevice = nullptr;
        dev.allocate(workspaceDevice, workspaceSize);
        uint8_t *matCHost = nullptr, *matCDevice = nullptr;
        uint8_t *tilingHost = nullptr, *tilingDevice = nullptr;
        dev.allocate(tilingHost, tilingDevice, tilingFileSize);
        dev.allocate(matCHost, matCDevice, cFileSize);
        std::memcpy(matAHost, matAData, aFileSize);
        dev.copyIn(matADevice, aFileSize, matAHost, aFileSize);
        std::memcpy(matBHost, matBData, bFileSize);
        dev.copyIn(matBDevice, bFileSize, matBHost, bFileSize);
        uint8_t* generatedTiling = makeTiling(socVersion);
        std::memcpy(tilingHost, generatedTiling, tilingFileSize);
        dev.copyIn(tilingDevice, tilingFileSize, tilingHost, tilingFileSize);
        dev.launch(ACLRT_LAUNCH_KERNEL(cube_kernel_0)(blockDim0, stream, matADevice, matBDevice, matCDevice, workspaceDevice, tilingDevice));
        dev.sync();
        dev.copyOut(matCHost, cFileSize, matCDevice, cFileSize);
        std::memcpy(matCData, matCHost, cFileSize);
        for (size_t i = 0; i < cElemCount; ++i) {
            matCData[i] = static_cast<float>(matCData[i]) + matCInitSnapshot[i];
        }
    {
        std::cout << "{";
        for (long long dacppPrintI0_0 = 0; dacppPrintI0_0 < static_cast<long long>(mElemCount); ++dacppPrintI0_0) {
            if (dacppPrintI0_0 != 0) { std::cout << ", "; }
            std::cout << "{";
            for (long long dacppPrintI0_1 = 0; dacppPrintI0_1 < static_cast<long long>(nElemCount); ++dacppPrintI0_1) {
                if (dacppPrintI0_1 != 0) { std::cout << ", "; }
                std::cout << (matCData)[static_cast<long long>(0) + dacppPrintI0_0 * static_cast<long long>(nElemCount) + dacppPrintI0_1 * static_cast<long long>(1)];
            }
            std::cout << "}";
        }
        std::cout << "}";
        std::cout << std::endl;
    }
        dev.release(matAHost, matADevice);
        dev.release(matBHost, matBDevice);
        dev.release(matCHost, matCDevice);
        dev.release(workspaceDevice);
        dev.release(tilingHost, tilingDevice);
        std::free(generatedTiling);
    }
#ifndef ASCENDC_CPU_DEBUG
    dev.close();
#endif
    return app.finish();
}

