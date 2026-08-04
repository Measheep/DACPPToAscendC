// liuliang-host.cpp
#include "acl/acl.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>

#define CHECK_ACL(x) do { aclError __ret = x; if (__ret != ACL_SUCCESS) { std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; return -1; } } while (0)

// 引用内核入口
extern void custom_kernel_0_do(uint32_t blockDim, void *stream, uint8_t *rho, uint8_t *new_rho, uint32_t totalLength, uint32_t coreNum);

const int WIDTH = 128;
const int TIME_STEPS = 256;
const int ACTIVE_LENGTH = WIDTH - 1;
constexpr float DELTA_T = 0.01f;
constexpr float DELTA_X = 1.0f;

static inline float Flux(float rho) {
    constexpr float V_MAX = 30.0f;
    constexpr float RHO_MAX = 50.0f;
    return rho * V_MAX * (1.0f - rho / RHO_MAX);
}

static inline void InitializeDensity(std::vector<float> &rho) {
    for (int i = 0; i < WIDTH; ++i) {
        if (i < WIDTH / 4) {
            rho[i] = 40.0f;
        } else if (i < 3 * WIDTH / 4) {
            rho[i] = 20.0f;
        } else {
            rho[i] = 10.0f;
        }
    }
}

static inline void ApplyDacStyleUpdate(std::vector<float> &rho, const std::vector<float> &newRho) {
    for (int i = 1; i <= ACTIVE_LENGTH - 1; ++i) {
        rho[i] = newRho[i - 1];
    }
    rho[0] = newRho[0];
}

int main() {
    // 1. 初始化：与 dac 源逻辑保持一致
    std::vector<float> rho(WIDTH, 0.0f);
    InitializeDensity(rho);

    // CPU golden: 使用与 dac 主循环相同的时间推进和回写逻辑
    std::vector<float> expectedState = rho;
    std::vector<float> expectedStep(ACTIVE_LENGTH, 0.0f);
    for (int t = 0; t < TIME_STEPS; ++t) {
        for (int i = 0; i < ACTIVE_LENGTH; ++i) {
            const float center = expectedState[i];
            const float shifted = expectedState[i + 1];
            const float update = shifted - (DELTA_T / DELTA_X) * (Flux(shifted) - Flux(center));
            expectedStep[i] = std::max(0.0f, update);
        }
        ApplyDacStyleUpdate(expectedState, expectedStep);
    }

    // 2. 计算 Ascend 架构强制的 32-Byte (8 float) 内存对齐
    uint32_t totalLength = ACTIVE_LENGTH;
    // 输入读取需要 i+1，故长度按 totalLength+1 预留并对齐
    uint32_t readAlignedLength = (totalLength + 1 + 8 - 1) / 8 * 8; 
    // 输出按 totalLength 对齐
    uint32_t writeAlignedLength = (totalLength + 8 - 1) / 8 * 8;    

    // 创建对齐后的内存容器
    std::vector<float> rhoAligned(readAlignedLength, 0.0f);
    std::vector<float> newRhoAligned(writeAlignedLength, 0.0f);

    uint32_t rhoByteSize = readAlignedLength * sizeof(float);
    uint32_t newRhoByteSize = writeAlignedLength * sizeof(float);

    // 3. ACL 初始化与 NPU 设备资源分配
    std::cout << "[INFO] Initializing ACL..." << std::endl;
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtContext context;
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));

    void *rhoDevice, *newRhoDevice;
    CHECK_ACL(aclrtMalloc(&rhoDevice, rhoByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&newRhoDevice, newRhoByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // 4. 启动 AscendC 内核并执行 TIME_STEPS 次推进
    std::cout << "[INFO] Start running Ascend C Kernel on NPU..." << std::endl;
    uint32_t blockDim = 1;
    uint32_t coreNum = 1;
    std::vector<float> npuState = rho;
    for (int t = 0; t < TIME_STEPS; ++t) {
        std::fill(rhoAligned.begin(), rhoAligned.end(), 0.0f);
        for (int i = 0; i < WIDTH; ++i) {
            rhoAligned[i] = npuState[i];
        }

        CHECK_ACL(aclrtMemcpy(rhoDevice, rhoByteSize, rhoAligned.data(), rhoByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemset(newRhoDevice, newRhoByteSize, 0, newRhoByteSize));

        custom_kernel_0_do(blockDim, stream, (uint8_t*)rhoDevice, (uint8_t*)newRhoDevice, totalLength, coreNum);
        CHECK_ACL(aclrtSynchronizeStream(stream));

        CHECK_ACL(aclrtMemcpy(newRhoAligned.data(), newRhoByteSize, newRhoDevice, newRhoByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
        ApplyDacStyleUpdate(npuState, newRhoAligned);
    }

    // 5. 严格数值验证
    bool pass = true;
    constexpr float kTolerance = 1e-4f;
    float maxAbsErr = 0.0f;
    int mismatchCount = 0;
    std::cout << "[NPU] Result middle_in_tensor after " << TIME_STEPS << " steps:\n";
    for(int i = 0; i < ACTIVE_LENGTH; ++i) {
        const float actual = npuState[i];
        const float golden = expectedState[i];
        const float absErr = std::fabs(actual - golden);
        if (absErr > kTolerance) {
            pass = false;
            ++mismatchCount;
        }
        if (absErr > maxAbsErr) {
            maxAbsErr = absErr;
        }

        std::cout << std::fixed << std::setprecision(1) << actual << " ";
        // 每 25 个元素换行，方便观察 3 个密度区间的阶梯现象
        if ((i + 1) % 25 == 0) std::cout << "\n";
    }
    std::cout << std::endl;

    std::cout << "[CPU] Expected middle_in_tensor after " << TIME_STEPS << " steps:\n";
    for (int i = 0; i < ACTIVE_LENGTH; ++i) {
        std::cout << std::fixed << std::setprecision(1) << expectedState[i] << " ";
        if ((i + 1) % 25 == 0) std::cout << "\n";
    }
    std::cout << std::endl;
    std::cout << "[CHECK] mismatch=" << mismatchCount
              << ", maxAbsErr=" << std::setprecision(6) << maxAbsErr
              << ", tol=" << kTolerance << std::endl;
    std::cout << "[RESULT] " << (pass ? "PASS" : "FAIL") << std::endl;

    std::cout << "[DAC] middle_in_tensor[15] = " << std::setprecision(6) << npuState[15] << std::endl;

    // 6. 释放所有硬件资源
    CHECK_ACL(aclrtFree(rhoDevice));
    CHECK_ACL(aclrtFree(newRhoDevice));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    std::cout << "[INFO] Execution finished." << std::endl;

    return pass ? 0 : 1;
}