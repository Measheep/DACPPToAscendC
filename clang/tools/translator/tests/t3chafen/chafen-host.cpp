#include "acl/acl.h"
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#define CHECK_ACL(x) do { aclError __ret = (x); if (__ret != ACL_SUCCESS) { std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; return -1; } } while (0)

extern void custom_kernel_0_do(uint32_t blockDim,
                               void *stream,
                               uint8_t *rho,
                               uint8_t *diff,
                               uint32_t totalLength,
                               uint32_t coreNum);

const int WIDTH = 16;

void initializeInput(std::vector<float>& rho) {
    for (int i = 0; i <= WIDTH; ++i) {
        rho[i] = static_cast<float>(i * i);
    }
}

int main() {
    std::vector<float> rho(WIDTH + 1, 0.0f);
    std::vector<float> expected(WIDTH, 0.0f);
    initializeInput(rho);
    for (int i = 0; i < WIDTH; ++i) {
        expected[i] = rho[i + 1] - rho[i];
    }

    uint32_t totalLength = WIDTH;
    uint32_t readAlignedLength = (totalLength + 1 + 8 - 1) / 8 * 8;
    uint32_t writeAlignedLength = (totalLength + 8 - 1) / 8 * 8;
    std::vector<float> rhoAligned(readAlignedLength, 0.0f);
    std::vector<float> diffAligned(writeAlignedLength, 0.0f);

    for (uint32_t i = 0; i <= static_cast<uint32_t>(WIDTH); ++i) {
        rhoAligned[i] = rho[i];
    }

    uint32_t rhoByteSize = readAlignedLength * sizeof(float);
    uint32_t diffByteSize = writeAlignedLength * sizeof(float);

    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));

    aclrtContext context;
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));

    void *rhoDevice = nullptr;
    void *diffDevice = nullptr;
    CHECK_ACL(aclrtMalloc(&rhoDevice, rhoByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&diffDevice, diffByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMemcpy(rhoDevice, rhoByteSize, rhoAligned.data(), rhoByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemset(diffDevice, diffByteSize, 0, diffByteSize));

    custom_kernel_0_do(1, stream, reinterpret_cast<uint8_t *>(rhoDevice), reinterpret_cast<uint8_t *>(diffDevice), totalLength, 1);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(diffAligned.data(), diffByteSize, diffDevice, diffByteSize, ACL_MEMCPY_DEVICE_TO_HOST));

    bool pass = true;
    constexpr float kTolerance = 1e-5f;
    std::cout << "[INPUT] rho:\n";
    for (int i = 0; i <= WIDTH; ++i) {
        std::cout << std::fixed << std::setprecision(1) << rho[i] << " ";
    }
    std::cout << "\n[OUTPUT] diff:\n";
    for (int i = 0; i < WIDTH; ++i) {
        float actual = diffAligned[i];
        float golden = expected[i];
        if (std::fabs(actual - golden) > kTolerance) {
            pass = false;
        }
        std::cout << std::fixed << std::setprecision(1) << actual << " ";
    }
    std::cout << "\n[EXPECT] diff:\n";
    for (int i = 0; i < WIDTH; ++i) {
        std::cout << std::fixed << std::setprecision(1) << expected[i] << " ";
    }
    std::cout << "\n[RESULT] " << (pass ? "PASS" : "FAIL") << std::endl;

    CHECK_ACL(aclrtFree(rhoDevice));
    CHECK_ACL(aclrtFree(diffDevice));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());

    return pass ? 0 : 1;
}
