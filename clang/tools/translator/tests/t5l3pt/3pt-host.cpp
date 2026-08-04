// 自动生成的 AscendC Vector Host 代码 - 修复版
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>
#include "acl/acl.h"

// ACL 错误检查宏
#ifndef CHECK_ACL
#define CHECK_ACL(x) do { \
    aclError __ret = x; \
    if (__ret != ACL_SUCCESS) { \
        std::cerr << "[ERROR] " << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        std::abort(); \
    } \
} while (0)
#endif

// 声明外部 NPU Kernel 调用函数
extern void custom_kernel_pde3_do(uint32_t blockDim, void *stream, uint8_t *u_kin, uint8_t *u_kout, uint8_t *r, uint32_t totalLength, uint32_t coreNum);

int32_t main() {
    // 1. 初始化 ACL
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));

    // 2. 核心语义参数设定（依据 a.dac.cpp 还原）
    // 输出长度为 4，滑窗大小为 3（需要额外读取 2 个数据），所以输入长度为 6
    uint32_t outLength = 8;
    uint32_t inLength = outLength + 2; 

    // 3. 内存与对齐规则 (AscendC 要求 32 字节/8 个 float 对齐)
    // 即使我们只处理 4/6 个有效数据，内存分配必须以 8 float (32 Byte) 的整数倍分配，避免 NPU 搬运报错
    size_t inAlignedCount = ((inLength + 7) / 8) * 8;   // 8
    size_t outAlignedCount = ((outLength + 7) / 8) * 8; // 8
    size_t rAlignedCount = 8; // 标量也占 1 个 32B block
    
    size_t inByteSize = inAlignedCount * sizeof(float);
    size_t outByteSize = outAlignedCount * sizeof(float);
    size_t rByteSize = rAlignedCount * sizeof(float);

    // 4. Host 侧数据准备
    std::vector<float> u_kinHost(inAlignedCount, 0.0f);
    std::vector<float> u_koutHost(outAlignedCount, 0.0f);
    std::vector<float> rHost(rAlignedCount, 0.0f);

    // 填入真实数据 (PDE3 u_data)
    for (size_t i = 0; i < inLength; ++i) u_kinHost[i] = static_cast<float>(i); // {0, 1, 2, 3, 4, 5}
    rHost[0] = 0.25f;

    // 5. Device 侧内存申请与数据搬运
    void *u_kinDevice = nullptr, *u_koutDevice = nullptr, *rDevice = nullptr;
    CHECK_ACL(aclrtMalloc(&u_kinDevice, inByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&u_koutDevice, outByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&rDevice, rByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMemcpy(u_kinDevice, inByteSize, u_kinHost.data(), inByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(rDevice, rByteSize, rHost.data(), rByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    
    // 初始化 Device 侧输出为全 0
    CHECK_ACL(aclrtMemset(u_koutDevice, outByteSize, 0, outByteSize));

    // 6. 发起 Kernel 执行任务
    uint32_t blockDim = 1; // 极小数据量，单核执行即可
    uint32_t coreNum = 1;
    std::cout << "[INFO] Launching Kernel PDE3... Valid Out Length: " << outLength << std::endl;
    custom_kernel_pde3_do(blockDim, stream, (uint8_t*)u_kinDevice, (uint8_t*)u_koutDevice, (uint8_t*)rDevice, outLength, coreNum);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // 7. 取回计算结果
    CHECK_ACL(aclrtMemcpy(u_koutHost.data(), outByteSize, u_koutDevice, outByteSize, ACL_MEMCPY_DEVICE_TO_HOST));

    // 8. 验证与打印结果
    std::cout << "[NPU] Result u_kout: ";
    for (size_t i = 0; i < outLength; ++i) {
        std::cout << u_koutHost[i] << " ";
    }
    std::cout << std::endl;

    // 9. 释放资源
    CHECK_ACL(aclrtFree(u_kinDevice));
    CHECK_ACL(aclrtFree(u_koutDevice));
    CHECK_ACL(aclrtFree(rDevice));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());

    return 0;
}