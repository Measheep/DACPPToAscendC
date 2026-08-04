// 自动生成的 AscendC Host 侧调度代码 - 线性层 (Linear / MatVec) 架构适配版
#include <iostream>
#include <vector>
#include <iomanip>
#include "acl/acl.h"

// ACL 错误检查宏，发生底层异常时第一时间拦截并抛出
#define CHECK_ACL(x) \
    do { \
        aclError __ret = x; \
        if (__ret != ACL_SUCCESS) { \
            std::cerr << "[ERROR] " << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
            std::abort(); \
        } \
    } while (0)

// 外部声明自动生成的 NPU 内核启动桩函数 (Stub)
extern void custom_kernel_0_do(uint32_t blockDim, void *stream, uint8_t *input, uint8_t *weight, uint8_t *bias, uint8_t *output, uint32_t totalLength, uint32_t coreNum);

constexpr int IN_FEATURES = 64;
constexpr int OUT_FEATURES = 64;

int main() {
    // =========================================================================
    // 1. 昇腾计算资源初始化与上下文绑定
    // =========================================================================
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    // =========================================================================
    // 2. Host 侧内存准备与测试用例构造 (严格匹配 linear.dac.cpp 语义)
    // =========================================================================
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

    // 数据字节量计算
    size_t input_bytes = input_data.size() * sizeof(float);
    size_t weight_bytes = weight_data.size() * sizeof(float);
    size_t bias_bytes = bias_data.size() * sizeof(float);
    size_t output_bytes = output_data.size() * sizeof(float);

    // =========================================================================
    // 3. 申请 Device (NPU) 侧全局内存 (Global Memory)
    // =========================================================================
    void *input_dev = nullptr, *weight_dev = nullptr, *bias_dev = nullptr, *output_dev = nullptr;
    CHECK_ACL(aclrtMalloc(&input_dev, input_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&weight_dev, weight_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&bias_dev, bias_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&output_dev, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));

    // =========================================================================
    // 4. 数据搬移: Host -> Device
    // =========================================================================
    CHECK_ACL(aclrtMemcpy(input_dev, input_bytes, input_data.data(), input_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(weight_dev, weight_bytes, weight_data.data(), weight_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(bias_dev, bias_bytes, bias_data.data(), bias_bytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // =========================================================================
    // 5. 启动 AscendC Kernel
    // =========================================================================
    std::cout << "[INFO] Start running Ascend C Kernel on NPU..." << std::endl;
    // 当前模板保持单核下发，便于稳定验证语义正确性
    uint32_t blockDim = 1; 
    custom_kernel_0_do(blockDim, stream,
                       reinterpret_cast<uint8_t*>(input_dev),
                       reinterpret_cast<uint8_t*>(weight_dev),
                       reinterpret_cast<uint8_t*>(bias_dev),
                       reinterpret_cast<uint8_t*>(output_dev),
                       OUT_FEATURES, 1);

    // 阻塞式同步，等待 Device 计算完毕
    CHECK_ACL(aclrtSynchronizeStream(stream));
    std::cout << "[INFO] Kernel execution finished successfully. (MTE Aligned)" << std::endl;

    // =========================================================================
    // 6. 数据搬移: Device -> Host
    // =========================================================================
    CHECK_ACL(aclrtMemcpy(output_data.data(), output_bytes, output_dev, output_bytes, ACL_MEMCPY_DEVICE_TO_HOST));

    // =========================================================================
    // 7. 打印与验证校验结果
    // =========================================================================
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "[INFO] Output (first 16 / " << OUT_FEATURES << "): ";
    for (int i = 0; i < 16 && i < static_cast<int>(output_data.size()); ++i) {
        std::cout << std::fixed << std::setprecision(6) << output_data[i] << ", ";
    }
    std::cout << "\n--------------------------------------------------------" << std::endl;

    // =========================================================================
    // 8. 释放资源与环境卸载
    // =========================================================================
    CHECK_ACL(aclrtFree(input_dev));
    CHECK_ACL(aclrtFree(weight_dev));
    CHECK_ACL(aclrtFree(bias_dev));
    CHECK_ACL(aclrtFree(output_dev));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());

    return 0;
}