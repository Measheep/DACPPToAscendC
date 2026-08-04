/**
 * @file main.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 * Modified for DACPPToAscendC test integration; see THIRD_PARTY_NOTICES.md.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "data_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_matmul_custom.h"
#else
#include "tikicpulib.h"
extern "C" void matmul_custom(uint8_t *a, uint8_t *b, uint8_t *c, uint8_t *workspace, uint8_t *tiling);
#endif
extern uint8_t *GenerateTiling(const char *socVersion);

int32_t main(int32_t argc, char *argv[])
{
    const char *socVersion = SOC_VERSION;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    size_t aFileSize = 512 * 512 * sizeof(uint16_t);  // uint16_t represent half
    size_t bFileSize = 512 * 1024 * sizeof(uint16_t); // uint16_t represent half
    size_t cFileSize = 512 * 1024 * sizeof(float);
    size_t tilingFileSize = sizeof(TCubeTiling);
    size_t userWorkspaceSize = 0;
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    size_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;

    uint32_t blockDim = 1;


    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *aHost;
    uint8_t *aDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&aHost), aFileSize));
    CHECK_ACL(aclrtMalloc((void **)&aDevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(aDevice, aFileSize, aHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *bHost;
    uint8_t *bDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&bHost), bFileSize));
    CHECK_ACL(aclrtMalloc((void **)&bDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(bDevice, bFileSize, bHost, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *workspaceDevice;
    CHECK_ACL(aclrtMalloc((void **)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *tilingHost;
    uint8_t *tilingDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&tilingHost), tilingFileSize));
    CHECK_ACL(aclrtMalloc((void **)&tilingDevice, tilingFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(
        aclrtMemcpy(tilingHost, tilingFileSize, GenerateTiling(socVersion), tilingFileSize, ACL_MEMCPY_HOST_TO_HOST));
    CHECK_ACL(aclrtMemcpy(tilingDevice, tilingFileSize, tilingHost, tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *cHost;
    uint8_t *cDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&cHost), cFileSize));
    CHECK_ACL(aclrtMalloc((void **)&cDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ACLRT_LAUNCH_KERNEL(matmul_custom)
    (blockDim, stream, aDevice, bDevice, cDevice, workspaceDevice, tilingDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(cHost, cFileSize, cDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST));

    CHECK_ACL(aclrtFree(aDevice));
    CHECK_ACL(aclrtFreeHost(aHost));
    CHECK_ACL(aclrtFree(bDevice));
    CHECK_ACL(aclrtFreeHost(bHost));
    CHECK_ACL(aclrtFree(workspaceDevice));
    CHECK_ACL(aclrtFree(tilingDevice));
    CHECK_ACL(aclrtFreeHost(tilingHost));
    CHECK_ACL(aclrtFree(cDevice));
    CHECK_ACL(aclrtFreeHost(cHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}
