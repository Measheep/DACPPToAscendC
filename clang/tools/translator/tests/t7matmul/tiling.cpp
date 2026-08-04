
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <cstdlib>
#include <limits>
#include <stdexcept>

#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
using namespace matmul_tiling;
using namespace std;

uint8_t *GetTilingBuf(optiling::TCubeTiling *tilingData)
{
    uint32_t tilingSize = tilingData->GetDataSize();
    uint8_t *buf = (uint8_t *)malloc(tilingSize);
    tilingData->SaveToBuffer(buf, tilingSize);
    return buf;
}


extern "C" uint8_t *makeShapeTiling(
    const char *socVersion,
    int M,
    int N,
    int K,
    int usedCoreNum);

extern "C" uint8_t *makeTiling(const char *socVersion)
{
    int M = 2048;
    int N = 2048;
    int K = 4096;
    int usedCoreNum = 1; // 默认使用1核测试
    return makeShapeTiling(socVersion, M, N, K, usedCoreNum);
}

extern "C" uint8_t *makeShapeTiling(
    const char *socVersion,
    int M,
    int N,
    int K,
    int usedCoreNum)
{
    M = std::max(M, 1);
    N = std::max(N, 1);
    K = std::max(K, 1);
    usedCoreNum = std::max(usedCoreNum, 1);

    TPosition leftPosition = TPosition::GM;
    CubeFormat leftFormat = CubeFormat::ND;
    DataType leftDtype = DataType::DT_FLOAT16;
    bool isTransA = false;

    TPosition rightPosition = TPosition::GM;
    CubeFormat rightFormat = CubeFormat::ND;
    DataType rightDtype = DataType::DT_FLOAT16;
    bool isTransB = false;

    TPosition resultPosition = TPosition::GM;
    CubeFormat resultFormat = CubeFormat::ND;
    DataType resultDtype = DataType::DT_FLOAT;

    bool isBias = false;
    int32_t baseM = 16;  // 基础切分
    int32_t baseN = 16;

    optiling::TCubeTiling tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    MultiCoreMatmulTiling tilingApi(*ascendcPlatform);

    tilingApi.SetDim(usedCoreNum);
    tilingApi.SetAType(leftPosition, leftFormat, leftDtype, isTransA);
    tilingApi.SetBType(rightPosition, rightFormat, rightDtype, isTransB);
    tilingApi.SetCType(resultPosition, resultFormat, resultDtype);

    tilingApi.SetOrgShape(M, N, K);
    tilingApi.SetShape(M, N, K);
    tilingApi.SetFixSplit(baseM, baseN, -1);
    tilingApi.SetBias(isBias);
    tilingApi.SetBufferSpace(-1, -1, -1);

    int64_t res = tilingApi.GetTiling(tilingData);
    if (res == -1) {
        std::cout << "gen tiling failed" << std::endl;
    }
    return GetTilingBuf(&tilingData);
}


