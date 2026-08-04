
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

__aicore__ inline void CopyTiling(TCubeTiling *tiling, GM_ADDR tilingGM) {
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);
    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
}

__aicore__ inline void CalcGMOffset(int blockIdx, const TCubeTiling &tiling, int &offsetA, int &offsetB, int &offsetC,
                                    int &tailM, int &tailN, bool isTransA, bool isTransB) {
    uint32_t mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
    uint32_t mCoreIndx = blockIdx % mSingleBlocks;
    uint32_t nCoreIndx = blockIdx / mSingleBlocks;

    offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
    if (isTransA) offsetA = mCoreIndx * tiling.singleCoreM;

    offsetB = nCoreIndx * tiling.singleCoreN;
    if (isTransB) offsetB = nCoreIndx * tiling.Kb * tiling.singleCoreN;

    offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;

    tailM = tiling.M - mCoreIndx * tiling.singleCoreM;
    tailM = tailM < tiling.singleCoreM ? tailM : tiling.singleCoreM;
    tailN = tiling.N - nCoreIndx * tiling.singleCoreN;
    tailN = tailN < tiling.singleCoreN ? tailN : tiling.singleCoreN;
}

extern "C" __global__ __aicore__ void cube_kernel_0(GM_ADDR matA, GM_ADDR matB, GM_ADDR matC, GM_ADDR workspace, GM_ADDR tilingGm)
{
    using A_T = half;
    using B_T = half;
    using C_T = float;
    bool isBias = false;

    AscendC::TPipe pipe;
    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);

    AscendC::GlobalTensor<A_T> aGlobal;
    AscendC::GlobalTensor<B_T> bGlobal;
    AscendC::GlobalTensor<C_T> cGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(matA), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(matB), tiling.Ka * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(matC), tiling.M * tiling.N);

    int offsetA = 0;
    int offsetB = 0;
    int offsetC = 0;
    bool isTransA = false;
    bool isTransB = false;

    int tailM = 0;
    int tailN = 0;
    CalcGMOffset(GetBlockIdx(), tiling, offsetA, offsetB, offsetC, tailM, tailN, isTransA, isTransB);

    auto gmA = aGlobal[offsetA];
    auto gmB = bGlobal[offsetB];
    auto gmC = cGlobal[offsetC];

    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_T>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_T>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_T>> mm;

    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);

    mm.SetTensorA(gmA, isTransA);
    mm.SetTensorB(gmB, isTransB);
    mm.SetTail(tailM, tailN);
    // Bias 累加能力由本地 SDK 决定，当前默认保守路径（A*B），Host 侧可做回加兜底。
    (void)isBias;
    mm.IterateAll(gmC);
    mm.End();
}
