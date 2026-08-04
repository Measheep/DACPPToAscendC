# Third-Party Notices

This file records third-party material shipped in DACPPToAscendC. It does not
replace the license text in `LICENSE.TXT` or copyright notices retained in
individual source files.

## LLVM Project

DACPPToAscendC contains a modified LLVM monorepo, including LLVM, Clang, MLIR,
and related subprojects.

- Upstream: <https://github.com/llvm/llvm-project>
- Version: `llvmorg-19.1.4`
- Commit: `aadaa00de76ed0c4987b97450dd638f63a385bed`
- License: Apache License 2.0 with LLVM Exceptions
- License text: `LICENSE.TXT`

Copyright and attribution notices in upstream LLVM files are retained.

## DACPP-OpenSource

The DACPP parser, tensor syntax, and original translator structure are derived
from DACPP-OpenSource.

- Upstream: <https://github.com/XJTU-HPC/DACPP-OpenSource>
- Reference release: `v0.1.0`
- Commit: `057618f234dbfd5078d12c54991bf4e50e302978`
- License: Apache License 2.0 with LLVM Exceptions

See `UPSTREAM.md` for the provenance limitation caused by the missing original
Git metadata.

## Huawei Ascend Samples

The following matrix-multiplication test files originate from or are modified
from Huawei Ascend Samples:

- `clang/tools/translator/tests/t7matmul/matmul_main.cpp` (modified)
- `clang/tools/translator/tests/t7matmul/data_utils.h` (unmodified at the
  recorded revision)
- `clang/tools/translator/tests/t7matmul/matmul_custom.cpp` (modified)
- `clang/tools/translator/tests/t7matmul/matmul_custom_tiling.cpp` (modified)

Recorded source:

- Project: Ascend Samples
- Canonical repository: <https://gitee.com/ascend/samples>
- GitHub mirror: <https://github.com/Huawei-Ascend/samples>
- Directory:
  `operator/ascendc/0_introduction/11_matmul_kernellaunch/MatmulInvocationNeo`
- Commit: `6511a5f4a45a1f68bd5e617989e68560f2f35cd6`
- License: Apache License 2.0
- Upstream notice: `Ascend Samples; Copyright 2019-2020 Huawei Technologies
  Co., Ltd`

The Huawei copyright headers are retained. Modified files carry an explicit
modification notice near the original header.

