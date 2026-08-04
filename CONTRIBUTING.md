# Contributing

Thank you for contributing to DACPPToAscendC. By participating, you agree to
follow `CODE_OF_CONDUCT.md` and to preserve all third-party license notices.

## Before Opening a Change

- Search existing issues and pull requests.
- Keep changes focused and describe the affected DACPP/AscendC behavior.
- Do not commit credentials, CANN SDK binaries, build directories, logs, or
  local editor/AI state.
- Do not replace LLVM or Huawei copyright headers with project-wide headers.
- Record any new third-party source in `THIRD_PARTY_NOTICES.md`.

## Build

Use an out-of-source build directory:

```bash
cmake -S llvm -B /tmp/dacpp-build -G Ninja \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON
cmake --build /tmp/dacpp-build \
  --target translator vector_tiling_test host_runtime_emitter_test \
           ascendc_flow_plan_test -j8
```

## Tests

```bash
ctest --test-dir /tmp/dacpp-build/tools/clang/tools/translator \
  --output-on-failure
BUILD_DIR=/tmp/dacpp-build \
  bash clang/tools/translator/tools/atest_diff.sh
BUILD_DIR=/tmp/dacpp-build \
  bash clang/tools/translator/tools/test_vector_tiling.sh
BUILD_DIR=/tmp/dacpp-build \
  bash clang/tools/translator/tools/atest_hostflow_arch_gate.sh
bash clang/tools/translator/tools/release_audit.sh
```

Add or update focused tests for behavior changes. Generated baselines must be
produced by the rebuilt translator; do not hand-edit them to hide differences.

## Pull Requests

Explain the motivation, implementation, verification commands, platform
assumptions, and any generated-file changes. A reviewer must be able to
distinguish tested device behavior from source-generation-only validation.
