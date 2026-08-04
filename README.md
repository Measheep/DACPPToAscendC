# DACPP

DACPP is a source-to-source translation project built on top of LLVM/Clang. It allows domain users to describe data organization and computation in DACPP C++ and translates the source into AscendC host, kernel, and tiling code.

The main translator lives in:

```text
clang/tools/translator
```

The repository is based on the LLVM monorepo layout and currently supports the AscendC backend only. Most upstream LLVM components remain in place because the translator is built as a Clang tool. The current source base is LLVM/Clang 19.1.4.

## Features

- Parses DACPP Shell/Calc expressions with Clang AST matchers.
- Lowers Calc semantics through the optional MLIR analysis pipeline.
- Generates AscendC Vector and Cube kernels.
- Generates structured host execution code through `AscendCFlowPlan` and HostFlow.
- Supports single-card and host-orchestrated multi-card generation.
- Plans Vector tiling from the available Unified Buffer budget.
- Provides representative applications for vector operations, stencil computation, image processing, matrix multiplication, and machine learning.
- Includes regression checks for Vector tiling, generated host code, and HostFlow architecture contracts.

## Project Goals

The project aims to migrate representative applications such as matrix computation, heat-conduction/stencil computation, and machine learning to DACPP, allowing domain users to write serial-style application code while the generated program runs in parallel on Ascend devices.

The challenge evaluation targets are:

- At least 80% of the performance of a comparable hand-written Ascend CANN implementation on single-card and multi-card platforms.
- Fewer than 60% of the source lines of the corresponding CANN implementation.

These are project evaluation targets, not unconditional performance guarantees for every device, input size, or CANN release.

## Dependencies

### Build Dependencies

To build the translator from source, install:

- CMake 3.20 or newer.
- Ninja, or another CMake-supported build tool.
- A C++17-capable host compiler.
- Python 3, required by the LLVM/Clang build system and project checks.
- Git.

### Runtime And Compilation Dependencies

Translating DACPP source does not require an Ascend device. To compile and run the generated AscendC code, install:

- A Huawei Ascend CANN/AscendC development environment compatible with the target device.
- The required Ascend runtime and development headers.
- One or more supported Ascend devices for device execution.

Use the compilation and deployment procedure for the installed CANN release;
the project does not currently wrap the CANN build process in a public script.

## Compatibility Status

| Component | Status |
| --- | --- |
| LLVM/Clang/MLIR | Source base and translator build verified against 19.1.4. |
| Host compiler | C++17 translator build; generated host runtime is checked as C++11. |
| CANN | No exact CANN release has been recorded as device-validated for this source snapshot. |
| Ascend devices | No physical device model has been recorded as validated for this source snapshot. |
| Single-card | Source generation, generated-host syntax, and fake-runtime failure tests are automated. |
| Multi-card | Eight-rank source generation and fake-runtime host checks are automated; physical multi-device execution remains experimental. |

Current coverage includes typed host declarations and control flow, Vector
pointwise operations, scalar/broadcast parameters, one-dimensional stencil and
halo paths, Cube matrix multiplication, runtime tiling, and the eight samples
listed below. MLIR-assisted lowering, collective communication, temporal
stencil scheduling, and production CANN/device integration should be treated as
experimental until validated on the intended CANN release and hardware.

## Build

Configure the LLVM/Clang/MLIR build from the repository root:

```bash
cmake -S llvm -B /tmp/dacpp-build -G Ninja \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON
```

Then build the DACPP translator:

```bash
cmake --build /tmp/dacpp-build --target translator -j8
```

The translator executable is written to:

```text
/tmp/dacpp-build/bin/translator/translator
```

MLIR is optional. A Clang-only build can still use the legacy AscendC path:

```bash
cmake -S llvm -B /tmp/dacpp-build -G Ninja \
  -DLLVM_ENABLE_PROJECTS="clang" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release
```

Do not reuse a CMake build directory after moving the repository, because the CMake cache records absolute source and output paths.

## Environment

The interactive environment helper is located at:

```text
clang/tools/translator/env.sh
```

Point the helper at the external build directory, then source it:

```bash
cd clang/tools/translator
export DACPP_BUILD_DIR=/tmp/dacpp-build
source env.sh
```

This defines a shell function named `dacpp`. The function calls the translator
from `DACPP_BUILD_DIR` and automatically supplies the DACPP include paths and
C++17 frontend option.

Common translation-time environment variables:

| Variable | Purpose |
| --- | --- |
| `DACPP_ASCENDC_USE_MLIR` | Enable the MLIR-based analysis path. |
| `DACPP_ASCENDC_CORE_NUM` | AI Core count launched per Ascend device. |
| `DACPP_ASCENDC_MULTICARD` | Enable multi-card host/kernel generation. |
| `DACPP_ASCENDC_RANK_DIM` | Requested device/rank count. |
| `DACPP_ASCENDC_RANK_ENV` | Runtime environment variable used for the active rank count. |
| `DACPP_ASCENDC_UB_BUDGET_BYTES` | Vector Unified Buffer budget in bytes. |
| `DACPP_ASCENDC_UB_RESERVE_BYTES` | Reserved Unified Buffer space in bytes. |
| `DACPP_ASCENDC_BUFFER_NUM` | Vector queue depth: `auto`, `1`, or `2`. |
| `DACPP_ASCENDC_MAX_TILE_ELEMENTS` | Maximum Vector tile element count. |

Command-line options take precedence over equivalent translation-time environment variables.

## Translate

After sourcing `env.sh`, enter a sample directory and translate its DACPP source:

```bash
cd tests/t1add
dacpp add.dac.cpp
```

The default backend is AscendC, so `--mode=ascendc` is optional. The equivalent explicit command is:

```bash
dacpp add.dac.cpp --mode=ascendc
```

Useful translator options include:

| Option | Purpose |
| --- | --- |
| `--ascendc-use-mlir` | Enable MLIR-based Calc analysis. |
| `--ascendc-core-num=<N>` | Set the AI Core count per device; the default is 8. |
| `--ascendc-multicard` | Generate multi-card host orchestration; the default request is 8 devices. |
| `--ascendc-rank-dim=<N>` | Override the requested multi-card device count. |
| `--ascendc-rank-env=<NAME>` | Select the runtime rank-count environment variable. |
| `--ascendc-dump-flow-plan` | Print the structured HostFlow plan to standard output. |
| `--ascendc-host-log=<LEVEL>` | Set generated host logging to `none`, `error`, or `debug`. |
| `--debug` | Print parser, matched AST, and code-generation diagnostics. |
| `--dump-parser` | Alias for `--debug`. |

## Generated Files

Generated files are written to the current sample working directory:

| Output | Purpose |
| --- | --- |
| `host0.cpp`, `host1.cpp`, ... | Generated AscendC host programs. |
| `kernel0.cpp`, `kernel1.cpp`, ... | Generated Vector or Cube kernels. |
| `tiling.cpp` | Cube-route tiling implementation, when required. |
| `dacpp_ascendc_host_runtime.h` | Shared generated host runtime. |
| `ascendc_flow_plan.txt` | Structured HostFlow plan. |
| `<input>.dac_translated.cpp` | Clang-rewritten DACPP source. |

The exact set of outputs depends on the application and selected execution route.

## Multi-Card Translation

Generate a four-device version of the vector-add sample:

```bash
dacpp add.dac.cpp \
  --ascendc-use-mlir \
  --ascendc-multicard \
  --ascendc-rank-dim=4
```

With `--ascendc-multicard` and no explicit rank count, the generated program requests eight devices. The generated host validates device availability and partitions the workload across active ranks. Multi-card generation is host-orchestrated; collective communication is emitted only when required by an explicit communication plan.

## Compile And Run Generated Code

The translator produces source files but does not compile or deploy them. Build the generated host, kernel, and optional tiling sources with the toolchain and project layout required by the installed CANN release, then run the resulting program on the target Ascend platform.

Keep these generated files in the same package unless the CANN build project is updated accordingly:

```text
host*.cpp
kernel*.cpp
tiling.cpp                    # Cube route only
dacpp_ascendc_host_runtime.h
```

## Examples

The sample applications are under `clang/tools/translator/tests`:

| Directory | Application |
| --- | --- |
| `t1add` | Vector addition. |
| `t2decay` | Isotope decay computation. |
| `t3chafen` | One-dimensional difference. |
| `t4liuliang` | LWR traffic-flow computation. |
| `t5l3pt` | Three-point stencil computation. |
| `t6image` | Image processing. |
| `t7matmul` | Matrix multiplication and Cube generation. |
| `t8linear` | Linear layer. |

## Tests

Build and run the CTest targets:

```bash
cmake --build /tmp/dacpp-build \
  --target translator vector_tiling_test host_runtime_emitter_test \
           ascendc_flow_plan_test \
  -j8
ctest --test-dir /tmp/dacpp-build/tools/clang/tools/translator \
  --output-on-failure
```

Run the translation and architecture checks:

```bash
BUILD_DIR=/tmp/dacpp-build \
  bash clang/tools/translator/tools/test_vector_tiling.sh
BUILD_DIR=/tmp/dacpp-build \
  bash clang/tools/translator/tools/atest_hostflow_arch_gate.sh
BUILD_DIR=/tmp/dacpp-build \
  bash clang/tools/translator/tools/atest_diff.sh
bash clang/tools/translator/tools/release_audit.sh
```

`atest_diff.sh` compares all committed single-card outputs and additionally
checks multi-card outputs for any case that contains committed multi-card
baselines.

## More Documentation

See the [translator guide](clang/tools/translator/README.md),
[architecture notes](clang/tools/translator/ARCHITECTURE.md),
[upstream record](UPSTREAM.md), and
[third-party notices](THIRD_PARTY_NOTICES.md) for implementation details,
source provenance, and the full processing flow.

Upstream LLVM documentation remains in component subdirectories because DACPP is built on the LLVM/Clang source tree.

## Contributing And Security

- See [CONTRIBUTING.md](CONTRIBUTING.md) before submitting an issue or pull request.
- Follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) in all project communication.
- Report vulnerabilities privately according to [SECURITY.md](SECURITY.md).
- See [SUPPORT.md](SUPPORT.md) for bug-reporting and support requirements.

## License

This repository follows the LLVM project license: Apache License 2.0 with LLVM
Exceptions. See [LICENSE.TXT](LICENSE.TXT), [NOTICE](NOTICE), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
