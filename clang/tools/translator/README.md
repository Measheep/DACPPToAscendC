# DACPP to AscendC Source-to-Source Translator

## Overview

This tool translates DACPP source code into AscendC host, kernel, and tiling code. The translator is implemented as a Clang-based source-to-source tool. It extracts DACPP expressions from the Clang AST, builds parser-level Shell/Calc structures, lowers Calc semantics through the `ascendc_mlir` analysis flow, and emits AscendC runtime code through the AscendC rewriter.

The host side is generated from a source-order, structured `AscendCFlowPlan`. Non-DAC host statements are reconstructed from typed AST descriptors, while DAC expression slots are replaced by generated Vector or Cube execution blocks; raw source replay is not an emission path.

## Build Steps

```shell
git clone https://github.com/Measheep/DACPPToAscendC.git
cd DACPPToAscendC
cmake -S llvm -B /tmp/dacpp-build -G Ninja \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON
cmake --build /tmp/dacpp-build --target translator -j8
```

Replace the clone placeholder with the URL shown by the hosting service after
the clean repository is created. The translator binary is generated at
`/tmp/dacpp-build/bin/translator/translator`.

Load the helper function, enter a sample directory, and translate its DACPP source:

```shell
cd clang/tools/translator
export DACPP_BUILD_DIR=/tmp/dacpp-build
source env.sh
cd tests/t1add
dacpp add.dac.cpp
```

## Directory Structure Description

```text
├── README.md                         // Project entry documentation
├── CMakeLists.txt                    // Translator build configuration
├── translator.cpp                    // Clang tool entry and ASTMatcher callbacks
├── ARCHITECTURE*.md                  // Architecture documentation
├── env.sh                            // Local environment configuration
├── std_lib                           // Minimal standard-library headers used by the translator
│   └── include
├── dacppLib                          // DACPP syntax and tensor data structures
│   ├── include
│   │   ├── Slice.h                   // Slice syntax support
│   │   ├── ReconTensor.h             // Tensor reconstruction interface
│   │   ├── FuncTensor.hpp            // Functional tensor representation
│   │   └── exception.h               // DACPP exception declarations
│   └── lib
│       └── exception.cpp             // DACPP exception implementation
├── parser                            // DACPP AST parsing and intermediate structures
│   ├── include
│   │   ├── ASTParse.h                // AST source extraction and helper analysis
│   │   ├── DacppStructure.h          // DacppFile, Expression, Shell, Calc structures
│   │   ├── Param.h                   // Parameter metadata
│   │   ├── Split.h                   // Split and index partition metadata
│   │   ├── Shell.h                   // Shell parser interface
│   │   ├── Calc.h                    // Calc parser interface
│   │   └── Dacfor.h                  // dac_for control metadata
│   └── lib
│       ├── ASTParse.cpp              // AST text extraction and parameter direction analysis
│       ├── DacppStructure.cpp        // Expression construction and parser result assembly
│       ├── Param.cpp                 // Parameter type, shape, and rw metadata
│       ├── Split.cpp                 // Split/index split representation
│       ├── Shell.cpp                 // Shell data layout and binding parser
│       ├── Calc.cpp                  // Calc body and expression parser
│       └── Dacfor.cpp                // dac_for control block support
├── rewriter                          // AscendC code generation
│   ├── include
│   │   ├── Rewriter.h                // Generic rewriter interface
│   │   ├── ascendc_template.h        // General AscendC template declarations
│   │   ├── universal_template.h      // Shared text replacement helper
│   │   ├── ascendc                   // AscendC planning and host-flow interfaces
│   │   └── ascendc_mlir              // Calc-to-MLIR analysis and planning interfaces
│   └── lib
│       ├── Rewriter_AscendC.cpp      // Thin AscendC backend entry; Vector/Cube routing lives under ascendc/
│       ├── ascendc_template.cpp      // General AscendC code templates
│       ├── universal_template.cpp    // Shared text replacement helper
│       ├── ascendc                   // AscendC planning and host-flow implementation
│       └── ascendc_mlir              // Calc lowering, semantic analysis, plan, and emitter
│           ├── CalcContext.cpp       // Calc semantic context extraction
│           ├── CalcShellInfo.cpp     // Shell summary construction
│           ├── CalcAstToMlir.cpp     // Calc AST to MLIR lowering
│           ├── CalcExprGraph.cpp     // MLIR expression graph construction
│           ├── CalcAnalysis.cpp      // Semantic classification
│           ├── AscendCPlan.cpp       // AscendC plan construction
│           ├── AscendCEmitter.cpp    // AscendC kernel code emission
│           └── CalcMlirPipeline.cpp  // MLIR analysis flow orchestration
├── tests                             // Functional samples and generated AscendC outputs
│   ├── t1add                         // Vector addition sample
│   ├── t2decay                       // Isotope decay sample
│   ├── t3chafen                      // One-dimensional difference sample
│   ├── t4liuliang                    // LWR traffic-flow sample
│   ├── t5l3pt                        // Three-point stencil sample
│   ├── t6image                       // Image-processing sample
│   ├── t7matmul                      // Matrix multiplication and Cube route sample
│   └── t8linear                      // Linear layer sample
└── tools                             // Regression and audit scripts
    ├── atest_diff.sh                 // Optional comparison when all output baselines are present
    ├── atest_host_compact_gate.py    // Host runtime and naming contract checks
    ├── atest_hostflow_arch_gate.sh   // HostFlow architecture checks
    ├── release_audit.sh              // Release content and provenance checks
    └── test_vector_tiling.sh         // Vector tiling regression checks
```

## Generated Files

The AscendC backend generates the following files for typical samples:

- `host0.cpp`, `host1.cpp`, ...: generated host programs.
- `kernel0.cpp`, `kernel1.cpp`, ...: generated Vector/Cube kernels with a shared global index.
- `tiling.cpp`: Cube-route tiling implementation.
- `dacpp_ascendc_host_runtime.h`: shared generated host runtime.
- `ascendc_flow_plan.txt`: structured HostFlow plan.
- `<input>.dac_translated.cpp`: rewritten source file.

## Multi-Card Options

- `--ascendc-core-num=<N>`: set the AI Core launch count per device. The default is 8.
- `--ascendc-multicard`: generate host-orchestrated multi-card AscendC code. With no rank override it requests 8 devices, each launching 8 AI Cores.
- `--ascendc-rank-dim=<N>`: override the requested rank/device count in multi-card mode. Single-card mode always uses one device; explicit `N=1` keeps the multi-card artifact and its synchronous debug fast path.
- `--ascendc-rank-env=<NAME>`: choose the runtime environment variable read by generated host code for rank count.
- `DACPP_ASCENDC_CORE_NUM`, `DACPP_ASCENDC_MULTICARD`, `DACPP_ASCENDC_RANK_DIM`, and `DACPP_ASCENDC_RANK_ENV` provide equivalent translation-time overrides; command-line values take precedence. The selected rank environment is also read by the generated host at runtime and overrides its compiled fallback.

The generated multi-card host distinguishes the compiled/requested rank count from the active rank count. It reports an error when fewer devices are available than requested, while a shard extent smaller than the request may reduce only the active workers. A route owns one persistent rank executor: each rank creates its context and stream once, reuses capacity-cached device buffers and contiguous pinned host slabs across launches, and uses ordered asynchronous H2D/launch/D2H operations. Ordinary shard strategies do not initialize HCCL; collective support is selected only by an explicit communication plan.

## Diagnostics

Normal translation prints errors, topology/route decisions, and generated-file
notifications. Parser structures, Calc bodies, matched ASTs, detailed tiling
allocation, and other internal diagnostics are disabled by default. Use
`--debug` (or its `--dump-parser` alias) only when those details are required;
debug logs can reveal user source structure.

## Main Processing Flow

```text
Clang AST
-> parser Shell/Calc extraction
-> DacppFile / Expression
-> Calc MLIR lowering and semantic analysis
-> AscendCPlan
-> Vector or Cube kernel generation
-> structured AscendCFlowPlan / HostFlow emission
-> AscendC host / kernel / tiling files
```
