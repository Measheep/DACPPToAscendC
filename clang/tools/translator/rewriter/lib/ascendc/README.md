# AscendC Rewriter Implementation

The active AscendC backend entrypoint is:

- Public wrapper: `rewriter/lib/Rewriter_AscendC.cpp`
- Internal driver: `rewriter/lib/ascendc/AscendCRewriteDriver.cpp`
- Public driver interface: `rewriter/include/ascendc/AscendCRewriteDriver.h`

This directory no longer uses `.inc` implementation fragments. The current
driver keeps the previous behavior while the low-coupling pieces move into real
`.h/.cpp` modules:

- `AscendCArtifacts.h`: per-expression route artifacts (kernel/tiling text,
  user prelude, runtime/name metadata, and structured `FlowKernelLaunch` /
  `FlowGeneratedCodeBlock` descriptors); it does not carry file-level runtime
  type text.
- `HostTensorAliasInfo.h`: host-side tensor alias descriptor (shape/stride/offset), shared between driver, HostFlowBuilder and HostFlowEmitter.
- `AscendCPolicy.h/.cpp`: translation policy and rewrite context.
- `AscendCRuntimeBinding.h/.cpp`: runtime binding shared types, tensor/record layout helpers, and alias key helpers.
- `AscendCSourceUtils.h/.cpp`: source/text helpers and simple constant resolution.
- `AscendCTensorAlias.h/.cpp`: tensor alias parsing, rank/type extraction, stride/offset helpers, index application, AST alias resolution, and live alias collection.
- `AscendCTypeMapping.h/.cpp`: host/kernel/tiling dtype mapping.
- `AscendCKernelSupport.h/.cpp`: MLIR pipeline facade and kernel support helpers.
- `HostCodeNames.h/.cpp`: deterministic short-name allocation, visible-name
  collision avoidance, reserved-identifier checks, and name audit records.
- `HostRuntimeFeatures.h`: feature bits and compact/embedded runtime packaging
  requirements carried by expression artifacts and the flow plan.
- `HostRuntimeEmitter.h/.cpp`: the only implementation of the versioned
  C++11 `dacpp_host` runtime (`Session`, `Device`, `Route`, buffers and tiling).

The driver routes Vector/Cube lowering through artifact-returning helper
functions, and delegates tensor alias resolution/collection to the tensor alias
module. Host body emission is fully driven by `HostFlowBuilder` →
`AscendCFlowPlan` → `HostFlowEmitter` →
`HostRuntimeEmitter`/the thin file template. The driver may lower Vector/Cube
semantics, but it must not add another public executor, raw ACL lifecycle, or
name-allocation policy. Future maintenance should continue
moving the remaining generator class bodies (Vector / Cube lowering) into
dedicated compilation units.
