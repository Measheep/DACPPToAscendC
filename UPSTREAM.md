# Upstream Sources

DACPPToAscendC is maintained as a modified LLVM monorepo. The repository keeps
the full upstream layout because the translator is built as a Clang tool; LLVM
top-level projects must not be removed without a clean configure, build, and
test proving that the reduced tree is complete.

## LLVM

- Project: `llvm/llvm-project`
- Release tag: `llvmorg-19.1.4`
- Commit: `aadaa00de76ed0c4987b97450dd638f63a385bed`
- Upstream URL: <https://github.com/llvm/llvm-project>
- Local license: `LICENSE.TXT` (Apache License 2.0 with LLVM Exceptions)

The LLVM, Clang, and MLIR directories retain their upstream notices and file
history conventions. DACPP-specific changes do not replace those notices.

## DACPP-OpenSource

- Project: `XJTU-HPC/DACPP-OpenSource`
- Reference release: `v0.1.0`
- Commit: `057618f234dbfd5078d12c54991bf4e50e302978`
- Upstream URL: <https://github.com/XJTU-HPC/DACPP-OpenSource>
- License: Apache License 2.0 with LLVM Exceptions

This reference identifies the public DACPP source line from which the parser,
tensor syntax, and translator structure were adapted. The imported workspace
arrived with an empty `.git` directory, so the original fork-point metadata is
not recoverable from this copy. The commit above is therefore the declared
public compatibility and provenance reference, not a claim that every local
file is byte-identical to that revision.

## Local Changes

The principal local work replaces the SYCL-oriented generation path with an
AscendC backend, including Vector and Cube code generation, structured
HostFlow emission, runtime packaging, tiling, and single-/multi-card source
generation. See `clang/tools/translator/ARCHITECTURE.md` for the current design
and `THIRD_PARTY_NOTICES.md` for additional sample code.

