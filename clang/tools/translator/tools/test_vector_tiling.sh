#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$(cd "$PROJ_DIR/../../../build" && pwd)}"
TRANSLATOR="${TRANSLATOR:-$BUILD_DIR/bin/translator/translator}"
CXX="${CXX:-g++}"

fail() { echo "[vector-tiling-test][FAIL] $*" >&2; exit 1; }
note() { echo "[vector-tiling-test] $*"; }

[[ -x "$TRANSLATOR" ]] || fail "translator not found: $TRANSLATOR"

work="$(mktemp -d -t vector-tiling-test.XXXXXX)"
trap 'rm -rf "$work"' EXIT

"$CXX" -std=c++17 -I"$PROJ_DIR/rewriter/include" \
  "$PROJ_DIR/rewriter/tests/VectorTilingTest.cpp" \
  "$PROJ_DIR/rewriter/lib/ascendc/VectorTiling.cpp" \
  -o "$work/vector_tiling_test"
"$work/vector_tiling_test"

run_case() {
  local name="$1"; shift
  local source="$1"; shift
  local out="$work/$name"
  mkdir -p "$out"
  cp "$source" "$out/"
  local base
  base="$(basename "$source")"
  (cd "$out" && "$TRANSLATOR" "$base" --mode=ascendc "$@" \
    -extra-arg=-std=c++17 -- \
    -I"$PROJ_DIR/std_lib/include" -I"$PROJ_DIR/dacppLib/include" \
    >stdout.log 2>stderr.log)
}

source_add="$PROJ_DIR/tests/t1add/add.dac.cpp"
run_case default "$source_add"
default="$work/default"
grep -q 'constexpr int32_t BUFFER_NUM = 2;' "$default/kernel0.cpp" || fail "auto mode did not select double buffering"
grep -q 'uint32_t tileLength, uint32_t tileCount' "$default/kernel0.cpp" || fail "kernel ABI lacks runtime tile parameters"
grep -q 'currentTileLength' "$default/kernel0.cpp" || fail "tail tile control flow missing"
if grep -q 'TILE_NUM' "$default/kernel0.cpp"; then fail "fixed TILE_NUM remains"; fi
grep -q 'bytes per tile estimate:' "$default/kernel0.cpp" || fail "kernel diagnostics missing"
grep -q 'tileLength0, tileCount0' "$default/host0.cpp" || fail "host launch ABI mismatch"
grep -q 'ICPU_RUN_KF(vector_kernel_0.*tileLength0, tileCount0' "$default/host0.cpp" || fail "CPU Debug launch argument order mismatch"
if grep -Eq 'Gm\[[^]]*progress[[:space:]]*\*[[:space:]]*tileLength|AscendC::(DataCopy|Add|Sub|Mul|Muls|Adds|Duplicate|Max)\([^;]*,[[:space:]]*tileLength\);' "$default/kernel0.cpp"; then
  fail "execution path still uses fixed tileLength instead of tileOffset/currentTileLength"
fi

mkdir -p "$work/fake_cpu"
cat >"$work/fake_cpu/tikicpulib.h" <<'EOF'
#ifndef DACPP_FAKE_TIKICPULIB_H
#define DACPP_FAKE_TIKICPULIB_H
#include <cstddef>
#include <cstdint>
#define __global__
#define __aicore__
using GM_ADDR = uint8_t*;
namespace AscendC {
inline void* GmAlloc(std::size_t) { return nullptr; }
inline void GmFree(void*) {}
}
#define ICPU_RUN_KF(kernel, blockDim, ...) ((void)0)
#endif
EOF
"$CXX" -std=c++17 -DASCENDC_CPU_DEBUG -I"$work/fake_cpu" \
  -fsyntax-only "$default/host0.cpp" || fail "CPU Debug host syntax/ABI check failed"
check_cpu_host() {
  local case_name="$1"
  "$CXX" -std=c++17 -DASCENDC_CPU_DEBUG -I"$work/fake_cpu" \
    -fsyntax-only "$work/$case_name/host0.cpp" || \
    fail "CPU Debug host syntax/ABI check failed for $case_name"
}

run_case low "$source_add" --ascendc-ub-budget-bytes=32768 --ascendc-ub-reserve-bytes=4096
default_max="$(sed -n 's/.*MAX_TILE_ELEMENTS = \([0-9][0-9]*\).*/\1/p' "$default/kernel0.cpp")"
low_max="$(sed -n 's/.*MAX_TILE_ELEMENTS = \([0-9][0-9]*\).*/\1/p' "$work/low/kernel0.cpp")"
[[ -n "$default_max" && -n "$low_max" && "$low_max" -le "$default_max" ]] || fail "max tile is not monotonic with UB budget"

run_case explicit "$source_add" --ascendc-buffer-num=1 --ascendc-max-tile-elements=1000
grep -q 'constexpr int32_t BUFFER_NUM = 1;' "$work/explicit/kernel0.cpp" || fail "explicit single buffer ignored"
grep -q 'aligned-down 1000->960' "$work/explicit/kernel0.cpp" || fail "tile override alignment diagnostic missing"

DACPP_ASCENDC_UB_BUDGET_BYTES=120000 \
run_case precedence "$source_add" --ascendc-ub-budget-bytes=100000
grep -q 'UB budget: 100000 bytes' "$work/precedence/kernel0.cpp" || fail "CLI did not override environment"

if run_case invalid "$source_add" --ascendc-buffer-num=invalid --ascendc-ub-budget-bytes=bad; then
  fail "invalid tiling configuration unexpectedly translated"
fi
grep -q 'invalid --ascendc-ub-budget-bytes' "$work/invalid/stderr.log" || fail "first invalid setting was not reported"

if run_case impossible "$source_add" --ascendc-ub-budget-bytes=700 --ascendc-ub-reserve-bytes=100; then
  fail "impossible minimum tile unexpectedly translated"
fi
grep -q 'cannot fit one aligned tile' "$work/impossible/stderr.log" || fail "resource failure lacks detail"

run_case linear "$PROJ_DIR/tests/t8linear/linear.dac.cpp"
if grep -Eq 'InitBuffer\([^\n]*totalLength' "$work/linear/kernel0.cpp"; then
  fail "linear fast path still allocates totalLength in UB"
fi
grep -q 'currentTileLength' "$work/linear/kernel0.cpp" || fail "linear path is not tiled"
grep -q 'kernelSize0 = outputBuf.logicalElems;' "$work/linear/host0.cpp" || fail "linear runtime length is not output-driven"
grep -q 'weightBuf.data.resize' "$work/linear/host0.cpp" || fail "linear weight padding is missing"
check_cpu_host linear

run_case halo "$PROJ_DIR/tests/t5l3pt/3pt.dac.cpp"
grep -q 'currentTileLength' "$work/halo/kernel0.cpp" || fail "halo/structured path lacks tail length"
if grep -Eq 'SetGlobalBuffer\([^;]*\+ [A-Za-z_]+GmStart' "$work/halo/kernel0.cpp" && \
   ! grep -q 'GmStart < 0' "$work/halo/kernel0.cpp"; then
  fail "halo/structured path can retain a negative GM start"
fi
check_cpu_host halo

run_case broadcast "$PROJ_DIR/tests/t2decay/decay.dac.cpp"
grep -Eq 'tGm.SetGlobalBuffer\([^;]*, 1\);' "$work/broadcast/kernel0.cpp" || fail "broadcast scalar GM view is not one element"
grep -q 'tGm.GetValue(0)' "$work/broadcast/kernel0.cpp" || fail "broadcast scalar is not read as one GM element"
grep -q 'fill < static_cast<int32_t>(currentTileLength)' "$work/broadcast/kernel0.cpp" || fail "broadcast fill ignores tail length"
check_cpu_host broadcast

run_case multicard "$source_add" --ascendc-multicard --ascendc-use-mlir
grep -q 'dacpp_host::readRankCount(app, "DACPP_ASCENDC_RANK_DIM", 8,' "$work/multicard/host0.cpp" || fail "multicard default rank is not 8"
grep -q 'dacpp_host::RankSpan ranks' "$work/multicard/host0.cpp" || fail "rank span missing"
grep -q 'tile.len, blockDim0, tile.size, tile.count' "$work/multicard/host0.cpp" || fail "rank launch ABI mismatch"
grep -q 'dacpp_host::makeTile(app, r.len' "$work/multicard/host0.cpp" || fail "rank does not use shared runtime tiling helper"

note "PASS"
