#!/usr/bin/env bash
# HostFlow architecture and compact-runtime gate.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$(cd "${PROJ_DIR}/../../../build" && pwd)}"
TRANSLATOR="${TRANSLATOR:-${BUILD_DIR}/bin/translator/translator}"
STD_LIB="${PROJ_DIR}/std_lib/include"
DACPP_LIB="${PROJ_DIR}/dacppLib/include"
CASES="${ARCH_GATE_CASES:-t1add t2decay t3chafen t4liuliang t5l3pt t6image t7matmul t8linear}"
WORK_ROOT="$(mktemp -d -t hostflow_gate.XXXXXX)"
trap 'rm -rf "${WORK_ROOT}"' EXIT
failed=0

fail() { echo "[hostflow-gate][FAIL] $*" >&2; failed=1; }
note() { echo "[hostflow-gate] $*"; }

if [[ ! -x "${TRANSLATOR}" ]]; then
  echo "[hostflow-gate][FAIL] translator unavailable: ${TRANSLATOR}" >&2
  exit 2
fi

make_fake_sdk() {
  local root=$1
  mkdir -p "${root}/acl" "${root}/kernel_tiling" "${root}/tiling/platform"
  cat >"${root}/acl/acl.h" <<'EOF'
#ifndef DACPP_FAKE_ACL_H
#define DACPP_FAKE_ACL_H
#include <cstddef>
#include <cstdint>
using aclError = int;
using aclFloat16 = uint16_t;
using aclrtContext = void*;
using aclrtStream = void*;
constexpr aclError ACL_SUCCESS = 0;
constexpr int ACL_MEM_MALLOC_HUGE_FIRST = 0;
constexpr int ACL_MEMCPY_HOST_TO_DEVICE = 0;
constexpr int ACL_MEMCPY_DEVICE_TO_HOST = 1;
constexpr int ACL_MEMCPY_DEVICE_TO_DEVICE = 2;
constexpr int ACL_MEMCPY_HOST_TO_HOST = 3;
inline aclError aclInit(const char*) { return 0; }
inline aclError aclFinalize() { return 0; }
inline aclError aclrtGetDeviceCount(uint32_t* n) { *n = 8; return 0; }
inline aclError aclrtSetDevice(int32_t) { return 0; }
inline aclError aclrtResetDevice(int32_t) { return 0; }
inline aclError aclrtCreateContext(aclrtContext* p, int32_t) { *p = nullptr; return 0; }
inline aclError aclrtDestroyContext(aclrtContext) { return 0; }
inline aclError aclrtSetCurrentContext(aclrtContext) { return 0; }
inline aclError aclrtCreateStream(aclrtStream* p) { *p = nullptr; return 0; }
inline aclError aclrtDestroyStream(aclrtStream) { return 0; }
inline aclError aclrtSynchronizeStream(aclrtStream) { return 0; }
inline aclError aclrtMalloc(void** p, std::size_t, int) { *p = reinterpret_cast<void*>(1); return 0; }
inline aclError aclrtFree(void*) { return 0; }
inline aclError aclrtMallocHost(void** p, std::size_t) { *p = reinterpret_cast<void*>(1); return 0; }
inline aclError aclrtFreeHost(void*) { return 0; }
inline aclError aclrtMemcpy(void*, std::size_t, const void*, std::size_t, int) { return 0; }
inline aclError aclrtMemcpyAsync(void*, std::size_t, const void*, std::size_t, int, aclrtStream) { return 0; }
#endif
EOF
  cat >"${root}/kernel_tiling/kernel_tiling.h" <<'EOF'
#ifndef DACPP_FAKE_TILING_H
#define DACPP_FAKE_TILING_H
struct TCubeTiling { int M=1,N=1,Ka=1,Kb=1,singleCoreM=1,singleCoreN=1; };
#endif
EOF
  cat >"${root}/tiling/platform/platform_ascendc.h" <<'EOF'
#ifndef DACPP_FAKE_PLATFORM_H
#define DACPP_FAKE_PLATFORM_H
#define SOC_VERSION "fake"
#include <cstddef>
namespace platform_ascendc {
class PlatformAscendCManager {
public:
 static PlatformAscendCManager* GetInstance(const char*) { static PlatformAscendCManager x; return &x; }
 std::size_t GetLibApiWorkSpaceSize() const { return 0; }
};
}
#endif
EOF
}

make_failure_sdk() {
  local root=$1
  mkdir -p "${root}/acl"
  cat >"${root}/acl/acl.h" <<'EOF'
#ifndef DACPP_FAKE_FAILURE_ACL_H
#define DACPP_FAKE_FAILURE_ACL_H
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
using aclError = int;
using aclFloat16 = uint16_t;
using aclrtContext = void*;
using aclrtStream = void*;
constexpr aclError ACL_SUCCESS = 0;
constexpr int ACL_MEM_MALLOC_HUGE_FIRST = 0;
constexpr int ACL_MEMCPY_HOST_TO_DEVICE = 0;
constexpr int ACL_MEMCPY_DEVICE_TO_HOST = 1;
constexpr int ACL_MEMCPY_DEVICE_TO_DEVICE = 2;
constexpr int ACL_MEMCPY_HOST_TO_HOST = 3;
inline bool dacppFakeStage(const char* stage) {
  const char* value = std::getenv("DACPP_FAKE_STAGE");
  return value && std::strcmp(value, stage) == 0;
}
inline uint32_t dacppFakeLaunchCode() {
  return dacppFakeStage("launch") ? 14u : 0u;
}
inline aclError aclInit(const char*) { return dacppFakeStage("init") ? 11 : 0; }
inline aclError aclFinalize() { return dacppFakeStage("cleanup") ? 17 : 0; }
inline aclError aclrtGetDeviceCount(uint32_t* n) { *n = 8; return 0; }
inline aclError aclrtSetDevice(int32_t) { return 0; }
inline aclError aclrtResetDevice(int32_t) { return 0; }
inline aclError aclrtCreateContext(aclrtContext* p, int32_t) { *p = reinterpret_cast<void*>(1); return 0; }
inline aclError aclrtDestroyContext(aclrtContext) { return 0; }
inline aclError aclrtSetCurrentContext(aclrtContext) { return 0; }
inline aclError aclrtCreateStream(aclrtStream* p) { *p = reinterpret_cast<void*>(1); return 0; }
inline aclError aclrtDestroyStream(aclrtStream) { return 0; }
inline aclError aclrtSynchronizeStream(aclrtStream) { return dacppFakeStage("sync") ? 15 : 0; }
inline aclError aclrtMalloc(void** p, std::size_t n, int) {
  if (dacppFakeStage("alloc")) { *p = nullptr; return 12; }
  *p = std::malloc(n ? n : 1); return *p ? 0 : 12;
}
inline aclError aclrtFree(void* p) {
  std::free(p); return dacppFakeStage("cleanup") ? 17 : 0;
}
inline aclError aclrtMallocHost(void** p, std::size_t n) {
  if (dacppFakeStage("alloc")) { *p = nullptr; return 12; }
  *p = std::malloc(n ? n : 1); return *p ? 0 : 12;
}
inline aclError aclrtFreeHost(void* p) {
  std::free(p); return dacppFakeStage("cleanup") ? 17 : 0;
}
inline aclError aclrtMemcpy(void* dst, std::size_t, const void* src,
                            std::size_t n, int kind) {
  if (kind == ACL_MEMCPY_HOST_TO_DEVICE && dacppFakeStage("copy_in")) return 13;
  if (kind == ACL_MEMCPY_DEVICE_TO_HOST && dacppFakeStage("copy_out")) return 16;
  if (n) std::memcpy(dst, src, n);
  return 0;
}
inline aclError aclrtMemcpyAsync(void* dst, std::size_t cap, const void* src,
                                 std::size_t n, int kind, aclrtStream) {
  return aclrtMemcpy(dst, cap, src, n, kind);
}
#endif
EOF
}

make_launch_headers() {
  local dir=$1 host=$2
  sed -n 's/.*#include "\([^"]*aclrtlaunch_[^"]*\.h\)".*/\1/p' "${host}" |
  while read -r header; do
    [[ -n "${header}" ]] || continue
    local path="${dir}/${header}" kernel
    kernel="$(basename "${header}" .h)"
    kernel="${kernel#aclrtlaunch_}"
    mkdir -p "$(dirname "${path}")"
    cat >"${path}" <<EOF
#ifndef DACPP_FAKE_LAUNCH_H
#define DACPP_FAKE_LAUNCH_H
#include <cstdint>
#define ACLRT_LAUNCH_KERNEL(kernel) kernel
extern "C" inline uint32_t ${kernel}(...) {
#ifdef DACPP_FAKE_FAILURES
  return dacppFakeLaunchCode();
#else
  return 0;
#endif
}
#endif
EOF
  done
}

make_failure_launch_stub() {
  local dir=$1
  cat >"${dir}/launch_stub.cpp" <<'EOF'
#include "acl/acl.h"
#include <cstdint>
extern "C" uint32_t vector_kernel_0_do(
    uint32_t, void*, uint8_t*, uint8_t*, uint8_t*,
    uint32_t, uint32_t, uint32_t, uint32_t) {
  return dacppFakeLaunchCode();
}
EOF
}

generate() {
  local source=$1 out=$2 mode=$3
  shift 3
  mkdir -p "${out}"
  cp "${source}" "${out}/"
  local base args=()
  base="$(basename "${source}")"
  [[ "${mode}" == multi ]] && args=(--ascendc-use-mlir --ascendc-multicard)
  (cd "${out}" && "${TRANSLATOR}" "${base}" --mode=ascendc "${args[@]}" "$@" \
      -extra-arg=-std=c++17 -- -I"${STD_LIB}" -I"${DACPP_LIB}" \
      >translator.out 2>translator.err)
}

FAKE_SDK="${WORK_ROOT}/fake"
make_fake_sdk "${FAKE_SDK}"
packages=()
for case_name in ${CASES}; do
  source="$(find "${PROJ_DIR}/tests/${case_name}" -maxdepth 1 \
      -name '*.dac.cpp' | sort | head -1)"
  [[ -n "${source}" ]] || { fail "${case_name}: source missing"; continue; }
  for mode in single multi; do
    out="${WORK_ROOT}/${case_name}/${mode}"
    if ! generate "${source}" "${out}" "${mode}"; then
      fail "${case_name}/${mode}: translation failed"
      sed -n '1,40p' "${out}/translator.err" >&2
      continue
    fi
    packages+=("${out}")
    make_launch_headers "${out}" "${out}/host0.cpp"
    if command -v g++ >/dev/null 2>&1 &&
       ! g++ -std=c++11 -pedantic-errors -pthread -fsyntax-only \
          -I"${FAKE_SDK}" -I"${out}" "${out}/host0.cpp" \
          2>"${out}/syntax.err"; then
      fail "${case_name}/${mode}: C++11 host syntax"
      sed -n '1,50p' "${out}/syntax.err" >&2
    fi
    cat >"${out}/runtime_smoke.cpp" <<'EOF'
#include "dacpp_ascendc_host_runtime.h"
static_assert(DACPP_ASCENDC_HOST_RUNTIME_VERSION == 19, "version");
int main() { dacpp_host::Session app; return app.finish(); }
EOF
    if command -v g++ >/dev/null 2>&1 &&
       ! g++ -std=c++11 -pedantic-errors -pthread -fsyntax-only \
          -I"${FAKE_SDK}" -I"${out}" "${out}/runtime_smoke.cpp" \
          2>"${out}/runtime.syntax.err"; then
      fail "${case_name}/${mode}: standalone runtime C++11 syntax"
      sed -n '1,40p' "${out}/runtime.syntax.err" >&2
    fi
  done
done

if ! "${SCRIPT_DIR}/atest_host_compact_gate.py" "${packages[@]}" \
     >"${WORK_ROOT}/compact.json"; then
  fail "compact/name/line contract"
  cat "${WORK_ROOT}/compact.json" >&2
else
  note "compact/name/line contract clean"
fi

# Runtime failure injection must preserve the first fatal stage and one-line log.
FAIL_SDK="${WORK_ROOT}/failure-sdk"
make_failure_sdk "${FAIL_SDK}"
failure_source="${WORK_ROOT}/failure_input.dac.cpp"
sed -e 's/134348800/16/g' -e '/vecC_tensor\.print();/d' \
    "${PROJ_DIR}/tests/t1add/add.dac.cpp" >"${failure_source}"
failure_pkg="${WORK_ROOT}/failure-log/error"
if ! generate "${failure_source}" \
     "${failure_pkg}" single; then
  fail "failure-injection package translation failed"
else
  make_launch_headers "${failure_pkg}" "${failure_pkg}/host0.cpp"
  make_failure_launch_stub "${failure_pkg}"
  if ! g++ -std=c++11 -pedantic-errors -pthread -DDACPP_FAKE_FAILURES \
       -I"${FAIL_SDK}" -I"${failure_pkg}" "${failure_pkg}/host0.cpp" \
       "${failure_pkg}/launch_stub.cpp" \
       -o "${failure_pkg}/host" 2>"${failure_pkg}/link.err"; then
    fail "failure-injection host build"
    sed -n '1,40p' "${failure_pkg}/link.err" >&2
  else
    for item in init:11 alloc:12 copy_in:13 launch:14 sync:15 \
                copy_out:16 cleanup:17; do
      stage="${item%%:*}"
      expected="${item##*:}"
      if DACPP_FAKE_STAGE="${stage}" timeout 10 "${failure_pkg}/host" \
           >"${failure_pkg}/${stage}.out" 2>"${failure_pkg}/${stage}.err"; then
        fail "injected ${stage} failure returned zero"
        continue
      fi
      lines="$(grep -cve '^[[:space:]]*$' "${failure_pkg}/${stage}.err")"
      if [[ "${lines}" -ne 1 ]] ||
         ! grep -q "DACPP runtime error: stage=${stage}.*code=${expected}" \
             "${failure_pkg}/${stage}.err"; then
        fail "injected ${stage} did not preserve one-line first-error"
      fi
    done
  fi
fi

# none emits no stderr while preserving the failure exit; debug emits resources.
for level in none debug; do
  log_pkg="${WORK_ROOT}/failure-log/${level}"
  if ! generate "${failure_source}" \
       "${log_pkg}" single "--ascendc-host-log=${level}"; then
    fail "${level} log package translation failed"
    continue
  fi
  make_launch_headers "${log_pkg}" "${log_pkg}/host0.cpp"
  make_failure_launch_stub "${log_pkg}"
  if ! g++ -std=c++11 -pedantic-errors -pthread -DDACPP_FAKE_FAILURES \
       -I"${FAIL_SDK}" -I"${log_pkg}" "${log_pkg}/host0.cpp" \
       "${log_pkg}/launch_stub.cpp" \
       -o "${log_pkg}/host" 2>"${log_pkg}/link.err"; then
    fail "${level} log host build"
    continue
  fi
  if [[ "${level}" == none ]]; then
    if DACPP_FAKE_STAGE=init timeout 10 "${log_pkg}/host" \
         >"${log_pkg}/run.out" 2>"${log_pkg}/run.err"; then
      fail "none log mode lost nonzero failure exit"
    elif [[ -s "${log_pkg}/run.err" ]]; then
      fail "none log mode emitted stderr"
    fi
  elif ! timeout 10 "${log_pkg}/host" \
           >"${log_pkg}/run.out" 2>"${log_pkg}/run.err" ||
       ! grep -q 'DACPP runtime debug: device=0' "${log_pkg}/run.err"; then
    fail "debug log mode lacks resource detail"
  fi
done

# Invalid logging values must be hard configuration errors.
probe="${WORK_ROOT}/invalid-log"
mkdir -p "${probe}"
cp "${PROJ_DIR}/tests/t1add/add.dac.cpp" "${probe}/"
if (cd "${probe}" && "${TRANSLATOR}" add.dac.cpp --mode=ascendc \
    --ascendc-host-log=verbose -extra-arg=-std=c++17 -- \
    -I"${STD_LIB}" -I"${DACPP_LIB}" >/dev/null 2>err); then
  fail "invalid --ascendc-host-log was accepted"
fi

# User-visible short-name collisions must select a deterministic fallback.
collision="${WORK_ROOT}/collision"
mkdir -p "${collision}/first" "${collision}/second"
sed '/int main()/a\
    int app=0, session=0, route=0, rankRoute=0, rankCount=0, deviceCount=0, dev=0, stream=0, ranks=0, tile=0, workLen=0, srcBegin=0;' \
    "${PROJ_DIR}/tests/t1add/add.dac.cpp" >"${collision}/collision.dac.cpp"
for copy in first second; do
  cp "${collision}/collision.dac.cpp" "${collision}/${copy}/"
  if ! (cd "${collision}/${copy}" && "${TRANSLATOR}" collision.dac.cpp \
      --mode=ascendc -extra-arg=-std=c++17 -- \
      -I"${STD_LIB}" -I"${DACPP_LIB}" >/dev/null 2>err); then
    fail "name-collision translation failed"
  fi
done
if grep -q 'runtime_names .*session=app .*routeObject=route ' \
      "${collision}/first/ascendc_flow_plan.txt" ||
   ! cmp -s "${collision}/first/host0.cpp" "${collision}/second/host0.cpp"; then
  fail "name collision fallback is unsafe or nondeterministic"
fi

# Compact and embedded modes share ABI and the same versioned emitter.
embed="${WORK_ROOT}/embed"
mkdir -p "${embed}"
cp "${PROJ_DIR}/tests/t1add/add.dac.cpp" "${embed}/"
if ! (cd "${embed}" && "${TRANSLATOR}" add.dac.cpp --mode=ascendc \
    --ascendc-embed-host-runtime -extra-arg=-std=c++17 -- \
    -I"${STD_LIB}" -I"${DACPP_LIB}" >/dev/null 2>err); then
  fail "embedded translation failed"
elif grep -qF '#include "dacpp_ascendc_host_runtime.h"' "${embed}/host0.cpp" ||
     [[ -e "${embed}/dacpp_ascendc_host_runtime.h" ]] ||
     ! grep -qF '#define DACPP_ASCENDC_HOST_RUNTIME_VERSION 19' "${embed}/host0.cpp"; then
  fail "embedded runtime packaging contract"
fi

if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi
note "all HostFlow architecture gates passed"
