#!/usr/bin/env bash
#
# atest_diff.sh：HostFlow 单路径验证脚本。
#
# HostReplay 渲染入口删除后，已经不存在另一条 host 路径可以对比。本脚本用于：
#
#   - 用当前 translator 在隔离工作目录里分别生成单卡与默认 8-rank 多卡产物；
#   - 单卡与 tests/<case>/ 中 committed 的产物对比；如果
#     tests/<case>/multicard/ 已保存产物，再对比默认 8-rank 多卡结果；
#   - 任何字节差异都视为回归（HostFlow 行为变化）。
#
# 任选用例：tools/atest_diff.sh t1add t2decay
# 全量：    tools/atest_diff.sh
#
# 把 atest_hostflow_arch_gate.sh 作为可选尾步触发，避免影响仅做
# fresh-vs-reference diff 的用法。设置 RUN_ARCH_GATE=1 即在比对完成后再跑一遍
# 架构闸门（任一闸门失败累计到 total_mismatches 并通过退出码体现）。

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$(cd "$PROJ_DIR/../../../build" && pwd)}"
TRANSLATOR="${TRANSLATOR:-$BUILD_DIR/bin/translator/translator}"
STD_LIB="$PROJ_DIR/std_lib/include"
DACPP_LIB="$PROJ_DIR/dacppLib/include"

if [[ ! -x "$TRANSLATOR" ]]; then
  echo "[atest_diff] translator not found or not executable: $TRANSLATOR" >&2
  exit 2
fi

DEFAULT_CASES=(t1add t2decay t3chafen t4liuliang t5l3pt t6image t7matmul t8linear)
CASES=("$@")
if [[ ${#CASES[@]} -eq 0 ]]; then
  CASES=("${DEFAULT_CASES[@]}")
fi

WORK_ROOT="$(mktemp -d -t atest_diff.XXXXXX)"
trap 'rm -rf "$WORK_ROOT"' EXIT

find_entry() {
  local case_dir="$1"
  ls "$case_dir"/*.dac.cpp 2>/dev/null | head -1
}

ARTIFACTS=(
  host0.cpp
  host1.cpp
  tiling.cpp
  ascendc_flow_plan.txt
  dacpp_ascendc_host_runtime.h
)
ARTIFACT_PREFIXES=('kernel[0-9]')

run_one() {
  local entry="$1"
  local out_dir="$2"
  local mode="$3"
  mkdir -p "$out_dir"
  cp "$entry" "$out_dir/"
  local base
  base="$(basename "$entry")"
  local args=("$TRANSLATOR" "$base" --mode=ascendc)
  if [[ "$mode" == "multicard" ]]; then
    # The multicard flag alone must compile an 8-rank fallback.
    args+=(--ascendc-use-mlir --ascendc-multicard)
  fi
  args+=(-extra-arg=-std=c++17 -- -I"$STD_LIB" -I"$DACPP_LIB")
  ( cd "$out_dir" && "${args[@]}" > stdout.log 2> stderr.log )
}

check_embed_equivalence() {
  local case_name="$1"
  local entry="$2"
  local compact_dir="$3"
  case "$case_name" in t1add|t4liuliang|t7matmul) ;; *) return 0 ;; esac
  local embed_dir="$WORK_ROOT/$case_name/embed"
  mkdir -p "$embed_dir"
  cp "$entry" "$embed_dir/"
  local base
  base="$(basename "$entry")"
  ( cd "$embed_dir" && "$TRANSLATOR" "$base" --mode=ascendc \
      --ascendc-embed-host-runtime -extra-arg=-std=c++17 -- \
      -I"$STD_LIB" -I"$DACPP_LIB" >stdout.log 2>stderr.log )
  if [[ -f "$embed_dir/dacpp_ascendc_host_runtime.h" ]] ||
     grep -qF '#include "dacpp_ascendc_host_runtime.h"' "$embed_dir/host0.cpp" ||
     ! grep -qF 'DACPP_ASCENDC_HOST_RUNTIME_VERSION 19' "$embed_dir/host0.cpp" ||
     ! cmp -s <(grep '^extern "C"' "$compact_dir/host0.cpp") \
                <(grep '^extern "C"' "$embed_dir/host0.cpp"); then
    echo "  [DIFF] $case_name :: compact/embed runtime or ABI contract"
    return 1
  fi
  return 0
}

compare_artifacts() {
  local case_name="$1"
  local fresh_dir="$2"
  local commit_dir="$3"
  local mismatches=0

  for artifact in "${ARTIFACTS[@]}"; do
    local f="$fresh_dir/$artifact"
    local c="$commit_dir/$artifact"
    if [[ -f "$c" && -f "$f" ]]; then
      local same=0
      if cmp -s "$c" "$f"; then
        same=1
      fi
      if [[ "$same" -eq 0 ]]; then
        echo "  [DIFF] $case_name :: $artifact"
        diff -u "$c" "$f" | head -40 | sed 's/^/    /'
        mismatches=$((mismatches+1))
      fi
    elif [[ -f "$c" && ! -f "$f" ]]; then
      echo "  [MISSING-FRESH] $case_name :: $artifact"
      mismatches=$((mismatches+1))
    elif [[ ! -f "$c" && -f "$f" ]]; then
      echo "  [MISSING-COMMITTED] $case_name :: $artifact"
      mismatches=$((mismatches+1))
    fi
  done

  for prefix in "${ARTIFACT_PREFIXES[@]}"; do
    local c_files f_files
    mapfile -t c_files < <(cd "$commit_dir" 2>/dev/null && ls ${prefix}*.cpp 2>/dev/null)
    mapfile -t f_files < <(cd "$fresh_dir" 2>/dev/null && ls ${prefix}*.cpp 2>/dev/null)
    local all
    all=$(printf '%s\n' "${c_files[@]}" "${f_files[@]}" | sort -u)
    for fname in $all; do
      [[ -z "$fname" ]] && continue
      local c="$commit_dir/$fname"
      local f="$fresh_dir/$fname"
      if [[ -f "$c" && -f "$f" ]]; then
        if ! cmp -s "$c" "$f"; then
          echo "  [DIFF] $case_name :: $fname"
          mismatches=$((mismatches+1))
        fi
      elif [[ -f "$c" && ! -f "$f" ]]; then
        echo "  [MISSING-FRESH] $case_name :: $fname"
        mismatches=$((mismatches+1))
      elif [[ ! -f "$c" && -f "$f" ]]; then
        echo "  [MISSING-COMMITTED] $case_name :: $fname"
        mismatches=$((mismatches+1))
      fi
    done
  done

  return $mismatches
}

total_mismatches=0
passed=0
failed=0

echo "[atest_diff] translator: $TRANSLATOR"
echo "[atest_diff] mode      : HostFlow single-card + committed 8-rank references"
echo "[atest_diff] cases     : ${CASES[*]}"
echo

for case_name in "${CASES[@]}"; do
  case_dir="$PROJ_DIR/tests/$case_name"
  if [[ ! -d "$case_dir" ]]; then
    echo "[$case_name] case directory missing, skipped"
    continue
  fi
  entry=$(find_entry "$case_dir")
  if [[ -z "$entry" ]]; then
    echo "[$case_name] no .dac.cpp entry, skipped"
    continue
  fi

  fresh_dir="$WORK_ROOT/$case_name/single"
  run_one "$entry" "$fresh_dir" single

  case_diffs=0
  if compare_artifacts "$case_name:single" "$fresh_dir" "$case_dir"; then
    :
  else
    case_diffs=$((case_diffs+$?))
  fi
  if [[ -n "$(find "$case_dir/multicard" -maxdepth 1 -type f -print -quit 2>/dev/null)" ]]; then
    fresh_multicard_dir="$WORK_ROOT/$case_name/multicard"
    run_one "$entry" "$fresh_multicard_dir" multicard
    if compare_artifacts "$case_name:multicard" "$fresh_multicard_dir" "$case_dir/multicard"; then
      :
    else
      case_diffs=$((case_diffs+$?))
    fi
  fi
  if ! check_embed_equivalence "$case_name" "$entry" "$fresh_dir"; then
    case_diffs=$((case_diffs+1))
  fi

  if [[ "$case_diffs" -eq 0 ]]; then
    echo "[$case_name] OK"
    passed=$((passed+1))
  else
    echo "[$case_name] $case_diffs artifact(s) differ vs committed reference"
    total_mismatches=$((total_mismatches+case_diffs))
    failed=$((failed+1))
  fi
done

echo
echo "[atest_diff] summary: passed=$passed failed=$failed total_diffs=$total_mismatches"

# 可选 arch gate。默认不跑（避免影响只想看 fresh-vs-reference diff 的用法）；
# RUN_ARCH_GATE=1 时，结束后再跑 atest_hostflow_arch_gate.sh，失败计入退出码。
if [[ "${RUN_ARCH_GATE:-0}" != "0" ]]; then
  echo
  echo "[atest_diff] RUN_ARCH_GATE=1 — invoking atest_hostflow_arch_gate.sh"
  if ! ARCH_GATE_CASES="${CASES[*]}" bash "$SCRIPT_DIR/atest_hostflow_arch_gate.sh"; then
    echo "[atest_diff] arch gate failed" >&2
    total_mismatches=$((total_mismatches+1))
  fi
fi

exit "$total_mismatches"
