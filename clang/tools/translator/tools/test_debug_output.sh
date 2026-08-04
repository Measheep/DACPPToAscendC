#!/usr/bin/env bash
set -euo pipefail

TRANSLATOR=${1:?translator path is required}
PROJ_DIR=${2:?translator source directory is required}
WORK_ROOT=$(mktemp -d -t translator-debug-output.XXXXXX)
trap 'rm -rf "$WORK_ROOT"' EXIT

run_translator() {
  local output_dir=$1
  shift
  mkdir -p "$output_dir"
  cp "$PROJ_DIR/tests/t1add/add.dac.cpp" "$output_dir/"
  (
    cd "$output_dir"
    "$TRANSLATOR" add.dac.cpp --mode=ascendc "$@" \
      -extra-arg=-std=c++17 -- \
      -I"$PROJ_DIR/std_lib/include" \
      -I"$PROJ_DIR/dacppLib/include" \
      >stdout.log 2>stderr.log
  )
}

run_translator "$WORK_ROOT/default"

if grep -Eq 'DACPP Parser 中间产物输出|发现 DACPP 表达式 AST|dacExprCount|Loop bound:|Found swap:' \
     "$WORK_ROOT/default/stdout.log" ||
   grep -Eq 'BinaryOperator 0x|\[AscendC tiling\]|allocation breakdown:' \
     "$WORK_ROOT/default/stderr.log"; then
  echo "default translation leaked parser or code-generation diagnostics" >&2
  exit 1
fi

run_translator "$WORK_ROOT/debug" --debug

grep -q 'DACPP Parser 中间产物输出' "$WORK_ROOT/debug/stdout.log"
grep -q '发现 DACPP 表达式 AST' "$WORK_ROOT/debug/stdout.log"
grep -q 'dacExprCount:' "$WORK_ROOT/debug/stdout.log"
grep -q 'BinaryOperator 0x' "$WORK_ROOT/debug/stderr.log"
grep -q '\[AscendC tiling\]' "$WORK_ROOT/debug/stderr.log"

run_translator "$WORK_ROOT/dump-parser" --dump-parser
grep -q 'DACPP Parser 中间产物输出' "$WORK_ROOT/dump-parser/stdout.log"
grep -q '\[AscendC tiling\]' "$WORK_ROOT/dump-parser/stderr.log"
