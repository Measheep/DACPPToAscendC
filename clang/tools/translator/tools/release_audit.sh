#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/../../../.." && pwd)
cd "$ROOT_DIR"

audit_paths=(
  README.md
  UPSTREAM.md
  THIRD_PARTY_NOTICES.md
  NOTICE
  SECURITY.md
  CONTRIBUTING.md
  CODE_OF_CONDUCT.md
  SUPPORT.md
  .github
  clang/tools/translator
)

# Construct these strings in pieces so this audit script does not flag itself.
private_path_pattern="/""home/|/""Users/|dacpp""A"
internal_marker_pattern='//[^\n]*SYCL|//[[:space:]]*gemini|\bPlan[0-9]+\b|\bPhase [0-9]+\b|\bplan[0-9]+\b|AscendC Designer'
secret_pattern='-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}'

if rg -n --glob '!release_audit.sh' "$private_path_pattern" "${audit_paths[@]}"; then
  echo "release audit: private or legacy absolute path found" >&2
  exit 1
fi

if rg -n -i --glob '!release_audit.sh' "$internal_marker_pattern" "${audit_paths[@]}"; then
  echo "release audit: obsolete internal marker found" >&2
  exit 1
fi

if rg -n --glob '!release_audit.sh' -- "$secret_pattern" "${audit_paths[@]}"; then
  echo "release audit: credential-like content found" >&2
  exit 1
fi

if [[ -e build || -e .ci ]]; then
  echo "release audit: in-tree build or inherited .ci directory found" >&2
  exit 1
fi

mapfile -t unwanted_files < <(
  find clang/tools/translator -type f \
    \( -name '*.log' -o -name '*.trace' -o -name '.env*' -o \
       -name '*.pem' -o -name '*.key' -o -name '*.p12' -o \
       -name '*.pfx' -o -name '*.jks' -o -name '*.so' -o \
       -name '*.a' -o -name '*.o' -o -path '*/credentials/*' -o \
       -path '*/secrets/*' \) -print
)
if (( ${#unwanted_files[@]} != 0 )); then
  printf 'release audit: unwanted translator artifact: %s\n' \
    "${unwanted_files[@]}" >&2
  exit 1
fi

required_docs=(
  README.md
  UPSTREAM.md
  THIRD_PARTY_NOTICES.md
  NOTICE
  SECURITY.md
  CONTRIBUTING.md
  CODE_OF_CONDUCT.md
  SUPPORT.md
  LICENSE.TXT
)
for required_doc in "${required_docs[@]}"; do
  [[ -f "$required_doc" ]] || {
    echo "release audit: missing $required_doc" >&2
    exit 1
  }
done

while IFS= read -r link; do
  target=${link%%#*}
  case "$target" in
    ''|http://*|https://*|mailto:*) continue ;;
  esac
  [[ -e "$target" ]] || {
    echo "release audit: README link target does not exist: $target" >&2
    exit 1
  }
done < <(
  rg -o '\[[^]]+\]\([^)]+\)' README.md |
    sed -E 's/^.*\]\(([^)]+)\)$/\1/'
)

baseline_count=$(
  find clang/tools/translator/tests -type f \
    \( -name 'host*.cpp' -o -name 'kernel*.cpp' -o \
       -name 'tiling.cpp' -o -name 'ascendc_flow_plan.txt' -o \
       -name 'dacpp_ascendc_host_runtime.h' \) -print |
    wc -l
)
if [[ $baseline_count -ne 34 ]]; then
  echo "release audit: expected 34 generated baselines, found $baseline_count" >&2
  exit 1
fi

huawei_files=(
  clang/tools/translator/tests/t7matmul/matmul_main.cpp
  clang/tools/translator/tests/t7matmul/data_utils.h
  clang/tools/translator/tests/t7matmul/matmul_custom.cpp
  clang/tools/translator/tests/t7matmul/matmul_custom_tiling.cpp
)
for huawei_file in "${huawei_files[@]}"; do
  head -n 12 "$huawei_file" | grep -q 'Huawei Technologies Co., Ltd' || {
    echo "release audit: Huawei copyright header missing: $huawei_file" >&2
    exit 1
  }
done

workflow_count=$(find .github/workflows -type f -name '*.yml' -print | wc -l)
if [[ $workflow_count -ne 1 ]] || rg -n 'pull_request_target' .github; then
  echo "release audit: unexpected GitHub workflow layout or trigger" >&2
  exit 1
fi

echo "release hygiene: PASS"
