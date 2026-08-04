#!/usr/bin/env bash

SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")
WORK_DIR=$(dirname "$SCRIPT_PATH")

dacpp() {
    local default_build_dir="$WORK_DIR/../../../build"
    local build_dir="${DACPP_BUILD_DIR:-$default_build_dir}"
    local translator_bin="${DACPP_TRANSLATOR:-$build_dir/bin/translator/translator}"

    if [[ ! -x "$translator_bin" ]]; then
        echo "No such file: $translator_bin" >&2
        return 1
    fi

    "$translator_bin" "$@" \
    -extra-arg=-std=c++17 -- \
    -I"$WORK_DIR"/std_lib/include \
    -I"$WORK_DIR"/dacppLib/include
}
