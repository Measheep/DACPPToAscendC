#!/usr/bin/env python3
"""Machine-readable checks for freshly generated compact Host packages."""

import argparse
import json
import re
from pathlib import Path


CAPS = {
    "t1add": 300, "t2decay": 420, "t3chafen": 280,
    "t4liuliang": 450, "t5l3pt": 310, "t6image": 520,
    "t7matmul": 260, "t8linear": 350,
}
ABI_DUNDER = {"__global__", "__aicore__"}
FORBIDDEN = (
    "DacppMulticardExecutor", "DacppReadRankDim", "PrepareHostSlab",
    "DacppVectorExpr", "RankWorker", "LinearWorker", "StorageElems",
    "CopyStart", "CopyEnd", "WriteCount", "std::is_same", "!false",
)
EXPECTED_LAUNCHES = {"t6image": 2}


def physical_lines(text):
    return len(text.splitlines())


def inspect(package):
    host = package / "host0.cpp"
    runtime = package / "dacpp_ascendc_host_runtime.h"
    plan = package / "ascendc_flow_plan.txt"
    errors = []
    if not host.is_file():
        return {"package": str(package), "errors": ["host0.cpp missing"]}
    host_text = host.read_text(encoding="utf-8")
    case = next((part for part in package.parts if part in CAPS), package.name)
    lines = physical_lines(host_text)
    if case in CAPS and lines > CAPS[case]:
        errors.append("host lines %d exceed %d" % (lines, CAPS[case]))
    include_count = host_text.count('#include "dacpp_ascendc_host_runtime.h"')
    if include_count != 1:
        errors.append("companion include count is %d" % include_count)
    if "static_assert(DACPP_ASCENDC_HOST_RUNTIME_VERSION == 19" not in host_text:
        errors.append("runtime version static_assert missing")
    if not runtime.is_file():
        errors.append("companion runtime missing")
        runtime_text = ""
    else:
        runtime_text = runtime.read_text(encoding="utf-8")
        if "#define DACPP_ASCENDC_HOST_RUNTIME_VERSION 19" not in runtime_text:
            errors.append("runtime header version mismatch")
        indentation_samples = (
            "\n    None, Config, Init, Device, Context, Stream, Alloc,\n",
            "\n        if (status_.stage == Stage::None) {\n",
            "\n            status_.stage = stage;\n",
        )
        if any(sample not in runtime_text for sample in indentation_samples):
            errors.append("runtime header indentation is missing")
    for token in FORBIDDEN:
        if token in host_text:
            errors.append("forbidden generated token: " + token)
    for token in re.findall(r"\b__[A-Za-z_][A-Za-z0-9_]*", host_text):
        if token not in ABI_DUNDER:
            errors.append("reserved identifier: " + token)
    if re.search(r"\bacl(?:rt)?(?:Malloc|Memcpy|Free|Synchronize|Create|Destroy|Set|Reset|Init|Finalize)\w*\s*\(", host_text):
        errors.append("raw ACL lifecycle/copy call escaped runtime")
    main_line = next((i for i, line in enumerate(host_text.splitlines(), 1)
                      if re.search(r"\bmain\s*\(", line)), 0)
    if main_line and main_line - 1 > 80:
        errors.append("main prelude exceeds 80 lines")
    if "std::cerr" in host_text:
        errors.append("host contains std::cerr")
    if plan.is_file():
        plan_text = plan.read_text(encoding="utf-8")
        launch_count = len(re.findall(
            r"^\s*step#\d+\s+KernelLaunch\b", plan_text, re.MULTILINE))
        expected_launches = EXPECTED_LAUNCHES.get(case, 1)
        if launch_count != expected_launches:
            errors.append("KernelLaunch count is %d, expected %d" %
                          (launch_count, expected_launches))
        if package.name == "multi":
            command_count = len(re.findall(
                r"^\s*step#\d+\s+GeneratedCodeBlock\s+"
                r"role=MulticardLaunch\b", plan_text, re.MULTILINE))
            if command_count != expected_launches:
                errors.append("MulticardLaunch count is %d, expected %d" %
                              (command_count, expected_launches))
            if len(re.findall(r"role=MulticardRouteSetup\b", plan_text)) != 1:
                errors.append("multicard route setup count is not one")
            if len(re.findall(r"role=MulticardRouteCleanup\b", plan_text)) != 1:
                errors.append("multicard route cleanup count is not one")

        buffer_rows = [
            (int(match.group(1)), match.group(2), int(match.group(3)))
            for match in re.finditer(
                r"buffer#(\d+) name=([A-Za-z_][A-Za-z0-9_]*) .* "
                r"resident=([01])\b", plan_text)
        ]
        if package.name == "multi" and case == "t2decay":
            required = ("N0s", "lambdas", "local_A", "t")
            rows = [(buffer_id, name) for buffer_id, name, _ in buffer_rows
                    if name in required]
            if len(rows) != 4 or {name for _, name in rows} != set(required):
                errors.append("t2 buffer descriptor set is incomplete")
            elif len({buffer_id for buffer_id, _ in rows}) != 4:
                errors.append("t2 buffer descriptors do not have distinct IDs")
        if package.name == "multi" and case == "t6image":
            resident_inputs = {
                name: buffer_id for buffer_id, name, resident in buffer_rows
                if resident and name in ("imageR", "imageG", "imageB")
            }
            if set(resident_inputs) != {"imageR", "imageG", "imageB"}:
                errors.append("t6 resident input descriptor set is incomplete")
            elif set(resident_inputs.values()) != {3, 4, 5}:
                errors.append("t6 resident input IDs do not reuse producer IDs")
            if not re.search(r"internalReuseIds=\[\s*3,\s*4,\s*5\s*\]",
                             plan_text):
                errors.append("t6 internal reuse edge is missing")
            expression_scopes = list(re.finditer(
                r"\bconst size_t workLen\b", host_text))
            if len(expression_scopes) != 2:
                errors.append("t6 second expression scope is missing")
            else:
                second_expr = expression_scopes[1].start()
                second_body = host_text[second_expr:]
                if re.search(
                        r"\.copyIn\([^;\n]*(?:imageR|imageG|imageB)Device",
                        second_body):
                    errors.append("t6 resident consumer emitted H2D")
                first_body = host_text[:second_expr]
                for stem in ("imageR", "imageG", "imageB"):
                    if not re.search(
                            r"\.copyIn\([^;\n]*" + stem + r"Device",
                            first_body):
                        errors.append("t6 producer input H2D missing: " + stem)
            if ".resident(" in host_text or "enableReuse" in host_text:
                errors.append("t6 emitted a dynamic resident branch")
        for match in re.finditer(
                r"generated_name text=([A-Za-z_][A-Za-z0-9_]*) "
                r"kind=(framework|public)", plan_text):
            limit = 24 if match.group(2) == "public" else 20
            if len(match.group(1)) > limit:
                errors.append("audited name exceeds limit: " + match.group(1))
    else:
        errors.append("flow plan missing")
    return {
        "package": str(package), "case": case, "host_lines": lines,
        "runtime_lines": physical_lines(runtime_text) if runtime_text else 0,
        "runtime_includes": include_count, "main_line": main_line,
        "if_count": len(re.findall(r"\bif\s*\(", host_text)),
        "errors": sorted(set(errors)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("packages", nargs="+", type=Path)
    args = parser.parse_args()
    reports = [inspect(path.resolve()) for path in args.packages]
    total = sum(report.get("host_lines", 0) for report in reports)
    if total > 2890:
        for report in reports:
            report["errors"].append("aggregate host lines exceed 2890")
    result = {"ok": not any(r["errors"] for r in reports),
              "total_host_lines": total, "packages": reports}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
