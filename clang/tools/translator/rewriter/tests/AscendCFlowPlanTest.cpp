#include "ascendc/AscendCFlowPlan.h"

#include <iostream>
#include <string>
#include <utility>

using namespace dacppTranslator::ascendc;

namespace {

bool ExpectContains(const std::string& text, const std::string& expected) {
    if (text.find(expected) != std::string::npos) return true;
    std::cerr << "FlowPlan dump is missing: " << expected << '\n';
    return false;
}

bool ExpectNotContains(const std::string& text, const std::string& unexpected) {
    if (text.find(unexpected) == std::string::npos) return true;
    std::cerr << "FlowPlan dump unexpectedly contains: " << unexpected << '\n';
    return false;
}

} // namespace

int main() {
    AscendCFlowPlan plan;

    FlowHostInput posixInput;
    posixInput.name = "posix";
    posixInput.location.file = "/workspace/alice/project/add.dac.cpp";
    posixInput.location.line = 15;
    posixInput.location.column = 5;
    plan.addHostInput(std::move(posixInput));

    FlowHostInput windowsInput;
    windowsInput.name = "windows";
    windowsInput.location.file = "C:\\work\\project\\linear.dac.cpp";
    windowsInput.location.line = 35;
    windowsInput.location.column = 7;
    plan.addHostInput(std::move(windowsInput));

    const std::string dump = DumpAscendCFlowPlan(plan);
    bool ok = true;
    ok = ExpectContains(dump, "loc=add.dac.cpp:15:5") && ok;
    ok = ExpectContains(dump, "loc=linear.dac.cpp:35:7") && ok;
    ok = ExpectNotContains(dump, "/workspace/alice") && ok;
    ok = ExpectNotContains(dump, "C:\\work") && ok;
    return ok ? 0 : 1;
}
