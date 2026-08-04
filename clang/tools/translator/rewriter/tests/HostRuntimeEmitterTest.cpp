#include "ascendc/HostRuntimeEmitter.h"

#include <iostream>
#include <string>

using namespace dacppTranslator::ascendc;

namespace {

bool ExpectContains(const std::string& text, const std::string& expected) {
    if (text.find(expected) != std::string::npos) return true;
    std::cerr << "generated Host runtime is missing the expected formatting:\n"
              << expected << '\n';
    return false;
}

} // namespace

int main() {
    HostRuntimeRequirements req;
    req.require(HostRuntimeFeature::CheckedMath);
    req.require(HostRuntimeFeature::VectorTiling);
    req.require(HostRuntimeFeature::SingleCard);

    const std::string header = EmitHostRuntimeHeader(req);
    bool ok = true;
    ok = ExpectContains(
             header,
             "enum class Stage : uint8_t {\n"
             "    None, Config, Init, Device, Context, Stream, Alloc,\n"
             "    CopyIn, Launch, Sync, CopyOut, Cleanup, HostFlow\n"
             "};") &&
         ok;
    ok = ExpectContains(
             header,
             "    bool fail(Stage stage, int32_t code, const char* detail,\n"
             "              uint32_t rank = UINT32_MAX) {\n"
             "        if (status_.stage == Stage::None) {\n"
             "            status_.stage = stage;") &&
         ok;
    ok = ExpectContains(
             header,
             "    for (uint32_t id = 0; id < rankCount; ++id) {\n"
             "        ranks[id].id = id; ranks[id].begin = len * id / rankCount;\n"
             "        ranks[id].len = len * (id + 1) / rankCount - ranks[id].begin;\n"
             "    }") &&
         ok;
    ok = ExpectContains(
             header,
             "class Device {\n"
             "public:\n"
             "    explicit Device(Session& session, int32_t id = 0)") &&
         ok;
    return ok ? 0 : 1;
}
