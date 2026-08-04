#include "ascendc/HostRuntimeEmitter.h"

#include "ascendc/AscendCArtifacts.h"
#include "ascendc/AscendCFlowPlan.h"
#include "ascendc/AscendCPolicy.h"

#include <sstream>

namespace dacppTranslator {
namespace ascendc {
namespace {

const char* LogLevelText(HostRuntimeLogLevel level) {
    switch (level) {
        case HostRuntimeLogLevel::None: return "0";
        case HostRuntimeLogLevel::Error: return "1";
        case HostRuntimeLogLevel::Debug: return "2";
    }
    return "1";
}

std::string CommonRuntime(const HostRuntimeRequirements& req) {
    std::string code = R"~~~(
namespace dacpp_host {

using BufferId = uint32_t;

enum class Stage : uint8_t {
    None, Config, Init, Device, Context, Stream, Alloc,
    CopyIn, Launch, Sync, CopyOut, Cleanup, HostFlow
};

struct Status {
    Stage stage; int32_t code; uint32_t rank; std::string detail;
    Status() : stage(Stage::None), code(0), rank(UINT32_MAX) {}
};

inline const char* stageName(Stage stage) {
    switch (stage) {
        case Stage::None: return "none"; case Stage::Config: return "config";
        case Stage::Init: return "init"; case Stage::Device: return "device";
        case Stage::Context: return "context"; case Stage::Stream: return "stream";
        case Stage::Alloc: return "alloc"; case Stage::CopyIn: return "copy_in";
        case Stage::Launch: return "launch"; case Stage::Sync: return "sync";
        case Stage::CopyOut: return "copy_out"; case Stage::Cleanup: return "cleanup";
        case Stage::HostFlow: return "host_flow";
    }
    return "unknown";
}

class Session {
public:
    bool ok() const { return status_.stage == Stage::None; }
    const Status& status() const { return status_; }
    bool fail(Stage stage, int32_t code, uint32_t rank = UINT32_MAX) {
        return fail(stage, code, static_cast<const char*>(nullptr), rank);
    }
    bool fail(Stage stage, int32_t code, const char* detail,
              uint32_t rank = UINT32_MAX) {
        if (status_.stage == Stage::None) {
            status_.stage = stage;
            status_.code = code;
            status_.rank = rank;
            if (detail) status_.detail = detail;
        }
        return false;
    }
    void debugValue(const char* label, size_t value) const {
        if (kLogLevel == 2)
            std::cerr << "DACPP runtime debug: " << label << "="
                      << value << std::endl;
    }
    void debugPair(const char* label, size_t first, size_t second) const {
        if (kLogLevel == 2)
            std::cerr << "DACPP runtime debug: " << label << "="
                      << first << "->" << second << std::endl;
    }
    int32_t finish() {
        if (!ok() && kLogLevel != 0) {
            std::cerr << "DACPP runtime error: stage="
                      << stageName(status_.stage);
            if (status_.rank != UINT32_MAX)
                std::cerr << " rank=" << status_.rank;
            std::cerr << " code=" << status_.code;
            if (!status_.detail.empty()) std::cerr << " " << status_.detail;
            std::cerr << std::endl;
        }
        return ok() ? 0 : (status_.code == 0 ? 1 : status_.code);
    }
private:
    static const int kLogLevel = )~~~";
    code += LogLevelText(req.logLevel);
    code += R"~~~(;
    Status status_;
};

struct Tile {
    uint32_t len, cores, block, size, count;
    Tile() : len(0), cores(1), block(0), size(0), count(0) {}
};

enum class Access : uint8_t { Read, Write, ReadWrite };
enum class Distribution : uint8_t { Sharded, Replicated, DeviceOnly };

struct BufferSpec {
    BufferId id; Access access;
    uint32_t elemBytes, leftHalo, rightHalo;
    bool broadcast; Distribution distribution; size_t stride;
    const char* label;
};

struct RankArgs {
    uint32_t id; size_t begin, len;
    RankArgs() : id(0), begin(0), len(0) {}
};
using RankSpan = std::vector<RankArgs>;

inline RankSpan splitRanks(size_t len, uint32_t rankCount) {
    rankCount = rankCount == 0 ? 1 : rankCount;
    if (len != 0 && rankCount > len) rankCount = static_cast<uint32_t>(len);
    RankSpan ranks(rankCount);
    for (uint32_t id = 0; id < rankCount; ++id) {
        ranks[id].id = id; ranks[id].begin = len * id / rankCount;
        ranks[id].len = len * (id + 1) / rankCount - ranks[id].begin;
    }
    return ranks;
}

inline uint32_t readRankCount(Session& app, const char* envName,
                              uint32_t fallback, size_t extent) {
    uint32_t result = fallback == 0 ? 1 : fallback;
    const char* value =
        envName && envName[0] != '\0' ? std::getenv(envName) : nullptr;
    if (value) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(value, &end, 10);
        if (value[0] == '\0' || end == value || *end != '\0' || parsed == 0 ||
            parsed > static_cast<unsigned long>(UINT32_MAX)) {
            app.fail(Stage::Config, 1);
            return 1;
        }
        result = static_cast<uint32_t>(parsed);
    }
    if (extent != 0 && result > extent) {
        app.debugPair("rank_clamp", result, extent);
        result = static_cast<uint32_t>(extent);
    } else {
        app.debugValue("rank_count", result);
    }
    return result == 0 ? 1 : result;
}

template <class T>
inline bool ensureSize(std::vector<T>& values, size_t required) {
    if (values.size() < required) values.resize(required);
    return true;
}

inline bool checkedAdd(Session& app, size_t lhs, size_t rhs, size_t& out) {
    if (rhs > std::numeric_limits<size_t>::max() - lhs)
        return app.fail(Stage::Config, 2);
    out = lhs + rhs;
    return true;
}

inline bool checkedMul(Session& app, size_t lhs, size_t rhs, size_t& out) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
        return app.fail(Stage::Config, 2);
    out = lhs * rhs;
    return true;
}

inline bool checkedAlign(Session& app, size_t value, size_t alignment,
                         size_t& out) {
    if (alignment == 0) return app.fail(Stage::Config, 2);
    const size_t remainder = value % alignment;
    return remainder == 0
        ? (out = value, true)
        : checkedAdd(app, value, alignment - remainder, out);
}

template <class T>
class HostBuffer {
public:
    HostBuffer(Session& app, size_t logicalCount, size_t leftHalo,
               size_t rightHalo, size_t align)
        : logicalElems(logicalCount), kernelElems(0),
          leftHaloElems(leftHalo), rightHaloElems(rightHalo),
          alignElems(align), host(nullptr), device(nullptr) {
        size_t halo = 0, storage = 0, physical = 0;
        if (!checkedAdd(app, leftHalo, rightHalo, halo)) return;
        const size_t compute = logicalCount > halo ? logicalCount - halo : 0;
        if (!checkedAlign(app, compute, align, kernelElems) ||
            !checkedAdd(app, kernelElems, halo, storage) ||
            !checkedAlign(app, storage, align, physical)) return;
        data.resize(std::max<size_t>(physical, 1), static_cast<T>(0));
    }
    size_t bytes() const { return data.size() * sizeof(T); }

    std::vector<T> data;
    size_t logicalElems, kernelElems, leftHaloElems, rightHaloElems, alignElems;
    uint8_t *host, *device;
};

} // namespace dacpp_host
)~~~";
    return code;
}

const char* CheckedMathAndTiling = R"~~~(
namespace dacpp_host {
inline Tile makeTile(Session& app, size_t len, uint32_t cores,
                     uint32_t align, uint32_t maxTile) {
    Tile tile;
    size_t combined = 0, total = 0;
    tile.cores = cores == 0 ? 1 : cores;
    if (align == 0 || maxTile == 0 ||
        !checkedMul(app, tile.cores, align, combined) ||
        (len != 0 && !checkedAlign(app, len, combined, total))) return tile;
    if (total > UINT32_MAX) {
        app.fail(Stage::Config, 2);
        return tile;
    }
    tile.block = static_cast<uint32_t>(total / tile.cores);
    size_t size = std::min<size_t>(tile.block, maxTile);
    if (size != 0) {
        size -= size % align;
        size = std::max<size_t>(size, align);
    }
    const size_t count = size == 0 ? 0 : (tile.block - 1) / size + 1;
    if (size > UINT32_MAX || count > UINT32_MAX) {
        app.fail(Stage::Config, 2);
        return tile;
    }
    tile.len = static_cast<uint32_t>(total);
    tile.size = static_cast<uint32_t>(size);
    tile.count = static_cast<uint32_t>(count);
    return tile;
}
} // namespace dacpp_host
)~~~";

const char* Fp16Bridge = R"~~~(
namespace dacpp_host {
inline uint16_t fp32ToFp16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    const uint32_t mantissa = bits & 0x7fffffu;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    if (exponent == 255)
        return static_cast<uint16_t>(sign | 0x7c00u | (mantissa ? 1u : 0u));
    const int32_t exp = static_cast<int32_t>(exponent) - 127 + 15;
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    if (exp <= 0) return sign;
    return static_cast<uint16_t>(
        sign | (static_cast<uint16_t>(exp) << 10) |
        static_cast<uint16_t>(mantissa >> 13));
}

inline float fp16ToFp32(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    uint32_t exponent = static_cast<uint32_t>(bits & 0x7c00u) >> 10;
    uint32_t mantissa = static_cast<uint32_t>(bits & 0x03ffu);
    uint32_t output = 0;
    if (exponent == 0) {
        if (mantissa == 0) output = sign;
        else {
            exponent = 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x03ffu;
            output = sign | ((exponent + 112) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        output = sign | 0x7f800000u | (mantissa << 13);
    } else {
        output = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &output, sizeof(value));
    return value;
}

} // namespace dacpp_host
)~~~";

const char* PersistentLoopControls = R"~~~(
namespace dacpp_host {
template <class T> class PersistentControl {
public:
    explicit PersistentControl(T initial) : value(initial) {}
    void advance(T delta) { value += delta; ++generation; }
    T value;
    size_t generation = 0;
};

class TemporalControl {
public:
    void beginGeneration() { applying = false; ++generation; }
    void beginApply() { applying = true; }
    void requestStop() { stopped = true; }
    size_t generation = 0;
    bool applying = false;
    bool stopped = false;
};
} // namespace dacpp_host
)~~~";

const char* SingleCardDevice = R"~~~(
#ifndef ASCENDC_CPU_DEBUG
namespace dacpp_host {
class Device {
public:
    explicit Device(Session& session, int32_t id = 0)
        : app(session), deviceId(id) {
        aclError code = aclInit(nullptr);
        if (code != ACL_SUCCESS) { app.fail(Stage::Init, code); return; }
        initialized = true;
        code = aclrtSetDevice(deviceId);
        if (code != ACL_SUCCESS) { app.fail(Stage::Device, code); return; }
        selected = true;
        code = aclrtCreateContext(&context, deviceId);
        if (code != ACL_SUCCESS) { app.fail(Stage::Context, code); return; }
        code = aclrtSetCurrentContext(context);
        if (code != ACL_SUCCESS) { app.fail(Stage::Context, code); return; }
        code = aclrtCreateStream(&streamValue);
        if (code != ACL_SUCCESS) app.fail(Stage::Stream, code);
        else {
            valid = true;
            app.debugValue("device", static_cast<size_t>(deviceId));
        }
    }
    ~Device() { close(); }
    bool ok() const { return valid && app.ok(); }
    aclrtStream stream() const { return streamValue; }
    bool allocate(uint8_t*& host, uint8_t*& device, size_t bytes) {
        if (!app.ok()) return false;
        aclError code = aclrtMallocHost(reinterpret_cast<void**>(&host), bytes);
        if (code != ACL_SUCCESS || !host)
            return app.fail(Stage::Alloc, code == ACL_SUCCESS ? 1 : code);
        code = aclrtMalloc(reinterpret_cast<void**>(&device), bytes,
                           ACL_MEM_MALLOC_HUGE_FIRST);
        if (code != ACL_SUCCESS || !device) {
            aclrtFreeHost(host); host = nullptr;
            return app.fail(Stage::Alloc, code == ACL_SUCCESS ? 1 : code);
        }
        return true;
    }
    bool allocate(uint8_t*& device, size_t bytes) {
        if (!app.ok()) return false;
        const aclError code = aclrtMalloc(
            reinterpret_cast<void**>(&device), bytes,
            ACL_MEM_MALLOC_HUGE_FIRST);
        return code == ACL_SUCCESS && device
            ? true : app.fail(Stage::Alloc, code == ACL_SUCCESS ? 1 : code);
    }
    template <class T> bool allocate(HostBuffer<T>& buffer) {
        return allocate(buffer.host, buffer.device, buffer.bytes());
    }
    template <class T> bool copyIn(HostBuffer<T>& buffer) {
        if (!app.ok() || !buffer.host || !buffer.device) return false;
        std::memcpy(buffer.host, buffer.data.data(), buffer.bytes());
        const aclError code = aclrtMemcpy(
            buffer.device, buffer.bytes(), buffer.host, buffer.bytes(),
            ACL_MEMCPY_HOST_TO_DEVICE);
        return code == ACL_SUCCESS || app.fail(Stage::CopyIn, code);
    }
    template <class T> bool copyOut(HostBuffer<T>& buffer) {
        if (!app.ok() || !buffer.host || !buffer.device) return false;
        const aclError code = aclrtMemcpy(
            buffer.host, buffer.bytes(), buffer.device, buffer.bytes(),
            ACL_MEMCPY_DEVICE_TO_HOST);
        if (code != ACL_SUCCESS) return app.fail(Stage::CopyOut, code);
        std::memcpy(buffer.data.data(), buffer.host, buffer.bytes());
        return true;
    }
    bool copyIn(void* device, size_t capacity, const void* host, size_t bytes) {
        if (!app.ok()) return false;
        const aclError code = aclrtMemcpy(
            device, capacity, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        return code == ACL_SUCCESS || app.fail(Stage::CopyIn, code);
    }
    bool copyOut(void* host, size_t capacity, const void* device, size_t bytes) {
        if (!app.ok()) return false;
        const aclError code = aclrtMemcpy(
            host, capacity, device, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        return code == ACL_SUCCESS || app.fail(Stage::CopyOut, code);
    }
    bool sync() {
        if (!app.ok()) return false;
        const aclError code = aclrtSynchronizeStream(streamValue);
        return code == ACL_SUCCESS || app.fail(Stage::Sync, code);
    }
    bool launch(int32_t code) {
        return code == 0 || app.fail(Stage::Launch, code);
    }
    void release(uint8_t*& host, uint8_t*& device) {
        if (device) {
            const aclError code = aclrtFree(device);
            if (code != ACL_SUCCESS) app.fail(Stage::Cleanup, code);
            device = nullptr;
        }
        if (host) {
            const aclError code = aclrtFreeHost(host);
            if (code != ACL_SUCCESS) app.fail(Stage::Cleanup, code);
            host = nullptr;
        }
    }
    void release(uint8_t*& device) {
        if (!device) return;
        const aclError code = aclrtFree(device);
        if (code != ACL_SUCCESS) app.fail(Stage::Cleanup, code);
        device = nullptr;
    }
    template <class T> void release(HostBuffer<T>& buffer) {
        release(buffer.host, buffer.device);
    }
    void close() {
        if (closed) return;
        closed = true;
        if (streamValue) {
            const aclError code = aclrtDestroyStream(streamValue);
            if (code != ACL_SUCCESS) app.fail(Stage::Cleanup, code);
            streamValue = nullptr;
        }
        if (context) {
            const aclError code = aclrtDestroyContext(context);
            if (code != ACL_SUCCESS) app.fail(Stage::Cleanup, code);
            context = nullptr;
        }
        if (selected) {
            const aclError code = aclrtResetDevice(deviceId);
            if (code != ACL_SUCCESS) app.fail(Stage::Cleanup, code);
            selected = false;
        }
        if (initialized) {
            const aclError code = aclFinalize();
            if (code != ACL_SUCCESS) app.fail(Stage::Cleanup, code);
            initialized = false;
        }
    }
private:
    Session& app;
    int32_t deviceId;
    aclrtContext context = nullptr;
    aclrtStream streamValue = nullptr;
    bool initialized = false, selected = false, closed = false, valid = false;
};
} // namespace dacpp_host
#endif
)~~~";

const char* MulticardRoute = R"~~~(
namespace dacpp_host {
class Route {
public:
    using Command =
        std::function<void(uint32_t, aclrtContext, aclrtStream)>;

    Route(Session& session, uint32_t requested)
        : app(session), rankCount(requested == 0 ? 1 : requested),
          ranks(rankCount) {
        if (!app.ok()) { valid = false; return; }
        const aclError initCode = aclInit(nullptr);
        if (initCode != ACL_SUCCESS) {
            valid = app.fail(Stage::Init, initCode);
            return;
        }
        initialized = true;
        uint32_t available = 0;
        const aclError countCode = aclrtGetDeviceCount(&available);
        if (countCode != ACL_SUCCESS || available < rankCount)
            valid = app.fail(Stage::Device,
                             countCode == ACL_SUCCESS ? 1 : countCode);
    }
    ~Route() { close(); }
    bool ok() const { return valid && app.ok(); }
    uint32_t size() const { return rankCount; }

    bool prepare(const BufferSpec* specs, size_t count) {
        if (!ok() || !ensureStarted()) return false;
        app.debugValue("buffers", count);
        BufferId maxId = 0;
        for (size_t i = 0; i < count; ++i)
            maxId = std::max(maxId, specs[i].id);
        const size_t bufferCount = count == 0
            ? 0 : static_cast<size_t>(maxId) + 1;
        slabs.resize(std::max(slabs.size(), bufferCount));
        for (RankState& rank : ranks)
            rank.buffers.resize(std::max(rank.buffers.size(), bufferCount));
        for (size_t i = 0; i < count; ++i) {
            const BufferSpec& spec = specs[i];
            const size_t stride = std::max<size_t>(spec.stride, 1);
            if (stride > std::numeric_limits<size_t>::max() / rankCount)
                return app.fail(Stage::Alloc, 2);
            HostSlab& slab = slabs[spec.id];
            if (slab.ptr && slab.stride >= stride) continue;
            uint8_t* next = nullptr;
            const aclError code = aclrtMallocHost(
                reinterpret_cast<void**>(&next), stride * rankCount);
            if (code != ACL_SUCCESS || !next)
                return app.fail(Stage::Alloc, code == ACL_SUCCESS ? 1 : code);
            if (slab.ptr) aclrtFreeHost(slab.ptr);
            slab.ptr = next; slab.stride = stride;
        }
        return true;
    }

    uint8_t* host(uint32_t rank, BufferId id, size_t bytes) {
        if (rank >= rankCount || id >= slabs.size() || !slabs[id].ptr ||
            bytes > slabs[id].stride) {
            fail(rank, Stage::Alloc, 1);
            return nullptr;
        }
        return slabs[id].ptr + static_cast<size_t>(rank) * slabs[id].stride;
    }

    uint8_t* device(uint32_t rank, BufferId id, size_t bytes) {
        if (rank >= rankCount || id >= ranks[rank].buffers.size()) {
            fail(rank, Stage::Alloc, 1);
            return nullptr;
        }
        bytes = std::max<size_t>(bytes, 1);
        BufferState& buffer = ranks[rank].buffers[id];
        if (buffer.ptr && buffer.capacity >= bytes) return buffer.ptr;
        buffer.resident = false;
        if (buffer.ptr) aclrtFree(buffer.ptr);
        buffer.ptr = nullptr;
        const aclError code = aclrtMalloc(
            reinterpret_cast<void**>(&buffer.ptr), bytes,
            ACL_MEM_MALLOC_HUGE_FIRST);
        if (code != ACL_SUCCESS || !buffer.ptr) {
            ranks[rank].commandOk = false;
            fail(rank, Stage::Alloc, code == ACL_SUCCESS ? 1 : code);
            return nullptr;
        }
        buffer.capacity = bytes;
        return buffer.ptr;
    }

    bool copyIn(uint32_t rank, void* device, size_t capacity,
                const void* host, size_t bytes, aclrtStream stream) {
        const aclError code = aclrtMemcpyAsync(
            device, capacity, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream);
        return code == ACL_SUCCESS || fail(rank, Stage::CopyIn, code);
    }
    bool copyOut(uint32_t rank, void* host, size_t capacity,
                 const void* device, size_t bytes, aclrtStream stream) {
        const aclError code = aclrtMemcpyAsync(
            host, capacity, device, bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream);
        return code == ACL_SUCCESS || fail(rank, Stage::CopyOut, code);
    }
    bool copyDevice(uint32_t rank, void* dst, size_t capacity,
                    const void* src, size_t bytes, aclrtStream stream) {
        const aclError code = aclrtMemcpyAsync(
            dst, capacity, src, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        return code == ACL_SUCCESS || fail(rank, Stage::CopyIn, code);
    }
    bool sync(uint32_t rank, aclrtStream stream) {
        const aclError code = aclrtSynchronizeStream(stream);
        return code == ACL_SUCCESS || fail(rank, Stage::Sync, code);
    }
    bool launch(uint32_t rank, int32_t code) {
        return code == 0 || fail(rank, Stage::Launch, code);
    }

    bool fail(uint32_t rank, Stage stage, int32_t code) {
        if (rank < rankCount) ranks[rank].commandOk = false;
        return recordFailure(stage, code, rank);
    }
    void markResident(uint32_t rank, BufferId id) {
        if (rank < rankCount && id < ranks[rank].buffers.size())
            ranks[rank].buffers[id].resident = true;
    }

    bool run(uint32_t active, const Command& next) {
        if (!ok() || !ensureStarted()) return false;
        active = std::max<uint32_t>(1, std::min(active, rankCount));
        if (rankCount == 1) {
            ranks[0].commandOk = true;
            aclrtSetCurrentContext(ranks[0].context);
            next(0, ranks[0].context, ranks[0].stream);
            return ranks[0].commandOk;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            command = next;
            activeRanks = active;
            completed = 0;
            for (uint32_t id = 0; id < active; ++id)
                ranks[id].commandOk = true;
            ++generation;
        }
        start.notify_all();
        {
            std::unique_lock<std::mutex> lock(mutex);
            done.wait(lock, [this]() { return completed == rankCount; });
            command = Command();
        }
        for (uint32_t id = 0; id < active; ++id)
            if (!ranks[id].commandOk) return false;
        return app.ok();
    }

    void close() {
        if (stopped) return;
        stopped = true;
        if (started) {
            if (rankCount == 1) cleanupRank(0);
            else {
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    stop = true;
                    ++generation;
                }
                start.notify_all();
                for (RankState& rank : ranks)
                    if (rank.thread.joinable()) rank.thread.join();
            }
        }
        for (HostSlab& slab : slabs)
            if (slab.ptr) aclrtFreeHost(slab.ptr);
        slabs.clear();
        if (initialized) {
            const aclError code = aclFinalize();
            if (code != ACL_SUCCESS) app.fail(Stage::Cleanup, code);
            initialized = false;
        }
    }

private:
    struct BufferState {
        uint8_t* ptr; size_t capacity; bool resident;
        BufferState() : ptr(nullptr), capacity(0), resident(false) {}
    };
    struct HostSlab {
        uint8_t* ptr; size_t stride;
        HostSlab() : ptr(nullptr), stride(0) {}
    };
    struct RankState {
        aclrtContext context; aclrtStream stream; std::thread thread;
        std::vector<BufferState> buffers; bool commandOk;
        RankState()
            : context(nullptr), stream(nullptr), commandOk(true) {}
    };

    bool setupRank(uint32_t id) {
        RankState& rank = ranks[id];
        aclError code = aclrtSetDevice(static_cast<int32_t>(id));
        if (code == ACL_SUCCESS)
            code = aclrtCreateContext(&rank.context, static_cast<int32_t>(id));
        if (code == ACL_SUCCESS) code = aclrtSetCurrentContext(rank.context);
        if (code == ACL_SUCCESS) code = aclrtCreateStream(&rank.stream);
        return code == ACL_SUCCESS ||
               recordFailure(Stage::Context, code, id);
    }
    bool ensureStarted() {
        if (started) return valid;
        started = true;
        if (rankCount == 1) {
            valid = setupRank(0);
            return valid;
        }
        for (uint32_t id = 0; id < rankCount; ++id)
            ranks[id].thread =
                std::thread([this, id]() { workerMain(id); });
        {
            std::unique_lock<std::mutex> lock(mutex);
            readyCv.wait(lock, [this]() { return ready == rankCount; });
        }
        const aclError code = aclrtSetDevice(0);
        if (setupFailed || code != ACL_SUCCESS)
            valid = app.fail(Stage::Device,
                             code == ACL_SUCCESS ? 1 : code);
        return valid;
    }
    void workerMain(uint32_t id) {
        const bool setupOk = setupRank(id);
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!setupOk) setupFailed = true;
            ++ready;
        }
        readyCv.notify_one();
        size_t seen = 0;
        while (true) {
            Command next;
            uint32_t active = 0;
            {
                std::unique_lock<std::mutex> lock(mutex);
                start.wait(lock, [this, seen]() {
                    return stop || generation != seen;
                });
                if (stop) break;
                seen = generation;
                next = command;
                active = activeRanks;
            }
            if (setupOk && id < active && next) {
                aclrtSetCurrentContext(ranks[id].context);
                next(id, ranks[id].context, ranks[id].stream);
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                ++completed;
            }
            done.notify_one();
        }
        cleanupRank(id);
    }
    void cleanupRank(uint32_t id) {
        RankState& rank = ranks[id];
        if (rank.context) aclrtSetCurrentContext(rank.context);
        for (BufferState& buffer : rank.buffers)
            if (buffer.ptr) aclrtFree(buffer.ptr);
        if (rank.stream) aclrtDestroyStream(rank.stream);
        if (rank.context) aclrtDestroyContext(rank.context);
        aclrtResetDevice(static_cast<int32_t>(id));
        rank.stream = nullptr;
        rank.context = nullptr;
    }
    bool recordFailure(Stage stage, int32_t code, uint32_t rank) {
        std::lock_guard<std::mutex> lock(statusMutex);
        return app.fail(stage, code, rank);
    }

    Session& app;
    uint32_t rankCount;
    std::vector<RankState> ranks;
    std::vector<HostSlab> slabs;
    bool valid = true, initialized = false, started = false, stopped = false;
    bool stop = false, setupFailed = false;
    size_t generation = 0, completed = 0, ready = 0;
    uint32_t activeRanks = 0;
    Command command;
    std::mutex mutex, statusMutex;
    std::condition_variable start, done, readyCv;
};
} // namespace dacpp_host
)~~~";

} // namespace

HostRuntimeRequirements CollectHostRuntimeRequirements(
    const std::vector<AscendCExpressionArtifact>& artifacts,
    const AscendCFlowPlan& plan,
    const TranslationPolicy& policy) {
    HostRuntimeRequirements req;
    req.packaging = policy.hostRuntimePackaging;
    req.logLevel = policy.hostRuntimeLogLevel;
    for (const AscendCExpressionArtifact& artifact : artifacts) {
        req.merge(artifact.runtimeRequirements);
        if (artifact.routeKind == AscendCRouteKind::Vector) {
            req.require(HostRuntimeFeature::CheckedMath);
            req.require(HostRuntimeFeature::VectorTiling);
        } else {
            req.require(HostRuntimeFeature::CubeTiling);
            req.require(HostRuntimeFeature::Fp16Bridge);
        }
    }
    bool hasMulticard = false;
    for (const FlowKernelLaunch& launch : plan.kernelLaunches) {
        hasMulticard = hasMulticard || launch.multicardEnabled;
        if (launch.multicard.persistentWorkers)
            req.require(HostRuntimeFeature::PersistentLoop);
    }
    req.require(hasMulticard ? HostRuntimeFeature::MulticardRoute
                             : HostRuntimeFeature::SingleCard);
    for (const FlowHostOutput& output : plan.hostOutputs) {
        if (output.kind == FlowHostOutputKind::TensorPrint) {
            req.require(HostRuntimeFeature::TensorPrint);
            break;
        }
    }
    return req;
}

std::string EmitHostRuntimeBody(const HostRuntimeRequirements& req) {
    std::string body = CommonRuntime(req);
    if (req.has(HostRuntimeFeature::CheckedMath) ||
        req.has(HostRuntimeFeature::VectorTiling)) {
        body += CheckedMathAndTiling;
    }
    if (req.has(HostRuntimeFeature::Fp16Bridge)) body += Fp16Bridge;
    if (req.has(HostRuntimeFeature::PersistentLoop))
        body += PersistentLoopControls;
    if (req.has(HostRuntimeFeature::SingleCard)) body += SingleCardDevice;
    if (req.has(HostRuntimeFeature::MulticardRoute)) {
        body += MulticardRoute;
    }
    return body;
}

std::string EmitHostRuntimeHeader(const HostRuntimeRequirements& req) {
    std::ostringstream out;
    out << "#ifndef DACPP_ASCENDC_HOST_RUNTIME_H\n"
           "#define DACPP_ASCENDC_HOST_RUNTIME_H\n\n"
           "#define DACPP_ASCENDC_HOST_RUNTIME_VERSION 19\n\n"
           "#include <algorithm>\n"
           "#include <cstddef>\n"
           "#include <cstdint>\n"
           "#include <cstdlib>\n"
           "#include <cstring>\n"
           "#include <iostream>\n"
           "#include <limits>\n"
           "#include <string>\n"
           "#include <vector>\n";
    if (req.has(HostRuntimeFeature::MulticardRoute)) {
        out << "#include <condition_variable>\n"
               "#include <functional>\n"
               "#include <mutex>\n"
               "#include <thread>\n";
    }
    out << "\n#ifndef ASCENDC_CPU_DEBUG\n"
           "#include \"acl/acl.h\"\n"
           "#else\n"
           "#include \"tikicpulib.h\"\n"
           "#endif\n";
    out << EmitHostRuntimeBody(req);
    out << "\n#endif // DACPP_ASCENDC_HOST_RUNTIME_H\n";
    return out.str();
}

std::string EmitHostRuntimeInsertion(const HostRuntimeRequirements& req) {
    std::string result;
    if (req.packaging == HostRuntimePackaging::Compact) {
        result = "#include \"dacpp_ascendc_host_runtime.h\"\n";
    } else {
        result = EmitHostRuntimeHeader(req);
    }
    result +=
        "static_assert(DACPP_ASCENDC_HOST_RUNTIME_VERSION == 19,\n"
        "              \"regenerate Host runtime\");\n";
    return result;
}

} // namespace ascendc
} // namespace dacppTranslator
