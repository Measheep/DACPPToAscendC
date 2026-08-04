#ifndef DACPP_ASCENDC_HOST_RUNTIME_H
#define DACPP_ASCENDC_HOST_RUNTIME_H

#define DACPP_ASCENDC_HOST_RUNTIME_VERSION 19

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#else
#include "tikicpulib.h"
#endif

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
    static const int kLogLevel = 1;
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

#endif // DACPP_ASCENDC_HOST_RUNTIME_H
