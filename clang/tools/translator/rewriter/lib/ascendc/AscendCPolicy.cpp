#include "ascendc/AscendCPolicy.h"

#include "DacppStructure.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace dacppTranslator {
namespace ascendc {

namespace {

bool IsTruthyEnv(const char* value) {
    if (!value) {
        return false;
    }
    std::string text(value);
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text == "1" || text == "true" || text == "on" || text == "yes";
}

bool HasProcessFlag(const std::vector<std::string>& args, const std::string& flag) {
    return std::find(args.begin(), args.end(), flag) != args.end();
}

std::string ParseProcessStringFlag(const std::vector<std::string>& args,
                                   const std::string& flag,
                                   std::string fallback) {
    const std::string prefix = flag + "=";
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg.rfind(prefix, 0) == 0) {
            return arg.substr(prefix.size());
        }
        if (arg == flag && i + 1 < args.size()) {
            return args[i + 1];
        }
    }
    return fallback;
}

std::optional<std::string> FindProcessFlagValue(
    const std::vector<std::string>& args,
    const std::string& flag) {
    const std::string prefix = flag + "=";
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (args[i].rfind(prefix, 0) == 0) {
            return args[i].substr(prefix.size());
        }
        if (args[i] == flag && i + 1 < args.size()) {
            return args[i + 1];
        }
    }
    return std::nullopt;
}

std::optional<size_t> ParsePositiveSize(const std::string& text) {
    if (text.empty() || text.front() == '-') {
        return std::nullopt;
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
    if (errno == ERANGE || end == text.c_str() || *end != '\0' || parsed == 0 ||
        parsed > std::numeric_limits<size_t>::max()) {
        return std::nullopt;
    }
    return static_cast<size_t>(parsed);
}

std::optional<BufferDepthMode> ParseBufferDepth(const std::string& value) {
    std::string normalized = value;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (normalized == "auto") return BufferDepthMode::Auto;
    if (normalized == "1") return BufferDepthMode::Single;
    if (normalized == "2") return BufferDepthMode::Double;
    return std::nullopt;
}

bool IsValidEnvName(const std::string& name) {
    if (name.empty()) {
        return false;
    }
    unsigned char first = static_cast<unsigned char>(name.front());
    if (!(std::isalpha(first) || name.front() == '_')) {
        return false;
    }
    for (char c : name) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (!(std::isalnum(uc) || c == '_')) {
            return false;
        }
    }
    return true;
}

void SetConfigErrorOnce(TranslationPolicy& policy,
                        const std::string& message) {
    if (!policy.valid) return;
    policy.valid = false;
    policy.firstConfigError = message;
}

std::vector<std::string> ReadProcessArgs() {
    std::vector<std::string> args;
#if defined(__linux__)
    const char* raw = std::getenv("_");
    (void)raw;
    FILE* file = std::fopen("/proc/self/cmdline", "rb");
    if (!file) {
        return args;
    }
    std::string current;
    int ch = 0;
    while ((ch = std::fgetc(file)) != EOF) {
        if (ch == '\0') {
            args.push_back(current);
            current.clear();
        } else {
            current.push_back(static_cast<char>(ch));
        }
    }
    if (!current.empty()) {
        args.push_back(current);
    }
    std::fclose(file);
#endif
    return args;
}

} // namespace

TranslationPolicy::TranslationPolicy()
    : stdTypeAliases({
          {"int", "int32_t"},
          {"unsigned int", "uint32_t"},
          {"short", "int16_t"},
          {"unsigned short", "uint16_t"},
          {"char", "int8_t"},
          {"unsigned char", "uint8_t"},
          {"long", "int64_t"},
          {"unsigned long", "uint64_t"},
      }),
      kernelTypeAliases({
          {"int", "int32_t"},
          {"unsigned int", "uint32_t"},
          {"short", "int16_t"},
          {"unsigned short", "uint16_t"},
          {"char", "int8_t"},
          {"unsigned char", "uint8_t"},
          {"long", "int64_t"},
          {"unsigned long", "uint64_t"},
      }),
      floatLikeTypes({"half", "float16", "float", "double"}) {
    const std::vector<std::string> args = ReadProcessArgs();
    debugOutput = HasProcessFlag(args, "--debug") ||
                  HasProcessFlag(args, "--dump-parser");
    emitTilingDiagnostics = debugOutput;
    hostRuntimePackaging =
        HasProcessFlag(args, "--ascendc-embed-host-runtime")
            ? HostRuntimePackaging::Embedded
            : HostRuntimePackaging::Compact;
    const std::string hostLog =
        ParseProcessStringFlag(args, "--ascendc-host-log", "error");
    if (hostLog == "none") {
        hostRuntimeLogLevel = HostRuntimeLogLevel::None;
    } else if (hostLog == "error") {
        hostRuntimeLogLevel = HostRuntimeLogLevel::Error;
    } else if (hostLog == "debug") {
        hostRuntimeLogLevel = HostRuntimeLogLevel::Debug;
    } else {
        SetConfigErrorOnce(
            *this,
            "invalid --ascendc-host-log='" + hostLog +
                "'; expected none, error, or debug");
    }
    auto ResolvePositiveInt = [&](const char* envName,
                                  const char* flagName,
                                  int defaultValue,
                                  std::string* source) {
        int value = defaultValue;
        if (const char* env = std::getenv(envName)) {
            const std::optional<size_t> parsed = ParsePositiveSize(env);
            if (parsed && *parsed <= static_cast<size_t>(std::numeric_limits<int>::max())) {
                value = static_cast<int>(*parsed);
                if (source) *source = "environment";
            } else {
                SetConfigErrorOnce(
                    *this, "invalid " + std::string(envName) + "='" + env + "'");
            }
        }
        const std::optional<std::string> cli = FindProcessFlagValue(args, flagName);
        if (cli) {
            const std::optional<size_t> parsed = ParsePositiveSize(*cli);
            if (parsed && *parsed <= static_cast<size_t>(std::numeric_limits<int>::max())) {
                value = static_cast<int>(*parsed);
                if (source) *source = "CLI";
            } else {
                SetConfigErrorOnce(
                    *this, "invalid " + std::string(flagName) + "='" + *cli + "'");
                value = defaultValue;
                if (source) *source = "default";
            }
        }
        return value;
    };
    defaultCoreNum = ResolvePositiveInt(
        "DACPP_ASCENDC_CORE_NUM", "--ascendc-core-num", 8, &coreNumSource);
    enableMulticard =
        HasProcessFlag(args, "--ascendc-multicard") ||
        IsTruthyEnv(std::getenv("DACPP_ASCENDC_MULTICARD"));
    std::string envNameFallback = rankDimEnvName;
    const char* envNameFromEnv = std::getenv("DACPP_ASCENDC_RANK_ENV");
    if (envNameFromEnv && IsValidEnvName(envNameFromEnv)) {
        envNameFallback = envNameFromEnv;
    } else if (envNameFromEnv) {
        SetConfigErrorOnce(
            *this, "invalid DACPP_ASCENDC_RANK_ENV='" +
                std::string(envNameFromEnv) + "'");
    }
    std::string parsedEnvName =
        ParseProcessStringFlag(args, "--ascendc-rank-env", envNameFallback);
    if (IsValidEnvName(parsedEnvName)) {
        rankDimEnvName = parsedEnvName;
    } else {
        rankDimEnvName = "DACPP_ASCENDC_RANK_DIM";
        SetConfigErrorOnce(
            *this, "invalid --ascendc-rank-env='" + parsedEnvName + "'");
    }
    const std::optional<std::string> rankDimCli =
        FindProcessFlagValue(args, "--ascendc-rank-dim");
    if (enableMulticard) {
        rankDimSource = "multicard-default";
        requestedRankDim = ResolvePositiveInt(
            rankDimEnvName.c_str(), "--ascendc-rank-dim",
            multicardDefaultRankDim, &rankDimSource);
        rankDimExplicit = rankDimCli.has_value() ||
                          std::getenv(rankDimEnvName.c_str()) != nullptr;
        if (rankDimSource == "default") {
            rankDimSource = "multicard-default";
        }
    } else {
        requestedRankDim = 1;
        rankDimSource = "single-device";
        const char* rankDimEnv = std::getenv(rankDimEnvName.c_str());
        if (rankDimCli || rankDimEnv) {
            SetConfigErrorOnce(
                *this,
                "rank dimension requires --ascendc-multicard");
        }
    }

    auto ResolvePositiveSize = [&](const char* envName,
                                   const char* flagName,
                                   size_t defaultValue,
                                   std::string& source) {
        size_t value = defaultValue;
        if (const char* env = std::getenv(envName)) {
            const std::optional<size_t> parsed = ParsePositiveSize(env);
            if (parsed) {
                value = *parsed;
                source = "environment";
            } else {
                SetConfigErrorOnce(
                    *this, "invalid " + std::string(envName) + "='" + env + "'");
            }
        }
        const std::optional<std::string> cli = FindProcessFlagValue(args, flagName);
        if (cli) {
            const std::optional<size_t> parsed = ParsePositiveSize(*cli);
            if (parsed) {
                value = *parsed;
                source = "CLI";
            } else {
                SetConfigErrorOnce(
                    *this, "invalid " + std::string(flagName) + "='" + *cli + "'");
                value = defaultValue;
                source = "default";
            }
        }
        return value;
    };

    vectorUbBudgetBytes = ResolvePositiveSize(
        "DACPP_ASCENDC_UB_BUDGET_BYTES", "--ascendc-ub-budget-bytes",
        128 * 1024, vectorUbBudgetSource);
    vectorUbReserveBytes = ResolvePositiveSize(
        "DACPP_ASCENDC_UB_RESERVE_BYTES", "--ascendc-ub-reserve-bytes",
        16 * 1024, vectorUbReserveSource);

    auto ApplyBufferDepth = [&](const std::string& raw,
                                const std::string& name,
                                const std::string& source) {
        const std::optional<BufferDepthMode> parsed = ParseBufferDepth(raw);
        if (!parsed) {
            SetConfigErrorOnce(
                *this, "invalid " + name + "='" + raw + "'");
            bufferDepthMode = BufferDepthMode::Auto;
            bufferDepthSource = "default";
            return;
        }
        bufferDepthMode = *parsed;
        bufferDepthSource = source;
    };
    if (const char* env = std::getenv("DACPP_ASCENDC_BUFFER_NUM")) {
        ApplyBufferDepth(env, "DACPP_ASCENDC_BUFFER_NUM", "environment");
    }
    if (const std::optional<std::string> cli =
            FindProcessFlagValue(args, "--ascendc-buffer-num")) {
        ApplyBufferDepth(*cli, "--ascendc-buffer-num", "CLI");
    }

    const size_t maxTile = ResolvePositiveSize(
        "DACPP_ASCENDC_MAX_TILE_ELEMENTS", "--ascendc-max-tile-elements",
        std::numeric_limits<uint32_t>::max(), maxTileElementsSource);
    if (maxTile == std::numeric_limits<uint32_t>::max() &&
        maxTileElementsSource == "default") {
        maxTileElementsOverride = 0;
    } else if (maxTile > std::numeric_limits<uint32_t>::max()) {
        SetConfigErrorOnce(
            *this, "max tile elements exceeds uint32_t");
        maxTileElementsOverride = 0;
        maxTileElementsSource = "default";
    } else {
        maxTileElementsOverride = static_cast<uint32_t>(maxTile);
    }

    if (vectorUbReserveBytes >= vectorUbBudgetBytes) {
        SetConfigErrorOnce(
            *this, "AscendC UB reserve must be smaller than UB budget");
    }
}

const TranslationPolicy& GetTranslationPolicy() {
    static const TranslationPolicy policy;
    return policy;
}

AscendCRewriteContext::AscendCRewriteContext(DacppFile* file, clang::Rewriter* rw)
    : dacppFile(file),
      rewriter(rw),
      sourceText(rw ? rw->getSourceMgr()
                          .getBufferData(rw->getSourceMgr().getMainFileID())
                          .str()
                    : std::string()),
      policy(GetTranslationPolicy()),
      mlirOptions(dacppTranslator::ascendc_mlir::ResolveCalcMlirPipelineOptions()) {}

} // namespace ascendc
} // namespace dacppTranslator
