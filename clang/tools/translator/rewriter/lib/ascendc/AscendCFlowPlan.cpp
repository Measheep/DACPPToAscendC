// AscendCFlowPlan：节点 append、metadata builder、dump 三件套。
//
// 关键不变量：
//   1. 头文件 AscendCFlowPlan.h 不引 MLIR / Clang；本 .cpp 仅引 DACPP
//      parser 头（DacppStructure.h）以读出 totalExpressions 元数据。
//   2. BuildAscendCFlowPlan 仅负责填充元数据；
//      实际的 host descriptor 由 HostFlowBuilder（AST builder）填充。
//   3. dump 文本必须稳定（不依赖 hash、迭代顺序）。

#include "ascendc/AscendCFlowPlan.h"

#include <cctype>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

#include "DacppStructure.h"

namespace dacppTranslator {
namespace ascendc {

// ============================================================
// Append helpers
// ============================================================
namespace {

template <typename Pool>
FlowStepId AppendStep(AscendCFlowPlan& plan, Pool& pool,
                      typename Pool::value_type node, FlowStepKind kind) {
    pool.push_back(std::move(node));
    FlowStep step;
    step.kind = kind;
    step.payloadIndex = pool.size() - 1;
    plan.steps.push_back(step);
    return plan.steps.size() - 1;
}

} // namespace

FlowStepId AscendCFlowPlan::addHostInput(FlowHostInput n) {
    return AppendStep(*this, hostInputs, std::move(n), FlowStepKind::HostInput);
}

FlowStepId AscendCFlowPlan::addTensorBuffer(FlowTensorBuffer n) {
    return AppendStep(*this, tensorBuffers, std::move(n), FlowStepKind::TensorBuffer);
}

FlowStepId AscendCFlowPlan::addScalarBinding(FlowScalarBinding n) {
    return AppendStep(*this, scalarBindings, std::move(n), FlowStepKind::ScalarBinding);
}

FlowStepId AscendCFlowPlan::addKernelLaunch(FlowKernelLaunch n) {
    return AppendStep(*this, kernelLaunches, std::move(n), FlowStepKind::KernelLaunch);
}

FlowStepId AscendCFlowPlan::addHostOutput(FlowHostOutput n) {
    return AppendStep(*this, hostOutputs, std::move(n), FlowStepKind::HostOutput);
}

FlowStepId AscendCFlowPlan::addCleanup(FlowCleanupStep n) {
    return AppendStep(*this, cleanups, std::move(n), FlowStepKind::Cleanup);
}

FlowStepId AscendCFlowPlan::addDiagnostic(FlowDiagnostic n) {
    return AppendStep(*this, diagnostics, std::move(n), FlowStepKind::Diagnostic);
}

FlowStepId AscendCFlowPlan::addHostLoop(FlowHostLoop n) {
    return AppendStep(*this, hostLoops, std::move(n), FlowStepKind::HostLoop);
}

FlowStepId AscendCFlowPlan::addHostBranch(FlowHostBranch n) {
    return AppendStep(*this, hostBranches, std::move(n), FlowStepKind::HostBranch);
}

FlowStepId AscendCFlowPlan::addHelperInvocation(FlowHelperInvocation n) {
    return AppendStep(*this, helperInvocations, std::move(n), FlowStepKind::HelperInvocation);
}

FlowStepId AscendCFlowPlan::addDataflowEdge(FlowExprDataflowEdge n) {
    return AppendStep(*this, dataflowEdges, std::move(n), FlowStepKind::ExprDataflowEdge);
}

FlowStepId AscendCFlowPlan::addAliasLifetime(FlowAliasLifetime n) {
    return AppendStep(*this, aliasLifetimes, std::move(n), FlowStepKind::AliasLifetime);
}

FlowStepId AscendCFlowPlan::addDacForNesting(FlowDacForNesting n) {
    return AppendStep(*this, dacForNestings, std::move(n), FlowStepKind::DacForNesting);
}

FlowStepId AscendCFlowPlan::addGeneratedCodeBlock(FlowGeneratedCodeBlock n) {
    return AppendStep(*this, generatedBlocks, std::move(n), FlowStepKind::GeneratedCodeBlock);
}

FlowStepId AscendCFlowPlan::addRawStmt(FlowRawStmt n) {
    return AppendStep(*this, rawStmts, std::move(n), FlowStepKind::Raw);
}

void AscendCFlowPlan::addGeneratedName(FlowGeneratedName n) {
    for (const FlowGeneratedName& existing : generatedNames) {
        if (existing.text == n.text && existing.kind == n.kind &&
            existing.ownerExpr == n.ownerExpr) {
            return;
        }
    }
    generatedNames.push_back(std::move(n));
}

const FlowRouteRuntimeNames* AscendCFlowPlan::findRouteRuntimeNames(
    FlowRouteKind route) const {
    const FlowRouteRuntimeNames* found = nullptr;
    for (const FlowRouteRuntimeNames& names : routeRuntimeNames) {
        if (names.route != route) continue;
        if (found) return nullptr;
        found = &names;
    }
    return found;
}

// ============================================================
// ToString
// ============================================================
#define FLOW_ENUM_CASE(val) case decltype(v)::val: return #val

const char* ToString(FlowRouteKind v) {
    switch (v) { FLOW_ENUM_CASE(Unknown); FLOW_ENUM_CASE(Vector); FLOW_ENUM_CASE(Cube); }
    return "?";
}
const char* ToString(FlowMulticardCommunicationKind v) {
    switch (v) {
        case FlowMulticardCommunicationKind::None: return "none";
        case FlowMulticardCommunicationKind::HostBoundary: return "host-boundary";
        case FlowMulticardCommunicationKind::PeerBoundary: return "peer-boundary";
        case FlowMulticardCommunicationKind::AllGather: return "all-gather";
        case FlowMulticardCommunicationKind::ReduceScatter: return "reduce-scatter";
        case FlowMulticardCommunicationKind::AllReduce: return "all-reduce";
    }
    return "?";
}
const char* ToString(FlowMulticardParamDistribution v) {
    switch (v) {
        case FlowMulticardParamDistribution::Sharded: return "sharded";
        case FlowMulticardParamDistribution::Replicated: return "replicated";
    }
    return "?";
}
const char* ToString(FlowBufferAccess v) {
    switch (v) {
        case FlowBufferAccess::Read: return "read";
        case FlowBufferAccess::Write: return "write";
        case FlowBufferAccess::ReadWrite: return "read-write";
    }
    return "?";
}
const char* ToString(FlowGeneratedNameKind v) {
    switch (v) {
        case FlowGeneratedNameKind::FrameworkLocal: return "framework";
        case FlowGeneratedNameKind::ParameterDerived: return "parameter";
        case FlowGeneratedNameKind::UserPreserved: return "user";
        case FlowGeneratedNameKind::PublicType: return "public-type";
    }
    return "?";
}
const char* ToString(FlowBufferOrigin v) {
    switch (v) { FLOW_ENUM_CASE(Unknown); FLOW_ENUM_CASE(StdVectorCtor); FLOW_ENUM_CASE(DacppSlice); FLOW_ENUM_CASE(HelperReturnAlias); }
    return "?";
}
const char* ToString(FlowHostOutputKind v) {
    switch (v) { FLOW_ENUM_CASE(Unknown); FLOW_ENUM_CASE(TensorPrint); FLOW_ENUM_CASE(StdOutStream); FLOW_ENUM_CASE(PrintfCall); FLOW_ENUM_CASE(TensorAssign); FLOW_ENUM_CASE(ScalarSelfUpdate); FLOW_ENUM_CASE(VectorElemWrite); }
    return "?";
}
const char* ToString(FlowCleanupKind v) {
    switch (v) { FLOW_ENUM_CASE(Unknown); FLOW_ENUM_CASE(HostFree); FLOW_ENUM_CASE(HostDelete); FLOW_ENUM_CASE(DeviceFree); FLOW_ENUM_CASE(HostStreamDestroy); FLOW_ENUM_CASE(AclFinalize); }
    return "?";
}
const char* ToString(FlowDiagnosticSeverity v) {
    switch (v) { FLOW_ENUM_CASE(Note); FLOW_ENUM_CASE(Warning); FLOW_ENUM_CASE(Error); }
    return "?";
}
const char* ToString(FlowHostLoopKind v) {
    switch (v) { FLOW_ENUM_CASE(For); FLOW_ENUM_CASE(While); FLOW_ENUM_CASE(DoWhile); }
    return "?";
}
const char* ToString(FlowHelperRole v) {
    switch (v) { FLOW_ENUM_CASE(Unknown); FLOW_ENUM_CASE(InitOnly); FLOW_ENUM_CASE(OutputMutating); FLOW_ENUM_CASE(TileSizeHint); FLOW_ENUM_CASE(PrintLike); }
    return "?";
}
const char* ToString(FlowExprDataflowKind v) {
    switch (v) { FLOW_ENUM_CASE(Unknown); FLOW_ENUM_CASE(HostToDevice); FLOW_ENUM_CASE(DeviceToHost); FLOW_ENUM_CASE(InternalReuse); }
    return "?";
}
const char* ToString(FlowDacForNestKind v) {
    switch (v) { FLOW_ENUM_CASE(Unknown); FLOW_ENUM_CASE(TimeStep); FLOW_ENUM_CASE(TemporalStencil); FLOW_ENUM_CASE(SpatialIteration); }
    return "?";
}
const char* ToString(FlowGeneratedCodeRole v) {
    switch (v) { FLOW_ENUM_CASE(Unknown); FLOW_ENUM_CASE(AclRuntimeInit); FLOW_ENUM_CASE(AclRuntimeFinalize); FLOW_ENUM_CASE(StreamCreate); FLOW_ENUM_CASE(StreamDestroy); FLOW_ENUM_CASE(CommonDecl); FLOW_ENUM_CASE(DeviceAlloc); FLOW_ENUM_CASE(DeviceFree); FLOW_ENUM_CASE(HostToDeviceCopy); FLOW_ENUM_CASE(DeviceToHostCopy); FLOW_ENUM_CASE(KernelLaunch); FLOW_ENUM_CASE(TemporalLoopBody); FLOW_ENUM_CASE(AuxiliaryPrelude); FLOW_ENUM_CASE(MulticardRouteSetup); FLOW_ENUM_CASE(MulticardLaunch); FLOW_ENUM_CASE(MulticardRouteCleanup); FLOW_ENUM_CASE(MulticardRuntime); }
    return "?";
}
const char* ToString(FlowStepKind v) {
    switch (v) { FLOW_ENUM_CASE(None); FLOW_ENUM_CASE(HostInput); FLOW_ENUM_CASE(TensorBuffer); FLOW_ENUM_CASE(ScalarBinding); FLOW_ENUM_CASE(KernelLaunch); FLOW_ENUM_CASE(HostOutput); FLOW_ENUM_CASE(Cleanup); FLOW_ENUM_CASE(Diagnostic); FLOW_ENUM_CASE(HostLoop); FLOW_ENUM_CASE(HostBranch); FLOW_ENUM_CASE(HelperInvocation); FLOW_ENUM_CASE(ExprDataflowEdge); FLOW_ENUM_CASE(AliasLifetime); FLOW_ENUM_CASE(DacForNesting); FLOW_ENUM_CASE(GeneratedCodeBlock); FLOW_ENUM_CASE(Raw); }
    return "?";
}

#undef FLOW_ENUM_CASE

// ============================================================
// 文本辅助
// ============================================================
namespace {


std::string Truncate(const std::string& s, std::size_t max_len) {
    if (s.size() <= max_len) return s;
    return s.substr(0, max_len) + "...";
}

std::string DumpFileName(const std::string& file) {
    // SourceManager keeps the full path for diagnostics, but a FlowPlan dump is
    // a reproducible test/debug artifact.  Accept both separator styles so a
    // dump generated from a #line path is stable on every host platform.
    const std::size_t separator = file.find_last_of("/\\");
    return separator == std::string::npos ? file : file.substr(separator + 1);
}

std::string DumpLoc(const FlowSourceLoc& loc) {
    if (!loc.isValid()) return "?";
    std::ostringstream os;
    if (!loc.file.empty()) os << DumpFileName(loc.file) << ":";
    os << loc.line;
    if (loc.column > 0) os << ":" << loc.column;
    return os.str();
}

std::string JoinCSV(const std::vector<std::string>& xs) {
    std::string out = "[";
    for (std::size_t i = 0; i < xs.size(); ++i) {
        if (i) out += ", ";
        out += xs[i];
    }
    out += "]";
    return out;
}

std::string JoinStepIds(const std::vector<FlowStepId>& xs) {
    std::string out = "[";
    for (std::size_t i = 0; i < xs.size(); ++i) {
        if (i) out += ", ";
        out += "#" + std::to_string(xs[i]);
    }
    out += "]";
    return out;
}

} // namespace

// ============================================================
// Builder：仅负责填充 plan 的元数据。
// 实际 host descriptor 由 HostFlowBuilder（AST builder）通过 add* 接口填充，
// 在更上层的 driver 流程中调用，不在这里生成。
// ============================================================
AscendCFlowPlan BuildAscendCFlowPlan(const AscendCFlowPlanBuildInput& in) {
    AscendCFlowPlan plan;
    plan.builderRan = true;

    if (in.sourceFileHint) plan.sourceFileHint = *in.sourceFileHint;
    if (in.dacppFile) plan.totalExpressions = in.dacppFile->getNumExpression();

    if (in.mixedRouteConflict) {
        FlowDiagnostic d;
        d.severity = FlowDiagnosticSeverity::Error;
        d.code = "hostflow.mixed_route_conflict";
        d.message = in.mixedRouteReason.empty()
                        ? "mixed Vector/Cube route order conflict"
                        : in.mixedRouteReason;
        plan.addDiagnostic(std::move(d));
        plan.partial = true;
    }

    return plan;
}

// ============================================================
// Dump
// ============================================================
namespace {

void DumpStep(std::ostringstream& os,
              const AscendCFlowPlan& plan,
              std::size_t stepIdx) {
    const FlowStep& s = plan.steps[stepIdx];
    os << "step#" << stepIdx << "  " << ToString(s.kind) << "  ";

    switch (s.kind) {
        case FlowStepKind::HostInput: {
            if (!s.hasValidPayload(plan.hostInputs.size())) { os << "<bad-ref>"; break; }
            const auto& n = plan.hostInputs[s.payloadIndex];
            os << "name=" << (n.name.empty() ? "?" : n.name)
               << " type=" << (n.declTypeText.empty() ? "?" : n.declTypeText)
               << " storage=" << (n.storageClass.empty() ? "-" : n.storageClass)
               << " const=" << (n.isConst ? "1" : "0")
               << " constexpr=" << (n.isConstexpr ? "1" : "0")
               << " global=" << (n.isHostGlobal ? "1" : "0")
               << " arrayExtent=" << (n.arrayExtentExpr.empty() ? "-" : n.arrayExtentExpr)
               << " init=" << Truncate(n.initializerText, 80)
               << " loc=" << DumpLoc(n.location);
            break;
        }
        case FlowStepKind::TensorBuffer: {
            if (!s.hasValidPayload(plan.tensorBuffers.size())) { os << "<bad-ref>"; break; }
            const auto& n = plan.tensorBuffers[s.payloadIndex];
            os << "name=" << (n.name.empty() ? "?" : n.name)
               << " container=" << (n.dacppContainerText.empty() ? "std::vector" : n.dacppContainerText)
               << " origin=" << ToString(n.origin)
               << " elem=" << (n.elementTypeText.empty() ? "?" : n.elementTypeText)
               << " shape=" << JoinCSV(n.shapeExprs);
            if (!n.sliceRootBufferName.empty())
                os << " sliceRoot=" << n.sliceRootBufferName;
            os << " loc=" << DumpLoc(n.location);
            break;
        }
        case FlowStepKind::ScalarBinding: {
            if (!s.hasValidPayload(plan.scalarBindings.size())) { os << "<bad-ref>"; break; }
            const auto& n = plan.scalarBindings[s.payloadIndex];
            os << "name=" << n.name << " dtype=" << n.dtypeText
               << " value=" << Truncate(n.valueExprText, 80)
               << " expr=" << n.ownerExpr;
            break;
        }
        case FlowStepKind::KernelLaunch: {
            if (!s.hasValidPayload(plan.kernelLaunches.size())) { os << "<bad-ref>"; break; }
            const auto& n = plan.kernelLaunches[s.payloadIndex];
            os << "expr=" << n.expressionIndex
               << " route=" << ToString(n.route)
               << " kernel=" << (n.kernelFunctionName.empty() ? "?" : n.kernelFunctionName)
               << " file=" << (n.kernelArtifactFileName.empty() ? "?" : n.kernelArtifactFileName)
               << " params=" << JoinCSV(n.paramOrder)
               << " scalars=" << JoinCSV(n.scalarParams)
               << " blockDim=" << (n.blockDimExpr.empty() ? "?" : n.blockDimExpr)
               << " tiling=" << (n.tilingDataSymbol.empty() ? "-" : n.tilingDataSymbol)
               << " multicard=" << (n.multicardEnabled ? "1" : "0")
               << " commonDecls=" << JoinStepIds(n.commonDeclStepIds)
               << " alloc=" << JoinStepIds(n.allocStepIds)
               << " copyIn=" << JoinStepIds(n.copyInStepIds)
               << " copyOut=" << JoinStepIds(n.copyOutStepIds)
               << " free=" << JoinStepIds(n.freeStepIds)
               << " postLaunch=" << JoinStepIds(n.postLaunchStepIds);
            if (n.multicardEnabled) {
                os << "\n            multicard rankDim="
                   << (n.multicard.rankDimExpr.empty() ? "?" : n.multicard.rankDimExpr)
                   << " requestedRanks="
                   << (n.multicard.requestedRankDimExpr.empty()
                           ? (n.multicard.rankDimExpr.empty() ? "?" : n.multicard.rankDimExpr)
                           : n.multicard.requestedRankDimExpr)
                   << " activeRanks="
                   << (n.multicard.activeRankDimExpr.empty()
                           ? "runtime" : n.multicard.activeRankDimExpr)
                   << " env="
                   << (n.multicard.rankDimEnvName.empty() ? "-" : n.multicard.rankDimEnvName)
                   << " coresPerDevice="
                   << (n.multicard.coreNumExpr.empty() ? "?" : n.multicard.coreNumExpr)
                   << " topology="
                   << (n.multicard.requestedRankDimExpr.empty()
                           ? (n.multicard.rankDimExpr.empty() ? "?" : n.multicard.rankDimExpr)
                           : n.multicard.requestedRankDimExpr)
                   << "x"
                   << (n.multicard.coreNumExpr.empty() ? "?" : n.multicard.coreNumExpr)
                   << " deviceMap="
                   << (n.multicard.deviceMapExpr.empty() ? "-" : n.multicard.deviceMapExpr)
                   << " shard="
                   << (n.multicard.shardDim.empty() ? "-" : n.multicard.shardDim)
                   << " shardAxis="
                   << (n.multicard.shardAxis.empty() ? "-" : n.multicard.shardAxis)
                   << " logical="
                   << (n.multicard.logicalLengthExpr.empty() ? "-" : n.multicard.logicalLengthExpr)
                   << " aligned="
                   << (n.multicard.alignedLengthExpr.empty() ? "-" : n.multicard.alignedLengthExpr)
                   << " halo=[" << n.multicard.readHaloLeft << ","
                   << n.multicard.readHaloRight << "]"
                   << " strategy="
                   << (n.multicard.strategy.empty() ? "-" : n.multicard.strategy)
                   << " replicatedParams="
                   << (n.multicard.replicatedParams.empty() ? "-" : n.multicard.replicatedParams)
                   << " shardedParams="
                   << (n.multicard.shardedParams.empty() ? "-" : n.multicard.shardedParams)
                   << " inputReplicated="
                   << (n.multicard.inputReplicated ? "1" : "0")
                   << " outputSharded="
                   << (n.multicard.outputSharded ? "1" : "0")
                   << " persistentWorkers="
                   << (n.multicard.persistentWorkers ? "1" : "0")
                   << " communication=" << ToString(n.multicard.communication)
                   << " resourceScope="
                   << (n.multicard.resourceScope.empty() ? "-" : n.multicard.resourceScope)
                   << " workerLifetime="
                   << (n.multicard.workerLifetime.empty() ? "-" : n.multicard.workerLifetime)
                   << " copyMode="
                   << (n.multicard.copyMode.empty() ? "-" : n.multicard.copyMode)
                   << " rankLaunch="
                   << (n.rankLaunchFunctionName.empty() ? "-" : n.rankLaunchFunctionName)
                   << " perRankParams=" << JoinCSV(n.perRankParamOrder);
                if (!n.internalReuseBufferIds.empty()) {
                    os << " internalReuseIds=[";
                    for (std::size_t id = 0;
                         id < n.internalReuseBufferIds.size(); ++id) {
                        if (id) os << ", ";
                        os << n.internalReuseBufferIds[id];
                    }
                    os << "]";
                }
                for (const FlowMulticardParamPartition& p :
                         n.multicard.paramPartitions) {
                    os << "\n                param name="
                       << (p.name.empty() ? "?" : p.name)
                       << " bufferId="
                       << (p.bufferId == kInvalidFlowBufferId
                               ? std::string("-")
                               : std::to_string(p.bufferId))
                       << " distribution=" << ToString(p.distribution)
                       << " axis=" << (p.shardAxis.empty() ? "-" : p.shardAxis)
                       << " halo=[" << p.readHaloLeft << "," << p.readHaloRight << "]"
                       << " rw=" << p.readWriteMode
                       << " deviceResident=" << (p.deviceResident ? "1" : "0");
                }
            }
            for (const FlowRuntimeBuffer& buffer : n.runtimeBuffers) {
                os << "\n            buffer#"
                   << (buffer.id == kInvalidFlowBufferId
                           ? std::string("?")
                           : std::to_string(buffer.id))
                   << " name="
                   << (buffer.logicalName.empty() ? "?" : buffer.logicalName)
                   << " access=" << ToString(buffer.access)
                   << " dist=" << ToString(buffer.distribution)
                   << " halo=[" << buffer.leftHalo << ","
                   << buffer.rightHalo << "]"
                   << " resident=" << (buffer.producerResident ? "1" : "0")
                   << " count="
                   << (buffer.logicalCountExpr.empty()
                           ? "-" : buffer.logicalCountExpr)
                   << " bytes="
                   << (buffer.byteCountExpr.empty()
                           ? "-" : buffer.byteCountExpr);
            }
            if (!n.launchDescriptorText.empty())
                os << "\n            descriptor=" << Truncate(n.launchDescriptorText, 200);
            break;
        }
        case FlowStepKind::HostOutput: {
            if (!s.hasValidPayload(plan.hostOutputs.size())) { os << "<bad-ref>"; break; }
            const auto& n = plan.hostOutputs[s.payloadIndex];
            os << "kind=" << ToString(n.kind)
               << " target=" << Truncate(n.targetExprText, 80)
               << " value=" << Truncate(n.valueExprText, 80);
            if (!n.streamName.empty())
                os << " stream=" << n.streamName
                   << " ops=" << JoinCSV(n.streamOperands);
            if (!n.printfCallee.empty())
                os << " printf=" << n.printfCallee
                   << " args=" << JoinCSV(n.printfArgs);
            if (!n.compoundOp.empty())
                os << " op=" << n.compoundOp;
            if (!n.generatedCode.empty())
                os << " gen=" << Truncate(n.generatedCode, 120);
            break;
        }
        case FlowStepKind::Cleanup: {
            if (!s.hasValidPayload(plan.cleanups.size())) { os << "<bad-ref>"; break; }
            const auto& n = plan.cleanups[s.payloadIndex];
            os << "kind=" << ToString(n.kind)
               << " target=" << Truncate(n.targetExprText, 80);
            break;
        }
        case FlowStepKind::Diagnostic: {
            if (!s.hasValidPayload(plan.diagnostics.size())) { os << "<bad-ref>"; break; }
            const auto& n = plan.diagnostics[s.payloadIndex];
            os << "severity=" << ToString(n.severity)
               << " code=" << (n.code.empty() ? "?" : n.code)
               << " message=" << Truncate(n.message, 200);
            if (n.relatedExpr != kInvalidFlowExprIndex)
                os << " expr=" << n.relatedExpr;
            break;
        }
        case FlowStepKind::HostLoop: {
            if (!s.hasValidPayload(plan.hostLoops.size())) { os << "<bad-ref>"; break; }
            const auto& n = plan.hostLoops[s.payloadIndex];
            os << "kind=" << ToString(n.kind)
               << " induct=" << (n.inductionVar.empty() ? "?" : n.inductionVar);
            if (n.initIsDecl) {
                os << " init=[decl " << n.initTypeText << " " << n.initVarName;
                if (!n.initValueExpr.empty()) os << "=" << n.initValueExpr;
                os << "]";
            } else if (n.initIsAssignment) {
                os << " init=[assign " << n.initAssignTargetExpr << "=" << n.initAssignValueExpr << "]";
            }
            os << " upper=" << n.upperBoundExpr
               << " step=" << n.stepExpr
               << " cond=" << Truncate(n.conditionText, 80)
               << " kernelInside=" << (n.containsKernelLaunch ? "1" : "0")
               << " bodySteps=" << n.bodyStepIds.size();
            break;
        }
        case FlowStepKind::HostBranch: {
            if (!s.hasValidPayload(plan.hostBranches.size())) { os << "<bad-ref>"; break; }
            const auto& n = plan.hostBranches[s.payloadIndex];
            os << "cond=" << Truncate(n.conditionText, 80)
               << " then=" << n.thenStepIds.size()
               << " else=" << n.elseStepIds.size();
            break;
        }
        case FlowStepKind::HelperInvocation: {
            if (!s.hasValidPayload(plan.helperInvocations.size())) { os << "<bad-ref>"; break; }
            const auto& n = plan.helperInvocations[s.payloadIndex];
            os << "name=" << (n.functionName.empty() ? "?" : n.functionName)
               << " role=" << ToString(n.role)
               << " args=" << JoinCSV(n.argTexts)
               << " returns=" << (n.returnsValue ? "1" : "0")
               << " target=" << (n.resultTargetName.empty() ? "-" : n.resultTargetName);
            break;
        }
        case FlowStepKind::ExprDataflowEdge: {
            if (!s.hasValidPayload(plan.dataflowEdges.size())) { os << "<bad-ref>"; break; }
            const auto& n = plan.dataflowEdges[s.payloadIndex];
            os << "from=expr#" << n.producerExpr
               << " to=expr#" << n.consumerExpr
               << " kind=" << ToString(n.kind)
               << " buffer=" << (n.bufferName.empty() ? "?" : n.bufferName);
            break;
        }
        case FlowStepKind::AliasLifetime: {
            if (!s.hasValidPayload(plan.aliasLifetimes.size())) { os << "<bad-ref>"; break; }
            const auto& n = plan.aliasLifetimes[s.payloadIndex];
            os << "alias=" << n.aliasName
               << " root=" << n.rootBufferName
               << " elem=" << n.elementTypeText
               << " rank=" << n.rank
               << " shape=" << JoinCSV(n.shapeExprs)
               << " stride=" << JoinCSV(n.strideExprs)
               << " offset=" << (n.offsetExpr.empty() ? "0" : n.offsetExpr);
            break;
        }
        case FlowStepKind::DacForNesting: {
            if (!s.hasValidPayload(plan.dacForNestings.size())) { os << "<bad-ref>"; break; }
            const auto& n = plan.dacForNestings[s.payloadIndex];
            os << "kind=" << ToString(n.kind)
               << " loopStep=" << n.enclosingLoopStep
               << " exprs=" << n.enclosedExpressions.size();
            if (!n.stateTensorName.empty())
                os << " state=" << n.stateTensorName
                   << " in=" << n.inputSliceVar << "@t" << n.inputTimeOffset
                   << " out=" << n.outputSliceVar << "@t" << n.outputTimeOffset;
            break;
        }
        case FlowStepKind::GeneratedCodeBlock: {
            if (!s.hasValidPayload(plan.generatedBlocks.size())) { os << "<bad-ref>"; break; }
            const auto& n = plan.generatedBlocks[s.payloadIndex];
            os << "role=" << ToString(n.role)
               << " label=" << (n.label.empty() ? "?" : n.label)
               << " expr=" << n.ownerExpr
               << " route=" << ToString(n.route)
               << " code=" << Truncate(n.code, 160);
            break;
        }
        case FlowStepKind::Raw: {
            if (!s.hasValidPayload(plan.rawStmts.size())) { os << "<bad-ref>"; break; }
            const auto& n = plan.rawStmts[s.payloadIndex];
            os << "owner=" << (n.ownerFunctionName.empty() ? "?" : n.ownerFunctionName)
               << " text=" << Truncate(n.text, 200)
               << " # diagnostic-only; emitter must not consume";
            break;
        }
        case FlowStepKind::None:
            os << "<none>";
            break;
    }

    if (!s.note.empty()) os << " # " << s.note;
    os << '\n';
}

} // namespace

std::string DumpAscendCFlowPlan(const AscendCFlowPlan& plan) {
    std::ostringstream os;
    os << "AscendCFlowPlan dump\n";
    os << "  source=" << (plan.sourceFileHint.empty() ? "<unset>" : plan.sourceFileHint) << "\n";
    os << "  builderRan=" << (plan.builderRan ? "1" : "0")
       << " partial=" << (plan.partial ? "1" : "0") << "\n";
    os << "  expressions=" << plan.totalExpressions
       << " vectorRoutes=" << plan.vectorExpressionCount
       << " cubeRoutes=" << plan.cubeExpressionCount << "\n";
    os << "  runtimeFeatureMask=" << plan.runtimeFeatureMask << "\n";
    for (const FlowRouteRuntimeNames& names : plan.routeRuntimeNames) {
        os << "  runtime_names route=" << ToString(names.route)
           << " session=" << names.session
           << " rank=" << names.rankCount
           << " routeObject=" << names.routeObject
           << " device=" << names.device
           << " stream=" << names.stream << "\n";
    }
    for (const FlowGeneratedName& name : plan.generatedNames) {
        os << "  generated_name text=" << name.text
           << " kind=" << ToString(name.kind)
           << " expr=" << name.ownerExpr << "\n";
    }
    os << "  pools:"
       << " hostInputs=" << plan.hostInputs.size()
       << " tensorBuffers=" << plan.tensorBuffers.size()
       << " scalarBindings=" << plan.scalarBindings.size()
       << " kernelLaunches=" << plan.kernelLaunches.size()
       << " hostOutputs=" << plan.hostOutputs.size()
       << " cleanups=" << plan.cleanups.size()
       << " diagnostics=" << plan.diagnostics.size()
       << " hostLoops=" << plan.hostLoops.size()
       << " hostBranches=" << plan.hostBranches.size()
       << " helpers=" << plan.helperInvocations.size()
       << " edges=" << plan.dataflowEdges.size()
       << " aliases=" << plan.aliasLifetimes.size()
       << " dacForNests=" << plan.dacForNestings.size()
       << " generatedBlocks=" << plan.generatedBlocks.size()
       << " rawStmts=" << plan.rawStmts.size() << "\n";
    os << "  steps (" << plan.steps.size() << "):\n";
    for (std::size_t i = 0; i < plan.steps.size(); ++i) {
        os << "    ";
        DumpStep(os, plan, i);
    }
    return os.str();
}

} // namespace ascendc
} // namespace dacppTranslator
