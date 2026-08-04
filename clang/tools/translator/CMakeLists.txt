set(LLVM_LINK_COMPONENTS support)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dacppLib/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/parser/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/rewriter/include)
include_directories(/usr/include)

find_package(Threads REQUIRED)

set(DACPPLIB_SOURCES
  dacppLib/lib/exception.cpp
  dacppLib/include/ReconTensor.h
)
set(PARSER_SOURCES
  parser/lib/ASTParse.cpp
  parser/lib/Calc.cpp
  parser/lib/DacppStructure.cpp
  parser/lib/Param.cpp
  parser/lib/Shell.cpp
  parser/lib/Split.cpp
  parser/lib/Dacfor.cpp
)

set(REWRITER_SOURCES
  rewriter/lib/universal_template.cpp
  rewriter/lib/Rewriter_AscendC.cpp
  rewriter/lib/ascendc/AscendCFlowPlan.cpp
  rewriter/lib/ascendc/AscendCKernelSupport.cpp
  rewriter/lib/ascendc/AscendCPolicy.cpp
  rewriter/lib/ascendc/AscendCRewriteDriver.cpp
  rewriter/lib/ascendc/AscendCRuntimeBinding.cpp
  rewriter/lib/ascendc/AscendCSourceUtils.cpp
  rewriter/lib/ascendc/AscendCTensorAlias.cpp
  rewriter/lib/ascendc/AscendCTypeMapping.cpp
  rewriter/lib/ascendc/HostFlowBuilder.cpp
  rewriter/lib/ascendc/HostFlowEmitter.cpp
  rewriter/lib/ascendc/HostCodeNames.cpp
  rewriter/lib/ascendc/HostRuntimeEmitter.cpp
  rewriter/lib/ascendc/VectorTiling.cpp
  rewriter/lib/ascendc_template.cpp
  rewriter/lib/ascendc_mlir/CalcContext.cpp
  rewriter/lib/ascendc_mlir/CalcShellInfo.cpp
  rewriter/lib/ascendc_mlir/CalcAstToMlir.cpp
  rewriter/lib/ascendc_mlir/CalcExprGraph.cpp
  rewriter/lib/ascendc_mlir/CalcMlirUtils.cpp
  rewriter/lib/ascendc_mlir/CalcAnalysis.cpp
  rewriter/lib/ascendc_mlir/AscendCPlan.cpp
  rewriter/lib/ascendc_mlir/AscendCEmitter.cpp
  rewriter/lib/ascendc_mlir/CalcMlirPipeline.cpp
)

add_library(DacppLib ${DACPPLIB_SOURCES})
add_library(ParserLib ${PARSER_SOURCES})
add_library(RewriterLib ${REWRITER_SOURCES})

add_clang_executable(translator
  translator.cpp
)

# Vector resource planning reports validation/overflow failures with typed
# exceptions; clang executables otherwise inherit LLVM's -fno-exceptions.
target_compile_options(translator PRIVATE -fexceptions)


set_target_properties(translator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator
)


clang_target_link_libraries(translator
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  DacppLib
  ParserLib
  RewriterLib
)

target_link_libraries(translator
  PRIVATE
  Threads::Threads
)

if(BUILD_TESTING)
  enable_testing()
  add_executable(vector_tiling_test
    rewriter/tests/VectorTilingTest.cpp
    rewriter/lib/ascendc/VectorTiling.cpp
  )
  add_test(NAME translator_vector_tiling COMMAND vector_tiling_test)

  add_executable(host_runtime_emitter_test
    rewriter/tests/HostRuntimeEmitterTest.cpp
    rewriter/lib/ascendc/HostRuntimeEmitter.cpp
  )
  add_test(NAME translator_host_runtime_emitter
           COMMAND host_runtime_emitter_test)

  add_executable(ascendc_flow_plan_test
    rewriter/tests/AscendCFlowPlanTest.cpp
  )
  clang_target_link_libraries(ascendc_flow_plan_test PRIVATE
    clangAST
    clangASTMatchers
    clangBasic
    clangFrontend
    clangSerialization
    clangTooling
    RewriterLib
    ParserLib
    DacppLib
  )
  target_link_libraries(ascendc_flow_plan_test PRIVATE
    Threads::Threads
  )
  add_test(NAME translator_ascendc_flow_plan
           COMMAND ascendc_flow_plan_test)

  add_test(NAME translator_cli_debug_output
    COMMAND bash
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_debug_output.sh"
      "$<TARGET_FILE:translator>"
      "${CMAKE_CURRENT_SOURCE_DIR}"
  )
endif()

if(TARGET MLIRIR)
  if(DEFINED LLVM_EXTERNAL_MLIR_SOURCE_DIR AND EXISTS "${LLVM_EXTERNAL_MLIR_SOURCE_DIR}/include")
    target_include_directories(RewriterLib PRIVATE
      "${LLVM_EXTERNAL_MLIR_SOURCE_DIR}/include"
    )
  endif()

  if(EXISTS "${LLVM_BINARY_DIR}/tools/mlir/include")
    target_include_directories(RewriterLib PRIVATE
      "${LLVM_BINARY_DIR}/tools/mlir/include"
    )
  endif()

  if(DEFINED MLIR_INCLUDE_DIRS)
    target_include_directories(RewriterLib PRIVATE ${MLIR_INCLUDE_DIRS})
  endif()

  if(DEFINED MLIR_TABLEGEN_OUTPUT_DIR)
    target_include_directories(RewriterLib PRIVATE ${MLIR_TABLEGEN_OUTPUT_DIR})
  endif()

  target_compile_definitions(RewriterLib PRIVATE DACPP_TRANSLATOR_HAS_MLIR=1)

  if(DEFINED LLVM_EXTERNAL_MLIR_SOURCE_DIR AND EXISTS "${LLVM_EXTERNAL_MLIR_SOURCE_DIR}/include")
    target_include_directories(translator PRIVATE
      "${LLVM_EXTERNAL_MLIR_SOURCE_DIR}/include"
    )
  endif()

  if(EXISTS "${LLVM_BINARY_DIR}/tools/mlir/include")
    target_include_directories(translator PRIVATE
      "${LLVM_BINARY_DIR}/tools/mlir/include"
    )
  endif()

  if(DEFINED MLIR_INCLUDE_DIRS)
    target_include_directories(translator PRIVATE ${MLIR_INCLUDE_DIRS})
  endif()

  if(DEFINED MLIR_TABLEGEN_OUTPUT_DIR)
    target_include_directories(translator PRIVATE ${MLIR_TABLEGEN_OUTPUT_DIR})
  endif()

  target_compile_definitions(translator PRIVATE DACPP_TRANSLATOR_HAS_MLIR=1)
  clang_target_link_libraries(translator
    PRIVATE
    MLIRIR
    MLIRFuncDialect
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRMathDialect
  )
else()
  message(STATUS "translator: MLIR targets are not available, building with legacy AscendC path only")
endif()
