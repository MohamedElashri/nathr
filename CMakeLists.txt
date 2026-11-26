cmake_minimum_required(VERSION 3.25)
project(nathr VERSION 0.4.0 LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
    message(FATAL_ERROR "Nathr requires x86-64 Linux")
endif()
if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
   AND NOT (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 16))
    message(FATAL_ERROR "Nathr requires GCC 12+ or Clang 16+")
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
set(NATHR_SANITIZER "" CACHE STRING "Sanitizer selection: address,undefined or thread")
option(NATHR_BUILD_FUZZERS "Build the Clang model-free libFuzzer targets" OFF)
if(NATHR_BUILD_FUZZERS)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR NATHR_SANITIZER)
        message(FATAL_ERROR "NATHR_BUILD_FUZZERS requires Clang and an empty NATHR_SANITIZER")
    endif()
    add_compile_options(-fsanitize=fuzzer-no-link,address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()
if(NATHR_SANITIZER)
    add_compile_options(-fsanitize=${NATHR_SANITIZER} -fno-omit-frame-pointer)
    add_link_options(-fsanitize=${NATHR_SANITIZER})
endif()

function(nathr_compile target)
    target_compile_features(${target} PUBLIC cxx_std_20)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -march=x86-64 -ffp-contract=off)
    target_include_directories(${target} PRIVATE src)
    target_include_directories(${target} SYSTEM PRIVATE third_party/json)
endfunction()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_package(Threads REQUIRED)
add_library(nathr_lib
    src/core/input.cpp
    src/core/sha256.cpp
    src/formats/gguf.cpp
    src/formats/safetensors.cpp
    src/quantization/decoders.cpp
    src/backend/cpu/scalar.cpp
    src/backend/cpu/kernels.cpp
    src/backend/cpu/thread_pool.cpp
    src/backend/cpu/x86/avx2.cpp
    src/models/llama/config.cpp
    src/models/llama/model.cpp
    src/runtime/session.cpp
    src/runtime/generation.cpp
    src/tokenizer/tokenizer.cpp
)
set_source_files_properties(src/backend/cpu/x86/avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
add_library(nathr::nathr ALIAS nathr_lib)
set_target_properties(nathr_lib PROPERTIES OUTPUT_NAME nathr EXPORT_NAME nathr
    VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
target_include_directories(nathr_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(nathr_lib PUBLIC Threads::Threads)
nathr_compile(nathr_lib)

include(CTest)
set(NATHR_TEST_FIXTURES "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")

add_executable(nathr apps/nathr.cpp)
target_compile_definitions(nathr PRIVATE NATHR_VERSION="${PROJECT_VERSION}")
target_link_libraries(nathr PRIVATE nathr::nathr)
nathr_compile(nathr)
if(BUILD_SHARED_LIBS AND NOT IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}" AND
   NOT IS_ABSOLUTE "${CMAKE_INSTALL_BINDIR}")
    file(RELATIVE_PATH NATHR_BIN_TO_LIB "/${CMAKE_INSTALL_BINDIR}" "/${CMAKE_INSTALL_LIBDIR}")
    set_target_properties(nathr PROPERTIES INSTALL_RPATH "$ORIGIN/${NATHR_BIN_TO_LIB}")
endif()

set(NATHR_PACKAGE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/nathr")
install(TARGETS nathr_lib EXPORT nathrTargets)
install(TARGETS nathr)
install(DIRECTORY include/nathr DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT nathrTargets NAMESPACE nathr:: DESTINATION ${NATHR_PACKAGE_DIR})
configure_package_config_file(cmake/nathrConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/nathrConfig.cmake" INSTALL_DESTINATION ${NATHR_PACKAGE_DIR})
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/nathrConfigVersion.cmake"
    COMPATIBILITY SameMinorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/nathrConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/nathrConfigVersion.cmake" DESTINATION ${NATHR_PACKAGE_DIR})
install(FILES third_party/json/LICENSE.MIT src/tokenizer/LICENSE.unicode src/quantization/LICENSE.ggml
    DESTINATION "${CMAKE_INSTALL_DOCDIR}/licenses")

if(BUILD_TESTING)
    add_executable(nathr_tests tests/runtime.cpp tests/allocation_counter.cpp)
    target_compile_definitions(nathr_tests PRIVATE NATHR_FIXTURES="${NATHR_TEST_FIXTURES}")
    target_link_libraries(nathr_tests PRIVATE nathr::nathr)
    nathr_compile(nathr_tests)
    add_test(NAME nathr-fast COMMAND nathr_tests)
    add_test(NAME nathr-cli COMMAND ${CMAKE_COMMAND} -DNATHR_BIN=$<TARGET_FILE:nathr>
        -DNATHR_FIXTURES=${NATHR_TEST_FIXTURES} -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli.cmake)
    set_tests_properties(nathr-fast nathr-cli PROPERTIES LABELS fast)
    if(NOT NATHR_SANITIZER AND NOT NATHR_BUILD_FUZZERS AND
       NOT IS_ABSOLUTE "${CMAKE_INSTALL_BINDIR}" AND NOT IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}" AND
       NOT IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}" AND NOT IS_ABSOLUTE "${CMAKE_INSTALL_DOCDIR}")
        add_test(NAME nathr-install COMMAND ${CMAKE_COMMAND}
            -DNATHR_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR} -DNATHR_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -DNATHR_CXX=${CMAKE_CXX_COMPILER} -DNATHR_CONFIG=$<CONFIG>
            -DNATHR_BINDIR=${CMAKE_INSTALL_BINDIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/install.cmake)
        set_tests_properties(nathr-install PROPERTIES LABELS packaging)
    endif()
endif()

if(NATHR_BUILD_FUZZERS)
    foreach(kind IN ITEMS safetensors gguf tokenizer quant)
        add_executable(nathr_fuzz_${kind} tests/fuzz.cpp)
        target_compile_definitions(nathr_fuzz_${kind} PRIVATE
            NATHR_FUZZ_${kind} NATHR_FIXTURES="${NATHR_TEST_FIXTURES}")
        target_link_libraries(nathr_fuzz_${kind} PRIVATE nathr::nathr)
        target_link_options(nathr_fuzz_${kind} PRIVATE -fsanitize=fuzzer)
        nathr_compile(nathr_fuzz_${kind})
    endforeach()
endif()

find_program(NATHR_CLANG_FORMAT clang-format)
if(NATHR_CLANG_FORMAT)
    file(GLOB_RECURSE NATHR_FORMAT_SOURCES CONFIGURE_DEPENDS
         apps/*.cpp include/*.hpp src/*.cpp src/*.hpp tests/*.cpp)
    list(FILTER NATHR_FORMAT_SOURCES EXCLUDE REGEX "/(unicode_tables|iq_codebooks)\\.hpp$")
    add_custom_target(format-check
        COMMAND "${NATHR_CLANG_FORMAT}" --dry-run --Werror ${NATHR_FORMAT_SOURCES}
        VERBATIM)
endif()
