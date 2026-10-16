cmake_minimum_required(VERSION 3.21)
project(rag VERSION 1.0.0 LANGUAGES CXX)

include(CTest)

set(MLN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../MyeLiN" CACHE PATH
    "MyeLiN source tree")
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(TURBOVEC_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/src/rag/turbovec/Cargo.toml")
set(TURBOVEC_LIBRARY
    "${CMAKE_CURRENT_SOURCE_DIR}/src/rag/turbovec/target/release/libturbovec_myelin.a")
file(GLOB_RECURSE TURBOVEC_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/rag/turbovec/src/*.rs"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/turbovec/turbovec/src/*.rs")

add_custom_command(
    OUTPUT "${TURBOVEC_LIBRARY}"
    COMMAND "${CARGO_EXECUTABLE}" build --manifest-path "${TURBOVEC_MANIFEST}"
            --locked --release
    DEPENDS
        "${TURBOVEC_MANIFEST}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/rag/turbovec/Cargo.lock"
        ${TURBOVEC_SOURCES}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Building TurboVec static library"
    VERBATIM)
add_custom_target(rag_turbovec_library DEPENDS "${TURBOVEC_LIBRARY}")

add_executable(rag_context_test
    tests/context_builder_test.cpp
    src/rag/chunk.cpp
    src/rag/context_builder.cpp)
target_include_directories(rag_context_test PRIVATE include)
target_compile_features(rag_context_test PRIVATE cxx_std_23)
target_compile_options(rag_context_test PRIVATE -UNDEBUG)

add_library(rag_turbovec_provider SHARED src/rag/turbovec_native.cpp)
add_dependencies(rag_turbovec_provider rag_turbovec_library)
target_include_directories(rag_turbovec_provider PRIVATE include "${MLN_ROOT}/include")
target_compile_definitions(rag_turbovec_provider PRIVATE MLN_NATIVE_BUILD=1)
target_link_libraries(rag_turbovec_provider PRIVATE
    "${TURBOVEC_LIBRARY}" pthread dl m)
target_compile_features(rag_turbovec_provider PRIVATE cxx_std_23)

add_executable(rag_turbovec_test tests/turbovec_test.cpp)
target_include_directories(rag_turbovec_test PRIVATE "${MLN_ROOT}/include")
target_link_libraries(rag_turbovec_test PRIVATE dl)
target_compile_features(rag_turbovec_test PRIVATE cxx_std_23)

if(BUILD_TESTING)
    add_test(NAME rag_context COMMAND rag_context_test)
    add_test(NAME rag_turbovec COMMAND rag_turbovec_test
        "$<TARGET_FILE:rag_turbovec_provider>")
endif()
