# =============================================================================
#  Barracuda - Cross-platform C++20 chess engine + NNUE + visual training dashboard
#  Targets: Fedora / Ubuntu (GCC, Clang)  and  Windows 10/11 (MSVC 2022, MinGW-w64)
# =============================================================================
cmake_minimum_required(VERSION 3.21)

project(Barracuda
    VERSION 1.0.0
    DESCRIPTION "Barracuda chess engine with NNUE and ImGui training dashboard"
    LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(BARRACUDA_BUILD_GUI      "Build the ImGui dashboard (fetches GLFW + Dear ImGui)" ON)
option(BARRACUDA_BUILD_CLI      "Build the UCI / command-line tool" ON)
option(BARRACUDA_BUILD_TESTS    "Build unit tests (perft, NNUE consistency)" ON)
option(BARRACUDA_NATIVE         "Optimise for the build machine (-march=native, GCC/Clang only)" OFF)
option(BARRACUDA_LTO            "Enable link-time optimisation in Release builds" ON)
option(BARRACUDA_STATIC_RUNTIME "MinGW: link libgcc/libstdc++/winpthread statically" ON)
set(BARRACUDA_SIMD "AUTO" CACHE STRING "SIMD level: AUTO, AVX2, SSE41, NONE")
set_property(CACHE BARRACUDA_SIMD PROPERTY STRINGS AUTO AVX2 SSE41 NONE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

get_property(BARRACUDA_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT BARRACUDA_MULTI_CONFIG AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "Barracuda: no build type given, defaulting to Release")
endif()

# Put every binary in one place so that the GUI and CLI find each other's data.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(BarracudaCompilerFlags)   # defines barracuda::options (warnings, -O3 / /O2, SIMD)

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Libraries
# ---------------------------------------------------------------------------
add_subdirectory(engine)   # barracuda::core (bitboards/movegen) + barracuda::engine (search)
add_subdirectory(nnue)     # barracuda::nnue (quantised inference + trainer)

# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------
if(BARRACUDA_BUILD_CLI)
    add_subdirectory(app)
endif()

if(BARRACUDA_BUILD_GUI)
    include(BarracudaDependencies) # FetchContent: GLFW + Dear ImGui
    add_subdirectory(gui)
endif()

if(BARRACUDA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "Barracuda ${PROJECT_VERSION}: compiler=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}, "
               "SIMD=${BARRACUDA_SIMD_RESOLVED}, GUI=${BARRACUDA_BUILD_GUI}, LTO=${BARRACUDA_LTO_ACTIVE}")
