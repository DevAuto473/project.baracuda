# -----------------------------------------------------------------------------
# Compiler-specific optimisation, warning and SIMD configuration.
# Everything is attached to the INTERFACE target `barracuda::options`, which all
# first-party targets link against (third-party code keeps its own flags).
# -----------------------------------------------------------------------------
include_guard(GLOBAL)
include(CheckIPOSupported)

add_library(barracuda_options INTERFACE)
add_library(barracuda::options ALIAS barracuda_options)

set(_is_msvc FALSE)
if(MSVC)
    set(_is_msvc TRUE) # cl.exe and clang-cl
endif()

# ---- warnings / conformance ---------------------------------------------------
if(_is_msvc)
    target_compile_options(barracuda_options INTERFACE
        /W4 /permissive- /Zc:__cplusplus /Zc:preprocessor /utf-8 /EHsc /MP
        /wd4244 /wd4267 /wd4146 /wd4324) # narrowing noise, unary minus on unsigned, alignment padding
    target_compile_definitions(barracuda_options INTERFACE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(barracuda_options INTERFACE -Wall -Wextra -Wpedantic -Wno-missing-field-initializers)
endif()

# ---- optimisation -------------------------------------------------------------
if(_is_msvc)
    target_compile_options(barracuda_options INTERFACE
        $<$<CONFIG:Release,RelWithDebInfo>:/O2 /Oi /Ot /Gy /GS->)
else()
    target_compile_options(barracuda_options INTERFACE
        $<$<CONFIG:Release>:-O3 -fno-math-errno -funroll-loops>
        $<$<CONFIG:RelWithDebInfo>:-O3 -g>)
    if(MINGW AND BARRACUDA_STATIC_RUNTIME)
        target_link_options(barracuda_options INTERFACE -static-libgcc -static-libstdc++ -static)
    endif()
endif()

# ---- SIMD detection -----------------------------------------------------------
set(_simd "${BARRACUDA_SIMD}")
string(TOUPPER "${_simd}" _simd)
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _cpu)
set(_is_x86 FALSE)
if(_cpu MATCHES "^(x86_64|amd64|x64|i[3-6]86|x86)$" OR CMAKE_GENERATOR_PLATFORM MATCHES "^(x64|Win32)$")
    set(_is_x86 TRUE)
endif()

if(_simd STREQUAL "AUTO")
    if(NOT _is_x86 OR CMAKE_CROSSCOMPILING)
        set(_simd "NONE")
    else()
        try_run(_avx2_run _avx2_compile
            ${CMAKE_BINARY_DIR}/simd_probe
            ${CMAKE_CURRENT_LIST_DIR}/detect_avx2.cpp
            COMPILE_OUTPUT_VARIABLE _avx2_log)
        if(_avx2_compile AND ("${_avx2_run}" STREQUAL "0"))
            set(_simd "AVX2")
        else()
            set(_simd "SSE41")
        endif()
        message(STATUS "Barracuda: host CPU probe -> ${_simd}")
    endif()
endif()

if(BARRACUDA_NATIVE AND NOT _is_msvc)
    target_compile_options(barracuda_options INTERFACE -march=native)
    set(_simd "NATIVE")
elseif(_simd STREQUAL "AVX2")
    if(_is_msvc)
        target_compile_options(barracuda_options INTERFACE /arch:AVX2)
    else()
        target_compile_options(barracuda_options INTERFACE -mavx2 -mbmi -mbmi2 -mpopcnt -mfma)
    endif()
elseif(_simd STREQUAL "SSE41")
    if(_is_msvc)
        # MSVC x64 has SSE2 by default; SSE4.1 intrinsics are available without a flag.
        target_compile_definitions(barracuda_options INTERFACE BARRACUDA_SSE41=1)
    else()
        target_compile_options(barracuda_options INTERFACE -msse4.1 -mpopcnt)
    endif()
endif()
set(BARRACUDA_SIMD_RESOLVED "${_simd}" CACHE INTERNAL "")

# ---- link-time optimisation ---------------------------------------------------
set(BARRACUDA_LTO_ACTIVE OFF CACHE INTERNAL "")
if(BARRACUDA_LTO)
    check_ipo_supported(RESULT _ipo_ok OUTPUT _ipo_msg LANGUAGES CXX)
    if(_ipo_ok)
        set(BARRACUDA_LTO_ACTIVE ON CACHE INTERNAL "")
    else()
        message(STATUS "Barracuda: LTO not supported here (${_ipo_msg})")
    endif()
endif()

# Helper that every first-party target calls.
function(barracuda_configure_target target)
    target_link_libraries(${target} PRIVATE barracuda::options)
    if(BARRACUDA_LTO_ACTIVE)
        set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    endif()
endfunction()
