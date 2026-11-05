# -----------------------------------------------------------------------------
# GUI dependencies, fetched at configure time (no system-wide framework installs):
#   * GLFW 3.4      - windowing / input / OpenGL context (X11 + Wayland on Linux, Win32)
#   * Dear ImGui    - immediate-mode UI (GLFW + OpenGL3 backends; its GL loader is built in)
#
# Offline / air-gapped builds: either
#   - put the sources in third_party/glfw and third_party/imgui (e.g. git submodules), or
#   - pass -DFETCHCONTENT_SOURCE_DIR_GLFW=<dir> -DFETCHCONTENT_SOURCE_DIR_IMGUI=<dir>
# -----------------------------------------------------------------------------
include_guard(GLOBAL)
include(FetchContent)

set(BARRACUDA_GLFW_TAG  "3.4"     CACHE STRING "GLFW git tag")
set(BARRACUDA_IMGUI_TAG "v1.91.5" CACHE STRING "Dear ImGui git tag")

foreach(_dep glfw imgui)
    string(TOUPPER ${_dep} _DEP)
    set(_vendored "${PROJECT_SOURCE_DIR}/third_party/${_dep}")
    if(NOT FETCHCONTENT_SOURCE_DIR_${_DEP} AND (EXISTS "${_vendored}/LICENSE.md" OR EXISTS "${_vendored}/LICENSE.txt"))
        set(FETCHCONTENT_SOURCE_DIR_${_DEP} "${_vendored}" CACHE PATH "" FORCE)
        message(STATUS "Barracuda: using vendored ${_dep} from third_party/${_dep}")
    endif()
endforeach()

# ---- GLFW configuration ------------------------------------------------------
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)

if(UNIX AND NOT APPLE)
    # Pick the Linux backends that can actually be built on this machine, so a missing
    # Wayland toolchain on an X11-only Ubuntu box doesn't break configuration (and vice versa).
    find_package(PkgConfig QUIET)
    set(_have_wayland FALSE)
    if(PKG_CONFIG_FOUND)
        find_program(_wayland_scanner wayland-scanner)
        pkg_check_modules(_wl QUIET wayland-client wayland-cursor wayland-egl xkbcommon)
        if(_wayland_scanner AND _wl_FOUND)
            set(_have_wayland TRUE)
        endif()
    endif()
    find_package(X11 QUIET)
    set(_have_x11 FALSE)
    if(X11_FOUND AND X11_Xrandr_INCLUDE_PATH AND X11_Xinerama_INCLUDE_PATH AND X11_Xkb_INCLUDE_PATH
       AND X11_Xcursor_INCLUDE_PATH AND X11_Xi_INCLUDE_PATH)
        set(_have_x11 TRUE)
    endif()
    if(NOT _have_x11 AND NOT _have_wayland)
        message(FATAL_ERROR
            "Barracuda GUI: neither X11 nor Wayland development headers were found.\n"
            "  Ubuntu: sudo apt install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev "
            "libgl1-mesa-dev libwayland-dev libxkbcommon-dev\n"
            "  Fedora: sudo dnf install libX11-devel libXrandr-devel libXinerama-devel libXcursor-devel "
            "libXi-devel mesa-libGL-devel wayland-devel libxkbcommon-devel\n"
            "  ...or configure with -DBARRACUDA_BUILD_GUI=OFF for the headless engine only.")
    endif()
    set(GLFW_BUILD_X11     ${_have_x11}     CACHE BOOL "" FORCE)
    set(GLFW_BUILD_WAYLAND ${_have_wayland} CACHE BOOL "" FORCE)
    message(STATUS "Barracuda GUI: GLFW backends X11=${_have_x11} Wayland=${_have_wayland}")
endif()

find_package(OpenGL REQUIRED)

FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        ${BARRACUDA_GLFW_TAG}
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE)
FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        ${BARRACUDA_IMGUI_TAG}
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE)

FetchContent_MakeAvailable(glfw)
FetchContent_GetProperties(imgui)
if(NOT imgui_POPULATED)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
        FetchContent_MakeAvailable(imgui) # no CMakeLists.txt upstream: this only downloads
    else()
        FetchContent_Populate(imgui)
    endif()
endif()

# ---- Dear ImGui static library ---------------------------------------------------
add_library(barracuda_imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
add_library(barracuda::imgui ALIAS barracuda_imgui)
target_include_directories(barracuda_imgui SYSTEM PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_compile_definitions(barracuda_imgui PUBLIC IMGUI_DISABLE_OBSOLETE_FUNCTIONS)
target_link_libraries(barracuda_imgui PUBLIC glfw OpenGL::GL)
if(CMAKE_DL_LIBS)
    target_link_libraries(barracuda_imgui PUBLIC ${CMAKE_DL_LIBS}) # GL loader uses dlopen on Linux
endif()
set_target_properties(barracuda_imgui PROPERTIES POSITION_INDEPENDENT_CODE ON)
