# barracuda (GUI): ImGui dashboard - interactive board, NNUE visualiser, metrics, console
add_executable(barracuda_gui
    src/main.cpp
    src/app.cpp
    src/theme.cpp
    src/widgets.cpp
    src/runtime/search_worker.cpp
    src/runtime/training_worker.cpp
    src/panels/board_panel.cpp
    src/panels/network_panel.cpp
    src/panels/metrics_panel.cpp
    src/panels/console_panel.cpp)
set_target_properties(barracuda_gui PROPERTIES OUTPUT_NAME barracuda)
target_include_directories(barracuda_gui PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(barracuda_gui PRIVATE barracuda::engine barracuda::nnue barracuda::imgui Threads::Threads)
barracuda_configure_target(barracuda_gui)

option(BARRACUDA_WINDOWED "Windows: build the dashboard as a GUI-subsystem app (no console window)" OFF)
if(WIN32 AND BARRACUDA_WINDOWED)
    set_target_properties(barracuda_gui PROPERTIES WIN32_EXECUTABLE TRUE)
    if(MSVC)
        target_link_options(barracuda_gui PRIVATE /ENTRY:mainCRTStartup) # keep using main()
    endif()
endif()
