# barracuda-cli: UCI engine + headless self-play / training / suite runner
add_executable(barracuda_cli main.cpp)
set_target_properties(barracuda_cli PROPERTIES OUTPUT_NAME barracuda-cli)
target_link_libraries(barracuda_cli PRIVATE barracuda::engine barracuda::nnue)
barracuda_configure_target(barracuda_cli)
