add_executable(barracuda_tests test_main.cpp)
target_link_libraries(barracuda_tests PRIVATE barracuda::engine barracuda::nnue)
# header-only lock-free primitives shared with the GUI runtime
target_include_directories(barracuda_tests PRIVATE ${PROJECT_SOURCE_DIR}/gui/src)
barracuda_configure_target(barracuda_tests)

add_test(NAME barracuda_tests COMMAND barracuda_tests WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
set_tests_properties(barracuda_tests PROPERTIES TIMEOUT 900)
