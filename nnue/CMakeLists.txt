# barracuda::nnue - quantised NNUE inference (SIMD), float master network, dataset + trainer, inspector
add_library(barracuda_nnue STATIC
    src/float_network.cpp
    src/network.cpp
    src/dataset.cpp
    src/trainer.cpp
    src/inspector.cpp)
add_library(barracuda::nnue ALIAS barracuda_nnue)
target_include_directories(barracuda_nnue PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(barracuda_nnue PUBLIC barracuda::core Threads::Threads)
barracuda_configure_target(barracuda_nnue)
