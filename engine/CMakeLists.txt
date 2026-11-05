# barracuda::core   - bitboards, magic attack tables, position, legal move generation, notation
# barracuda::engine - evaluation, transposition table, search, self-play, tactical suites
add_library(barracuda_core STATIC
    src/bitboard.cpp
    src/zobrist.cpp
    src/position.cpp
    src/movegen.cpp
    src/notation.cpp)
add_library(barracuda::core ALIAS barracuda_core)
target_include_directories(barracuda_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(barracuda_core PUBLIC cxx_std_20)
barracuda_configure_target(barracuda_core)

add_library(barracuda_engine STATIC
    src/evaluate.cpp
    src/tt.cpp
    src/search.cpp
    src/selfplay.cpp
    src/match.cpp
    src/publish.cpp
    src/suite.cpp)
add_library(barracuda::engine ALIAS barracuda_engine)
target_include_directories(barracuda_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(barracuda_engine PUBLIC barracuda::core barracuda::nnue Threads::Threads)
barracuda_configure_target(barracuda_engine)
