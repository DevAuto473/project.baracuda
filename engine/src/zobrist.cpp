#include "barracuda/engine/zobrist.h"

#include <mutex>

namespace barracuda::zobrist {

Key PIECE_SQUARE[PIECE_NB][SQUARE_NB];
Key CASTLING[16];
Key EN_PASSANT_FILE[8];
Key SIDE_TO_MOVE;

namespace {
std::uint64_t splitmix64(std::uint64_t& state) {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void init_impl() {
    std::uint64_t state = 0xBA22AC0DAULL; // fixed seed -> reproducible hashes across platforms
    for (auto& piece : PIECE_SQUARE)
        for (auto& k : piece) k = splitmix64(state);
    for (auto& k : CASTLING) k = splitmix64(state);
    for (auto& k : EN_PASSANT_FILE) k = splitmix64(state);
    SIDE_TO_MOVE = splitmix64(state);
}
} // namespace

void init() {
    static std::once_flag flag;
    std::call_once(flag, init_impl);
}

} // namespace barracuda::zobrist
