// Barracuda - fully legal move generation (pins, checks, en passant, castling, promotions).
#pragma once

#include <array>
#include <cstdint>

#include "barracuda/engine/position.h"

namespace barracuda {

struct MoveList {
    std::array<Move, 256> moves{};
    int size = 0;

    void push(Move m) { moves[size++] = m; }
    Move* begin() { return moves.data(); }
    Move* end() { return moves.data() + size; }
    const Move* begin() const { return moves.data(); }
    const Move* end() const { return moves.data() + size; }
    Move operator[](int i) const { return moves[i]; }
    bool contains(Move m) const {
        for (int i = 0; i < size; ++i)
            if (moves[i] == m) return true;
        return false;
    }
};

enum class GenType { ALL, NOISY }; // NOISY = captures + queen promotions

void generate_moves(const Position& pos, MoveList& list, GenType type = GenType::ALL);
inline MoveList legal_moves(const Position& pos) {
    MoveList l;
    generate_moves(pos, l, GenType::ALL);
    return l;
}

std::uint64_t perft(Position& pos, int depth);

} // namespace barracuda
