// Barracuda - 64-bit bitboard utilities and attack tables (fancy magic bitboards).
#pragma once

#include <bit>
#include <string>

#include "barracuda/engine/types.h"

namespace barracuda {

constexpr Bitboard FILE_A_BB = 0x0101010101010101ULL;
constexpr Bitboard FILE_H_BB = FILE_A_BB << 7;
constexpr Bitboard RANK_1_BB = 0xFFULL;
constexpr Bitboard RANK_2_BB = RANK_1_BB << 8;
constexpr Bitboard RANK_3_BB = RANK_1_BB << 16;
constexpr Bitboard RANK_4_BB = RANK_1_BB << 24;
constexpr Bitboard RANK_5_BB = RANK_1_BB << 32;
constexpr Bitboard RANK_6_BB = RANK_1_BB << 40;
constexpr Bitboard RANK_7_BB = RANK_1_BB << 48;
constexpr Bitboard RANK_8_BB = RANK_1_BB << 56;

constexpr Bitboard square_bb(Square s) { return 1ULL << s; }
constexpr Bitboard file_bb(int f) { return FILE_A_BB << f; }
constexpr Bitboard rank_bb(int r) { return RANK_1_BB << (8 * r); }

inline int popcount(Bitboard b) { return std::popcount(b); }
inline Square lsb(Bitboard b) { return Square(std::countr_zero(b)); }
inline Square msb(Bitboard b) { return Square(63 - std::countl_zero(b)); }
inline Square pop_lsb(Bitboard& b) {
    const Square s = lsb(b);
    b &= b - 1;
    return s;
}
constexpr bool more_than_one(Bitboard b) { return (b & (b - 1)) != 0; }

template <int D>
constexpr Bitboard shift(Bitboard b) {
    if constexpr (D == NORTH) return b << 8;
    else if constexpr (D == SOUTH) return b >> 8;
    else if constexpr (D == EAST) return (b & ~FILE_H_BB) << 1;
    else if constexpr (D == WEST) return (b & ~FILE_A_BB) >> 1;
    else if constexpr (D == NORTH_EAST) return (b & ~FILE_H_BB) << 9;
    else if constexpr (D == NORTH_WEST) return (b & ~FILE_A_BB) << 7;
    else if constexpr (D == SOUTH_EAST) return (b & ~FILE_H_BB) >> 7;
    else if constexpr (D == SOUTH_WEST) return (b & ~FILE_A_BB) >> 9;
    else return 0;
}

// ---------------------------------------------------------------------------
// Attack tables. Must be initialised once via attacks::init() before use.
// ---------------------------------------------------------------------------
namespace attacks {

struct Magic {
    Bitboard mask = 0;
    Bitboard magic = 0;
    Bitboard* table = nullptr;
    unsigned shift = 0;

    unsigned index(Bitboard occupied) const {
        return unsigned(((occupied & mask) * magic) >> shift);
    }
};

extern Bitboard PAWN_ATTACKS[COLOR_NB][SQUARE_NB];
extern Bitboard KNIGHT_ATTACKS[SQUARE_NB];
extern Bitboard KING_ATTACKS[SQUARE_NB];
extern Magic ROOK_MAGICS[SQUARE_NB];
extern Magic BISHOP_MAGICS[SQUARE_NB];
extern Bitboard BETWEEN[SQUARE_NB][SQUARE_NB]; // squares strictly between a and b (exclusive), plus b
extern Bitboard LINE[SQUARE_NB][SQUARE_NB];    // full line through a and b (0 if not aligned)

void init(); // thread-safe, idempotent

inline Bitboard pawn(Color c, Square s) { return PAWN_ATTACKS[c][s]; }
inline Bitboard knight(Square s) { return KNIGHT_ATTACKS[s]; }
inline Bitboard king(Square s) { return KING_ATTACKS[s]; }
inline Bitboard bishop(Square s, Bitboard occ) {
    const Magic& m = BISHOP_MAGICS[s];
    return m.table[m.index(occ)];
}
inline Bitboard rook(Square s, Bitboard occ) {
    const Magic& m = ROOK_MAGICS[s];
    return m.table[m.index(occ)];
}
inline Bitboard queen(Square s, Bitboard occ) { return bishop(s, occ) | rook(s, occ); }

inline Bitboard piece(PieceType pt, Square s, Bitboard occ) {
    switch (pt) {
    case KNIGHT: return knight(s);
    case BISHOP: return bishop(s, occ);
    case ROOK: return rook(s, occ);
    case QUEEN: return queen(s, occ);
    case KING: return king(s);
    default: return 0;
    }
}

} // namespace attacks

inline Bitboard between_bb(Square a, Square b) { return attacks::BETWEEN[a][b]; }
inline Bitboard line_bb(Square a, Square b) { return attacks::LINE[a][b]; }
inline bool aligned(Square a, Square b, Square c) { return (line_bb(a, b) & square_bb(c)) != 0; }

std::string bitboard_to_string(Bitboard b);

} // namespace barracuda
