// Barracuda - attack table initialisation (fancy magic bitboards generated at startup).
#include "barracuda/engine/bitboard.h"

#include <cstdlib>
#include <mutex>
#include <vector>

namespace barracuda {
namespace attacks {

Bitboard PAWN_ATTACKS[COLOR_NB][SQUARE_NB];
Bitboard KNIGHT_ATTACKS[SQUARE_NB];
Bitboard KING_ATTACKS[SQUARE_NB];
Magic ROOK_MAGICS[SQUARE_NB];
Magic BISHOP_MAGICS[SQUARE_NB];
Bitboard BETWEEN[SQUARE_NB][SQUARE_NB];
Bitboard LINE[SQUARE_NB][SQUARE_NB];

namespace {

// Rook needs at most 4096 entries/square, bishop 512.
Bitboard ROOK_TABLE[0x19000];
Bitboard BISHOP_TABLE[0x1480];

// xorshift64* - deterministic so that magic search is reproducible.
class Prng {
public:
    explicit Prng(std::uint64_t seed) : s_(seed) {}
    std::uint64_t next() {
        s_ ^= s_ >> 12;
        s_ ^= s_ << 25;
        s_ ^= s_ >> 27;
        return s_ * 2685821657736338717ULL;
    }
    std::uint64_t sparse() { return next() & next() & next(); }

private:
    std::uint64_t s_;
};

int distance(Square a, Square b) {
    return std::max(std::abs(file_of(a) - file_of(b)), std::abs(rank_of(a) - rank_of(b)));
}

Bitboard safe_step(Square s, int step) {
    const int to = int(s) + step;
    if (to < 0 || to > 63) return 0;
    return distance(s, Square(to)) <= 2 ? square_bb(Square(to)) : 0;
}

Bitboard sliding_attack(PieceType pt, Square sq, Bitboard occ) {
    static constexpr int ROOK_DIRS[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    static constexpr int BISHOP_DIRS[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    const auto& dirs = pt == ROOK ? ROOK_DIRS : BISHOP_DIRS;
    Bitboard result = 0;
    for (const auto& d : dirs) {
        int f = file_of(sq) + d[0];
        int r = rank_of(sq) + d[1];
        while (f >= 0 && f < 8 && r >= 0 && r < 8) {
            const Bitboard b = square_bb(make_square(f, r));
            result |= b;
            if (occ & b) break;
            f += d[0];
            r += d[1];
        }
    }
    return result;
}

void init_magics(PieceType pt, Bitboard* table, Magic* magics) {
    // Seeds that find magics quickly (one per rank), as popularised by Stockfish.
    static constexpr std::uint64_t SEEDS[8] = {728, 10316, 55013, 32803, 12281, 15100, 16645, 255};
    std::vector<Bitboard> occupancy(4096), reference(4096);
    std::vector<int> epoch(4096, 0);
    int cnt = 0;
    int size = 0;

    for (Square s = A1; s <= H8; ++s) {
        const Bitboard edges = ((RANK_1_BB | RANK_8_BB) & ~rank_bb(rank_of(s))) |
                               ((FILE_A_BB | FILE_H_BB) & ~file_bb(file_of(s)));
        Magic& m = magics[s];
        m.mask = sliding_attack(pt, s, 0) & ~edges;
        m.shift = 64 - unsigned(popcount(m.mask));
        m.table = s == A1 ? table : magics[s - 1].table + size;

        // Carry-Rippler enumeration of all subsets of the mask.
        Bitboard b = 0;
        size = 0;
        do {
            occupancy[size] = b;
            reference[size] = sliding_attack(pt, s, b);
            size++;
            b = (b - m.mask) & m.mask;
        } while (b);

        Prng rng(SEEDS[rank_of(s)]);
        for (int i = 0; i < size;) {
            for (m.magic = 0; popcount((m.magic * m.mask) >> 56) < 6;) m.magic = rng.sparse();
            for (++cnt, i = 0; i < size; ++i) {
                const unsigned idx = m.index(occupancy[i]);
                if (epoch[idx] < cnt) {
                    epoch[idx] = cnt;
                    m.table[idx] = reference[i];
                } else if (m.table[idx] != reference[i]) {
                    break;
                }
            }
        }
    }
}

void init_impl() {
    for (Square s = A1; s <= H8; ++s) {
        PAWN_ATTACKS[WHITE][s] = shift<NORTH_EAST>(square_bb(s)) | shift<NORTH_WEST>(square_bb(s));
        PAWN_ATTACKS[BLACK][s] = shift<SOUTH_EAST>(square_bb(s)) | shift<SOUTH_WEST>(square_bb(s));
        KNIGHT_ATTACKS[s] = 0;
        for (int step : {17, 15, 10, 6, -6, -10, -15, -17}) KNIGHT_ATTACKS[s] |= safe_step(s, step);
        KING_ATTACKS[s] = 0;
        for (int step : {9, 8, 7, 1, -1, -7, -8, -9}) KING_ATTACKS[s] |= safe_step(s, step);
    }

    init_magics(ROOK, ROOK_TABLE, ROOK_MAGICS);
    init_magics(BISHOP, BISHOP_TABLE, BISHOP_MAGICS);

    for (Square a = A1; a <= H8; ++a) {
        for (Square b = A1; b <= H8; ++b) {
            LINE[a][b] = 0;
            BETWEEN[a][b] = square_bb(b);
            if (a == b) continue;
            for (PieceType pt : {BISHOP, ROOK}) {
                if (sliding_attack(pt, a, 0) & square_bb(b)) {
                    LINE[a][b] = (sliding_attack(pt, a, 0) & sliding_attack(pt, b, 0)) | square_bb(a) | square_bb(b);
                    BETWEEN[a][b] = (sliding_attack(pt, a, square_bb(b)) & sliding_attack(pt, b, square_bb(a))) |
                                    square_bb(b);
                }
            }
        }
    }
}

} // namespace

void init() {
    static std::once_flag flag;
    std::call_once(flag, init_impl);
}

} // namespace attacks

std::string bitboard_to_string(Bitboard b) {
    std::string s = "+---+---+---+---+---+---+---+---+\n";
    for (int r = 7; r >= 0; --r) {
        for (int f = 0; f < 8; ++f) s += (b & square_bb(make_square(f, r))) ? "| X " : "|   ";
        s += "| " + std::to_string(r + 1) + "\n+---+---+---+---+---+---+---+---+\n";
    }
    s += "  a   b   c   d   e   f   g   h\n";
    return s;
}

} // namespace barracuda
