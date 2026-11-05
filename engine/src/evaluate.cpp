#include "barracuda/engine/evaluate.h"

#include <algorithm>
#include <array>

namespace barracuda::eval {

namespace {

// Piece-square tables (from White's point of view, rank 8 first, as printed on a diagram).
// Based on Tomasz Michniewski's "Simplified Evaluation Function".
// clang-format off
constexpr int PAWN_PST[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0};
constexpr int KNIGHT_PST[64] = {
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50};
constexpr int BISHOP_PST[64] = {
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -20,-10,-10,-10,-10,-10,-10,-20};
constexpr int ROOK_PST[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0};
constexpr int QUEEN_PST[64] = {
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20};
constexpr int KING_MG_PST[64] = {
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20};
constexpr int KING_EG_PST[64] = {
   -50,-40,-30,-20,-20,-30,-40,-50,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-30,  0,  0,  0,  0,-30,-30,
   -50,-30,-30,-30,-30,-30,-30,-50};
// clang-format on

constexpr int MG_VALUE[6] = {100, 320, 330, 500, 900, 0};
constexpr int EG_VALUE[6] = {120, 300, 320, 530, 950, 0};
constexpr int PHASE_WEIGHT[6] = {0, 1, 1, 2, 4, 0};
constexpr int PASSED_BONUS_MG[8] = {0, 5, 10, 15, 25, 40, 60, 0};
constexpr int PASSED_BONUS_EG[8] = {0, 10, 20, 35, 60, 95, 140, 0};

constexpr const int* PST[6] = {PAWN_PST, KNIGHT_PST, BISHOP_PST, ROOK_PST, QUEEN_PST, KING_MG_PST};

// table index for a piece of colour c on square s (tables are written rank 8 first)
constexpr int pst_index(Color c, Square s) { return c == WHITE ? (int(s) ^ 56) : int(s); }

Bitboard forward_files(Color c, Square s) {
    Bitboard files = file_bb(file_of(s));
    if (file_of(s) > 0) files |= file_bb(file_of(s) - 1);
    if (file_of(s) < 7) files |= file_bb(file_of(s) + 1);
    Bitboard ahead = 0;
    if (c == WHITE) {
        for (int r = rank_of(s) + 1; r < 8; ++r) ahead |= rank_bb(r);
    } else {
        for (int r = rank_of(s) - 1; r >= 0; --r) ahead |= rank_bb(r);
    }
    return files & ahead;
}

} // namespace

Value classical(const Position& pos) {
    int mg[2] = {0, 0}, eg[2] = {0, 0};
    int phase = 0;
    const Bitboard occ = pos.pieces();

    for (Color c : {WHITE, BLACK}) {
        for (int pt = PAWN; pt <= KING; ++pt) {
            Bitboard bb = pos.pieces(c, PieceType(pt));
            phase += PHASE_WEIGHT[pt] * popcount(bb);
            while (bb) {
                const Square s = pop_lsb(bb);
                const int idx = pst_index(c, s);
                mg[c] += MG_VALUE[pt] + PST[pt][idx];
                eg[c] += EG_VALUE[pt] + (pt == KING ? KING_EG_PST[idx] : PST[pt][idx]);
                if (pt >= KNIGHT && pt <= QUEEN) {
                    // light mobility term
                    const int mob = popcount(attacks::piece(PieceType(pt), s, occ) & ~pos.pieces(c));
                    mg[c] += 2 * mob;
                    eg[c] += 3 * mob;
                }
            }
        }

        // bishop pair
        if (popcount(pos.pieces(c, BISHOP)) >= 2) {
            mg[c] += 30;
            eg[c] += 50;
        }

        // pawn structure
        const Bitboard own_pawns = pos.pieces(c, PAWN);
        const Bitboard their_pawns = pos.pieces(~c, PAWN);
        Bitboard pawns = own_pawns;
        while (pawns) {
            const Square s = pop_lsb(pawns);
            const int f = file_of(s);
            Bitboard adjacent = 0;
            if (f > 0) adjacent |= file_bb(f - 1);
            if (f < 7) adjacent |= file_bb(f + 1);
            if (!(own_pawns & adjacent)) { mg[c] -= 12; eg[c] -= 18; } // isolated
            if (popcount(own_pawns & file_bb(f)) > 1) { mg[c] -= 8; eg[c] -= 16; } // doubled (per pawn)
            if (!(forward_files(c, s) & their_pawns)) { // passed
                const int r = relative_rank(c, s);
                mg[c] += PASSED_BONUS_MG[r];
                eg[c] += PASSED_BONUS_EG[r];
            }
        }

        // rooks on open / semi-open files
        Bitboard rooks = pos.pieces(c, ROOK);
        while (rooks) {
            const Bitboard f = file_bb(file_of(pop_lsb(rooks)));
            if (!(f & own_pawns)) {
                mg[c] += (f & their_pawns) ? 10 : 20;
                eg[c] += 10;
            }
        }
    }

    phase = std::min(phase, 24);
    const Color us = pos.side_to_move();
    const int mg_score = mg[us] - mg[~us];
    const int eg_score = eg[us] - eg[~us];
    int score = (mg_score * phase + eg_score * (24 - phase)) / 24;
    score += 10; // tempo
    return score;
}

} // namespace barracuda::eval
