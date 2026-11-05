#include "barracuda/engine/movegen.h"

namespace barracuda {

namespace {

inline void add_promotions(MoveList& list, Square from, Square to, bool capture, bool noisy_only) {
    const int base = capture ? PROMO_CAP_N : PROMO_N;
    list.push(Move(from, to, base + 3)); // queen first
    if (noisy_only) return;
    list.push(Move(from, to, base + 0));
    list.push(Move(from, to, base + 1));
    list.push(Move(from, to, base + 2));
}

// All squares attacked by `by`, computed with `occ` (callers remove our king to see through it).
Bitboard attacked_squares(const Position& pos, Color by, Bitboard occ) {
    Bitboard att = 0;
    const Bitboard pawns = pos.pieces(by, PAWN);
    att |= by == WHITE ? (shift<NORTH_EAST>(pawns) | shift<NORTH_WEST>(pawns))
                       : (shift<SOUTH_EAST>(pawns) | shift<SOUTH_WEST>(pawns));
    Bitboard b = pos.pieces(by, KNIGHT);
    while (b) att |= attacks::knight(pop_lsb(b));
    b = pos.pieces(by, BISHOP, QUEEN);
    while (b) att |= attacks::bishop(pop_lsb(b), occ);
    b = pos.pieces(by, ROOK, QUEEN);
    while (b) att |= attacks::rook(pop_lsb(b), occ);
    att |= attacks::king(pos.king_square(by));
    return att;
}

} // namespace

void generate_moves(const Position& pos, MoveList& list, GenType type) {
    const bool noisy = type == GenType::NOISY;
    const Color us = pos.side_to_move();
    const Color them = ~us;
    const Square ksq = pos.king_square(us);
    const Bitboard ours = pos.pieces(us);
    const Bitboard theirs = pos.pieces(them);
    const Bitboard occ = ours | theirs;
    const Bitboard checkers = pos.checkers();

    // --- king moves ---------------------------------------------------------
    const Bitboard danger = attacked_squares(pos, them, occ ^ square_bb(ksq));
    Bitboard kmoves = attacks::king(ksq) & ~ours & ~danger;
    if (noisy) kmoves &= theirs;
    while (kmoves) {
        const Square to = pop_lsb(kmoves);
        list.push(Move(ksq, to, (theirs & square_bb(to)) ? CAPTURE : QUIET));
    }
    if (more_than_one(checkers)) return; // double check: only king moves

    // --- target mask (check evasions) --------------------------------------
    Bitboard target = ~ours;
    if (checkers) target = between_bb(ksq, lsb(checkers)); // includes the checker itself

    // --- pins ---------------------------------------------------------------
    Bitboard pinned = 0;
    Bitboard snipers = (attacks::rook(ksq, 0) & pos.pieces(them, ROOK, QUEEN)) |
                       (attacks::bishop(ksq, 0) & pos.pieces(them, BISHOP, QUEEN));
    while (snipers) {
        const Square s = pop_lsb(snipers);
        const Bitboard blockers = between_bb(ksq, s) & ~square_bb(s) & occ;
        if (blockers && !more_than_one(blockers) && (blockers & ours)) pinned |= blockers;
    }

    const Bitboard capture_target = theirs & target;
    const Bitboard quiet_target = ~occ & target;

    // --- pawns --------------------------------------------------------------
    const int push = pawn_push(us);
    const Bitboard promo_rank = us == WHITE ? RANK_8_BB : RANK_1_BB;
    Bitboard pawns = pos.pieces(us, PAWN);
    while (pawns) {
        const Square from = pop_lsb(pawns);
        const Bitboard pin_mask = (pinned & square_bb(from)) ? line_bb(ksq, from) : ~0ULL;

        const Square to1 = Square(int(from) + push);
        if (pos.empty(to1)) {
            if (square_bb(to1) & quiet_target & pin_mask) {
                if (square_bb(to1) & promo_rank) add_promotions(list, from, to1, false, noisy);
                else if (!noisy) list.push(Move(from, to1, QUIET));
            }
            if (!noisy && relative_rank(us, from) == 1) {
                const Square to2 = Square(int(to1) + push);
                if (pos.empty(to2) && (square_bb(to2) & quiet_target & pin_mask))
                    list.push(Move(from, to2, DOUBLE_PUSH));
            }
        }

        Bitboard caps = attacks::pawn(us, from) & capture_target & pin_mask;
        while (caps) {
            const Square to = pop_lsb(caps);
            if (square_bb(to) & promo_rank) add_promotions(list, from, to, true, noisy);
            else list.push(Move(from, to, CAPTURE));
        }

        const Square ep = pos.ep_square();
        if (ep != SQ_NONE && (attacks::pawn(us, from) & square_bb(ep))) {
            // Full legality test: simulate the capture and look for any attack on our king.
            const Square capsq = Square(int(ep) - push);
            const Bitboard occ2 = (occ ^ square_bb(from) ^ square_bb(capsq)) | square_bb(ep);
            const Bitboard their_after = theirs & ~square_bb(capsq);
            const bool exposed = (attacks::rook(ksq, occ2) & their_after & pos.pieces(ROOK, QUEEN)) ||
                                 (attacks::bishop(ksq, occ2) & their_after & pos.pieces(BISHOP, QUEEN)) ||
                                 (attacks::knight(ksq) & their_after & pos.pieces(KNIGHT)) ||
                                 (attacks::pawn(us, ksq) & their_after & pos.pieces(PAWN));
            if (!exposed) list.push(Move(from, ep, EP_CAPTURE));
        }
    }

    // --- knights, bishops, rooks, queens -------------------------------------
    for (PieceType pt : {KNIGHT, BISHOP, ROOK, QUEEN}) {
        Bitboard bb = pos.pieces(us, pt);
        while (bb) {
            const Square from = pop_lsb(bb);
            Bitboard att = attacks::piece(pt, from, occ) & (noisy ? capture_target : (capture_target | quiet_target));
            if (pinned & square_bb(from)) att &= line_bb(ksq, from);
            while (att) {
                const Square to = pop_lsb(att);
                list.push(Move(from, to, (theirs & square_bb(to)) ? CAPTURE : QUIET));
            }
        }
    }

    // --- castling -------------------------------------------------------------
    if (!noisy && !checkers) {
        if (pos.can_castle(us == WHITE ? WHITE_OO : BLACK_OO)) {
            const Square f = relative_square(us, F1), g = relative_square(us, G1);
            if (pos.empty(f) && pos.empty(g) && !(danger & (square_bb(f) | square_bb(g))) &&
                pos.piece_on(relative_square(us, H1)) == make_piece(us, ROOK))
                list.push(Move(ksq, g, KING_CASTLE));
        }
        if (pos.can_castle(us == WHITE ? WHITE_OOO : BLACK_OOO)) {
            const Square d = relative_square(us, D1), c = relative_square(us, C1), b = relative_square(us, B1);
            if (pos.empty(d) && pos.empty(c) && pos.empty(b) && !(danger & (square_bb(d) | square_bb(c))) &&
                pos.piece_on(relative_square(us, A1)) == make_piece(us, ROOK))
                list.push(Move(ksq, c, QUEEN_CASTLE));
        }
    }
}

std::uint64_t perft(Position& pos, int depth) {
    MoveList list;
    generate_moves(pos, list);
    if (depth <= 1) return std::uint64_t(depth == 1 ? list.size : 1);
    std::uint64_t nodes = 0;
    for (Move m : list) {
        pos.do_move(m);
        nodes += perft(pos, depth - 1);
        pos.undo_move();
    }
    return nodes;
}

} // namespace barracuda
