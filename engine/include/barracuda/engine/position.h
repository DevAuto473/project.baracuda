// Barracuda - board representation (bitboards + mailbox) with make/unmake.
#pragma once

#include <array>
#include <string>
#include <vector>

#include "barracuda/engine/bitboard.h"
#include "barracuda/engine/types.h"

namespace barracuda {

// Up to three piece changes per move; consumed by the NNUE accumulator for incremental updates.
// from == SQ_NONE  -> piece was added on `to`
// to   == SQ_NONE  -> piece was removed from `from`
struct DirtyPiece {
    int count = 0;
    Piece piece[3] = {NO_PIECE, NO_PIECE, NO_PIECE};
    Square from[3] = {SQ_NONE, SQ_NONE, SQ_NONE};
    Square to[3] = {SQ_NONE, SQ_NONE, SQ_NONE};

    void add(Piece p, Square f, Square t) {
        piece[count] = p;
        from[count] = f;
        to[count] = t;
        ++count;
    }
};

struct StateInfo {
    Key key = 0;
    int castling = NO_CASTLING;
    Square ep = SQ_NONE;
    int rule50 = 0;
    int plies_from_null = 0;
    Piece captured = NO_PIECE;
    Move move;
    Bitboard checkers = 0;
    DirtyPiece dirty;
};

class Position {
public:
    static constexpr const char* START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    // Initialises global tables (attacks, zobrist). Safe to call many times / from many threads.
    static void init();

    Position();
    explicit Position(const std::string& fen);

    bool set_fen(const std::string& fen, std::string* error = nullptr);
    std::string fen() const;
    std::string pretty() const;

    // --- accessors -------------------------------------------------------
    Piece piece_on(Square s) const { return board_[s]; }
    bool empty(Square s) const { return board_[s] == NO_PIECE; }
    Bitboard pieces() const { return by_color_[WHITE] | by_color_[BLACK]; }
    Bitboard pieces(Color c) const { return by_color_[c]; }
    Bitboard pieces(PieceType pt) const { return by_type_[pt]; }
    Bitboard pieces(PieceType a, PieceType b) const { return by_type_[a] | by_type_[b]; }
    Bitboard pieces(Color c, PieceType pt) const { return by_color_[c] & by_type_[pt]; }
    Bitboard pieces(Color c, PieceType a, PieceType b) const { return by_color_[c] & (by_type_[a] | by_type_[b]); }
    Square king_square(Color c) const { return lsb(pieces(c, KING)); }

    Color side_to_move() const { return stm_; }
    Square ep_square() const { return st().ep; }
    int castling_rights() const { return st().castling; }
    bool can_castle(int right) const { return (st().castling & right) != 0; }
    int rule50() const { return st().rule50; }
    int game_ply() const { return game_ply_; }
    int fullmove_number() const { return 1 + game_ply_ / 2; }
    Key key() const { return st().key; }
    Bitboard checkers() const { return st().checkers; }
    bool in_check() const { return st().checkers != 0; }
    const StateInfo& st() const { return states_.back(); }
    const DirtyPiece& dirty() const { return st().dirty; }
    Move last_move() const { return st().move; }
    Piece captured_piece() const { return st().captured; }
    Piece moved_piece(Move m) const { return board_[m.from()]; }
    std::size_t history_size() const { return states_.size(); }

    Bitboard attackers_to(Square s, Bitboard occupied) const;
    Bitboard attackers_to(Square s) const { return attackers_to(s, pieces()); }
    bool is_attacked(Square s, Color by) const { return (attackers_to(s) & pieces(by)) != 0; }
    bool has_non_pawn_material(Color c) const { return (pieces(c) & ~pieces(PAWN, KING)) != 0; }
    int non_pawn_material(Color c) const;

    // --- move making -----------------------------------------------------
    void do_move(Move m);
    void undo_move();
    void do_null_move();
    void undo_null_move();

    // --- game state ------------------------------------------------------
    // `ply` = distance from search root; repetitions inside the tree count as draws immediately.
    bool is_repetition(int ply) const;
    bool is_draw(int ply) const;
    bool insufficient_material() const;

    // Static exchange evaluation: true if the move wins at least `threshold` centipawns.
    bool see_ge(Move m, int threshold) const;

    // Pseudo-check of move sanity against the current position (used to validate TT moves).
    bool is_pseudo_legal_basic(Move m) const;

private:
    void put_piece(Piece p, Square s);
    void remove_piece(Square s);
    void move_piece(Square from, Square to);
    Key compute_key() const;
    void clear();
    StateInfo& st_mut() { return states_.back(); }

    std::array<Bitboard, PIECE_TYPE_NB> by_type_{};
    std::array<Bitboard, COLOR_NB> by_color_{};
    std::array<Piece, SQUARE_NB> board_{};
    Color stm_ = WHITE;
    int game_ply_ = 0;
    std::vector<StateInfo> states_;
};

} // namespace barracuda
