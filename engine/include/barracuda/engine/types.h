// Barracuda Chess Engine - core type definitions.
// Squares are numbered a1 = 0 ... h8 = 63 (little-endian rank-file mapping).
#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace barracuda {

using Bitboard = std::uint64_t;
using Key = std::uint64_t;

enum Color : int { WHITE = 0, BLACK = 1, COLOR_NB = 2 };

constexpr Color operator~(Color c) { return Color(c ^ 1); }

enum PieceType : int { PAWN = 0, KNIGHT, BISHOP, ROOK, QUEEN, KING, PIECE_TYPE_NB, NO_PIECE_TYPE = 6 };

// Piece = color * 6 + type. 12 = empty square.
enum Piece : int {
    W_PAWN = 0, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
    NO_PIECE = 12, PIECE_NB = 12
};

constexpr Piece make_piece(Color c, PieceType pt) { return Piece(c * 6 + pt); }
constexpr PieceType type_of(Piece p) { return PieceType(p % 6); }
constexpr Color color_of(Piece p) { return Color(p / 6); }

// clang-format off
enum Square : int {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
    SQ_NONE = 64, SQUARE_NB = 64
};
// clang-format on

constexpr int file_of(Square s) { return s & 7; }
constexpr int rank_of(Square s) { return s >> 3; }
constexpr Square make_square(int file, int rank) { return Square(rank * 8 + file); }
constexpr Square flip_rank(Square s) { return Square(s ^ 56); }
// Square from the perspective of color c (black sees the board mirrored vertically).
constexpr Square relative_square(Color c, Square s) { return c == WHITE ? s : flip_rank(s); }
constexpr int relative_rank(Color c, Square s) { return c == WHITE ? rank_of(s) : 7 - rank_of(s); }
constexpr bool is_ok(Square s) { return s >= A1 && s <= H8; }

inline Square& operator++(Square& s) { return s = Square(int(s) + 1); }
constexpr Square operator+(Square s, int d) { return Square(int(s) + d); }
constexpr Square operator-(Square s, int d) { return Square(int(s) - d); }

enum Direction : int { NORTH = 8, SOUTH = -8, EAST = 1, WEST = -1,
                       NORTH_EAST = 9, NORTH_WEST = 7, SOUTH_EAST = -7, SOUTH_WEST = -9 };

constexpr int pawn_push(Color c) { return c == WHITE ? NORTH : SOUTH; }

enum CastlingRights : int {
    NO_CASTLING = 0,
    WHITE_OO = 1, WHITE_OOO = 2, BLACK_OO = 4, BLACK_OOO = 8,
    ALL_CASTLING = 15
};

// ---------------------------------------------------------------------------
// Move encoding (16 bits): from(6) | to(6) << 6 | flags(4) << 12
// Flags follow the chessprogramming.org convention.
// ---------------------------------------------------------------------------
enum MoveFlag : std::uint16_t {
    QUIET = 0, DOUBLE_PUSH = 1, KING_CASTLE = 2, QUEEN_CASTLE = 3,
    CAPTURE = 4, EP_CAPTURE = 5,
    PROMO_N = 8, PROMO_B = 9, PROMO_R = 10, PROMO_Q = 11,
    PROMO_CAP_N = 12, PROMO_CAP_B = 13, PROMO_CAP_R = 14, PROMO_CAP_Q = 15
};

class Move {
public:
    constexpr Move() = default;
    constexpr explicit Move(std::uint16_t raw) : data_(raw) {}
    constexpr Move(Square from, Square to, int flags = QUIET)
        : data_(std::uint16_t(from | (to << 6) | (flags << 12))) {}

    constexpr Square from() const { return Square(data_ & 63); }
    constexpr Square to() const { return Square((data_ >> 6) & 63); }
    constexpr int flags() const { return data_ >> 12; }
    constexpr bool is_capture() const { return (flags() & CAPTURE) != 0; }
    constexpr bool is_promotion() const { return (flags() & 8) != 0; }
    constexpr bool is_castle() const { return flags() == KING_CASTLE || flags() == QUEEN_CASTLE; }
    constexpr bool is_ep() const { return flags() == EP_CAPTURE; }
    constexpr bool is_quiet() const { return !is_capture() && !is_promotion(); }
    constexpr PieceType promotion_type() const { return PieceType(KNIGHT + (flags() & 3)); }
    constexpr std::uint16_t raw() const { return data_; }
    constexpr bool is_null() const { return data_ == 0; }
    constexpr explicit operator bool() const { return data_ != 0; }
    constexpr bool operator==(const Move&) const = default;

    static constexpr Move none() { return Move(); }
    // A special "null move" sentinel used by null-move pruning (from == to == B1 is never legal).
    static constexpr Move null_move() { return Move(std::uint16_t(65)); }

    std::string to_uci() const;

private:
    std::uint16_t data_ = 0;
};

// ---------------------------------------------------------------------------
// Score constants
// ---------------------------------------------------------------------------
using Value = int;
constexpr int MAX_PLY = 128;
constexpr Value VALUE_ZERO = 0;
constexpr Value VALUE_DRAW = 0;
constexpr Value VALUE_MATE = 32000;
constexpr Value VALUE_INFINITE = 32001;
constexpr Value VALUE_NONE = 32002;
constexpr Value VALUE_MATE_IN_MAX_PLY = VALUE_MATE - MAX_PLY;
constexpr Value VALUE_MATED_IN_MAX_PLY = -VALUE_MATE_IN_MAX_PLY;

constexpr Value mate_in(int ply) { return VALUE_MATE - ply; }
constexpr Value mated_in(int ply) { return -VALUE_MATE + ply; }
constexpr bool is_mate_score(Value v) { return v >= VALUE_MATE_IN_MAX_PLY || v <= VALUE_MATED_IN_MAX_PLY; }

constexpr std::array<int, 7> PIECE_VALUE = {100, 320, 330, 500, 900, 20000, 0};

constexpr char PIECE_CHARS[] = "PNBRQKpnbrqk.";

} // namespace barracuda
