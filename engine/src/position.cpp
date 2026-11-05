#include "barracuda/engine/position.h"

#include <sstream>

#include "barracuda/engine/zobrist.h"

namespace barracuda {

namespace {

// castling-rights mask applied on every move touching a square
constexpr std::array<int, SQUARE_NB> make_castling_mask() {
    std::array<int, SQUARE_NB> m{};
    for (auto& v : m) v = ALL_CASTLING;
    m[A1] &= ~WHITE_OOO;
    m[H1] &= ~WHITE_OO;
    m[E1] &= ~(WHITE_OO | WHITE_OOO);
    m[A8] &= ~BLACK_OOO;
    m[H8] &= ~BLACK_OO;
    m[E8] &= ~(BLACK_OO | BLACK_OOO);
    return m;
}
constexpr std::array<int, SQUARE_NB> CASTLING_MASK = make_castling_mask();

int char_to_piece(char c) {
    for (int i = 0; i < 12; ++i)
        if (PIECE_CHARS[i] == c) return i;
    return -1;
}

} // namespace

void Position::init() {
    attacks::init();
    zobrist::init();
}

Position::Position() {
    init();
    set_fen(START_FEN);
}

Position::Position(const std::string& fen) {
    init();
    if (!set_fen(fen)) set_fen(START_FEN);
}

void Position::clear() {
    by_type_.fill(0);
    by_color_.fill(0);
    board_.fill(NO_PIECE);
    stm_ = WHITE;
    game_ply_ = 0;
    states_.clear();
    states_.reserve(1024);
    states_.emplace_back();
}

void Position::put_piece(Piece p, Square s) {
    board_[s] = p;
    by_type_[type_of(p)] |= square_bb(s);
    by_color_[color_of(p)] |= square_bb(s);
}

void Position::remove_piece(Square s) {
    const Piece p = board_[s];
    by_type_[type_of(p)] ^= square_bb(s);
    by_color_[color_of(p)] ^= square_bb(s);
    board_[s] = NO_PIECE;
}

void Position::move_piece(Square from, Square to) {
    const Piece p = board_[from];
    const Bitboard ft = square_bb(from) | square_bb(to);
    by_type_[type_of(p)] ^= ft;
    by_color_[color_of(p)] ^= ft;
    board_[from] = NO_PIECE;
    board_[to] = p;
}

bool Position::set_fen(const std::string& fen, std::string* error) {
    auto fail = [&](const std::string& msg) {
        if (error) *error = msg;
        return false;
    };

    std::istringstream ss(fen);
    std::string placement, side, castle, ep;
    int halfmove = 0, fullmove = 1;
    if (!(ss >> placement >> side)) return fail("FEN: missing fields");
    if (!(ss >> castle)) castle = "-";
    if (!(ss >> ep)) ep = "-";
    if (!(ss >> halfmove)) halfmove = 0;
    if (!(ss >> fullmove)) fullmove = 1;

    // Validate into a temporary so that a malformed FEN never corrupts *this.
    Position tmp_holder_guard = *this;
    clear();

    int rank = 7, file = 0;
    for (char c : placement) {
        if (c == '/') {
            if (file != 8) { *this = tmp_holder_guard; return fail("FEN: bad rank length"); }
            --rank;
            file = 0;
        } else if (c >= '1' && c <= '8') {
            file += c - '0';
        } else {
            const int p = char_to_piece(c);
            if (p < 0 || file > 7 || rank < 0) { *this = tmp_holder_guard; return fail("FEN: bad piece placement"); }
            put_piece(Piece(p), make_square(file, rank));
            ++file;
        }
        if (file > 8) { *this = tmp_holder_guard; return fail("FEN: rank overflow"); }
    }
    if (rank != 0 || file != 8) { *this = tmp_holder_guard; return fail("FEN: incomplete placement"); }
    if (popcount(pieces(WHITE, KING)) != 1 || popcount(pieces(BLACK, KING)) != 1) {
        *this = tmp_holder_guard;
        return fail("FEN: each side needs exactly one king");
    }

    if (side != "w" && side != "b") { *this = tmp_holder_guard; return fail("FEN: bad side to move"); }
    stm_ = side == "w" ? WHITE : BLACK;

    StateInfo& s = st_mut();
    s.castling = NO_CASTLING;
    for (char c : castle) {
        switch (c) {
        case 'K': if (piece_on(E1) == W_KING && piece_on(H1) == W_ROOK) s.castling |= WHITE_OO; break;
        case 'Q': if (piece_on(E1) == W_KING && piece_on(A1) == W_ROOK) s.castling |= WHITE_OOO; break;
        case 'k': if (piece_on(E8) == B_KING && piece_on(H8) == B_ROOK) s.castling |= BLACK_OO; break;
        case 'q': if (piece_on(E8) == B_KING && piece_on(A8) == B_ROOK) s.castling |= BLACK_OOO; break;
        default: break;
        }
    }

    s.ep = SQ_NONE;
    if (ep.size() == 2 && ep[0] >= 'a' && ep[0] <= 'h' && (ep[1] == '3' || ep[1] == '6')) {
        const Square eps = make_square(ep[0] - 'a', ep[1] - '1');
        // only keep the ep square if a capture is actually possible (canonical hashing)
        if (attacks::pawn(~stm_, eps) & pieces(stm_, PAWN)) s.ep = eps;
    }

    s.rule50 = std::max(0, halfmove);
    game_ply_ = std::max(0, 2 * (fullmove - 1)) + (stm_ == BLACK ? 1 : 0);
    s.key = compute_key();
    s.checkers = attackers_to(king_square(stm_)) & pieces(~stm_);

    // The side not to move must not be in check.
    if (attackers_to(king_square(~stm_)) & pieces(stm_)) {
        *this = tmp_holder_guard;
        return fail("FEN: side not to move is in check");
    }
    return true;
}

std::string Position::fen() const {
    std::string out;
    for (int r = 7; r >= 0; --r) {
        int empty_count = 0;
        for (int f = 0; f < 8; ++f) {
            const Piece p = piece_on(make_square(f, r));
            if (p == NO_PIECE) {
                ++empty_count;
            } else {
                if (empty_count) out += char('0' + empty_count);
                empty_count = 0;
                out += PIECE_CHARS[p];
            }
        }
        if (empty_count) out += char('0' + empty_count);
        if (r) out += '/';
    }
    out += stm_ == WHITE ? " w " : " b ";
    std::string c;
    if (can_castle(WHITE_OO)) c += 'K';
    if (can_castle(WHITE_OOO)) c += 'Q';
    if (can_castle(BLACK_OO)) c += 'k';
    if (can_castle(BLACK_OOO)) c += 'q';
    out += c.empty() ? "-" : c;
    out += ' ';
    if (ep_square() == SQ_NONE) {
        out += '-';
    } else {
        out += char('a' + file_of(ep_square()));
        out += char('1' + rank_of(ep_square()));
    }
    out += ' ' + std::to_string(rule50()) + ' ' + std::to_string(fullmove_number());
    return out;
}

std::string Position::pretty() const {
    std::string s = " +---+---+---+---+---+---+---+---+\n";
    for (int r = 7; r >= 0; --r) {
        for (int f = 0; f < 8; ++f) {
            s += " | ";
            s += PIECE_CHARS[piece_on(make_square(f, r))];
        }
        s += " | " + std::to_string(r + 1) + "\n +---+---+---+---+---+---+---+---+\n";
    }
    s += "   a   b   c   d   e   f   g   h\n\nFen: " + fen() + "\n";
    std::ostringstream k;
    k << std::hex << key();
    s += "Key: " + k.str() + "\n";
    return s;
}

Key Position::compute_key() const {
    Key k = 0;
    for (Square s = A1; s <= H8; ++s)
        if (board_[s] != NO_PIECE) k ^= zobrist::PIECE_SQUARE[board_[s]][s];
    k ^= zobrist::CASTLING[st().castling];
    if (st().ep != SQ_NONE) k ^= zobrist::EN_PASSANT_FILE[file_of(st().ep)];
    if (stm_ == BLACK) k ^= zobrist::SIDE_TO_MOVE;
    return k;
}

Bitboard Position::attackers_to(Square s, Bitboard occ) const {
    return (attacks::pawn(BLACK, s) & pieces(WHITE, PAWN)) | (attacks::pawn(WHITE, s) & pieces(BLACK, PAWN)) |
           (attacks::knight(s) & pieces(KNIGHT)) | (attacks::king(s) & pieces(KING)) |
           (attacks::bishop(s, occ) & pieces(BISHOP, QUEEN)) | (attacks::rook(s, occ) & pieces(ROOK, QUEEN));
}

int Position::non_pawn_material(Color c) const {
    return popcount(pieces(c, KNIGHT)) * PIECE_VALUE[KNIGHT] + popcount(pieces(c, BISHOP)) * PIECE_VALUE[BISHOP] +
           popcount(pieces(c, ROOK)) * PIECE_VALUE[ROOK] + popcount(pieces(c, QUEEN)) * PIECE_VALUE[QUEEN];
}

void Position::do_move(Move m) {
    const StateInfo& prev = states_.back();
    StateInfo next;
    next.key = prev.key;
    next.castling = prev.castling;
    next.rule50 = prev.rule50 + 1;
    next.plies_from_null = prev.plies_from_null + 1;
    next.move = m;
    next.ep = SQ_NONE;

    const Color us = stm_;
    const Color them = ~us;
    const Square from = m.from();
    const Square to = m.to();
    const Piece pc = board_[from];

    next.key ^= zobrist::SIDE_TO_MOVE;
    if (prev.ep != SQ_NONE) next.key ^= zobrist::EN_PASSANT_FILE[file_of(prev.ep)];

    if (m.is_castle()) {
        const bool king_side = m.flags() == KING_CASTLE;
        const Square rfrom = relative_square(us, king_side ? H1 : A1);
        const Square rto = relative_square(us, king_side ? F1 : D1);
        const Piece rook = board_[rfrom];
        move_piece(from, to);
        move_piece(rfrom, rto);
        next.key ^= zobrist::PIECE_SQUARE[pc][from] ^ zobrist::PIECE_SQUARE[pc][to];
        next.key ^= zobrist::PIECE_SQUARE[rook][rfrom] ^ zobrist::PIECE_SQUARE[rook][rto];
        next.dirty.add(pc, from, to);
        next.dirty.add(rook, rfrom, rto);
    } else {
        if (m.is_capture()) {
            const Square capsq = m.is_ep() ? Square(int(to) - pawn_push(us)) : to;
            next.captured = board_[capsq];
            next.key ^= zobrist::PIECE_SQUARE[next.captured][capsq];
            next.dirty.add(next.captured, capsq, SQ_NONE);
            remove_piece(capsq);
            next.rule50 = 0;
        }
        move_piece(from, to);
        next.key ^= zobrist::PIECE_SQUARE[pc][from] ^ zobrist::PIECE_SQUARE[pc][to];

        if (type_of(pc) == PAWN) {
            next.rule50 = 0;
            if (m.flags() == DOUBLE_PUSH) {
                const Square eps = Square(int(from) + pawn_push(us));
                if (attacks::pawn(us, eps) & pieces(them, PAWN)) {
                    next.ep = eps;
                    next.key ^= zobrist::EN_PASSANT_FILE[file_of(eps)];
                }
            }
            if (m.is_promotion()) {
                const Piece promo = make_piece(us, m.promotion_type());
                remove_piece(to);
                put_piece(promo, to);
                next.key ^= zobrist::PIECE_SQUARE[pc][to] ^ zobrist::PIECE_SQUARE[promo][to];
                next.dirty.add(pc, from, SQ_NONE);
                next.dirty.add(promo, SQ_NONE, to);
            } else {
                next.dirty.add(pc, from, to);
            }
        } else {
            next.dirty.add(pc, from, to);
        }
    }

    const int new_castling = prev.castling & CASTLING_MASK[from] & CASTLING_MASK[to];
    if (new_castling != prev.castling) {
        next.key ^= zobrist::CASTLING[prev.castling] ^ zobrist::CASTLING[new_castling];
        next.castling = new_castling;
    }

    stm_ = them;
    ++game_ply_;
    next.checkers = attackers_to(king_square(them)) & pieces(us);
    states_.push_back(next);
}

void Position::undo_move() {
    const StateInfo& s = states_.back();
    const Move m = s.move;
    const Color us = ~stm_; // side that made the move
    const Square from = m.from();
    const Square to = m.to();

    if (m.is_castle()) {
        const bool king_side = m.flags() == KING_CASTLE;
        const Square rfrom = relative_square(us, king_side ? H1 : A1);
        const Square rto = relative_square(us, king_side ? F1 : D1);
        move_piece(to, from);
        move_piece(rto, rfrom);
    } else {
        if (m.is_promotion()) {
            remove_piece(to);
            put_piece(make_piece(us, PAWN), to);
        }
        move_piece(to, from);
        if (s.captured != NO_PIECE) {
            const Square capsq = m.is_ep() ? Square(int(to) - pawn_push(us)) : to;
            put_piece(s.captured, capsq);
        }
    }

    stm_ = us;
    --game_ply_;
    states_.pop_back();
}

void Position::do_null_move() {
    const StateInfo& prev = states_.back();
    StateInfo next;
    next.key = prev.key ^ zobrist::SIDE_TO_MOVE;
    if (prev.ep != SQ_NONE) next.key ^= zobrist::EN_PASSANT_FILE[file_of(prev.ep)];
    next.castling = prev.castling;
    next.rule50 = prev.rule50 + 1;
    next.plies_from_null = 0;
    next.move = Move::null_move();
    stm_ = ~stm_;
    ++game_ply_;
    next.checkers = 0; // null move is never made while in check
    states_.push_back(next);
}

void Position::undo_null_move() {
    stm_ = ~stm_;
    --game_ply_;
    states_.pop_back();
}

bool Position::is_repetition(int ply) const {
    const StateInfo& cur = st();
    const int end = std::min(cur.rule50, cur.plies_from_null);
    const int n = int(states_.size());
    int count = 0;
    for (int i = 4; i <= end && n - 1 - i >= 0; i += 2) {
        if (states_[n - 1 - i].key == cur.key) {
            if (i <= ply) return true; // repetition inside the search tree
            if (++count >= 2) return true; // three-fold in game history
        }
    }
    return false;
}

bool Position::insufficient_material() const {
    if (pieces(PAWN) | pieces(ROOK) | pieces(QUEEN)) return false;
    const int minors = popcount(pieces(KNIGHT) | pieces(BISHOP));
    if (minors <= 1) return true;
    // K+B vs K+B with same-coloured bishops
    if (!pieces(KNIGHT) && popcount(pieces(WHITE, BISHOP)) <= 1 && popcount(pieces(BLACK, BISHOP)) <= 1) {
        constexpr Bitboard DARK = 0xAA55AA55AA55AA55ULL;
        const Bitboard b = pieces(BISHOP);
        return (b & DARK) == b || (b & ~DARK) == b;
    }
    return false;
}

bool Position::is_draw(int ply) const {
    if (st().rule50 >= 100) return true; // (ignoring the mate-on-100th-ply corner case)
    return insufficient_material() || is_repetition(ply);
}

bool Position::see_ge(Move m, int threshold) const {
    if (m.is_castle() || m.is_promotion() || m.is_ep()) return 0 >= threshold;

    const Square from = m.from();
    const Square to = m.to();
    int swap = PIECE_VALUE[type_of(piece_on(to)) == NO_PIECE_TYPE ? 6 : type_of(piece_on(to))];
    if (piece_on(to) == NO_PIECE) swap = 0;
    swap -= threshold;
    if (swap < 0) return false;

    swap = PIECE_VALUE[type_of(piece_on(from))] - swap;
    if (swap <= 0) return true;

    Bitboard occ = pieces() ^ square_bb(from) ^ square_bb(to);
    Color side = color_of(piece_on(from));
    Bitboard attackers = attackers_to(to, occ);
    int res = 1;

    while (true) {
        side = ~side;
        attackers &= occ;
        Bitboard mine = attackers & pieces(side);
        if (!mine) break;
        res ^= 1;

        Bitboard bb;
        if ((bb = mine & pieces(PAWN))) {
            if ((swap = PIECE_VALUE[PAWN] - swap) < res) break;
            occ ^= square_bb(lsb(bb));
            attackers |= attacks::bishop(to, occ) & pieces(BISHOP, QUEEN);
        } else if ((bb = mine & pieces(KNIGHT))) {
            if ((swap = PIECE_VALUE[KNIGHT] - swap) < res) break;
            occ ^= square_bb(lsb(bb));
        } else if ((bb = mine & pieces(BISHOP))) {
            if ((swap = PIECE_VALUE[BISHOP] - swap) < res) break;
            occ ^= square_bb(lsb(bb));
            attackers |= attacks::bishop(to, occ) & pieces(BISHOP, QUEEN);
        } else if ((bb = mine & pieces(ROOK))) {
            if ((swap = PIECE_VALUE[ROOK] - swap) < res) break;
            occ ^= square_bb(lsb(bb));
            attackers |= attacks::rook(to, occ) & pieces(ROOK, QUEEN);
        } else if ((bb = mine & pieces(QUEEN))) {
            if ((swap = PIECE_VALUE[QUEEN] - swap) < res) break;
            occ ^= square_bb(lsb(bb));
            attackers |= (attacks::bishop(to, occ) & pieces(BISHOP, QUEEN)) |
                         (attacks::rook(to, occ) & pieces(ROOK, QUEEN));
        } else {
            // king: can only capture if the opponent has no more attackers
            return (attackers & ~pieces(side)) ? bool(res ^ 1) : bool(res);
        }
    }
    return bool(res);
}

bool Position::is_pseudo_legal_basic(Move m) const {
    if (!m || m == Move::null_move()) return false;
    const Piece p = piece_on(m.from());
    if (p == NO_PIECE || color_of(p) != stm_) return false;
    const Piece target = piece_on(m.to());
    if (target != NO_PIECE && color_of(target) == stm_) return false;
    if (m.is_capture() && !m.is_ep() && target == NO_PIECE) return false;
    if (!m.is_capture() && target != NO_PIECE) return false;
    return true;
}

std::string Move::to_uci() const {
    if (is_null()) return "0000";
    if (*this == null_move()) return "0000";
    std::string s;
    s += char('a' + file_of(from()));
    s += char('1' + rank_of(from()));
    s += char('a' + file_of(to()));
    s += char('1' + rank_of(to()));
    if (is_promotion()) s += "nbrq"[flags() & 3];
    return s;
}

} // namespace barracuda
