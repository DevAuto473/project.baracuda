#include "barracuda/engine/notation.h"

#include "barracuda/engine/movegen.h"

namespace barracuda {

Move parse_uci_move(const Position& pos, const std::string& uci) {
    if (uci.size() < 4) return Move::none();
    MoveList list;
    generate_moves(pos, list);
    for (Move m : list)
        if (m.to_uci() == uci) return m;
    return Move::none();
}

std::string move_to_san(Position& pos, Move m) {
    const Piece pc = pos.piece_on(m.from());
    const PieceType pt = type_of(pc);
    std::string s;

    if (m.is_castle()) {
        s = m.flags() == KING_CASTLE ? "O-O" : "O-O-O";
    } else if (pt == PAWN) {
        if (m.is_capture()) {
            s += char('a' + file_of(m.from()));
            s += 'x';
        }
    } else {
        s += "PNBRQK"[pt];
        // disambiguation
        MoveList list;
        generate_moves(pos, list);
        bool ambiguous = false, same_file = false, same_rank = false;
        for (Move o : list) {
            if (o == m || o.to() != m.to() || pos.piece_on(o.from()) != pc) continue;
            ambiguous = true;
            if (file_of(o.from()) == file_of(m.from())) same_file = true;
            if (rank_of(o.from()) == rank_of(m.from())) same_rank = true;
        }
        if (ambiguous) {
            if (!same_file) s += char('a' + file_of(m.from()));
            else if (!same_rank) s += char('1' + rank_of(m.from()));
            else {
                s += char('a' + file_of(m.from()));
                s += char('1' + rank_of(m.from()));
            }
        }
        if (m.is_capture()) s += 'x';
    }
    if (!m.is_castle()) {
        s += char('a' + file_of(m.to()));
        s += char('1' + rank_of(m.to()));
    }
    if (m.is_promotion()) {
        s += '=';
        s += "NBRQ"[m.flags() & 3];
    }

    pos.do_move(m);
    if (pos.in_check()) {
        MoveList replies;
        generate_moves(pos, replies);
        s += replies.size == 0 ? '#' : '+';
    }
    pos.undo_move();
    return s;
}

static std::string strip_san(std::string s) {
    std::string out;
    for (char c : s)
        if (c != '+' && c != '#' && c != '!' && c != '?' && c != '=') out += c;
    // normalise castling written with zeros
    if (out == "0-0") out = "O-O";
    if (out == "0-0-0") out = "O-O-O";
    return out;
}

Move parse_san_move(const Position& pos, const std::string& san) {
    const std::string want = strip_san(san);
    if (want.empty()) return Move::none();
    Position tmp = pos;
    MoveList list;
    generate_moves(tmp, list);
    for (Move m : list) {
        if (strip_san(move_to_san(tmp, m)) == want) return m;
    }
    // Lenient fallback: accept SAN without capture marks (e.g. "Qh7" for "Qxh7") or UCI.
    for (Move m : list) {
        std::string got = strip_san(move_to_san(tmp, m));
        std::string a, b;
        for (char c : got) if (c != 'x') a += c;
        for (char c : want) if (c != 'x') b += c;
        if (a == b) return m;
    }
    return parse_uci_move(pos, san);
}

std::string pv_to_san(const Position& root, const std::vector<Move>& pv) {
    Position pos = root;
    std::string out;
    for (Move m : pv) {
        MoveList legal;
        generate_moves(pos, legal);
        if (!legal.contains(m)) break;
        if (pos.side_to_move() == WHITE) out += std::to_string(pos.fullmove_number()) + ". ";
        else if (out.empty()) out += std::to_string(pos.fullmove_number()) + "... ";
        out += move_to_san(pos, m) + " ";
        pos.do_move(m);
    }
    if (!out.empty()) out.pop_back();
    return out;
}

GameResult game_result(const Position& pos, std::string* reason) {
    auto set = [&](const char* r) {
        if (reason) *reason = r;
    };
    MoveList list;
    generate_moves(pos, list);
    if (list.size == 0) {
        if (pos.in_check()) {
            set("checkmate");
            return pos.side_to_move() == WHITE ? GameResult::BLACK_WINS : GameResult::WHITE_WINS;
        }
        set("stalemate");
        return GameResult::DRAW;
    }
    if (pos.rule50() >= 100) { set("50-move rule"); return GameResult::DRAW; }
    if (pos.insufficient_material()) { set("insufficient material"); return GameResult::DRAW; }
    // three-fold: ply argument 0 => only true repetitions in game history count
    if (pos.is_repetition(0)) { set("threefold repetition"); return GameResult::DRAW; }
    return GameResult::ONGOING;
}

} // namespace barracuda
