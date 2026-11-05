#include "barracuda/engine/import_pgn.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_set>
#include <vector>

#include "barracuda/engine/movegen.h"
#include "barracuda/engine/notation.h"
#include "barracuda/engine/search.h"

namespace barracuda {
namespace {

struct Sample {
    std::string fen;
    const char* result;      // "1.0" / "0.5" / "0.0" (White's point of view)
    bool has_eval = false;
    int eval_cp = 0;         // White's point of view
};

struct PgnGame {
    std::string white_elo, black_elo, result, fen, movetext;
};

// "[Key \"value\"]" -> true and fills key/value
bool parse_tag(const std::string& line, std::string& key, std::string& value) {
    if (line.size() < 5 || line[0] != '[' || line.back() != ']') return false;
    const auto sp = line.find(' ');
    const auto q1 = line.find('"');
    const auto q2 = line.rfind('"');
    if (sp == std::string::npos || q1 == std::string::npos || q2 <= q1) return false;
    key = line.substr(1, sp - 1);
    value = line.substr(q1 + 1, q2 - q1 - 1);
    return true;
}

// Reads the next game; false at end of input.
bool read_game(std::istream& in, PgnGame& g) {
    g = PgnGame{};
    std::string line;
    bool have_any = false, in_moves = false;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) {
            if (in_moves) return true; // blank line after the movetext ends the game
            continue;
        }
        if (line[0] == '[' && !in_moves) {
            std::string k, v;
            if (parse_tag(line, k, v)) {
                have_any = true;
                if (k == "WhiteElo") g.white_elo = v;
                else if (k == "BlackElo") g.black_elo = v;
                else if (k == "Result") g.result = v;
                else if (k == "FEN") g.fen = v;
            }
            continue;
        }
        if (!have_any) continue; // junk before the first tag
        in_moves = true;
        g.movetext += line;
        g.movetext += ' ';
    }
    return in_moves; // last game without a trailing blank line
}

struct Token {
    std::string san;
    bool has_eval = false;
    int eval_cp = 0;
};

// Splits movetext into SAN tokens, attaching the "[%eval x]" that follows a move to that move.
std::vector<Token> tokenize(const std::string& text) {
    std::vector<Token> out;
    std::size_t i = 0;
    const std::size_t n = text.size();
    while (i < n) {
        const char c = text[i];
        if (c == '{') { // comment: may hold [%eval ...]
            const auto end = text.find('}', i);
            const std::string body = text.substr(i + 1, end == std::string::npos ? std::string::npos : end - i - 1);
            const auto p = body.find("[%eval ");
            if (p != std::string::npos && !out.empty()) {
                const std::string v = body.substr(p + 7, body.find(']', p) - (p + 7));
                if (!v.empty() && v[0] != '#') {
                    out.back().has_eval = true;
                    out.back().eval_cp = int(std::lround(std::atof(v.c_str()) * 100.0));
                } else if (!v.empty()) { // mate score: not a usable training score
                    out.back().has_eval = false;
                }
            }
            if (end == std::string::npos) break;
            i = end + 1;
        } else if (c == '(') { // variation: skip (rare)
            int depth = 1;
            ++i;
            while (i < n && depth) { depth += text[i] == '(' ? 1 : text[i] == ')' ? -1 : 0; ++i; }
        } else if (c == ';') { // rest-of-line comment
            while (i < n && text[i] != '\n') ++i;
        } else if (c == ' ' || c == '\t' || c == '\n') {
            ++i;
        } else {
            std::size_t j = i;
            while (j < n && text[j] != ' ' && text[j] != '\n' && text[j] != '{' && text[j] != '(') ++j;
            std::string tok = text.substr(i, j - i);
            i = j;
            if (tok.empty() || tok[0] == '$') continue;               // NAG
            if (tok == "1-0" || tok == "0-1" || tok == "1/2-1/2" || tok == "*") continue;
            // strip a leading move number: "12." or "12..."
            std::size_t k = 0;
            while (k < tok.size() && std::isdigit(static_cast<unsigned char>(tok[k]))) ++k;
            if (k > 0 && k < tok.size() && tok[k] == '.') {
                while (k < tok.size() && tok[k] == '.') ++k;
                tok = tok.substr(k);
            } else if (k == tok.size()) {
                continue; // a bare number
            }
            if (tok.empty()) continue;
            out.push_back({tok, false, 0});
        }
    }
    return out;
}

} // namespace

bool import_pgn(std::istream& in, const std::string& out_path, const ImportPgnOptions& opt,
                std::shared_ptr<const nnue::Network> net, const std::atomic<bool>& stop, ImportPgnStats& st,
                std::string* error, const ImportPgnLog& log) {
    std::ofstream out;
    if (!opt.moves_only) {
        out.open(out_path, std::ios::trunc);
        if (!out) {
            if (error) *error = "cannot write '" + out_path + "'";
            return false;
        }
    }
    std::ofstream moves_file;
    if (!opt.moves_out.empty()) {
        moves_file.open(opt.moves_out, std::ios::trunc);
        if (!moves_file) {
            if (error) *error = "cannot write '" + opt.moves_out + "'";
            return false;
        }
    }
    std::unordered_set<Key> seen_moves;
    const int moves_every = std::max(1, opt.moves_every);

    // ---- pass 1: read, filter and sample (cheap, single thread) ----
    std::vector<Sample> samples;
    std::unordered_set<Key> seen;
    PgnGame g;
    const int every = std::max(1, opt.every);
    while (st.games_used < opt.max_games && !stop.load() && read_game(in, g)) {
        ++st.games_seen;
        const int we = std::atoi(g.white_elo.c_str()), be = std::atoi(g.black_elo.c_str());
        if (opt.min_elo && (we < opt.min_elo || be < opt.min_elo)) { ++st.rejected_elo; continue; }
        if (opt.max_elo && (we > opt.max_elo || be > opt.max_elo || we == 0 || be == 0)) { ++st.rejected_elo; continue; }
        const char* res = g.result == "1-0" ? "1.0" : g.result == "0-1" ? "0.0" : g.result == "1/2-1/2" ? "0.5" : nullptr;
        if (!res && opt.require_result) { ++st.rejected_other; continue; }
        if (!res) continue;
        if (!g.fen.empty() && g.fen != "startpos") { ++st.rejected_other; continue; } // variants / set-ups
        const std::vector<Token> toks = tokenize(g.movetext);
        if (int(toks.size()) < opt.min_plies) { ++st.rejected_short; continue; }

        Position pos;
        pos.set_fen(Position::START_FEN);
        bool ok = true;
        int next_eval = 0;
        bool next_has_eval = false; // eval of the position about to be considered = eval after the previous move
        int eligible = 0;
        std::vector<Sample> game_samples;
        std::string game_moves; // written only if the whole game parses (no half games)
        std::uint64_t game_moves_n = 0;
        for (std::size_t i = 0; i < toks.size(); ++i) {
            const Move m = parse_san_move(pos, toks[i].san);
            if (!m) { ok = false; break; }
            if (moves_file.is_open() && int(i) >= opt.skip_plies && int(i) % moves_every == 0 &&
                seen_moves.insert(pos.key()).second) {
                game_moves += pos.fen() + " | " + m.to_uci() + "\n";
                ++game_moves_n;
            }
            if (!opt.moves_only && int(i) >= opt.skip_plies && !pos.in_check() && m.is_quiet() && game_result(pos) == GameResult::ONGOING) {
                if (eligible++ % every == 0 && seen.insert(pos.key()).second) {
                    Sample s;
                    s.fen = pos.fen();
                    s.result = res;
                    if (opt.use_pgn_eval && next_has_eval) {
                        s.has_eval = true;
                        s.eval_cp = next_eval;
                    }
                    game_samples.push_back(std::move(s));
                }
            }
            pos.do_move(m);
            next_has_eval = toks[i].has_eval;
            // the PGN gives the evaluation from White's point of view already
            next_eval = toks[i].eval_cp;
        }
        if (!ok) { ++st.illegal; continue; }
        ++st.games_used;
        if (moves_file.is_open()) { moves_file << game_moves; st.moves_written += game_moves_n; }
        for (Sample& s : game_samples) samples.push_back(std::move(s));
        if (log && st.games_used % 5000 == 0)
            log("read " + std::to_string(st.games_used) + " games, " + std::to_string(samples.size()) + " positions so far");
    }

    if (opt.moves_only) return true;

    // ---- pass 2: scores. Positions with a PGN evaluation are free; the rest are searched in parallel ----
    std::atomic<std::size_t> cursor{0};
    std::atomic<std::uint64_t> searched{0};
    std::vector<int> score(samples.size(), 0);
    std::vector<char> valid(samples.size(), 0);
    const int threads = std::max(1, opt.threads);
    auto worker = [&] {
        Search search(8);
        search.set_network(net);
        while (!stop.load()) {
            const std::size_t i = cursor.fetch_add(1);
            if (i >= samples.size()) break;
            const Sample& s = samples[i];
            if (s.has_eval) {
                score[i] = s.eval_cp;
                valid[i] = 1;
                continue;
            }
            Position pos;
            if (!pos.set_fen(s.fen)) continue;
            SearchLimits lim;
            lim.depth = opt.depth;
            const SearchResult r = search.go(pos, lim);
            if (r.best && !is_mate_score(r.score)) {
                score[i] = pos.side_to_move() == WHITE ? r.score : -r.score;
                valid[i] = 1;
            }
            const std::uint64_t n = ++searched;
            if (log && n % 50000 == 0) log("scored " + std::to_string(n) + " positions by search");
        }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) pool.emplace_back(worker);
    for (auto& t : pool) t.join();

    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (!valid[i]) continue;
        // a score beyond +-2500 cp carries no extra information and only slows down training
        const int cp = std::clamp(score[i], -2500, 2500);
        out << samples[i].fen << " | " << cp << " | " << samples[i].result << "\n";
        ++st.positions;
        (samples[i].has_eval ? st.from_pgn_eval : st.from_search) += 1;
    }
    return true;
}

} // namespace barracuda
