#include "barracuda/engine/match.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>

#include "barracuda/engine/movegen.h"
#include "barracuda/engine/notation.h"

namespace barracuda {

namespace {
double elo_of(double s) {
    s = std::clamp(s, 1e-6, 1.0 - 1e-6);
    return -400.0 * std::log10(1.0 / s - 1.0);
}
} // namespace

EloEstimate elo_from_results(int w, int d, int l) {
    EloEstimate e;
    const int n = w + d + l;
    if (n <= 0) return e;
    const double lo_clamp = 0.5 / n;
    const double s = std::clamp((w + 0.5 * d) / n, lo_clamp, 1.0 - lo_clamp);
    e.score = (w + 0.5 * d) / n;
    e.elo = elo_of(s);
    const double var = (w * (1 - s) * (1 - s) + d * (0.5 - s) * (0.5 - s) + l * s * s) / n;
    const double se = std::sqrt(var / n);
    e.error = (elo_of(s + 1.96 * se) - elo_of(s - 1.96 * se)) / 2.0;
    return e;
}

void run_match(Search& a, Search& b, const MatchConfig& cfg, const std::atomic<bool>& stop, const MatchCallbacks& cb,
               MatchStats* out) {
    const std::uint64_t seed =
        cfg.seed ? cfg.seed : std::uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
    MatchStats st;

    for (int g = 0; g < cfg.games && !stop.load(); ++g) {
        // both games of a pair share the opening; colours are swapped
        std::mt19937_64 rng(seed + std::uint64_t(g / 2) * 0x9E3779B97F4A7C15ULL);
        Position pos;
        for (int p = 0; p < cfg.random_plies; ++p) {
            MoveList l;
            generate_moves(pos, l);
            if (l.size == 0) break;
            pos.do_move(l[int(rng() % std::uint64_t(l.size))]);
        }
        if (game_result(pos) != GameResult::ONGOING) pos = Position();

        a.new_game();
        b.new_game();
        const Color a_color = g % 2 == 0 ? WHITE : BLACK;
        GameResult result = GameResult::ONGOING;
        std::string reason;
        int ply = 0, big_count = 0, big_sign = 0, quiet_count = 0;
        bool aborted = false;

        while (true) {
            result = game_result(pos, &reason);
            if (result != GameResult::ONGOING) break;
            if (ply >= cfg.max_plies) {
                result = GameResult::DRAW;
                reason = "max plies";
                break;
            }
            if (stop.load()) {
                aborted = true;
                break;
            }
            Search& mover = pos.side_to_move() == a_color ? a : b;
            SearchLimits lim;
            lim.movetime_ms = cfg.movetime_ms;
            const SearchResult r = mover.go(pos, lim);
            if (!r.best) {
                aborted = true;
                break;
            }
            const Value sw = pos.side_to_move() == WHITE ? r.score : -r.score;
            pos.do_move(r.best);
            ++ply;
            if (cb.on_move) cb.on_move(pos, r.best, sw, g, ply);

            // adjudication: sustained large advantage, or a long dead-drawn endgame
            const int sign = sw > 0 ? 1 : -1;
            if (std::abs(sw) >= 1000) big_count = sign == big_sign ? big_count + 1 : 1, big_sign = sign;
            else big_count = 0;
            quiet_count = (ply >= 120 && std::abs(sw) <= 5) ? quiet_count + 1 : 0;
            if (big_count >= 5) {
                result = big_sign > 0 ? GameResult::WHITE_WINS : GameResult::BLACK_WINS;
                reason = "adjudicated win";
                break;
            }
            if (quiet_count >= 12) {
                result = GameResult::DRAW;
                reason = "adjudicated draw";
                break;
            }
        }
        if (aborted) break;

        int a_score; // +1 win, 0 draw, -1 loss for A
        if (result == GameResult::DRAW) a_score = 0;
        else a_score = (result == GameResult::WHITE_WINS) == (a_color == WHITE) ? 1 : -1;
        st.games_done++;
        (a_score > 0 ? st.wins : a_score == 0 ? st.draws : st.losses)++;
        st.elo = elo_from_results(st.wins, st.draws, st.losses);

        if (cb.on_game_end) {
            const char* what = a_score > 0 ? "win" : a_score == 0 ? "draw" : "loss";
            cb.on_game_end(st, std::string("Barracuda (") + (a_color == WHITE ? "White" : "Black") + ") " + what +
                                   " - " + reason + ", " + std::to_string(ply) + " plies");
        }
    }
    if (out) *out = st;
}

} // namespace barracuda
