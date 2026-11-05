#include "barracuda/engine/selfplay.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <random>
#include <vector>

#include "barracuda/engine/movegen.h"
#include "barracuda/engine/notation.h"

namespace barracuda {

namespace {
struct Record {
    std::string fen;
    int score_white;
};
} // namespace

bool run_selfplay(Search& search, const SelfPlayConfig& cfg, const std::atomic<bool>& stop,
                  const SelfPlayCallbacks& cb, SelfPlayStats* stats_out, std::string* error) {
    std::ofstream out(cfg.output_path, std::ios::app);
    if (!out) {
        if (error) *error = "cannot open '" + cfg.output_path + "' for writing";
        return false;
    }
    const std::uint64_t seed =
        cfg.seed ? cfg.seed : std::uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
    std::mt19937_64 rng(seed);
    SelfPlayStats stats;

    for (int game = 0; game < cfg.games && !stop.load(); ++game) {
        search.new_game();
        Position pos;
        std::vector<Record> records;

        // ---- random opening ----
        bool aborted = false;
        for (int p = 0; p < cfg.random_plies; ++p) {
            MoveList legal;
            generate_moves(pos, legal);
            if (legal.size == 0) {
                aborted = true;
                break;
            }
            pos.do_move(legal[int(rng() % std::uint64_t(legal.size))]);
        }
        if (aborted || game_result(pos) != GameResult::ONGOING) {
            --game; // retry with a fresh opening
            continue;
        }

        // ---- play ----
        GameResult result = GameResult::ONGOING;
        std::string reason;
        int resign_count = 0, draw_count = 0;
        int ply = 0;
        while (!stop.load()) {
            result = game_result(pos, &reason);
            if (result != GameResult::ONGOING) break;
            if (ply >= cfg.max_plies) {
                result = GameResult::DRAW;
                reason = "max plies";
                break;
            }

            SearchLimits lim;
            lim.depth = cfg.nodes ? MAX_PLY - 2 : cfg.depth;
            lim.nodes = cfg.nodes;
            const SearchResult r = search.go(pos, lim);
            if (stop.load()) break;
            if (!r.best) break;

            const Value score_white = pos.side_to_move() == WHITE ? r.score : -r.score;

            // record quiet, non-check, non-mate positions
            if (!pos.in_check() && r.best.is_quiet() && !is_mate_score(r.score))
                records.push_back({pos.fen(), score_white});

            // adjudication
            if (std::abs(r.score) >= cfg.resign_score) ++resign_count; else resign_count = 0;
            if (ply >= cfg.draw_min_ply && std::abs(r.score) <= cfg.draw_score) ++draw_count; else draw_count = 0;

            pos.do_move(r.best);
            ++ply;
            ++stats.moves;
            if (cb.on_move) cb.on_move(pos, r.best, score_white, game, ply);

            if (resign_count >= cfg.resign_plies) {
                result = score_white > 0 ? GameResult::WHITE_WINS : GameResult::BLACK_WINS;
                reason = "adjudicated win";
                break;
            }
            if (draw_count >= cfg.draw_plies) {
                result = GameResult::DRAW;
                reason = "adjudicated draw";
                break;
            }
        }
        if (stop.load() && result == GameResult::ONGOING) break; // drop incomplete game

        const char* res_str = result == GameResult::WHITE_WINS ? "1.0" : result == GameResult::BLACK_WINS ? "0.0" : "0.5";
        for (const Record& rec : records) out << rec.fen << " | " << rec.score_white << " | " << res_str << '\n';
        out.flush();
        if (!out) {
            if (error) *error = "write error on '" + cfg.output_path + "'";
            return false;
        }

        stats.games_done++;
        stats.positions += records.size();
        if (result == GameResult::WHITE_WINS) stats.white_wins++;
        else if (result == GameResult::BLACK_WINS) stats.black_wins++;
        else stats.draws++;

        const std::string summary = std::string(result == GameResult::WHITE_WINS ? "1-0" : result == GameResult::BLACK_WINS ? "0-1" : "1/2-1/2") +
                                    " (" + reason + ", " + std::to_string(ply) + " plies, " +
                                    std::to_string(records.size()) + " positions)";
        if (cb.on_game_end) cb.on_game_end(stats, summary);
    }

    if (stats_out) *stats_out = stats;
    return true;
}

} // namespace barracuda
