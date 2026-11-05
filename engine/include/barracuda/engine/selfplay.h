// Barracuda - self-play game generator producing NNUE training data.
// Output line format:  <fen> | <score_cp_white> | <result_white 1.0/0.5/0.0>
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

#include "barracuda/engine/search.h"

namespace barracuda {

struct SelfPlayConfig {
    int games = 100;
    int depth = 6;              // fixed search depth per move (ignored if nodes > 0)
    std::uint64_t nodes = 0;    // optional fixed node budget per move
    int random_plies = 8;       // random opening plies for diversity
    int max_plies = 400;
    int resign_score = 1500;    // adjudicate when |score| stays above this...
    int resign_plies = 6;       // ...for this many consecutive plies
    int draw_score = 8;
    int draw_plies = 12;
    int draw_min_ply = 80;
    std::string output_path = "selfplay.txt";
    std::uint64_t seed = 0;     // 0 = time based
};

struct SelfPlayStats {
    int games_done = 0;
    int white_wins = 0, black_wins = 0, draws = 0;
    std::uint64_t positions = 0;
    std::uint64_t moves = 0;
};

struct SelfPlayCallbacks {
    // After each move: position after the move, the move, white-relative score, game index, ply.
    std::function<void(const Position&, Move, Value, int, int)> on_move;
    std::function<void(const SelfPlayStats&, const std::string& result)> on_game_end;
    std::function<void(const std::string&)> log;
};

// Blocking; returns false on I/O error. Honours `stop` between moves.
bool run_selfplay(Search& search, const SelfPlayConfig& cfg, const std::atomic<bool>& stop,
                  const SelfPlayCallbacks& cb, SelfPlayStats* stats, std::string* error);

} // namespace barracuda
