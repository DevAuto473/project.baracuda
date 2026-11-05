// Barracuda - head-to-head matches between two Search instances and Elo estimation.
#pragma once

#include <atomic>
#include <functional>
#include <string>

#include "barracuda/engine/search.h"

namespace barracuda {

struct EloEstimate {
    double score = 0.5; // A's score fraction
    double elo = 0.0;   // A's Elo advantage over B
    double error = 0.0; // 95% confidence half-width, in Elo
};

// Elo difference from W/D/L (A's point of view) with a 95% normal-approximation error bar.
EloEstimate elo_from_results(int wins, int draws, int losses);

struct MatchConfig {
    int games = 40;          // played in colour-swapped pairs from the same random opening
    int movetime_ms = 100;
    int random_plies = 6;
    int max_plies = 240;
    std::uint64_t seed = 0;  // 0 = time based
};

struct MatchStats {
    int games_done = 0, wins = 0, draws = 0, losses = 0; // from A's point of view
    EloEstimate elo;
};

struct MatchCallbacks {
    // After each move: position, move, white-relative score, game index (0-based), ply.
    std::function<void(const Position&, Move, Value, int, int)> on_move;
    std::function<void(const MatchStats&, const std::string&)> on_game_end;
};

// Blocking. Honours `stop` between moves; an interrupted game is not counted.
void run_match(Search& a, Search& b, const MatchConfig& cfg, const std::atomic<bool>& stop, const MatchCallbacks& cb,
               MatchStats* out);

} // namespace barracuda
