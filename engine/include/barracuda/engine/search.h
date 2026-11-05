// Barracuda - iterative-deepening PVS/negamax alpha-beta search.
//
// Features: aspiration windows, transposition table (Zobrist), quiescence search with SEE,
// move ordering (TT move, MVV-LVA, killer moves, history heuristic), null-move pruning,
// reverse futility pruning, late move reductions/pruning, check extensions, mate distance pruning.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "barracuda/engine/position.h"
#include "barracuda/engine/movegen.h"
#include "barracuda/engine/tt.h"
#include "barracuda/nnue/network.h"

namespace barracuda {

struct SearchLimits {
    int depth = MAX_PLY - 1;
    std::int64_t movetime_ms = 0;
    std::int64_t time_ms[COLOR_NB] = {0, 0};
    std::int64_t inc_ms[COLOR_NB] = {0, 0};
    int movestogo = 0;
    std::uint64_t nodes = 0;
    bool infinite = false;
    std::int64_t move_overhead_ms = 30;
};

struct SearchInfo {
    int depth = 0;
    int seldepth = 0;
    Value score = 0;
    std::uint64_t nodes = 0;
    std::uint64_t nps = 0;
    std::int64_t time_ms = 0;
    int hashfull = 0;
    std::vector<Move> pv;
    bool nnue = false;
};

struct SearchResult {
    Move best;
    Move ponder;
    Value score = 0;
    int depth = 0;
    std::uint64_t nodes = 0;
    std::int64_t time_ms = 0;
    std::vector<Move> pv;
};

// AUTO uses the NNUE only once it has seen enough training data to beat the hand-crafted eval;
// a network trained on a few thousand positions is much weaker than the classical evaluation.
enum class EvalMode { AUTO, CLASSICAL, NNUE };
constexpr std::uint64_t kAutoNnueMinSamples = 1'000'000;

// UCI-style score string: "cp 35" or "mate -3"
std::string score_to_uci(Value v);
// Human readable: "+0.35" or "#-3"
std::string score_to_string(Value v);

class Search {
public:
    using InfoCallback = std::function<void(const SearchInfo&)>;

    explicit Search(std::size_t hash_mb = 64);
    ~Search();
    Search(const Search&) = delete;
    Search& operator=(const Search&) = delete;

    void set_hash_mb(std::size_t mb);
    void new_game(); // clears TT and heuristics

    void set_network(std::shared_ptr<const nnue::Network> net);
    std::shared_ptr<const nnue::Network> network() const { return net_; }
    void set_eval_mode(EvalMode m) { eval_mode_ = m; }
    EvalMode eval_mode() const { return eval_mode_; }
    bool nnue_active() const;

    // Blocking. Thread-safe to call stop() from another thread while this runs.
    SearchResult go(const Position& root, const SearchLimits& limits, const InfoCallback& on_info = {});
    void stop() { stop_.store(true, std::memory_order_relaxed); }

    // Static evaluation of `pos` with the active evaluator (side-to-move relative).
    Value static_eval(const Position& pos) const;

private:

    Value negamax(Value alpha, Value beta, int depth, int ply, bool allow_null);
    Value qsearch(Value alpha, Value beta, int ply);
    Value evaluate_node();
    void make(Move m);
    void unmake();
    void make_null();
    void unmake_null();
    bool should_stop();
    std::int64_t elapsed_ms() const;
    void score_moves(const MoveList& list, int* scores, Move tt_move, int ply, bool qs) const;
    void update_quiet_stats(Move best, const Move* quiets, int n_quiets, int depth, int ply);

    TranspositionTable tt_;
    Position pos_;
    std::shared_ptr<const nnue::Network> net_;
    std::unique_ptr<nnue::AccumulatorStack> acc_;
    bool use_nnue_ = false;
    EvalMode eval_mode_ = EvalMode::AUTO;

    std::atomic<bool> stop_{false};
    SearchLimits limits_;
    std::chrono::steady_clock::time_point start_;
    std::int64_t soft_ms_ = 0, hard_ms_ = 0;
    bool time_limited_ = false;
    bool can_stop_ = false;

    std::uint64_t nodes_ = 0;
    int seldepth_ = 0;
    Move killers_[MAX_PLY + 4][2] = {};
    int history_[COLOR_NB][SQUARE_NB][SQUARE_NB] = {};
    Move pv_[MAX_PLY + 4][MAX_PLY + 4] = {};
    int pv_len_[MAX_PLY + 4] = {};
    Value static_eval_[MAX_PLY + 4] = {};
};

} // namespace barracuda
