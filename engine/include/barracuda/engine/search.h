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
#include <random>
#include <vector>

#include "barracuda/engine/human_model.h"
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
    Move objective_best; // what plain search preferred (differs from `best` when the style chose another move)
    bool styled = false;
    Move trap_reply;     // set when `best` is a trap: the most likely reply that loses
    float trap_prob = 0; // chance (by the human model) that the opponent falls for it
    bool defending = false; // adaptive style: this move was chosen in defence mode
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

// "Strange" playing style. After the normal search, the engine looks at the most unusual-looking
// alternatives (king walks, sacrifices, rim knights, under-promotions, retreats...) and plays one of
// them only if a verification search proves it loses at most `style_margin(best)` centipawns compared
// with the best move. The margin is 0 when the position is bad, small when it is close, and grows
// when the engine is clearly winning - so it is precise when it matters and bizarre when it can afford it.
struct StyleOptions {
    bool enabled = false;
    // share of the move time kept aside for proving strange moves. 0 = only the time the
    // main search leaves unused (it stops early rather than start an iteration it cannot finish):
    // measured to cost nothing, while holding back 30% cost about 50 Elo
    int verify_percent = 0;
    int margin_percent = 100; // scales style_margin()
    // Traps: 0..100, how readily the opponent falls for the natural-looking reply (the opponent
    // model; the website measures it from the player's own moves). 0 disables traps.
    int trap = 0;
    // Draws are worth -contempt centipawns to the engine: it avoids them and plays for the win.
    int contempt = 0;
    // Once clearly winning, steer for the mate (losing king to the edge) instead of more material.
    bool hunt = false;
    // Attack hunger, 0..100: the engine values its own pieces bearing on the enemy king's
    // surroundings (from the opening on), so it builds attacks instead of quiet edges.
    int aggression = 0;
    // Switch between attack (aggression on) and defence (guard its own king, no gambles) according
    // to how the game is going.
    bool adaptive = false;
    std::uint64_t seed = 0;  // 0 = random
};
int style_margin(Value best_score);                 // allowed loss in centipawns
int move_strangeness(const Position& pos, Move m);  // 0 = ordinary, higher = more surprising to humans
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
    void set_style(const StyleOptions& s);
    const StyleOptions& style() const { return style_; }
    std::uint64_t styled_moves() const { return styled_moves_; }
    std::uint64_t style_decisions() const { return style_decisions_; }
    EvalMode eval_mode() const { return eval_mode_; }
    bool nnue_active() const;

    // Blocking. Thread-safe to call stop() from another thread while this runs.
    SearchResult go(const Position& root, const SearchLimits& limits, const InfoCallback& on_info = {});
    void stop() { stop_.store(true, std::memory_order_relaxed); }

    // What a human would most likely play here: the best move of a very shallow search
    // (captures and threats it sees, deep tactics it does not). Used as the opponent model.
    Move predict_human(const Position& pos, int depth = 2);

    // How people choose moves (see human_model.h). Without one, a 2-ply search stands in for it.
    void set_human_model(std::shared_ptr<const HumanModel> m) { human_ = std::move(m); }
    bool has_human_model() const { return human_ != nullptr; }

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
    bool time_up_ = false;
    std::uint64_t styled_moves_ = 0, style_decisions_ = 0; // stop_ was raised by the time/node limit (not by the caller)
    StyleOptions style_;
    std::shared_ptr<const HumanModel> human_;
    std::mt19937_64 style_rng_{12345};
    void apply_style(const Position& root, const MoveList& root_moves, std::int64_t full_hard_ms, SearchResult& result);
    bool try_trap(const MoveList& root_moves, SearchResult& result, bool no_risk);
    Move shallow_best(int ply, int depth);
    Value hunt_bonus(Value v) const;
    Value aggression_bonus() const;
    Color root_color_ = WHITE;
    bool defending_ = false;
    bool defending_by_[COLOR_NB] = {false, false};
    Value last_score_[COLOR_NB] = {0, 0};
  public:
    bool defending() const { return defending_; }
  private:
    Value draw_value(int ply) const {
        // ply 0 is the engine to move: a draw costs it `contempt`, and gains the opponent as much
        const int c = defending_ ? 0 : style_.contempt; // when defending, a draw is welcome
        return style_.enabled ? Value((ply & 1) ? c : -c) : VALUE_DRAW;
    }

    std::uint64_t nodes_ = 0;
    int seldepth_ = 0;
    Move killers_[MAX_PLY + 4][2] = {};
    int history_[COLOR_NB][SQUARE_NB][SQUARE_NB] = {};
    Move pv_[MAX_PLY + 4][MAX_PLY + 4] = {};
    int pv_len_[MAX_PLY + 4] = {};
    Value static_eval_[MAX_PLY + 4] = {};
};

} // namespace barracuda
