#include "barracuda/engine/search.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <thread>

#include "barracuda/engine/evaluate.h"

namespace barracuda {

namespace {

// Late-move-reduction table, initialised once.
struct LmrTable {
    int r[64][64];
    LmrTable() {
        for (int d = 0; d < 64; ++d)
            for (int m = 0; m < 64; ++m)
                r[d][m] = (d == 0 || m == 0) ? 0 : int(0.75 + std::log(double(d)) * std::log(double(m)) / 2.25);
    }
};
const LmrTable LMR;

constexpr int SCORE_TT = 2'000'000;
constexpr int SCORE_GOOD_CAPTURE = 1'000'000;
constexpr int SCORE_PROMO = 950'000;
constexpr int SCORE_KILLER1 = 900'000;
constexpr int SCORE_KILLER2 = 800'000;
constexpr int SCORE_BAD_CAPTURE = -1'000'000;
constexpr int HISTORY_MAX = 16384;

inline int mvv_lva(const Position& pos, Move m) {
    const int victim = m.is_ep() ? PAWN : int(type_of(pos.piece_on(m.to())));
    const int attacker = int(type_of(pos.piece_on(m.from())));
    return PIECE_VALUE[victim] * 16 - PIECE_VALUE[attacker] / 10;
}

// Selection-sort step: bring the best remaining move to index i.
inline Move pick_move(MoveList& list, int* scores, int i) {
    int best = i;
    for (int j = i + 1; j < list.size; ++j)
        if (scores[j] > scores[best]) best = j;
    std::swap(list.moves[i], list.moves[best]);
    std::swap(scores[i], scores[best]);
    return list.moves[i];
}

} // namespace

std::string score_to_uci(Value v) {
    if (v >= VALUE_MATE_IN_MAX_PLY) return "mate " + std::to_string((VALUE_MATE - v + 1) / 2);
    if (v <= VALUE_MATED_IN_MAX_PLY) return "mate -" + std::to_string((VALUE_MATE + v) / 2);
    return "cp " + std::to_string(v);
}

std::string score_to_string(Value v) {
    if (v >= VALUE_MATE_IN_MAX_PLY) return "#" + std::to_string((VALUE_MATE - v + 1) / 2);
    if (v <= VALUE_MATED_IN_MAX_PLY) return "#-" + std::to_string((VALUE_MATE + v) / 2);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%+.2f", double(v) / 100.0);
    return buf;
}

Search::Search(std::size_t hash_mb) : tt_(hash_mb), acc_(std::make_unique<nnue::AccumulatorStack>()) {
    Position::init();
}

Search::~Search() = default;

void Search::set_hash_mb(std::size_t mb) { tt_.resize(mb); }

void Search::new_game() {
    tt_.clear();
    std::fill(&killers_[0][0], &killers_[0][0] + sizeof(killers_) / sizeof(Move), Move());
    std::fill(&history_[0][0][0], &history_[0][0][0] + sizeof(history_) / sizeof(int), 0);
}

void Search::set_network(std::shared_ptr<const nnue::Network> net) { net_ = std::move(net); }

bool Search::nnue_active() const {
    if (!net_) return false;
    switch (eval_mode_) {
    case EvalMode::CLASSICAL: return false;
    case EvalMode::NNUE: return true;
    case EvalMode::AUTO: default: return net_->is_trained() && net_->samples_trained() >= kAutoNnueMinSamples;
    }
}

Value Search::static_eval(const Position& pos) const {
    if (nnue_active()) return std::clamp(net_->evaluate(pos), VALUE_MATED_IN_MAX_PLY + 1, VALUE_MATE_IN_MAX_PLY - 1);
    return eval::classical(pos);
}

std::int64_t Search::elapsed_ms() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_).count();
}

bool Search::should_stop() {
    if (stop_.load(std::memory_order_relaxed)) return true;
    if (!can_stop_) return false;
    if ((limits_.nodes && nodes_ >= limits_.nodes) || (time_limited_ && elapsed_ms() >= hard_ms_)) {
        stop_.store(true, std::memory_order_relaxed);
        return true;
    }
    return false;
}

Value Search::evaluate_node() {
    Value v = use_nnue_ ? net_->evaluate(acc_->current(), pos_.side_to_move()) : eval::classical(pos_);
    // fade towards a draw as the 50-move counter grows
    v = v * (200 - pos_.rule50()) / 200;
    return std::clamp(v, VALUE_MATED_IN_MAX_PLY + 1, VALUE_MATE_IN_MAX_PLY - 1);
}

void Search::make(Move m) {
    pos_.do_move(m);
    if (use_nnue_) acc_->push(*net_, pos_.dirty());
}

void Search::unmake() {
    pos_.undo_move();
    if (use_nnue_) acc_->pop();
}

void Search::make_null() {
    pos_.do_null_move();
    if (use_nnue_) acc_->push_copy();
}

void Search::unmake_null() {
    pos_.undo_null_move();
    if (use_nnue_) acc_->pop();
}

void Search::score_moves(const MoveList& list, int* scores, Move tt_move, int ply, bool qs) const {
    const Color us = pos_.side_to_move();
    for (int i = 0; i < list.size; ++i) {
        const Move m = list.moves[i];
        int s;
        if (m == tt_move) {
            s = SCORE_TT;
        } else if (m.is_capture()) {
            const bool good = qs || pos_.see_ge(m, -50);
            s = (good ? SCORE_GOOD_CAPTURE : SCORE_BAD_CAPTURE) + mvv_lva(pos_, m);
            if (m.is_promotion()) s += 50'000;
        } else if (m.is_promotion()) {
            s = m.promotion_type() == QUEEN ? SCORE_PROMO : SCORE_BAD_CAPTURE - 100'000;
        } else if (m == killers_[ply][0]) {
            s = SCORE_KILLER1;
        } else if (m == killers_[ply][1]) {
            s = SCORE_KILLER2;
        } else {
            s = history_[us][m.from()][m.to()];
        }
        scores[i] = s;
    }
}

void Search::update_quiet_stats(Move best, const Move* quiets, int n_quiets, int depth, int ply) {
    const Color us = pos_.side_to_move();
    if (killers_[ply][0] != best) {
        killers_[ply][1] = killers_[ply][0];
        killers_[ply][0] = best;
    }
    const int bonus = std::min(depth * depth + 2 * depth, 1600);
    auto gravity = [&](int& h, int b) { h += b - h * std::abs(b) / HISTORY_MAX; };
    gravity(history_[us][best.from()][best.to()], bonus);
    for (int i = 0; i < n_quiets; ++i)
        if (quiets[i] != best) gravity(history_[us][quiets[i].from()][quiets[i].to()], -bonus);
}

// ---------------------------------------------------------------------------
// Quiescence search
// ---------------------------------------------------------------------------
Value Search::qsearch(Value alpha, Value beta, int ply) {
    if ((++nodes_ & 2047) == 0 && should_stop()) return 0;
    if (stop_.load(std::memory_order_relaxed)) return 0;
    seldepth_ = std::max(seldepth_, ply);

    if (pos_.is_draw(ply)) return VALUE_DRAW;
    if (ply >= MAX_PLY - 1) return pos_.in_check() ? VALUE_DRAW : evaluate_node();

    const bool in_check = pos_.in_check();
    const bool pv_node = beta - alpha > 1;

    TTEntry tte;
    const bool tt_hit = tt_.probe(pos_.key(), tte);
    Move tt_move = tt_hit ? Move(tte.move) : Move();
    if (tt_hit && !pv_node) {
        const Value s = TranspositionTable::score_from_tt(tte.score, ply);
        if ((tte.bound() == BOUND_EXACT) || (tte.bound() == BOUND_LOWER && s >= beta) ||
            (tte.bound() == BOUND_UPPER && s <= alpha))
            return s;
    }

    Value best = -VALUE_INFINITE;
    Value stand_pat = -VALUE_INFINITE;
    if (!in_check) {
        stand_pat = tt_hit && tte.eval != 0 ? Value(tte.eval) : evaluate_node();
        if (stand_pat >= beta) return stand_pat;
        if (stand_pat > alpha) alpha = stand_pat;
        best = stand_pat;
    }

    MoveList list;
    generate_moves(pos_, list, in_check ? GenType::ALL : GenType::NOISY);
    if (in_check && list.size == 0) return mated_in(ply);

    int scores[256];
    score_moves(list, scores, tt_move, ply, true);

    Move best_move;
    const Value old_alpha = alpha;
    for (int i = 0; i < list.size; ++i) {
        const Move m = pick_move(list, scores, i);
        if (!in_check) {
            // delta pruning
            if (!m.is_promotion()) {
                const int victim = m.is_ep() ? PAWN : int(type_of(pos_.piece_on(m.to())));
                if (stand_pat + PIECE_VALUE[victim] + 200 <= alpha) continue;
            }
            if (!pos_.see_ge(m, 0)) continue;
        }
        make(m);
        const Value score = -qsearch(-beta, -alpha, ply + 1);
        unmake();
        if (stop_.load(std::memory_order_relaxed)) return 0;

        if (score > best) {
            best = score;
            if (score > alpha) {
                alpha = score;
                best_move = m;
                if (score >= beta) break;
            }
        }
    }

    const Bound b = best >= beta ? BOUND_LOWER : (alpha > old_alpha ? BOUND_EXACT : BOUND_UPPER);
    tt_.store(pos_.key(), best_move, best, in_check ? 0 : stand_pat, 0, b, ply);
    return best;
}

// ---------------------------------------------------------------------------
// Main search
// ---------------------------------------------------------------------------
Value Search::negamax(Value alpha, Value beta, int depth, int ply, bool allow_null) {
    const bool pv_node = beta - alpha > 1;
    const bool root = ply == 0;
    pv_len_[ply] = ply;

    const bool in_check = pos_.in_check();
    if (in_check && depth < MAX_PLY / 2) ++depth; // check extension
    if (depth <= 0) return qsearch(alpha, beta, ply);

    if ((++nodes_ & 2047) == 0 && should_stop()) return 0;
    if (stop_.load(std::memory_order_relaxed)) return 0;
    seldepth_ = std::max(seldepth_, ply);

    if (!root) {
        if (pos_.is_draw(ply)) return VALUE_DRAW;
        if (ply >= MAX_PLY - 1) return in_check ? VALUE_DRAW : evaluate_node();
        // mate distance pruning
        alpha = std::max(alpha, mated_in(ply));
        beta = std::min(beta, mate_in(ply + 1));
        if (alpha >= beta) return alpha;
    }

    // ---- transposition table ----
    TTEntry tte;
    const bool tt_hit = tt_.probe(pos_.key(), tte);
    Move tt_move = tt_hit ? Move(tte.move) : Move();
    if (tt_move && !pos_.is_pseudo_legal_basic(tt_move)) tt_move = Move();
    if (tt_hit && !pv_node && tte.depth >= depth) {
        const Value s = TranspositionTable::score_from_tt(tte.score, ply);
        if ((tte.bound() == BOUND_EXACT) || (tte.bound() == BOUND_LOWER && s >= beta) ||
            (tte.bound() == BOUND_UPPER && s <= alpha))
            return s;
    }

    // ---- static evaluation ----
    Value eval;
    if (in_check) {
        eval = VALUE_NONE;
        static_eval_[ply] = VALUE_NONE;
    } else {
        eval = tt_hit && tte.eval != 0 ? Value(tte.eval) : evaluate_node();
        static_eval_[ply] = eval;
        // TT score is a better estimate when its bound agrees
        if (tt_hit) {
            const Value s = TranspositionTable::score_from_tt(tte.score, ply);
            if ((tte.bound() == BOUND_LOWER && s > eval) || (tte.bound() == BOUND_UPPER && s < eval)) eval = s;
        }
    }
    const bool improving = !in_check && ply >= 2 && static_eval_[ply - 2] != VALUE_NONE &&
                           static_eval_[ply] > static_eval_[ply - 2];

    if (!pv_node && !in_check && !is_mate_score(beta)) {
        // reverse futility pruning
        if (depth <= 8 && eval - (80 - 20 * improving) * depth >= beta) return eval;

        // null-move pruning
        if (allow_null && depth >= 3 && eval >= beta && pos_.has_non_pawn_material(pos_.side_to_move())) {
            const int R = 3 + depth / 4 + std::min(3, (eval - beta) / 200);
            make_null();
            const Value s = -negamax(-beta, -beta + 1, depth - 1 - R, ply + 1, false);
            unmake_null();
            if (stop_.load(std::memory_order_relaxed)) return 0;
            if (s >= beta) return is_mate_score(s) ? beta : s;
        }
    }

    // internal iterative reduction
    if (depth >= 4 && !tt_move) --depth;

    MoveList list;
    generate_moves(pos_, list);
    if (list.size == 0) return in_check ? mated_in(ply) : VALUE_DRAW;

    int scores[256];
    score_moves(list, scores, tt_move, ply, false);

    Value best = -VALUE_INFINITE;
    Move best_move;
    const Value old_alpha = alpha;
    Move quiets[64];
    int n_quiets = 0;
    int searched = 0;

    for (int i = 0; i < list.size; ++i) {
        const Move m = pick_move(list, scores, i);
        const bool quiet = m.is_quiet();
        const bool is_killer = m == killers_[ply][0] || m == killers_[ply][1];

        // ---- move-level pruning (never before a move has been searched) ----
        if (!root && best > VALUE_MATED_IN_MAX_PLY && searched > 0) {
            if (quiet && !in_check) {
                // late move pruning
                if (!pv_node && depth <= 5 && searched >= 3 + depth * depth * (improving ? 2 : 1)) continue;
                // futility pruning
                if (!pv_node && depth <= 6 && eval + 100 + 90 * depth <= alpha) continue;
                // SEE pruning of quiet moves that hang material
                if (depth <= 6 && !pos_.see_ge(m, -40 * depth * depth)) continue;
            } else if (!quiet && depth <= 6 && scores[i] < SCORE_GOOD_CAPTURE && !pos_.see_ge(m, -90 * depth)) {
                continue;
            }
        }

        make(m);
        ++searched;
        const bool gives_check = pos_.in_check();
        const int new_depth = depth - 1;
        Value score;

        if (searched == 1) {
            score = -negamax(-beta, -alpha, new_depth, ply + 1, true);
        } else {
            int r = 0;
            if (depth >= 3 && searched > (pv_node ? 3 : 1) && quiet) {
                r = LMR.r[std::min(depth, 63)][std::min(searched, 63)];
                r += !pv_node;
                r += !improving;
                r -= is_killer;
                r -= gives_check;
                r -= history_[~pos_.side_to_move()][m.from()][m.to()] / 6000;
                r = std::clamp(r, 0, new_depth - 1);
            }
            score = -negamax(-alpha - 1, -alpha, new_depth - r, ply + 1, true);
            if (score > alpha && r > 0) score = -negamax(-alpha - 1, -alpha, new_depth, ply + 1, true);
            if (score > alpha && score < beta) score = -negamax(-beta, -alpha, new_depth, ply + 1, true);
        }
        unmake();
        if (stop_.load(std::memory_order_relaxed)) return 0;

        if (score > best) {
            best = score;
            if (score > alpha) {
                alpha = score;
                best_move = m;
                // update principal variation
                pv_[ply][ply] = m;
                for (int k = ply + 1; k < pv_len_[ply + 1]; ++k) pv_[ply][k] = pv_[ply + 1][k];
                pv_len_[ply] = std::max(pv_len_[ply + 1], ply + 1);
                if (score >= beta) {
                    if (quiet) update_quiet_stats(m, quiets, n_quiets, depth, ply);
                    break;
                }
            }
        }
        if (quiet && n_quiets < 64) quiets[n_quiets++] = m;
    }

    const Bound b = best >= beta ? BOUND_LOWER : (alpha > old_alpha ? BOUND_EXACT : BOUND_UPPER);
    tt_.store(pos_.key(), best_move, best, static_eval_[ply] == VALUE_NONE ? 0 : static_eval_[ply], depth, b, ply);
    return best;
}

SearchResult Search::go(const Position& root, const SearchLimits& limits, const InfoCallback& on_info) {
    start_ = std::chrono::steady_clock::now();
    stop_.store(false, std::memory_order_relaxed);
    limits_ = limits;
    pos_ = root;
    nodes_ = 0;
    can_stop_ = false;
    tt_.new_search();
    use_nnue_ = nnue_active();
    if (use_nnue_) acc_->reset(*net_, pos_);
    // age history so that old information slowly fades
    for (auto& side : history_)
        for (auto& from : side)
            for (int& h : from) h /= 2;
    for (auto& k : killers_) k[0] = k[1] = Move();

    // ---- time management ----
    time_limited_ = false;
    const Color us = root.side_to_move();
    if (limits.movetime_ms > 0) {
        time_limited_ = true;
        soft_ms_ = hard_ms_ = std::max<std::int64_t>(1, limits.movetime_ms - limits.move_overhead_ms / 3);
    } else if (!limits.infinite && limits.time_ms[us] > 0) {
        time_limited_ = true;
        const std::int64_t left = std::max<std::int64_t>(1, limits.time_ms[us] - limits.move_overhead_ms);
        const int mtg = limits.movestogo > 0 ? std::min(limits.movestogo, 40) : 30;
        const std::int64_t base = left / mtg + limits.inc_ms[us] * 3 / 4;
        soft_ms_ = std::max<std::int64_t>(1, std::min(base, left / 2));
        hard_ms_ = std::max<std::int64_t>(1, std::min(base * 4, left * 2 / 3));
    }

    SearchResult result;
    MoveList root_moves;
    generate_moves(pos_, root_moves);
    if (root_moves.size == 0) {
        result.score = pos_.in_check() ? mated_in(0) : VALUE_DRAW;
        return result;
    }
    result.best = root_moves[0];

    Value prev_score = 0;
    const int max_depth = std::clamp(limits.depth, 1, MAX_PLY - 2);
    for (int depth = 1; depth <= max_depth; ++depth) {
        seldepth_ = 0;
        Value score;
        // aspiration windows
        Value delta = 25;
        Value alpha = -VALUE_INFINITE, beta = VALUE_INFINITE;
        if (depth >= 5 && !is_mate_score(prev_score)) {
            alpha = std::max(prev_score - delta, -VALUE_INFINITE);
            beta = std::min(prev_score + delta, VALUE_INFINITE);
        }
        while (true) {
            score = negamax(alpha, beta, depth, 0, false);
            if (stop_.load(std::memory_order_relaxed)) break;
            if (score <= alpha) {
                beta = (alpha + beta) / 2;
                alpha = std::max(score - delta, -VALUE_INFINITE);
            } else if (score >= beta) {
                beta = std::min(score + delta, VALUE_INFINITE);
            } else {
                break;
            }
            delta += delta / 2 + 10;
            if (delta > 1000) {
                alpha = -VALUE_INFINITE;
                beta = VALUE_INFINITE;
            }
        }
        if (stop_.load(std::memory_order_relaxed) && depth > 1) break;

        prev_score = score;
        result.depth = depth;
        result.score = score;
        result.pv.assign(pv_[0], pv_[0] + pv_len_[0]);
        if (!result.pv.empty()) result.best = result.pv[0];
        result.ponder = result.pv.size() > 1 ? result.pv[1] : Move();
        can_stop_ = true;

        const std::int64_t ms = elapsed_ms();
        if (on_info) {
            SearchInfo info;
            info.depth = depth;
            info.seldepth = seldepth_;
            info.score = score;
            info.nodes = nodes_;
            info.time_ms = ms;
            info.nps = ms > 0 ? nodes_ * 1000 / std::uint64_t(ms) : nodes_ * 1000;
            info.hashfull = tt_.hashfull();
            info.pv = result.pv;
            info.nnue = use_nnue_;
            on_info(info);
        }

        if (stop_.load(std::memory_order_relaxed)) break;
        if (limits.nodes && nodes_ >= limits.nodes) break;
        // don't start an iteration we are unlikely to finish
        if (time_limited_ && ms >= soft_ms_ * 6 / 10) break;
        // a forced mate has been found and fully proven at this depth
        if (!limits.infinite && is_mate_score(score) && depth >= 2 * (VALUE_MATE - std::abs(score)) + 2) break;
        // only one legal move: no point thinking long
        if (root_moves.size == 1 && time_limited_ && depth >= 4) break;
    }

    // In "infinite" mode, the GUI/UCI protocol expects us to wait for an explicit stop.
    if (limits.infinite) {
        while (!stop_.load(std::memory_order_relaxed)) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    result.nodes = nodes_;
    result.time_ms = elapsed_ms();
    return result;
}

} // namespace barracuda
