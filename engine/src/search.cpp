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
    defending_ = false;
    defending_by_[WHITE] = defending_by_[BLACK] = false;
    last_score_[WHITE] = last_score_[BLACK] = 0;
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
        time_up_ = true;
        stop_.store(true, std::memory_order_relaxed);
        return true;
    }
    return false;
}

Value Search::evaluate_node() {
    Value v = use_nnue_ ? net_->evaluate(acc_->current(), pos_.side_to_move()) : eval::classical(pos_);
    if (style_.enabled && style_.hunt) v = hunt_bonus(v);
    if (style_.enabled && (style_.aggression > 0 || defending_)) v += aggression_bonus();
    // fade towards a draw as the 50-move counter grows
    v = v * (200 - pos_.rule50()) / 200;
    return std::clamp(v, VALUE_MATED_IN_MAX_PLY + 1, VALUE_MATE_IN_MAX_PLY - 1);
}

// "Hunt" (killer instinct): once one side is clearly winning, reward driving the losing king to
// the edge of the board and, when the defender has little left, walking the winning king up to it.
// This turns a won position into a mate in fewer moves instead of collecting more material.
Value Search::hunt_bonus(Value v) const {
    if (v > -200 && v < 200) return v;
    const Color stm = pos_.side_to_move();
    const Color winner = v > 0 ? stm : ~stm, loser = ~winner;
    const Square kl = pos_.king_square(loser), kw = pos_.king_square(winner);
    const int fl = file_of(kl), rl = rank_of(kl);
    const int edge = std::max(3 - fl, fl - 4) + std::max(3 - rl, rl - 4); // 0 in the centre .. 6 in a corner
    const int dist = std::abs(file_of(kw) - fl) + std::abs(rank_of(kw) - rl); // 1 .. 14
    int bonus = 16 * edge;
    if (pos_.non_pawn_material(loser) <= 1300 && !pos_.pieces(loser, QUEEN)) bonus += 8 * (14 - dist);
    return Value(v + (winner == stm ? bonus : -bonus));
}

// Pressure that `attacker`'s pieces put on the other king: squares of the king's surroundings hit,
// queens and rooks counting more, and several attackers together far more than one.
static int king_pressure(const Position& pos, Color attacker) {
    const Square ksq = pos.king_square(~attacker);
    const Bitboard zone = attacks::king(ksq) | square_bb(ksq);
    const Bitboard occ = pos.pieces();
    static constexpr int kWeight[PIECE_TYPE_NB] = {0, 2, 2, 3, 5, 0}; // P N B R Q K
    int units = 0, attackers = 0;
    for (PieceType pt : {KNIGHT, BISHOP, ROOK, QUEEN}) {
        Bitboard b = pos.pieces(attacker, pt);
        while (b) {
            const int n = popcount(attacks::piece(pt, pop_lsb(b), occ) & zone);
            if (n) units += n * kWeight[pt], ++attackers;
        }
    }
    return units * (2 + attackers);
}

// Attack mode: the engine is rewarded for its own pressure on the enemy king (the opponent is not
// credited the same way, so it is optimistic about attacking). Defence mode: it is instead
// penalised for the opponent's pressure on its own king, and digs in until the danger has passed.
// Returned from the side to move's point of view.
Value Search::aggression_bonus() const {
    const Color us = root_color_;
    int bonus;
    if (defending_) bonus = -std::min(160, king_pressure(pos_, ~us));
    else bonus = std::min(160, king_pressure(pos_, us) * std::clamp(style_.aggression, 0, 150) / 100);
    return Value(pos_.side_to_move() == us ? bonus : -bonus);
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

    if (pos_.is_draw(ply)) return draw_value(ply);
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
        if (pos_.is_draw(ply)) return draw_value(ply);
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
    if (list.size == 0) return in_check ? mated_in(ply) : draw_value(ply);

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
    root_color_ = root.side_to_move();
    // attack / defence switch, with hysteresis so it does not flicker: the last search's verdict
    // decides. In defence the engine stops gambling: no contempt-driven risks, no strange moves.
    // (kept per colour: self-play uses one search for both sides)
    if (style_.enabled && style_.adaptive) {
        bool& d = defending_by_[root_color_];
        const Value last = last_score_[root_color_];
        if (!d && last < -60) d = true;
        else if (d && last > -10) d = false;
        defending_ = d;
    } else {
        defending_ = false;
    }
    nodes_ = 0;
    can_stop_ = false;
    time_up_ = false;
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
    // with the style on, part of the time is kept for verifying the unusual alternatives
    const std::int64_t full_hard_ms = hard_ms_;
    if (style_.enabled && time_limited_) {
        hard_ms_ = std::max<std::int64_t>(1, full_hard_ms * (100 - style_.verify_percent) / 100);
        soft_ms_ = std::min(soft_ms_, hard_ms_);
    }

    SearchResult result;
    MoveList root_moves;
    generate_moves(pos_, root_moves);
    if (root_moves.size == 0) {
        result.score = pos_.in_check() ? mated_in(0) : VALUE_DRAW;
        return result;
    }
    result.best = root_moves[0];
    const Position root_copy = pos_;

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

    result.objective_best = result.best;
    last_score_[root_color_] = result.score;
    result.defending = defending_;
    if (style_.enabled && !limits.infinite) {
        apply_style(root_copy, root_moves, full_hard_ms, result);
        ++style_decisions_;
        if (result.styled) ++styled_moves_;
    }

    result.nodes = nodes_;
    result.time_ms = elapsed_ms();
    return result;
}

// ---------------------------------------------------------------------------
// Strange style
// ---------------------------------------------------------------------------
void Search::set_style(const StyleOptions& s) {
    style_ = s;
    style_rng_.seed(s.seed ? s.seed : std::uint64_t(std::chrono::steady_clock::now().time_since_epoch().count()));
}

int style_margin(Value best) {
    // the concession is paid on every styled move, so it stays small while the game is open
    // and only grows once the advantage is large enough to absorb it
    if (best < -40) return 0;   // in trouble: only the best move will do
    if (best < 60) return 8;    // balanced game: almost free
    if (best < 200) return 12;
    if (best < 500) return 25;
    if (best < 900) return 55;
    return 110;                 // clearly winning: free to be bizarre
}

int move_strangeness(const Position& pos, Move m) {
    const Piece pc = pos.piece_on(m.from());
    const PieceType pt = type_of(pc);
    const Color us = pos.side_to_move();
    const int from_rank = relative_rank(us, m.from()), to_rank = relative_rank(us, m.to());
    const int to_file = file_of(m.to());
    const bool middlegame = pos.pieces(QUEEN) && pos.non_pawn_material(~us) > 1300;
    int s = 0;
    if (m.is_promotion() && m.promotion_type() != QUEEN) s += 45;                      // under-promotion
    if (pt == KING && !m.is_castle() && middlegame) s += 30;                             // king walk
    if (!m.is_castle() && !m.is_promotion() && !pos.see_ge(m, 0)) s += 35;              // sacrifice / piece left en prise
    if (pt == PAWN && !m.is_capture()) {
        const int f = file_of(m.from());
        if (f == 0 || f == 7) s += 16;                                                    // rook-pawn push
        else if (f == 1 || f == 6) s += 8;
    }
    if (pt == KNIGHT && (to_file == 0 || to_file == 7)) s += 15;                         // knight on the rim
    if (pt == QUEEN && pos.game_ply() < 20) s += 12;                                     // early queen trip
    if (pt != PAWN && pt != KING && to_rank < from_rank && !m.is_capture()) s += 10;     // retreat
    if (pt == ROOK && (to_rank == 2 || to_rank == 3) && !m.is_capture()) s += 10;        // rook lift
    if (m.is_quiet()) {                                                                   // ignoring a winning capture
        MoveList noisy;
        generate_moves(pos, noisy, GenType::NOISY);
        for (Move x : noisy)
            if (x.is_capture() && pos.see_ge(x, 1)) { s += 8; break; }
    }
    return s;
}

void Search::apply_style(const Position& root, const MoveList& root_moves, std::int64_t full_hard_ms, SearchResult& result) {
    // the caller asked us to stop: play what we have
    if (stop_.load(std::memory_order_relaxed) && !time_up_) return;
    if (!result.best || root_moves.size < 2 || is_mate_score(result.score) || result.depth < 2) return;
    const bool worse = result.score < -40; // in trouble: no strange moves, only risk-free traps

    // resume with the time the main search left unused
    hard_ms_ = full_hard_ms;
    time_up_ = false;
    stop_.store(false, std::memory_order_relaxed);
    if (limits_.nodes) limits_.nodes = nodes_ + std::max<std::uint64_t>(1, limits_.nodes * std::uint64_t(std::max(style_.verify_percent, 30)) / 100);
    pos_ = root;
    if (use_nnue_) acc_->reset(*net_, pos_);

    if (style_.trap > 0 && try_trap(root_moves, result, worse || defending_)) {
        stop_.store(false, std::memory_order_relaxed);
        return;
    }
    if (worse || should_stop()) { stop_.store(false, std::memory_order_relaxed); return; }

    const int margin = style_margin(result.score) * style_.margin_percent / 100;
    if (margin <= 0) return;

    struct Cand { Move m; int s; };
    std::vector<Cand> cands;
    for (Move m : root_moves) {
        if (m == result.best) continue;
        const int s = move_strangeness(root, m);
        if (s >= 10) cands.push_back({m, s * 8 + int(style_rng_() % 8)}); // small random tie-break: never the same game twice
    }
    if (cands.empty()) return;
    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.s > b.s; });
    if (cands.size() > 6) cands.resize(6);

    const Value threshold = result.score - margin;
    const int depth = std::max(1, result.depth - 2);
    for (const Cand& c : cands) {
        if (should_stop()) break;
        // null-window proof that the move scores at least `threshold`
        // 1) cheap filter at reduced depth, 2) confirmation at the full depth of the main search:
        //    a shallow search alone lets through "sacrifices" that only look sound
        make(c.m);
        Value v = -negamax(-threshold, -threshold + 1, depth - 1, 1, true);
        if (v >= threshold && !stop_.load(std::memory_order_relaxed))
            v = -negamax(-threshold, -threshold + 1, result.depth - 1, 1, true);
        unmake();
        if (stop_.load(std::memory_order_relaxed)) break;
        if (v >= threshold) {
            result.best = c.m;
            result.ponder = Move();
            result.pv.assign(1, c.m);
            result.score = v;
            result.styled = true;
            break;
        }
    }
    stop_.store(false, std::memory_order_relaxed);
}

// Best move for the side to move at `ply` by a shallow full-width search: the opponent model.
Move Search::shallow_best(int ply, int depth) {
    MoveList list;
    generate_moves(pos_, list);
    Move best;
    Value best_v = -VALUE_INFINITE;
    for (Move m : list) {
        make(m);
        const Value v = -negamax(-VALUE_INFINITE, VALUE_INFINITE, depth - 1, ply + 1, false);
        unmake();
        if (stop_.load(std::memory_order_relaxed)) return best;
        if (v > best_v) best_v = v, best = m;
    }
    return best;
}

void Search::shallow_scores(const Position& pos, int depth, std::vector<ShallowScore>& out) {
    out.clear();
    stop_.store(false, std::memory_order_relaxed);
    time_limited_ = false;
    can_stop_ = false;
    limits_ = SearchLimits{};
    pos_ = pos;
    use_nnue_ = nnue_active();
    if (use_nnue_) acc_->reset(*net_, pos_);
    MoveList list;
    generate_moves(pos_, list);
    for (Move m : list) {
        make(m);
        const Value v = -negamax(-VALUE_INFINITE, VALUE_INFINITE, std::max(0, depth - 1), 1, false);
        unmake();
        out.push_back({m, int(v)});
    }
}

// The human model needs a look-ahead of the position; a separate classical searcher provides it, so the
// main search's table and evaluator are untouched and the numbers match what the model was trained on.
void Search::human_rank(Position& pos, std::vector<RankedMove>& out) {
    if (!look_) {
        look_ = std::make_unique<Search>(1);
        look_->set_eval_mode(EvalMode::CLASSICAL);
    }
    std::vector<ShallowScore> sc;
    look_->shallow_scores(pos, 2, sc);
    human_->rank(pos, out, &sc);
}

Move Search::predict_human(const Position& pos, int depth) {
    stop_.store(false, std::memory_order_relaxed);
    time_limited_ = false;
    can_stop_ = false;
    limits_ = SearchLimits{};
    pos_ = pos;
    if (human_) {
        std::vector<RankedMove> r;
        Position copy = pos;
        human_rank(copy, r);
        return r.empty() ? Move() : r[0].move;
    }
    use_nnue_ = nnue_active();
    if (use_nnue_) acc_->reset(*net_, pos_);
    return shallow_best(0, std::max(1, depth));
}

// A trap: a move that is (nearly) as good as the best one, after which the replies a person is
// likely to play are serious mistakes. With a human model the replies come with probabilities:
// the trap is judged by the chance that the opponent falls for it (the summed probability of the
// likely replies that lose at least 1.5 pawns), and the most promising one is played. Without a
// model, the single reply a shallow search prefers stands in for the person.
// The opponent model `trap` (0..100) sets how much evaluation may be risked for it (up to 0.4 pawn)
// and how likely the fall must be. pos_ is the root on entry and on exit.
bool Search::try_trap(const MoveList& root_moves, SearchResult& result, bool no_risk) {
    // counter-traps while defending risk nothing: only moves as good as the best one
    const int risk = no_risk ? 0 : std::clamp(style_.trap, 0, 100) * 40 / 100; // up to 0.4 pawn
    const Value gain = 150;                                                      // the reply must lose at least this much
    const Value sound = result.score - risk;
    const Value punish = result.score + gain;
    const int depth = std::max(1, result.depth - 1);
    // the more eager the style, the lower the chance that is still worth a trap
    const float need = human_ ? std::max(0.20f, 0.65f - 0.0045f * float(std::clamp(style_.trap, 0, 100))) : 0.0f;

    // most forcing moves first: checks and captures give the opponent the most chances to go wrong
    struct Cand { Move m; int k; };
    std::vector<Cand> cands;
    for (Move m : root_moves) {
        int k = int(style_rng_() % 4);
        { Position t = pos_; t.do_move(m); if (t.in_check()) k += 40; }
        if (pos_.piece_on(m.to()) != NO_PIECE) k += 20;
        if (m == result.best) k += 60; // the best move itself may already be a trap
        cands.push_back({m, k});
    }
    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.k > b.k; });
    if (cands.size() > 10) cands.resize(10);

    Move best_move, best_reply;
    Value best_score = 0;
    float best_prob = 0.0f;
    std::vector<RankedMove> replies;
    for (const Cand& c : cands) {
        if (should_stop()) break;
        make(c.m);
        replies.clear();
        if (human_) { Position copy = pos_; human_rank(copy, replies); }
        else if (const Move nat = shallow_best(1, 2)) replies.push_back({nat, 1.0f});

        float fall = 0.0f, covered = 0.0f;
        Move likely_loser;
        for (std::size_t k = 0; k < replies.size() && k < 4 && covered < 0.85f; ++k) {
            covered += replies[k].p;
            make(replies[k].move);
            const Value after = negamax(punish - 1, punish, std::max(1, depth - 1), 2, true); // engine to move
            unmake();
            if (stop_.load(std::memory_order_relaxed)) break;
            if (after >= punish) {
                fall += replies[k].p;
                if (!likely_loser) likely_loser = replies[k].move;
            }
        }
        bool ok = false;
        Value v = 0;
        if (!stop_.load(std::memory_order_relaxed) && likely_loser && fall >= need && fall > best_prob) {
            // is the trap itself sound against the best defence?
            v = c.m == result.best ? result.score : -negamax(-sound, -sound + 1, depth, 1, true);
            ok = v >= sound && !stop_.load(std::memory_order_relaxed);
        }
        unmake();
        if (stop_.load(std::memory_order_relaxed)) break;
        if (ok) {
            best_move = c.m;
            best_reply = likely_loser;
            best_score = v;
            best_prob = fall;
            if (!human_ || fall >= 0.75f) break; // good enough: do not spend more time
        }
    }
    if (!best_move) return false;
    if (best_move != result.best) {
        result.best = best_move;
        result.ponder = best_reply;
        result.pv.assign({best_move, best_reply});
        result.score = best_score;
        result.styled = true;
    }
    result.trap_reply = best_reply;
    result.trap_prob = best_prob;
    return true;
}

} // namespace barracuda
