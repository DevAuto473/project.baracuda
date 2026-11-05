#include "barracuda/engine/human_model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>

#include "barracuda/engine/movegen.h"
#include "barracuda/engine/notation.h"
#include "barracuda/engine/search.h"

namespace barracuda {
namespace {

const char* const kNames[kHumanFeatures] = {
    "moves a pawn", "moves a knight", "moves a bishop", "moves a rook", "moves the queen", "moves the king",
    "capture", "value of the captured piece", "gives check", "promotion", "castling",
    "lands on an attacked square", "leaves the piece hanging", "size of the loss if hanging",
    "piece was attacked before moving", "escapes an attack", "recapture", "wins material in a trade",
    "forward progress (pieces)", "pawn push", "pawn double push", "centrality of the target square",
    "centrality gained", "develops a minor piece", "king move (no castling)", "early queen move",
    "creates a threat on a valuable piece", "steps next to an enemy pawn's attack", "retreat", "safe capture"};

constexpr float kValue9[7] = {1, 3, 3, 5, 9, 0, 0}; // pawn units

float centrality(Square s) {
    const int f = std::abs(2 * file_of(s) - 7), r = std::abs(2 * rank_of(s) - 7);
    return 1.0f - float(std::max(f, r)) / 7.0f;
}

} // namespace

const char* HumanModel::feature_name(int i) { return i >= 0 && i < kHumanFeatures ? kNames[i] : "?"; }

void HumanModel::features(Position& pos, Move m, float* f) {
    std::fill(f, f + kHumanFeatures, 0.0f);
    const Color us = pos.side_to_move(), them = ~us;
    const Square from = m.from(), to = m.to();
    const PieceType pt = type_of(pos.piece_on(from));
    const float mv = kValue9[pt];
    PieceType victim = NO_PIECE_TYPE;
    if (m.is_ep()) victim = PAWN;
    else if (m.is_capture()) victim = type_of(pos.piece_on(to));
    const float vv = victim == NO_PIECE_TYPE ? 0.0f : kValue9[victim];

    f[int(pt)] = 1.0f;
    if (m.is_capture()) { f[6] = 1.0f; f[7] = vv / 9.0f; }
    if (m.is_promotion()) f[9] = 1.0f;
    if (m.is_castle()) f[10] = 1.0f;
    const bool attacked_before = pos.is_attacked(from, them);
    f[14] = attacked_before ? 1.0f : 0.0f;
    const Move last = pos.history_size() > 1 ? pos.last_move() : Move();
    if (m.is_capture() && last && last.to() == to) f[16] = 1.0f;
    if (m.is_capture() && vv > mv) f[17] = 1.0f;
    const float rel = float(relative_rank(us, to) - relative_rank(us, from)) / 7.0f;
    if (pt != PAWN && pt != KING) { f[18] = rel; if (rel < 0) f[28] = 1.0f; }
    if (pt == PAWN && !m.is_capture()) f[m.flags() == DOUBLE_PUSH ? 20 : 19] = 1.0f;
    f[21] = centrality(to);
    f[22] = centrality(to) - centrality(from);
    if ((pt == KNIGHT || pt == BISHOP) && relative_rank(us, from) == 0) f[23] = 1.0f;
    if (pt == KING && !m.is_castle()) f[24] = 1.0f;
    if (pt == QUEEN && pos.fullmove_number() <= 10) f[25] = 1.0f;
    if ((attacks::pawn(us, to) & pos.pieces(them, PAWN)) != 0) f[27] = 1.0f;

    pos.do_move(m);
    f[8] = pos.in_check() ? 1.0f : 0.0f;
    const Bitboard att = pos.attackers_to(to);
    const Bitboard theirs = att & pos.pieces(them), ours = att & pos.pieces(us);
    const bool attacked = theirs != 0, defended = ours != 0;
    f[11] = attacked ? 1.0f : 0.0f;
    float lowest = 100.0f;
    for (int t = PAWN; t <= KING; ++t)
        if (theirs & pos.pieces(them, PieceType(t))) { lowest = t == KING ? 100.0f : kValue9[t]; break; }
    const bool hanging = attacked && (!defended || lowest < mv);
    if (hanging) {
        f[12] = 1.0f;
        f[13] = std::max(0.0f, mv - vv) / 9.0f;
    }
    if (attacked_before && !attacked) f[15] = 1.0f;
    if (m.is_capture() && !attacked) f[29] = 1.0f;
    // threat: the moved piece now attacks an enemy piece worth at least as much as itself
    const PieceType now = m.is_promotion() ? m.promotion_type() : pt;
    const Bitboard hit = (now == PAWN ? attacks::pawn(us, to) : attacks::piece(now, to, pos.pieces())) & pos.pieces(them);
    float best = 0.0f;
    for (int t = PAWN; t < KING; ++t)
        if (hit & pos.pieces(them, PieceType(t))) best = std::max(best, kValue9[t]);
    if (best > 0.0f && best >= kValue9[now]) f[26] = best / 9.0f;
    pos.undo_move();
}

float HumanModel::logit(Position& pos, Move m) const {
    float f[kHumanFeatures];
    features(pos, m, f);
    float s = 0.0f;
    for (int i = 0; i < kHumanFeatures; ++i) s += w[i] * f[i];
    return s;
}

void HumanModel::rank(Position& pos, std::vector<RankedMove>& out) const {
    out.clear();
    const MoveList list = legal_moves(pos);
    if (list.size == 0) return;
    float mx = -1e30f;
    for (Move m : list) {
        const float s = logit(pos, m);
        out.push_back({m, s});
        mx = std::max(mx, s);
    }
    float sum = 0.0f;
    for (RankedMove& r : out) { r.p = std::exp(r.p - mx); sum += r.p; }
    for (RankedMove& r : out) r.p /= sum;
    std::sort(out.begin(), out.end(), [](const RankedMove& a, const RankedMove& b) { return a.p > b.p; });
}

bool HumanModel::save(const std::string& path, std::string* err) const {
    std::ofstream f(path, std::ios::trunc);
    if (!f) {
        if (err) *err = "cannot write '" + path + "'";
        return false;
    }
    f << "BRCHUMAN 1 " << kHumanFeatures << " " << (unsigned long long)samples << " " << top1 << " " << top3 << "\n";
    for (int i = 0; i < kHumanFeatures; ++i) f << w[i] << (i + 1 < kHumanFeatures ? " " : "\n");
    return bool(f);
}

std::shared_ptr<HumanModel> HumanModel::load(const std::string& path, std::string* err) {
    std::ifstream f(path);
    if (!f) {
        if (err) *err = "cannot open '" + path + "'";
        return nullptr;
    }
    std::string magic;
    int version = 0, n = 0;
    unsigned long long samples = 0;
    auto m = std::make_shared<HumanModel>();
    if (!(f >> magic >> version >> n >> samples >> m->top1 >> m->top3) || magic != "BRCHUMAN" || version != 1 || n != kHumanFeatures) {
        if (err) *err = "'" + path + "' is not a Barracuda human model";
        return nullptr;
    }
    m->samples = samples;
    for (int i = 0; i < kHumanFeatures; ++i)
        if (!(f >> m->w[i]) || !std::isfinite(m->w[i])) {
            if (err) *err = "'" + path + "' is truncated or corrupt";
            return nullptr;
        }
    return m;
}

// ---------------------------------------------------------------------------------------------
// Training: conditional logit, Adam, mini-batches split over threads.
// ---------------------------------------------------------------------------------------------
namespace {

struct Sample {
    std::string fen;
    std::uint16_t move;
};

struct Grad {
    double g[kHumanFeatures] = {};
    double nll = 0.0;
    double top1 = 0, top3 = 0, uni_nll = 0, uni_top1 = 0;
    std::uint64_t n = 0;
};

// Gradient of the negative log-likelihood (and the held-out metrics) for one sample.
void accumulate(const HumanModel& model, Position& pos, const Sample& s, Grad& out, bool want_grad) {
    if (!pos.set_fen(s.fen)) return;
    const MoveList list = legal_moves(pos);
    if (list.size < 2) return;
    float feat[256][kHumanFeatures];
    float logit[256];
    int played = -1;
    float mx = -1e30f;
    for (int i = 0; i < list.size; ++i) {
        HumanModel::features(pos, list.moves[std::size_t(i)], feat[i]);
        float v = 0.0f;
        for (int k = 0; k < kHumanFeatures; ++k) v += model.w[k] * feat[i][k];
        logit[i] = v;
        mx = std::max(mx, v);
        if (list.moves[std::size_t(i)].raw() == s.move) played = i;
    }
    if (played < 0) return; // the recorded move is not legal here (corrupt line)
    double sum = 0.0;
    float p[256];
    for (int i = 0; i < list.size; ++i) { p[i] = std::exp(logit[i] - mx); sum += p[i]; }
    for (int i = 0; i < list.size; ++i) p[i] = float(p[i] / sum);
    out.nll += -std::log(std::max(1e-9f, p[played]));
    ++out.n;
    if (want_grad) {
        for (int i = 0; i < list.size; ++i)
            for (int k = 0; k < kHumanFeatures; ++k) out.g[k] += double(p[i]) * feat[i][k];
        for (int k = 0; k < kHumanFeatures; ++k) out.g[k] -= feat[played][k];
    } else {
        int better = 0;
        for (int i = 0; i < list.size; ++i) better += p[i] > p[played];
        out.top1 += better == 0;
        out.top3 += better < 3;
        out.uni_nll += std::log(double(list.size));
        out.uni_top1 += 1.0 / list.size;
    }
}

} // namespace

bool train_human_model(const std::string& data_path, HumanModel& model, const HumanTrainOptions& opt,
                       const std::atomic<bool>& stop, HumanTrainStats& stats, std::string* err, const HumanLog& log) {
    std::ifstream in(data_path);
    if (!in) {
        if (err) *err = "cannot open '" + data_path + "'";
        return false;
    }
    std::vector<Sample> all;
    {
        std::string line;
        Position pos;
        while (std::getline(in, line)) {
            const auto bar = line.find(" | ");
            if (bar == std::string::npos) continue;
            Sample s;
            s.fen = line.substr(0, bar);
            std::string uci = line.substr(bar + 3);
            while (!uci.empty() && (uci.back() == '\r' || uci.back() == ' ')) uci.pop_back();
            if (!pos.set_fen(s.fen)) continue;
            const Move m = parse_uci_move(pos, uci);
            if (!m) continue;
            s.move = m.raw();
            all.push_back(std::move(s));
            if (opt.max_samples && int(all.size()) >= opt.max_samples) break;
        }
    }
    if (all.size() < 1000) {
        if (err) *err = "need at least 1000 samples, found " + std::to_string(all.size()) + " (see: barracuda-cli import-pgn --moves-out)";
        return false;
    }
    if (log) log("loaded " + std::to_string(all.size()) + " (position, move) samples");

    // every 20th sample is held out
    std::vector<Sample> train, val;
    for (std::size_t i = 0; i < all.size(); ++i) (i % 20 == 0 ? val : train).push_back(std::move(all[i]));
    std::vector<Sample>().swap(all);
    stats.samples = train.size();
    stats.val_samples = val.size();

    const int threads = std::max(1, opt.threads ? opt.threads : int(std::thread::hardware_concurrency()));
    std::mt19937_64 rng(12345);
    std::vector<std::size_t> order(train.size());
    std::iota(order.begin(), order.end(), 0);

    double m1[kHumanFeatures] = {}, m2[kHumanFeatures] = {};
    std::uint64_t step = 0;
    auto evaluate = [&](const std::vector<Sample>& set, Grad& total) {
        std::vector<Grad> parts(static_cast<std::size_t>(threads));
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; ++t)
            pool.emplace_back([&, t] {
                Position pos;
                for (std::size_t i = std::size_t(t); i < set.size(); i += std::size_t(threads))
                    accumulate(model, pos, set[i], parts[std::size_t(t)], false);
            });
        for (auto& th : pool) th.join();
        for (const Grad& g : parts) {
            total.nll += g.nll; total.n += g.n; total.top1 += g.top1; total.top3 += g.top3;
            total.uni_nll += g.uni_nll; total.uni_top1 += g.uni_top1;
        }
    };

    for (int epoch = 1; epoch <= opt.epochs && !stop.load(); ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        const float lr = opt.lr * std::pow(0.7f, float(epoch - 1));
        double epoch_nll = 0.0;
        std::uint64_t epoch_n = 0;
        for (std::size_t b = 0; b < order.size() && !stop.load(); b += std::size_t(opt.batch)) {
            const std::size_t e = std::min(order.size(), b + std::size_t(opt.batch));
            std::vector<Grad> parts(static_cast<std::size_t>(threads));
            std::vector<std::thread> pool;
            for (int t = 0; t < threads; ++t)
                pool.emplace_back([&, t] {
                    Position pos;
                    for (std::size_t i = b + std::size_t(t); i < e; i += std::size_t(threads))
                        accumulate(model, pos, train[order[i]], parts[std::size_t(t)], true);
                });
            for (auto& th : pool) th.join();
            Grad total;
            for (const Grad& g : parts) {
                for (int k = 0; k < kHumanFeatures; ++k) total.g[k] += g.g[k];
                total.nll += g.nll;
                total.n += g.n;
            }
            if (total.n == 0) continue;
            epoch_nll += total.nll;
            epoch_n += total.n;
            ++step;
            for (int k = 0; k < kHumanFeatures; ++k) { // Adam
                const double g = total.g[k] / double(total.n) + 1e-5 * model.w[k];
                m1[k] = 0.9 * m1[k] + 0.1 * g;
                m2[k] = 0.999 * m2[k] + 0.001 * g * g;
                const double mh = m1[k] / (1.0 - std::pow(0.9, double(step)));
                const double vh = m2[k] / (1.0 - std::pow(0.999, double(step)));
                model.w[k] -= float(lr * mh / (std::sqrt(vh) + 1e-8));
            }
        }
        Grad v;
        evaluate(val, v);
        stats.train_nll = epoch_n ? epoch_nll / double(epoch_n) : 0.0;
        stats.val_nll = v.n ? v.nll / double(v.n) : 0.0;
        stats.val_top1 = v.n ? v.top1 / double(v.n) : 0.0;
        stats.val_top3 = v.n ? v.top3 / double(v.n) : 0.0;
        stats.baseline_nll = v.n ? v.uni_nll / double(v.n) : 0.0;
        stats.baseline_top1 = v.n ? v.uni_top1 / double(v.n) : 0.0;
        if (log) {
            char buf[256];
            std::snprintf(buf, sizeof(buf), "epoch %d/%d  train %.4f  held-out %.4f (uniform %.4f)  top-1 %.1f%% (uniform %.1f%%)  top-3 %.1f%%",
                          epoch, opt.epochs, stats.train_nll, stats.val_nll, stats.baseline_nll, 100 * stats.val_top1,
                          100 * stats.baseline_top1, 100 * stats.val_top3);
            log(buf);
        }
    }
    // the stand-in used before the model existed, measured on the same held-out games
    {
        Search shallow(1);
        shallow.set_eval_mode(EvalMode::CLASSICAL);
        Position pos;
        int tested = 0, hit = 0;
        for (std::size_t i = 0; i < val.size() && tested < 3000; i += std::max<std::size_t>(1, val.size() / 3000)) {
            if (!pos.set_fen(val[i].fen)) continue;
            ++tested;
            hit += shallow.predict_human(pos, 2).raw() == val[i].move;
        }
        stats.val_top1_shallow = tested ? double(hit) / tested : 0.0;
    }
    model.samples = train.size();
    model.top1 = float(stats.val_top1);
    model.top3 = float(stats.val_top3);
    return true;
}

} // namespace barracuda
