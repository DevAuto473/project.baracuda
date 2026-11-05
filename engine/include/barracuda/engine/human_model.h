// Barracuda - a model of how people choose moves.
//
// People do not search: they pick the move that "looks right" - captures, checks, developing a
// piece, a recapture, escaping an attacked piece - and they miss what is hidden two or three
// moves deeper. This model learns exactly that surface preference from real games: every legal
// move is described by ~30 cheap features (piece, capture, check, is the piece left hanging,
// does it create a threat, ...) and the probability of a move is a softmax over a weighted sum of
// them (a conditional logit trained on Lichess games, see `barracuda-cli train-human`).
//
// The search uses it to set traps: it prefers a move after which the reply a person is most
// likely to play is a mistake. The model file is ~1 KB of text, so it also runs in the browser.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "barracuda/engine/position.h"

namespace barracuda {

inline constexpr int kHumanFeatures = 30;

struct RankedMove {
    Move move;
    float p = 0.0f; // probability that a person plays it
};

class HumanModel {
public:
    float w[kHumanFeatures] = {};
    std::uint64_t samples = 0; // positions it was trained on
    float top1 = 0.0f, top3 = 0.0f; // held-out accuracy (0 when unknown)

    static const char* feature_name(int i);

    // Features of `m` in `pos` (the position is restored before returning).
    static void features(Position& pos, Move m, float* f);

    float logit(Position& pos, Move m) const;
    // All legal moves with their probability, most likely first.
    void rank(Position& pos, std::vector<RankedMove>& out) const;

    bool save(const std::string& path, std::string* err = nullptr) const;
    static std::shared_ptr<HumanModel> load(const std::string& path, std::string* err = nullptr);
};

struct HumanTrainOptions {
    int epochs = 6;
    int batch = 2048;
    float lr = 0.05f;
    int threads = 0;      // 0 = all cores
    int max_samples = 0;  // 0 = all
};

struct HumanTrainStats {
    std::uint64_t samples = 0, val_samples = 0;
    double train_nll = 0, val_nll = 0;      // average negative log-likelihood per move
    double val_top1 = 0, val_top3 = 0;      // how often the move people played was the model's 1st / top-3 choice
    double baseline_nll = 0, baseline_top1 = 0; // choosing uniformly among the legal moves
    double val_top1_shallow = 0;                // the old stand-in: the move a 2-ply search prefers (first-choice accuracy)
};

using HumanLog = std::function<void(const std::string&)>;

// Data: one "fen | uci_move" per line. Trains `model` in place.
bool train_human_model(const std::string& data_path, HumanModel& model, const HumanTrainOptions& opt,
                       const std::atomic<bool>& stop, HumanTrainStats& stats, std::string* err, const HumanLog& log = {});

} // namespace barracuda
