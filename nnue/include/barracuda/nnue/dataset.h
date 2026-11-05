// Barracuda NNUE - training dataset (FEN + centipawn score + W/D/L outcome).
//
// Accepted text line formats (one position per line, '#' starts a comment):
//   <fen> | <score_cp> | <result>        (Barracuda self-play output)
//   <fen> [<result>] <score_cp>          (common "legacy" text format)
// <score_cp> is from WHITE's point of view. <result> is 1.0 / 0.5 / 0.0, 1-0 / 1/2-1/2 / 0-1
// or w / d / b, also from White's point of view.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "barracuda/engine/position.h"

namespace barracuda::nnue {

struct PackedSample {
    std::uint16_t pieces[32]; // (piece << 6) | square
    std::uint8_t count = 0;
    std::uint8_t stm = 0;     // Color
    std::int16_t score = 0;   // centipawns, white-relative
    std::uint8_t result = 1;  // 0 = black wins, 1 = draw, 2 = white wins

    int features(Color perspective, int* out) const;
};

class Dataset {
public:
    struct LoadStats {
        std::uint64_t lines = 0, accepted = 0, rejected = 0;
    };
    using ProgressFn = std::function<void(std::uint64_t lines, std::uint64_t accepted)>;

    // Appends to the dataset. `max_samples` = 0 means unlimited.
    bool load_text(const std::string& path, std::uint64_t max_samples, LoadStats* stats, std::string* error,
                   const std::atomic<bool>* stop = nullptr, const ProgressFn& progress = {});

    void add(const Position& pos, int score_white, int result_white /*0,1,2*/);
    static bool parse_line(const std::string& line, PackedSample& out);
    static bool pack_fen(const std::string& fen, PackedSample& out);

    void shuffle(std::uint64_t seed);
    std::size_t size() const { return samples_.size(); }
    bool empty() const { return samples_.empty(); }
    const PackedSample& operator[](std::size_t i) const { return samples_[i]; }
    void clear() { samples_.clear(); }
    std::vector<PackedSample>& samples() { return samples_; }

private:
    std::vector<PackedSample> samples_;
};

} // namespace barracuda::nnue
