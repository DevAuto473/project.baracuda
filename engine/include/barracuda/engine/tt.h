// Barracuda - transposition table (2-way buckets: depth-preferred + always-replace).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "barracuda/engine/types.h"

namespace barracuda {

enum Bound : std::uint8_t { BOUND_NONE = 0, BOUND_UPPER = 1, BOUND_LOWER = 2, BOUND_EXACT = 3 };

struct TTEntry {
    std::uint32_t key32 = 0;
    std::uint16_t move = 0;
    std::int16_t score = 0;
    std::int16_t eval = 0;
    std::uint8_t depth = 0;
    std::uint8_t gen_bound = 0; // generation << 2 | bound

    Bound bound() const { return Bound(gen_bound & 3); }
    std::uint8_t generation() const { return gen_bound >> 2; }
};

struct alignas(32) TTBucket {
    TTEntry e[2];
};

class TranspositionTable {
public:
    explicit TranspositionTable(std::size_t mb = 64) { resize(mb); }

    void resize(std::size_t mb);
    void clear();
    void new_search() { generation_ = std::uint8_t((generation_ + 1) & 63); }

    // Returns a copy (so a concurrent overwrite never yields a torn entry for the caller).
    bool probe(Key key, TTEntry& out) const;
    void store(Key key, Move m, Value score, Value eval, int depth, Bound b, int ply);
    int hashfull() const; // permille

    static Value score_to_tt(Value v, int ply);
    static Value score_from_tt(Value v, int ply);

    std::size_t size_mb() const { return buckets_.size() * sizeof(TTBucket) / (1024 * 1024); }

private:
    std::size_t index(Key key) const { return std::size_t(key) & mask_; }

    std::vector<TTBucket> buckets_;
    std::size_t mask_ = 0;
    std::uint8_t generation_ = 0;
};

} // namespace barracuda
