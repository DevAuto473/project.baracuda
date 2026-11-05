#include "barracuda/engine/tt.h"

#include <algorithm>

namespace barracuda {

void TranspositionTable::resize(std::size_t mb) {
    mb = std::clamp<std::size_t>(mb, 1, 65536);
    const std::size_t bytes = mb * 1024 * 1024;
    std::size_t count = 1;
    while (count * 2 * sizeof(TTBucket) <= bytes) count *= 2; // power of two for masking
    buckets_.assign(count, TTBucket{});
    mask_ = count - 1;
    generation_ = 0;
}

void TranspositionTable::clear() {
    std::fill(buckets_.begin(), buckets_.end(), TTBucket{});
    generation_ = 0;
}

Value TranspositionTable::score_to_tt(Value v, int ply) {
    if (v >= VALUE_MATE_IN_MAX_PLY) return v + ply;
    if (v <= VALUE_MATED_IN_MAX_PLY) return v - ply;
    return v;
}

Value TranspositionTable::score_from_tt(Value v, int ply) {
    if (v >= VALUE_MATE_IN_MAX_PLY) return v - ply;
    if (v <= VALUE_MATED_IN_MAX_PLY) return v + ply;
    return v;
}

bool TranspositionTable::probe(Key key, TTEntry& out) const {
    const TTBucket& b = buckets_[index(key)];
    const std::uint32_t k32 = std::uint32_t(key >> 32);
    for (const TTEntry& e : b.e) {
        if (e.key32 == k32 && e.bound() != BOUND_NONE) {
            out = e;
            return true;
        }
    }
    return false;
}

void TranspositionTable::store(Key key, Move m, Value score, Value eval, int depth, Bound bound, int ply) {
    TTBucket& b = buckets_[index(key)];
    const std::uint32_t k32 = std::uint32_t(key >> 32);
    TTEntry* slot;
    TTEntry& deep = b.e[0];
    if (deep.key32 == k32 || deep.bound() == BOUND_NONE || deep.generation() != generation_ ||
        depth >= deep.depth || bound == BOUND_EXACT) {
        slot = &deep;
        // keep the deeper entry around in the always-replace slot
        if (deep.key32 != k32 && deep.bound() != BOUND_NONE) b.e[1] = deep;
    } else {
        slot = &b.e[1];
    }
    // preserve an existing best move if the new store has none
    if (!m && slot->key32 == k32) m = Move(slot->move);

    slot->key32 = k32;
    slot->move = m.raw();
    slot->score = std::int16_t(score_to_tt(score, ply));
    slot->eval = std::int16_t(std::clamp(eval, -32000, 32000));
    slot->depth = std::uint8_t(std::clamp(depth, 0, 255));
    slot->gen_bound = std::uint8_t((generation_ << 2) | bound);
}

int TranspositionTable::hashfull() const {
    int used = 0;
    const std::size_t n = std::min<std::size_t>(500, buckets_.size());
    for (std::size_t i = 0; i < n; ++i)
        for (const TTEntry& e : buckets_[i].e)
            if (e.bound() != BOUND_NONE && e.generation() == generation_) ++used;
    return int(used * 1000 / std::max<std::size_t>(1, n * 2));
}

} // namespace barracuda
