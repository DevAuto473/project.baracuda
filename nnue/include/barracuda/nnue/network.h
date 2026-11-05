// Barracuda NNUE - quantised inference network with incrementally updated accumulators.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include "barracuda/engine/position.h"
#include "barracuda/nnue/architecture.h"
#include "barracuda/nnue/float_network.h"

namespace barracuda::nnue {

// One accumulator per perspective (index = Color, *not* side to move).
struct alignas(64) Accumulator {
    alignas(64) std::int16_t v[COLOR_NB][kHidden];
};

// Which SIMD kernel set was compiled in ("AVX2", "SSE4.1", "scalar").
const char* simd_backend();

class Network {
public:
    // Quantises a float master network (round-to-nearest, saturating).
    static std::shared_ptr<Network> from_float(const FloatNetwork& f);

    // --- accumulator maintenance ------------------------------------------
    void refresh(const Position& pos, Accumulator& acc) const;
    // next = prev + dirty (feature add/sub for both perspectives in a single fused pass)
    void update(const Accumulator& prev, Accumulator& next, const DirtyPiece& dirty) const;
    void add_feature(Accumulator& acc, Color perspective, int feature) const;
    void sub_feature(Accumulator& acc, Color perspective, int feature) const;

    // --- inference ----------------------------------------------------------
    struct Trace {
        alignas(64) std::uint8_t ft[2 * kHidden]; // [stm | nstm], ClippedReLU output 0..127
        std::int32_t l1_raw[kL1];
        std::uint8_t l1[kL1];
        std::int32_t l2_raw[kL2];
        std::uint8_t l2[kL2];
        std::int32_t out_raw;
    };
    // Centipawns from the side-to-move point of view.
    int evaluate(const Accumulator& acc, Color stm, Trace* trace = nullptr) const;
    int evaluate(const Position& pos, Trace* trace = nullptr) const;

    // --- raw parameter access (for the visualiser) -----------------------------
    float ft_weight(int feature, int neuron) const { return float(ft_w_[feature * kHidden + neuron]) / kQA; }
    float l1_weight(int out, int in) const { return float(l1_w_[out * 2 * kHidden + in]) / kQB; }
    float l2_weight(int out, int in) const { return float(l2_w_[out * kL1 + in]) / kQB; }
    float out_weight(int in) const { return float(out_w_[in]) / kQB; }

    std::uint32_t epochs_trained() const { return epochs_trained_; }
    std::uint64_t samples_trained() const { return samples_trained_; }
    bool is_trained() const { return epochs_trained_ > 0; }

private:
    alignas(64) std::int16_t ft_w_[kInputs * kHidden];
    alignas(64) std::int16_t ft_b_[kHidden];
    alignas(64) std::int8_t l1_w_[kL1 * 2 * kHidden];
    alignas(64) std::int32_t l1_b_[kL1];
    alignas(64) std::int8_t l2_w_[kL2 * kL1];
    alignas(64) std::int32_t l2_b_[kL2];
    alignas(64) std::int8_t out_w_[kL2];
    std::int32_t out_b_ = 0;
    std::uint32_t epochs_trained_ = 0;
    std::uint64_t samples_trained_ = 0;
};

// Loads a float network file (*.bnn) and quantises it. Returns nullptr on failure.
std::shared_ptr<Network> load_network(const std::string& path, std::string* error = nullptr);

// Stack of accumulators that mirrors the search's make/unmake.
class AccumulatorStack {
public:
    void reset(const Network& net, const Position& pos) {
        top_ = 0;
        net.refresh(pos, stack_[0]);
    }
    // Call right after pos.do_move(m) / do_null_move().
    void push(const Network& net, const DirtyPiece& dirty) {
        net.update(stack_[top_], stack_[top_ + 1], dirty);
        ++top_;
    }
    void push_copy() {
        stack_[top_ + 1] = stack_[top_];
        ++top_;
    }
    void pop() { --top_; }
    const Accumulator& current() const { return stack_[top_]; }

private:
    std::array<Accumulator, MAX_PLY + 16> stack_{};
    int top_ = 0;
};

} // namespace barracuda::nnue
